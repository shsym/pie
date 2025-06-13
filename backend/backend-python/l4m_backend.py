import time
import argparse
import warnings
import asyncio
import threading

import os
import torch
import zmq
from transformers import TorchAoConfig, AutoTokenizer
from typing import Dict, Any, Optional

import config
import l4m_pb2
import l4m_vision_pb2
import ping_pb2
import handshake_pb2

from common import ceil_div, handle_request
from driver import Driver
from llama import LlamaForCausalLM
from config import MAX_NUM_PAGES, MAX_NUM_EMBEDS
from base_backend_agent import BaseBackendAgent, ModelLoadRequest, ModelUnloadRequest

import logging

logger = logging.getLogger(__name__)

class LlamaBackendAgent(BaseBackendAgent):
    """
    Llama-specific backend agent that handles Llama model loading and ZMQ communication.
    """

    def __init__(self, model_name: str, ipc_endpoint: str, management_service_url: str,
                 backend_host: str, backend_api_port: int):
        # Only support the model that was passed when running this script
        supported_models = [model_name]

        super().__init__(
            management_service_url=management_service_url,
            backend_host=backend_host,
            backend_api_port=backend_api_port,
            backend_type="llama",
            service_name=f"LlamaBackend-{model_name.split('/')[-1]}",
            ipc_endpoint=ipc_endpoint,
            supported_models=supported_models,
            shutdown_callback=self._shutdown_zmq_server
        )

        self.model_name = model_name
        self.device = "cuda:0"
        self.model = None
        self.engine = None
        self.zmq_context = None
        self.zmq_router = None
        self.zmq_thread = None
        self.zmq_stop_event = threading.Event()

    async def _load_model_impl(self, request: ModelLoadRequest) -> Dict[str, Any]:
        """Load a Llama model."""
        try:
            # Check if this model is already loaded
            if self.model is not None:
                return {
                    "success": False,
                    "error": f"Model {self.model_name} is already loaded. Unload it first."
                }

            # Use the model name from request or fall back to default
            model_to_load = request.model_name or self.model_name

            # Load model using the same logic as original
            models_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "symphony", "models")
            local_dir = os.path.join(models_cache_dir, model_to_load.replace("/", "--"))

            if os.path.isdir(local_dir):
                logger.info(f"Loading Llama model from local cache: {local_dir}")
                model_path_or_name = local_dir
                local_only = True
            else:
                logger.info(f"Loading Llama model from HuggingFace Hub: {model_to_load}")
                model_path_or_name = model_to_load
                local_only = False

            # Suppress rope_scaling warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized keys in `rope_scaling`")
                self.model = LlamaForCausalLM.from_pretrained(
                    model_path_or_name,
                    torch_dtype="bfloat16",
                    device_map=self.device,
                    cache_dir=models_cache_dir,
                    local_files_only=local_only,
                )

            logger.info(f"Model {model_to_load} loaded successfully")

            # Initialize the engine
            self.engine = Driver(self.model, MAX_NUM_PAGES, MAX_NUM_EMBEDS, torch.bfloat16, self.device)
            logger.info("Driver engine initialized")

            # Start ZMQ server if not already running
            if not self.zmq_thread or not self.zmq_thread.is_alive():
                self._start_zmq_server()

            return {
                "success": True,
                "model_name": model_to_load,
                "device": self.device,
                "message": f"Llama model {model_to_load} loaded successfully"
            }

        except Exception as e:
            logger.error(f"Failed to load model {request.model_name}: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _unload_model_impl(self, request: ModelUnloadRequest) -> Dict[str, Any]:
        """Unload the current Llama model."""
        try:
            if self.model is None:
                return {
                    "success": False,
                    "error": "No model is currently loaded"
                }

            # Stop ZMQ server
            self._shutdown_zmq_server()

            # Clear model and engine
            model_name = self.model_name
            self.model = None
            self.engine = None

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Model {model_name} unloaded successfully")

            return {
                "success": True,
                "model_name": model_name,
                "message": f"Model {model_name} unloaded successfully"
            }

        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _start_zmq_server(self):
        """Start the ZMQ server for handling inference requests."""
        if self.zmq_thread and self.zmq_thread.is_alive():
            logger.warning("ZMQ server thread is already running")
            return

        logger.info(f"Starting ZMQ server on {self.ipc_endpoint}")
        self.zmq_stop_event.clear()
        self.zmq_thread = threading.Thread(target=self._zmq_server_loop, daemon=True)
        self.zmq_thread.name = f"{self.service_name}-ZMQThread"
        self.zmq_thread.start()

    def _shutdown_zmq_server(self):
        """Shutdown the ZMQ server."""
        logger.info("Shutting down ZMQ server")
        self.zmq_stop_event.set()

        if self.zmq_router:
            self.zmq_router.close()
            self.zmq_router = None

        if self.zmq_context:
            self.zmq_context.term()
            self.zmq_context = None

        if self.zmq_thread and self.zmq_thread.is_alive():
            self.zmq_thread.join(timeout=5)
            if self.zmq_thread.is_alive():
                logger.warning("ZMQ thread did not terminate in time")

    def _zmq_server_loop(self):
        """Main ZMQ server loop - handles inference requests."""
        try:
            self.zmq_context = zmq.Context()
            self.zmq_router = self.zmq_context.socket(zmq.ROUTER)
            self.zmq_router.bind(self.ipc_endpoint)

            # Set socket timeout to allow periodic checking of stop event
            self.zmq_router.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

            logger.info(f"ZMQ server listening on {self.ipc_endpoint}")

            connected_clients = {}
            protocols = ["l4m", "l4m-vision", "ping"]

            while not self.zmq_stop_event.is_set():
                try:
                    # ROUTER sockets receive multipart messages
                    frames = self.zmq_router.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    # Timeout occurred, check stop event and continue
                    continue
                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        # Context was terminated
                        break
                    logger.error(f"ZMQ error: {e}")
                    continue

                if not frames:
                    continue

                client_identity = frames[0]
                start = time.time()

                # Check if the client has already established a protocol
                if client_identity in connected_clients:
                    if len(frames) != 3:
                        logger.warning("Invalid message format")
                        continue

                    protocol_raw = frames[1]  # should be a single byte
                    protocol_idx = int.from_bytes(protocol_raw, byteorder="little")

                    if protocol_idx >= len(protocols):
                        logger.warning(f"Invalid protocol: {protocol_idx}")
                        continue

                    protocol = protocols[protocol_idx]
                    payload = frames[2]

                    if protocol == "l4m":
                        self._handle_l4m_request(client_identity, protocol_raw, payload)
                    elif protocol == "l4m-vision":
                        self._handle_l4m_vision_request(client_identity, protocol_raw, payload)
                    elif protocol == "ping":
                        self._handle_ping_request(client_identity, protocol_raw, payload)

                else:
                    # Handle handshake
                    self._handle_handshake(client_identity, frames, connected_clients, protocols)

        except Exception as e:
            logger.error(f"Error in ZMQ server loop: {e}", exc_info=True)
        finally:
            logger.info("ZMQ server loop ended")

    def _handle_l4m_request(self, client_identity, protocol_raw, payload):
        """Handle L4M inference requests."""
        if not self.engine:
            logger.error("Engine not initialized")
            return

        try:
            # Deserialize the protobuf message
            request = l4m_pb2.Request()
            request.ParseFromString(payload)

            # Handle the request using the existing logic
            response = handle_request(self.engine, request)

            if response is not None:
                reply_payload = response.SerializeToString()
                self.zmq_router.send_multipart([client_identity, protocol_raw, reply_payload])

        except Exception as e:
            logger.error(f"Error handling L4M request: {e}")

    def _handle_l4m_vision_request(self, client_identity, protocol_raw, payload):
        """Handle L4M vision requests."""
        try:
            request = l4m_vision_pb2.Request()
            request.ParseFromString(payload)
            
            # Note: Currently vision support is not implemented in the original backend
            # This is a placeholder for future implementation
            logger.warning("L4M vision requests are not yet implemented")

        except Exception as e:
            logger.error(f"Error handling L4M vision request: {e}")

    def _handle_ping_request(self, client_identity, protocol_raw, payload):
        """Handle ping requests."""
        try:
            ping = ping_pb2.Ping()
            ping.ParseFromString(payload)

            pong = ping_pb2.Pong(
                correlation_id=ping.correlation_id,
                message="Pong:" + ping.message
            ).SerializeToString()

            self.zmq_router.send_multipart([client_identity, protocol_raw, pong])

        except Exception as e:
            logger.error(f"Error handling ping request: {e}")

    def _handle_handshake(self, client_identity, frames, connected_clients, protocols):
        """Handle client handshake."""
        try:
            payload = frames[1]

            # Deserialize the protobuf message
            hs = handshake_pb2.Request()
            hs.ParseFromString(payload)

            # Send available protocols to the client
            response = handshake_pb2.Response(protocols=protocols)
            payload = response.SerializeToString()

            connected_clients[client_identity] = True
            self.zmq_router.send_multipart([client_identity, payload])

        except Exception:
            logger.error("Invalid handshake message")
            self.zmq_router.send_multipart([client_identity, b"\\x00"])


def main_run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Llama Backend Server with Management Agent')
    parser.add_argument('--ipc-endpoint', type=str, default='ipc:///tmp/pie-ipc',
                       help='IPC endpoint to bind to')
    parser.add_argument('--model-name', type=str, required=True,
                       help='Model name to load (e.g., meta-llama/Llama-3.1-8B-Instruct)')
    parser.add_argument('--management-service-url', type=str, default='http://localhost:9000',
                       help='URL of the engine management service')
    parser.add_argument('--backend-host', type=str, default='0.0.0.0',
                       help='Host for the backend management API')
    parser.add_argument('--backend-api-port', type=int, default=8082,
                       help='Port for the backend management API')
    parser.add_argument('--auto-load', action='store_true',
                       help='Automatically load the specified model on startup')

    args = parser.parse_args()

    # Create and start the Llama backend agent
    agent = LlamaBackendAgent(
        model_name=args.model_name,
        ipc_endpoint=args.ipc_endpoint,
        management_service_url=args.management_service_url,
        backend_host=args.backend_host,
        backend_api_port=args.backend_api_port
    )

    # Start the agent (registration, heartbeat, management API)
    if not agent.start():
        logger.error("Failed to start Llama backend agent")
        return 1

    logger.info(f"Llama backend agent started successfully")
    logger.info(f"Management API: http://{args.backend_host}:{args.backend_api_port}")
    logger.info(f"ZMQ endpoint: {args.ipc_endpoint}")

    # Auto-load model if requested
    if args.auto_load:
        logger.info(f"Auto-loading model: {args.model_name}")
        try:
            # Use asyncio to call the async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            request = ModelLoadRequest(model_name=args.model_name)
            result = loop.run_until_complete(agent._load_model_impl(request))

            if result.get("success", False):
                logger.info("Model auto-loaded successfully")
            else:
                logger.error(f"Failed to auto-load model: {result.get('error')}")

            loop.close()

        except Exception as e:
            logger.error(f"Error during model auto-load: {e}")

    # Keep the main thread alive
    try:
        agent.wait_until_stopped()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        agent.stop()
        logger.info("Llama backend shutdown complete")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main_run())
