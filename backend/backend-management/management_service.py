#!/usr/bin/env python3
"""
Symphony Management Service

A long-running daemon that manages backend model instances and handles
client handshakes, providing dynamic routing to model-specific endpoints.
"""

import os
import sys
import time
import uuid
import json
import signal
import logging
import argparse
import subprocess
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import zmq

try:
    import handshake_pb2
except ImportError:
    print("Error: Could not import handshake_pb2. Make sure protobuf files are generated.")
    print("Run './build_proto.sh' to generate protobuf files.")
    sys.exit(1)


@dataclass
class ModelInstance:
    """Represents a running model backend instance."""
    model_name: str
    model_type: str
    endpoint: str
    process: subprocess.Popen
    config_path: Optional[str] = None
    started_at: float = 0.0
    
    def __post_init__(self):
        if self.started_at == 0.0:
            self.started_at = time.time()
    
    def is_alive(self) -> bool:
        """Check if the backend process is still running."""
        return self.process.poll() is None
    
    def terminate(self):
        """Gracefully terminate the backend process."""
        if self.is_alive():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()


@dataclass 
class ManagementCommand:
    """Represents a command from the CLI tool to the management service."""
    command: str
    params: Dict[str, Any]
    correlation_id: str = ""
    
    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


class ManagementService:
    """
    The main management service that handles client handshakes and manages
    model backend instances.
    """
    
    def __init__(self, 
                 config_path: str = None,
                 backend_base_path: str = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set endpoints from config
        self.client_endpoint = self.config["endpoints"]["client_handshake"]
        self.cli_endpoint = self.config["endpoints"]["cli_management"]
        
        self.backend_base_path = backend_base_path or self._get_default_backend_path()
        
        # Registry of running model instances
        self.model_instances: Dict[str, ModelInstance] = {}
        
        # Model type to backend script mapping (loaded from config)
        self.model_backends = self.config["model_backends"]
        
        # Model name to type mapping (loaded from config)
        self.model_type_mapping = {}
        for model_info in self.config.get("supported_models", []):
            self.model_type_mapping[model_info["name"]] = model_info["type"]

        # ZMQ setup
        self.context = zmq.Context()
        self.client_router = None  # For client handshakes
        self.cli_router = None     # For CLI commands
        
        # Shutdown flag and lock for thread safety
        self.shutdown_requested = False
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def _get_default_backend_path(self) -> str:
        """Get the default path to backend scripts.
        
        This method tries to locate the backend-flashinfer directory using
        multiple strategies to ensure it works regardless of how the service
        is invoked (from CLI, tests, or different working directories).
        """
        # Strategy 1: Use the file's location to find backend directory
        # This works when the file structure is intact
        current_file_dir = Path(__file__).parent.resolve()
        backend_dir = current_file_dir.parent / "backend-flashinfer"
        
        if backend_dir.exists():
            return str(backend_dir)
        
        # Strategy 2: Look for backend directory relative to current working directory
        # This handles cases where the service is run from the project root
        cwd_backend = Path.cwd() / "backend" / "backend-flashinfer"
        if cwd_backend.exists():
            return str(cwd_backend)
        
        # Strategy 3: Walk up the directory tree to find the symphony project root
        # Look for characteristic files/directories that indicate project root
        current_path = current_file_dir
        for _ in range(5):  # Limit search depth to avoid infinite loops
            # Check if this looks like the symphony project root
            if ((current_path / "backend").exists() and 
                (current_path / "engine").exists() and
                (current_path / "example-apps").exists()):
                
                potential_backend = current_path / "backend" / "backend-flashinfer"
                if potential_backend.exists():
                    return str(potential_backend)
            
            # Move up one directory
            parent = current_path.parent
            if parent == current_path:  # Reached filesystem root
                break
            current_path = parent
        
        # Strategy 4: Return the original calculated path as fallback
        # Even if it doesn't exist, let the calling code handle the error
        print(f"Warning: Could not find backend-flashinfer directory, using fallback path: {backend_dir}")
        return str(backend_dir)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get("format", "%(asctime)s [%(levelname)8s] %(name)s: %(message)s")
        date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
        
        # Setup root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            datefmt=date_format,
            force=True  # Override any existing configuration
        )
        
        self.logger = logging.getLogger('symphony-mgmt')
        
        # Also setup a separate logger for subprocess output
        self.subprocess_logger = logging.getLogger('symphony-backend')
    
    def _generate_unique_endpoint(self) -> str:
        """Generate a unique IPC endpoint for a model instance."""
        instance_id = str(uuid.uuid4())[:8]
        return f"ipc:///tmp/symphony-model-{instance_id}"
    
    def start_service(self):
        """Start the management service."""
        self.logger.info(f"Starting Symphony Management Service")
        self.logger.info(f"Client endpoint: {self.client_endpoint}")
        self.logger.info(f"CLI endpoint: {self.cli_endpoint}")
        
        # Setup signal handlers (only in main thread)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except ValueError:
            # Signal handling not available in threads
            self.logger.debug("Signal handlers not available (running in thread)")
        
        # Setup ZMQ sockets
        self.client_router = self.context.socket(zmq.ROUTER)
        self.cli_router = self.context.socket(zmq.ROUTER)
        
        try:
            self.client_router.bind(self.client_endpoint)
            self.cli_router.bind(self.cli_endpoint)
        except zmq.ZMQError as e:
            self.logger.error(f"Failed to bind to endpoints: {e}")
            return False
            
        self.logger.info("Management service started successfully")
        
        # Main event loop
        try:
            self._main_loop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self._cleanup()
        
        return True
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
    
    def _main_loop(self):
        """Main event loop handling incoming messages."""
        poller = zmq.Poller()
        poller.register(self.client_router, zmq.POLLIN)
        poller.register(self.cli_router, zmq.POLLIN)
        
        while not self.shutdown_requested:
            try:
                # Poll with timeout to allow checking shutdown flag
                socks = dict(poller.poll(timeout=1000))
                
                if self.client_router in socks and socks[self.client_router] == zmq.POLLIN:
                    self._handle_client_message()
                    
                if self.cli_router in socks and socks[self.cli_router] == zmq.POLLIN:
                    self._handle_cli_message()
                    
                # Periodically check health of model instances
                self._check_model_instances_health()
                    
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break
                self.logger.error(f"ZMQ error in main loop: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
    
    def _handle_client_message(self):
        """Handle an incoming message from a client (handshake request)."""
        try:
            frames = self.client_router.recv_multipart(zmq.NOBLOCK)
            
            if len(frames) < 2:
                self.logger.warning("Received invalid client message format")
                return
                
            client_identity = frames[0]
            payload = frames[1]
            
            # Parse as handshake request
            try:
                handshake_req = handshake_pb2.Request()
                handshake_req.ParseFromString(payload)
                self._handle_client_handshake(client_identity, handshake_req)
            except Exception as e:
                self.logger.error(f"Error parsing client handshake: {e}")
                
        except zmq.Again:
            # No message available
            pass
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
    
    def _parse_cli_message_frames(self, frames):
        """Parse CLI message frames using DEALER-ROUTER pattern.
        
        DEALER-ROUTER pattern (enforced):
        - DEALER socket sends: [payload]
        - ROUTER receives: [identity, payload]
        
        This is cleaner than REQ-ROUTER which has an extra empty delimiter frame.
        
        Returns:
            tuple: (cli_identity, payload) or (None, None) if invalid
        """
        if len(frames) != 2:
            self.logger.error(f"Invalid CLI message: expected exactly 2 frames (DEALER-ROUTER), got {len(frames)}")
            return None, None
            
        cli_identity, payload = frames
        return cli_identity, payload

    def _send_cli_response(self, cli_identity, response_data):
        """Send response back to CLI using DEALER-ROUTER pattern.
        
        DEALER-ROUTER response format:
        - ROUTER sends: [identity, response]
        - DEALER receives: [response]
        """
        response_payload = json.dumps(response_data).encode('utf-8')
        self.cli_router.send_multipart([cli_identity, response_payload])

    def _handle_cli_message(self):
        """Handle an incoming message from CLI tool."""
        try:
            frames = self.cli_router.recv_multipart(zmq.NOBLOCK)
            
            # Parse message frames
            cli_identity, payload = self._parse_cli_message_frames(frames)
            if cli_identity is None:
                return
            
            # Parse as CLI command (JSON)
            try:
                command_data = json.loads(payload.decode('utf-8'))
                command = ManagementCommand(**command_data)
                self._handle_cli_command(cli_identity, command)
            except Exception as e:
                self.logger.error(f"Error parsing CLI command: {e}")
                # Send structured error response
                error_response = {
                    "success": False,
                    "error": f"Invalid command format: {e}",
                    "correlation_id": "unknown"
                }
                self._send_cli_response(cli_identity, error_response)
                
        except zmq.Again:
            # No message available
            pass
        except Exception as e:
            self.logger.error(f"Error handling CLI message: {e}")
    
    def _handle_client_handshake(self, client_identity: bytes, request: handshake_pb2.Request):
        """Handle a client handshake request."""
        self.logger.info("Handling client handshake request")
        
        # For now, assume clients want "llama" model if not specified
        # TODO: Update handshake.proto to include requested_model_type
        requested_model = "llama"
        
        try:
            # Get or create model instance
            endpoint = self._get_or_create_model_instance(requested_model)
            
            if endpoint:
                # Send successful response with model endpoint
                response = handshake_pb2.Response()
                response.protocols.extend(["l4m", "l4m-vision", "ping"])
                # TODO: Add model_ipc_endpoint field to handshake.proto
                
                response_payload = response.SerializeToString()
                self.client_router.send_multipart([client_identity, response_payload])
                
                self.logger.info(f"Provided endpoint {endpoint} to client")
            else:
                # Send error response
                response = handshake_pb2.Response()
                # TODO: Add error field to handshake.proto
                response_payload = response.SerializeToString()
                self.client_router.send_multipart([client_identity, response_payload])
                
        except Exception as e:
            self.logger.error(f"Error in client handshake: {e}")
    
    def _handle_cli_command(self, cli_identity: bytes, command: ManagementCommand):
        """Handle a CLI command."""
        self.logger.info(f"Handling CLI command: {command.command}")
        
        response = {"correlation_id": command.correlation_id, "success": False, "data": {}}
        
        try:
            if command.command == "status":
                response = self._handle_status_command(command)
            elif command.command == "load-model":
                response = self._handle_load_model_command(command)
            elif command.command == "unload-model":
                response = self._handle_unload_model_command(command)
            elif command.command == "stop-service":
                response = self._handle_stop_service_command(command)
            else:
                response["error"] = f"Unknown command: {command.command}"
                
        except Exception as e:
            self.logger.error(f"Error handling CLI command {command.command}: {e}")
            response["error"] = str(e)
        
        # Send response back to CLI using structured method
        self._send_cli_response(cli_identity, response)
    
    def _handle_status_command(self, command: ManagementCommand) -> Dict[str, Any]:
        """Handle status command from CLI."""
        running_models = []
        for model_name, instance in self.model_instances.items():
            if instance.is_alive():
                running_models.append({
                    "name": model_name,
                    "type": instance.model_type,
                    "endpoint": instance.endpoint,
                    "started_at": instance.started_at,
                    "uptime": time.time() - instance.started_at
                })
        
        return {
            "correlation_id": command.correlation_id,
            "success": True,
            "data": {
                "service_status": "running",
                "client_endpoint": self.client_endpoint,
                "cli_endpoint": self.cli_endpoint,
                "models": running_models
            }
        }
    
    def _handle_load_model_command(self, command: ManagementCommand) -> Dict[str, Any]:
        """Handle load-model command from CLI."""
        model_name = command.params.get("model_name")
        config_path = command.params.get("config_path")
        
        if not model_name:
            return {
                "correlation_id": command.correlation_id,
                "success": False,
                "error": "model_name parameter required"
            }
        
        # Validate model exists in configuration early
        try:
            model_type = self._get_model_type(model_name)
            if model_type not in self.model_backends:
                return {
                    "correlation_id": command.correlation_id,
                    "success": False,
                    "error": f"Backend for model type '{model_type}' not configured"
                }
        except ValueError as e:
            return {
                "correlation_id": command.correlation_id,
                "success": False,
                "error": str(e)
            }
        
        # Config path should be ignored as config is fixed at service launch time
        if config_path:
            self.logger.warning(f"Ignoring config_path parameter in load-model command: {config_path}")
        
        try:
            endpoint = self._get_or_create_model_instance(model_name, None)
            return {
                "correlation_id": command.correlation_id,
                "success": True,
                "data": {"endpoint": endpoint}
            }
        except Exception as e:
            return {
                "correlation_id": command.correlation_id,
                "success": False,
                "error": str(e)
            }
    
    def _handle_unload_model_command(self, command: ManagementCommand) -> Dict[str, Any]:
        """Handle unload-model command from CLI."""
        model_name = command.params.get("model_name")
        
        if not model_name:
            return {
                "correlation_id": command.correlation_id,
                "success": False,
                "error": "model_name parameter required"
            }
        
        if model_name in self.model_instances:
            instance = self.model_instances[model_name]
            instance.terminate()
            del self.model_instances[model_name]
            self.logger.info(f"Unloaded model: {model_name}")
            
            return {
                "correlation_id": command.correlation_id,
                "success": True,
                "data": {"message": f"Model {model_name} unloaded"}
            }
        else:
            return {
                "correlation_id": command.correlation_id,
                "success": False,
                "error": f"Model {model_name} not found"
            }
    
    def _handle_stop_service_command(self, command: ManagementCommand) -> Dict[str, Any]:
        """Handle stop-service command from CLI."""
        self.shutdown_requested = True
        
        return {
            "correlation_id": command.correlation_id,
            "success": True,
            "data": {"message": "Service shutdown requested"}
        }
    
    def _get_or_create_model_instance(self, model_name: str, config_path: str = None) -> Optional[str]:
        """Get existing model instance or create a new one."""
        # Check if model is already running
        if model_name in self.model_instances:
            instance = self.model_instances[model_name]
            if instance.is_alive():
                self.logger.info(f"Using existing model instance: {model_name}")
                return instance.endpoint
            else:
                # Clean up dead instance
                del self.model_instances[model_name]
        
        # Create new instance
        return self._create_model_instance(model_name, config_path)
    
    def _create_model_instance(self, model_name: str, config_path: str = None) -> Optional[str]:
        """Create a new model backend instance."""
        # Determine model type and backend script
        try:
            model_type = self._get_model_type(model_name)
        except ValueError as e:
            self.logger.error(f"Failed to determine model type: {e}")
            return None
            
        if model_type not in self.model_backends:
            self.logger.error(f"Unknown model type: {model_type}")
            return None
        
        backend_script = self.model_backends[model_type]
        backend_path = os.path.join(self.backend_base_path, backend_script)
        
        if not os.path.exists(backend_path):
            self.logger.error(f"Backend script not found: {backend_path}")
            return None
        
        # Generate unique endpoint
        endpoint = self._generate_unique_endpoint()
        
        # Build command to start backend
        cmd = [
            sys.executable, backend_path,
            "--ipc-endpoint", endpoint
        ]
        
        if config_path:
            cmd.extend(["--config", config_path])
        
        self.logger.info(f"Starting backend process: {' '.join(cmd)}")
        
        try:
            # Start backend process with piped output
            process = subprocess.Popen(cmd, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True,
                                     bufsize=1,
                                     universal_newlines=True)
            
            # Start a thread to log subprocess output
            import threading
            def log_subprocess_output():
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line and line.strip():
                            # Handle mock objects in tests
                            line_str = str(line).strip() if hasattr(line, 'strip') else str(line)
                            if not line_str.startswith('<Mock'):  # Skip mock object strings
                                try:
                                    self.subprocess_logger.info(f"[{model_name}] {line_str}")
                                except (ValueError, OSError):
                                    # Logger or file handler might be closed
                                    break
                    process.stdout.close()
                except (ValueError, AttributeError) as e:
                    # Handle closed file or mock objects
                    self.logger.debug(f"Subprocess output logging ended: {e}")
            
            log_thread = threading.Thread(target=log_subprocess_output, daemon=True)
            log_thread.start()
            
            # Give it a moment to start
            time.sleep(5)
            
            if process.poll() is not None:
                # Process died immediately
                self.logger.error(f"Backend process for {model_name} failed to start")
                return None
            
            # Create instance record
            instance = ModelInstance(
                model_name=model_name,
                model_type=model_type,
                endpoint=endpoint,
                process=process,
                config_path=config_path
            )
            
            self.model_instances[model_name] = instance
            self.logger.info(f"Started model instance: {model_name} on {endpoint}")
            
            return endpoint
            
        except Exception as e:
            self.logger.error(f"Failed to start model instance {model_name}: {e}")
            return None
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name using configuration mapping."""
        # Try exact match with model name (case-sensitive)
        if model_name in self.model_type_mapping:
            return self.model_type_mapping[model_name]
        
        # If no exact match found, check for similar names with different capitalization
        similar_names = []
        for config_model_name in self.model_type_mapping.keys():
            if model_name.lower() == config_model_name.lower():
                similar_names.append(config_model_name)
        
        # If found similar names with different capitalization, suggest them
        if similar_names:
            if len(similar_names) == 1:
                raise ValueError(f"Unknown model '{model_name}' - not found in configuration. Similar model name found. Do you mean: {similar_names[0]}?")
            else:
                similar_list = ", ".join(similar_names)
                raise ValueError(f"Unknown model '{model_name}' - not found in configuration. Similar model names found. Do you mean: {similar_list}?")
        
        # If no similar names found, raise a generic error
        raise ValueError(f"Unknown model '{model_name}' - not found in configuration")
    
    def _check_model_instances_health(self):
        """Check health of all model instances and clean up dead ones."""
        dead_instances = []
        
        for model_name, instance in self.model_instances.items():
            if not instance.is_alive():
                self.logger.warning(f"Model instance {model_name} has died")
                dead_instances.append(model_name)
        
        # Clean up dead instances
        for model_name in dead_instances:
            del self.model_instances[model_name]
    
    def _cleanup(self):
        """Cleanup resources before shutdown."""
        with self._cleanup_lock:
            if self._cleaned_up:
                return  # Already cleaned up
                
            self.logger.info("Cleaning up management service...")
            
            # Terminate all model instances
            for model_name, instance in self.model_instances.items():
                self.logger.info(f"Terminating model instance: {model_name}")
                try:
                    instance.terminate()
                except Exception as e:
                    self.logger.warning(f"Error terminating model instance {model_name}: {e}")
            
            # Close ZMQ sockets with proper error handling
            if self.client_router:
                try:
                    self.client_router.close()
                except Exception as e:
                    self.logger.warning(f"Error closing client router: {e}")
                self.client_router = None
                
            if self.cli_router:
                try:
                    self.cli_router.close()
                except Exception as e:
                    self.logger.warning(f"Error closing CLI router: {e}")
                self.cli_router = None

            # Terminate ZMQ context with proper error handling and thread safety
            if hasattr(self, 'context') and self.context:
                try:
                    # Use a short linger to avoid hanging
                    self.context.setsockopt(zmq.LINGER, 100)  # 100ms linger
                    self.context.term()
                except Exception as e:
                    self.logger.warning(f"Error terminating ZMQ context: {e}")
                finally:
                    self.context = None
                    
            self._cleaned_up = True
            self.logger.info("Management service cleanup complete")
    
    def initialize_sockets(self):
        """Initialize ZMQ sockets without starting the main loop (for testing)."""
        if self.client_router is not None or self.cli_router is not None:
            return True  # Already initialized
            
        self.client_router = self.context.socket(zmq.ROUTER)
        self.cli_router = self.context.socket(zmq.ROUTER)
        
        try:
            self.client_router.bind(self.client_endpoint)
            self.cli_router.bind(self.cli_endpoint)
            self.logger.info("Sockets initialized successfully")
            return True
        except zmq.ZMQError as e:
            self.logger.error(f"Failed to bind sockets: {e}")
            return False

    def handle_single_message(self, timeout=100):
        """Handle a single message with timeout (for testing)."""
        if not self.client_router or not self.cli_router:
            return False
            
        poller = zmq.Poller()
        poller.register(self.client_router, zmq.POLLIN)
        poller.register(self.cli_router, zmq.POLLIN)
        
        try:
            socks = dict(poller.poll(timeout))
            
            if self.client_router in socks:
                self._handle_client_message()
                return True
            elif self.cli_router in socks:
                self._handle_cli_message()
                return True
                
        except zmq.ZMQError as e:
            self.logger.error(f"Error polling sockets: {e}")
            
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Symphony Management Service")
    parser.add_argument("--config", 
                       help="Path to configuration file (default: config.json in script directory)")
    parser.add_argument("--backend-path", 
                       help="Path to backend scripts directory")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (overrides config file)")
    
    args = parser.parse_args()
    
    # Create and start service
    service = ManagementService(
        config_path=args.config,
        backend_base_path=args.backend_path
    )
    
    # Override log level if specified via command line
    if args.log_level != "INFO":
        logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    success = service.start_service()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
