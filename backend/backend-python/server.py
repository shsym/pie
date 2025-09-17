import os
import random
import struct
import sys
import threading
import time
from pathlib import Path

import fire
import msgpack
import msgspec
import torch
import ztensor
import zmq
from msgspec import structs
from platformdirs import user_cache_dir
from tqdm import tqdm
from websockets.sync.client import connect

from config.common import ModelInfo
from handler import Handler
from model.l4ma import L4maForCausalLM, create_fusion_map as create_l4ma_fusion_map
from message import (
    EmbedImageRequest,
    ForwardPassRequest,
    HandshakeRequest,
    InitializeAdapterRequest,
    QueryRequest,
    UpdateAdapterRequest,
)
from model.qwen3 import Qwen3ForCausalLM, create_fusion_map as create_qwen3_fusion_map


def main(
    model: str,
    host: str = "localhost",
    port: int = 10123,
    controller_host: str = "localhost",
    controller_port: int = 9123,
    auth_token: str = None,
    cache_dir: str = None,
    kv_page_size: int = 16,
    max_dist_size: int = 64,
    max_num_kv_pages: int = 1024,
    max_num_embeds: int = 128,
    max_num_adapters: int = 48,
    max_adapter_rank: int = 8,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
):
    """
    Runs the application with configuration provided as command-line arguments.

    Args:
        model: Name of the model to load (required).
        host: Hostname for the ZMQ service to bind to.
        port: Port for the ZMQ service to bind to.
        controller_host: Hostname of the controller to register with.
        controller_port: Port of the controller to register with.
        auth_token: Authentication token for connecting to the controller.
        cache_dir: Directory for model cache. Defaults to PIE_HOME env var,
                   then the platform-specific user cache dir.
        kv_page_size: The size of each page in the key-value cache.
        max_dist_size: Maximum distance for embeddings.
        max_num_kv_pages: Maximum number of pages in the key-value cache.
        max_num_embeds: Maximum number of embeddings to store.
        max_num_adapters: Maximum number of adapters that can be loaded.
        max_adapter_rank: Maximum rank for any loaded adapter.
        device: The device to run the model on (e.g., 'cuda:0' or 'cpu').
        dtype: The data type for model weights (e.g., 'bfloat16', 'float16').
    """
    # Resolve cache_dir using the precedence: CLI arg > Environment Var > Platform Default
    resolved_cache_dir = (
        cache_dir or os.environ.get("PIE_HOME") or str(Path(user_cache_dir("pie")))
    )

    # Create a config dictionary from function arguments for downstream use.
    # The 'locals()' function conveniently captures all arguments.
    config = locals()
    config["cache_dir"] = resolved_cache_dir  # Update with the resolved path

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("----------------------")

    model_instance, model_metadata = load_model(config)
    start_service(config, model_instance, model_metadata)


def load_model(config: dict):
    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    # Instantiate model and create fusion map based on architecture
    if model_info.architecture.type.lower() == "qwen3":
        model = Qwen3ForCausalLM(model_info.architecture)
        fusion_map = create_qwen3_fusion_map(model)
    else:
        model = L4maForCausalLM(model_info.architecture)
        fusion_map = create_l4ma_fusion_map(model)

    # Create a reverse map for quick lookup of fusion targets
    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    pending_fusion_tensors = {}
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    try:
        for param_file in model_info.parameters:
            weights_path = model_path / model_name / param_file
            with ztensor.Reader(str(weights_path)) as reader:
                tensor_names = reader.get_tensor_names()
                pbar_desc = (
                    f"Loading {param_file[:30]}..."
                    if len(param_file) > 30
                    else f"Loading {param_file}"
                )
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    # If tensor is part of a fusion, buffer it
                    if name in source_to_fusion_target:
                        pending_fusion_tensors[name] = reader.read_tensor(
                            name, to="torch"
                        )
                        continue

                    # Load standard, non-fused tensor
                    if name in model_state_keys and name not in loaded_keys:
                        param = model.state_dict()[name]
                        tensor_data = reader.read_tensor(name, to="torch")
                        if tensor_data.shape != param.shape:
                            print(
                                f"    Warning: Shape mismatch for tensor '{name}'. Skipping."
                            )
                            continue
                        with torch.no_grad():
                            param.copy_(tensor_data, non_blocking=True)
                        loaded_keys.add(name)

        # Process all buffered tensors for fusion
        for target_name, details in fusion_map.items():
            source_names = details["sources"]
            if all(s in pending_fusion_tensors for s in source_names):
                tensors_to_fuse = [pending_fusion_tensors[s] for s in source_names]
                fused_tensor = torch.cat(tensors_to_fuse, dim=details["dim"])
                param = model.state_dict()[target_name]
                if fused_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for fused tensor '{target_name}'. Skipping."
                    )
                    continue
                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)
                loaded_keys.add(target_name)

        # Handle weight tying for lm_head
        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            model.state_dict()["lm_head.weight"].copy_(
                model.model.embed_tokens.weight, non_blocking=True
            )
            loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model, model_info

    except ztensor.ZTensorError as e:
        print(
            f"Fatal Error: Failed to read a ztensor file. Error: {e}", file=sys.stderr
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"An unexpected fatal error occurred during model loading: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def start_service(config, model, model_info):
    """
    Initializes and starts the service, including the ZMQ server and registration threads.
    """
    if config["controller_host"] in ["127.0.0.1", "localhost"]:
        unique_id = random.randint(1000, 9999)
        endpoint = f"ipc:///tmp/pie-service-{unique_id}"
        real_endpoint = endpoint
    else:
        endpoint = f"tcp://{config['host']}:{config['port']}"
        real_endpoint = f"tcp://*:{config['port']}"

    handler = Handler(
        model=model,
        model_info=model_info,
        kv_page_size=config["kv_page_size"],
        max_dist_size=config["max_dist_size"],
        max_num_kv_pages=config["max_num_kv_pages"],
        max_num_embeds=config["max_num_embeds"],
        max_num_adapters=config["max_num_adapters"],
        max_adapter_rank=config["max_adapter_rank"],
        dtype=getattr(torch, config["dtype"]),
        device=config["device"],
    )

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(real_endpoint)

    # Start ZMQ server and registration in daemon threads
    threading.Thread(target=run_zmq_server, args=(socket, handler), daemon=True).start()
    threading.Thread(target=register, args=(config, endpoint), daemon=True).start()

    # Keep main thread alive to handle shutdown
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        socket.close()
        context.term()
        print("Server shutdown complete.")


def register(config, endpoint):
    """
    Registers this service with the controller in a separate thread.
    """
    controller_addr = f"ws://{config['controller_host']}:{config['controller_port']}"
    try:
        with connect(controller_addr) as websocket:
            # Authenticate with the controller
            websocket.send(
                msgpack.packb(
                    {
                        "type": "authenticate",
                        "corr_id": 0,
                        "token": config["auth_token"],
                    },
                    use_bin_type=True,
                )
            )
            auth_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not auth_response.get("successful"):
                print(
                    f"Authentication failed: {auth_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            # Register the service endpoint
            websocket.send(
                msgpack.packb(
                    {
                        "type": "attach_remote_service",
                        "corr_id": 0,
                        "endpoint": endpoint,
                        "service_name": config["model"],
                        "service_type": "model",
                    },
                    use_bin_type=True,
                )
            )
            reg_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not reg_response.get("successful"):
                print(
                    f"Controller registration failed: {reg_response.get('result', 'Unknown error')}"
                )
                sys.exit(1)

            print(f"Registered with controller at {controller_addr}")

    except ConnectionRefusedError:
        print(f"Failed to connect to the controller at {controller_addr}.")
        print("Please ensure the controller is running and accessible.")
    except Exception as e:
        print(f"An error occurred during registration: {e}")


def run_zmq_server(socket, handler):
    """
    Runs the ZMQ server loop, listening for and processing client requests.
    """
    MSGPACK_ENCODER = msgspec.msgpack.Encoder()
    DECODERS = {
        0: msgspec.msgpack.Decoder(HandshakeRequest),
        1: msgspec.msgpack.Decoder(QueryRequest),
        2: msgspec.msgpack.Decoder(ForwardPassRequest),
        3: msgspec.msgpack.Decoder(EmbedImageRequest),
        4: msgspec.msgpack.Decoder(InitializeAdapterRequest),
        5: msgspec.msgpack.Decoder(UpdateAdapterRequest),
    }

    while True:
        try:
            # ROUTER sockets expect [client_id, corr_id, handler_id, payload...]
            message = socket.recv_multipart()

            if len(message) < 3:
                print(f"[!] Received invalid message: {message}", file=sys.stderr)
                continue

            client_identity, corr_id_bytes, handler_id_bytes = message[:3]
            try:
                corr_id = struct.unpack(">I", corr_id_bytes)[0]
                handler_id = struct.unpack(">I", handler_id_bytes)[0]
                reqs = [DECODERS[handler_id].decode(m) for m in message[3:]]
            except (struct.error, KeyError, msgspec.DecodeError) as e:
                print(
                    f"[!] Error decoding request header or payload: {e}",
                    file=sys.stderr,
                )
                continue

            if not reqs:
                print(f"[!] Received empty request body", file=sys.stderr)
                continue

            resps = []
            match handler_id:
                case 0:
                    resps = handler.handshake(reqs)
                case 1:
                    resps = handler.query(reqs)
                case 2:
                    resps = handler.forward_pass(reqs)
                case 3:
                    handler.embed_image(reqs)
                case 4:
                    handler.initialize_adapter(reqs)
                case 5:
                    handler.update_adapter(reqs)
                case _:
                    print(f"[!] Unknown handler ID: {handler_id}", file=sys.stderr)

            if resps:
                response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + [
                    MSGPACK_ENCODER.encode(r) for r in resps
                ]
                socket.send_multipart(response_msg)

        except zmq.ZMQError as e:
            print(f"ZMQ Error in server loop: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    fire.Fire(main)
