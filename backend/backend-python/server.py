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
import tomli
import ztensor
import zmq
from platformdirs import user_cache_dir
from tqdm import tqdm
from websockets.sync.client import connect

from config.common import ModelInfo
from handler import Handler
from model.l4ma import L4maForCausalLM, create_fusion_map as create_l4ma_fusion_map
from message import (EmbedImageRequest, ForwardPassRequest, HandshakeRequest,
                     InitializeAdapterRequest, QueryRequest, UpdateAdapterRequest)
from model.qwen3 import Qwen3ForCausalLM, create_fusion_map as create_qwen3_fusion_map


def get_final_config(config_path: str) -> dict:
    """
    Resolves the final configuration from a TOML file, environment variables, and defaults.

    The precedence order for settings is:
    1.  **TOML Configuration File**
    2.  **Environment Variables** (only for `cache_dir` via `PIE_HOME`)
    3.  **Default Values**

    Args:
        config_path: The path to the TOML configuration file.

    Returns:
        A dictionary containing the final, resolved configuration.
    """
    # 1. Default configuration values
    DEFAULTS = {
        'host': 'localhost',
        'port': 10123,
        'controller_host': 'localhost',
        'controller_port': 9123,
        'auth_token': None,
        'model': None,
        'cache_dir': None,
        'kv_page_size': 32,
        'dist_size': 32,
        'max_num_kv_pages': 1000,
        'max_num_embeds': 50000,
        'max_num_adapters': 48,
        'max_adapter_rank': 8,
        'device': 'cuda:0',
        'dtype': 'bfloat16'
    }

    # 2. Load configuration from TOML file
    config_from_file = {}
    if config_path:
        try:
            with open(config_path, "rb") as f:
                config_from_file = tomli.load(f)
        except FileNotFoundError:
            print(f"Error: Configuration file not found at '{config_path}'", file=sys.stderr)
            sys.exit(1)
        except tomli.TOMLDecodeError as e:
            print(f"Error decoding TOML file '{config_path}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: A configuration file must be provided via the --config argument.", file=sys.stderr)
        sys.exit(1)

    # 3. Build the final configuration dictionary
    final_config = DEFAULTS.copy()
    final_config.update(config_from_file)

    # 4. Handle `cache_dir` with its unique precedence: File > Env > Platform Default
    final_config['cache_dir'] = (
            config_from_file.get('cache_dir') or
            os.environ.get('PIE_HOME') or
            str(Path(user_cache_dir('pie')))
    )

    if not final_config.get('model'):
        print("Error: A 'model' must be specified in the config file.", file=sys.stderr)
        sys.exit(1)

    return final_config


def main(config: str):
    """
    Runs the application using settings from the specified configuration file.

    Args:
        config (str): The path to a TOML configuration file.
    """
    final_config = get_final_config(config)

    print("--- Configuration ---")
    for key, value in final_config.items():
        print(f"{key}: {value}")
    print("----------------------")

    model_instance, model_metadata = load_model(final_config)

    start_service(final_config, model_instance, model_metadata)


def load_model(config: dict):
    model_name = config['model']
    cache_dir = config['cache_dir']
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config['device']
    model_dtype = getattr(torch, config['dtype'])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    # Instantiate model and create fusion map based on architecture
    if model_info.architecture.type.lower() == 'qwen3':
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
                pbar_desc = f"Loading {param_file[:30]}..." if len(param_file) > 30 else f"Loading {param_file}"
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    # If tensor is part of a fusion, buffer it
                    if name in source_to_fusion_target:
                        pending_fusion_tensors[name] = reader.read_tensor(name, to="torch")
                        continue

                    # Load standard, non-fused tensor
                    if name in model_state_keys and name not in loaded_keys:
                        param = model.state_dict()[name]
                        tensor_data = reader.read_tensor(name, to="torch")
                        if tensor_data.shape != param.shape:
                            print(f"    Warning: Shape mismatch for tensor '{name}'. Skipping.")
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
                    print(f"    Warning: Shape mismatch for fused tensor '{target_name}'. Skipping.")
                    continue
                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)
                loaded_keys.add(target_name)

        # Handle weight tying for lm_head
        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            model.state_dict()["lm_head.weight"].copy_(model.model.embed_tokens.weight, non_blocking=True)
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
        print(f"Fatal Error: Failed to read a ztensor file. Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected fatal error occurred during model loading: {e}", file=sys.stderr)
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

    handler = Handler(model=model,
                      model_info=model_info,
                      kv_page_size=config["kv_page_size"],
                      dist_size=config["dist_size"],
                      max_num_kv_pages=config["max_num_kv_pages"],
                      max_num_embeds=config["max_num_embeds"],
                      max_num_adapters=config["max_num_adapters"],
                      max_adapter_rank=config["max_adapter_rank"],
                      dtype=getattr(torch, config['dtype']),
                      device=config["device"])

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
            websocket.send(msgpack.packb({
                "type": "authenticate", "corr_id": 0, "token": config["auth_token"]
            }, use_bin_type=True))
            auth_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not auth_response.get("successful"):
                print(f"Authentication failed: {auth_response.get('result', 'Unknown error')}")
                sys.exit(1)

            # Register the service endpoint
            websocket.send(msgpack.packb({
                "type": "attach_remote_service", "corr_id": 0, "endpoint": endpoint,
                "service_name": "example_service", "service_type": "l4m"
            }, use_bin_type=True))
            reg_response = msgpack.unpackb(websocket.recv(), raw=False)
            if not reg_response.get("successful"):
                print(f"Controller registration failed: {reg_response.get('result', 'Unknown error')}")
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
                corr_id = struct.unpack('>I', corr_id_bytes)[0]
                handler_id = struct.unpack('>I', handler_id_bytes)[0]
                reqs = [DECODERS[handler_id].decode(m) for m in message[3:]]
            except (struct.error, KeyError, msgspec.DecodeError) as e:
                print(f"[!] Error decoding request header or payload: {e}", file=sys.stderr)
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
                response_msg = [client_identity, corr_id_bytes, handler_id_bytes] + \
                               [MSGPACK_ENCODER.encode(r) for r in resps]
                socket.send_multipart(response_msg)

        except zmq.ZMQError as e:
            print(f"ZMQ Error in server loop: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    fire.Fire(main)
