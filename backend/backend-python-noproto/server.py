import random
import msgpack
import torch
from websockets.sync.client import connect
import zmq
import time
import fire
import os
import tomli
from pathlib import Path
from platformdirs import user_cache_dir
from config import parse_model_metadata
from driver import Driver
from l4ma import L4maForCausalLM, create_fusion_map
from qwen3 import Qwen3ForCausalLM, create_fusion_map as create_qwen3_fusion_map
import ztensor
from tqdm import tqdm
import threading  # Import the threading module


def main(config: str = None,
         host: str = 'localhost',
         port: int = 10123,
         controller_host: str = 'localhost',
         controller_port: int = 9123,
         auth_token: str = None,
         model: str = None,
         version: str = None,
         cache_dir: str = None,
         kv_page_size: int = 32,
         dist_size: int = 32,
         max_num_kv_pages: int = 1000,
         max_num_embeds: int = 50000,
         max_num_adapters: int = 48,
         max_adapter_rank: int = 8,
         device: str = 'cuda:0',
         dtype: str = 'bfloat16'):
    """
    Runs the application with the specified configuration.

    Args:
        config (str, optional): Path to a TOML configuration file.
        host (str, optional): The hostname. Defaults to 'localhost'.
        port (int, optional): The port number. Defaults to 10123.
        controller_host (str, optional): The controller hostname. Defaults to 'localhost'.
        controller_port (int, optional): The controller port number. Defaults to 9123.
        auth_token (str, optional): The authentication token. Defaults to None.
        model (str, optional): The model to use. Defaults to None.
        version (str, optional): The version of the model. Defaults to None.
        cache_dir (str, optional): The directory for caching. Defaults to a system-appropriate cache directory.
        kv_page_size (int, optional): The KV page size. Defaults to 32.
        dist_size (int, optional): The distribution size. Defaults to 32.
        max_num_kv_pages (int, optional): The maximum number of KV pages. Defaults to 1000.
        max_num_embeds (int, optional): The maximum number of embeddings. Defaults to 50000.
        device (str, optional): The device to use. Defaults to 'cuda:0'.
        dtype (str, optional): The data type. Defaults to 'bfloat16'.
    """
    config_from_file = {}
    if config:
        try:
            with open(config, "rb") as f:
                config_from_file = tomli.load(f)
        except FileNotFoundError:
            print(f"Warning: Config file not found at '{config}'")
        except tomli.TOMLDecodeError as e:
            print(f"Error decoding TOML file '{config}': {e}")

    # Prioritize CLI arguments over config file values
    final_config = {
        'host': host if host != 'localhost' else config_from_file.get('host', 'localhost'),
        'port': port if port != 10123 else config_from_file.get('port', 10123),
        'controller_host': controller_host if controller_host != 'localhost' else config_from_file.get(
            'controller_host', 'localhost'),
        'controller_port': controller_port if controller_port != 9123 else config_from_file.get('controller_port',
                                                                                                9123),
        'auth_token': auth_token if auth_token is not None else config_from_file.get('auth_token'),
        'model': model if model is not None else config_from_file.get('model'),
        'kv_page_size': kv_page_size if kv_page_size != 32 else config_from_file.get('kv_page_size', 32),
        'dist_size': dist_size if dist_size != 32 else config_from_file.get('dist_size', 32),
        'max_num_kv_pages': max_num_kv_pages if max_num_kv_pages != 1000 else config_from_file.get('max_num_kv_pages',
                                                                                                   1000),
        'max_num_embeds': max_num_embeds if max_num_embeds != 50000 else config_from_file.get('max_num_embeds', 50000),
        'max_num_adapters': max_num_adapters if max_num_adapters != 48 else config_from_file.get('max_num_adapters',
                                                                                                   2048),
        'max_adapter_rank': max_adapter_rank if max_adapter_rank != 8 else config_from_file.get('max_adapter_rank', 8),
        'device': device if device != 'cuda:0' else config_from_file.get('device', 'cuda:0'),
        'dtype': dtype if dtype != 'bfloat16' else config_from_file.get('dtype', 'bfloat16'),
    }

    # Special handling for cache-dir
    if cache_dir:
        final_config['cache_dir'] = cache_dir
    elif 'cache_dir' in config_from_file:
        final_config['cache_dir'] = config_from_file['cache_dir']
    elif os.environ.get('PIE_HOME'):
        final_config['cache_dir'] = os.environ.get('PIE_HOME')
    else:
        final_config['cache_dir'] = str(Path(user_cache_dir('pie')))

    print("--- Configuration ---")
    for key, value in final_config.items():
        print(f"{key}: {value}")
    print("----------------------")

    model, model_metadata = load_model(final_config)

    start_service(final_config, model, model_metadata)


def detect_architecture_type(metadata_path: str) -> str:
    """Detect the architecture type from the TOML file."""
    try:
        with open(metadata_path, "rb") as f:
            data = tomli.load(f)
        arch_data = data.get("architecture", {})
        return arch_data.get("type", "").lower()
    except Exception as e:
        raise RuntimeError(f"Failed to detect architecture type from {metadata_path}: {e}")



def load_model(config: dict):
    model_name = config.get('model')
    if not model_name:
        raise ValueError("Model name must be specified in config or arguments.")

    cache_dir = config.get('cache_dir')

    # Define the path to the model directory and metadata file
    model_path = os.path.join(cache_dir, "models")
    metadata_path = os.path.join(model_path, f"{model_name}.toml")

    # Read the metadata file
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config.get('device', 'cuda:0')
    model_dtype = getattr(torch, config.get('dtype', 'bfloat16'))
    model_info = ModelInfo.load_from_file(metadata_path, model_device, model_dtype)

    # 1. Map fused tensor names to their original sources
    # Choose the appropriate model based on architecture type
    if model_info.architecture.type.lower() == 'qwen3':
        model = Qwen3ForCausalLM(model_info.architecture)
        fusion_map = create_qwen3_fusion_map(model)
    else:
        # Default to L4MA for backward compatibility
        model = L4maForCausalLM(model_info.architecture)
        fusion_map = create_fusion_map(model)
    # 2. Create a reverse map for fast lookups (original source -> fused target)
    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    # 3. Buffer to hold source tensors until they are all collected
    pending_fusion_tensors = {}
    # ===================================================================

    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    # print(f"Found {len(metadata.parameters)} parameter file(s) to load.")

    try:
        for param_file in model_info.parameters:
            weights_path = os.path.join(model_path, model_name, param_file)
            # ... (existing file existence check) ...

            # print(f"Loading weights from ztensor file: {param_file}")
            with ztensor.Reader(weights_path) as reader:
                tensors_in_file = reader.get_tensor_names()

                for name in tqdm(tensors_in_file, desc=f"Loading {param_file}", unit="tensors"):
                    # Check if this tensor from the file is part of a planned fusion
                    if name in source_to_fusion_target:
                        tensor_data = reader.read_tensor(name, to="torch")
                        pending_fusion_tensors[name] = tensor_data
                        continue  # Skip direct loading, we'll fuse it later

                    # This is a standard, non-fused tensor
                    if name in model_state_keys and name not in loaded_keys:
                        param = model.state_dict()[name]
                        tensor_data = reader.read_tensor(name, to="torch")
                        if tensor_data.shape != param.shape:
                            print(f"    Warning: Shape mismatch for tensor '{name}'. Skipping.")
                            continue
                        with torch.no_grad():
                            param.copy_(tensor_data, non_blocking=True)
                        loaded_keys.add(name)

        # ================= ELEGANT FUSION HANDLING: PROCESSING =================
        # print("\nProcessing fused tensors...")
        for target_name, details in fusion_map.items():
            source_names = details["sources"]

            # Check if all required source tensors have been collected
            if all(s in pending_fusion_tensors for s in source_names):
                # Collect tensors in the correct order
                tensors_to_fuse = [pending_fusion_tensors[s] for s in source_names]

                # Concatenate them along the specified dimension
                fused_tensor = torch.cat(tensors_to_fuse, dim=details["dim"])

                # Load the newly created fused tensor into the model
                param = model.state_dict()[target_name]
                if fused_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for fused tensor '{target_name}'. ZT: {fused_tensor.shape}, Model: {param.shape}. Skipping.")
                    continue

                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)

                loaded_keys.add(target_name)
        # ========================================================================

        # L4ma models often reuse the embed_tokens for the lm_head, so we need to copy it explicitly
        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            model.state_dict()["lm_head.weight"].copy_(model.model.embed_tokens.weight, non_blocking=True)
            loaded_keys.add("lm_head.weight")

        # After trying all files, check if any keys are missing
        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        # Move the entire model to the specified device
        model.eval()  # Set the model to evaluation mode

        return model, model_info


    except ztensor.ZTensorError as e:
        print(f"Fatal Error: Failed to read a ztensor file. Error: {e}")
    except Exception as e:
        print(f"An unexpected fatal error occurred: {e}")


def start_service(config, model, model_metadata):
    """
    Initializes and starts the service, including the ZMQ server and registration threads.
    """
    if config.get("controller_host") in ["127.0.0.1", "localhost"]:
        unique_id = random.randint(1000, 9999)
        endpoint = f"ipc:///tmp/pie-service-{unique_id}"
        real_endpoint = endpoint
    else:
        endpoint = f"tcp://{config.get('host')}:{config.get('port')}"
        real_endpoint = f"tcp://*:{config.get('port')}"

    engine = Driver(model=model,
                    kv_page_size=config.get("kv_page_size"),
                    dist_size=config.get("dist_size"),
                    max_num_kv_pages=config.get("max_num_kv_pages"),
                    max_num_embeds=config.get("max_num_embeds"),
                    max_num_adapters=config.get("max_num_adapters"),
                    max_adapter_rank=config.get("max_adapter_rank"),
                    dtype=getattr(torch, config.get('dtype', 'bfloat16')),
                    device=config.get("device"))

    # ===================================================================
    # RUN THE TEST ROUTINE AT STARTUP
    # run_test_routine(engine)
    # ===================================================================

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(real_endpoint)

    # print(f"Server listening on {endpoint}")

    # Create and start the ZMQ server thread
    zmq_thread = threading.Thread(
        target=run_zmq_server,
        args=(router, engine, config, model_metadata),
        daemon=True  # Daemonize thread to exit when the main thread exits
    )
    zmq_thread.start()

    # Create and start the registration thread
    register_thread = threading.Thread(
        target=register,
        args=(config, endpoint),
        daemon=True
    )
    register_thread.start()

    # Keep the main thread alive to allow daemon threads to run, and handle shutdown
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        # Cleanly close ZMQ resources
        router.close()
        context.term()
        print("Server shutdown complete.")


def register(config, endpoint):
    """
    Registers the service with the controller. Runs in a separate thread.
    """
    # Notify the controller.
    try:
        auth_token = config.get("auth_token")
        with connect(f"ws://{config.get('controller_host')}:{config.get('controller_port')}") as websocket:
            # Do authentication
            websocket.send(msgpack.packb({
                "type": "authenticate",
                "corr_id": 0,
                "token": auth_token,
            }, use_bin_type=True))

            message = msgpack.unpackb(websocket.recv(), raw=False)

            if not message.get("successful"):
                print(f"Authentication failed: {message.get('result', 'Unknown error')}")
                exit(1)

            # Register the service
            websocket.send(msgpack.packb({
                "type": "attach_remote_service",
                "corr_id": 0,
                "endpoint": endpoint,
                "service_name": "example_service",
                "service_type": "l4m",
            }, use_bin_type=True))

            message = msgpack.unpackb(websocket.recv(), raw=False)
            if not message.get("successful"):
                print(f"Controller could not attack the backend: {message.get('result', 'Unknown error')}")
                exit(1)

            print(f"Registered with the controller at {config.get('controller_host')}:{config.get('controller_port')}")

    except ConnectionRefusedError:
        print(
            f"Failed to connect to the controller at {config.get('controller_host')}:{config.get('controller_port')}.")
        print("Please ensure the controller is running and accessible.")
    except Exception as e:
        print(f"An error occurred during registration: {e}")


def run_zmq_server(router, engine, config, model_metadata):
    """
    This function runs the ZMQ server loop in a dedicated thread.
    It listens for incoming client requests and processes them.
    """
    connected_clients = {}
    protocols = ["l4m", "l4m-vision", "ping"]
    idle_start = time.time()

    while True:
        try:
            # ROUTER sockets receive multipart messages.
            # Expected format: [client_identity, empty_frame, payload]
            frames = router.recv_multipart()
            # print(f"Idle time: {(time.time() - idle_start) * 1000}ms")

            client_identity = frames[0]
            start = time.time()

            # print("received", frames)

            # Check if the client has already established a protocol
            if client_identity in connected_clients:
                if len(frames) != 3:
                    print("Invalid message format.")
                    continue

                protocol_raw = frames[1]  # Should be a single byte
                protocol_idx = int.from_bytes(protocol_raw, byteorder="little")

                if protocol_idx >= len(protocols):
                    print("Invalid protocol:", protocol_idx)
                    continue

                protocol = protocols[protocol_idx]
                payload = frames[2]

                if protocol == "l4m":
                    request = l4m_pb2.Request()
                    request.ParseFromString(payload)
                    #print(payload, "@", request)
                    command = request.WhichOneof("command")
                    response = None

                    if command == "allocate":
                        engine.allocate(request.allocate)
                    elif command == "deallocate":
                        engine.deallocate(request.deallocate)
                    elif command == "embed_text":
                        engine.embed_text(request.embed_text)
                    elif command == "fill_block":
                        res = engine.fill_block(request.fill_block)
                        response = l4m_pb2.Response(correlation_id=request.correlation_id, batch_sync=res)
                    elif command == "forward_text":
                        res = engine.forward_text(request.forward_text)
                        response = l4m_pb2.Response(correlation_id=request.correlation_id, forward_text=res)
                    elif command == "mask_block":
                        engine.mask_block(request.mask_block)
                    elif command == "copy_block":
                        engine.copy_block(request.copy_block)
                    elif command == "decode_token_distribution":
                        engine.decode_token_distribution(request.decode_token_distribution)
                    elif command == "sample_top_k_request":
                        res = engine.sample_top_k_request(request.sample_top_k_request)
                        response = l4m_pb2.Response(correlation_id=request.correlation_id, sample_top_k=res)
                    elif command == "debug_query_request":
                        res = engine.debug_query_request(request.debug_query_request)
                        response = l4m_pb2.Response(correlation_id=request.correlation_id, debug_query=res)
                    elif command == "initialize_adapter":
                        engine.initialize_adapter(request.initialize_adapter)
                    elif command == "update_adapter":
                        engine.update_adapter(request.update_adapter)
                    elif command == "forward_with_adapter":
                        res = engine.forward_with_adapter(request.forward_with_adapter)
                        response = l4m_pb2.Response(correlation_id=request.correlation_id, forward_text=res)
                    elif command == "get_info":
                        print("Getting info from the engine.")
                        response = l4m_pb2.Response(correlation_id=request.correlation_id,
                                                    get_info=l4m_pb2.GetInfoResponse(
                                                        version="0.1",
                                                        model_name=f"{config.get('model')}-{config.get('version', '')}",
                                                        kv_page_size=config.get("kv_page_size"),
                                                        num_available_kv_pages=config.get("max_num_kv_pages"),
                                                        num_available_embeddings=config.get("max_num_embeds"),
                                                        num_available_adapters=config.get("max_num_adapters"),
                                                        tokenizer=l4m_pb2.Tokenizer(
                                                            merge_table=model_metadata.tokenizer.merge_table,
                                                            special_tokens=model_metadata.tokenizer.special_tokens,
                                                            split_regex=model_metadata.tokenizer.split_regex,
                                                            escape_non_printable=model_metadata.tokenizer.escape_non_printable,
                                                        )
                                                    ))
                    else:
                        print("No valid command found in request.", command)

                    if response is not None:
                        reply_payload = response.SerializeToString()
                        router.send_multipart([client_identity, protocol_raw, reply_payload])

                elif protocol == "l4m-vision":
                    request = l4m_vision_pb2.Request()
                    request.ParseFromString(payload)
                elif protocol == "ping":
                    ping = ping_pb2.Ping()
                    ping.ParseFromString(payload)
                    pong = ping_pb2.Pong(
                        correlation_id=ping.correlation_id,
                        message="Pong:" + ping.message
                    ).SerializeToString()
                    router.send_multipart([client_identity, protocol_raw, pong])
            else:
                # Handle handshake for new clients
                payload = frames[1]
                try:
                    hs = handshake_pb2.Request()
                    hs.ParseFromString(payload)
                except:
                    print("Invalid handshake message.")
                    router.send_multipart([client_identity, b"\x00"])
                    continue

                response = handshake_pb2.Response(protocols=protocols)
                payload = response.SerializeToString()
                connected_clients.update({client_identity: True})
                router.send_multipart([client_identity, payload])

            idle_start = time.time()
        except zmq.ZMQError as e:
            # This can happen if the context is terminated, which is expected on shutdown
            if e.errno == zmq.ETERM:
                print("ZMQ server thread exiting.")
                break
            else:
                raise


if __name__ == "__main__":
    fire.Fire(main)
