import asyncio
import msgpack
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from websockets.asyncio.client import connect
import zmq
import time
import fire
import os
import tomli
from pathlib import Path
from platformdirs import user_cache_dir
import math
import handshake_pb2
import l4m_pb2
import l4m_vision_pb2
import ping_pb2
from config import MAX_NUM_PAGES, MAX_NUM_EMBEDS, VERSION, NUM_TOKENS_IN_BLOCK
from driver import Driver
from l4ma import LlamaForCausalLM


def run(config: str = None,
        host: str = 'localhost',
        port: int = 9123,
        auth_token: str = None,
        model: str = None,
        cache_dir: str = None,
        kv_page_size: int = 32,
        dist_size: int = 32,
        max_num_kv_pages: int = 1000,
        max_num_embeds: int = 50000,
        device: str = 'cuda:0',
        dtype: str = 'bfloat16'):
    """
    Runs the application with the specified configuration.

    Args:
        config (str, optional): Path to a TOML configuration file.
        host (str, optional): The hostname. Defaults to 'localhost'.
        port (int, optional): The port number. Defaults to 9123.
        auth_token (str, optional): The authentication token. Defaults to None.
        model (str, optional): The model to use. Defaults to None.
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
        'port': port if port != 9123 else config_from_file.get('port', 9123),
        'auth_token': auth_token if auth_token is not None else config_from_file.get('auth_token'),
        'model': model if model is not None else config_from_file.get('model'),
        'kv_page_size': kv_page_size if kv_page_size != 32 else config_from_file.get('kv_page_size', 32),
        'dist_size': dist_size if dist_size != 32 else config_from_file.get('dist_size', 32),
        'max_num_kv_pages': max_num_kv_pages if max_num_kv_pages != 1000 else config_from_file.get('max_num_kv_pages', 1000),
        'max_num_embeds': max_num_embeds if max_num_embeds != 50000 else config_from_file.get('max_num_embeds', 50000),
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

    print("--- Final Configuration ---")
    for key, value in final_config.items():
        print(f"{key}: {value}")
    print("---------------------------")



def handle_request(d: Driver, request: l4m_pb2.Request) -> l4m_pb2.Response | None:
    # Determine which command was set in the oneof field "command"
    command = request.WhichOneof("command")

    # check locks on inputs & outputs

    # no pending computations on input -> do it RN (except for Fills - cannot do them in parallel) & register "pending" status on inputs & outputs.
    # ...
    # print("Handling request:", command)
    if command == "allocate":
        d.allocate(request.allocate)

    elif command == "deallocate":
        d.deallocate(request.deallocate)

    elif command == "embed_text":
        d.embed_text(request.embed_text)

    elif command == "fill_block":
        d.fill_block(request.fill_block)

    elif command == "mask_block":
        d.mask_block(request.mask_block)

    elif command == "copy_block":
        d.copy_block(request.copy_block)

    elif command == "decode_token_distribution":
        d.decode_token_distribution(request.decode_token_distribution)

    elif command == "sample_top_k_request":
        res = d.sample_top_k_request(request.sample_top_k_request)
        return l4m_pb2.Response(correlation_id=request.correlation_id, sample_top_k=res)

    elif command == "get_info":
        return l4m_pb2.Response(correlation_id=request.correlation_id, get_info=l4m_pb2.GetInfoResponse(
            version=VERSION,  # This needs to be defined or passed in
            model_name=d.model_name_or_path,  # Use the stored model name
            kv_page_size=NUM_TOKENS_IN_BLOCK,  # This needs to be defined or passed in
            num_available_kv_pages=d.max_num_pages,
            num_available_embeddings=d.max_num_embeds,
            num_available_distributions=0,
            tokenizer=l4m_pb2.Tokenizer(
                merge_table=None,
                special_tokens=None,
                split_regex=None,
            )
        ))

    else:
        print("No valid command found in request.")

    return None


def start_service(endpoint, model_name, device="cuda:0"):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map=device,
    )

    engine = Driver(model, MAX_NUM_PAGES, MAX_NUM_EMBEDS, torch.bfloat16, device)

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)

    print(f"Server listening on {endpoint}")
    tokenizer = AutoTokenizer.from_pretrained(engine.model_name_or_path)

    connected_clients = {}
    protocols = ["l4m", "l4m-vision", "ping"]
    idle_start = time.time()
    while True:
        # ROUTER sockets receive multipart messages.
        # Expected format: [client_identity, empty_frame, payload]

        frames = router.recv_multipart()
        # print(f"Idle time: {(time.time() - idle_start) * 1000}ms")

        client_identity = frames[0]
        start = time.time()

        # print("received", frames)

        # Check if an empty frame is present. If so, payload is at index 2.

        # check if the client has already a protocol
        if client_identity in connected_clients:

            if len(frames) != 3:
                print("Invalid message format.")
                # send an error message back to the client
                continue

            protocol_raw = frames[1]  # should be a single byte
            protocol_idx = int.from_bytes(protocol_raw, byteorder="little")

            if protocol_idx >= len(protocols):
                print("Invalid protocol:", protocol_idx)
                # send an error message back to the client
                continue

            protocol = protocols[protocol_idx]
            payload = frames[2]

            if protocol == "l4m":
                # Deserialize the protobuf message

                request = l4m_pb2.Request()
                request.ParseFromString(payload)

                # handle the request
                response = handle_request(engine, request)

                if response is not None:
                    reply_payload = response.SerializeToString()

                    # print("Sending reply back to the client.")
                    # Send reply back to the client.
                    # Include the client identity and an empty frame to maintain the envelope.
                    router.send_multipart([client_identity, protocol_raw, reply_payload])

                # print(f"elapsed time: {(time.time() - start) * 1000}ms")


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
            # do a handshake
            payload = frames[1]

            try:
                # Deserialize the protobuf message
                hs = handshake_pb2.Request()
                hs.ParseFromString(payload)

            except:
                print("Invalid handshake message.")
                # send an error message back to the client
                router.send_multipart([client_identity, b"\x00"])
                continue

            # send available protocols to the client
            response = handshake_pb2.Response(protocols=protocols)

            # Serialize the response
            payload = response.SerializeToString()

            connected_clients.update({client_identity: True})

            # send the response back to the client
            router.send_multipart([client_identity, payload])

        idle_start = time.time()


# Example usage
async def main():
    controller_host = "127.0.0.1"
    controller_port = 8080

    endpoint = ""

    if controller_host == "127.0.1" or controller_host == "localhost":
        endpoint = "ipc:///tmp/pie-ipc-test"

    else:
        endpoint = "tcp://*:8080"

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    start_service(endpoint, model_name)

    return
    # Notify the controller.
    auth_token = "<PASSWORD>"
    async with connect(f"ws://{controller_host}:{controller_port}") as websocket:

        # do authentication
        await websocket.send(msgpack.packb({
            "type": "authenticate",
            "corr_id": 0,
            "token": auth_token,
        }, use_bin_type=True))

        message = msgpack.unpackb(await websocket.recv(), raw=False)
        print(message)

        await websocket.send(msgpack.packb({
            "type": "attach_remote_service",
            "corr_id": 0,
            "endpoint": endpoint,
            "service_name": "example_service",
            "service_type": "l4m",
        }, use_bin_type=True))

        # register a remote backend

        await websocket.close()


if __name__ == "__main__":
    fire.Fire(run)
    #asyncio.run(main())
