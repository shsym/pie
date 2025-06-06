import time
import argparse

import os
import warnings
import torch
import zmq
from transformers import TorchAoConfig, AutoTokenizer

import config
import l4m_pb2
import l4m_vision_pb2
import ping_pb2
import handshake_pb2

from common import ceil_div, handle_request # Updated import
from driver import Driver
from llama import LlamaForCausalLM
from config import MAX_NUM_PAGES, MAX_NUM_EMBEDS


def main_run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='L4M Backend Server')
    parser.add_argument('--ipc-endpoint', type=str, default='ipc:///tmp/symphony-ipc', 
                        help='IPC endpoint to bind to')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name to load (e.g., deepseek-ai/DeepSeek-V3, meta-llama/Llama-3.1-8B-Instruct)')
    args = parser.parse_args()
    
    device = "cuda:0"

    # Determine Symphony models cache directory
    models_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "symphony", "models")
    # Local model path uses '--' in place of '/'
    local_dir = os.path.join(models_cache_dir, args.model_name.replace("/", "--"))
    if os.path.isdir(local_dir):
        print(f"Loading Llama model from local cache: {local_dir}")
        model_source = local_dir
        local_only = True
    else:
        print(f"Loading Llama model from HuggingFace Hub: {args.model_name}")
        model_source = args.model_name
        local_only = False

    # Suppress rope_scaling warnings if present
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized keys in `rope_scaling`")
        model = LlamaForCausalLM.from_pretrained(
            model_source,
            torch_dtype="bfloat16",
            device_map=device,
            cache_dir=models_cache_dir,
            local_files_only=local_only,
        )

    endpoint = args.ipc_endpoint

    engine = Driver(model, MAX_NUM_PAGES, MAX_NUM_EMBEDS, torch.bfloat16, device)

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)

    print(f"Server listening on {endpoint}")

    connected_clients = {}
    protocols = ["l4m", "l4m-vision", "ping"]
    idle_start = time.time()
    while True:
        # ROUTER sockets receive multipart messages.
        # Expected format: [client_identity, empty_frame, payload]

        frames = router.recv_multipart()
        #print(f"Idle time: {(time.time() - idle_start) * 1000}ms")

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



if __name__ == "__main__":
    main_run()
