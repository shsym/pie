import time
import argparse

import torch
import zmq
from transformers import TorchAoConfig, AutoTokenizer

import config
import l4m_pb2
import l4m_vision_pb2
import ping_pb2
import handshake_pb2

from common import ceil_div
from driver import Driver
from llama import LlamaForCausalLM
from config import VERSION, MODEL_NAME, FULL_MODEL_NAME, NUM_TOKENS_IN_BLOCK, MAX_NUM_PAGES, MAX_NUM_EMBEDS


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
            version=VERSION,
            model_name=MODEL_NAME,
            block_size=NUM_TOKENS_IN_BLOCK,
            num_available_blocks=d.max_num_pages,
            num_available_embeddings=d.max_num_embeds,
            num_available_distributions=0
        ))

    else:
        print("No valid command found in request.")

    return None


def main_run():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='L4M Backend Server')
    parser.add_argument('--ipc-endpoint', type=str, default='ipc:///tmp/symphony-ipc', 
                       help='IPC endpoint to bind to')
    args = parser.parse_args()
    
    device = "cuda:0"

    # quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
    # , quantization_config=quantization_config
    model = LlamaForCausalLM.from_pretrained(
        FULL_MODEL_NAME, torch_dtype="bfloat16", device_map=device)

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
