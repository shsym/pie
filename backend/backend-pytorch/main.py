import time

import torch
import zmq
from transformers import TorchAoConfig, AutoTokenizer

import l4m_pb2
import l4m_vision_pb2
import ping_pb2
import handshake_pb2

from common import ceil_div
from driver import Driver
from l4ma import AttentionStorage, VectorStorage
from llama import LlamaForCausalLM
from config import VERSION, MODEL_NAME, FULL_MODEL_NAME, NUM_TOKENS_IN_BLOCK


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
            num_available_blocks=d.block_storage.num_blocks,
            num_available_embeddings=d.embed_storage.num_vectors,
            num_available_distributions=0
        ))

    else:
        print("No valid command found in request.")

    return None


def main_run():
    device = "cuda:0"

    # quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
    # , quantization_config=quantization_config
    model = LlamaForCausalLM.from_pretrained(
        FULL_MODEL_NAME, torch_dtype="bfloat16", device_map=device)

    block_storage = AttentionStorage(
        num_layers=model.config.num_hidden_layers,
        num_blocks=6000,
        num_heads=model.config.num_key_value_heads,
        block_size=NUM_TOKENS_IN_BLOCK,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=torch.bfloat16,
        device=device
    )

    embed_storage = VectorStorage(
        num_vectors=6000,
        embed_dim=model.config.vocab_size,
        dtype=torch.bfloat16,
        device=device
    )

    # dist_storage = VectorStorage(
    #     num_vectors=6000,
    #     embed_dim=model.config.vocab_size,
    #     dtype=torch.bfloat16,
    #     device=device
    # )

    engine = Driver(model, block_storage, embed_storage)

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://*:8888")
    print("Server listening on tcp://*:8888")

    connected_clients = {}
    protocols = ["l4m", "l4m-vision", "ping"]

    while True:
        # ROUTER sockets receive multipart messages.
        # Expected format: [client_identity, empty_frame, payload]
        frames = router.recv_multipart()
        client_identity = frames[0]

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


###====================Test====================###

def llama3_format(prompt: str, hint: str | None, system: str = "You are a helpful, respectful and honest assistant."):
    temp = "<|begin_of_text|>"
    temp += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    temp += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    temp += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    if hint:
        temp += hint

    return temp


def main_test():
    device = "cuda:0"

    # quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
    # , quantization_config=quantization_config
    model = LlamaForCausalLM.from_pretrained(
        FULL_MODEL_NAME, torch_dtype="bfloat16", device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_NAME)

    block_storage = AttentionStorage(
        num_layers=model.config.num_hidden_layers,
        num_blocks=1000,
        num_heads=model.config.num_key_value_heads,
        block_size=NUM_TOKENS_IN_BLOCK,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=torch.bfloat16,
        device=device
    )

    embed_storage = VectorStorage(
        num_vectors=1000,
        embed_dim=model.config.vocab_size,
        dtype=torch.bfloat16,
        device=device
    )

    # dist_storage = VectorStorage(
    #     num_vectors=1000,
    #     embed_dim=model.config.vocab_size,
    #     dtype=torch.bfloat16,
    #     device=device
    # )

    engine = Driver(model, block_storage, embed_storage)

    test_prompt = llama3_format("What is Pinon coffee? ELI 5", None)

    token_ids = tokenizer.encode(test_prompt)
    print("token_ids:", token_ids)

    num_blocks_needed = ceil_div(len(token_ids), NUM_TOKENS_IN_BLOCK)
    print("num blocks needed:", num_blocks_needed)

    next_token_pointer_idx = (len(token_ids) % NUM_TOKENS_IN_BLOCK) - 1
    print("next token pointer idx:", next_token_pointer_idx)

    engine.embed_text(l4m_pb2.BatchEmbedText(items=[
        l4m_pb2.EmbedText(embedding_id=i, token_id=token_ids[i], position_id=i)
        for i in range(len(token_ids))
    ]))

    engine.allocate(l4m_pb2.BatchAllocate(items=[l4m_pb2.Allocate(kind=l4m_pb2.ObjectKind.OBJECT_KIND_KV_BLOCK, object_id_offset=0, count=5)]))

    OUT_EMB_OFFSET = 100

    engine.fill_block(l4m_pb2.BatchFillBlock(items=[
        l4m_pb2.FillBlock(block_id=i,
                          context_block_ids=list(range(i + 1)),
                          input_embedding_ids=list(range(NUM_TOKENS_IN_BLOCK * i, NUM_TOKENS_IN_BLOCK * (i + 1))),
                          output_embedding_ids=list(range(OUT_EMB_OFFSET, OUT_EMB_OFFSET + NUM_TOKENS_IN_BLOCK)) if i == num_blocks_needed - 1 else [])
        for i in range(num_blocks_needed)
    ]))

    decoded_tokens = []

    last_block_id = num_blocks_needed - 1
    last_token_idx = (len(token_ids) % NUM_TOKENS_IN_BLOCK) - 1

    for i in range(10):
        torch.cuda.synchronize()

        time_start = time.time()

        engine.decode_token_distribution(l4m_pb2.BatchDecodeTokenDistribution(items=[
            l4m_pb2.DecodeTokenDistribution(embedding_id=OUT_EMB_OFFSET + last_token_idx + i, distribution_id=0)
        ]))
        res = engine.sample_top_k_request(l4m_pb2.BatchSampleTopKRequest(items=[
            l4m_pb2.SampleTopKRequest(distribution_id=0, k=5)
        ]))

        new_token = res.items[0].token_ids[0]

        # print("new token:", new_token)
        decoded_tokens.append(new_token)
        print(tokenizer.decode(decoded_tokens), f"({new_token})")

        engine.embed_text(l4m_pb2.BatchEmbedText(items=[
            l4m_pb2.EmbedText(embedding_id=len(token_ids) + i, token_id=new_token, position_id=len(token_ids) + i)
        ]))
        engine.fill_block(l4m_pb2.BatchFillBlock(items=[
            l4m_pb2.FillBlock(block_id=last_block_id,
                              context_block_ids=list(range(last_block_id + 1)),
                              input_embedding_ids=list(range(NUM_TOKENS_IN_BLOCK * last_block_id, NUM_TOKENS_IN_BLOCK * (last_block_id + 1))),
                              output_embedding_ids=list(range(OUT_EMB_OFFSET, OUT_EMB_OFFSET + NUM_TOKENS_IN_BLOCK))),
        ]))
        torch.cuda.synchronize()
        time_end = time.time()

        # print the elapsed time in milliseconds
        print(f"Elapsed time: {(time_end - time_start) * 1000:.2f}ms")

    print("done!")


if __name__ == "__main__":
    main_run()
    # main_test()
