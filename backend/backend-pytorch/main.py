import torch
import zmq
from transformers import TorchAoConfig

import sdi_pb2
from driver import Driver, NUM_TOKENS_IN_BLOCK
from l4ma import AttentionStorage, VectorStorage
from llama import LlamaForCausalLM


def handle_request(d: Driver, request: sdi_pb2.Request) -> sdi_pb2.Response | None:
    # Determine which command was set in the oneof field "command"
    command = request.WhichOneof("command")

    # check locks on inputs & outputs

    # no pending computations on input -> do it RN (except for Fills - cannot do them in parallel) & register "pending" status on inputs & outputs.
    # ...

    if command == "allocate":
        d.allocate(request.allocate)

    elif command == "deallocate":
        d.deallocate(request.deallocate)

    elif command == "embed_text":
        d.embed_text(request.embed_text)

    elif command == "embed_image":
        d.embed_image(request.embed_image)

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
        return sdi_pb2.Response(correlation_id=request.correlation_id, sample_top_k=res)

    elif command == "get_token_distribution":
        res = d.get_token_distribution(request.get_token_distribution)
        return sdi_pb2.Response(correlation_id=request.correlation_id, get_token_distribution=res)

    else:
        print("No valid command found in request.")

    return None


def main():
    device = "cuda:0"

    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype="bfloat16", device_map=device, quantization_config=quantization_config)

    block_storage = AttentionStorage(
        num_layers=model.config.num_hidden_layers,
        num_blocks=1000,
        num_heads=model.config.num_attention_heads,
        block_size=NUM_TOKENS_IN_BLOCK,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        dtype=torch.bfloat16,
        device=device
    )

    embed_storage = VectorStorage(
        num_embeds=100,
        embed_dim=model.config.hidden_size,
        dtype=torch.bfloat16,
        device=device
    )

    dist_storage = VectorStorage(
        num_embeds=100,
        embed_dim=model.config.hidden_size,
        dtype=torch.bfloat16,
        device=device
    )

    engine = Driver(model, block_storage, embed_storage, dist_storage)

    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://*:5555")
    print("Server listening on tcp://*:5555")

    while True:
        # ROUTER sockets receive multipart messages.
        # Expected format: [client_identity, empty_frame, payload]
        frames = router.recv_multipart()
        client_identity = frames[0]

        # Check if an empty frame is present. If so, payload is at index 2.
        payload = frames[1]

        # Deserialize the protobuf message
        request = sdi_pb2.Request()
        request.ParseFromString(payload)

        # handle the request
        response = handle_request(engine, request)

        if response is not None:
            reply_payload = response.SerializeToString()

            # Send reply back to the client.
            # Include the client identity and an empty frame to maintain the envelope.
            router.send_multipart([client_identity, reply_payload])


if __name__ == "__main__":
    main()
