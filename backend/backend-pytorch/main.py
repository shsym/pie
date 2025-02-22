import torch
import zmq
from flatbuffers.flexbuffers import Vector
from transformers import TorchAoConfig

import sdi_pb2
from engine import Engine, BLOCK_SIZE
from l4ma import L4maModel, AttentionBuffer, AttentionStorage, VectorStorage
from llama import LlamaForCausalLM


def handle_request(request: sdi_pb2.Request):
    # Determine which command was set in the oneof field "command"
    command = request.WhichOneof("command")

    # check locks on inputs & outputs

    # no pending computations on input -> do it RN (except for Fills - cannot do them in parallel) & register "pending" status on inputs & outputs.
    # ...

    if command == "allocate":
        batch_allocate = request.allocate  # This is a BatchAllocate message
        print("Handling BatchAllocate command")
        # Process each Allocate item...
        for allocate in batch_allocate.items:
            print(f"Allocate object: kind={allocate.kind}, offset={allocate.object_id_offset}, count={allocate.count}")

    elif command == "deallocate":
        batch_deallocate = request.deallocate  # This is a BatchDeallocate message
        print("Handling BatchDeallocate command")
        # Process each Allocate item (or deallocate items if they differ)...

    elif command == "embed_text":
        batch_embed_text = request.embed_text  # This is a BatchEmbedText message
        print("Handling BatchEmbedText command")
        for embed in batch_embed_text.items:
            print(f"EmbedText: embedding_id={embed.embedding_id}, token_id={embed.token_id}, position_id={embed.position_id}")

    elif command == "embed_image":
        batch_embed_image = request.embed_image  # This is a BatchEmbedImage message
        print("Handling BatchEmbedImage command")
        for embed in batch_embed_image.items:
            print(f"EmbedImage: embedding_ids={embed.embedding_ids}, url={embed.url}")

    elif command == "fill_block":
        batch_fill_block = request.fill_block  # This is a BatchFillBlock message
        print("Handling BatchFillBlock command")
        # Process each FillBlock item...

    elif command == "mask_block":
        batch_mask_block = request.mask_block  # This is a BatchMaskBlock message
        print("Handling BatchMaskBlock command")
        # Process each MaskBlock item...

    elif command == "copy_block":
        batch_copy_block = request.copy_block  # This is a BatchCopyBlock message
        print("Handling BatchCopyBlock command")
        # Process each CopyBlock item...

    elif command == "decode_token_distribution":
        batch_decode = request.decode_token_distribution  # This is a BatchDecodeTokenDistribution message
        print("Handling BatchDecodeTokenDistribution command")
        # Process each DecodeTokenDistribution item...

    elif command == "sample_top_k_request":
        batch_sample_topk = request.sample_top_k_request  # This is a BatchSampleTopKRequest message
        print("Handling BatchSampleTopKRequest command")
        # Process each SampleTopKRequest item...

    elif command == "get_token_distribution":
        batch_get_distribution = request.get_token_distribution  # This is a BatchGetTokenDistributionRequest message
        print("Handling BatchGetTokenDistribution command")
        # Process each GetTokenDistributionRequest item...

    else:
        print("No valid command found in request.")


def main():
    device = "cuda:0"

    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype="bfloat16", device_map=device, quantization_config=quantization_config)

    block_storage = AttentionStorage(
        num_layers=model.config.num_hidden_layers,
        num_blocks=1000,
        num_heads=model.config.num_attention_heads,
        block_size=BLOCK_SIZE,
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

    engine = Engine(model, block_storage, embed_storage, dist_storage)

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
        person = sdi_pb2.Request()
        person.ParseFromString(payload)

        print(f"Received message from {client_identity.decode('utf-8')}:")
        print(person)

        # Send reply back to the client.
        # Include the client identity and an empty frame to maintain the envelope.
        router.send_multipart([client_identity, reply_payload])


if __name__ == "__main__":
    main()
