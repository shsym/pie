import zmq
import sdi_pb2


def handle_request(request: sdi_pb2.Request):
    # Determine which command was set in the oneof field "command"
    command = request.WhichOneof("command")

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
        if len(frames) == 3 and frames[1] == b'':
            payload = frames[2]
        else:
            payload = frames[1]

        # Deserialize the protobuf message
        person = sdi_pb2.Request()
        person.ParseFromString(payload)

        print(f"Received message from {client_identity.decode('utf-8')}:")
        print(person)


        # Send reply back to the client.
        # Include the client identity and an empty frame to maintain the envelope.
        router.send_multipart([client_identity, b'', reply_payload])

    if __name__ == "__main__":
        main()