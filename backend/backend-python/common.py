import l4m_pb2 # Added import
from driver import Driver # Added import
from config import VERSION, MODEL_NAME, NUM_TOKENS_IN_BLOCK


def ceil_div(a, b):
    return -(-a // b)


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
            version=VERSION, # This needs to be defined or passed in
            model_name=d.model_name_or_path, # Use the stored model name
            block_size=NUM_TOKENS_IN_BLOCK, # This needs to be defined or passed in
            num_available_blocks=d.max_num_pages,
            num_available_embeddings=d.max_num_embeds,
            num_available_distributions=0
        ))

    else:
        print("No valid command found in request.")

    return None
