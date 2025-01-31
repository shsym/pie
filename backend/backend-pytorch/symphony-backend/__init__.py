import zmq
import msgpack
import uuid  # for parsing the instance_id string into a Python UUID (if desired)

# --------------------------------------------------
# Simple in-memory "state"
# --------------------------------------------------
CAPACITY = 1000
allocated_count = 0

# --------------------------------------------------
# Helper: build externally tagged "Ok" or "Error" response
# matching Rust's enum `Response`
# --------------------------------------------------
def make_ok_response(instance_id, data=None):
    # Externally tagged enum:
    # {
    #   "Ok": {
    #     "instance_id": <uuid-string>,
    #     "data": <another externally tagged object, or null>
    #   }
    # }
    resp = {
        "Ok": {
            "instance_id": str(instance_id),  # store as string
        }
    }
    if data is not None:
        resp["Ok"]["data"] = data
    return resp

def make_error_response(instance_id, error_code, message):
    # {
    #   "Error": {
    #     "instance_id": <uuid-string>,
    #     "error_code": <u32>,
    #     "message": <string>
    #   }
    # }
    return {
        "Error": {
            "instance_id": str(instance_id),
            "error_code": error_code,
            "message": message
        }
    }

# --------------------------------------------------
# Helper: build externally tagged data for the `ResponseData` enum
# e.g. AllocatedBlocks(RleVec(...)) => {"AllocatedBlocks": [[1000, 5]]}
# --------------------------------------------------
def make_allocated_blocks_rle(rle_list):
    return {
        "AllocatedBlocks": rle_list  # e.g. [[1000, 5]]
    }

def make_available_count(n):
    return {
        "AvailableCount": n
    }

# --------------------------------------------------
# Process a single Command
# We decode the externally tagged enum inside "command"
# and produce a matching Response
# --------------------------------------------------
def process_command(instance_id, command_obj):
    """
    command_obj is a dict with exactly one key, e.g.:
      { "AllocateBlocks": 5 }
      { "AllocateBlock": None? }
      { "Copy": { "src_block_id": 123, ... } }
    We'll handle a few as examples.
    """
    global allocated_count

    # We'll extract the first (and only) key to see the variant name
    if len(command_obj) != 1:
        return make_error_response(instance_id, 99, "Malformed command enum.")

    (variant, payload) = next(iter(command_obj.items()))

    if variant == "AllocateBlocks":
        # payload is an integer
        if not isinstance(payload, int):
            return make_error_response(instance_id, 100, "AllocateBlocks expects an integer payload.")
        num = payload
        if allocated_count + num > CAPACITY:
            return make_error_response(
                instance_id,
                1,
                f"Not enough capacity (allocated={allocated_count}, requested={num}, max={CAPACITY})"
            )
        allocated_count += num
        # For demonstration, pretend we allocated block IDs in a single RLE run: [[1000, num]]
        data_obj = make_allocated_blocks_rle([[1000, num]])
        return make_ok_response(instance_id, data=data_obj)

    elif variant == "AvailableBlocks":
        # No payload
        remaining = CAPACITY - allocated_count
        data_obj = make_available_count(remaining)
        return make_ok_response(instance_id, data=data_obj)

    elif variant == "AllocateBlock":
        # Example: same as AllocateBlocks(1)
        if allocated_count + 1 > CAPACITY:
            return make_error_response(
                instance_id,
                1,
                f"Not enough capacity (allocated={allocated_count}, requested=1, max={CAPACITY})"
            )
        allocated_count += 1
        data_obj = make_allocated_blocks_rle([[1000, 1]])
        return make_ok_response(instance_id, data=data_obj)

    # We'll just demonstrate an error for unimplemented variants:
    else:
        return make_error_response(
            instance_id,
            99,
            f"Command variant '{variant}' not implemented in Python server example."
        )

# --------------------------------------------------
# Main server loop (single-threaded)
# --------------------------------------------------
def main():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://*:5555")

    print("Python server listening on tcp://*:5555")
    try:
        while True:
            frames = router.recv_multipart()
            if len(frames) < 2:
                # Expect [client_id, data]
                continue

            client_id, packed_data = frames[0], frames[1]

            # Unpack the Request, which is also externally tagged for the "command" field
            # The Rust struct is:
            #   struct Request {
            #       pub instance_id: Uuid,
            #       pub command: Command,
            #   }
            #
            # By default, `serde` => `externally tagged` for the enum. So we might get:
            # {
            #   "instance_id": "...(string)...

            #   "command": {
            #     "AllocateBlocks": 5
            #   }
            # }
            try:
                request_obj = msgpack.unpackb(packed_data, raw=False)
            except Exception as e:
                # Malformed message
                error_resp = make_error_response("00000000-0000-0000-0000-000000000000", 400, f"Malformed msgpack: {e}")
                router.send_multipart([client_id, msgpack.packb(error_resp, use_bin_type=True)])
                continue

            # Extract instance_id (as a string), parse or keep as string
            instance_id_str = request_obj.get("instance_id", "00000000-0000-0000-0000-000000000000")
            # If we want, convert to Python's uuid.UUID
            try:
                inst_id = uuid.UUID(instance_id_str)
            except ValueError:
                inst_id = uuid.UUID("00000000-0000-0000-0000-000000000000")

            # Extract the "command" object
            command_obj = request_obj.get("command", {})
            if not isinstance(command_obj, dict):
                # It's probably an error or wrong shape
                resp = make_error_response(
                    inst_id,
                    400,
                    "Request missing 'command' object or 'command' not a dict."
                )
            else:
                # Process it
                resp = process_command(inst_id, command_obj)

            # Send response
            packed_resp = msgpack.packb(resp, use_bin_type=True)
            router.send_multipart([client_id, packed_resp])

    except KeyboardInterrupt:
        print("Server interrupted, shutting down...")
    finally:
        router.close()
        context.term()

if __name__ == "__main__":
    main()