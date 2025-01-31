#!/usr/bin/env python3
import zmq
import msgpack
import uuid
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Union


# =============================================================================
# 1) Pythonic Data Classes to Mirror Your Commands & Responses
# =============================================================================

@dataclass
class AllocateBlocksCmd:
    """Represents Command::AllocateBlocks(num_blocks)."""
    num_blocks: int


@dataclass
class AllocateBlockCmd:
    """Represents Command::AllocateBlock (no parameters)."""
    pass


@dataclass
class CopyCmd:
    """Represents Command::Copy { src_block_id, dst_block_id, src_start, dst_start, length }."""
    src_block_id: int
    dst_block_id: int
    src_start: int
    dst_start: int
    length: int


@dataclass
class DropCmd:
    """Represents Command::Drop { block_id, start, end }."""
    block_id: int
    start: int
    end: int


@dataclass
class FreeBlockCmd:
    """Represents Command::FreeBlock { block_id }."""
    block_id: int


@dataclass
class FreeBlocksCmd:
    """Represents Command::FreeBlocks { block_id_offset, count }."""
    block_id_offset: int
    count: int


@dataclass
class AvailableBlocksCmd:
    """Represents Command::AvailableBlocks (no parameters)."""
    pass


# For convenience, define a Python "enum-like" union of all commands:
Command = Union[
    AllocateBlocksCmd,
    AllocateBlockCmd,
    CopyCmd,
    DropCmd,
    FreeBlockCmd,
    FreeBlocksCmd,
    AvailableBlocksCmd,
]


@dataclass
class Request:
    """Represents an incoming client request."""
    instance_id: bytes  # The raw 16 bytes from the message
    command: Command


# --- For responses, define typed wrappers as well ---

@dataclass
class AllocatedBlocksResp:
    block_id_offset: int
    count: int


@dataclass
class AvailableCountResp:
    count: int


@dataclass
class ErrorResp:
    error_code: int
    message: str


@dataclass
class AwkResp:
    message: str


ResponseData = Union[
    AllocatedBlocksResp,
    AvailableCountResp,
    ErrorResp,
    AwkResp
]


@dataclass
class Response:
    instance_id: bytes
    data: ResponseData


# =============================================================================
# 2) Server State
# =============================================================================

class ServerState:
    """
    Maintains a set of allocated block IDs, a global counter for new blocks,
    and enforces a finite capacity.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.allocated = set()  # set of allocated block IDs (u32)
        self.next_block_id = 1
        self.lock = threading.Lock()

    def allocate_blocks(self, n: int) -> List[int]:
        with self.lock:
            if len(self.allocated) + n > self.capacity:
                raise ValueError("Capacity exceeded")
            allocated_ids = []
            for _ in range(n):
                # Find the next available block id.
                while self.next_block_id in self.allocated:
                    self.next_block_id += 1
                allocated_ids.append(self.next_block_id)
                self.allocated.add(self.next_block_id)
                self.next_block_id += 1
            return allocated_ids

    def available_blocks(self) -> int:
        with self.lock:
            return self.capacity - len(self.allocated)

    def free_block(self, block_id: int):
        with self.lock:
            if block_id not in self.allocated:
                raise ValueError(f"Block {block_id} not allocated")
            self.allocated.remove(block_id)

    def free_blocks_range(self, offset: int, count: int):
        with self.lock:
            to_free = list(range(offset, offset + count))
            for bid in to_free:
                if bid not in self.allocated:
                    raise ValueError(f"Block {bid} not allocated")
            for bid in to_free:
                self.allocated.remove(bid)

    def copy_blocks(self, src: int, dst: int):
        with self.lock:
            if src not in self.allocated:
                raise ValueError(f"Source block {src} not allocated")
            if dst not in self.allocated:
                raise ValueError(f"Destination block {dst} not allocated")
            # Simulate copy (no state change)

    def drop_block(self, block_id: int):
        with self.lock:
            if block_id not in self.allocated:
                raise ValueError(f"Block {block_id} not allocated")
            # Simulate partial drop (no state change)


# =============================================================================
# 3) Parsing and Handling Commands
# =============================================================================

def parse_incoming_message(msg: list) -> Request:
    """
    The raw message is a list of length 2:
      [ <16-byte UUID>, { "AllocateBlocks": [5] } ]
    or similar for other commands.

    We'll parse:
    - instance_id_bytes = msg[0]  (16-byte array)
    - command_dict = msg[1]      (dict, e.g. {"AllocateBlocks": [5]})
    """
    if not isinstance(msg, list) or len(msg) != 2:
        raise ValueError("Message must be a 2-element list: [uuid_bytes, command_dict]")
    instance_id = msg[0]  # 16-byte UUID
    command_dict = msg[1]

    print("instance_id", instance_id)
    print("command_dict", command_dict)

    if isinstance(command_dict, dict):

        (cmd_type, cmd_payload_list) = next(iter(command_dict.items()))
    else:
        cmd_type = command_dict
        cmd_payload_list = []

    if not isinstance(cmd_payload_list, list):
        raise ValueError("Command payload must be a list of parameters")

    # Now parse each command type:
    if cmd_type == "AllocateBlocks":
        if len(cmd_payload_list) != 1:
            raise ValueError("AllocateBlocks requires [num_blocks]")
        return Request(
            instance_id,
            AllocateBlocksCmd(num_blocks=cmd_payload_list[0]),
        )

    elif cmd_type == "AllocateBlock":
        # Typically "AllocateBlock": []
        return Request(instance_id, AllocateBlockCmd())

    elif cmd_type == "Copy":
        # Expect 5 parameters
        if len(cmd_payload_list) != 5:
            raise ValueError("Copy requires 5 parameters [src_block_id, dst_block_id, src_start, dst_start, length]")
        return Request(
            instance_id,
            CopyCmd(
                src_block_id=cmd_payload_list[0],
                dst_block_id=cmd_payload_list[1],
                src_start=cmd_payload_list[2],
                dst_start=cmd_payload_list[3],
                length=cmd_payload_list[4],
            ),
        )

    elif cmd_type == "Drop":
        # Expect 3 parameters
        if len(cmd_payload_list) != 3:
            raise ValueError("Drop requires 3 parameters [block_id, start, end]")
        return Request(
            instance_id,
            DropCmd(block_id=cmd_payload_list[0],
                    start=cmd_payload_list[1],
                    end=cmd_payload_list[2]),
        )

    elif cmd_type == "FreeBlock":
        # Expect 1 parameter
        if len(cmd_payload_list) != 1:
            raise ValueError("FreeBlock requires [block_id]")
        return Request(
            instance_id,
            FreeBlockCmd(block_id=cmd_payload_list[0]),
        )

    elif cmd_type == "FreeBlocks":
        # Expect 2 parameters: [block_id_offset, count]
        if len(cmd_payload_list) != 2:
            raise ValueError("FreeBlocks requires [block_id_offset, count]")
        return Request(
            instance_id,
            FreeBlocksCmd(
                block_id_offset=cmd_payload_list[0],
                count=cmd_payload_list[1],
            )
        )

    elif cmd_type == "AvailableBlocks":
        # Expect 0 parameters
        return Request(instance_id, AvailableBlocksCmd())

    else:
        raise ValueError(f"Unknown command type: {cmd_type}")


def handle_command(req: Request, state: ServerState) -> Optional[Response]:
    """
    Process a typed request. Return a `Response` if the command needs one.
    If the command does not produce a "success" response, return None
    (but on error, we'll generate an error response).
    """
    c = req.command
    if isinstance(c, AllocateBlocksCmd):
        allocated_ids = state.allocate_blocks(c.num_blocks)
        if not allocated_ids:
            raise ValueError("Allocation returned empty")
        offset = allocated_ids[0]
        count = len(allocated_ids)
        return Response(
            instance_id=req.instance_id,
            data=AllocatedBlocksResp(offset, count),
        )

    elif isinstance(c, AllocateBlockCmd):
        allocated_ids = state.allocate_blocks(1)
        if not allocated_ids:
            raise ValueError("Allocation returned empty")
        return Response(
            instance_id=req.instance_id,
            data=AllocatedBlocksResp(allocated_ids[0], 1)
        )

    elif isinstance(c, AvailableBlocksCmd):
        available = state.available_blocks()
        return Response(
            instance_id=req.instance_id,
            data=AvailableCountResp(count=available),
        )

    elif isinstance(c, FreeBlockCmd):
        state.free_block(c.block_id)
        return None  # no success response

    elif isinstance(c, FreeBlocksCmd):
        state.free_blocks_range(c.block_id_offset, c.count)
        return None

    elif isinstance(c, CopyCmd):
        state.copy_blocks(c.src_block_id, c.dst_block_id)
        return None

    elif isinstance(c, DropCmd):
        state.drop_block(c.block_id)
        return None

    # Should never get here if all commands are covered
    raise ValueError("Unhandled command variant")


# =============================================================================
# 4) Convert Response to a Dict for msgpack
# =============================================================================

def response_to_dict(resp: Response) -> list:
    """
    Convert a typed Response into a dictionary that can be msgpacked.
    """
    if isinstance(resp.data, AllocatedBlocksResp):
        return [
            resp.instance_id,
            {
                "AllocatedBlocks": [
                    resp.data.block_id_offset,
                    resp.data.count
                ]
            }
        ]
    elif isinstance(resp.data, AvailableCountResp):
        return [
            resp.instance_id,
            {"AvailableCount": [resp.data.count]},
        ]
    elif isinstance(resp.data, ErrorResp):
        return [
            resp.instance_id,

            {"Error": [
                resp.data.error_code,
                resp.data.message
            ]}
        ]
    elif isinstance(resp.data, AwkResp):
        return [
            resp.instance_id,
            {"Awk": [resp.data.message]},
        ]
    else:
        # If you add new response variants, handle them here
        raise ValueError("Unknown response type")


# =============================================================================
# 5) Worker Thread
# =============================================================================

def worker_thread(worker_queue, state: ServerState, response_queue):
    """
    Each worker processes items of the form (client_id, message_list).
    message_list is something like:
       [ <16-byte UUID>, { "AllocateBlocks": [5] } ]
    We parse it into a Request, handle the command, and if needed place
    the response on response_queue. If there's an error, we place an error
    response on the queue.
    """
    while True:
        item = worker_queue.get()
        if item is None:
            # shutdown
            break
        client_id, req = item

        try:

            print(req)

            resp = handle_command(req, state)

            if resp is not None:
                # Convert typed Response to dict
                response_queue.put((client_id, resp))
        except Exception as exc:
            print(f"[Worker] Error: {exc} for message {req}")

            resp = Response(
                instance_id=req.instance_id,
                data=ErrorResp(
                    error_code=100,
                    message=str(exc),
                ),
            )
            response_queue.put((client_id, resp))

        worker_queue.task_done()


# =============================================================================
# 6) Main Server Loop
# =============================================================================

def main():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://*:5555")
    print("Server listening on tcp://*:5555")

    # Shared server state
    state = ServerState(capacity=10_000)

    # Create worker threads
    num_workers = 4
    worker_queues = [queue.Queue() for _ in range(num_workers)]
    response_queue = queue.Queue()

    for i in range(num_workers):
        t = threading.Thread(
            target=worker_thread,
            args=(worker_queues[i], state, response_queue),
            daemon=True
        )
        t.start()

    while True:
        # 1) Receive a multi-part message: [client_id, payload]
        msg_parts = router.recv_multipart()
        if len(msg_parts) < 2:
            continue

        client_id = msg_parts[0]
        payload = msg_parts[1]

        # 2) Unpack from MessagePack. The result should be a list of length 2:
        #    [instance_id_bytes, command_obj]
        message_list = msgpack.unpackb(payload, raw=False)
        req = parse_incoming_message(message_list)

        worker_index = hash(req.instance_id) % num_workers

        # 4) Enqueue the work

        worker_queues[worker_index].put((client_id, req))

        # return awk
        # router.send_multipart([client_id, msgpack.packb(response_to_dict(
        #     Response(req.instance_id, AwkResp(message="Received"))
        # ), use_bin_type=True)])

        # 5) Check for any responses in the response_queue
        while not response_queue.empty():
            try:
                resp_client_id, resp = response_queue.get_nowait()
                resp_dict = response_to_dict(resp)

                packed_resp = msgpack.packb(resp_dict, use_bin_type=True)
                router.send_multipart([resp_client_id, packed_resp])
                response_queue.task_done()
            except queue.Empty:
                break


if __name__ == "__main__":
    main()
