#!/usr/bin/env python3

import zmq
import msgpack
import uuid
import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import StrEnum, auto


# =============================================================================
# 1) Enums & Dataclasses for Commands/Responses
# =============================================================================

class CommandKind(StrEnum):
    ALLOCATE_BLOCKS = "AllocateBlocks"
    ALLOCATE_BLOCK = "AllocateBlock"
    COPY = "Copy"
    DROP = "Drop"
    FREE_BLOCK = "FreeBlock"
    FREE_BLOCKS = "FreeBlocks"
    AVAILABLE_BLOCKS = "AvailableBlocks"
    # Add more if needed


@dataclass
class AllocateBlocksCmd:
    num_blocks: int


@dataclass
class AllocateBlockCmd:
    pass


@dataclass
class CopyCmd:
    src_block_id: int
    dst_block_id: int
    src_start: int
    dst_start: int
    length: int


@dataclass
class DropCmd:
    block_id: int
    start: int
    end: int


@dataclass
class FreeBlockCmd:
    block_id: int


@dataclass
class FreeBlocksCmd:
    block_id_offset: int
    count: int


@dataclass
class AvailableBlocksCmd:
    pass


CommandPayload = Union[
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
    instance_id: bytes
    kind: CommandKind
    payload: CommandPayload


# For responses, likewise we define the kind:

class ResponseKind(StrEnum):
    ALLOCATED_BLOCKS = "AllocatedBlocks"
    AVAILABLE_COUNT = "AvailableCount"
    ERROR = "Error"
    AWK = "Awk"


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


ResponsePayload = Union[
    AllocatedBlocksResp,
    AvailableCountResp,
    ErrorResp,
    AwkResp
]


@dataclass
class Response:
    instance_id: bytes
    kind: ResponseKind
    payload: ResponsePayload


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
            # Simulate copy

    def drop_block(self, block_id: int):
        with self.lock:
            if block_id not in self.allocated:
                raise ValueError(f"Block {block_id} not allocated")
            # Simulate partial drop


def parse_incoming_message(msg: list) -> Request:
    """
    The raw message is a list of length 2:
      [ <16-byte UUID>, { "AllocateBlocks": [5] } ]
    or similar for other commands.

    We'll parse them using structural pattern matching.
    """
    if not isinstance(msg, list) or len(msg) != 2:
        raise ValueError("Message must be [uuid_bytes, command_dict]")

    instance_id = msg[0]
    command_dict = msg[1]

    if not isinstance(command_dict, dict):
        (cmd_str, parameters) = (command_dict, [])

    else:
        # There's only one key in command_dict, e.g. {"AllocateBlocks": [5]}
        (cmd_str, parameters) = next(iter(command_dict.items()))

    # Convert string -> CommandKind
    try:
        kind = CommandKind(cmd_str)
    except ValueError:
        raise ValueError(f"Unknown command type: {cmd_str}")

    # In a typical scenario, parameters is a list. We'll match on (kind, parameters).
    match kind, parameters:
        case (CommandKind.ALLOCATE_BLOCKS, [int(num_blocks)]):
            payload = AllocateBlocksCmd(num_blocks=num_blocks)
        case (CommandKind.ALLOCATE_BLOCK, []):
            payload = AllocateBlockCmd()
        case (CommandKind.COPY, [int(src), int(dst), int(src_start), int(dst_start), int(length)]):
            payload = CopyCmd(
                src_block_id=src, dst_block_id=dst,
                src_start=src_start, dst_start=dst_start, length=length
            )
        case (CommandKind.DROP, [int(block_id), int(start), int(end)]):
            payload = DropCmd(block_id=block_id, start=start, end=end)
        case (CommandKind.FREE_BLOCK, [int(block_id)]):
            payload = FreeBlockCmd(block_id=block_id)
        case (CommandKind.FREE_BLOCKS, [int(offset), int(count)]):
            payload = FreeBlocksCmd(block_id_offset=offset, count=count)
        case (CommandKind.AVAILABLE_BLOCKS, []):
            payload = AvailableBlocksCmd()
        case _:
            raise ValueError(f"Invalid parameters for command: {cmd_str}")

    return Request(
        instance_id=instance_id,
        kind=kind,
        payload=payload,
    )


def handle_command(req: Request, state: ServerState) -> Optional[Response]:
    """
    Process a typed request, returning a Response if the command needs one.
    """
    match req.kind, req.payload:
        case (CommandKind.ALLOCATE_BLOCKS, AllocateBlocksCmd(num_blocks)):
            allocated_ids = state.allocate_blocks(num_blocks)
            if not allocated_ids:
                raise ValueError("Allocation returned empty")
            offset = allocated_ids[0]
            count = len(allocated_ids)
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.ALLOCATED_BLOCKS,
                payload=AllocatedBlocksResp(block_id_offset=offset, count=count)
            )

        case (CommandKind.ALLOCATE_BLOCK, AllocateBlockCmd()):
            allocated_ids = state.allocate_blocks(1)
            if not allocated_ids:
                raise ValueError("Allocation returned empty")
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.ALLOCATED_BLOCKS,
                payload=AllocatedBlocksResp(block_id_offset=allocated_ids[0], count=1)
            )

        case (CommandKind.AVAILABLE_BLOCKS, AvailableBlocksCmd()):
            available = state.available_blocks()
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.AVAILABLE_COUNT,
                payload=AvailableCountResp(count=available)
            )

        case (CommandKind.FREE_BLOCK, FreeBlockCmd(block_id)):
            state.free_block(block_id)
            return None

        case (CommandKind.FREE_BLOCKS, FreeBlocksCmd(block_id_offset=offset, count=c)):
            state.free_blocks_range(offset, c)
            return None

        case (CommandKind.COPY, CopyCmd(src_block_id=src, dst_block_id=dst, src_start=_, dst_start=_, length=_)):
            state.copy_blocks(src, dst)
            return None

        case (CommandKind.DROP, DropCmd(block_id=b, start=_, end=_)):
            state.drop_block(b)
            return None

        case _:
            # This should never happen if all commands are covered
            raise ValueError("Unhandled command variant")


def response_to_dict(resp: Response) -> list:
    """
    Convert a typed Response into a list/dict structure suitable for msgpack.
    Returns something like:
        [ <16-byte instance_id>,
          { "AllocatedBlocks": [offset, count] }
        ]
    """
    match resp.kind, resp.payload:
        case (ResponseKind.ALLOCATED_BLOCKS, AllocatedBlocksResp(offset, count)):
            return [
                resp.instance_id,
                {"AllocatedBlocks": [offset, count]}
            ]
        case (ResponseKind.AVAILABLE_COUNT, AvailableCountResp(count)):
            return [
                resp.instance_id,
                {"AvailableCount": [count]}
            ]
        case (ResponseKind.ERROR, ErrorResp(error_code, msg)):
            return [
                resp.instance_id,
                {"Error": [error_code, msg]}
            ]
        case (ResponseKind.AWK, AwkResp(message)):
            return [
                resp.instance_id,
                {"Awk": [message]}
            ]
        case _:
            raise ValueError("Unknown response type")


def worker_thread(worker_queue, state: ServerState, response_queue):
    while True:
        item = worker_queue.get()
        if item is None:
            # shutdown
            break
        client_id, req = item

        try:
            resp = handle_command(req, state)
            if resp is not None:
                response_queue.put((client_id, resp))
        except Exception as exc:
            print(f"[Worker] Error: {exc} for message {req}")

            err_resp = Response(
                instance_id=req.instance_id,
                kind=ResponseKind.ERROR,
                payload=ErrorResp(error_code=100, message=str(exc)),
            )
            response_queue.put((client_id, err_resp))

        worker_queue.task_done()


def main():
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.bind("tcp://*:5555")
    print("Server listening on tcp://*:5555")

    state = ServerState(capacity=10_000)

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
        # Receive a multipart message from the client: [client_id, payload]
        msg_parts = router.recv_multipart()
        if len(msg_parts) < 2:
            continue

        client_id = msg_parts[0]
        payload = msg_parts[1]

        # Unpack from MessagePack
        message_list = msgpack.unpackb(payload, raw=False)
        req = parse_incoming_message(message_list)

        worker_index = hash(req.instance_id) % num_workers
        worker_queues[worker_index].put((client_id, req))

        # Drain any ready responses
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
