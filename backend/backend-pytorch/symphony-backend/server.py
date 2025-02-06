import threading
import typing
from dataclasses import dataclass
from enum import StrEnum
from typing import Union
import numpy as np

from blocks import BlockManager, BlockStorage, Block, BlockId


class TextTokenEmbedding:
    token_id: int
    position_id: int


class ImageTokenEmbedding:
    image_url: str  # the url is used as an identifier
    position_id: tuple[int, int, int]


class VideoTokenEmbedding:
    video_url: str  # the url is used as an identifier
    position_id: tuple[int, int, int]


TokenEmbedding = Union[TextTokenEmbedding, ImageTokenEmbedding]


# =============================================================================
# 1) Enums & Dataclasses for Commands/Responses
# =============================================================================

class CommandKind(StrEnum):
    # Block-level operations
    ALLOCATE_BLOCK = "AllocateBlock"
    FILL_BLOCK = "FillBlock"
    FREE_BLOCK = "FreeBlock"
    AVAILABLE_BLOCKS = "AvailableBlocks"

    # Token-level operations
    COPY_TOKENS = "Copy"
    DROP = "Drop"

    # Embedding-level operations
    EMBED_IMAGE = "EmbedImage"
    EMBED_VIDEO = "EmbedVideo"

    # Add more if needed


@dataclass
class AllocateBlockCmd:
    num_blocks: int


@dataclass
class FillBlockCmd:
    block_ids: list[int]
    ctx_block_ids: list[int]
    block_mask: list[list[bool]]  # 2d array of bools
    embeddings: list[TokenEmbedding]


@dataclass
class CopyCmd:
    src_block_id: int
    dst_block_id: int
    src_offset: int
    dst_offset: int
    size: int


@dataclass
class DropCmd:
    block_id: int
    offset: int
    size: int


@dataclass
class FreeBlockCmd:
    block_id_offset: int
    count: int


@dataclass
class AvailableBlocksCmd:
    pass


@dataclass
class EmbedImageCmd:
    image_url: str


@dataclass
class EmbedVideoCmd:
    video_url: str


CommandPayload = Union[
    AllocateBlockCmd,
    FillBlockCmd,
    FreeBlockCmd,
    AvailableBlocksCmd,
    CopyCmd,
    DropCmd,
    EmbedImageCmd,
    EmbedVideoCmd
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

    block_manager: BlockManager

    def __init__(self, block_storage: BlockStorage):

        self.block_manager = BlockManager(block_storage)

    def allocate_blocks(self, num_blocks: int) -> list[BlockId]:
        new_block_ids = self.block_manager.create_blocks(num_blocks)
        return new_block_ids

    def available_blocks(self) -> int:
        return self.block_manager.num_free_blocks()

    def free_block(self, block_id: BlockId):
        self.block_manager.delete_block(block_id)

    def free_blocks_range(self, offset: BlockId, count: int):
        self.block_manager.delete_blocks(list(range(offset, offset + count)))

    def copy_block(self, src_block_id: BlockId, dst_block_id: BlockId, src_offset:int, dst_offset:int, size:int):
        self.block_manager.copy_block(src_block_id, dst_block_id, src_offset, dst_offset, size)

    def drop_block(self, block_id: int, offset: int, size: int):
        self.block_manager.drop_block


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


def ceil_div(a, b):
    return -(a // -b)


class TokenEmbeddingElement:
    # token embeddings are not separately computed
    token_id: int
    position_id: int


class ImageEmbeddingElement:
    vector_id: int  # handle to the already computed image embedding vector
    position_id: tuple[int, int, int]


Embedding = typing.Union[TokenEmbeddingElement, ImageEmbeddingElement]


class Block:
    block_id: int
    output_embed_id: int
    # two needed for masking
    empty_mask: list[int]
    position_ids: list[int]


class BlockFillCmd:
    target_block_id: int
    context_blocks: list[int]
    embeddings: list[Embedding]
    mask_indices: list[int]  # indices of the blocks to mask
    retain_output_embed: bool


########
# 1  2  3  4  [13]
# 5  6  7  8  [13]
# 9  10 11 12 [13]
# 13  .  .  . [13]
# 1  2  3  4  [14]
# 5  6  7  8  [14]
# 9  10 11 12 [14]
# 13 14  .  . [14]


# ([Block] [Block] [Block]) -> Chunk
# ([Chunk]
#   [Chunk]
#   [Chunk]) -> Command


#######


class BatchItem:
    target_block_ids: int
    # len(block_ids) = segment_size
    ctx_seg_block_ids: list[int]
    mask_blocks: list[bool]  # indices of the blocks to mask

    def __init__(self, target_block_ids, ctx_seg_block_ids, mask_blocks):
        self.target_block_ids = target_block_ids
        self.ctx_seg_block_ids = ctx_seg_block_ids
        self.mask_blocks = mask_blocks


class CommandBatcher:
    tasks: list[BlockFillCmd]
    items: list[BatchItem]

    def __init__(self):

        self.tasks = []
        self.items = []

    def add_task(self, task: BlockFillCmd):

        self.tasks.append(task)

    def get_segment_size(self) -> int:

        # analyze the ideal segment size
        # wasted computation due to sparsity vs. extra reduction operation

        # just use the median of the context block sizes

        num_blocks = []

        for task in self.tasks:
            num_blocks.append(len(task.context_blocks))

        return int(np.median(num_blocks).item())

    def batch(self):

        segment_size = self.get_segment_size()

        for task in self.tasks:

            num_segments = ceil_div(len(task.context_blocks), segment_size)

            for i in range(num_segments):
                start = i * segment_size
                end = min((i + 1) * segment_size, len(task.context_blocks))

                ctx_seg_block_ids = task.context_blocks[start:end]
                mask_blocks = [False] * segment_size

                if len(ctx_seg_block_ids) < segment_size:
                    # 0 is a special block id for padding
                    pad_size = segment_size - len(ctx_seg_block_ids)
                    ctx_seg_block_ids += [0] * pad_size
                    mask_blocks[segment_size - pad_size:] = [True] * pad_size

                # check if the mask is needed
                for mi in task.mask_indices:
                    if start <= mi < end:
                        mask_blocks[mi - start] = True

                item = BatchItem(
                    target_block_ids=task.target_block_id,
                    ctx_seg_block_ids=ctx_seg_block_ids,
                    mask_blocks=mask_blocks
                )

                self.items.append(item)

        ...

    def clear(self):
        ...


# a single inference task
class Task:
    token_ids: list[int]
    position_ids: list[int]

    block_ids: list[int]  # could be compressed bitset

    # sink ids (basically the placement of new tokens in the block)
    sink_block_ids: list[int]
    first_block_offset: int

    # mask_flag
    mask_flag: list[int]  # causal & empty
    mask_blocks: list[tuple[int, int]]

    # number of distributions to retain
    n_dist: int
    ...


# block-based
# {block, new_block, position_ids, token_ids, mask}


def main():
    # list of inference commands
    cmd_list = []

    ...
