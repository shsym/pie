from __future__ import annotations

import threading
import typing
from dataclasses import dataclass
from enum import StrEnum
from typing import Union
import numpy as np

from blocks import BlockManager, BlockStorage, Block, BlockId

type InstanceId = bytes


@dataclass
class TextToken:
    token_id: int
    position_id: int


@dataclass
class ImageToken:
    image_url: str  # the url is used as an identifier
    position_id: tuple[int, int, int]


@dataclass
class VideoToken:
    video_url: str  # the url is used as an identifier
    position_id: tuple[int, int, int]


Token = Union[TextToken, ImageToken, VideoToken]


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
    COPY_TOKENS = "CopyTokens"
    DROP_TOKENS = "DropTokens"

    # Embedding-level operations
    CREATE_IMAGE_TOKENS = "CreateImageTokens"
    CREATE_VIDEO_TOKENS = "CreateVideoTokens"

    # C
    GET_NEXT_TOKEN_DIST = "GetNextTokenDist"
    GET_FEATURE_VECTOR = "GetFeatureVector"

    # Add more if needed


@dataclass
class AllocateBlockCmd:
    num_blocks: int


@dataclass
class FillBlockCmd:
    block_id: int
    ctx_block_ids: list[int]
    block_mask: list[bool]
    embeddings: list[Token]
    retain_output_embed: bool


@dataclass
class GetNextTokenDistCmd:
    block_id: int
    offset: int
    size: int
    drop_output_embed: bool


@dataclass
class GetFeatureVectorCmd:
    block_id: int
    offset: int
    size: int
    drop_output_embed: bool


@dataclass
class CopyTokensCmd:
    src_block_id: int
    dst_block_id: int
    src_offset: int
    dst_offset: int
    size: int


@dataclass
class DropTokensCmd:
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
class CreateImageTokensCmd:
    image_url: str


@dataclass
class CreateVideoTokensCmd:
    video_url: str


CommandPayload = Union[
    AllocateBlockCmd,
    FillBlockCmd,
    FreeBlockCmd,
    AvailableBlocksCmd,
    CopyTokensCmd,
    DropTokensCmd,
    CreateImageTokensCmd,
    CreateVideoTokensCmd
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
    fill_cmd_batcher: FillBlockCmdBatcher
    img_cmd_batcher: CreateImageTokensCmdBatcher

    def __init__(self, block_storage: BlockStorage):
        self.block_manager = BlockManager(block_storage)
        self.fill_cmd_batcher = FillBlockCmdBatcher()

    def allocate_blocks(self, inst_id: InstanceId, num_blocks: int) -> list[BlockId]:
        new_block_ids = self.block_manager.create_blocks(inst_id, num_blocks)
        return new_block_ids

    def fill_blocks(self, inst_id: InstanceId, block_id: BlockId, ctx_block_ids: list[BlockId], block_mask: list[list[bool]], embeddings: list[Token], retain_output_embed: bool):
        # first validate if all the blocks are in the storage

        addr_space = self.block_manager.virtual_addr_space[inst_id]

        # translate all block ids into block ptr
        ctx_block_ptrs = []
        for b_id in ctx_block_ids:
            b_ptr = addr_space.translate(b_id)
            ctx_block_ptrs.append(b_ptr)

        block_ptr = addr_space.translate(block_id)

        # self.cmd_batcher << accepts block pointers.
        self.fill_cmd_batcher.add
        ...

    def free_blocks(self, inst_id: InstanceId, offset: BlockId, count: int):
        self.block_manager.delete_blocks(inst_id, list(range(offset, offset + count)))

    def available_blocks(self) -> int:
        return self.block_manager.num_free_blocks()

    def copy_tokens(self, inst_id: InstanceId, src_block_id: BlockId, dst_block_id: BlockId, src_offset: int, dst_offset: int, size: int):
        self.block_manager.copy_tokens(inst_id, src_block_id, dst_block_id, src_offset, dst_offset, size)

    def drop_tokens(self, inst_id: InstanceId, block_id: BlockId, offset: int, size: int):
        self.block_manager.drop_tokens(inst_id, block_id, offset, size)


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
        case (CommandKind.ALLOCATE_BLOCK, [num_blocks]):
            payload = AllocateBlockCmd(num_blocks=num_blocks)

        case (CommandKind.FREE_BLOCK, [offset, count]):
            payload = FreeBlockCmd(block_id_offset=offset, count=count)

        case (CommandKind.FILL_BLOCK, [block_id, ctx_block_ids, block_mask, embeddings, retain_output_embed]):
            payload = FillBlockCmd(block_id=block_id, ctx_block_ids=ctx_block_ids, block_mask=block_mask, embeddings=embeddings, retain_output_embed=retain_output_embed)

        case (CommandKind.AVAILABLE_BLOCKS, []):
            payload = AvailableBlocksCmd()

        case (CommandKind.COPY_TOKENS, [src, dst, src_offset, dst_offset, size]):
            payload = CopyTokensCmd(
                src_block_id=src, dst_block_id=dst,
                src_offset=src_offset, dst_offset=dst_offset, size=size
            )

        case (CommandKind.DROP_TOKENS, [block_id, offset, size]):
            payload = DropTokensCmd(block_id=block_id, offset=offset, size=size)

        case (CommandKind.CREATE_IMAGE_TOKENS, [image_url]):
            payload = CreateImageTokensCmd(image_url=image_url)

        case (CommandKind.CREATE_VIDEO_TOKENS, [video_url]):
            payload = CreateVideoTokensCmd(video_url=video_url)

        case (CommandKind.GET_NEXT_TOKEN_DIST, [block_id, offset, size, drop_output_embed]):
            payload = GetNextTokenDistCmd(block_id=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case (CommandKind.GET_FEATURE_VECTOR, [block_id, offset, size, drop_output_embed]):
            payload = GetFeatureVectorCmd(block_id=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case _:
            raise ValueError(f"Invalid parameters for command: {cmd_str}")

    return Request(
        instance_id=instance_id,
        kind=kind,
        payload=payload,
    )


def handle_command(req: Request, state: ServerState) -> Response | None:
    """
    Process a typed request, returning a Response if the command needs one.
    """
    match req.kind, req.payload:
        case (CommandKind.ALLOCATE_BLOCK, AllocateBlockCmd(num_blocks)):
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

        case (CommandKind.FILL_BLOCK, FillBlockCmd(block_id=block_id, ctx_block_ids=ctx_block_ids, block_mask=block_mask, embeddings=embeddings, retain_output_embed=retain_output_embed)):

            ...

        case (CommandKind.FREE_BLOCK, FreeBlockCmd(block_id_offset=offset, count=c)):
            state.free_blocks(offset, c)
            return None

        case (CommandKind.AVAILABLE_BLOCKS, AvailableBlocksCmd()):
            available = state.available_blocks()
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.AVAILABLE_COUNT,
                payload=AvailableCountResp(count=available)
            )

        case (CommandKind.COPY_TOKENS, CopyTokensCmd(src_block_id=src, dst_block_id=dst, src_offset=src_offset, dst_offset=dst_offset, size=size)):
            state.copy_tokens(src, dst, src_offset, dst_offset, size)
            return None

        case (CommandKind.DROP_TOKENS, DropTokensCmd(block_id=b, offset=offset, size=size)):
            state.drop_tokens(b, offset, size)
            return None

        case (CommandKind.CREATE_IMAGE_TOKENS, CreateImageTokensCmd(image_url=url)):
            return None

        case (CommandKind.CREATE_VIDEO_TOKENS, CreateVideoTokensCmd(video_url=url)):
            return None

        case (CommandKind.GET_NEXT_TOKEN_DIST, GetNextTokenDistCmd(block_id=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None

        case (CommandKind.GET_FEATURE_VECTOR, GetFeatureVectorCmd(block_id=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None

        case _:
            # This should never happen if all commands are covered
            raise ValueError("Unhandled command variant")


def ceil_div(a, b):
    return -(a // -b)


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


class ChunkedContext:
    block: Block
    ctx_blocks: list[Block]
    ctx_block_mask: list[bool]


class FillBlockCmdBatcher:
    tasks: list[BlockFillCmd]
    items: list[BatchItem]

    def __init__(self):

        self.tasks = []
        self.items = []

    def add(self, block_ptr: int, ctx_block_ptrs: list[int], block_mask: list[bool], embeddings: list[Token], retain_output_embed: bool = False):

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


class CreateImageTokensCmdBatcher:
    pass
