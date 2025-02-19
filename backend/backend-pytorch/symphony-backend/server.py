from __future__ import annotations

import threading
import typing
from dataclasses import dataclass
from enum import StrEnum
from typing import Union
import numpy as np
import torch

from blocks import KvBlockManager, BlockStorage, KvBlock, Address, Address, TokenEmbedManager, TokenEmbedStorage, KvBlockStorage

type InstanceId = bytes


@dataclass
class TextToken:
    token_id: int
    position_id: int


@dataclass
class ImageToken:
    embed_addr: Address
    position_id: tuple[int, int, int]


@dataclass
class VideoToken:
    embed_addr: Address
    position_id: tuple[int, int, int]


Token = Union[TextToken, ImageToken, VideoToken]


# =============================================================================
# 1) Enums & Dataclasses for Commands/Responses
# =============================================================================

class CommandKind(StrEnum):
    # Block-level operations
    ALLOCATE_KV_BLOCKS = "AllocateKvBlocks"
    DEALLOCATE_KV_BLOCKS = "FreeKvBlocks"
    AVAILABLE_KV_BLOCKS = "AvailableKvBlocks"

    # Block sharing
    EXPORT_KV_BLOCKS = "ExportKvBlocks"
    IMPORT_KV_BLOCKS = "ImportKvBlocks"

    # Token-level operations
    COPY_KV_BLOCK = "CopyTokens"
    MASK_KV_BLOCK = "DropTokens"
    FILL_KV_BLOCK = "FillKvBlock"

    # Embedding vector operations (input/output)
    ALLOCATE_TOKEN_EMBEDS = "AllocateTokenEmbeds"
    DEALLOCATE_TOKEN_EMBEDS = "DeallocateTokenEmbeds"
    AVAILABLE_TOKEN_EMBEDS = "AvailableTokenEmbeds"

    #  Input embedding operations
    EMBED_IMAGE = "EmbedImage"
    EMBED_VIDEO = "EmbedVideo"

    # Output embedding operations
    GET_NEXT_TOKEN_DIST = "GetNextTokenDist"
    GET_FEATURE_VECTOR = "GetFeatureVector"
    DECODE = "Decode"

    # Add more if needed


######## BLOCK ALLOC/DEALLOC ########

@dataclass
class AllocateKvBlocksCmd:
    num_blocks: int


@dataclass
class DeallocateKvBlocksCmd:
    addr_offset: int
    count: int


@dataclass
class AvailableKvBlocksCmd:
    pass


@dataclass
class ExportKvBlocksCmd:
    resource_name: str
    addr_offset: int
    count: int


@dataclass
class ImportKvBlocksCmd:
    resource_name: str


######## BLOCK CTRL ########

@dataclass
class FillKvBlockCmd:
    addr: Address
    ctx_addrs: list[Address]
    mask: list[bool]

    input_embeds: list[Token]
    output_embeds: list[Address]
    output_embed_offset: int


@dataclass
class CopyKvBlockCmd:
    src_addr: int
    dst_addr: int
    src_token_offset: int
    dst_token_offset: int
    token_count: int


@dataclass
class MaskKvBlockCmd:
    addr: int
    token_offset: int
    token_count: int


######## BLOCK ALLOC/DEALLOC ########

@dataclass
class AllocateTokenEmbedsCmd:
    num_tokens: int


@dataclass
class DeallocateTokenEmbedsCmd:
    addr_offset: int
    count: int


@dataclass
class AvailableTokenEmbedsCmd:
    ...


#### INPUT EMBEDDING ####

@dataclass
class EmbedImageCmd:
    image_url: str


@dataclass
class EmbedVideoCmd:
    video_url: str


#### OUTPUT EMBEDDING ####


@dataclass
class DecodeCmd:
    ...


@dataclass
class GetNextTokenDistCmd:
    addr: int
    offset: int
    size: int
    drop_output_embed: bool


@dataclass
class GetFeatureVectorCmd:
    addr: int
    offset: int
    size: int
    drop_output_embed: bool


CommandPayload = Union[
    AllocateKvBlocksCmd,
    DeallocateKvBlocksCmd,
    AvailableKvBlocksCmd,
    ExportKvBlocksCmd,
    ImportKvBlocksCmd,
    FillKvBlockCmd,
    CopyKvBlockCmd,
    MaskKvBlockCmd,
    AllocateTokenEmbedsCmd,
    DeallocateTokenEmbedsCmd,
    AvailableTokenEmbedsCmd,
    DecodeCmd,
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


# Any reusable states, blocks and embeddings
class Resource:
    owner_inst_id: InstanceId

    block_addr: Address
    block_offset: int

    def __init__(self, owner_inst_id: InstanceId, block_addr: Address, block_offset: int):
        self.owner_inst_id = owner_inst_id
        self.block_addr = block_addr
        self.block_offset = block_offset


class Instance:
    owned_resources: list[str]
    usage_stats: dict[str, int]

    def __init__(self):
        self.owned_resources = []
        self.usage_stats = {}


class ServerState:
    # resource name resolution.

    # public resources should be visible to the users.
    resources: dict[str, Resource]
    instances: dict[InstanceId, Instance]

    # state management
    block_manager: KvBlockManager
    input_manager: TokenEmbedManager
    output_manager: TokenEmbedManager

    # command batcher
    fill_cmd_batcher: FillBlockCmdBatcher
    img_cmd_batcher: CreateImageTokensCmdBatcher

    def __init__(self, block_storage: KvBlockStorage, embedding_storage: TokenEmbedStorage):
        self.resources = {}
        self.instances = {}

        self.block_manager = KvBlockManager(block_storage)

        # input and output managers actually share the same physical storage
        self.input_manager = TokenEmbedManager(embedding_storage)
        self.output_manager = TokenEmbedManager(embedding_storage)

        # command batchers
        self.fill_cmd_batcher = FillBlockCmdBatcher()
        self.img_cmd_batcher = CreateImageTokensCmdBatcher()

    def init_instance(self, inst_id: InstanceId):
        self.instances[inst_id] = Instance()

    def delete_instance(self, inst_id: InstanceId):

        # release all the resources
        for r in self.instances[inst_id].owned_resources:
            del self.resources[r]

        del self.instances[inst_id]

    def allocate_kv_blocks(self, inst_id: InstanceId, num_blocks: int) -> list[Address]:
        new_block_ids = self.block_manager.allocate_blocks(inst_id, num_blocks)
        return new_block_ids

    def deallocate_kv_blocks(self, inst_id: InstanceId, addr_offset: Address, count: int):
        inst = self.instances[inst_id]

        # check if it deallocates the exported resources
        if inst.owned_resources:
            for r in inst.owned_resources:
                resource = self.resources[r]
                if addr_offset <= resource.block_addr < addr_offset + count:
                    del self.resources[r]

        # even if the other instances are using the deleted blocks,
        # that's okay because their refcount will not be zero yet.
        # they will be released when the refcount reaches zero.

        self.block_manager.delete_blocks(inst_id, list(range(addr_offset, addr_offset + count)))

    def fill_kv_block(self, inst_id: InstanceId, addr: Address, ctx_addrs: list[Address], block_mask: list[list[bool]], embeddings: list[Token], retain_output_embed: bool):
        # first validate if all the blocks are in the storage

        addr_space = self.block_manager.addr_space[inst_id]

        # translate all block ids into block ptr
        ctx_block_ptrs = []
        for b_id in ctx_addrs:
            b_ptr = addr_space.resolve(b_id)
            ctx_block_ptrs.append(b_ptr)

        block_ptr = addr_space.resolve(addr)

        # self.cmd_batcher << accepts block pointers.
        self.fill_cmd_batcher.add
        ...

    def available_kv_blocks(self) -> int:
        return self.block_manager.num_free_blocks()

    def export_blocks(self, inst_id: InstanceId, resource_name: str, offset: Address, count: int):
        self.resources[resource_name] = Resource(inst_id, offset, count)
        self.instances[inst_id].owned_resources.append(resource_name)

    def import_blocks(self, inst_id: InstanceId, resource_name: str):
        resource = self.resources[resource_name]
        src_addrs = list(range(resource.block_addr, resource.block_addr + resource.block_offset))
        self.block_manager.allocate_linked_blocks(inst_id, resource.owner_inst_id, src_addrs)

    def copy_tokens(self, inst_id: InstanceId, src_block_id: Address, dst_block_id: Address, src_offset: int, dst_offset: int, size: int):
        self.block_manager.copy_tokens(inst_id, src_block_id, dst_block_id, src_offset, dst_offset, size)

    def drop_tokens(self, inst_id: InstanceId, block_id: Address, offset: int, size: int):
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
        case (CommandKind.ALLOCATE_KV_BLOCKS, [num_blocks]):
            payload = AllocateKvBlocksCmd(num_blocks=num_blocks)

        case (CommandKind.DEALLOCATE_KV_BLOCKS, [offset, count]):
            payload = DeallocateKvBlocksCmd(addr_offset=offset, count=count)

        case (CommandKind.FILL_KV_BLOCK, [block_id, ctx_block_ids, block_mask, embeddings, retain_output_embed]):
            payload = FillKvBlockCmd(addr=block_id, ctx_addrs=ctx_block_ids, mask=block_mask, tokens=embeddings, retain_output_embed=retain_output_embed)

        case (CommandKind.AVAILABLE_KV_BLOCKS, []):
            payload = AvailableKvBlocksCmd()

        case (CommandKind.COPY_TOKENS, [src, dst, src_offset, dst_offset, size]):
            payload = CopyKvBlockCmd(
                src_addr=src, dst_addr=dst,
                src_token_offset=src_offset, dst_token_offset=dst_offset, size=size
            )

        case (CommandKind.MASK_KV_BLOCK, [block_id, offset, size]):
            payload = MaskKvBlockCmd(addr=block_id, offset=offset, size=size)

        case (CommandKind.EMBED_IMAGE, [image_url]):
            payload = EmbedImageCmd(image_url=image_url)

        case (CommandKind.EMBED_VIDEO, [video_url]):
            payload = EmbedVideoCmd(video_url=video_url)

        case (CommandKind.GET_NEXT_TOKEN_DIST, [block_id, offset, size, drop_output_embed]):
            payload = GetNextTokenDistCmd(addr=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case (CommandKind.GET_FEATURE_VECTOR, [block_id, offset, size, drop_output_embed]):
            payload = GetFeatureVectorCmd(addr=block_id, offset=offset, size=size, drop_output_embed=drop_output_embed)

        case _:
            raise ValueError(f"Invalid parameters for command: {cmd_str}")

    return Request(
        instance_id=instance_id,
        kind=kind,
        payload=payload,
    )


def pad_list(x, max_len, pad_value):
    if len(x) > max_len:
        raise ValueError("List is too long")
    else:
        return x + [pad_value] * (max_len - len(x))


def handle_command(state: ServerState, req: Request) -> tuple[Response | None, bool]:
    """
    Process a typed request, returning a Response if the command needs one.
    """

    # this function may DENY the request if the command is not ready to be processed.

    match req.kind, req.payload:
        case (CommandKind.ALLOCATE_KV_BLOCKS, AllocateKvBlocksCmd(num_blocks)):
            allocated_ids = state.allocate_kv_blocks(num_blocks)
            if not allocated_ids:
                raise ValueError("Allocation returned empty")
            offset = allocated_ids[0]
            count = len(allocated_ids)
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.ALLOCATED_BLOCKS,
                payload=AllocatedBlocksResp(block_id_offset=offset, count=count)
            ), True

        case (CommandKind.FILL_KV_BLOCK, FillKvBlockCmd(addr=block_id, ctx_addrs=ctx_block_ids, mask=block_mask, tokens=embeddings, retain_output_embed=retain_output_embed)):

            # inspect the embedding to see if they are ready. if not, return None, False
            for b in embeddings:
                if isinstance(b, ImageToken):
                    if not state.img_cmd_batcher.is_ready(b.image_url):
                        return None, False

            # fill the block with the position id and occupancy mask
            tgt_block = state.block_manager.get_block(req.instance_id, block_id)
            tgt_block.set_occupancy(pad_list([True] * len(embeddings), tgt_block.size, False))
            tgt_block.set_position_ids(pad_list([b.position_id for b in embeddings], tgt_block.size, 0))

            tgt_block.set_filled(False)

            cmd = _FillBlockCmd(
                block=tgt_block,
                ctx_blocks=[state.block_manager.get_block(req.instance_id, b_id) for b_id in ctx_block_ids],
                block_mask=block_mask,
                embeddings=embeddings,
                retain_output_embed=retain_output_embed
            )

            state.fill_cmd_batcher.add(cmd)

            return None, True

        case (CommandKind.DEALLOCATE_KV_BLOCKS, DeallocateKvBlocksCmd(addr_offset=offset, count=c)):
            state.deallocate_kv_blocks(req.instance_id, offset, c)
            return None, True

        case (CommandKind.AVAILABLE_KV_BLOCKS, AvailableKvBlocksCmd()):
            available = state.available_kv_blocks()
            return Response(
                instance_id=req.instance_id,
                kind=ResponseKind.AVAILABLE_COUNT,
                payload=AvailableCountResp(count=available)
            ), True

        case (CommandKind.COPY_TOKENS, CopyKvBlockCmd(src_addr=src, dst_addr=dst, src_token_offset=src_offset, dst_token_offset=dst_offset, size=size)):
            state.copy_tokens(req.instance_id, src, dst, src_offset, dst_offset, size)
            return None, True

        case (CommandKind.MASK_KV_BLOCK, MaskKvBlockCmd(addr=b, offset=offset, size=size)):
            state.drop_tokens(req.instance_id, b, offset, size)
            return None, True

        case (CommandKind.EMBED_IMAGE, EmbedImageCmd(image_url=url)):
            return None, True

        case (CommandKind.EMBED_VIDEO, EmbedVideoCmd(video_url=url)):
            return None, True

        case (CommandKind.GET_NEXT_TOKEN_DIST, GetNextTokenDistCmd(addr=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None, True

        case (CommandKind.GET_FEATURE_VECTOR, GetFeatureVectorCmd(addr=b, offset=offset, size=size, drop_output_embed=drop_output_embed)):
            return None, True

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
    block: KvBlock
    ctx_blocks: list[KvBlock]
    ctx_block_mask: list[bool]


@dataclass
class _FillBlockCmd:
    block: KvBlock
    ctx_blocks: list[KvBlock]
    block_mask: list[bool]
    embeddings: list[Token]
    retain_output_embed: bool



class CreateImageTokensCmdBatcher:
    pass
