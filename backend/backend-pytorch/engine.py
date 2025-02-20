from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch

BLOCK_SIZE = 64


def ceil_div(a, b):
    return -(-a // b)


class ObjectKind(Enum):
    BLOCK = "Block"
    EMBED = "Embed"
    DIST = "Dist"


@dataclass
class AllocateCommand:
    kind: ObjectKind
    id_offset: int
    count: int


@dataclass
class EmbedTextCommand:
    embed_id: int
    token_id: int
    position_id: int


@dataclass
class EmbedImageCommand:
    embed_id: int
    url: str


@dataclass
class FillBlockCommand:
    block_id: int
    context_block_ids: list[int]
    input_embed_ids: list[int]
    output_embed_ids: list[int]


@dataclass
class MaskBlockCommand:
    block_id: int
    mask: list[bool]


@dataclass
class CopyBlockCommand:
    src_block_id: int
    dst_block_id: int
    src_start: int
    dst_start: int
    length: int


@dataclass
class DecodeTokenDistributionCommand:
    dist_id: int
    token_id: int


@dataclass
class SampleTopKCommand:
    dist_id: int
    k: int


@dataclass
class SampleTopKResponse:
    token_id: int
    prob: float


@dataclass
class GetTokenDistributionCommand:
    dist_id: int


@dataclass
class GetTokenDistributionResponse:
    token_id: int
    prob: float


@dataclass
class TextEmbed:
    token_id: int
    position_id: int


@dataclass
class ImageEmbed:
    vec_id: int
    position_id: (int, int)


@dataclass
class Block:
    position_ids: list[int]
    occupancy: list[bool]


EMPTY_BLOCK = Block(
    position_ids=[0] * BLOCK_SIZE,
    occupancy=[False] * BLOCK_SIZE
)

# union type
Embed = Union[TextEmbed, ImageEmbed]


class Engine:
    embeds: dict[int, Embed]
    blocks: dict[int, Block]

    def __init__(self, model):
        self.embeds = {}
        self.blocks = {}

        # llm
        # tokens

    def allocate(self, cmds: list[AllocateCommand]):
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity, by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.

        for cmd in cmds:
            if cmd.kind == ObjectKind.BLOCK:
                for i in range(cmd.count):
                    self.blocks[cmd.id_offset + i] = EMPTY_BLOCK

            elif cmd.kind == ObjectKind.EMBED:
                # do nothing. Embeds are allocated on the fly.
                ...
            elif cmd.kind == ObjectKind.DIST:
                # do nothing. Dists are allocated on the fly.
                ...

    def deallocate(self, cmds: list[AllocateCommand]):
        ...

    def embed_text(self, cmds: list[EmbedTextCommand]):
        for cmd in cmds:
            self.embeds[cmd.embed_id] = TextEmbed(token_id=cmd.token_id, position_id=cmd.position_id)

    def embed_image(self, cmds: list[EmbedImageCommand]):
        # unimplemented
        ...

    def mask_block(self, cmds: list[MaskBlockCommand]):
        for cmd in cmds:
            block = self.blocks[cmd.block_id]
            for i, m in enumerate(cmd.mask):
                block.occupancy[i] = m

    def copy_block(self, cmds: list[CopyBlockCommand]):
        ...

    def decode_token_distribution(self, cmds: list[DecodeTokenDistributionCommand]):
        ...

    def sample_top_k_request(self, cmds: list[SampleTopKCommand]) -> list[SampleTopKResponse]:
        ...

    def get_token_distribution(self, cmds: list[GetTokenDistributionCommand]) -> list[GetTokenDistributionResponse]:
        ...

    def fill_block(self, cmds: list[FillBlockCommand]):

        ### Step 1.Decide the `chunk size`
        # first estimate the `chunk_size` for the batch (chunk = number of blocks in one batch row)
        # if chunk size is too large -> performance will be nice, but most blocks will be "empty" ones, leading to wasted computation.
        # if chunk size is too small -> there will be only few empty blocks, but the performance could be bad for fill requests with very large contexts.
        # so we need to find a balance between the two, which I use the median of the number of context blocks in the commands.

        num_ctx_blocks = []
        for cmd in cmds:
            num_ctx_blocks.append(len(cmd.context_block_ids))

        CHUNK_SIZE = int(np.median(num_ctx_blocks))

        cmd_groups = []  # 2d (NUM_CMDS, MAX_NUM_CHUNKS)

        # N = sum(NUM_CHUNKS per command)

        batched_tgt_block_ptrs = []  # 1d (N, 1)
        batched_ctx_block_ptrs = []  # 2d (N, CHUNK_SIZE)

        batched_token_ids = []  # 2d (N, BLOCK_SIZE)
        batched_pos_ids = []  # 2d (N, BLOCK_SIZE)
        batched_mask = []  # 3d (N, BLOCK_SIZE * CHUNK_SIZE, BLOCK_SIZE)

        token_id_map = []

        for cmd in cmds:

            if len(cmd.input_embed_ids) > BLOCK_SIZE:
                raise ValueError("Too many tokens in the block")

            num_chunks = ceil_div(len(cmd.context_block_ids), CHUNK_SIZE)
            cmd_grp = []
            for i in range(num_chunks):
                start = i * CHUNK_SIZE
                end = min((i + 1) * CHUNK_SIZE, len(cmd.context_block_ids))

                ctx_block_ids = cmd.context_block_ids[start:end]
                ctx_block_mask = [False] * CHUNK_SIZE

                # pad the chunk if it's not full
                if len(ctx_block_ids) < CHUNK_SIZE:
                    # 0 is a special block id for padding
                    pad_size = CHUNK_SIZE - len(ctx_block_ids)
                    ctx_block_ids += [0] * pad_size
                    ctx_block_mask[CHUNK_SIZE - pad_size:] = [True] * pad_size

                # get token ids and position ids
                token_ids = []
                position_ids = []
                for embed_id in cmd.input_embed_ids:

                    embed = self.embeds[embed_id]

                    if isinstance(embed, TextEmbed):

                        token_ids.append(embed.token_id)
                        position_ids.append(embed.position_id)

                    else:
                        # just fill in stubs. It will be filled in later
                        token_ids.append(0)

                        # get the image token
                        token_id_map.append({
                            "n": len(batched_token_ids),
                            "i": len(token_ids) - 1,
                            "vec_id": embed.vec_id,
                            "pos_id": embed.position_id
                        })

                # this is better than computing masks for the entire command.
                # most of the masks will be zeros anyway. so this kinda leverages the sparsity.

                ctx_pos_ids = np.hstack([self.blocks[ctx_id].position_ids for ctx_id in ctx_block_ids])  # int
                tgt_pos_ids = np.array(position_ids)
                ctx_occupancy = np.hstack([self.blocks[ctx_id].occupancy for ctx_id in ctx_block_ids])  # bool

                # get the full attn mask
                block_mask = np.repeat(ctx_block_mask, BLOCK_SIZE)
                casual_mask = ctx_pos_ids[None, :] > tgt_pos_ids[: None]
                valid_mask = np.logical_not(ctx_occupancy[None, :])

                attn_mask = np.logical_or(np.logical_or(casual_mask, valid_mask), block_mask)

                # add the chunk to the batch
                batched_tgt_block_ptrs.append(cmd.block_id)
                batched_ctx_block_ptrs.append(ctx_block_ids)
                batched_token_ids.append(np.array(token_ids))
                batched_pos_ids.append(tgt_pos_ids)
                batched_mask.append(attn_mask)
                cmd_grp.append(len(batched_token_ids) - 1)

            cmd_groups.append(cmd_grp)

        # create a torch tensor
        batched_tgt_block_ptrs = torch.as_tensor(batched_tgt_block_ptrs)
        batched_ctx_block_ptrs = torch.as_tensor(batched_ctx_block_ptrs)
        batched_token_ids = torch.as_tensor(batched_token_ids)
        batched_pos_ids = torch.as_tensor(batched_pos_ids)
        batched_mask = torch.as_tensor(batched_mask)

        cmd_grp_max_size = max(len(g) for g in cmd_groups)
        cmd_groups = [g + [-1] * (cmd_grp_max_size - len(g)) for g in cmd_groups]
        cmd_groups = torch.as_tensor(cmd_groups)


        # invoke the model

        return {
            "tgt_block_ptrs": batched_tgt_block_ptrs,
            "ctx_block_ptrs": batched_ctx_block_ptrs,
            "token_ids": batched_token_ids,
            "pos_ids": batched_pos_ids,
            "mask": batched_mask,
            "cmd_groups": cmd_groups,
            "token_id_map": token_id_map
        }
