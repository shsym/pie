from dataclasses import dataclass
from enum import Enum

import numpy as np


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
    start: int
    end: int


@dataclass
class MaskBlockCommand:
    block_id: int
    start: int
    end: int


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


class Engine:
    def __init__(self):
        self.embeds = {}

        # llm
        # tokens

    def allocate(self, cmds: list[AllocateCommand]):
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity, by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.
        ...

    def deallocate(self, cmds: list[AllocateCommand]):
        ...

    def embed_text(self, cmds: list[EmbedTextCommand]):
        ...

    def embed_image(self, cmds: list[EmbedImageCommand]):
        # if the input/output embeds are not "in use", do it one the second buffer.
        ...

    def fill_block(self, cmds: list[FillBlockCommand]):



        ...

    def mask_block(self, cmds: list[MaskBlockCommand]):
        ...

    def copy_block(self, cmds: list[CopyBlockCommand]):
        ...

    def decode_token_distribution(self, cmds: list[DecodeTokenDistributionCommand]):
        ...

    def sample_top_k_request(self, cmds: list[SampleTopKCommand]) -> list[SampleTopKResponse]:
        ...

    def get_token_distribution(self, cmds: list[GetTokenDistributionCommand]) -> list[GetTokenDistributionResponse]:
        ...



def batch_fill_block(cmds: list[FillBlockCommand]):

    # first, let's get the chunk size.

    ...




class FillBlockCmdBatcher:
    queue: list[_FillBlockCmd]
    redundancy_check: dict[Address, _FillBlockCmd]

    def __init__(self):
        self.queue = []
        self.redundancy_check = {}

    def add(self, cmd: _FillBlockCmd):

        if cmd.block.pointer not in self.redundancy_check:
            self.redundancy_check[cmd.block.pointer] = cmd
            self.queue.append(cmd)

        else:
            # override the previous command
            prev_cmd = self.redundancy_check[cmd.block.pointer]
            self.queue.remove(prev_cmd)
            self.queue.append(cmd)

    def get_chunk_size(self) -> int:

        # analyze the ideal segment size
        # wasted computation due to sparsity vs. extra reduction operation

        # just use the median of the context block sizes

        num_blocks = []

        for cmd in self.queue:
            num_blocks.append(len(cmd.ctx_blocks))

        return int(np.median(num_blocks).item())

    def batch(self):

        chunk_size = self.get_chunk_size()

        cmd_groups = []  # 2d (NUM_CMDS, MAX_NUM_CHUNKS)

        # N = sum(NUM_CHUNKS per command)

        batched_tgt_block_ptrs = []  # 1d (N, 1)
        batched_ctx_block_ptrs = []  # 2d (N, CHUNK_SIZE)

        batched_token_ids = []  # 2d (N, BLOCK_SIZE)
        batched_pos_ids = []  # 2d (N, BLOCK_SIZE)
        batched_mask = []  # 3d (N, BLOCK_SIZE * CHUNK_SIZE, BLOCK_SIZE)

        token_id_map = []

        for cmd in self.queue:

            num_chunks = ceil_div(len(cmd.ctx_blocks), chunk_size)
            cmd_grp = []
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(cmd.ctx_blocks))

                ctx_blocks = cmd.ctx_blocks[start:end]
                ctx_block_ptrs = [b.pointer for b in ctx_blocks]
                ctx_block_mask = [False] * chunk_size

                # pad the chunk if it's not full
                if len(ctx_blocks) < chunk_size:
                    # 0 is a special block id for padding
                    pad_size = chunk_size - len(ctx_blocks)
                    ctx_blocks += [0] * pad_size
                    ctx_block_mask[chunk_size - pad_size:] = [True] * pad_size

                # check if the mask is needed
                for mi in cmd.block_mask:
                    if start <= mi < end:
                        ctx_block_mask[mi - start] = True

                # if the entire chunk is masked, we can save some computation. skip it
                if all(ctx_block_mask):
                    continue

                if len(cmd.embeddings) > cmd.block.size:
                    raise ValueError("Too many tokens in the block")

                # get token ids and position ids
                token_ids = []

                for b in cmd.embeddings:
                    if isinstance(b, TextToken):
                        token_ids.append(b.token_id)

                    else:
                        # just fill in stubs. It will be filled in later
                        token_ids.append(0)

                        # get the image token
                        token_id_map.append({
                            "n": len(batched_token_ids),
                            "i": len(token_ids) - 1,
                            "token": b
                        })

                # this is better than computing masks for the entire command.
                # most of the masks will be zeros anyway. so this kinda leverages the sparsity.

                ctx_pos_ids = np.hstack([b.position_ids for b in ctx_blocks])  # int
                tgt_pos_ids = np.array(cmd.block.position_ids)
                ctx_occupancy = np.hstack([b.occupancy for b in ctx_blocks])  # bool

                # get the full attn mask
                block_mask = np.repeat(ctx_block_mask, cmd.block.size)
                casual_mask = ctx_pos_ids[None, :] > tgt_pos_ids[: None]
                valid_mask = np.logical_not(ctx_occupancy[None, :])

                attn_mask = np.logical_or(np.logical_or(casual_mask, valid_mask), block_mask)

                # add the chunk to the batch
                batched_tgt_block_ptrs.append(cmd.block.pointer)
                batched_ctx_block_ptrs.append(ctx_block_ptrs)
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

        return {
            "tgt_block_ptrs": batched_tgt_block_ptrs,
            "ctx_block_ptrs": batched_ctx_block_ptrs,
            "token_ids": batched_token_ids,
            "pos_ids": batched_pos_ids,
            "mask": batched_mask,
            "cmd_groups": cmd_groups,
            "token_id_map": token_id_map
        }

    def clear(self):
        self.queue.clear()
        self.redundancy_check.clear()
