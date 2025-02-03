import typing
from typing import Union
import numpy as np


def ceil_div(a, b):
    return -(a // -b)


class TokenEmbedding:
    # token embeddings are not separately computed
    token_id: int
    position_id: int


class ImageEmbedding:
    vector_id: int  # handle to the already computed image embedding vector


Embedding = typing.Union[TokenEmbedding, ImageEmbedding]


class Block:
    block_id: int
    # two needed for masking
    empty_mask: list[int]
    position_ids: list[int]


class BlockFillCmd:
    target_block_id: int
    context_blocks: list[int]
    embeddings: list[Embedding]
    mask_indices: list[int]  # indices of the blocks to mask
    retain_output: bool


########
# 1  2  3  4  [13]
# 5  6  7  8  [13]
# 9  10 11 12 [13]
# 13  .  .  . [13]
# 1  2  3  4  [14]
# 5  6  7  8  [14]
# 9  10 11 12 [14]
# 13 14  .  . [14]

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
