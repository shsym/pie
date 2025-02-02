import typing
from typing import Union


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


class BlockFill:
    target_block_id: int
    context_blocks: list[int]
    embeddings: list[Embedding]
    mask: list[int]  # indices of the blocks to mask
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

    target_block_ids:int
    # len(block_ids) = segment_size
    ctx_seg_block_ids: list[int]
    mask_blocks: list[bool]  # indices of the blocks to mask


class Batcher:

    def __init__(self):

        ...

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
