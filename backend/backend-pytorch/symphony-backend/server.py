# a single inference task
class Task:
    token_ids: list[int]
    position_ids: list[int]

    block_ids: list[int] # could be compressed bitset

    # sink ids (basically the placement of new tokens in the block)
    sink_block_ids: list[int]
    first_block_offset: int

    # mask_flag
    mask_flag: list[int] # causal & empty
    mask_blocks: list[tuple[int, int]]


    # number of distributions to retain
    n_dist: int
    ...

# block-based
# {block, new_block, position_ids, token_ids, mask}