import sys
import os
import time

# Add the parent directory to sys.path for importing
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from ops import *

@torch.inference_mode()
def test_rope_correctness():
    head_dim = 64
    batch_size = 100
    block_size = 8
    num_head = 32
    max_pos = 8192
    device = 'cuda'

    k = torch.randn((batch_size, num_head, block_size, head_dim), device=device)
    k2 = torch.clone(k)
    k3 = torch.clone(k)
    # Shape: (batch_size, )
    start_pos = torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device)
    # Shape: (max_pos, head_dim)
    rope_cache = create_rope_cache(max_pos, head_dim, torch.float32, device)
    # Shape: (batch_size, block_size)
    cache_idxs = start_pos[:, None] + torch.tensor(list(range(block_size)), dtype=torch.int32, device=device)[None, :]
    # Shape: (batch_size, block_size, head_dim)
    pre_indexed_cache = rope_cache[cache_idxs]

    k_pos1 = rope_baseline(rope_cache, k, start_pos)
    k_pos2 = rope_baseline_no_cache(k, start_pos)
    k_pos_triton = rope(rope_cache, k, start_pos)
    k_pos_new = rope_pre_indexed(pre_indexed_cache, k2)
    k_pos_sg = rope_scatter_gather(rope_cache, k3, cache_idxs)

    assert torch.allclose(k_pos2, k_pos1, atol=1e-6)
    assert torch.allclose(k_pos1, k_pos_triton, atol=1e-6)
    assert torch.allclose(k_pos_new, k_pos_triton, atol=1e-6)
    assert torch.allclose(k_pos_sg, k_pos_triton, atol=1e-6)


@torch.inference_mode()
def test_rope_performance():
    head_dim = 64
    batch_size = 1000
    block_size = 64
    num_head = 32
    max_pos = 8192
    warmup_round = 4
    test_round = 16
    device = 'cuda'

    k = torch.randn((batch_size, num_head, block_size, head_dim), device=device)
    start_pos = torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device)
    rope_cache = create_rope_cache(max_pos, head_dim, torch.float32, device)

    compiled_baseline = torch.compile(rope_baseline)

    torch.cuda.synchronize()
    for _ in range(warmup_round):
        rope_baseline(rope_cache, k, start_pos)

    torch.cuda.synchronize()
    pytorch_start = time.time()
    for _ in range(test_round):
        compiled_baseline(rope_cache, k, start_pos)
    torch.cuda.synchronize()
    pytorch_elapsed = time.time() - pytorch_start

    torch.cuda.synchronize()
    for _ in range(warmup_round):
        rope(rope_cache, k, start_pos)

    torch.cuda.synchronize()
    triton_start = time.time()
    for _ in range(test_round):
        rope(rope_cache, k, start_pos)
    torch.cuda.synchronize()
    triton_elapsed = time.time() - triton_start

    cache_idxs = start_pos[:, None] + torch.tensor(list(range(block_size)), dtype=torch.int32, device=device)[None, :]

    torch.cuda.synchronize()
    for _ in range(warmup_round):
        pre_indexed_cache = rope_cache[cache_idxs]
        rope_pre_indexed(pre_indexed_cache, k)

    torch.cuda.synchronize()
    pre_indexed_start = time.time()
    for _ in range(test_round):
        pre_indexed_cache = rope_cache[cache_idxs]
        rope_pre_indexed(pre_indexed_cache, k)
    torch.cuda.synchronize()
    pre_indexed_elapsed = time.time() - pre_indexed_start

    torch.cuda.synchronize()
    for _ in range(warmup_round):
        rope_scatter_gather(rope_cache, k, cache_idxs)

    torch.cuda.synchronize()
    sg_start = time.time()
    for _ in range(test_round):
        rope_scatter_gather(rope_cache, k, cache_idxs)
    torch.cuda.synchronize()
    sg_elapsed = time.time() - sg_start

    # assert sg_elapsed < triton_elapsed < pytorch_elapsed < pre_indexed_elapsed

    print('Pytorch: %.2f ms' % (pytorch_elapsed * 1000))
    print('Triton-old: %.2f ms' % (triton_elapsed * 1000))
    print('Triton-pre-indexed: %.2f ms' % (pre_indexed_elapsed * 1000))
    print('Triton-scatter-gather: %.2f ms' % (sg_elapsed * 1000))


@torch.inference_mode()
def test_qkv_attention():
    device = torch.device('cuda')

    #### CREATE A DUMMY MODEL ####

    num_head = 9
    head_dim = 32
    hidden_size = head_dim * num_head
    num_kv_head = num_head // 1

    q_proj = nn.Linear(hidden_size, num_head * head_dim, bias=False, device=device)

    #### CREATE A DUMMY KV CACHE ####
    NUM_TOTAL_BLOCKS = 128
    NUM_TOK_PER_BLK = 32

    kv_cache_table = torch.randn(NUM_TOTAL_BLOCKS, num_kv_head, NUM_TOK_PER_BLK * 2, head_dim, device=device)

    #### CREATE A DUMMY TASK BATCH ####
    NUM_BLK_PER_CHUNK = 3
    BATCH_SIZE = 5
    tasks = []

    for _ in range(BATCH_SIZE):
        # pick random number between 1 and 10
        num_blk_in_req = random.randint(1, 10)

        # select `num_total_blocks` amount of random block ids (does not need to be unique)
        ctx_ids = random.choices(range(NUM_TOTAL_BLOCKS), k=num_blk_in_req)

        # create a random mask (num_total_blocks * block_size, block_size) using numpy
        mask = np.random.choice([0, 1], size=(NUM_TOK_PER_BLK, num_blk_in_req * NUM_TOK_PER_BLK), p=[0.1, 0.9])

        # ensure the first block is always True (This is to ensure that the softmax is not NaN)
        mask[:, 0] = 1

        tasks.append((ctx_ids, mask))

    inp_baseline = construct_input_baseline(tasks, num_tok_per_blk=NUM_TOK_PER_BLK, device=device)
    inp = construct_input(tasks, num_blk_per_chunk=NUM_BLK_PER_CHUNK, num_tok_per_blk=NUM_TOK_PER_BLK, device=device)

    # simulate the previous state
    hidden_states = torch.randn(BATCH_SIZE, NUM_TOK_PER_BLK, hidden_size, device=device)

    q = q_proj(hidden_states)
    q = q.view(BATCH_SIZE, NUM_TOK_PER_BLK, num_head, head_dim).transpose(1, 2)

    # attention
    y1 = qkv_attention_baseline(
        q,
        kv_cache_table,
        inp_baseline['q_idxs'],
        inp_baseline['kv_idxs'],
        inp_baseline['mask']
    )

    y2 = qkv_attention(
        q,
        kv_cache_table,
        inp['q_idxs'],
        inp['kv_idxs'],
        inp['masks'],
        inp['reduce_grps'],
        False
    )

    print('y1, y2', torch.abs(y1 - y2).sum())
    assert torch.allclose(y1, y2, atol=1e-6)


def construct_input(
    reqs: list[tuple[list[int], np.ndarray]],
    num_blk_per_chunk: int,
    num_tok_per_blk: int,
    device: torch.device
) -> dict:
    num_chunk_of_reqs = [ceil_div(len(block_idxs), num_blk_per_chunk) for block_idxs, _ in reqs]
    num_chunk_in_batch = sum(num_chunk_of_reqs)
    batch_size = len(reqs)

    q_idxs = np.zeros((num_chunk_in_batch, 1), dtype=np.int32)
    kv_idxs = np.zeros((num_chunk_in_batch, num_blk_per_chunk), dtype=np.int32)
    reduce_grps = np.zeros((batch_size, max(num_chunk_of_reqs)), dtype=np.int32) - 1
    masks = np.zeros((num_chunk_in_batch, num_tok_per_blk, num_blk_per_chunk * num_tok_per_blk), dtype=np.bool_)

    # Unique index for each chunk in the batch
    glb_chunk_idx = 0

    for req_idx, req in enumerate(reqs):

        block_idxs, mask = req

        num_chunk_in_req = ceil_div(len(block_idxs), num_blk_per_chunk)

        for req_chunk_idx in range(num_chunk_in_req):
            start = req_chunk_idx * num_blk_per_chunk
            end = min(start + num_blk_per_chunk, len(block_idxs))

            q_idxs[glb_chunk_idx] = req_idx
            kv_idxs[glb_chunk_idx, :end - start] = block_idxs[start:end]
            masks[glb_chunk_idx, :, : (end - start) * num_tok_per_blk] = mask[
                :, start * num_tok_per_blk : end * num_tok_per_blk
            ]

            # if all items in the chunk are False, then it will cause NaN in softmax. Check:
            if not masks[glb_chunk_idx].any():
                raise ValueError('All items in the chunk are False. This will cause NaN in softmax.')

            reduce_grps[req_idx, req_chunk_idx] = glb_chunk_idx

            glb_chunk_idx += 1

    return {
        'q_idxs': torch.as_tensor(q_idxs, dtype=torch.long, device=device),
        'kv_idxs': torch.as_tensor(kv_idxs, dtype=torch.long, device=device),
        'reduce_grps': torch.as_tensor(reduce_grps, dtype=torch.long, device=device),
        'masks': torch.as_tensor(masks, dtype=torch.bool, device=device)
    }


def construct_input_baseline(
    reqs: list[tuple[list[int], np.ndarray]],
    num_tok_per_blk: int,
    device: torch.device
) -> dict:
    batch_size = len(reqs)

    # Pad all requests to the same maximum length
    num_blk_per_req = max(len(block_idxs) for block_idxs, _ in reqs)

    kv_idxs = np.zeros((batch_size, num_blk_per_req), dtype=np.int32)
    q_idxs = np.zeros((batch_size, 1), dtype=np.int32)
    masks = np.zeros((batch_size, num_tok_per_blk, num_blk_per_req * num_tok_per_blk), dtype=np.bool_)

    for i, req in enumerate(reqs):
        block_idxs, mask = req
        q_idxs[i] = i
        kv_idxs[i, : len(block_idxs)] = block_idxs
        masks[i, :, : len(block_idxs) * num_tok_per_blk] = mask

    return {
        'q_idxs': torch.as_tensor(q_idxs, device=device),
        'kv_idxs': torch.as_tensor(kv_idxs, device=device),
        'mask': torch.as_tensor(masks, device=device)
    }
