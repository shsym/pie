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
    start_pos = torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device)
    rope_cache = create_rope_cache(max_pos, head_dim, torch.float32, device)

    k_pos1 = rope_baseline(rope_cache, k, start_pos)
    k_pos2 = rope_baseline_no_cache(k, start_pos)
    k_pos_triton = rope(rope_cache, k, start_pos)

    assert torch.allclose(k_pos2, k_pos1, atol=1e-6)
    assert torch.allclose(k_pos1, k_pos_triton, atol=1e-6)


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

    for _ in range(warmup_round):
        compiled_baseline(rope_cache, k, start_pos)

    pytorch_start = time.time()
    for _ in range(test_round):
        compiled_baseline(rope_cache, k, start_pos)
    pytorch_elapsed = time.time() - pytorch_start

    for _ in range(warmup_round):
        rope(rope_cache, k, start_pos)

    triton_start = time.time()
    for _ in range(test_round):
        rope(rope_cache, k, start_pos)
    triton_elapsed = time.time() - triton_start

    assert triton_elapsed < pytorch_elapsed

    speedup = pytorch_elapsed / triton_elapsed - 1.0
    print('Speed up: %.1f%%' % (speedup * 100))




@torch.inference_mode()
def test_qkv_attention():
    device = torch.device('cuda')

    #### CREATE A DUMMY MODEL ####

    # create a dummy model
    num_heads = 9
    head_dim = 32
    hidden_size = head_dim * num_heads
    num_key_value_heads = num_heads // 1

    # create a rope cache
    # rope_cache = create_rope_cache(8192, head_dim, torch.float32, device)

    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, device=device)
    # k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device)
    # v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device)

    ###############################

    #### CREATE A DUMMY KV CACHE ####
    NUM_TOTAL_BLOCKS = 128
    BLOCK_SIZE = 32

    # create a dummy kv cache
    kv_cache_table = torch.randn(NUM_TOTAL_BLOCKS, num_key_value_heads, BLOCK_SIZE * 2, head_dim, device=device)

    ###############################

    #### CREATE A DUMMY TASK BATCH ####
    CHUNK_SIZE = 3
    NUM_REQS = 5
    tasks = []

    for _ in range(NUM_REQS):
        # pick random number between 1 and 10
        num_blocks = 10  # random.randint(1, 8)

        # select `num_total_blocks` amount of random block ids (does not need to be unique)
        ctx_ids = random.choices(range(NUM_TOTAL_BLOCKS), k=num_blocks)

        # create a random mask (num_total_blocks * block_size, block_size) using numpy
        mask = np.random.choice([0, 1], size=(BLOCK_SIZE, num_blocks * BLOCK_SIZE), p=[0.1, 0.9])

        # ensure the first block is always True (This is to ensure that the softmax is not NaN)
        mask[:, 0] = 1

        # create a full true mask
        # mask = np.ones((BLOCK_SIZE, num_blocks * BLOCK_SIZE), dtype=np.bool_)

        tasks.append((ctx_ids, mask))

    inp_baseline = construct_input_baseline(tasks, num_tok_per_blk=BLOCK_SIZE, device=device)
    inp = construct_input(tasks, num_blk_per_chunk=CHUNK_SIZE, num_tok_per_blk=BLOCK_SIZE, device=device)

    # simulate the previous state
    hidden_states = torch.randn(NUM_REQS, BLOCK_SIZE, hidden_size, device=device)

    q = q_proj(hidden_states)
    # k = k_proj(hidden_states)
    # v = v_proj(hidden_states)

    q = q.view(NUM_REQS, BLOCK_SIZE, num_heads, head_dim).transpose(1, 2)
    # k = k.view(NUM_REQS, BLOCK_SIZE, num_key_value_heads, head_dim).transpose(1, 2)
    # v = v.view(NUM_REQS, BLOCK_SIZE, num_key_value_heads, head_dim).transpose(1, 2)

    # rope(rope_cache, q, batch.position_offsets)
    # rope(rope_cache, k, batch.position_offsets)

    # attention
    y1 = qkv_attention_baseline(
        q,
        kv_cache_table,
        inp_baseline['q_lut'],
        inp_baseline['kv_lut'],
        inp_baseline['mask']
    )

    # print(y1[0])

    y2 = qkv_attention(
        q,
        kv_cache_table,
        inp['q_lut'],
        inp['kv_lut'],
        inp['masks'],
        inp['reduce_grps']
    )

    # print(y2[0])
    # print('baseline:')
    # print(y1.unsqueeze(0).unsqueeze(0))
    #
    # print('triton:')
    # print(y2.unsqueeze(0).unsqueeze(0))

    # shape
    print('y1, y2', y1.shape, y2.shape)

    print('y1, y2', torch.abs(y1 - y2).sum())

    print('done')
