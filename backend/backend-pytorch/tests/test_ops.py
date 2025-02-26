import sys
import os
import time

# Add the parent directory to sys.path for importing
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from ops import *

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
