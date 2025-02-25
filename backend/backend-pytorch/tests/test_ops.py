import sys
import os

# Add the parent directory to sys.path for importing
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from ops import *

def test_rope():
    head_dim = 64
    batch_size = 100
    device = 'cuda'

    k = torch.randn((batch_size, 32, 8, head_dim), device=device)
    position_offsets = torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device)
    rope_cache = create_rope_cache(8192, head_dim, torch.float32, device)

    k_pos1 = rope_baseline(rope_cache, k, position_offsets)
    k_pos2 = rope_baseline_no_cache(k, position_offsets)
    k_pos_triton = rope(rope_cache, k, position_offsets)

    print('kpos1, kpos2', torch.abs(k_pos1 - k_pos2).sum())
    assert torch.allclose(k_pos2, k_pos1, atol=1e-3)

    print('kpos1, kpostriton', torch.abs(k_pos1 - k_pos_triton).sum())
    assert torch.allclose(k_pos1, k_pos_triton, atol=1e-3)
