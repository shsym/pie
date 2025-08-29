#!/usr/bin/env python3
"""
Create a medium bf16 test case with head_size=128 but smaller token count
"""
import json
import numpy as np
from pathlib import Path

# Create a test with head_size=128 like the original, but smaller token count
config = {
    "version": "1",
    "op": "batch_prefill_attention", 
    "case_id": "medium_bf16",
    "config": {
        "num_tokens": 32,      # Smaller than original 128
        "num_query_heads": 2,  # Smaller than original 32  
        "num_kv_heads": 2,     # Smaller than original 8
        "head_size": 128,      # Same as original
        "kv_len": 512,         # Smaller than original 2048
        "page_size": 16,       # Same as original
        "batch_size": 1,       # Same
        "num_pages": 32        # Smaller than original 128
    },
    "dtype_map": {
        "q_input": "bf16", "k_input": "bf16", "v_input": "bf16", 
        "paged_k_cache": "bf16", "paged_v_cache": "bf16", "output": "bf16",
        "qo_indptr": "s32", "kv_page_indptr": "s32", "kv_page_indices": "s32", "kv_last_page_lens": "s32"
    },
    "shape_map": {
        "q_input": [32, 256],          # 32 tokens * (2 heads * 128 head_size)
        "k_input": [512, 256],         # 512 kv_len * 256 head_dim  
        "v_input": [512, 256],         # 512 kv_len * 256 head_dim
        "paged_k_cache": [32, 16, 256], # 32 pages * 16 page_size * 256 head_dim
        "paged_v_cache": [32, 16, 256], # 32 pages * 16 page_size * 256 head_dim 
        "output": [32, 256],           # 32 tokens * 256 head_dim
        "qo_indptr": [2], "kv_page_indptr": [2], "kv_page_indices": [32], "kv_last_page_lens": [1]
    }
}

# Create test directory
test_dir = Path("/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/medium_bf16")
test_dir.mkdir(exist_ok=True)

# Write meta.json
with open(test_dir / "meta.json", "w") as f:
    json.dump(config, f, indent=2)

# Generate synthetic test data
np.random.seed(42)

def generate_bf16_data(shape):
    # Generate as f32, then convert to bf16
    f32_data = np.random.normal(0.0, 1.0, shape).astype(np.float32)
    # Convert to bf16 by truncating the lower 16 bits
    f32_as_uint32 = f32_data.view(np.uint32)
    bf16_as_uint16 = (f32_as_uint32 >> 16).astype(np.uint16)
    return bf16_as_uint16.tobytes()

# Generate input tensors
tensors = {
    "q_input": generate_bf16_data([32, 256]),
    "k_input": generate_bf16_data([512, 256]),  
    "v_input": generate_bf16_data([512, 256]),
    "paged_k_cache": generate_bf16_data([32, 16, 256]),
    "paged_v_cache": generate_bf16_data([32, 16, 256]),
    "output": np.zeros([32, 256], dtype=np.uint16).tobytes(),  # BF16 zeros
    "qo_indptr": np.array([0, 32], dtype=np.int32),
    "kv_page_indptr": np.array([0, 32], dtype=np.int32), 
    "kv_page_indices": np.arange(32, dtype=np.int32),
    "kv_last_page_lens": np.array([0], dtype=np.int32)  # Full pages
}

# Write binary files (integers need to be written as bytes)
for name, data in tensors.items():
    with open(test_dir / f"{name}.bin", "wb") as f:
        if isinstance(data, np.ndarray):
            f.write(data.tobytes())
        else:
            f.write(data)  # Already bytes for BF16 data

print(f"Created medium bf16 test case at {test_dir}")
print(f"head_dim = {config['config']['num_query_heads']} * {config['config']['head_size']} = {config['config']['num_query_heads'] * config['config']['head_size']}")
print(f"head_size = {config['config']['head_size']} (should be < MAX_HEAD_DIM=256)")