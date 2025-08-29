#!/usr/bin/env python3
"""
Create a smaller f32 test case that fits within MAX_HEAD_DIM=256
"""
import json
import numpy as np
from pathlib import Path

# Create a smaller test configuration that fits within MAX_HEAD_DIM=256
config = {
    "version": "1",
    "op": "batch_prefill_attention", 
    "case_id": "small_f32",
    "config": {
        "num_tokens": 32,      # Reduced from 128
        "num_query_heads": 4,  # Reduced from 32  
        "num_kv_heads": 4,     # Reduced from 8
        "head_size": 64,       # Reduced from 128
        "kv_len": 512,         # Reduced from 2048
        "page_size": 16,       # Keep same
        "batch_size": 1,       # Keep same
        "num_pages": 32        # Reduced from 128
    },
    "dtype_map": {
        "q_input": "fp32", "k_input": "fp32", "v_input": "fp32", 
        "paged_k_cache": "fp32", "paged_v_cache": "fp32", "output": "fp32",
        "qo_indptr": "s32", "kv_page_indptr": "s32", "kv_page_indices": "s32", "kv_last_page_lens": "s32"
    },
    "shape_map": {
        "q_input": [32, 256],          # 32 tokens * (4 heads * 64 head_size)
        "k_input": [512, 256],         # 512 kv_len * 256 head_dim  
        "v_input": [512, 256],         # 512 kv_len * 256 head_dim
        "paged_k_cache": [32, 16, 256], # 32 pages * 16 page_size * 256 head_dim
        "paged_v_cache": [32, 16, 256], # 32 pages * 16 page_size * 256 head_dim 
        "output": [32, 256],           # 32 tokens * 256 head_dim
        "qo_indptr": [2], "kv_page_indptr": [2], "kv_page_indices": [32], "kv_last_page_lens": [1]
    }
}

# Create test directory
test_dir = Path("/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/small_f32")
test_dir.mkdir(exist_ok=True)

# Write meta.json
with open(test_dir / "meta.json", "w") as f:
    json.dump(config, f, indent=2)

# Generate synthetic test data
np.random.seed(42)

def generate_data(shape, dtype=np.float32):
    return np.random.normal(0.0, 1.0, shape).astype(dtype)

# Generate input tensors
tensors = {
    "q_input": generate_data([32, 256]),
    "k_input": generate_data([512, 256]),  
    "v_input": generate_data([512, 256]),
    "paged_k_cache": generate_data([32, 16, 256]),
    "paged_v_cache": generate_data([32, 16, 256]),
    "output": np.zeros([32, 256], dtype=np.float32),  # Output starts as zeros
    "qo_indptr": np.array([0, 32], dtype=np.int32),
    "kv_page_indptr": np.array([0, 32], dtype=np.int32), 
    "kv_page_indices": np.arange(32, dtype=np.int32),
    "kv_last_page_lens": np.array([0], dtype=np.int32)  # Full pages
}

# Write binary files
for name, data in tensors.items():
    with open(test_dir / f"{name}.bin", "wb") as f:
        f.write(data.tobytes())

print(f"Created small f32 test case at {test_dir}")
print(f"head_dim = {config['config']['num_query_heads']} * {config['config']['head_size']} = {config['config']['num_query_heads'] * config['config']['head_size']}")
print("This should fit within MAX_HEAD_DIM=256")