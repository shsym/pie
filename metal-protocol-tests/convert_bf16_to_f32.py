#!/usr/bin/env python3
"""
Convert bf16 binary files to f32 binary files for testing
"""
import os
import struct
import numpy as np
from pathlib import Path

def bf16_to_f32(bf16_bytes):
    """Convert bf16 bytes to f32 bytes"""
    # bf16 is 16-bit, f32 is 32-bit
    bf16_data = np.frombuffer(bf16_bytes, dtype=np.uint16)
    
    # Convert bf16 to f32 by padding with zeros in the lower 16 bits
    f32_data = np.zeros(len(bf16_data), dtype=np.uint32)
    f32_data = (bf16_data.astype(np.uint32) << 16)
    
    # Convert to actual float32
    f32_floats = f32_data.view(np.float32)
    return f32_floats.tobytes()

def convert_files(source_dir, dest_dir):
    """Convert all .bin files from bf16 to f32"""
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Skip integer files (indices and pointers)
    skip_files = {'qo_indptr.bin', 'kv_page_indptr.bin', 'kv_page_indices.bin', 'kv_last_page_lens.bin'}
    
    for bin_file in source_path.glob('*.bin'):
        if bin_file.name in skip_files:
            print(f"Copying (unchanged): {bin_file.name}")
            # Just copy integer files unchanged
            with open(bin_file, 'rb') as src, open(dest_path / bin_file.name, 'wb') as dst:
                dst.write(src.read())
        else:
            print(f"Converting bf16->f32: {bin_file.name}")
            with open(bin_file, 'rb') as f:
                bf16_bytes = f.read()
            
            f32_bytes = bf16_to_f32(bf16_bytes)
            
            with open(dest_path / bin_file.name, 'wb') as f:
                f.write(f32_bytes)

if __name__ == "__main__":
    source_dir = "/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/final_test"
    dest_dir = "/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/test_f32"
    
    print(f"Converting files from {source_dir} to {dest_dir}")
    convert_files(source_dir, dest_dir)
    print("Conversion complete!")