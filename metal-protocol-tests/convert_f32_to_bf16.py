#!/usr/bin/env python3
"""
Convert f32 binary files to bf16 binary files
"""
import numpy as np
from pathlib import Path

def f32_to_bf16(f32_bytes):
    """Convert f32 bytes to bf16 bytes by truncating lower 16 bits"""
    f32_data = np.frombuffer(f32_bytes, dtype=np.float32)
    
    # Convert to bf16 by truncating the lower 16 bits
    f32_as_uint32 = f32_data.view(np.uint32)
    bf16_as_uint16 = (f32_as_uint32 >> 16).astype(np.uint16)
    
    return bf16_as_uint16.tobytes()

def convert_files(source_dir, dest_dir):
    """Convert all .bin files from f32 to bf16"""
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
            print(f"Converting f32->bf16: {bin_file.name}")
            with open(bin_file, 'rb') as f:
                f32_bytes = f.read()
            
            bf16_bytes = f32_to_bf16(f32_bytes)
            
            with open(dest_path / bin_file.name, 'wb') as f:
                f.write(bf16_bytes)

if __name__ == "__main__":
    source_dir = "/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/small_f32"
    dest_dir = "/Users/seung-seoblee/Dev/pie/metal-protocol-tests/tests/artifacts/batch_prefill_attention/small_bf16"
    
    print(f"Converting files from {source_dir} to {dest_dir}")
    convert_files(source_dir, dest_dir)
    print("Conversion complete!")