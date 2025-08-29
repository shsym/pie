#!/usr/bin/env python3
"""
Debug script to understand the CUDA extract_k_values pattern
"""

# Parse the hex dump manually since we don't have numpy
def parse_hex_line(line):
    """Parse a hexdump line to extract int32 values"""
    parts = line.split('|')[0].strip().split()
    if len(parts) < 2:
        return []
    
    # Skip the offset (first part)
    hex_bytes = parts[1:]
    
    # Group into 4-byte int32 values (little endian)
    values = []
    for i in range(0, len(hex_bytes), 4):
        if i + 3 < len(hex_bytes):
            # Little endian: LSB first
            val = (int(hex_bytes[i+3], 16) << 24) | \
                  (int(hex_bytes[i+2], 16) << 16) | \
                  (int(hex_bytes[i+1], 16) << 8) | \
                   int(hex_bytes[i], 16)
            values.append(val)
    return values

# CUDA indices from hexdump
cuda_lines = [
    "00000000  00 00 00 00 11 00 00 00  44 00 00 00 55 00 00 00",
    "00000010  22 00 00 00 33 00 00 00  66 00 00 00 77 00 00 00", 
    "00000020  88 00 00 00 99 00 00 00  cc 00 00 00 dd 00 00 00",
    "00000030  aa 00 00 00 bb 00 00 00  ee 00 00 00 ff 00 00 00",
    "00000040  43 01 00 00 54 01 00 00  10 01 00 00 21 01 00 00"
]

# Metal indices from hexdump  
metal_lines = [
    "00000000  00 00 00 00 11 00 00 00  22 00 00 00 33 00 00 00",
    "00000010  44 00 00 00 55 00 00 00  66 00 00 00 77 00 00 00",
    "00000020  88 00 00 00 99 00 00 00  aa 00 00 00 bb 00 00 00", 
    "00000030  cc 00 00 00 dd 00 00 00  ee 00 00 00 ff 00 00 00",
    "00000040  10 01 00 00 21 01 00 00  32 01 00 00 43 01 00 00"
]

print("CUDA extraction pattern (first row, k=50):")
cuda_indices = []
for line in cuda_lines:
    cuda_indices.extend(parse_hex_line(line))

print("CUDA first 20 indices:", cuda_indices[:20])

print("\nMetal extraction pattern (first row, k=50):")  
metal_indices = []
for line in metal_lines:
    metal_indices.extend(parse_hex_line(line))

print("Metal first 20 indices:", metal_indices[:20])

# Analyze pattern
print(f"\nCUDA indices: {cuda_indices[:10]}")
print(f"Metal indices: {metal_indices[:10]}")

# Check if CUDA follows the hash pattern from the generation
print("\nPredicted pattern based on CUDA generation code:")
print("for (int j = 0; j < k; ++j) {")
print("    int col = (m * 131 + j * 17) % N;  // N = 4096")
print("}")

predicted = []
for j in range(20):  # First 20 values for row m=0
    col = (0 * 131 + j * 17) % 4096
    predicted.append(col)

print(f"Predicted indices: {predicted}")
print(f"CUDA matches predicted: {cuda_indices[:20] == predicted}")
print(f"Metal matches predicted: {metal_indices[:20] == predicted}")