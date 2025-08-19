#!/usr/bin/env python3

import numpy as np
import struct

def bfloat16_to_float32(bfloat16_bytes):
    """Convert bfloat16 bytes to float32"""
    results = []
    for i in range(0, len(bfloat16_bytes), 2):
        # bfloat16 is stored as uint16, extend to float32 by shifting left 16 bits
        bfloat16_uint16 = struct.unpack('<H', bfloat16_bytes[i:i+2])[0]
        float32_bits = bfloat16_uint16 << 16
        float32_val = struct.unpack('<f', struct.pack('<I', float32_bits))[0]
        results.append(float32_val)
    return np.array(results)

def float32_to_bfloat16_bytes(float32_array):
    """Convert float32 array to bfloat16 bytes"""
    result = bytearray()
    for val in float32_array:
        float32_bits = struct.unpack('<I', struct.pack('<f', val))[0]
        # Truncate to bfloat16 by taking upper 16 bits
        bfloat16_bits = (float32_bits + 0x8000) >> 16  # Round to nearest
        result.extend(struct.pack('<H', bfloat16_bits))
    return bytes(result)

def rmsnorm_reference(input_tensor, weight, eps=1e-5):
    """Reference implementation of RMS normalization"""
    # input_tensor shape: [num_tokens, hidden_size]
    # weight shape: [hidden_size]
    
    # Compute RMS (Root Mean Square) for each token
    input_squared = input_tensor ** 2
    mean_squared = np.mean(input_squared, axis=1, keepdims=True)  # [num_tokens, 1]
    rms = np.sqrt(mean_squared + eps)
    
    # Normalize and apply weight
    normalized = input_tensor / rms  # Broadcasting: [num_tokens, hidden_size] / [num_tokens, 1]
    output = normalized * weight[None, :]  # Broadcasting: [num_tokens, hidden_size] * [1, hidden_size]
    
    return output

def test_rmsnorm():
    """Test RMS normalization with known values"""
    print("=== RMS Norm Validation ===")
    
    # Test parameters
    num_tokens = 8
    hidden_size = 16
    eps = 1e-5
    
    # Generate test data (same pattern as Metal test)
    np.random.seed(42)  # For reproducibility
    
    # Create input tensor: values from -0.5 to 0.49 like in the Metal test
    input_data = []
    for i in range(num_tokens * hidden_size):
        val = (i % 100) / 100.0 - 0.5  # Range: -0.5 to 0.49
        input_data.append(val)
    
    input_tensor = np.array(input_data, dtype=np.float32).reshape(num_tokens, hidden_size)
    
    # Create weight tensor: values around 1.0 like in the Metal test
    weight_data = []
    for i in range(hidden_size):
        val = 1.0 + (i % 10) / 100.0  # Range: 1.0 to 1.09
        weight_data.append(val)
    
    weight = np.array(weight_data, dtype=np.float32)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Input sample: {input_tensor[0, :5]}")
    print(f"Weight sample: {weight[:5]}")
    
    # Compute reference RMS norm
    output_ref = rmsnorm_reference(input_tensor, weight, eps)
    
    print(f"Reference output sample: {output_ref[0, :5]}")
    
    # The Metal test reported: -1.17188 -1.15625 -1.14844 -1.13281 -1.11719
    # Let's see how close we are
    metal_output_sample = np.array([-1.17188, -1.15625, -1.14844, -1.13281, -1.11719])
    
    print(f"Metal output sample:     {metal_output_sample}")
    print(f"Difference:              {output_ref[0, :5] - metal_output_sample}")
    print(f"Max difference:          {np.max(np.abs(output_ref[0, :5] - metal_output_sample))}")
    
    # Check if they're close (within bfloat16 precision)
    bfloat16_tolerance = 1e-2  # bfloat16 has ~3 decimal digits of precision
    if np.allclose(output_ref[0, :5], metal_output_sample, atol=bfloat16_tolerance):
        print("✅ Metal RMS norm output matches reference implementation!")
    else:
        print("❌ Metal RMS norm output differs from reference")
    
    # Test the mathematical properties
    print("\n=== Mathematical Properties ===")
    
    # Property 1: Output should have specific RMS
    for token_idx in range(min(3, num_tokens)):
        token_output = output_ref[token_idx, :]
        token_input = input_tensor[token_idx, :]
        
        # Compute RMS of input
        input_rms = np.sqrt(np.mean(token_input ** 2) + eps)
        
        # The normalized input should have RMS ≈ 1.0
        normalized = token_input / input_rms
        normalized_rms = np.sqrt(np.mean(normalized ** 2))
        
        print(f"Token {token_idx}: Input RMS = {input_rms:.6f}, Normalized RMS = {normalized_rms:.6f}")
    
    print("\n✅ RMS Norm validation completed!")

if __name__ == "__main__":
    test_rmsnorm()