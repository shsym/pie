#!/usr/bin/env python3
"""
Simple Metal Kernel Test

Tests basic Metal kernel functionality with minimal dependencies.
"""

import sys
import os
import numpy as np

# Test without Metal bindings first - CPU reference implementations
print("🧮 Testing CPU Reference Implementations")
print("=" * 50)

def cpu_softmax(x):
    """CPU reference softmax implementation."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cpu_rmsnorm(x, eps=1e-6):
    """CPU reference RMS normalization."""
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    rms_scale = 1.0 / np.sqrt(mean_square + eps)
    return x * rms_scale

def test_softmax_cpu():
    """Test softmax sum=1.0 correctness with CPU implementation."""
    print("\n✅ Testing CPU Softmax...")

    test_cases = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([10.0, 1.0, 0.1], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.random.randn(100).astype(np.float32)
    ]

    all_passed = True
    for i, test_input in enumerate(test_cases):
        output = cpu_softmax(test_input)
        output_sum = np.sum(output)
        sum_error = abs(output_sum - 1.0)
        has_nan = np.any(np.isnan(output))
        has_inf = np.any(np.isinf(output))

        passed = sum_error < 1e-6 and not has_nan and not has_inf
        all_passed &= passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Case {i+1}: {status} (sum={output_sum:.8f}, error={sum_error:.2e})")

    return all_passed

def test_rmsnorm_cpu():
    """Test RMS norm no-NaN outputs with CPU implementation."""
    print("\n✅ Testing CPU RMS Norm...")

    test_cases = [
        np.random.randn(4, 8).astype(np.float32),
        np.random.randn(4, 8).astype(np.float32) * 1e-6,
        np.random.randn(4, 8).astype(np.float32) * 100.0,
        np.zeros((4, 8), dtype=np.float32)
    ]

    all_passed = True
    for i, test_input in enumerate(test_cases):
        output = cpu_rmsnorm(test_input, eps=1e-6)
        has_nan = np.any(np.isnan(output))
        has_inf = np.any(np.isinf(output))
        shape_correct = output.shape == test_input.shape

        passed = not has_nan and not has_inf and shape_correct
        all_passed &= passed

        status = "✅ PASS" if passed else "❌ FAIL"
        issues = []
        if has_nan: issues.append("NaN")
        if has_inf: issues.append("Inf")
        if not shape_correct: issues.append("Shape")
        issue_str = f" ({', '.join(issues)})" if issues else ""

        print(f"  Case {i+1}: {status}{issue_str}")

    return all_passed

def test_basic_operations():
    """Test basic tensor operations and shapes."""
    print("\n✅ Testing Basic Operations...")

    # Test embedding lookup
    vocab_size, embed_dim = 10, 8
    embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    indices = np.array([0, 2, 5], dtype=np.int32)

    # Simple embedding lookup
    embedding_output = embedding_table[indices]
    embedding_passed = embedding_output.shape == (3, embed_dim)

    print(f"  Embedding lookup: {'✅ PASS' if embedding_passed else '❌ FAIL'} "
          f"(shape: {embedding_output.shape})")

    # Test MLP (ReLU activation)
    mlp_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    mlp_output = np.maximum(0, mlp_input)  # ReLU
    expected_mlp = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    mlp_passed = np.allclose(mlp_output, expected_mlp)

    print(f"  MLP (ReLU): {'✅ PASS' if mlp_passed else '❌ FAIL'}")

    return embedding_passed and mlp_passed

def test_attention_shapes():
    """Test attention operation shapes."""
    print("\n✅ Testing Attention Shapes...")

    # Simple dot-product attention
    num_tokens, num_heads, head_size = 4, 2, 8
    kv_len = 16

    q = np.random.randn(num_tokens, num_heads * head_size).astype(np.float32)
    k = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)
    v = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)

    # Simple attention computation
    scores = np.dot(q, k.T) / np.sqrt(num_heads * head_size)  # (num_tokens, kv_len)

    # Apply softmax to each row
    attn_weights = np.array([cpu_softmax(row) for row in scores])

    # Compute output
    output = np.dot(attn_weights, v)  # (num_tokens, num_heads * head_size)

    # Verify shapes and properties
    shape_correct = output.shape == q.shape
    no_nan = not np.any(np.isnan(output))
    no_inf = not np.any(np.isinf(output))

    # Verify attention weights sum to 1
    weights_sum_correct = np.allclose(np.sum(attn_weights, axis=1), 1.0)

    passed = shape_correct and no_nan and no_inf and weights_sum_correct

    print(f"  Attention: {'✅ PASS' if passed else '❌ FAIL'} "
          f"(shape: {output.shape}, weights_sum: {np.sum(attn_weights, axis=1)[:2]})")

    return passed

def main():
    """Run all CPU reference tests."""
    print("🚀 Metal Kernel Validation - CPU Reference Tests")
    print("This validates the expected numerical properties that Metal kernels should match.")

    start_time = __import__('time').time()

    # Run tests
    test_results = {
        "softmax": test_softmax_cpu(),
        "rmsnorm": test_rmsnorm_cpu(),
        "basic_ops": test_basic_operations(),
        "attention": test_attention_shapes()
    }

    end_time = __import__('time').time()

    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)

    passed_count = sum(test_results.values())
    total_count = len(test_results)

    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\n📊 Results: {passed_count}/{total_count} tests passed")
    print(f"⏱️ Runtime: {end_time - start_time:.2f}s")

    if passed_count == total_count:
        print("\n🎉 All CPU reference tests passed!")
        print("Metal kernels should match these numerical properties.")
    else:
        print(f"\n⚠️ {total_count - passed_count} test(s) failed.")

    return passed_count == total_count

if __name__ == "__main__":
    # Import numpy with proper error handling
    try:
        import numpy as np
        success = main()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Please install numpy: pip install numpy")
        sys.exit(1)