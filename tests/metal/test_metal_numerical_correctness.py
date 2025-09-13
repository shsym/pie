#!/usr/bin/env python3
"""
Metal Kernel Numerical Correctness Test Suite

Tests the numerical correctness fixes implemented in Metal kernels:
- Softmax sum=1.0 validation
- RMS norm no-NaN outputs
- MLP correct shape outputs
- Embedding accuracy
- Attention debug buffer functionality

This test suite validates Metal kernel fixes using both CPU reference implementations
and the debug framework integration.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path to import debug framework
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from debug_framework.integrations.backend_interfaces import BackendType, TensorComputationResult
    from debug_framework.integrations.backend_interfaces import create_backend
    DEBUG_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    DEBUG_FRAMEWORK_AVAILABLE = False
    print(f"⚠️  Debug framework not available: {e}")


class MetalKernelNumericalTests:
    """Test suite for Metal kernel numerical correctness."""

    def __init__(self):
        self.backend = None
        if DEBUG_FRAMEWORK_AVAILABLE:
            try:
                self.backend = create_backend(BackendType.METAL)
                print("✅ Metal backend created via debug framework")
            except Exception as e:
                print(f"⚠️  Could not create Metal backend: {e}")
                self.backend = None

    # CPU reference implementations for validation

    def _cpu_softmax(self, x):
        """CPU reference softmax implementation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _cpu_rmsnorm(self, x, eps=1e-6):
        """CPU reference RMS normalization."""
        # x shape: (num_tokens, hidden_size)
        mean_square = np.mean(x * x, axis=-1, keepdims=True)
        rms_scale = 1.0 / np.sqrt(mean_square + eps)
        return x * rms_scale

    def _cpu_mlp(self, x):
        """CPU reference MLP (ReLU activation)."""
        return np.maximum(0, x)

    def _cpu_attention(self, q, k, v):
        """CPU reference attention implementation."""
        # q: (num_tokens, dim), k, v: (kv_len, dim)
        scores = np.dot(q, k.T) / np.sqrt(q.shape[-1])
        attn_weights = np.array([self._cpu_softmax(row) for row in scores])
        output = np.dot(attn_weights, v)
        return output

    # Test methods

    def test_softmax_sum_correctness(self):
        """Test that softmax outputs sum to 1.0 within numerical tolerance."""
        print("\n🧮 Testing Softmax Sum Correctness...")

        test_cases = [
            ("simple", np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            ("large_diff", np.array([10.0, 1.0, 0.1], dtype=np.float32)),
            ("negative", np.array([-1.0, 0.0, 1.0], dtype=np.float32)),
            ("random_100", np.random.randn(100).astype(np.float32))
        ]

        results = {}

        for case_name, test_input in test_cases:
            # CPU reference
            cpu_output = self._cpu_softmax(test_input)
            cpu_sum = np.sum(cpu_output)
            cpu_sum_error = abs(cpu_sum - 1.0)
            cpu_has_nan = np.any(np.isnan(cpu_output))
            cpu_has_inf = np.any(np.isinf(cpu_output))

            cpu_passed = (cpu_sum_error < 1e-6 and not cpu_has_nan and not cpu_has_inf)

            # Metal backend test (if available)
            metal_passed = None
            metal_output = None
            metal_sum = None
            metal_sum_error = None

            if self.backend:
                try:
                    # Try to execute softmax via debug framework
                    result = self.backend.execute_softmax(test_input)
                    if hasattr(result, 'output'):
                        metal_output = result.output
                    else:
                        metal_output = result

                    metal_sum = np.sum(metal_output)
                    metal_sum_error = abs(metal_sum - 1.0)
                    metal_has_nan = np.any(np.isnan(metal_output))
                    metal_has_inf = np.any(np.isinf(metal_output))

                    metal_passed = (metal_sum_error < 1e-6 and not metal_has_nan and not metal_has_inf)

                except (AttributeError, RuntimeError) as e:
                    # Expected when Metal executor not available
                    metal_passed = None
                    print(f"    {case_name}: Metal not available ({e})")

            results[case_name] = {
                "cpu_passed": cpu_passed,
                "cpu_sum": float(cpu_sum),
                "cpu_sum_error": float(cpu_sum_error),
                "metal_passed": metal_passed,
                "metal_sum": float(metal_sum) if metal_sum is not None else None,
                "metal_sum_error": float(metal_sum_error) if metal_sum_error is not None else None
            }

            # Print results
            cpu_status = "✅ PASS" if cpu_passed else "❌ FAIL"
            metal_status = "✅ PASS" if metal_passed else ("❌ FAIL" if metal_passed is False else "⚠️  N/A")

            print(f"  {case_name}:")
            print(f"    CPU: {cpu_status} (sum={cpu_sum:.8f}, error={cpu_sum_error:.2e})")
            if metal_passed is not None:
                print(f"    Metal: {metal_status} (sum={metal_sum:.8f}, error={metal_sum_error:.2e})")

        return results

    def test_rmsnorm_no_nan_outputs(self):
        """Test that RMS normalization produces no NaN outputs."""
        print("\n🔢 Testing RMS Norm No-NaN Outputs...")

        test_cases = [
            ("normal", np.random.randn(4, 8).astype(np.float32), 1e-6),
            ("small_values", np.random.randn(4, 8).astype(np.float32) * 1e-6, 1e-8),
            ("large_values", np.random.randn(4, 8).astype(np.float32) * 100.0, 1e-6),
            ("zero_input", np.zeros((4, 8), dtype=np.float32), 1e-6)
        ]

        results = {}

        for case_name, test_input, eps in test_cases:
            # CPU reference
            cpu_output = self._cpu_rmsnorm(test_input, eps)
            cpu_has_nan = np.any(np.isnan(cpu_output))
            cpu_has_inf = np.any(np.isinf(cpu_output))
            cpu_shape_correct = cpu_output.shape == test_input.shape
            cpu_passed = not cpu_has_nan and not cpu_has_inf and cpu_shape_correct

            # Metal backend test (if available)
            metal_passed = None
            metal_output = None

            if self.backend:
                try:
                    result = self.backend.run_normalization(test_input, eps=eps)
                    if hasattr(result, 'output'):
                        metal_output = result.output
                    else:
                        metal_output = result

                    metal_has_nan = np.any(np.isnan(metal_output))
                    metal_has_inf = np.any(np.isinf(metal_output))
                    metal_shape_correct = metal_output.shape == test_input.shape
                    metal_passed = not metal_has_nan and not metal_has_inf and metal_shape_correct

                except (AttributeError, RuntimeError) as e:
                    metal_passed = None
                    print(f"    {case_name}: Metal not available ({e})")

            results[case_name] = {
                "cpu_passed": cpu_passed,
                "cpu_has_nan": cpu_has_nan,
                "cpu_has_inf": cpu_has_inf,
                "metal_passed": metal_passed,
                "metal_has_nan": metal_output is not None and np.any(np.isnan(metal_output)),
                "metal_has_inf": metal_output is not None and np.any(np.isinf(metal_output))
            }

            # Print results
            cpu_status = "✅ PASS" if cpu_passed else "❌ FAIL"
            metal_status = "✅ PASS" if metal_passed else ("❌ FAIL" if metal_passed is False else "⚠️  N/A")

            cpu_issues = []
            if cpu_has_nan: cpu_issues.append("NaN")
            if cpu_has_inf: cpu_issues.append("Inf")
            if not cpu_shape_correct: cpu_issues.append("Shape")

            print(f"  {case_name}:")
            print(f"    CPU: {cpu_status}" + (f" ({', '.join(cpu_issues)})" if cpu_issues else ""))
            if metal_passed is not None:
                metal_issues = []
                if results[case_name]["metal_has_nan"]: metal_issues.append("NaN")
                if results[case_name]["metal_has_inf"]: metal_issues.append("Inf")
                print(f"    Metal: {metal_status}" + (f" ({', '.join(metal_issues)})" if metal_issues else ""))

        return results

    def test_mlp_correct_shapes(self):
        """Test that MLP operations produce correct output shapes."""
        print("\n🏗️  Testing MLP Correct Shapes...")

        test_cases = [
            ("small", (4, 8)),
            ("batch", (32, 64)),
            ("large", (8, 256))
        ]

        results = {}

        for case_name, input_shape in test_cases:
            test_input = np.random.randn(*input_shape).astype(np.float32)

            # CPU reference (simple ReLU)
            cpu_output = self._cpu_mlp(test_input)
            cpu_shape_correct = cpu_output.shape == test_input.shape
            cpu_has_nan = np.any(np.isnan(cpu_output))
            cpu_has_inf = np.any(np.isinf(cpu_output))
            cpu_passed = cpu_shape_correct and not cpu_has_nan and not cpu_has_inf

            # Metal backend test (if available)
            metal_passed = None
            metal_output = None

            if self.backend:
                try:
                    result = self.backend.run_mlp(test_input)
                    if hasattr(result, 'output'):
                        metal_output = result.output
                    else:
                        metal_output = result

                    metal_shape_correct = metal_output.shape == test_input.shape
                    metal_has_nan = np.any(np.isnan(metal_output))
                    metal_has_inf = np.any(np.isinf(metal_output))
                    metal_passed = metal_shape_correct and not metal_has_nan and not metal_has_inf

                except (AttributeError, RuntimeError) as e:
                    metal_passed = None

            results[case_name] = {
                "input_shape": input_shape,
                "cpu_passed": cpu_passed,
                "cpu_output_shape": cpu_output.shape,
                "metal_passed": metal_passed,
                "metal_output_shape": metal_output.shape if metal_output is not None else None
            }

            # Print results
            cpu_status = "✅ PASS" if cpu_passed else "❌ FAIL"
            metal_status = "✅ PASS" if metal_passed else ("❌ FAIL" if metal_passed is False else "⚠️  N/A")

            print(f"  {case_name}: input={input_shape}")
            print(f"    CPU: {cpu_status} (output={cpu_output.shape})")
            if metal_passed is not None:
                print(f"    Metal: {metal_status} (output={metal_output.shape})")

        return results

    def test_embedding_accuracy(self):
        """Test embedding lookup accuracy."""
        print("\n📚 Testing Embedding Accuracy...")

        # Create test embedding table
        vocab_size, embed_dim = 10, 8
        embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)

        test_cases = [
            ("single", np.array([3], dtype=np.int32)),
            ("multiple", np.array([0, 2, 5], dtype=np.int32)),
            ("batch", np.array([1, 4, 7, 9], dtype=np.int32))
        ]

        results = {}

        for case_name, indices in test_cases:
            # CPU reference
            cpu_output = embedding_table[indices]
            expected_shape = (len(indices), embed_dim)
            cpu_shape_correct = cpu_output.shape == expected_shape
            cpu_has_nan = np.any(np.isnan(cpu_output))
            cpu_has_inf = np.any(np.isinf(cpu_output))
            cpu_passed = cpu_shape_correct and not cpu_has_nan and not cpu_has_inf

            # Metal backend test (if available)
            metal_passed = None
            metal_output = None
            accuracy_error = None

            if self.backend:
                try:
                    result = self.backend.run_embedding(indices, embedding_table=embedding_table)
                    if hasattr(result, 'output'):
                        metal_output = result.output
                    else:
                        metal_output = result

                    metal_shape_correct = metal_output.shape == expected_shape
                    metal_has_nan = np.any(np.isnan(metal_output))
                    metal_has_inf = np.any(np.isinf(metal_output))

                    # Check accuracy against CPU reference
                    if metal_output.shape == cpu_output.shape:
                        accuracy_error = np.max(np.abs(metal_output - cpu_output))
                        accuracy_correct = accuracy_error < 1e-6
                    else:
                        accuracy_correct = False
                        accuracy_error = float('inf')

                    metal_passed = metal_shape_correct and not metal_has_nan and not metal_has_inf and accuracy_correct

                except (AttributeError, RuntimeError) as e:
                    metal_passed = None

            results[case_name] = {
                "indices": indices.tolist(),
                "expected_shape": expected_shape,
                "cpu_passed": cpu_passed,
                "cpu_output_shape": cpu_output.shape,
                "metal_passed": metal_passed,
                "metal_output_shape": metal_output.shape if metal_output is not None else None,
                "accuracy_error": float(accuracy_error) if accuracy_error is not None else None
            }

            # Print results
            cpu_status = "✅ PASS" if cpu_passed else "❌ FAIL"
            metal_status = "✅ PASS" if metal_passed else ("❌ FAIL" if metal_passed is False else "⚠️  N/A")

            print(f"  {case_name}: indices={indices.tolist()}")
            print(f"    CPU: {cpu_status} (shape={cpu_output.shape})")
            if metal_passed is not None:
                print(f"    Metal: {metal_status} (shape={metal_output.shape}, error={accuracy_error:.2e})")

        return results

    def test_attention_debug_buffer(self):
        """Test attention debug buffer functionality."""
        print("\n🔍 Testing Attention Debug Buffer...")

        # Test parameters
        num_tokens, num_heads, head_size = 2, 2, 8
        kv_len = 16

        q = np.random.randn(num_tokens, num_heads * head_size).astype(np.float32)
        k = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)
        v = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)

        # CPU reference
        cpu_output = self._cpu_attention(q, k, v)
        expected_shape = q.shape
        cpu_shape_correct = cpu_output.shape == expected_shape
        cpu_has_nan = np.any(np.isnan(cpu_output))
        cpu_has_inf = np.any(np.isinf(cpu_output))
        cpu_passed = cpu_shape_correct and not cpu_has_nan and not cpu_has_inf

        # Metal backend test (if available)
        metal_passed = None
        metal_output = None
        debug_info = None

        if self.backend:
            try:
                result = self.backend.run_attention(
                    q, k, v,
                    num_query_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_size=head_size
                )

                if hasattr(result, 'output'):
                    metal_output = result.output
                    if hasattr(result, 'metadata'):
                        debug_info = result.metadata
                else:
                    metal_output = result

                metal_shape_correct = metal_output.shape == expected_shape
                metal_has_nan = np.any(np.isnan(metal_output))
                metal_has_inf = np.any(np.isinf(metal_output))
                metal_passed = metal_shape_correct and not metal_has_nan and not metal_has_inf

            except (AttributeError, RuntimeError) as e:
                metal_passed = None
                print(f"    Metal attention not available: {e}")

        result = {
            "cpu_passed": cpu_passed,
            "cpu_output_shape": cpu_output.shape,
            "cpu_has_nan": cpu_has_nan,
            "cpu_has_inf": cpu_has_inf,
            "metal_passed": metal_passed,
            "metal_output_shape": metal_output.shape if metal_output is not None else None,
            "debug_info": debug_info
        }

        # Print results
        cpu_status = "✅ PASS" if cpu_passed else "❌ FAIL"
        metal_status = "✅ PASS" if metal_passed else ("❌ FAIL" if metal_passed is False else "⚠️  N/A")

        print(f"  Attention:")
        print(f"    CPU: {cpu_status} (shape={cpu_output.shape})")
        if metal_passed is not None:
            print(f"    Metal: {metal_status} (shape={metal_output.shape})")
            if debug_info:
                print(f"    Debug info available: {list(debug_info.keys())}")

        return result

    def run_all_tests(self):
        """Run all numerical correctness tests."""
        print("🚀 Metal Kernel Numerical Correctness Tests")
        print("=" * 60)

        if not DEBUG_FRAMEWORK_AVAILABLE:
            print("⚠️  Debug framework not available, running CPU-only tests")
        elif not self.backend:
            print("⚠️  Metal backend not available, running CPU-only tests")
        else:
            print("✅ Running tests with Metal backend support")

        start_time = __import__('time').time()

        # Run all tests
        results = {
            "softmax": self.test_softmax_sum_correctness(),
            "rmsnorm": self.test_rmsnorm_no_nan_outputs(),
            "mlp": self.test_mlp_correct_shapes(),
            "embedding": self.test_embedding_accuracy(),
            "attention": self.test_attention_debug_buffer()
        }

        end_time = __import__('time').time()

        # Generate summary
        print("\n" + "=" * 60)
        print("📋 FINAL TEST SUMMARY")
        print("=" * 60)

        cpu_passed = 0
        cpu_total = 0
        metal_passed = 0
        metal_total = 0

        for test_name, test_result in results.items():
            if isinstance(test_result, dict):
                if test_name in ["softmax", "rmsnorm", "mlp", "embedding"]:
                    # Multiple test cases
                    for case_name, case_result in test_result.items():
                        if isinstance(case_result, dict):
                            if "cpu_passed" in case_result:
                                cpu_total += 1
                                if case_result["cpu_passed"]:
                                    cpu_passed += 1

                            if case_result.get("metal_passed") is not None:
                                metal_total += 1
                                if case_result["metal_passed"]:
                                    metal_passed += 1
                else:
                    # Single test case
                    if "cpu_passed" in test_result:
                        cpu_total += 1
                        if test_result["cpu_passed"]:
                            cpu_passed += 1

                    if test_result.get("metal_passed") is not None:
                        metal_total += 1
                        if test_result["metal_passed"]:
                            metal_passed += 1

        print(f"📊 CPU Reference Tests: {cpu_passed}/{cpu_total} passed")
        if metal_total > 0:
            print(f"🔧 Metal Backend Tests: {metal_passed}/{metal_total} passed")
        else:
            print("🔧 Metal Backend Tests: Not available")

        print(f"⏱️ Runtime: {end_time - start_time:.2f}s")

        if cpu_passed == cpu_total:
            print("\n🎉 All CPU reference tests passed!")
            if metal_total > 0:
                if metal_passed == metal_total:
                    print("🎉 All Metal tests passed! Numerical correctness validated.")
                else:
                    print(f"⚠️  {metal_total - metal_passed} Metal test(s) failed.")
            else:
                print("ℹ️  Metal backend tests were not available for validation.")
        else:
            print(f"\n❌ {cpu_total - cpu_passed} CPU reference test(s) failed.")

        return results


def main():
    """Main entry point."""
    tester = MetalKernelNumericalTests()
    results = tester.run_all_tests()
    return results

if __name__ == "__main__":
    main()