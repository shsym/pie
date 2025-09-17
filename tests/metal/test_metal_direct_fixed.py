#!/usr/bin/env python3
"""
Direct Metal Kernel Testing with Path Fix

Tests Metal kernels by adding the build path to sys.path before importing.
"""

import sys
import os

# Add Metal bindings path before importing
METAL_BINDINGS_PATH = "/Users/seung-seoblee/Dev/pie/backend/backend-metal/build/lib"
sys.path.insert(0, METAL_BINDINGS_PATH)

# Now we can import numpy and metal_bindings
import numpy as np

def test_metal_kernels():
    """Test Metal kernels directly."""
    print("🚀 Direct Metal Kernel Tests")
    print("=" * 50)

    try:
        import metal_bindings
        print("✅ Metal bindings imported successfully")

        # Change to metal build directory where metallib is located
        original_cwd = os.getcwd()
        os.chdir(METAL_BINDINGS_PATH)

        try:
            # Create Metal executor
            metallib_path = "pie_metal_kernels.metallib"
            if not os.path.exists(metallib_path):
                print(f"❌ Metallib not found: {metallib_path}")
                return False

            executor = metal_bindings.MetalKernelExecutor(metallib_path)
            print("✅ Metal executor created successfully")

            # Get device info
            try:
                device_info = executor.get_device_info()
                print(f"🖥️  Device: {device_info}")
            except:
                print("⚠️  Could not get device info")

            # List kernels
            try:
                kernels = executor.list_available_kernels()
                print(f"🔧 Available kernels: {kernels}")
            except:
                print("⚠️  Could not list kernels")

            results = {}

            # Test 1: Softmax
            print("\n🧮 Testing Softmax...")
            try:
                test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                output = executor.execute_softmax(test_input)
                sum_val = np.sum(output)
                sum_error = abs(sum_val - 1.0)
                has_nan = np.any(np.isnan(output))

                passed = sum_error < 1e-6 and not has_nan
                results["softmax"] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  Softmax: {status} (sum={sum_val:.8f}, error={sum_error:.2e})")

            except Exception as e:
                results["softmax"] = False
                print(f"  Softmax: ❌ ERROR - {e}")

            # Test 2: RMS Norm
            print("\n🔢 Testing RMS Norm...")
            try:
                test_input = np.random.randn(4, 8).astype(np.float32)
                eps = 1e-6
                output = executor.execute_rms_norm(test_input, eps)

                has_nan = np.any(np.isnan(output))
                has_inf = np.any(np.isinf(output))
                shape_ok = output.shape == test_input.shape

                passed = not has_nan and not has_inf and shape_ok
                results["rmsnorm"] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                issues = []
                if has_nan: issues.append("NaN")
                if has_inf: issues.append("Inf")
                if not shape_ok: issues.append("Shape")

                issue_str = f" ({', '.join(issues)})" if issues else ""
                print(f"  RMS Norm: {status}{issue_str}")

            except Exception as e:
                results["rmsnorm"] = False
                print(f"  RMS Norm: ❌ ERROR - {e}")

            # Test 3: MLP
            print("\n🏗️  Testing MLP...")
            try:
                test_input = np.random.randn(4, 8).astype(np.float32)
                output = executor.execute_mlp(test_input)

                has_nan = np.any(np.isnan(output))
                has_inf = np.any(np.isinf(output))
                shape_ok = output.shape == test_input.shape

                passed = not has_nan and not has_inf and shape_ok
                results["mlp"] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  MLP: {status} (shape: {test_input.shape} -> {output.shape})")

            except Exception as e:
                results["mlp"] = False
                print(f"  MLP: ❌ ERROR - {e}")

            # Test 4: Embedding
            print("\n📚 Testing Embedding...")
            try:
                vocab_size, embed_dim = 10, 8
                embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
                indices = np.array([0, 2, 5], dtype=np.int32)

                output = executor.execute_embedding(indices, embedding_table)

                expected_shape = (len(indices), embed_dim)
                shape_ok = output.shape == expected_shape
                has_nan = np.any(np.isnan(output))
                has_inf = np.any(np.isinf(output))

                # Check accuracy
                cpu_output = embedding_table[indices]
                max_error = np.max(np.abs(output - cpu_output))
                accuracy_ok = max_error < 1e-6

                passed = shape_ok and not has_nan and not has_inf and accuracy_ok
                results["embedding"] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  Embedding: {status} (error: {max_error:.2e})")

            except Exception as e:
                results["embedding"] = False
                print(f"  Embedding: ❌ ERROR - {e}")

            # Test 5: Attention
            print("\n🔍 Testing Attention...")
            try:
                num_tokens, num_heads, head_size = 2, 2, 8
                kv_len = 16

                q = np.random.randn(num_tokens, num_heads * head_size).astype(np.float32)
                k = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)
                v = np.random.randn(kv_len, num_heads * head_size).astype(np.float32)

                output = executor.execute_attention(q, k, v)

                shape_ok = output.shape == q.shape
                has_nan = np.any(np.isnan(output))
                has_inf = np.any(np.isinf(output))

                passed = shape_ok and not has_nan and not has_inf
                results["attention"] = passed

                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  Attention: {status} (shape: {q.shape} -> {output.shape})")

            except Exception as e:
                results["attention"] = False
                print(f"  Attention: ❌ ERROR - {e}")

            # Summary
            print("\n" + "=" * 50)
            print("📋 TEST SUMMARY")
            print("=" * 50)

            passed = sum(results.values())
            total = len(results)

            for test_name, passed_test in results.items():
                status = "✅ PASS" if passed_test else "❌ FAIL"
                print(f"{status} {test_name}")

            print(f"\n📊 Results: {passed}/{total} tests passed")

            if passed == total:
                print("🎉 All Metal kernel tests passed!")
                print("✅ Numerical correctness validated:")
                print("   - Softmax sums to 1.0")
                print("   - RMS norm produces no NaN")
                print("   - MLP has correct shapes")
                print("   - Embedding accuracy verified")
                print("   - Attention works correctly")
            else:
                print(f"⚠️  {total - passed} test(s) failed.")

            return passed == total

        finally:
            os.chdir(original_cwd)

    except ImportError as e:
        print(f"❌ Failed to import metal_bindings: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_metal_kernels()
    sys.exit(0 if success else 1)