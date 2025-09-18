#!/usr/bin/env python3
"""
Critical Tensor Validation Tests (T063-T065)

This test suite implements three critical validation tests to verify that recorded
tensor values are correctly stored as binary with metadata, can be loaded/extracted
properly, and reloaded values match original memory exactly.

Tests:
- T063: Binary storage with metadata validation
- T064: Tensor loading/extraction validation
- T065: Memory matching validation

Uses existing working tensor capture system that records events during inference.
"""

import sys
import os
import tempfile
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import threading
import time

# Add backend-python to path
backend_python_path = Path(__file__).parent.parent.parent.parent / "backend" / "backend-python"
sys.path.insert(0, str(backend_python_path))

# Import debug framework components
try:
    from debug_framework.models.tensor_recording import TensorRecording, TensorRecordingManager
    from debug_framework.services.database_manager import DatabaseManager
    from debug_framework.integrations.l4ma_real_integration import L4MARealDebugIntegration
    DEBUG_FRAMEWORK_AVAILABLE = True
except ImportError as e:
    print(f"Debug framework not available: {e}")
    DEBUG_FRAMEWORK_AVAILABLE = False

# Import backend integration test for tensor capture
try:
    from test_l4ma_backend_integration import BackendReuseIntegrationTest
    BACKEND_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Backend integration not available: {e}")
    BACKEND_INTEGRATION_AVAILABLE = False


class TensorValidationCriticalTest:
    """
    Critical validation tests for tensor recording, loading, and memory matching.

    This test suite validates the core functionality of the debug framework's
    tensor recording system using real tensor data captured during inference.
    """

    def __init__(self):
        self.temp_dir = None
        self.test_recordings: List[TensorRecording] = []
        self.captured_tensors: List[Dict[str, Any]] = []
        self.capture_lock = threading.Lock()
        self.recording_active = False
        self.backend_test = None

    def setup_test_environment(self) -> bool:
        """Set up test environment with temporary storage and backend integration."""
        print("🔧 Setting up test environment...")

        if not DEBUG_FRAMEWORK_AVAILABLE:
            print("❌ Debug framework not available")
            return False

        if not BACKEND_INTEGRATION_AVAILABLE:
            print("❌ Backend integration not available")
            return False

        try:
            # Create temporary directory for test files
            self.temp_dir = tempfile.mkdtemp(prefix="tensor_validation_test_")
            print(f"📁 Created temp directory: {self.temp_dir}")

            # Register pytorch backend for testing
            self._register_pytorch_backend()

            # Set up backend integration for tensor capture
            self.backend_test = BackendReuseIntegrationTest()
            if not self.backend_test.load_model_with_backend_handler():
                print("❌ Failed to load model with backend")
                return False

            if not self.backend_test.integrate_with_debug_framework():
                print("❌ Failed to setup debug framework integration")
                return False

            # Set up tensor capture callback
            self.backend_test.debug_integration.set_tensor_capture_callback(self._tensor_capture_callback)

            print("✅ Test environment setup complete")
            return True

        except Exception as e:
            print(f"❌ Failed to setup test environment: {e}")
            return False

    def _register_pytorch_backend(self):
        """Register pytorch backend for testing purposes."""
        try:
            from debug_framework.services.plugin_registry import PluginRegistry
            import debug_framework.services.plugin_registry as plugin_registry_module

            # Create a shared registry instance and monkey-patch the module
            self.shared_registry = PluginRegistry()

            # Monkey-patch the PluginRegistry class to return our shared instance
            original_init = PluginRegistry.__init__
            def patched_init(self_inner, *args, **kwargs):
                # Copy the shared registry's state to this instance
                self_inner.__dict__.update(self.shared_registry.__dict__)

            PluginRegistry.__init__ = patched_init

            # Create a dummy source file for the pytorch backend
            dummy_source = os.path.join(self.temp_dir, "pytorch_backend.py")
            with open(dummy_source, 'w') as f:
                f.write("# Dummy PyTorch backend for testing\n")

            # Register the pytorch backend
            plugin_metadata = {
                "name": "pytorch",
                "version": "1.0.0",
                "backend_type": "python",
                "supported_operations": ["forward", "backward", "inference"],
                "source_file": dummy_source
            }

            plugin_id = self.shared_registry.register_plugin(plugin_metadata)
            print(f"✅ Registered PyTorch backend with ID: {plugin_id}")

            # Store original for cleanup
            self.original_plugin_registry_init = original_init

        except Exception as e:
            print(f"⚠️ Failed to register PyTorch backend: {e}")
            # Continue anyway - we'll handle this in the test

    def _tensor_capture_callback(self, checkpoint_name: str, tensor_data: Any, metadata: Dict[str, Any]):
        """Callback to capture tensor data during inference, including input tensors."""
        with self.capture_lock:
            if not self.recording_active:
                return

            # Store captured tensor information for output tensor
            capture_info = {
                'checkpoint_name': checkpoint_name,
                'tensor_data': tensor_data,
                'metadata': metadata,
                'timestamp': time.perf_counter(),
                'memory_hash': None,  # Will compute for validation
                'tensor_type': 'output'  # Mark as output tensor
            }

            # Compute memory hash for output tensor
            if hasattr(tensor_data, 'numpy'):
                # PyTorch tensor
                numpy_data = tensor_data.detach().cpu().numpy()
                capture_info['memory_hash'] = hashlib.sha256(numpy_data.tobytes()).hexdigest()
                capture_info['numpy_data'] = numpy_data
            elif isinstance(tensor_data, np.ndarray):
                # NumPy array
                capture_info['memory_hash'] = hashlib.sha256(tensor_data.tobytes()).hexdigest()
                capture_info['numpy_data'] = tensor_data.copy()
            elif isinstance(tensor_data, dict):
                # Dict of tensors - process first tensor found
                for key, value in tensor_data.items():
                    if hasattr(value, 'numpy'):
                        numpy_data = value.detach().cpu().numpy()
                        capture_info['memory_hash'] = hashlib.sha256(numpy_data.tobytes()).hexdigest()
                        capture_info['numpy_data'] = numpy_data
                        capture_info['tensor_key'] = key
                        break
                    elif isinstance(value, np.ndarray):
                        capture_info['memory_hash'] = hashlib.sha256(value.tobytes()).hexdigest()
                        capture_info['numpy_data'] = value.copy()
                        capture_info['tensor_key'] = key
                        break

            # Capture output tensor if we got valid data
            if capture_info['memory_hash']:
                self.captured_tensors.append(capture_info)

            # Capture input tensors if available
            if 'captured_inputs' in metadata:
                for input_name, input_tensor_data in metadata['captured_inputs'].items():
                    input_capture_info = {
                        'checkpoint_name': f"{checkpoint_name}_input_{input_name}",
                        'tensor_data': input_tensor_data,
                        'metadata': metadata,
                        'timestamp': time.perf_counter(),
                        'memory_hash': None,
                        'tensor_type': 'input',
                        'input_name': input_name,
                        'parent_checkpoint': checkpoint_name
                    }

                    # Input tensors are already numpy arrays from our decorator
                    if isinstance(input_tensor_data, np.ndarray):
                        input_capture_info['memory_hash'] = hashlib.sha256(input_tensor_data.tobytes()).hexdigest()
                        input_capture_info['numpy_data'] = input_tensor_data.copy()
                        self.captured_tensors.append(input_capture_info)

    def capture_real_tensors(self) -> bool:
        """Capture real tensor data during inference to test with."""
        print("\n🔍 Capturing real tensor data from inference...")

        if not self.backend_test or not self.backend_test.debug_integration:
            print("❌ Backend integration not available")
            return False

        try:
            self.recording_active = True
            self.captured_tensors.clear()

            # Run inference to capture tensors
            test_prompt = "The capital of France is"
            print(f"🔍 Running inference with prompt: '{test_prompt}'")

            # Temporarily disable debug output for cleaner test results
            import os
            old_debug_level = os.environ.get('DEBUG_LEVEL', '')
            os.environ['DEBUG_LEVEL'] = 'ERROR'

            result = self.backend_test.test_prompt_inference_with_handler(test_prompt)

            # Restore debug level
            os.environ['DEBUG_LEVEL'] = old_debug_level

            if not result.get('success', False):
                print(f"❌ Inference failed: {result.get('error', 'unknown')}")
                return False

            print(f"✅ Captured {len(self.captured_tensors)} tensors from inference")
            if len(self.captured_tensors) > 0:
                print(f"   Sample tensor shapes: {[c['numpy_data'].shape for c in self.captured_tensors[:3]]}")
                print(f"   Sample tensor hashes: {[c['memory_hash'][:8] + '...' for c in self.captured_tensors[:3]]}")

            # Ensure we have enough tensors for testing
            if len(self.captured_tensors) < 3:
                print(f"⚠️ Warning: Only captured {len(self.captured_tensors)} tensors, need at least 3 for comprehensive testing")
                # Continue anyway for partial testing

            return len(self.captured_tensors) > 0

        except Exception as e:
            print(f"❌ Failed to capture tensors: {e}")
            return False
        finally:
            self.recording_active = False

    def test_t063_binary_storage_with_metadata(self) -> bool:
        """
        T063: Binary storage with metadata validation

        Verifies that recorded tensor values are correctly stored as binary
        with complete metadata including shape, dtype, device info, etc.
        """
        print("\n🧪 T063: Binary Storage with Metadata Validation")
        print("=" * 55)

        if len(self.captured_tensors) == 0:
            print("❌ No captured tensors available for testing")
            return False

        try:
            validation_results = []

            for i, capture_info in enumerate(self.captured_tensors):  # Test all captured tensors
                print(f"\n📋 Testing tensor {i+1}: {capture_info['checkpoint_name']}")

                numpy_data = capture_info['numpy_data']
                checkpoint_name = capture_info['checkpoint_name']

                # Create TensorRecording with binary storage
                device_info = {
                    'platform': 'cpu',  # Simplified for testing
                    'device': 'cpu'
                }

                tensor_recording = TensorRecording.create_from_tensor(
                    session_id=1,
                    checkpoint_id=i + 1,
                    tensor_name=f"test_tensor_{checkpoint_name}_{i}",
                    tensor_data=numpy_data,
                    backend_name='pytorch',  # Now registered in setup
                    device_info=device_info,
                    storage_dir=self.temp_dir
                )

                # Validate binary file creation
                binary_file_exists = os.path.exists(tensor_recording.tensor_data_path)

                # Validate metadata completeness
                metadata = tensor_recording.tensor_metadata
                required_fields = {'dtype', 'shape', 'strides', 'device', 'memory_layout', 'byte_order'}
                has_required_metadata = required_fields.issubset(set(metadata.keys()))

                # Validate file size matches tensor size
                expected_size = numpy_data.nbytes
                actual_file_size = os.path.getsize(tensor_recording.tensor_data_path) if binary_file_exists else 0
                size_matches = (actual_file_size == expected_size)

                # Validate integrity check
                integrity_results = tensor_recording.validate_recording_integrity()
                integrity_valid = all(integrity_results.values())

                result = {
                    'tensor_name': f"test_tensor_{checkpoint_name}_{i}",
                    'binary_file_exists': binary_file_exists,
                    'has_required_metadata': has_required_metadata,
                    'size_matches': size_matches,
                    'integrity_valid': integrity_valid,
                    'file_path': tensor_recording.tensor_data_path,
                    'metadata': metadata
                }

                validation_results.append(result)
                self.test_recordings.append(tensor_recording)

                # Print detailed results
                print(f"   📄 Binary file exists: {'✅' if binary_file_exists else '❌'}")
                print(f"   📋 Required metadata: {'✅' if has_required_metadata else '❌'}")
                print(f"   📏 Size matches: {'✅' if size_matches else '❌'} ({actual_file_size} == {expected_size})")
                print(f"   🔍 Integrity valid: {'✅' if integrity_valid else '❌'}")

            # Overall validation
            all_valid = all(
                r['binary_file_exists'] and r['has_required_metadata'] and
                r['size_matches'] and r['integrity_valid']
                for r in validation_results
            )

            print(f"\n📊 T063 Results: {len(validation_results)} tensors tested")
            print(f"Overall result: {'✅ PASS' if all_valid else '❌ FAIL'}")

            return all_valid

        except Exception as e:
            print(f"❌ T063 failed with exception: {e}")
            return False

    def test_t064_tensor_loading_extraction(self) -> bool:
        """
        T064: Tensor loading/extraction validation

        Verifies that stored tensors can be loaded/extracted properly
        from binary files with correct shape, dtype, and metadata.
        """
        print("\n🧪 T064: Tensor Loading/Extraction Validation")
        print("=" * 50)

        if len(self.test_recordings) == 0:
            print("❌ No recorded tensors available for loading test")
            return False

        try:
            loading_results = []

            for i, tensor_recording in enumerate(self.test_recordings):
                print(f"\n📋 Testing load {i+1}: {tensor_recording.tensor_name}")

                # Load tensor data from binary file
                loaded_tensor = tensor_recording.load_tensor_data()

                # Get original tensor data for comparison
                original_tensor = self.captured_tensors[i]['numpy_data']

                # Validate loaded tensor properties
                shape_matches = loaded_tensor.shape == original_tensor.shape
                dtype_matches = loaded_tensor.dtype == original_tensor.dtype

                # Validate metadata consistency
                metadata = tensor_recording.tensor_metadata
                metadata_shape_matches = tuple(metadata['shape']) == loaded_tensor.shape
                metadata_dtype_matches = metadata['dtype'] == str(loaded_tensor.dtype)

                # Check if data can be read without errors
                loading_successful = True
                try:
                    _ = loaded_tensor.flat[0]  # Try to access data
                except Exception:
                    loading_successful = False

                result = {
                    'tensor_name': tensor_recording.tensor_name,
                    'loading_successful': loading_successful,
                    'shape_matches': shape_matches,
                    'dtype_matches': dtype_matches,
                    'metadata_shape_matches': metadata_shape_matches,
                    'metadata_dtype_matches': metadata_dtype_matches,
                    'original_shape': original_tensor.shape,
                    'loaded_shape': loaded_tensor.shape,
                    'original_dtype': original_tensor.dtype,
                    'loaded_dtype': loaded_tensor.dtype
                }

                loading_results.append(result)

                # Print detailed results
                print(f"   📦 Loading successful: {'✅' if loading_successful else '❌'}")
                print(f"   📐 Shape matches: {'✅' if shape_matches else '❌'} ({original_tensor.shape} == {loaded_tensor.shape})")
                print(f"   🔤 Dtype matches: {'✅' if dtype_matches else '❌'} ({original_tensor.dtype} == {loaded_tensor.dtype})")
                print(f"   📋 Metadata shape: {'✅' if metadata_shape_matches else '❌'}")
                print(f"   📋 Metadata dtype: {'✅' if metadata_dtype_matches else '❌'}")

            # Overall validation
            all_valid = all(
                r['loading_successful'] and r['shape_matches'] and r['dtype_matches'] and
                r['metadata_shape_matches'] and r['metadata_dtype_matches']
                for r in loading_results
            )

            print(f"\n📊 T064 Results: {len(loading_results)} tensors tested")
            print(f"Overall result: {'✅ PASS' if all_valid else '❌ FAIL'}")

            return all_valid

        except Exception as e:
            print(f"❌ T064 failed with exception: {e}")
            return False

    def test_t065_memory_matching_validation(self) -> bool:
        """
        T065: Memory matching validation

        Verifies that reloaded tensor values match original memory exactly,
        including bit-level comparison and hash validation.
        """
        print("\n🧪 T065: Memory Matching Validation")
        print("=" * 40)

        if len(self.test_recordings) == 0:
            print("❌ No recorded tensors available for memory matching test")
            return False

        try:
            memory_results = []

            for i, tensor_recording in enumerate(self.test_recordings):
                print(f"\n📋 Testing memory match {i+1}: {tensor_recording.tensor_name}")

                # Load tensor data from binary file
                loaded_tensor = tensor_recording.load_tensor_data()

                # Get original tensor data and hash
                original_tensor = self.captured_tensors[i]['numpy_data']
                original_hash = self.captured_tensors[i]['memory_hash']

                # Compute hash of loaded tensor
                loaded_hash = hashlib.sha256(loaded_tensor.tobytes()).hexdigest()

                # Exact memory comparison
                memory_exact_match = np.array_equal(original_tensor, loaded_tensor)

                # Hash comparison (more reliable for detecting corruption)
                hash_matches = original_hash == loaded_hash

                # Byte-level comparison
                original_bytes = original_tensor.tobytes()
                loaded_bytes = loaded_tensor.tobytes()
                bytes_match = original_bytes == loaded_bytes

                # Value-wise comparison with tolerance (for floating point)
                values_close = np.allclose(original_tensor, loaded_tensor, rtol=1e-15, atol=1e-15)

                # Size comparison
                size_matches = original_tensor.nbytes == loaded_tensor.nbytes

                result = {
                    'tensor_name': tensor_recording.tensor_name,
                    'memory_exact_match': memory_exact_match,
                    'hash_matches': hash_matches,
                    'bytes_match': bytes_match,
                    'values_close': values_close,
                    'size_matches': size_matches,
                    'original_hash': original_hash,
                    'loaded_hash': loaded_hash,
                    'original_size': original_tensor.nbytes,
                    'loaded_size': loaded_tensor.nbytes
                }

                memory_results.append(result)

                # Print focused memory validation results
                status = "✅ PASS" if (memory_exact_match and hash_matches and bytes_match) else "❌ FAIL"
                print(f"   {status} - Memory validation")
                print(f"   Original hash: {original_hash[:16]}...")
                print(f"   Loaded hash:   {loaded_hash[:16]}...")
                print(f"   Hash match: {'✅' if hash_matches else '❌'} | Exact match: {'✅' if memory_exact_match else '❌'} | Size: {original_tensor.nbytes} bytes")

            # Overall validation - all memory checks must pass
            all_valid = all(
                r['memory_exact_match'] and r['hash_matches'] and r['bytes_match'] and
                r['values_close'] and r['size_matches']
                for r in memory_results
            )

            print(f"\n📊 T065 Results: {len(memory_results)} tensors tested")
            print(f"Overall result: {'✅ PASS' if all_valid else '❌ FAIL'}")

            return all_valid

        except Exception as e:
            print(f"❌ T065 failed with exception: {e}")
            return False

    def cleanup_test_environment(self):
        """Clean up test environment and temporary files."""
        print("\n🧹 Cleaning up test environment...")

        try:
            # Restore original PluginRegistry init if we patched it
            if hasattr(self, 'original_plugin_registry_init'):
                from debug_framework.services.plugin_registry import PluginRegistry
                PluginRegistry.__init__ = self.original_plugin_registry_init
                print("✅ Restored original PluginRegistry")

            # Clean up temporary files
            if self.temp_dir and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                print(f"✅ Cleaned up temp directory: {self.temp_dir}")

            # Clean up backend integration
            if self.backend_test and hasattr(self.backend_test, 'cleanup'):
                self.backend_test.cleanup()

        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")

    def run_all_critical_tests(self) -> bool:
        """Run all critical validation tests (T063-T065)."""
        print("🚀 Critical Tensor Validation Tests (T063-T065)")
        print("=" * 70)

        try:
            # Setup
            if not self.setup_test_environment():
                return False

            # Capture real tensors for testing
            if not self.capture_real_tensors():
                return False

            # Run the three critical tests
            t063_pass = self.test_t063_binary_storage_with_metadata()
            t064_pass = self.test_t064_tensor_loading_extraction()
            t065_pass = self.test_t065_memory_matching_validation()

            # Summary
            print("\n📊 Critical Tensor Validation Test Summary:")
            print("=" * 70)
            print(f"T063 - Binary storage with metadata: {'✅ PASS' if t063_pass else '❌ FAIL'}")
            print(f"T064 - Tensor loading/extraction: {'✅ PASS' if t064_pass else '❌ FAIL'}")
            print(f"T065 - Memory matching validation: {'✅ PASS' if t065_pass else '❌ FAIL'}")

            overall_success = all([t063_pass, t064_pass, t065_pass])

            if overall_success:
                print(f"\n🎉 All critical validation tests passed!")
                print("✅ Tensor recording system is working correctly")
            else:
                print(f"\n❌ Some critical validation tests failed")
                print("   Tensor recording system needs attention")

            return overall_success

        finally:
            self.cleanup_test_environment()


def main():
    """Main test function."""
    if not DEBUG_FRAMEWORK_AVAILABLE:
        print("❌ Debug framework not available")
        return False

    if not BACKEND_INTEGRATION_AVAILABLE:
        print("❌ Backend integration not available")
        return False

    test = TensorValidationCriticalTest()
    return test.run_all_critical_tests()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)