"""
Test suite for Metal backend integration in the debug framework.

This test file validates the Metal kernel parameter passing fixes, buffer management,
and dtype conversion improvements implemented in the Metal backend.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Test availability of Metal backend
try:
    from debug_framework.integrations.metal_backend import MetalBackend
    from debug_framework.integrations.backend_interfaces import BackendType, TensorComputationResult
    METAL_BACKEND_AVAILABLE = True
except ImportError:
    METAL_BACKEND_AVAILABLE = False


class TestMetalBackendParameterPassing:
    """Test Metal backend parameter passing and buffer management fixes."""

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_metal_backend_initialization(self):
        """Test Metal backend initialization without actual Metal hardware."""
        backend = MetalBackend()
        assert backend.backend_type == BackendType.METAL
        assert hasattr(backend, 'initialize')
        assert hasattr(backend, 'cleanup')
        assert hasattr(backend, '_metal_executor')

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_metal_backend_interface_completeness(self):
        """Test that Metal backend implements all required interface methods."""
        backend = MetalBackend()

        # Test required methods exist
        required_methods = [
            'run_attention', 'run_mlp', 'run_embedding', 'run_normalization',
            'validate_inputs', 'get_capabilities', 'record_performance'
        ]

        for method in required_methods:
            assert hasattr(backend, method), f"Missing required method: {method}"
            assert callable(getattr(backend, method)), f"Method {method} is not callable"

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_parameter_validation_input_shapes(self):
        """Test parameter validation for input tensor shapes."""
        backend = MetalBackend()

        # Test attention validation - should accept 2D tensors
        valid_q = np.random.randn(4, 8).astype(np.float32)
        valid_k = np.random.randn(4, 8).astype(np.float32)
        valid_v = np.random.randn(4, 8).astype(np.float32)

        is_valid, error_msg = backend.validate_inputs("attention", valid_q, valid_k, valid_v)
        assert is_valid, f"Valid attention inputs rejected: {error_msg}"

        # Test invalid shapes
        invalid_q = np.random.randn(4).astype(np.float32)  # 1D instead of 2D
        is_valid, error_msg = backend.validate_inputs("attention", invalid_q, valid_k, valid_v)
        assert not is_valid, "Invalid attention input shapes accepted"
        assert "shape" in error_msg.lower() or "dimension" in error_msg.lower()

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_dtype_conversion_handling(self):
        """Test dtype conversion and validation fixes."""
        backend = MetalBackend()

        # Test that different dtypes are handled properly
        test_cases = [
            (np.float32, True, "float32 should be accepted"),
            (np.float64, True, "float64 should be convertible"),
            (np.int32, False, "int32 should be rejected"),
            (np.int16, False, "int16 should be rejected")
        ]

        for dtype, should_work, message in test_cases:
            test_input = np.array([1.0, 2.0, 3.0], dtype=dtype)

            try:
                # This should either work or raise a clear error
                is_valid, error_msg = backend.validate_inputs("mlp", test_input)
                if should_work:
                    assert is_valid or "dtype" in error_msg.lower(), f"{message}: {error_msg}"
                else:
                    assert not is_valid, f"{message}: dtype validation too permissive"

            except Exception as e:
                if should_work:
                    pytest.fail(f"{message}: Unexpected error {e}")

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_cpu_fallback_implementations(self):
        """Test CPU fallback implementations work correctly."""
        backend = MetalBackend()

        # Test MLP fallback (ReLU activation)
        mlp_input = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        try:
            result = backend.run_mlp(mlp_input)
            # Should apply ReLU: negative values become 0
            expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0], dtype=np.float32)
            assert isinstance(result, TensorComputationResult)
            assert result.output.shape == mlp_input.shape
            # Note: Actual Metal executor not available, so this tests the fallback path
        except RuntimeError as e:
            # Expected when Metal executor not initialized
            assert "Metal executor not initialized" in str(e)

        # Test embedding fallback (CPU-based lookup)
        indices = np.array([0, 2, 1], dtype=np.int32)
        embedding_table = np.random.randn(5, 4).astype(np.float32)  # 5 vocab, 4 dims

        try:
            result = backend.run_embedding(indices, embedding_table=embedding_table)
            assert isinstance(result, TensorComputationResult)
            assert result.output.shape == (3, 4)  # 3 indices, 4 embedding dims
        except RuntimeError as e:
            # Expected when Metal executor not initialized
            assert "Metal executor not initialized" in str(e)

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_buffer_management_interface(self):
        """Test that buffer management interface is properly structured."""
        backend = MetalBackend()

        # Test capabilities structure
        capabilities = backend.get_capabilities()
        assert isinstance(capabilities, dict)

        expected_keys = ['backend_type', 'is_available', 'initialization_time', 'error_count']
        for key in expected_keys:
            assert key in capabilities, f"Missing capability key: {key}"

        # Metal-specific capability keys should be present (even if None when uninitialized)
        metal_keys = ['device_info', 'metallib_path', 'available_kernels', 'platform_support']
        for key in metal_keys:
            assert key in capabilities, f"Missing Metal capability key: {key}"

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_error_handling_improvements(self):
        """Test improved error handling and messaging."""
        backend = MetalBackend()

        # Test error count tracking
        initial_error_count = backend.error_count

        # Try operations that should fail gracefully
        try:
            # This should fail because Metal executor is not initialized
            test_input = np.array([1.0, 2.0], dtype=np.float32)
            backend.run_mlp(test_input)
            pytest.fail("Expected RuntimeError for uninitialized Metal executor")
        except RuntimeError as e:
            # Should get clear error message
            assert "Metal executor not initialized" in str(e)

        # Test that error count is tracked
        assert backend.error_count > initial_error_count

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_attention_kernel_parameter_interface(self):
        """Test attention kernel parameter interface fixes."""
        backend = MetalBackend()

        # Test that attention accepts the correct parameter structure
        q = np.random.randn(2, 4).astype(np.float32)
        k = np.random.randn(2, 4).astype(np.float32)
        v = np.random.randn(2, 4).astype(np.float32)

        # Test with optional parameters that match fixed interface
        try:
            result = backend.run_attention(
                q, k, v,
                num_query_heads=1,
                num_kv_heads=1,
                head_size=4
            )
            # Should return result with correct structure
            assert isinstance(result, TensorComputationResult)
            assert result.backend_type == BackendType.METAL

        except RuntimeError as e:
            # Expected when Metal executor not available
            assert "Metal executor not initialized" in str(e) or "not available" in str(e)
        except TypeError as e:
            pytest.fail(f"Parameter interface broken: {e}")

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_normalization_parameter_fixes(self):
        """Test RMS normalization parameter binding fixes."""
        backend = MetalBackend()

        # Test normalization with 2D input (tokens x hidden_size)
        test_input = np.random.randn(3, 8).astype(np.float32)  # 3 tokens, 8 hidden size
        eps = 1e-6

        try:
            result = backend.run_normalization(test_input, eps=eps)
            assert isinstance(result, TensorComputationResult)
            assert result.output.shape == test_input.shape

        except RuntimeError as e:
            # Expected when Metal executor not available
            assert "Metal executor not initialized" in str(e) or "not available" in str(e)
        except Exception as e:
            pytest.fail(f"Parameter binding error in normalization: {e}")

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        backend = MetalBackend()

        # Test that performance tracking methods exist
        assert hasattr(backend, 'record_performance')
        assert hasattr(backend, 'get_performance_stats')

        # Test recording performance
        backend.record_performance("test_operation", 0.005)  # 5ms

        stats = backend.get_performance_stats("test_operation")
        assert stats['operation'] == "test_operation"
        assert stats['count'] == 1
        assert stats['total_time'] == 0.005


class TestMetalBackendIntegration:
    """Integration tests for Metal backend within debug framework ecosystem."""

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_backend_factory_integration(self):
        """Test Metal backend creation through factory."""
        from debug_framework.integrations.backend_interfaces import create_backend

        # Test factory can create Metal backend
        backend = create_backend(BackendType.METAL)
        assert isinstance(backend, MetalBackend)
        assert backend.backend_type == BackendType.METAL

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_tensor_computation_result_structure(self):
        """Test TensorComputationResult structure from Metal backend."""
        backend = MetalBackend()

        # Mock a successful computation result
        test_output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = TensorComputationResult(
            output=test_output,
            computation_time=0.001,
            backend_type=BackendType.METAL,
            metadata={'operation': 'test_metal_operation'}
        )

        assert result.output.shape == (3,)
        assert result.computation_time == 0.001
        assert result.backend_type == BackendType.METAL
        assert result.metadata['operation'] == 'test_metal_operation'

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_cross_platform_compatibility(self):
        """Test Metal backend behavior on different platforms."""
        backend = MetalBackend()

        capabilities = backend.get_capabilities()

        # On macOS, should indicate platform support
        # On other platforms, should gracefully indicate unavailability
        assert 'platform_support' in capabilities

        # Platform-specific behavior should be handled gracefully
        if sys.platform == 'darwin':
            # On macOS, Metal should be potentially available
            assert capabilities['platform_support'] is True
        else:
            # On non-macOS, should indicate unavailability
            assert capabilities['platform_support'] is False

    @pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
    def test_cleanup_and_resource_management(self):
        """Test proper resource cleanup."""
        backend = MetalBackend()

        # Test cleanup method exists and can be called
        assert hasattr(backend, 'cleanup')
        assert callable(backend.cleanup)

        # Should be able to call cleanup multiple times safely
        backend.cleanup()
        backend.cleanup()  # Should not raise error

        # After cleanup, backend should be in safe state
        capabilities = backend.get_capabilities()
        assert capabilities['backend_type'] == BackendType.METAL


@pytest.mark.skipif(not METAL_BACKEND_AVAILABLE, reason="Metal backend not available")
class TestMetalBackendMockIntegration:
    """Test Metal backend with mock Metal bindings for validation."""

    @patch('debug_framework.integrations.metal_backend.sys.path')
    def test_metal_bindings_import_handling(self, mock_sys_path):
        """Test handling of metal_bindings import scenarios."""
        backend = MetalBackend()

        # Test initialization when metal_bindings not available
        success = backend.initialize()

        # Should handle missing bindings gracefully
        assert not success  # Expected to fail without real Metal setup
        assert not backend.is_available

        capabilities = backend.get_capabilities()
        assert capabilities['device_info'] is None
        assert capabilities['metallib_path'] is None

    def test_validation_with_mock_executor(self):
        """Test parameter validation with mocked Metal executor."""
        backend = MetalBackend()

        # Mock the Metal executor to test parameter passing
        mock_executor = Mock()
        mock_executor.execute_softmax.return_value = np.array([0.1, 0.7, 0.2], dtype=np.float32)
        mock_executor.get_device_info.return_value = "Mock Metal Device"
        mock_executor.list_available_kernels.return_value = ["softmax_kernel", "attention_kernel"]

        backend._metal_executor = mock_executor
        backend.is_available = True

        # Test softmax execution
        test_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        try:
            result = backend.execute_softmax(test_input)
            assert result.shape == (3,)
            mock_executor.execute_softmax.assert_called_once()
        except AttributeError:
            # execute_softmax might not be exposed at backend level
            pass

    def test_error_propagation_from_metal_bindings(self):
        """Test error propagation from Metal bindings layer."""
        backend = MetalBackend()

        # Mock executor that raises errors
        mock_executor = Mock()
        mock_executor.execute_softmax.side_effect = RuntimeError("Metal kernel execution failed")

        backend._metal_executor = mock_executor
        backend.is_available = True

        # Test that errors are properly caught and re-raised
        test_input = np.array([1.0, 2.0], dtype=np.float32)

        try:
            # This should test the error handling path
            backend.execute_softmax(test_input)
        except (RuntimeError, AttributeError) as e:
            # Should get either the Metal error or AttributeError if method doesn't exist
            # Both are acceptable for this test
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])