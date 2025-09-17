#!/usr/bin/env python3
"""
Test script for the new backend hierarchy implementation.

This script tests the L4MA Python backend integration to ensure
it properly interfaces with L4MA model components and records tensors.
"""

import sys
import os
import numpy as np
import pytest

# Add the backend path to Python path
sys.path.insert(0, '/home/sslee/Workspace/pie/backend/backend-python')

def test_mock_backend():
    """Test the mock backend implementation."""
    from debug_framework.integrations.backend_interfaces import create_backend, BackendType

    # Create mock backend
    backend = create_backend(BackendType.MOCK, seed=42)

    assert backend.backend_type == BackendType.MOCK
    assert backend.is_available is True

    capabilities = backend.get_capabilities()
    assert capabilities['backend_type'] == 'mock'
    assert capabilities['is_available'] is True

    # Test attention computation
    query = np.random.rand(2, 8, 64).astype(np.float32)
    key = np.random.rand(2, 8, 64).astype(np.float32)
    value = np.random.rand(2, 8, 64).astype(np.float32)

    attention_result = backend.run_attention(query, key, value)
    assert attention_result.output.shape == query.shape
    assert attention_result.computation_time > 0
    assert attention_result.backend_type == BackendType.MOCK
    assert 'mock_attention' in attention_result.metadata['operation']

    # Test MLP computation
    hidden_states = np.random.rand(2, 8, 4096).astype(np.float32)
    mlp_result = backend.run_mlp(hidden_states)
    assert mlp_result.output.shape == hidden_states.shape
    assert mlp_result.computation_time > 0
    assert mlp_result.backend_type == BackendType.MOCK

    # Test embedding computation
    input_ids = np.random.randint(0, 1000, size=(2, 8)).astype(np.int64)
    embedding_result = backend.run_embedding(input_ids, hidden_size=4096)
    expected_shape = input_ids.shape + (4096,)
    assert embedding_result.output.shape == expected_shape
    assert embedding_result.computation_time > 0
    assert embedding_result.backend_type == BackendType.MOCK

    # Test performance metrics
    capabilities = backend.get_capabilities()
    assert capabilities['performance_metrics']['attention_times']['count'] == 1
    assert capabilities['performance_metrics']['mlp_times']['count'] == 1
    assert capabilities['performance_metrics']['embedding_times']['count'] == 1

    backend.cleanup()


def test_l4ma_python_backend():
    """Test the L4MA Python backend implementation."""
    from debug_framework.integrations.backend_interfaces import create_backend, BackendType

    # Create L4MA Python backend without model reference (standalone mode)
    backend = create_backend(BackendType.L4MA_PYTHON, device='cpu')

    assert backend.backend_type == BackendType.L4MA_PYTHON

    if not backend.is_available:
        pytest.skip("L4MA Python backend not available - missing dependencies")

    capabilities = backend.get_capabilities()
    assert capabilities['backend_type'] == 'l4ma_python'
    assert capabilities['is_available'] is True

    # Test attention computation
    query = np.random.rand(1, 4, 128).astype(np.float32)
    key = np.random.rand(1, 4, 128).astype(np.float32)
    value = np.random.rand(1, 4, 128).astype(np.float32)

    attention_result = backend.run_attention(query, key, value, num_heads=32)
    assert attention_result.output.shape == query.shape
    assert attention_result.computation_time > 0
    assert attention_result.backend_type == BackendType.L4MA_PYTHON
    assert attention_result.metadata['operation'] == 'l4ma_attention'
    assert attention_result.metadata['device'] == 'cpu'

    # Test MLP computation
    hidden_states = np.random.rand(1, 4, 4096).astype(np.float32)
    mlp_result = backend.run_mlp(hidden_states)
    assert mlp_result.output.shape == hidden_states.shape
    assert mlp_result.computation_time > 0
    assert mlp_result.backend_type == BackendType.L4MA_PYTHON
    assert mlp_result.metadata['operation'] == 'l4ma_mlp'

    # Test embedding computation
    input_ids = np.random.randint(0, 32000, size=(1, 4)).astype(np.int64)
    embedding_result = backend.run_embedding(input_ids, hidden_size=4096, vocab_size=32768)
    expected_shape = input_ids.shape + (4096,)
    assert embedding_result.output.shape == expected_shape
    assert embedding_result.computation_time > 0
    assert embedding_result.backend_type == BackendType.L4MA_PYTHON
    assert embedding_result.metadata['operation'] == 'l4ma_embedding'

    # Test normalization computation
    hidden_states = np.random.rand(1, 4, 4096).astype(np.float32)
    norm_result = backend.run_normalization(hidden_states)
    assert norm_result.output.shape == hidden_states.shape
    assert norm_result.computation_time > 0
    assert norm_result.backend_type == BackendType.L4MA_PYTHON
    assert norm_result.metadata['operation'] == 'l4ma_normalization'

    backend.cleanup()


def test_backend_validation():
    """Test backend input validation."""
    from debug_framework.integrations.backend_interfaces import create_backend, BackendType

    backend = create_backend(BackendType.MOCK)

    # Test empty tensor validation
    empty_tensor = np.array([])
    with pytest.raises(ValueError, match="empty"):
        backend.run_mlp(empty_tensor)

    # Test non-finite values validation
    inf_tensor = np.array([[np.inf, 1.0], [2.0, np.nan]]).astype(np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        backend.run_mlp(inf_tensor)

    # Test invalid embedding input (negative indices)
    invalid_ids = np.array([[-1, 0], [1, 2]]).astype(np.int64)
    with pytest.raises(ValueError, match="negative"):
        backend.run_embedding(invalid_ids)

    # Test mismatched QKV shapes
    q = np.random.rand(2, 4, 64).astype(np.float32)
    k = np.random.rand(2, 8, 64).astype(np.float32)  # Different seq_len
    v = np.random.rand(2, 4, 64).astype(np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        backend.run_attention(q, k, v)

    backend.cleanup()


def test_backend_factory():
    """Test the backend factory function."""
    from debug_framework.integrations.backend_interfaces import create_backend, BackendType

    # Test creating different backend types
    mock_backend = create_backend(BackendType.MOCK, seed=123)
    assert mock_backend.backend_type == BackendType.MOCK
    assert mock_backend.is_available is True

    l4ma_backend = create_backend(BackendType.L4MA_PYTHON, device='cpu')
    assert l4ma_backend.backend_type == BackendType.L4MA_PYTHON

    # Test invalid backend type
    with pytest.raises(ValueError, match="Unsupported backend type"):
        class InvalidBackendType:
            value = "invalid"
        create_backend(InvalidBackendType())

    mock_backend.cleanup()
    l4ma_backend.cleanup()


def test_tensor_computation_result():
    """Test TensorComputationResult class."""
    from debug_framework.integrations.backend_interfaces import TensorComputationResult, BackendType

    output = np.random.rand(2, 4, 64).astype(np.float32)
    computation_time = 0.001
    backend_type = BackendType.MOCK
    metadata = {'test': 'value'}

    result = TensorComputationResult(
        output=output,
        computation_time=computation_time,
        backend_type=backend_type,
        metadata=metadata
    )

    assert np.array_equal(result.output, output)
    assert result.computation_time == computation_time
    assert result.backend_type == backend_type
    assert result.metadata == metadata
    assert result.timestamp > 0


if __name__ == "__main__":
    """Run tests directly if called as script."""
    import subprocess
    import sys

    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], cwd="/home/sslee/Workspace/pie")

    sys.exit(result.returncode)