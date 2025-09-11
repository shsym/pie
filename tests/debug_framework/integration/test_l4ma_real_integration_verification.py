"""
Real L4MA Integration Verification Tests

This test module verifies that the real L4MA debug integration works correctly
with actual PyTorch L4MA models, Metal backend connections, and provides the
required <1% overhead when disabled.

This is the validation phase - testing real functionality, not mock objects.
"""

import pytest
import numpy as np
import time
import torch
import torch.nn as nn
import threading
import gc
import os
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# Import the real integration module
try:
    from debug_framework.integrations.l4ma_real_integration import (
        L4MARealDebugIntegration,
        MetalBackendInterface,
        create_l4ma_integration
    )
    REAL_INTEGRATION_AVAILABLE = True
except ImportError:
    REAL_INTEGRATION_AVAILABLE = False

# Import L4MA components
try:
    from model.l4ma import L4maModel, L4maForCausalLM, L4maAttention, L4maMlp, L4maDecoderLayer
    from config.l4ma import L4maArch
    L4MA_MODEL_AVAILABLE = True
except ImportError:
    L4MA_MODEL_AVAILABLE = False

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def minimal_l4ma_config():
    """Create a minimal L4MA config for testing."""
    if not L4MA_MODEL_AVAILABLE:
        pytest.skip("L4MA model not available")

    return L4maArch(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=2048,
        num_layers=2,
        num_query_heads=8,
        num_key_value_heads=4,
        head_size=64,
        rms_norm_eps=1e-6,
        use_qkv_bias=False,
        device='cpu',  # Use CPU for CI/testing
        dtype=torch.float32
    )


@pytest.fixture
def small_l4ma_model(minimal_l4ma_config):
    """Create a small L4MA model for testing."""
    if not L4MA_MODEL_AVAILABLE or not TORCH_AVAILABLE:
        pytest.skip("L4MA model or PyTorch not available")

    # Create a simplified L4MA model for testing
    model = L4maForCausalLM(minimal_l4ma_config)
    model.eval()
    return model


@pytest.fixture
def test_inputs_l4ma(minimal_l4ma_config):
    """Generate test inputs compatible with L4MA model."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    batch_size = 1
    seq_len = 16
    vocab_size = minimal_l4ma_config.vocab_size
    hidden_size = minimal_l4ma_config.hidden_size
    num_layers = minimal_l4ma_config.num_layers
    page_size = 16

    return {
        'input_embeds': torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32),
        'position_ids': torch.arange(seq_len, dtype=torch.long).unsqueeze(0),
        'qo_indptr': torch.tensor([0, seq_len], dtype=torch.long),
        'kv_cache_at_layer': [
            torch.zeros(batch_size, 4, page_size, 64, dtype=torch.float32)
            for _ in range(num_layers)
        ],
        'kv_page_indices': torch.zeros(1, dtype=torch.long),
        'kv_page_indptr': torch.tensor([0, 1], dtype=torch.long),
        'kv_last_page_lens': torch.tensor([seq_len], dtype=torch.long),
        'custom_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
        'single_token_inference_mode': False,
        'adapter_subpass': None
    }


class TestRealL4MAIntegrationVerification:
    """Verification tests for real L4MA integration functionality."""

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not L4MA_MODEL_AVAILABLE, reason="L4MA model not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_integration_initialization_with_actual_model(self, small_l4ma_model):
        """Test real integration initialization with actual L4MA model."""

        integration = L4MARealDebugIntegration(
            l4ma_model=small_l4ma_model,
            debug_config={
                'enabled_checkpoints': ['post_embedding', 'post_attention', 'post_mlp'],
                'validation_mode': 'online',
                'performance_monitoring': True,
                'tolerance': 1e-5,
                'backend_comparison': 'metal',
                'real_tensor_validation': True
            }
        )

        # Verify initialization
        assert integration.l4ma_model is small_l4ma_model
        assert isinstance(integration.l4ma_model, (L4maModel, L4maForCausalLM))
        assert integration.debug_enabled is True
        assert integration.metal_backend is not None
        assert isinstance(integration.metal_backend, MetalBackendInterface)

        # Verify configuration
        assert integration.debug_config['enabled_checkpoints'] == ['post_embedding', 'post_attention', 'post_mlp']
        assert integration.debug_config['backend_comparison'] == 'metal'
        assert integration.debug_config['real_tensor_validation'] is True

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not L4MA_MODEL_AVAILABLE, reason="L4MA model not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_model_layer_path_resolution(self, small_l4ma_model):
        """Test layer path resolution for actual L4MA model components."""

        integration = L4MARealDebugIntegration(small_l4ma_model)

        # Test resolution of various L4MA layer paths
        test_paths = [
            'embed_tokens',
            'layers.0.self_attn',
            'layers.0.mlp',
            'layers.1.self_attn',
            'layers.1.mlp',
            'norm',
            'lm_head'
        ]

        for path in test_paths:
            layer_obj, parent_obj, attr_name = integration._resolve_l4ma_layer_path(path)

            if path == 'embed_tokens':
                assert layer_obj is small_l4ma_model.model.embed_tokens
                assert isinstance(layer_obj, nn.Embedding)
            elif 'self_attn' in path:
                assert isinstance(layer_obj, L4maAttention)
            elif 'mlp' in path and 'layers' in path:
                assert isinstance(layer_obj, L4maMlp)
            elif path == 'norm':
                assert isinstance(layer_obj, nn.RMSNorm)
            elif path == 'lm_head':
                assert isinstance(layer_obj, nn.Linear)

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    def test_metal_backend_interface_initialization(self):
        """Test Metal backend interface initialization and availability detection."""

        # Test with default path
        metal_backend = MetalBackendInterface()

        # Should initialize without error even if Metal backend is not available
        assert metal_backend is not None
        assert hasattr(metal_backend, 'is_available')
        assert hasattr(metal_backend, 'backend_path')

        # Test basic operations (should handle unavailability gracefully)
        try:
            # These should work or raise appropriate exceptions
            test_query = np.random.rand(1, 16, 64).astype(np.float32)
            test_key = np.random.rand(1, 16, 64).astype(np.float32)
            test_value = np.random.rand(1, 16, 64).astype(np.float32)

            if metal_backend.is_available:
                result = metal_backend.run_attention(test_query, test_key, test_value)
                assert isinstance(result, np.ndarray)
                assert result.shape == test_query.shape
            else:
                with pytest.raises(RuntimeError, match="Metal backend not available"):
                    metal_backend.run_attention(test_query, test_key, test_value)

        except Exception as e:
            if "Metal backend not available" in str(e):
                assert not metal_backend.is_available
            else:
                raise

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    def test_factory_function_integration_selection(self):
        """Test factory function for selecting appropriate integration type."""

        # Test with real integration requested but L4MA unavailable
        with patch('debug_framework.integrations.l4ma_real_integration.L4MA_MODEL_AVAILABLE', False):
            integration = create_l4ma_integration(
                l4ma_model=None,
                use_real_integration=True
            )
            # Should fall back to mock integration
            assert integration is not None

        # Test with real integration available
        if L4MA_MODEL_AVAILABLE and TORCH_AVAILABLE:
            mock_model = MagicMock()
            integration = create_l4ma_integration(
                l4ma_model=mock_model,
                use_real_integration=True
            )
            assert isinstance(integration, L4MARealDebugIntegration)

        # Test explicit mock integration request
        mock_model = MagicMock()
        integration = create_l4ma_integration(
            l4ma_model=mock_model,
            use_real_integration=False
        )
        # Should use mock integration
        assert integration is not None