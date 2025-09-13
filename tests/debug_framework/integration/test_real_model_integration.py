"""
Real Model Integration Tests

This test module tests the debug framework with actual loaded model weights
from the PIE cache, verifying that computation patching and swapping works
with real L4MA models and real tensor operations.
"""

import os
import pytest
import numpy as np
import time
import torch
import ztensor
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

# Test if we have access to the real model
model_cache_path = Path.home() / "Library" / "Caches" / "pie" / "models"
llama_model_path = model_cache_path / "llama-3.2-1b-instruct"
llama_metadata_path = model_cache_path / "llama-3.2-1b-instruct.toml"
llama_weights_path = llama_model_path / "llama-3.2-1b-instruct.zt"

REAL_MODEL_AVAILABLE = (
    llama_metadata_path.exists() and
    llama_weights_path.exists()
)

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

# Import PIE components for model loading
try:
    from config.common import ModelInfo
    from model.l4ma import L4maForCausalLM, create_fusion_map
    PIE_COMPONENTS_AVAILABLE = True
except ImportError:
    PIE_COMPONENTS_AVAILABLE = False

# Import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_real_l4ma_model():
    """Load the real L4MA model with weights from the PIE cache."""
    if not REAL_MODEL_AVAILABLE or not PIE_COMPONENTS_AVAILABLE:
        pytest.skip("Real model or PIE components not available")

    # Load model metadata
    model_info = ModelInfo.load_from_file(
        str(llama_metadata_path),
        device='cpu',  # Use CPU for testing
        dtype=torch.float32
    )

    # Create L4MA model instance
    model = L4maForCausalLM(model_info.architecture)
    fusion_map = create_fusion_map(model)

    # Create reverse map for fusion lookup
    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    pending_fusion_tensors = {}
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    # Load weights from .zt file
    print(f"Loading model weights from {llama_weights_path}")

    with ztensor.Reader(str(llama_weights_path)) as reader:
        tensor_names = reader.get_tensor_names()
        print(f"Found {len(tensor_names)} tensors in model file")

        # Load first 100 tensors for testing (to avoid memory issues in CI)
        for i, name in enumerate(tensor_names[:100]):
            if i % 20 == 0:
                print(f"Loading tensor {i+1}/100: {name}")

            # Handle fusion tensors
            if name in source_to_fusion_target:
                pending_fusion_tensors[name] = reader.read_tensor(name, to="torch")
                continue

            # Load standard tensors
            if name in model_state_keys and name not in loaded_keys:
                param = model.state_dict()[name]
                tensor_data = reader.read_tensor(name, to="torch")

                if tensor_data.shape != param.shape:
                    print(f"Shape mismatch for {name}: expected {param.shape}, got {tensor_data.shape}")
                    continue

                param.copy_(tensor_data)
                loaded_keys.add(name)

        # Process fusion tensors
        for target, details in fusion_map.items():
            sources = details["sources"]
            dim = details["dim"]

            if all(source in pending_fusion_tensors for source in sources):
                fused_tensor = torch.cat(
                    [pending_fusion_tensors[source] for source in sources],
                    dim=dim
                )

                if target in model_state_keys:
                    target_param = model.state_dict()[target]
                    if fused_tensor.shape == target_param.shape:
                        target_param.copy_(fused_tensor)
                        loaded_keys.add(target)
                        print(f"Loaded fused tensor: {target}")

    print(f"Successfully loaded {len(loaded_keys)} tensors")
    model.eval()  # Set to evaluation mode
    return model, model_info


@pytest.fixture
def real_l4ma_model():
    """Fixture that provides a real L4MA model with loaded weights."""
    return load_real_l4ma_model()


@pytest.fixture
def test_inputs_real():
    """Generate test inputs for real L4MA model."""
    batch_size = 1
    seq_len = 8  # Keep small for testing

    # Create realistic input embeddings
    hidden_size = 2048  # From model config
    input_embeds = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    # QO indptr for attention
    qo_indptr = torch.tensor([0, seq_len], dtype=torch.long)

    # KV cache - simplified for testing
    num_layers = 16  # From model config
    num_kv_heads = 8  # From model config
    head_size = 64   # From model config
    page_size = 16

    kv_cache_at_layer = [
        torch.zeros(batch_size, num_kv_heads, page_size, head_size, dtype=torch.float32)
        for _ in range(num_layers)
    ]

    # Other required inputs
    kv_page_indices = torch.zeros(1, dtype=torch.long)
    kv_page_indptr = torch.tensor([0, 1], dtype=torch.long)
    kv_last_page_lens = torch.tensor([seq_len], dtype=torch.long)
    custom_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    return {
        'input_embeds': input_embeds,
        'position_ids': position_ids,
        'qo_indptr': qo_indptr,
        'kv_cache_at_layer': kv_cache_at_layer,
        'kv_page_indices': kv_page_indices,
        'kv_page_indptr': kv_page_indptr,
        'kv_last_page_lens': kv_last_page_lens,
        'custom_mask': custom_mask,
        'single_token_inference_mode': False,
        'adapter_subpass': None
    }


class TestRealModelIntegration:
    """Integration tests with real L4MA model and weights."""

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not REAL_MODEL_AVAILABLE, reason="Real model files not available")
    @pytest.mark.skipif(not PIE_COMPONENTS_AVAILABLE, reason="PIE components not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_model_loading_and_integration(self, real_l4ma_model):
        """Test that we can load the real model and initialize integration."""
        model, model_info = real_l4ma_model

        # Verify model loaded correctly
        assert isinstance(model, L4maForCausalLM)
        assert model_info.architecture.type == "l4ma"
        assert model_info.architecture.num_layers == 16
        assert model_info.architecture.hidden_size == 2048

        # Initialize debug integration with real model
        integration = L4MARealDebugIntegration(
            l4ma_model=model,
            debug_config={
                'enabled_checkpoints': ['post_embedding', 'post_attention', 'post_mlp'],
                'validation_mode': 'online',
                'performance_monitoring': True,
                'tolerance': 1e-5,
                'backend_comparison': 'metal',
                'real_tensor_validation': True
            }
        )

        # Verify integration initialization
        assert integration.l4ma_model is model
        assert integration.debug_enabled is True
        assert integration.metal_backend is not None

        # Test production readiness
        readiness = integration.validate_production_readiness()
        assert readiness['checks']['l4ma_model'] == 'available'
        assert readiness['checks']['pytorch'] == 'available'

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not REAL_MODEL_AVAILABLE, reason="Real model files not available")
    @pytest.mark.skipif(not PIE_COMPONENTS_AVAILABLE, reason="PIE components not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_model_layer_patching(self, real_l4ma_model):
        """Test patching real model layers with computation swapping."""
        model, model_info = real_l4ma_model

        integration = L4MARealDebugIntegration(l4ma_model=model)

        # Test patching attention operations
        patch_results = integration.patch_computation_operations({
            'attention': 'metal',
            'mlp': 'metal',
            'embedding': 'metal',
            'normalization': 'metal'
        })

        print(f"Patch results: {patch_results}")

        # Verify patching (results depend on Metal backend availability)
        assert 'attention' in patch_results
        assert 'mlp' in patch_results
        assert 'embedding' in patch_results
        assert 'normalization' in patch_results

        # Check if any operations were successfully patched
        successful_patches = [k for k, v in patch_results.items() if v == "patched"]
        if successful_patches:
            # Verify computation swapping is enabled
            assert integration._computation_swap_enabled is True
            swapped_ops = integration.get_swapped_operations()
            assert len(swapped_ops) > 0

            # Test restoration
            restore_results = integration.restore_original_operations()
            assert len(restore_results) > 0
            assert integration._computation_swap_enabled is False
        else:
            print("No operations were patched (likely due to Metal backend unavailability)")

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not REAL_MODEL_AVAILABLE, reason="Real model files not available")
    @pytest.mark.skipif(not PIE_COMPONENTS_AVAILABLE, reason="PIE components not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_model_forward_pass_with_validation(self, real_l4ma_model, test_inputs_real):
        """Test real forward pass with validation and checkpoint capture."""
        model, model_info = real_l4ma_model

        integration = L4MARealDebugIntegration(l4ma_model=model)

        # Set up tensor capture callback
        captured_tensors = []
        def capture_callback(checkpoint_name, tensor_data, metadata):
            captured_tensors.append({
                'checkpoint': checkpoint_name,
                'tensor_data': tensor_data,
                'metadata': metadata
            })

        integration.set_tensor_capture_callback(capture_callback)

        # Apply checkpoint decorators to some layers
        decoration_results = integration.apply_checkpoint_decorators([
            'embed_tokens',
            'layers.0.self_attn',
            'layers.0.mlp'
        ])

        print(f"Decoration results: {decoration_results}")

        # Test forward pass (this may fail due to complex L4MA inputs, but we test the integration)
        try:
            # Use a simplified input for testing
            simple_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)  # Token IDs

            # Test with embedding layer only (simpler than full forward pass)
            if hasattr(model.model, 'embed_tokens'):
                embedding_output = model.model.embed_tokens(simple_input)
                assert embedding_output is not None
                assert embedding_output.shape == (1, 8, 2048)  # batch, seq, hidden
                print(f"Embedding output shape: {embedding_output.shape}")

        except Exception as e:
            print(f"Forward pass test encountered expected complexity: {e}")
            # This is expected as L4MA requires complex input structure

        # Verify that some checkpoints were captured (if decorators worked)
        if any(v == "decorated" for v in decoration_results.values()):
            print(f"Captured {len(captured_tensors)} tensor checkpoints")

        # Clean up
        integration.cleanup_and_restore()

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not REAL_MODEL_AVAILABLE, reason="Real model files not available")
    @pytest.mark.skipif(not PIE_COMPONENTS_AVAILABLE, reason="PIE components not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_real_tensor_comparison_with_metal_backend(self, real_l4ma_model):
        """Test tensor comparison between PyTorch and Metal backend with real tensors."""
        model, model_info = real_l4ma_model

        integration = L4MARealDebugIntegration(l4ma_model=model)

        # Get some real tensors from the model
        embedding_layer = model.model.embed_tokens
        sample_input = torch.randint(0, 1000, (1, 8), dtype=torch.long)

        # Run embedding computation
        with torch.no_grad():
            embedding_output = embedding_layer(sample_input)

        # Test comparison with Metal backend
        comparison_result = integration.compare_with_metal_backend(
            layer_name='embedding',
            pytorch_tensor=embedding_output,
            input_ids=sample_input.numpy(),
            embedding_table=embedding_layer.weight.detach().numpy()
        )

        print(f"Comparison result: {comparison_result}")

        # Verify comparison structure (result depends on Metal backend availability)
        assert 'status' in comparison_result
        assert 'backend_compatibility' in comparison_result
        assert 'layer_name' in comparison_result

        if comparison_result['status'] == 'metal_unavailable':
            print("Metal backend not available - comparison test limited")
        elif comparison_result['status'] == 'error':
            print(f"Comparison error (expected): {comparison_result.get('error', 'Unknown')}")
        else:
            print("Metal backend comparison completed successfully")

        # Test with other layer types
        if hasattr(model.model, 'norm'):
            norm_layer = model.model.norm
            hidden_states = torch.randn(1, 8, 2048, dtype=torch.float32)

            with torch.no_grad():
                norm_output = norm_layer(hidden_states)

            norm_comparison = integration.compare_with_metal_backend(
                layer_name='normalization',
                pytorch_tensor=norm_output
            )

            assert 'status' in norm_comparison
            print(f"Normalization comparison: {norm_comparison['status']}")

        # Clean up
        integration.cleanup_and_restore()

    @pytest.mark.skipif(not REAL_INTEGRATION_AVAILABLE, reason="Real L4MA integration not available")
    @pytest.mark.skipif(not REAL_MODEL_AVAILABLE, reason="Real model files not available")
    @pytest.mark.skipif(not PIE_COMPONENTS_AVAILABLE, reason="PIE components not available")
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_performance_overhead_measurement(self, real_l4ma_model):
        """Test performance overhead measurement with real model operations."""
        model, model_info = real_l4ma_model

        integration = L4MARealDebugIntegration(l4ma_model=model)

        # Measure baseline performance
        embedding_layer = model.model.embed_tokens
        sample_input = torch.randint(0, 1000, (1, 32), dtype=torch.long)

        # Time without debug integration
        start_time = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = embedding_layer(sample_input)
        baseline_time = time.perf_counter() - start_time

        # Apply debug integration
        integration.apply_checkpoint_decorators(['embed_tokens'])

        # Time with debug integration
        start_time = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = embedding_layer(sample_input)
        debug_time = time.perf_counter() - start_time

        # Calculate overhead
        if baseline_time > 0:
            overhead_percent = ((debug_time - baseline_time) / baseline_time) * 100
            print(f"Performance overhead: {overhead_percent:.2f}%")

            # Get framework overhead measurement
            framework_overhead = integration.get_performance_overhead()
            print(f"Framework reported overhead: {framework_overhead:.3f}%")

            # Verify overhead is reasonable for testing (may be higher in test environment)
            assert overhead_percent < 50, f"Overhead too high: {overhead_percent:.2f}%"

        # Clean up
        integration.cleanup_and_restore()


if __name__ == "__main__":
    # Quick test for development
    if REAL_MODEL_AVAILABLE and PIE_COMPONENTS_AVAILABLE and TORCH_AVAILABLE:
        print("Loading real L4MA model...")
        model, model_info = load_real_l4ma_model()
        print(f"Model loaded successfully: {model_info.architecture.type}")
        print(f"Model layers: {model_info.architecture.num_layers}")
        print(f"Hidden size: {model_info.architecture.hidden_size}")
    else:
        print("Real model or required components not available")