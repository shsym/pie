"""
FlashInfer-compatible API for pie-metal.

This module provides a drop-in replacement for FlashInfer operations using
Metal acceleration on macOS with Apple Silicon. When Metal is not available,
it raises informative errors directing users to use PyTorch implementations.
"""

import platform
import sys
from typing import Tuple, Dict, Any, Optional
import torch

# Detect hardware capabilities
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.processor() == "arm"

# Global MPS shader compiler instance (initialized lazily)
_mps_compiler = None
_mps_available = False


def _validate_mps_device(tensor: torch.Tensor, name: str) -> None:
    """Validate that a tensor is on MPS device.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Raises:
        RuntimeError: If tensor is not on MPS device
    """
    if tensor.device.type != 'mps':
        raise RuntimeError(
            f"pie-metal requires all tensors to be on MPS device. "
            f"Tensor '{name}' is on {tensor.device}. "
            f"Please move all tensors to MPS before calling pie-metal operations:\n"
            f"  tensor = tensor.to('mps')\n"
            f"Or set config.device='mps' when initializing the model."
        )


def _initialize_mps_backend() -> bool:
    """Initialize MPS shader backend - returns success status"""
    global _mps_compiler, _mps_available

    if not IS_APPLE_SILICON:
        print("❌ pie-metal requires macOS with Apple Silicon (M1/M2/M3)")
        print("   Install FlashInfer for other platforms:")
        print("   pip install flashinfer")
        return False

    # Validate MPS is available in PyTorch
    if not torch.backends.mps.is_available():
        print("❌ PyTorch MPS backend is not available")
        print("   Please ensure you have PyTorch 2.0+ with MPS support:")
        print("   pip install torch>=2.0.0")
        sys.exit(1)

    # Validate MPS device can be created
    try:
        test_tensor = torch.tensor([1.0], device='mps')
        del test_tensor
    except Exception as e:
        print(f"❌ Cannot create tensors on MPS device: {e}")
        print("   Please check your PyTorch MPS installation")
        sys.exit(1)

    # Set default device to MPS for all PyTorch operations
    torch.set_default_device('mps')
    print("✅ Set default PyTorch device to 'mps'")
    print("   All tensors will be created on MPS device by default")

    try:
        # Try MPS shader approach first (preferred)
        from ._internal.mps_shader_integration import get_mps_compiler
        _mps_compiler = get_mps_compiler()
        _mps_available = _mps_compiler.can_use_mps_kernels()

        if _mps_available:
            print("✅ pie-metal initialized with PyTorch MPS shader compilation")
            print(f"   Available shader libraries: {list(_mps_compiler.compiled_libraries.keys())}")
            return True
        else:
            print("❌ MPS shader compilation not available")

    except Exception as e:
        print(f"❌ MPS shader initialization failed: {e}")

    # MPS initialization failed
    print("❌ MPS shader compilation failed")
    print("   Please check that:")
    print("   1. You have PyTorch 2.0+ with MPS support")
    print("   2. Xcode Command Line Tools are installed")
    print("   3. Your system supports Metal compute")
    print("\n   Install FlashInfer as an alternative:")
    print("   pip install flashinfer")
    sys.exit(1)


# Initialize on import
_backend_available = _initialize_mps_backend()




class BatchPrefillWithPagedKVCacheWrapper:
    """
    Drop-in replacement for flashinfer.BatchPrefillWithPagedKVCacheWrapper

    Handles prefill attention operations (processing multiple tokens per sequence).
    Plans operation parameters and executes attention using Metal kernels.
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """
        Initialize prefill wrapper

        Args:
            workspace_buffer: Workspace memory (for FlashInfer compatibility, unused in Metal)
            kv_layout: KV cache layout, either "NHD" or "HND" (default: "NHD")
        """
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False

    def plan(self,
             qo_indptr: torch.Tensor,
             paged_kv_indptr: torch.Tensor,
             paged_kv_indices: torch.Tensor,
             paged_kv_last_page_len: torch.Tensor,
             num_qo_heads: int,
             num_kv_heads: int,
             head_dim_qk: int,
             page_size: int,
             pos_encoding_mode: str = "NONE",
             custom_mask: Optional[torch.Tensor] = None,
             q_data_type: torch.dtype = torch.float16) -> None:
        """
        Plan the prefill attention operation

        Stores all parameters needed for attention computation. Must be called before run().

        Args:
            qo_indptr: Query-output indptr tensor [batch_size + 1]
            paged_kv_indptr: KV page indptr tensor [batch_size + 1]
            paged_kv_indices: KV page indices tensor [num_pages]
            paged_kv_last_page_len: Last page lengths [batch_size]
            num_qo_heads: Number of query heads
            num_kv_heads: Number of key-value heads
            head_dim_qk: Head dimension for query/key
            page_size: KV cache page size
            pos_encoding_mode: Position encoding mode (unused, for compatibility)
            custom_mask: Optional attention mask
            q_data_type: Query tensor data type
        """
        # Validate all input tensors are on MPS device
        _validate_mps_device(qo_indptr, "qo_indptr")
        _validate_mps_device(paged_kv_indptr, "paged_kv_indptr")
        _validate_mps_device(paged_kv_indices, "paged_kv_indices")
        _validate_mps_device(paged_kv_last_page_len, "paged_kv_last_page_len")
        if custom_mask is not None:
            _validate_mps_device(custom_mask, "custom_mask")

        self._planned_params = {
            'qo_indptr': qo_indptr,
            'kv_page_indptr': paged_kv_indptr,
            'kv_page_indices': paged_kv_indices,
            'kv_last_page_lens': paged_kv_last_page_len,
            'num_query_heads': num_qo_heads,
            'num_kv_heads': num_kv_heads,
            'head_size': head_dim_qk,
            'page_size': page_size,
            'custom_mask': custom_mask,
            'q_data_type': q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """
        Execute prefill attention computation

        Args:
            query: Query tensor [num_tokens, num_query_heads, head_dim]
            kv_cache: Paged KV cache tensor [num_pages, 2, page_size, num_kv_heads, head_dim]

        Returns:
            Attention output tensor [num_tokens, num_query_heads * head_dim]
        """
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        # Validate input tensors are on MPS device
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        return self._run_metal(query, kv_cache)

    def _run_metal(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """Execute using MPS shader kernels"""
        assert self._planned_params is not None, "plan() must be called before run()"

        if not _mps_available or _mps_compiler is None:
            raise RuntimeError("MPS backend not initialized - pie-metal requires MPS support")

        return _mps_compiler.run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params['kv_page_indices'],
            kv_page_indptr=self._planned_params['kv_page_indptr'],
            kv_last_page_lens=self._planned_params['kv_last_page_lens'],
            qo_indptr=self._planned_params['qo_indptr'],
            custom_mask=self._planned_params['custom_mask'],
        )


class BatchDecodeWithPagedKVCacheWrapper:
    """
    Drop-in replacement for flashinfer.BatchDecodeWithPagedKVCacheWrapper

    Handles decode attention operations (single token generation per sequence).
    """

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """Initialize decode wrapper - similar to prefill wrapper"""
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False

    def plan(self,
             indptr: torch.Tensor,
             indices: torch.Tensor,
             last_page_len: torch.Tensor,
             num_qo_heads: int,
             num_kv_heads: int,
             head_dim: int,
             page_size: int,
             pos_encoding_mode: str = "NONE",
             q_data_type: torch.dtype = torch.float16) -> None:
        """Plan decode operation - simpler than prefill (single token per sequence)"""
        # Validate all input tensors are on MPS device
        _validate_mps_device(indptr, "indptr")
        _validate_mps_device(indices, "indices")
        _validate_mps_device(last_page_len, "last_page_len")

        self._planned_params = {
            'kv_page_indptr': indptr,
            'kv_page_indices': indices,
            'kv_last_page_lens': last_page_len,
            'num_query_heads': num_qo_heads,
            'num_kv_heads': num_kv_heads,
            'head_size': head_dim,
            'page_size': page_size,
            'q_data_type': q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        """Execute decode attention - create qo_indptr for single tokens"""
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        # Validate input tensors are on MPS device
        _validate_mps_device(query, "query")
        _validate_mps_device(kv_cache, "kv_cache")

        # For decode, we have one token per batch element
        assert self._planned_params is not None, "plan() must be called before run()"

        if not _mps_available or _mps_compiler is None:
            raise RuntimeError("MPS backend not initialized - pie-metal requires MPS support")

        batch_size = self._planned_params['kv_page_indptr'].shape[0] - 1
        qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)

        return _mps_compiler.run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params['kv_page_indices'],
            kv_page_indptr=self._planned_params['kv_page_indptr'],
            kv_last_page_lens=self._planned_params['kv_last_page_lens'],
            qo_indptr=qo_indptr,
            custom_mask=None,
        )


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 8.0,
    rope_theta: float = 500000.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> None:
    """
    Apply LLaMA 3.1-style RoPE encoding in-place using Metal kernels.

    This is a drop-in replacement for FlashInfer's apply_llama31_rope_pos_ids_inplace.
    cos/sin values are computed on the fly inside the kernel.

    Parameters:
        q: Query ragged tensor, shape: (nnz, num_q_heads, head_dim) - modified in-place
        k: Key ragged tensor, shape: (nnz, num_k_heads, head_dim) - modified in-place
        pos_ids: Position indices, shape: (nnz)
        rotary_dim: The dimensions to apply RoPE. If None, apply to entire head dimension.
        interleave: Whether to use interleaved layout. If True, rotate even/odd dims separately.
                   If False, rotate first half and second half separately.
        rope_scale: The scaling factor used in rope embedding (default: 8.0)
        rope_theta: The theta value used in rope embedding (default: 500000.0)
        low_freq_factor: The low frequency factor used in Llama 3.1 RoPE (default: 1.0)
        high_freq_factor: The high frequency factor used in Llama 3.1 RoPE (default: 4.0)
        old_context_len: The old context length used in Llama 3.1 RoPE (default: 8192)

    Raises:
        ValueError: If unsupported parameters have non-default values
    """
    # Validate that unsupported parameters are using default values
    if rotary_dim is not None:
        raise ValueError(
            f"rotary_dim parameter is not supported in Metal RoPE implementation. "
            f"Got rotary_dim={rotary_dim}, expected None."
        )
    if low_freq_factor != 1.0:
        raise ValueError(
            f"low_freq_factor parameter is not supported in Metal RoPE implementation. "
            f"Got low_freq_factor={low_freq_factor}, expected 1.0."
        )
    if high_freq_factor != 4.0:
        raise ValueError(
            f"high_freq_factor parameter is not supported in Metal RoPE implementation. "
            f"Got high_freq_factor={high_freq_factor}, expected 4.0."
        )
    if old_context_len != 8192:
        raise ValueError(
            f"old_context_len parameter is not supported in Metal RoPE implementation. "
            f"Got old_context_len={old_context_len}, expected 8192."
        )

    # Verify Metal kernels are available
    if not _mps_available or _mps_compiler is None or 'rope' not in _mps_compiler.compiled_libraries:
        raise RuntimeError("Metal RoPE kernels not available. pie-metal requires MPS support.")

    # Validate all tensors are on MPS device
    _validate_mps_device(q, "q")
    _validate_mps_device(k, "k")
    _validate_mps_device(pos_ids, "pos_ids")

    # Apply RoPE to both query and key tensors using Metal kernels
    _mps_compiler.run_rope_mps(
        q, pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave
    )
    _mps_compiler.run_rope_mps(
        k, pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave
    )


def append_paged_kv_cache(
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    kv_layout: str = "NHD"
) -> None:
    """
    Append key-value states to paged KV cache using Metal kernels

    Args:
        append_key: Key states to append [num_tokens, num_kv_heads, head_dim]
        append_value: Value states to append [num_tokens, num_kv_heads, head_dim]
        batch_indices: Batch index for each token [num_tokens]
        positions: Position within sequence for each token [num_tokens]
        paged_kv_cache: Paged KV cache [num_pages, 2, page_size, num_kv_heads, head_dim]
        kv_indices: Page indices [num_pages]
        kv_indptr: Page indptr [batch_size + 1]
        kv_last_page_len: Last page lengths [batch_size]
        kv_layout: Layout string (unused, for compatibility)
    """
    # Validate all input tensors are on MPS device
    _validate_mps_device(append_key, "append_key")
    _validate_mps_device(append_value, "append_value")
    _validate_mps_device(batch_indices, "batch_indices")
    _validate_mps_device(positions, "positions")
    _validate_mps_device(paged_kv_cache, "paged_kv_cache")
    _validate_mps_device(kv_indices, "kv_indices")
    _validate_mps_device(kv_indptr, "kv_indptr")
    _validate_mps_device(kv_last_page_len, "kv_last_page_len")

    try:
        # Try to use Metal append_kv_cache kernel via MPSShaderCompiler
        from ._internal.mps_shader_integration import get_mps_compiler

        compiler = get_mps_compiler()
        if compiler.can_use_mps_kernels() and 'append_kv_cache' in compiler.compiled_libraries:
            # Extract dimensions
            num_tokens, num_kv_heads, head_dim = append_key.shape

            # Reshape inputs for Metal kernel
            # Metal expects [num_tokens, num_kv_heads * head_dim] for key/value
            k_flat = append_key.reshape(num_tokens, num_kv_heads * head_dim)
            v_flat = append_value.reshape(num_tokens, num_kv_heads * head_dim)

            # Split paged_kv_cache into K and V caches
            # Expected shape: [num_pages, 2, page_size, num_kv_heads, head_dim]
            paged_k = paged_kv_cache[:, 0, :, :, :].reshape(paged_kv_cache.shape[0], paged_kv_cache.shape[2], -1)
            paged_v = paged_kv_cache[:, 1, :, :, :].reshape(paged_kv_cache.shape[0], paged_kv_cache.shape[2], -1)

            compiler.run_append_paged_kv_cache_mps(
                k_flat, v_flat, paged_k, paged_v,
                batch_indices, positions, kv_indices, kv_indptr, kv_last_page_len,
                num_kv_heads=num_kv_heads,
                head_size=head_dim
            )
            return
    except Exception as e:
        # Re-raise to expose the issue - NO FALLBACK for Metal operations
        raise RuntimeError(f"Metal append_paged_kv_cache failed: {e}") from e

    # NOTE: PyTorch fallback removed - we must use Metal kernels
    # If you reach here, Metal kernel was not available
    raise RuntimeError(
        "Metal append_paged_kv_cache kernel not available. "
        "pie-metal requires Metal support on Apple Silicon."
    )


def get_seq_lens(
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    page_size: int
) -> torch.Tensor:
    """Calculate sequence lengths from paging metadata"""
    # Validate input tensors are on MPS device
    _validate_mps_device(kv_page_indptr, "kv_page_indptr")
    _validate_mps_device(kv_last_page_lens, "kv_last_page_lens")

    batch_size = kv_page_indptr.shape[0] - 1
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=kv_page_indptr.device)

    for i in range(batch_size):
        num_pages = kv_page_indptr[i + 1] - kv_page_indptr[i]
        if num_pages > 0:
            seq_lens[i] = (num_pages - 1) * page_size + kv_last_page_lens[i]

    return seq_lens


def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get batch indices and positions for tokens"""
    # Validate input tensors are on MPS device
    _validate_mps_device(append_indptr, "append_indptr")
    _validate_mps_device(seq_lens, "seq_lens")

    device = append_indptr.device

    # Compute batch indices (self-contained implementation)
    batch_indices = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        batch_indices[start_idx:end_idx] = batch_idx

    # Compute positions
    positions = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        positions[start_idx:end_idx] = torch.arange(
            int(seq_lens[batch_idx].item()) - (end_idx - start_idx),
            int(seq_lens[batch_idx].item()),
            dtype=torch.int32,
            device=device
        )

    return batch_indices, positions


# Optional: Basic sampling operations for completeness
class sampling:
    """Sampling operations namespace (basic PyTorch implementations)"""

    @staticmethod
    def sampling_from_probs(probs: torch.Tensor) -> torch.Tensor:
        """Sample tokens from probability distribution"""
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_p_sampling_from_probs(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
        """Top-p (nucleus) sampling from probability distribution"""
        # Simple implementation - can be enhanced
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for tokens to keep
        mask = cumulative_probs <= top_p.unsqueeze(-1)
        mask[..., 0] = True  # Always keep at least one token

        # Zero out tokens beyond threshold
        sorted_probs[~mask] = 0

        # Sample from filtered distribution
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)

        # Map back to original token indices
        return torch.gather(sorted_indices, -1, sampled_sorted_idx.unsqueeze(-1)).squeeze(-1)


# Export all public functions and classes
__all__ = [
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchDecodeWithPagedKVCacheWrapper",
    "apply_llama31_rope_pos_ids_inplace",
    "append_paged_kv_cache",
    "get_seq_lens",
    "get_batch_indices_positions",
    "sampling",
]
