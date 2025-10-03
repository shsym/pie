"""
Backend Operations - Unified abstraction layer for compute backends

This module provides:
1. BackendOps abstract base class defining the interface
2. get_backend_ops() factory function for automatic backend selection
3. Unified 'ops' instance for direct import by models

Backends are selected based on platform:
- Apple Silicon: MetalOps (pie-metal)
- Other platforms: FlashInferOps (flashinfer)
"""

from __future__ import annotations
import torch
from platform_detection import is_apple_silicon


class BackendOps:
    """Abstract base class for backend operations."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        self.available = False

    # Image operations
    def decode_image(
        self, image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image from bytes to tensor."""
        raise NotImplementedError(
            f"decode_image not implemented for {self.backend_name}"
        )

    # Sampling operations
    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample tokens from probability distribution."""
        raise NotImplementedError(
            f"sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling from probability distribution."""
        raise NotImplementedError(
            f"top_p_sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_k_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling from probability distribution."""
        raise NotImplementedError(
            f"top_k_sampling_from_probs not implemented for {self.backend_name}"
        )

    def min_p_sampling_from_probs(
        self, probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling from probability distribution."""
        raise NotImplementedError(
            f"min_p_sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_k_top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling from probability distribution."""
        raise NotImplementedError(
            f"top_k_top_p_sampling_from_probs not implemented for {self.backend_name}"
        )

    # Model operations (RoPE, KV cache, attention wrappers)
    def apply_llama31_rope_pos_ids_inplace(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
        rope_scale: float = 8.0,
        rope_theta: float = 500000.0,
        low_freq_factor: float = 1.0,
        high_freq_factor: float = 4.0,
    ) -> None:
        """Apply LLaMA-style RoPE encoding in-place."""
        raise NotImplementedError(
            f"apply_llama31_rope_pos_ids_inplace not implemented for {self.backend_name}"
        )

    def append_paged_kv_cache(
        self,
        append_key: torch.Tensor,
        append_value: torch.Tensor,
        batch_indices: torch.Tensor,
        positions: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_last_page_len: torch.Tensor,
        kv_layout: str = "NHD",
    ) -> None:
        """Append key-value states to paged KV cache."""
        raise NotImplementedError(
            f"append_paged_kv_cache not implemented for {self.backend_name}"
        )

    def BatchPrefillWithPagedKVCacheWrapper(  # pylint: disable=invalid-name
        self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ):
        """Create BatchPrefillWithPagedKVCacheWrapper instance."""
        raise NotImplementedError(
            f"BatchPrefillWithPagedKVCacheWrapper not implemented for {self.backend_name}"
        )

    def BatchDecodeWithPagedKVCacheWrapper(  # pylint: disable=invalid-name
        self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ):
        """Create BatchDecodeWithPagedKVCacheWrapper instance."""
        raise NotImplementedError(
            f"BatchDecodeWithPagedKVCacheWrapper not implemented for {self.backend_name}"
        )

    def get_seq_lens(
        self,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """Calculate sequence lengths from paging metadata."""
        raise NotImplementedError(
            f"get_seq_lens not implemented for {self.backend_name}"
        )

    def get_batch_indices_positions(
        self, append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch indices and positions for tokens."""
        raise NotImplementedError(
            f"get_batch_indices_positions not implemented for {self.backend_name}"
        )


def get_backend_ops() -> BackendOps:
    """Factory function to get appropriate backend operations.

    Returns:
        BackendOps instance (MetalOps or FlashInferOps)

    Raises:
        RuntimeError: If required backend is not available
    """
    if is_apple_silicon():
        from metal_ops import MetalOps  # pylint: disable=import-outside-toplevel

        return MetalOps()

    from flashinfer_ops import FlashInferOps  # pylint: disable=import-outside-toplevel

    return FlashInferOps()


# Create global ops instance for direct import by models
# This replaces unified_ops.py
ops = get_backend_ops()

__all__ = ["BackendOps", "get_backend_ops", "ops"]
