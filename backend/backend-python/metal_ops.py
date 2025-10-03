"""
Metal Backend Operations - pie-metal implementation

This module provides Metal-accelerated operations using pie_metal.ops
for macOS with Apple Silicon.
"""

from __future__ import annotations
import torch
from backend_ops import BackendOps


class MetalOps(BackendOps):
    """Metal backend operations using pie_metal."""

    def __init__(self):
        super().__init__("pie-metal")
        try:
            import pie_metal.ops as ops  # pylint: disable=import-outside-toplevel

            self.ops = ops
            self.available = True
            print("✅ Using pie-metal for backend operations (Metal acceleration)")
        except ImportError as e:
            self.ops = None
            self.available = False
            raise RuntimeError(
                f"pie-metal is required on Apple Silicon but not available: {e}\n"
                "Install with: pip install ./pie-metal"
            ) from e

    def decode_image(
        self, image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image using pie-metal (not yet implemented)."""
        raise NotImplementedError("Image decoding not yet implemented in pie-metal")

    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for sampling")
        return self.ops.sampling.sampling_from_probs(probs)  # type: ignore

    def top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p sampling using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for top-p sampling")
        return self.ops.sampling.top_p_sampling_from_probs(probs, top_p=top_p)  # type: ignore

    def top_k_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling using pie-metal (not yet implemented)."""
        raise NotImplementedError("top_k_sampling not yet implemented in pie-metal")

    def min_p_sampling_from_probs(
        self, probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling using pie-metal (not yet implemented)."""
        raise NotImplementedError("min_p_sampling not yet implemented in pie-metal")

    def top_k_top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling using pie-metal (not yet implemented)."""
        raise NotImplementedError(
            "top_k_top_p_sampling not yet implemented in pie-metal"
        )

    # Model operations
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
        """Apply RoPE using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for RoPE")
        self.ops.apply_llama31_rope_pos_ids_inplace(
            q=q,
            k=k,
            pos_ids=pos_ids,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
        )  # type: ignore

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
        """Append KV cache using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for KV cache append")
        self.ops.append_paged_kv_cache(  # type: ignore
            append_key=append_key,
            append_value=append_value,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=paged_kv_cache,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout=kv_layout,
        )

    def BatchPrefillWithPagedKVCacheWrapper(
        self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ):
        """Create pie-metal prefill wrapper."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for prefill wrapper")
        # type: ignore
        return self.ops.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout)

    def BatchDecodeWithPagedKVCacheWrapper(
        self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ):
        """Create pie-metal decode wrapper."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for decode wrapper")
        # type: ignore
        return self.ops.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)

    def get_seq_lens(
        self,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        """Get sequence lengths using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for seq lens")
        return self.ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)  # type: ignore

    def get_batch_indices_positions(
        self, append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch indices and positions using pie-metal."""
        if not self.available or self.ops is None:
            raise RuntimeError("pie-metal not available for batch indices/positions")
        return self.ops.get_batch_indices_positions(append_indptr, seq_lens, nnz)  # type: ignore
