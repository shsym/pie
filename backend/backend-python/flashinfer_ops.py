"""
FlashInfer Backend Operations

This module provides FlashInfer-specific operations implementation.
Only imports flashinfer when instantiated.
"""

from __future__ import annotations
import torch
from backend_ops import BackendOps


class FlashInferOps(BackendOps):
    """FlashInfer backend operations."""

    def __init__(self):
        super().__init__("flashinfer")
        try:
            import flashinfer as ops  # pylint: disable=import-outside-toplevel

            self.ops = ops
            self.available = True
            print("✅ Using FlashInfer for backend operations")
        except ImportError:
            self.ops = None
            self.available = False
            print("⚠️ FlashInfer not available - some operations may not work")

    def decode_image(
        self, image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for image decoding")
        return self.ops.image.decode_image(image_blob, dtype=dtype, device=device)  # type: ignore

    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for sampling")
        return self.ops.sampling.sampling_from_probs(probs)  # type: ignore

    def top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for top-p sampling")
        return self.ops.sampling.top_p_sampling_from_probs(probs, top_p=top_p)  # type: ignore

    def top_k_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for top-k sampling")
        return self.ops.sampling.top_k_sampling_from_probs(probs, top_k=top_k)  # type: ignore

    def min_p_sampling_from_probs(
        self, probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for min-p sampling")
        return self.ops.sampling.min_p_sampling_from_probs(probs, min_p=min_p)  # type: ignore

    def top_k_top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for combined sampling")
        return self.ops.sampling.top_k_top_p_sampling_from_probs(  # type: ignore
            probs, top_k=top_k, top_p=top_p
        )

    # Model operations
    def apply_llama31_rope_pos_ids_inplace(
        self, q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor
    ) -> None:
        """Apply RoPE using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for RoPE")
        self.ops.apply_llama31_rope_pos_ids_inplace(q=q, k=k, pos_ids=pos_ids)  # type: ignore

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
        kv_layout: str = "NHD"
    ) -> None:
        """Append KV cache using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for KV cache append")
        self.ops.append_paged_kv_cache(  # type: ignore
            append_key=append_key,
            append_value=append_value,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=paged_kv_cache,
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout=kv_layout
        )

    def get_batch_prefill_wrapper(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """Get FlashInfer prefill wrapper."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for prefill wrapper")
        return self.ops.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, kv_layout)  # type: ignore

    def get_batch_decode_wrapper(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        """Get FlashInfer decode wrapper."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for decode wrapper")
        return self.ops.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)  # type: ignore

    def get_seq_lens(
        self,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        page_size: int
    ) -> torch.Tensor:
        """Get sequence lengths using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for seq lens")
        return self.ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)  # type: ignore

    def get_batch_indices_positions(
        self,
        append_indptr: torch.Tensor,
        seq_lens: torch.Tensor,
        nnz: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get batch indices and positions using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for batch indices/positions")
        return self.ops.get_batch_indices_positions(append_indptr, seq_lens, nnz)  # type: ignore