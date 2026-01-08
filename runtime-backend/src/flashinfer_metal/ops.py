"""
FlashInfer-compatible API for Metal kernels.

Provides drop-in replacement for FlashInfer operations using
Metal acceleration on macOS with Apple Silicon.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from .attention import AttentionCompiler
from .config import IS_APPLE_SILICON, MPS_DEVICE_AVAILABLE
from .kv_cache import AppendKVCacheCompiler
from .rope import RoPECompiler
from .utils import validate_mps_device, validate_page_size


class MPSShaderCompiler:
    """Unified facade for all MPS Metal kernel operations (Singleton)."""

    _instance: Optional["MPSShaderCompiler"] = None

    def __new__(cls, page_size: int = 16):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, page_size: int = 16):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.page_size = page_size
        self.attention_compiler = AttentionCompiler(page_size=page_size)
        self.rope_compiler = RoPECompiler()
        self.append_kv_cache_compiler = AppendKVCacheCompiler()

    @property
    def compiled_libraries(self):
        """Unified view of all compiled libraries."""
        libraries = {}
        libraries.update(self.attention_compiler.compiled_libraries)
        libraries.update(self.rope_compiler.compiled_libraries)
        libraries.update(self.append_kv_cache_compiler.compiled_libraries)
        return libraries

    def can_use_mps_kernels(self) -> bool:
        """Check if we can use compiled MPS kernels."""
        return (
            self.attention_compiler.can_use_mps_kernels()
            or self.rope_compiler.can_use_mps_kernels()
            or self.append_kv_cache_compiler.can_use_mps_kernels()
        )

    def run_attention_mps(
        self,
        query,
        kv_cache,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        qo_indptr,
        custom_mask=None,
    ):
        return self.attention_compiler.run_attention_mps(
            query,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            qo_indptr,
            custom_mask,
        )

    def run_rope_mps(
        self,
        input_qk,
        position_ids,
        rope_theta=10000.0,
        rope_factor=1.0,
        interleaved=False,
    ):
        self.rope_compiler.run_rope_mps(
            input_qk, position_ids, rope_theta, rope_factor, interleaved
        )

    def run_append_paged_kv_cache_mps(
        self,
        k_input,
        v_input,
        paged_kv_cache,
        kv_batch_indices,
        kv_positions,
        kv_page_indices,
        kv_page_indptr,
        kv_last_page_lens,
        num_kv_heads,
        head_size,
        page_size,
    ):
        self.append_kv_cache_compiler.run_append_paged_kv_cache_mps(
            k_input,
            v_input,
            paged_kv_cache,
            kv_batch_indices,
            kv_positions,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            num_kv_heads,
            head_size,
            page_size,
        )


def get_mps_compiler(page_size: int = 16) -> MPSShaderCompiler:
    """Get or create the singleton MPS shader compiler instance."""
    return MPSShaderCompiler(page_size=page_size)


def _initialize_mps_backend(page_size: int = 16) -> bool:
    """Initialize MPS shader backend."""
    try:
        validate_page_size(page_size)
    except ValueError as e:
        print(f"❌ Invalid page_size: {e}")
        sys.exit(1)

    if not IS_APPLE_SILICON:
        print("❌ flashinfer_metal requires macOS with Apple Silicon")
        return False

    if not MPS_DEVICE_AVAILABLE:
        print("❌ PyTorch MPS backend is not available")
        sys.exit(1)

    try:
        test_tensor = torch.tensor([1.0], device="mps")
        del test_tensor
    except (RuntimeError, OSError) as e:
        print(f"❌ Cannot create tensors on MPS device: {e}")
        sys.exit(1)

    torch.set_default_device("mps")

    try:
        compiler = get_mps_compiler(page_size=page_size)
        if compiler.can_use_mps_kernels():
            return True
    except (RuntimeError, ImportError, AttributeError) as e:
        print(f"❌ MPS shader initialization failed: {e}")

    sys.exit(1)


# Initialize on import
_PAGE_SIZE_FROM_ENV = int(os.environ.get("PIE_METAL_PAGE_SIZE", "16"))
_initialize_mps_backend(page_size=_PAGE_SIZE_FROM_ENV)


class BatchPrefillWithPagedKVCacheWrapper:
    """Drop-in replacement for flashinfer.BatchPrefillWithPagedKVCacheWrapper."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False

    def plan(
        self,
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
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        validate_mps_device(qo_indptr, "qo_indptr")
        validate_mps_device(paged_kv_indptr, "paged_kv_indptr")
        validate_mps_device(paged_kv_indices, "paged_kv_indices")
        validate_mps_device(paged_kv_last_page_len, "paged_kv_last_page_len")
        if custom_mask is not None:
            validate_mps_device(custom_mask, "custom_mask")

        self._planned_params = {
            "qo_indptr": qo_indptr,
            "kv_page_indptr": paged_kv_indptr,
            "kv_page_indices": paged_kv_indices,
            "kv_last_page_lens": paged_kv_last_page_len,
            "num_query_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_dim_qk,
            "page_size": page_size,
            "custom_mask": custom_mask,
            "q_data_type": q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        validate_mps_device(query, "query")
        validate_mps_device(kv_cache, "kv_cache")

        assert self._planned_params is not None
        return get_mps_compiler().run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params["kv_page_indices"],
            kv_page_indptr=self._planned_params["kv_page_indptr"],
            kv_last_page_lens=self._planned_params["kv_last_page_lens"],
            qo_indptr=self._planned_params["qo_indptr"],
            custom_mask=self._planned_params["custom_mask"],
        )


class BatchDecodeWithPagedKVCacheWrapper:
    """Drop-in replacement for flashinfer.BatchDecodeWithPagedKVCacheWrapper."""

    def __init__(self, workspace_buffer: torch.Tensor, kv_layout: str = "NHD"):
        self.workspace_buffer = workspace_buffer
        self.kv_layout = kv_layout
        self._planned_params: Optional[Dict[str, Any]] = None
        self._is_planned = False

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str = "NONE",
        q_data_type: torch.dtype = torch.float16,
    ) -> None:
        validate_mps_device(indptr, "indptr")
        validate_mps_device(indices, "indices")
        validate_mps_device(last_page_len, "last_page_len")

        self._planned_params = {
            "kv_page_indptr": indptr,
            "kv_page_indices": indices,
            "kv_last_page_lens": last_page_len,
            "num_query_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_dim,
            "page_size": page_size,
            "q_data_type": q_data_type,
        }
        self._is_planned = True

    def run(self, query: torch.Tensor, kv_cache: torch.Tensor) -> torch.Tensor:
        if not self._is_planned:
            raise RuntimeError("Must call plan() before run()")

        validate_mps_device(query, "query")
        validate_mps_device(kv_cache, "kv_cache")

        assert self._planned_params is not None
        batch_size = self._planned_params["kv_page_indptr"].shape[0] - 1
        qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)

        return get_mps_compiler().run_attention_mps(
            query=query,
            kv_cache=kv_cache,
            kv_page_indices=self._planned_params["kv_page_indices"],
            kv_page_indptr=self._planned_params["kv_page_indptr"],
            kv_last_page_lens=self._planned_params["kv_last_page_lens"],
            qo_indptr=qo_indptr,
            custom_mask=None,
        )


def apply_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rope_theta: float = 10000.0,
    interleave: bool = False,
) -> None:
    """Apply standard RoPE encoding in-place using Metal kernels."""
    validate_mps_device(q, "q")
    validate_mps_device(k, "k")
    validate_mps_device(pos_ids, "pos_ids")

    compiler = get_mps_compiler()
    compiler.run_rope_mps(
        q, pos_ids, rope_theta=rope_theta, rope_factor=1.0, interleaved=interleave
    )
    compiler.run_rope_mps(
        k, pos_ids, rope_theta=rope_theta, rope_factor=1.0, interleaved=interleave
    )


def apply_llama31_rope_pos_ids_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
    interleave: bool = False,
    rope_scale: float = 32.0,
    rope_theta: float = 500000.0,
    low_freq_factor: float = 1.0,
    high_freq_factor: float = 4.0,
    old_context_len: int = 8192,
) -> None:
    """Apply LLaMA 3.1-style RoPE encoding in-place using Metal kernels."""
    if rotary_dim is not None:
        raise ValueError("rotary_dim not supported in Metal RoPE")
    if low_freq_factor != 1.0:
        raise ValueError("low_freq_factor not supported in Metal RoPE")
    if high_freq_factor != 4.0:
        raise ValueError("high_freq_factor not supported in Metal RoPE")
    if old_context_len != 8192:
        raise ValueError("old_context_len not supported in Metal RoPE")

    validate_mps_device(q, "q")
    validate_mps_device(k, "k")
    validate_mps_device(pos_ids, "pos_ids")

    compiler = get_mps_compiler()
    compiler.run_rope_mps(
        q,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave,
    )
    compiler.run_rope_mps(
        k,
        pos_ids,
        rope_theta=rope_theta,
        rope_factor=rope_scale,
        interleaved=interleave,
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
    kv_layout: str = "NHD",
) -> None:
    """Append key-value states to paged KV cache using Metal kernels."""
    validate_mps_device(append_key, "append_key")
    validate_mps_device(append_value, "append_value")
    validate_mps_device(batch_indices, "batch_indices")
    validate_mps_device(positions, "positions")
    validate_mps_device(paged_kv_cache, "paged_kv_cache")
    validate_mps_device(kv_indices, "kv_indices")
    validate_mps_device(kv_indptr, "kv_indptr")
    validate_mps_device(kv_last_page_len, "kv_last_page_len")

    num_tokens, num_kv_heads, head_dim = append_key.shape
    _num_pages, _, page_size, _, _ = paged_kv_cache.shape

    k_flat = append_key.contiguous().reshape(num_tokens, num_kv_heads * head_dim)
    v_flat = append_value.contiguous().reshape(num_tokens, num_kv_heads * head_dim)
    paged_kv_unified = paged_kv_cache.view(-1)

    get_mps_compiler().run_append_paged_kv_cache_mps(
        k_flat,
        v_flat,
        paged_kv_unified,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        num_kv_heads,
        head_dim,
        page_size,
    )


def get_seq_lens(
    kv_page_indptr: torch.Tensor, kv_last_page_lens: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Calculate sequence lengths from paging metadata."""
    validate_mps_device(kv_page_indptr, "kv_page_indptr")
    validate_mps_device(kv_last_page_lens, "kv_last_page_lens")

    batch_size = kv_page_indptr.shape[0] - 1
    seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=kv_page_indptr.device)

    for i in range(batch_size):
        num_pages = kv_page_indptr[i + 1] - kv_page_indptr[i]
        if num_pages > 0:
            seq_lens[i] = (num_pages - 1) * page_size + kv_last_page_lens[i]

    return seq_lens


def get_batch_indices_positions(
    append_indptr: torch.Tensor, seq_lens: torch.Tensor, nnz: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get batch indices and positions for tokens."""
    validate_mps_device(append_indptr, "append_indptr")
    validate_mps_device(seq_lens, "seq_lens")

    device = append_indptr.device

    batch_indices = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        batch_indices[start_idx:end_idx] = batch_idx

    positions = torch.empty(nnz, dtype=torch.int32, device=device)
    for batch_idx in range(append_indptr.numel() - 1):
        start_idx = int(append_indptr[batch_idx].item())
        end_idx = int(append_indptr[batch_idx + 1].item())
        num_new = end_idx - start_idx
        seq_len = int(seq_lens[batch_idx].item())
        pos_start = seq_len - num_new
        positions[start_idx:end_idx] = torch.arange(
            pos_start, seq_len, dtype=torch.int32, device=device
        )

    return batch_indices, positions


class sampling:
    """Basic sampling operations (PyTorch implementations)."""

    @staticmethod
    def sampling_from_probs(probs: torch.Tensor) -> torch.Tensor:
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_p_sampling_from_probs(
        probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p.unsqueeze(-1)
        mask[..., 0] = True
        sorted_probs[~mask] = 0
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        return torch.gather(
            sorted_indices, -1, sampled_sorted_idx.unsqueeze(-1)
        ).squeeze(-1)

    @staticmethod
    def top_k_sampling_from_probs(
        probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling: sample from the top k tokens."""
        # Get the maximum k value (handles per-sample k values)
        k = int(top_k.max().item()) if top_k.numel() > 0 else 50
        k = min(k, probs.shape[-1])  # Clamp to vocab size

        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)
        # Renormalize probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Sample from top-k
        sampled_indices = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
        return topk_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def min_p_sampling_from_probs(
        probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling: filter tokens below min_p * max_prob."""
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = min_p.unsqueeze(-1) * max_probs
        filtered_probs = probs.clone()
        filtered_probs[probs < threshold] = 0
        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)

    @staticmethod
    def top_k_top_p_sampling_from_probs(
        probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling."""
        # First apply top-k
        k = int(top_k.max().item()) if top_k.numel() > 0 else 50
        k = min(k, probs.shape[-1])

        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        # Then apply top-p on the top-k subset
        sorted_probs, sorted_order = torch.sort(topk_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p.unsqueeze(-1)
        mask[..., 0] = True
        sorted_probs[~mask] = 0

        # Sample from filtered distribution
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        # Map back through the sorting to get top-k indices
        sampled_topk_idx = sorted_order.gather(
            -1, sampled_sorted_idx.unsqueeze(-1)
        ).squeeze(-1)
        # Map back through top-k to get original vocab indices
        return topk_indices.gather(-1, sampled_topk_idx.unsqueeze(-1)).squeeze(-1)


__all__ = [
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchDecodeWithPagedKVCacheWrapper",
    "apply_rope_pos_ids_inplace",
    "apply_llama31_rope_pos_ids_inplace",
    "append_paged_kv_cache",
    "get_seq_lens",
    "get_batch_indices_positions",
    "get_mps_compiler",
    "sampling",
]
