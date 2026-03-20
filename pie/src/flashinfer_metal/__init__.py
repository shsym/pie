"""
flashinfer_metal: Metal-accelerated FlashInfer replacement for macOS

A drop-in replacement for FlashInfer that uses Apple's Metal Performance Shaders
for attention operations on macOS with Apple Silicon.
"""

__version__ = "0.2.0"
__author__ = "Pie Team"

# Import main API for convenience
try:
    from . import ops
    from .ops import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        append_paged_kv_cache,
        apply_llama31_rope_pos_ids_inplace,
        apply_rope_pos_ids_inplace,
        get_batch_indices_positions,
        get_mps_compiler,
        get_seq_lens,
        sampling,
    )

    __all__ = [
        "ops",
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
except ImportError:
    # Metal backend not available
    __all__ = []
