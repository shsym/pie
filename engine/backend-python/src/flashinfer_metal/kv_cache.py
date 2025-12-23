"""
Metal append_paged_kv_cache operation implementation.

Compiles and executes Metal append KV cache kernels for PyTorch MPS backend.
"""

import torch

from .config import MPS_COMPILE_AVAILABLE
from .shader_compiler import BaseShaderCompiler


class AppendKVCacheCompiler(BaseShaderCompiler):
    """Compiles and runs Metal append_paged_kv_cache kernels."""

    def __init__(self):
        super().__init__()
        self._compile_append_kv_cache_kernels()

    def _compile_append_kv_cache_kernels(self):
        """Compile append_paged_kv_cache Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        append_source = self._read_metal_file("metal_append_paged_kv_cache.metal")
        if not append_source:
            return

        self._compile_shader(append_source, "append_kv_cache")

    def run_append_paged_kv_cache_mps(
        self,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        paged_kv_cache: torch.Tensor,
        kv_batch_indices: torch.Tensor,
        kv_positions: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        page_size: int,
    ) -> None:
        """Run append_paged_kv_cache using compiled MPS kernels.

        Modifies paged_kv_cache in-place.

        Args:
            k_input: Key states to append [num_tokens, num_kv_heads * head_size]
            v_input: Value states to append [num_tokens, num_kv_heads * head_size]
            paged_kv_cache: Unified KV cache buffer (flattened 1D tensor)
            kv_batch_indices: Batch index for each token [num_tokens]
            kv_positions: Position within sequence [num_tokens]
            kv_page_indices: Page indices [max_num_pages]
            kv_page_indptr: Page indptr [batch_size + 1]
            kv_last_page_lens: Last page lengths [batch_size]
            num_kv_heads: Number of KV heads
            head_size: Head dimension
            page_size: Page size
        """
        if not self.can_use_mps_kernels() or "append_kv_cache" not in self.compiled_libraries:
            raise RuntimeError("Append KV cache MPS kernels not available")

        k_input = k_input.to("mps") if k_input.device.type != "mps" else k_input
        v_input = v_input.to("mps") if v_input.device.type != "mps" else v_input
        paged_kv_cache = paged_kv_cache.to("mps") if paged_kv_cache.device.type != "mps" else paged_kv_cache

        num_tokens = k_input.shape[0]
        batch_size = kv_page_indptr.shape[0] - 1
        max_num_pages = paged_kv_cache.numel() // (2 * page_size * num_kv_heads * head_size)

        params = torch.tensor(
            [
                num_tokens,
                num_kv_heads,
                head_size,
                page_size,
                max_num_pages,
                batch_size,
                num_kv_heads * head_size,  # k_stride_token
                head_size,                 # k_stride_head
                num_kv_heads * head_size,  # v_stride_token
                head_size,                 # v_stride_head
            ],
            dtype=torch.int32,
            device="mps",
        )

        lib = self.compiled_libraries["append_kv_cache"]

        # Select kernel based on dtype
        if k_input.dtype == torch.bfloat16:
            kernel_name = "metal_append_paged_kv_cache_bfloat16"
        elif k_input.dtype == torch.float32:
            kernel_name = "metal_append_paged_kv_cache_float32"
        elif k_input.dtype == torch.float16:
            kernel_name = "metal_append_paged_kv_cache_float16"
        else:
            raise RuntimeError(f"Unsupported dtype {k_input.dtype}")

        if not hasattr(lib, kernel_name):
            raise RuntimeError(f"Kernel {kernel_name} not found")

        getattr(lib, kernel_name)(
            k_input,
            v_input,
            paged_kv_cache,
            kv_batch_indices.to(torch.int32),
            kv_positions.to(torch.int32),
            kv_page_indices.to(torch.int32),
            kv_page_indptr.to(torch.int32),
            kv_last_page_lens.to(torch.int32),
            params.float(),
            threads=(num_tokens, num_kv_heads, head_size),
            group_size=(8, 8, 8),
        )
