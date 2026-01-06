"""
Metal attention operation implementation.

Compiles and executes Metal attention kernels for PyTorch MPS backend.
"""

from typing import Optional

import torch

from .config import MPS_COMPILE_AVAILABLE
from .shader_compiler import BaseShaderCompiler


class AttentionCompiler(BaseShaderCompiler):
    """Compiles and runs Metal attention kernels."""

    def __init__(self, page_size: int = 16):
        """Initialize AttentionCompiler with dynamic BLOCK_SIZE.

        Args:
            page_size: KV cache page size for BLOCK_SIZE compilation (default: 16)
        """
        super().__init__(page_size=page_size)
        self._compile_attention_kernels()

    def _compile_attention_kernels(self):
        """Compile the MLX Steel Attention-based kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        kernel_source = self._read_metal_file("metal_attention_simdgroup_opt.metal")
        if not kernel_source:
            return

        # Transform Params struct for torch.mps.compile_shader compatibility
        processed_source = kernel_source.replace(
            "constant Params& params [[buffer(7)]]",
            "device const float* params_raw [[buffer(7)]]",
        )

        param_replacements = [
            (
                "const int num_qo = params.num_qo;",
                "const int num_qo = (int)params_raw[0];",
            ),
            (
                "const int head_dim = params.head_dim;",
                "const int head_dim = (int)params_raw[1];",
            ),
            (
                "const int kv_head_dim = params.kv_head_dim;",
                "const int kv_head_dim = (int)params_raw[2];",
            ),
            (
                "const int head_size = params.head_size;",
                "const int head_size = (int)params_raw[3];",
            ),
            (
                "const int page_size = params.page_size;",
                "const int page_size = (int)params_raw[4];",
            ),
            (
                "const int num_query_heads = params.num_query_heads;",
                "const int num_query_heads = (int)params_raw[5];",
            ),
            (
                "const int num_kv_heads = params.num_kv_heads;",
                "const int num_kv_heads = (int)params_raw[6];",
            ),
            ("const float scale = params.scale;", "const float scale = params_raw[7];"),
        ]

        for old, new in param_replacements:
            processed_source = processed_source.replace(old, new)

        full_source = f"""
// Dynamically injected BLOCK_SIZE from configuration
#define BLOCK_SIZE {self.page_size}

{processed_source}
"""

        if self._compile_shader(full_source, "attention"):
            # Warmup prefill kernels
            self._warmup_kernel(
                "attention", "batch_prefill_attention_unified_fp16_simdgroup_kernel"
            )
            self._warmup_kernel(
                "attention", "batch_prefill_attention_unified_f32_simdgroup_kernel"
            )
            # Warmup decode kernels
            for head_dim in [64, 128]:
                self._warmup_kernel("attention", f"attention_decode_v2_fp16_{head_dim}")
                self._warmup_kernel("attention", f"attention_decode_v2_f32_{head_dim}")

    # Metal constant memory limit (64KB)
    CONSTANT_MEMORY_LIMIT = 64 * 1024

    def run_attention_mps(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention using compiled MPS kernels."""
        if not self.can_use_mps_kernels():
            raise RuntimeError("MPS kernels not available")

        if "attention" not in self.compiled_libraries:
            raise RuntimeError("Attention kernel not compiled")

        original_dtype = query.dtype

        # Dtype validation
        if query.dtype not in [torch.float32, torch.float16]:
            raise ValueError(
                f"Unsupported dtype {query.dtype}. Supported: float32, float16."
            )

        query = query.to("mps") if query.device.type != "mps" else query
        kv_cache = kv_cache.to("mps") if kv_cache.device.type != "mps" else kv_cache

        num_tokens, num_heads, head_dim = query.shape
        output = torch.empty(
            num_tokens * num_heads * head_dim, device="mps", dtype=query.dtype
        )

        result = self._run_full_attention(
            query,
            kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            qo_indptr,
            output,
        )

        if original_dtype != result.dtype:
            result = result.to(original_dtype)

        return result

    def _run_full_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run the actual optimized Metal attention kernel."""
        lib = self.compiled_libraries["attention"]

        num_tokens, num_heads, head_dim = query.shape
        _num_pages, _, page_size, num_kv_heads, _ = kv_cache.shape

        if head_dim > 128:
            raise ValueError(
                f"Head dimension {head_dim} exceeds Metal kernel limit of 128."
            )

        scale = 1.0 / (head_dim**0.5)

        params_data = [
            num_tokens,
            num_heads * head_dim,
            num_kv_heads * head_dim,
            head_dim,
            page_size,
            num_heads,
            num_kv_heads,
            scale,
        ]
        params = torch.tensor(params_data, dtype=torch.float32, device="mps")

        paged_kv_cache = kv_cache.contiguous().view(-1)
        q_input = query.contiguous().view(-1)
        debug_out = torch.zeros(20, dtype=torch.float32, device="mps")

        is_decode = num_tokens == 1

        if is_decode:
            dtype_prefix = "f32" if query.dtype == torch.float32 else "fp16"
            kernel_name = f"attention_decode_v2_{dtype_prefix}_{head_dim}"

            if not hasattr(lib, kernel_name):
                raise RuntimeError(
                    f"Decode kernel {kernel_name} not found. Supported head_dim: 64, 128"
                )

            getattr(lib, kernel_name)(
                q_input,
                paged_kv_cache,
                qo_indptr.to(torch.int32),
                kv_page_indptr.to(torch.int32),
                kv_page_indices.to(torch.int32),
                kv_last_page_lens.to(torch.int32),
                output,
                params,
                debug_out,
                threads=(num_heads * 1024, 1, 1),
                group_size=(1024, 1, 1),
            )
        else:
            if query.dtype == torch.float32:
                kernel_name = "batch_prefill_attention_unified_f32_simdgroup_kernel"
                bq = 16
                threads_per_threadgroup = 64
            else:
                kernel_name = "batch_prefill_attention_unified_fp16_simdgroup_kernel"
                bq = 32
                threads_per_threadgroup = 128

            num_q_blocks = (num_tokens + bq - 1) // bq
            total_threads_x = num_q_blocks * threads_per_threadgroup
            total_threads_y = num_heads

            getattr(lib, kernel_name)(
                q_input,
                paged_kv_cache,
                qo_indptr.to(torch.int32),
                kv_page_indptr.to(torch.int32),
                kv_page_indices.to(torch.int32),
                kv_last_page_lens.to(torch.int32),
                output,
                params,
                debug_out,
                threads=(total_threads_x, total_threads_y, 1),
                group_size=(threads_per_threadgroup, 1, 1),
            )

        total_head_dim = num_heads * head_dim
        return output.view(num_tokens, total_head_dim)
