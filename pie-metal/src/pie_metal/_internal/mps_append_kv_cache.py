"""
Metal append_paged_kv_cache operation implementation.

This module handles compilation and execution of Metal append KV cache kernels
for PyTorch MPS backend.
"""

import torch
from typing import Optional
from .mps_shader_compiler import BaseShaderCompiler
from .mps_config import (
    MPS_COMPILE_AVAILABLE,
    DEBUG_ENABLED,
    DEBUG_ATOL,
    DEBUG_RTOL,
    DEBUG_VERBOSITY,
    VERBOSITY_DETAILED
)


class AppendKVCacheCompiler(BaseShaderCompiler):
    """Compiles and runs Metal append_paged_kv_cache kernels."""

    def __init__(self):
        super().__init__()
        self._compile_append_kv_cache_kernels()

    def _compile_append_kv_cache_kernels(self):
        """Compile append_paged_kv_cache Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read append_kv_cache kernel source
        append_source = self._read_metal_file("metal_append_paged_kv_cache.metal")
        if not append_source:
            print("⚠️  Append KV cache kernel source not found")
            return

        try:
            # Compile the append_kv_cache shader library
            if self._compile_shader(append_source, 'append_kv_cache'):
                print("✅ Compiled append_paged_kv_cache kernels for MPS")
        except Exception as e:
            print(f"⚠️  Failed to compile append_kv_cache kernels: {e}")

    def run_append_paged_kv_cache_mps(
        self,
        k_input: torch.Tensor,
        v_input: torch.Tensor,
        paged_k_cache: torch.Tensor,
        paged_v_cache: torch.Tensor,
        kv_batch_indices: torch.Tensor,
        kv_positions: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        num_kv_heads: Optional[int] = None,
        head_size: Optional[int] = None
    ) -> None:
        """
        Run append_paged_kv_cache using compiled MPS kernels.

        Modifies paged_k_cache and paged_v_cache in-place.

        Args:
            k_input: Key states to append [num_tokens, num_kv_heads * head_size]
            v_input: Value states to append [num_tokens, num_kv_heads * head_size]
            paged_k_cache: Paged K cache [max_num_pages, page_size, num_kv_heads * head_size]
            paged_v_cache: Paged V cache [max_num_pages, page_size, num_kv_heads * head_size]
            kv_batch_indices: Batch index for each token [num_tokens]
            kv_positions: Position within sequence [num_tokens]
            kv_page_indices: Page indices [max_num_pages]
            kv_page_indptr: Page indptr [batch_size + 1]
            kv_last_page_lens: Last page lengths [batch_size]
        """
        if not self.can_use_mps_kernels() or 'append_kv_cache' not in self.compiled_libraries:
            raise RuntimeError("Append KV cache MPS kernels not available")

        # Ensure all tensors are on MPS device first
        k_input = k_input.to('mps') if k_input.device.type != 'mps' else k_input
        v_input = v_input.to('mps') if v_input.device.type != 'mps' else v_input
        paged_k_cache = paged_k_cache.to('mps') if paged_k_cache.device.type != 'mps' else paged_k_cache
        paged_v_cache = paged_v_cache.to('mps') if paged_v_cache.device.type != 'mps' else paged_v_cache

        # DEBUG MODE: Clone caches for comparison (before any modifications)
        if DEBUG_ENABLED:
            from . import debug_utils
            from . import pytorch_reference

            # Clone all caches before modification
            paged_k_cache_clone_pytorch = paged_k_cache.clone()
            paged_v_cache_clone_pytorch = paged_v_cache.clone()

            # Collect input metadata
            input_metadata = [
                debug_utils.collect_tensor_metadata(k_input, "k_input"),
                debug_utils.collect_tensor_metadata(v_input, "v_input"),
                debug_utils.collect_tensor_metadata(paged_k_cache, "paged_k_cache (before)"),
                debug_utils.collect_tensor_metadata(paged_v_cache, "paged_v_cache (before)"),
            ]

            # Run PyTorch reference on cloned caches (do this first)
            if num_kv_heads is None or head_size is None:
                raise ValueError("num_kv_heads and head_size must be provided for debug mode")

            pytorch_reference.append_paged_kv_cache_reference(
                k_input, v_input,
                paged_k_cache_clone_pytorch, paged_v_cache_clone_pytorch,
                kv_batch_indices, kv_positions,
                kv_page_indices, kv_page_indptr, kv_last_page_lens,
                num_kv_heads, head_size
            )

        # Get dimensions
        num_tokens = k_input.shape[0]
        page_size = paged_k_cache.shape[1]
        max_num_pages = paged_k_cache.shape[0]
        batch_size = kv_page_indptr.shape[0] - 1

        # Use provided num_kv_heads and head_size (required parameters)
        if num_kv_heads is None or head_size is None:
            raise ValueError(
                "num_kv_heads and head_size must be provided to run_append_paged_kv_cache_mps"
            )

        # Create params tensor matching AppendPagedKVCacheParams struct
        params = torch.tensor([
            num_tokens,
            num_kv_heads,
            head_size,
            page_size,
            max_num_pages,
            batch_size,
            num_kv_heads * head_size,  # k_stride_token
            head_size,                  # k_stride_head
            num_kv_heads * head_size,  # v_stride_token
            head_size                   # v_stride_head
        ], dtype=torch.int32, device='mps')

        lib = self.compiled_libraries['append_kv_cache']

        # Select kernel based on dtype
        if k_input.dtype == torch.bfloat16:
            kernel_name = 'metal_append_paged_kv_cache_bfloat16'
        elif k_input.dtype == torch.float32:
            kernel_name = 'metal_append_paged_kv_cache_float32'
        elif k_input.dtype == torch.float16:
            kernel_name = 'metal_append_paged_kv_cache_float16'
        else:
            raise RuntimeError(
                f"append_paged_kv_cache only supports float16, bfloat16, and float32, got {k_input.dtype}. "
                f"Please convert tensors before calling (e.g., tensor.to(torch.float32))."
            )

        if hasattr(lib, kernel_name):
            # append_kv_cache uses 3D grid: (num_tokens, num_kv_heads, head_size)
            # Each thread handles one element at position (token_idx, head_idx, head_offset)
            # The Metal kernel expects thread_position_in_grid with x=token, y=head, z=offset

            # Call Metal append_kv_cache kernel (modifies caches in-place)
            getattr(lib, kernel_name)(
                k_input,                        # buffer(0)
                v_input,                        # buffer(1)
                paged_k_cache,                  # buffer(2)
                paged_v_cache,                  # buffer(3)
                kv_batch_indices.to(torch.int32),  # buffer(4)
                kv_positions.to(torch.int32),       # buffer(5)
                kv_page_indices.to(torch.int32),    # buffer(6)
                kv_page_indptr.to(torch.int32),     # buffer(7)
                kv_last_page_lens.to(torch.int32),  # buffer(8)
                params.float(),                  # buffer(9) - Metal expects float buffer
                threads=(num_tokens, num_kv_heads, head_size),
                group_size=(8, 8, 8)  # Use 3D threadgroup to match 3D dispatch
            )
        else:
            raise RuntimeError(f"Append KV cache kernel {kernel_name} not found in compiled library")

        # DEBUG MODE: Compare Metal output with PyTorch reference
        if DEBUG_ENABLED:
            from . import debug_utils

            # Compare Metal vs PyTorch outputs (K cache)
            matches_k, diagnostics_k = debug_utils.compare_tensors(
                paged_k_cache, paged_k_cache_clone_pytorch,
                atol=DEBUG_ATOL, rtol=DEBUG_RTOL,
                operation_name="Append KV Cache (K)"
            )

            # Compare Metal vs PyTorch outputs (V cache)
            matches_v, diagnostics_v = debug_utils.compare_tensors(
                paged_v_cache, paged_v_cache_clone_pytorch,
                atol=DEBUG_ATOL, rtol=DEBUG_RTOL,
                operation_name="Append KV Cache (V)"
            )

            # Generate and print reports
            report_k = debug_utils.generate_report(
                diagnostics_k, input_metadata, verbosity=DEBUG_VERBOSITY
            )
            if report_k:
                print(report_k)

            report_v = debug_utils.generate_report(
                diagnostics_v, input_metadata, verbosity=DEBUG_VERBOSITY
            )
            if report_v:
                print(report_v)

            # Warnings if mismatches detected
            if (not matches_k or not matches_v) and DEBUG_VERBOSITY >= VERBOSITY_DETAILED:
                print("WARNING: Append KV cache Metal kernel output differs from PyTorch reference")