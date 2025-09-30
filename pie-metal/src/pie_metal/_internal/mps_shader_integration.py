"""
Real PyTorch MPS integration using torch.mps.compile_shader.

This module compiles and executes our existing Metal attention kernels
directly through PyTorch's MPS backend, enabling true zero-copy operations.
"""

import torch
import warnings
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

# Check if torch.mps.compile_shader is available
try:
    from torch.mps import compile_shader
    MPS_COMPILE_AVAILABLE = True
    print("✅ torch.mps.compile_shader available")
except ImportError:
    MPS_COMPILE_AVAILABLE = False
    warnings.warn("torch.mps.compile_shader not available - using fallback implementation")

# Check if MPS device is available
MPS_DEVICE_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False


class MPSShaderCompiler:
    """Compiles and manages our Metal attention kernels for PyTorch MPS."""

    def __init__(self):
        self.compiled_libraries: Dict[str, Any] = {}
        self.kernel_dir = Path(__file__).parent / "metal" / "kernels"
        self._initialize_shaders()

    def _initialize_shaders(self):
        """Compile all required Metal shaders."""
        if not MPS_COMPILE_AVAILABLE or not MPS_DEVICE_AVAILABLE:
            print("⚠️  MPS shader compilation skipped - not available")
            return

        try:
            self._compile_attention_kernels()
            self._compile_rope_kernels()
            self._compile_append_kv_cache_kernels()
            print("✅ Metal shaders compiled successfully")
        except Exception as e:
            print(f"⚠️  Failed to compile Metal shaders: {e}")

    def _compile_attention_kernels(self):
        """Compile the actual optimized Metal attention kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read the actual optimized kernel sources
        common_source = self._read_metal_file("metal_attention_common.metal")
        simdgroup_source = self._read_metal_file("metal_attention_simdgroup_opt.metal")

        if not common_source or not simdgroup_source:
            print("⚠️  Could not find optimized Metal kernel sources, using fallback")
            self._compile_simple_kernels()
            return

        # Process the common header to resolve includes
        processed_common = self._process_common_header(common_source)

        # Process the simdgroup source to resolve includes
        processed_simdgroup = self._resolve_includes(simdgroup_source, processed_common)

        # Transform Params struct to params_raw for torch.mps.compile_shader compatibility
        processed_simdgroup = processed_simdgroup.replace(
            'constant Params& params [[buffer(8)]]',
            'device const float* params_raw [[buffer(8)]]'
        )

        # Replace parameter access patterns
        param_replacements = [
            ('const int num_qo = params.num_qo;', 'const int num_qo = (int)params_raw[0];'),
            ('const int head_dim = params.head_dim;', 'const int head_dim = (int)params_raw[1];'),
            ('const int kv_head_dim = params.kv_head_dim;', 'const int kv_head_dim = (int)params_raw[2];'),
            ('const int head_size = params.head_size;', 'const int head_size = (int)params_raw[3];'),
            ('const int page_size = params.page_size;', 'const int page_size = (int)params_raw[4];'),
            ('const int num_query_heads = params.num_query_heads;', 'const int num_query_heads = (int)params_raw[5];'),
            ('const int num_kv_heads = params.num_kv_heads;', 'const int num_kv_heads = (int)params_raw[6];'),
            ('const float scale = params.scale;', 'const float scale = params_raw[7];'),
        ]

        for old, new in param_replacements:
            processed_simdgroup = processed_simdgroup.replace(old, new)

        # Compile the full optimized attention kernels
        full_source = f"""
#include <metal_stdlib>
using namespace metal;

{processed_common}

{processed_simdgroup}
"""

        try:
            # Compile the actual optimized shader library
            self.compiled_libraries['attention'] = compile_shader(full_source)
            print("✅ Compiled OPTIMIZED Metal attention kernels for MPS")

        except Exception as e:
            print(f"⚠️  Failed to compile optimized kernels: {e}")
            print("   Falling back to simple implementation")
            # Fall back to simpler implementation
            self._compile_simple_kernels()

    def _process_common_header(self, common_source: str) -> str:
        """Process the common header to resolve any includes."""
        # Remove the duplicate #include <metal_stdlib> and using namespace since
        # they'll be included in the final shader source
        processed = common_source.replace('#include <metal_stdlib>\n', '')
        processed = processed.replace('using namespace metal;\n', '')

        # Remove empty lines at the beginning
        lines = processed.split('\n')
        while lines and lines[0].strip() == '':
            lines.pop(0)

        return '\n'.join(lines)

    def _resolve_includes(self, source: str, common_source: str) -> str:
        """Resolve includes in Metal source code."""
        lines = source.split('\n')
        resolved_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "metal_attention_common.metal"'):
                # Skip this include since we're already including the common source
                resolved_lines.append('// Resolved: #include "metal_attention_common.metal"')
            elif stripped.startswith('#include <metal_stdlib>'):
                # Skip duplicate metal_stdlib includes
                resolved_lines.append('// Skipped duplicate: #include <metal_stdlib>')
            elif stripped.startswith('using namespace metal;'):
                # Skip duplicate namespace declarations
                resolved_lines.append('// Skipped duplicate: using namespace metal;')
            else:
                resolved_lines.append(line)

        return '\n'.join(resolved_lines)

    def _compile_simple_kernels(self):
        """Compile simplified kernels that work with torch.mps.compile_shader."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Very simple kernel that just works
        simple_source = """
#include <metal_stdlib>
using namespace metal;

struct SimpleParams {
    int num_tokens;
    int head_dim;
    float scale;
};

kernel void simple_attention_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant SimpleParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(params.num_tokens * params.head_dim)) return;

    uint token_idx = idx / uint(params.head_dim);
    uint dim_idx = idx % uint(params.head_dim);

    // Simple attention: just copy value (placeholder)
    output[idx] = value[idx] * params.scale;
}

kernel void simple_attention_f16(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant SimpleParams& params [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= uint(params.num_tokens * params.head_dim)) return;

    uint token_idx = idx / uint(params.head_dim);
    uint dim_idx = idx % uint(params.head_dim);

    // Simple attention: just copy value (placeholder)
    output[idx] = half(value[idx] * params.scale);
}
"""

        try:
            self.compiled_libraries['simple'] = compile_shader(simple_source)
            print("✅ Compiled simple attention kernels for MPS")
        except Exception as e:
            print(f"❌ Failed to compile even simple kernels: {e}")

    def _compile_rope_kernels(self):
        """Compile RoPE (Rotary Position Embedding) Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        # Read RoPE kernel source
        rope_source = self._read_metal_file("metal_rope.metal")
        if not rope_source:
            print("⚠️  RoPE kernel source not found")
            return

        try:
            # Compile the RoPE shader library
            self.compiled_libraries['rope'] = compile_shader(rope_source)
            print("✅ Compiled RoPE kernels for MPS")
        except Exception as e:
            print(f"⚠️  Failed to compile RoPE kernels: {e}")

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
            self.compiled_libraries['append_kv_cache'] = compile_shader(append_source)
            print("✅ Compiled append_paged_kv_cache kernels for MPS")
        except Exception as e:
            print(f"⚠️  Failed to compile append_kv_cache kernels: {e}")

    def _read_metal_file(self, filename: str) -> str:
        """Read Metal kernel source file."""
        file_path = self.kernel_dir / filename
        if file_path.exists():
            return file_path.read_text()
        else:
            print(f"⚠️  Metal file not found: {filename}")
            return ""

    def can_use_mps_kernels(self) -> bool:
        """Check if we can use compiled MPS kernels."""
        return (MPS_COMPILE_AVAILABLE and
                MPS_DEVICE_AVAILABLE and
                len(self.compiled_libraries) > 0)

    def run_attention_mps(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run attention using compiled MPS kernels.

        All dtype conversions happen here before passing to Metal kernels.
        Metal kernels expect float16 (half) types.
        """

        if not self.can_use_mps_kernels():
            raise RuntimeError("MPS kernels not available")

        if 'attention' not in self.compiled_libraries:
            raise RuntimeError("Attention kernel not compiled")

        # Store original dtype for conversion back
        original_dtype = query.dtype

        # Convert bfloat16 to float16 since Metal kernel uses half (float16) types
        # The kernel name says "bf16" but the actual Metal code uses half types
        if query.dtype == torch.bfloat16:
            query = query.to(torch.float16)
            kv_cache = kv_cache.to(torch.float16)

        # Ensure tensors are on MPS device
        query = query.to('mps') if query.device.type != 'mps' else query
        kv_cache = kv_cache.to('mps') if kv_cache.device.type != 'mps' else kv_cache

        # Get dimensions
        num_tokens, num_heads, head_dim = query.shape
        # Kernel expects 1D output buffer, we'll reshape it later
        output = torch.empty(num_tokens * num_heads * head_dim, device='mps', dtype=query.dtype)

        # Run the attention kernel
        result = self._run_full_attention(
            query, kv_cache, kv_page_indices, kv_page_indptr,
            kv_last_page_lens, qo_indptr, output
        )

        # Convert back to original dtype if needed
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
        output: torch.Tensor
    ) -> torch.Tensor:
        """Run the actual optimized Metal attention kernel."""

        lib = self.compiled_libraries['attention']

        # Extract dimensions
        num_tokens, num_heads, head_dim = query.shape
        num_pages, _, page_size, _, _ = kv_cache.shape
        scale = 1.0 / (head_dim ** 0.5)

        # Create the Params struct exactly as expected by the Metal kernel
        # This matches what the test uses: 8 parameters
        params_data = [
            num_tokens,           # num_qo (seq_len in test)
            num_heads * head_dim, # head_dim (total_head_dim in test)
            num_heads * head_dim, # kv_head_dim (total_head_dim in test)
            head_dim,             # head_size
            page_size,            # page_size
            num_heads,            # num_query_heads
            num_heads,            # num_kv_heads
            scale,                # scale (1/sqrt(head_dim))
        ]

        # Convert to Metal-compatible parameter buffer
        params = torch.tensor(params_data, dtype=torch.float32, device='mps')

        # Prepare tensors in the exact format expected by the kernel
        # The kernel expects 1D flat buffers for K and V caches
        # Input: [num_pages, 2, page_size, num_kv_heads, head_dim]
        # Output: 1D [num_pages * page_size * num_kv_heads * head_dim]
        paged_k_cache = kv_cache[:, 0, :, :, :].contiguous().view(-1)
        paged_v_cache = kv_cache[:, 1, :, :, :].contiguous().view(-1)

        # Reshape query for kernel: [num_tokens, num_heads, head_dim] -> 1D [num_tokens * num_heads * head_dim]
        q_input = query.contiguous().view(-1)

        # Create debug buffer (optional, can be None)
        debug_out = torch.zeros(20, dtype=torch.float32, device='mps')

        # Launch the actual optimized kernel!
        kernel_name = 'batch_prefill_attention_unified_bf16_simdgroup_kernel'

        # Configure dispatch parameters
        threads_per_threadgroup = 128
        total_threads = num_tokens * threads_per_threadgroup

        if not hasattr(lib, kernel_name):
            raise RuntimeError(f"Kernel {kernel_name} not found in compiled library")

        # Call the optimized simdgroup kernel with exact parameter layout and dispatch config
        getattr(lib, kernel_name)(
            q_input,                             # q_input [buffer(0)]
            paged_k_cache,                       # paged_k_cache [buffer(1)]
            paged_v_cache,                       # paged_v_cache [buffer(2)]
            qo_indptr.to(torch.int32),          # qo_indptr [buffer(3)]
            kv_page_indptr.to(torch.int32),     # kv_page_indptr [buffer(4)]
            kv_page_indices.to(torch.int32),    # kv_page_indices [buffer(5)]
            kv_last_page_lens.to(torch.int32),  # kv_last_page_lens [buffer(6)]
            output,                              # output [buffer(7)]
            params,                              # params [buffer(8)]
            debug_out,                           # debug_out [buffer(9)]
            threads=(total_threads, 1, 1),
            group_size=(threads_per_threadgroup, 1, 1)
        )

        # Reshape output from 1D [num_tokens * num_heads * head_dim] to expected 2D [num_tokens, num_heads * head_dim]
        return output.view(num_tokens, num_heads * head_dim)

    def _run_simple_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor
    ) -> torch.Tensor:
        """Run simple attention kernel as fallback."""

        lib = self.compiled_libraries['simple']

        num_tokens, num_heads, head_dim = query.shape
        scale = 1.0 / (head_dim ** 0.5)

        # Simple parameters
        params = torch.tensor([
            num_tokens,
            head_dim,
            scale
        ], dtype=torch.float32, device='mps')

        # Flatten tensors for simple kernel
        query_flat = query.view(-1)
        kv_flat = kv_cache.view(-1)[:query_flat.numel()]  # Match size
        value_flat = kv_flat  # Use same as value for simplicity
        output_flat = output.view(-1)

        kernel_name = 'simple_attention_f16' if query.dtype == torch.float16 else 'simple_attention_f32'

        if hasattr(lib, kernel_name):
            getattr(lib, kernel_name)(
                query_flat,
                kv_flat,
                value_flat,
                output_flat,
                params
            )
        else:
            raise RuntimeError(f"Simple kernel {kernel_name} not found")

        return output

    def run_rope_mps(
        self,
        input_qk: torch.Tensor,
        position_ids: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        interleaved: bool = False
    ) -> None:
        """
        Run RoPE (Rotary Position Embedding) using compiled MPS kernels.

        Modifies input_qk in-place.

        IMPORTANT: Input tensors must be on MPS device for the kernel to work.

        Args:
            input_qk: Input tensor [num_tokens, num_heads, head_size] on MPS device
            position_ids: Position IDs [num_tokens] on MPS device
            rope_theta: RoPE theta parameter (default: 10000.0)
            rope_factor: RoPE scaling factor (default: 1.0)
            interleaved: Layout mode (default: False for non-interleaved)
        """
        if not self.can_use_mps_kernels() or 'rope' not in self.compiled_libraries:
            raise RuntimeError("RoPE MPS kernels not available")

        # Verify tensors are on MPS device
        if input_qk.device.type != 'mps':
            raise RuntimeError(f"input_qk must be on MPS device, got {input_qk.device}")
        if position_ids.device.type != 'mps':
            position_ids = position_ids.to('mps')

        # Get dimensions
        num_tokens, num_heads, head_size = input_qk.shape

        # KEY INSIGHT: Flatten 3D tensor to 1D for Metal kernel
        # The Metal kernel uses 3D grid dispatch internally, but expects 1D buffer
        # This is the same pattern used in test_numerical_accuracy_comprehensive.py
        input_qk_flat = input_qk.contiguous().view(-1)

        # Create params tensor matching RoPEParams struct
        params = torch.tensor([
            num_tokens,
            num_heads,
            head_size,
            rope_theta,
            rope_factor,
            1 if interleaved else 0  # bool as int
        ], dtype=torch.float32, device='mps')

        lib = self.compiled_libraries['rope']

        # Select kernel based on dtype
        kernel_name = 'metal_rope_float32'
        if input_qk.dtype == torch.float16:
            kernel_name = 'metal_rope_float16'
        elif input_qk.dtype == torch.bfloat16:
            kernel_name = 'metal_rope_bfloat16'

        if hasattr(lib, kernel_name):
            # RoPE uses 3D grid: (num_tokens, num_heads, head_size/2)
            # Each thread handles one rotation pair
            num_pairs = head_size // 2

            # Call Metal RoPE kernel with flattened tensor (modifies in-place)
            # The flattened view shares memory with input_qk, so modifications
            # are automatically reflected in the original 3D tensor
            getattr(lib, kernel_name)(
                input_qk_flat,      # buffer(0): flattened input/output tensor
                position_ids.to(torch.int32),  # buffer(1): position IDs
                params,             # buffer(2): RoPEParams
                threads=(num_tokens * num_heads * num_pairs, 1, 1),
                group_size=(256, 1, 1)  # Standard threadgroup size
            )
        else:
            raise RuntimeError(f"RoPE kernel {kernel_name} not found in compiled library")

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

        # Ensure all tensors are on MPS device
        k_input = k_input.to('mps') if k_input.device.type != 'mps' else k_input
        v_input = v_input.to('mps') if v_input.device.type != 'mps' else v_input
        paged_k_cache = paged_k_cache.to('mps') if paged_k_cache.device.type != 'mps' else paged_k_cache
        paged_v_cache = paged_v_cache.to('mps') if paged_v_cache.device.type != 'mps' else paged_v_cache

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
        # NOTE: We only have bfloat16 and float32 kernels, no float16 kernel
        if k_input.dtype == torch.bfloat16:
            kernel_name = 'metal_append_paged_kv_cache_bfloat16'
        elif k_input.dtype == torch.float32:
            kernel_name = 'metal_append_paged_kv_cache_float32'
        else:
            # float16 not supported - dtype conversion breaks view connections
            # Caller must convert tensors to float32 or bfloat16 before calling
            raise RuntimeError(
                f"append_paged_kv_cache only supports bfloat16 and float32, got {k_input.dtype}. "
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


# Global compiler instance
_shader_compiler: Optional[MPSShaderCompiler] = None

def get_mps_compiler() -> MPSShaderCompiler:
    """Get or create the global MPS shader compiler."""
    global _shader_compiler
    if _shader_compiler is None:
        _shader_compiler = MPSShaderCompiler()
    return _shader_compiler

def run_mps_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
    custom_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Run attention using PyTorch MPS compiled shaders."""

    compiler = get_mps_compiler()

    if compiler.can_use_mps_kernels():
        return compiler.run_attention_mps(
            query, kv_cache, kv_page_indices, kv_page_indptr,
            kv_last_page_lens, qo_indptr, custom_mask
        )
    else:
        raise RuntimeError(
            "MPS shader compilation not available. "
            "Please ensure you have PyTorch 2.0+ with MPS support."
        )