"""
Metal RoPE (Rotary Position Embedding) operation implementation.

Compiles and executes Metal RoPE kernels for PyTorch MPS backend.
"""

import torch

from .config import MPS_COMPILE_AVAILABLE
from .shader_compiler import BaseShaderCompiler


class RoPECompiler(BaseShaderCompiler):
    """Compiles and runs Metal RoPE kernels."""

    def __init__(self):
        super().__init__()
        self._compile_rope_kernels()

    def _compile_rope_kernels(self):
        """Compile RoPE Metal kernels."""
        if not MPS_COMPILE_AVAILABLE:
            return

        rope_source = self._read_metal_file("metal_rope.metal")
        if not rope_source:
            return

        self._compile_shader(rope_source, "rope")

    def run_rope_mps(
        self,
        input_qk: torch.Tensor,
        position_ids: torch.Tensor,
        rope_theta: float = 10000.0,
        rope_factor: float = 1.0,
        interleaved: bool = False,
    ) -> None:
        """Run RoPE using compiled MPS kernels.

        Modifies input_qk in-place.

        Args:
            input_qk: Input tensor [num_tokens, num_heads, head_size] on MPS device
            position_ids: Position IDs [num_tokens] on MPS device
            rope_theta: RoPE theta parameter (default: 10000.0)
            rope_factor: RoPE scaling factor (default: 1.0)
            interleaved: Layout mode (default: False for non-interleaved)
        """
        if not self.can_use_mps_kernels() or "rope" not in self.compiled_libraries:
            raise RuntimeError("RoPE MPS kernels not available")

        if input_qk.device.type != "mps":
            raise RuntimeError(f"input_qk must be on MPS device, got {input_qk.device}")
        if position_ids.device.type != "mps":
            position_ids = position_ids.to("mps")

        num_tokens, num_heads, head_size = input_qk.shape

        # Flatten for Metal kernel
        was_contiguous = input_qk.is_contiguous()
        input_qk_contiguous = None

        if was_contiguous:
            input_qk_flat = input_qk.view(-1)
        else:
            input_qk_contiguous = input_qk.contiguous()
            input_qk_flat = input_qk_contiguous.view(-1)

        params = torch.tensor(
            [
                num_tokens,
                num_heads,
                head_size,
                rope_theta,
                rope_factor,
                1 if interleaved else 0,
            ],
            dtype=torch.float32,
            device="mps",
        )

        lib = self.compiled_libraries["rope"]

        # Select kernel based on dtype
        if input_qk.dtype == torch.float16:
            kernel_name = "metal_rope_float16"
        elif input_qk.dtype == torch.bfloat16:
            kernel_name = "metal_rope_bfloat16"
        else:
            kernel_name = "metal_rope_float32"

        if not hasattr(lib, kernel_name):
            raise RuntimeError(f"RoPE kernel {kernel_name} not found")

        num_pairs = head_size // 2

        getattr(lib, kernel_name)(
            input_qk_flat,
            position_ids.to(torch.int32),
            params,
            threads=(num_tokens, num_heads, num_pairs),
            group_size=(8, 8, 4),
        )

        # Copy back if not contiguous
        if not was_contiguous:
            assert input_qk_contiguous is not None
            input_qk.copy_(input_qk_contiguous)
