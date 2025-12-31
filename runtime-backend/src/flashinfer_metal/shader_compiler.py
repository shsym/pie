"""
Base shader compiler with Metal file utilities.

Provides the base class for compiling Metal shaders and utilities
for reading and processing Metal source files.
"""

from pathlib import Path
from typing import Any, Dict

import torch

from .config import MPS_COMPILE_AVAILABLE, MPS_DEVICE_AVAILABLE


class BaseShaderCompiler:
    """Base class for Metal shader compilation and management."""

    def __init__(self, page_size: int = 16):
        """Initialize the base shader compiler.

        Args:
            page_size: KV cache page size for BLOCK_SIZE compilation (default: 16)
        """
        self.compiled_libraries: Dict[str, Any] = {}
        self.kernel_dir = Path(__file__).parent / "kernels"
        self.page_size = page_size

    def can_use_mps_kernels(self) -> bool:
        """Check if we can use compiled MPS kernels."""
        return (
            MPS_COMPILE_AVAILABLE
            and MPS_DEVICE_AVAILABLE
            and len(self.compiled_libraries) > 0
        )

    def _read_metal_file(self, filename: str) -> str:
        """Read Metal kernel source file."""
        file_path = self.kernel_dir / filename
        if file_path.exists():
            return file_path.read_text()
        return ""

    def _resolve_includes(self, source: str) -> str:
        """Resolve includes in Metal source code."""
        lines = source.split("\n")
        resolved_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#include "metal_attention_common.metal"'):
                resolved_lines.append(
                    '// Resolved: #include "metal_attention_common.metal"'
                )
            elif stripped.startswith("#include <metal_stdlib>"):
                resolved_lines.append("// Skipped duplicate: #include <metal_stdlib>")
            elif stripped.startswith("using namespace metal;"):
                resolved_lines.append("// Skipped duplicate: using namespace metal;")
            else:
                resolved_lines.append(line)

        return "\n".join(resolved_lines)

    def _compile_shader(self, source: str, library_name: str) -> bool:
        """Compile a Metal shader and store it in compiled_libraries.

        Returns True if compilation succeeded, False otherwise.
        """
        if not MPS_COMPILE_AVAILABLE:
            return False

        try:
            self.compiled_libraries[library_name] = torch.mps.compile_shader(source)
            return True
        except (RuntimeError, OSError, AttributeError):
            return False

    def _warmup_kernel(self, library_name: str, kernel_name: str) -> None:
        """Warm up a kernel by checking its existence (triggers PSO creation)."""
        if library_name not in self.compiled_libraries:
            return

        lib = self.compiled_libraries[library_name]
        if not hasattr(lib, kernel_name):
            raise RuntimeError(
                f"Kernel '{kernel_name}' not found in library '{library_name}'."
            )
