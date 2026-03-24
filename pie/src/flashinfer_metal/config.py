"""
MPS configuration and availability detection.

This module checks if Metal Performance Shaders are available
for hardware-accelerated operations on Apple Silicon.
"""

import platform

import torch

# Platform detection
IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.processor() == "arm"

# MPS availability checks
try:
    import torch.mps

    MPS_COMPILE_AVAILABLE = hasattr(torch.mps, "compile_shader")
except (ImportError, AttributeError):
    MPS_COMPILE_AVAILABLE = False

MPS_DEVICE_AVAILABLE = (
    torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
)
