"""
Common utilities for Metal kernel operations.
"""

import torch


def validate_mps_device(tensor: torch.Tensor, name: str) -> None:
    """Validate that a tensor is on MPS device.

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Raises:
        RuntimeError: If tensor is not on MPS device
    """
    if tensor.device.type != "mps":
        raise RuntimeError(
            f"flashinfer_metal requires all tensors to be on MPS device. "
            f"Tensor '{name}' is on {tensor.device}. "
            f"Use: tensor = tensor.to('mps')"
        )


def validate_page_size(page_size: int) -> None:
    """Validate page_size configuration for Metal kernel compilation.

    Args:
        page_size: The KV cache page size to validate

    Raises:
        ValueError: If page_size is invalid
    """
    # Must be power of 2
    if page_size <= 0 or (page_size & (page_size - 1)) != 0:
        raise ValueError(
            f"page_size must be a power of 2. Got: {page_size}. "
            f"Valid values: 1, 2, 4, 8, 16"
        )

    # Metal threadgroup memory limit: 32KB
    # With MAX_HEAD_DIM=256, page_size=32 would need ~34KB
    if page_size > 16:
        raise ValueError(
            f"page_size={page_size} exceeds Metal threadgroup memory limit (32KB). "
            f"Maximum supported: 16. Valid values: 1, 2, 4, 8, 16"
        )
