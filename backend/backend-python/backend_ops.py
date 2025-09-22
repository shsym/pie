"""
Backend Operations - Abstraction layer for different compute backends

This module provides a unified interface for operations that can be implemented
by different backends (flashinfer, Metal, etc.). The specific backend is determined
at runtime based on availability and configuration.
"""

from __future__ import annotations
import torch


class BackendOps:
    """Abstract base class for backend operations."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    # Image operations
    def decode_image(
        self, image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image from bytes to tensor."""
        raise NotImplementedError(
            f"decode_image not implemented for {self.backend_name}"
        )

    # Sampling operations
    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample tokens from probability distribution."""
        raise NotImplementedError(
            f"sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p (nucleus) sampling from probability distribution."""
        raise NotImplementedError(
            f"top_p_sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_k_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling from probability distribution."""
        raise NotImplementedError(
            f"top_k_sampling_from_probs not implemented for {self.backend_name}"
        )

    def min_p_sampling_from_probs(
        self, probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling from probability distribution."""
        raise NotImplementedError(
            f"min_p_sampling_from_probs not implemented for {self.backend_name}"
        )

    def top_k_top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling from probability distribution."""
        raise NotImplementedError(
            f"top_k_top_p_sampling_from_probs not implemented for {self.backend_name}"
        )


class FlashInferOps(BackendOps):
    """FlashInfer backend operations."""

    def __init__(self):
        super().__init__("flashinfer")
        try:
            import flashinfer as ops  # pylint: disable=import-outside-toplevel

            self.ops = ops
            self.available = True
        except ImportError:
            self.ops = None
            self.available = False
            print("⚠️ FlashInfer not available - some operations may not work")

    def decode_image(
        self, image_blob: bytes, dtype: torch.dtype, device: str
    ) -> torch.Tensor:
        """Decode image using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for image decoding")
        return self.ops.image.decode_image(image_blob, dtype=dtype, device=device)  # type: ignore

    def sampling_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Sample using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for sampling")
        return self.ops.sampling.sampling_from_probs(probs)  # type: ignore

    def top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Top-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for top-p sampling")
        return self.ops.sampling.top_p_sampling_from_probs(probs, top_p=top_p)  # type: ignore

    def top_k_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor
    ) -> torch.Tensor:
        """Top-k sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for top-k sampling")
        return self.ops.sampling.top_k_sampling_from_probs(probs, top_k=top_k)  # type: ignore

    def min_p_sampling_from_probs(
        self, probs: torch.Tensor, min_p: torch.Tensor
    ) -> torch.Tensor:
        """Min-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for min-p sampling")
        return self.ops.sampling.min_p_sampling_from_probs(probs, min_p=min_p)  # type: ignore

    def top_k_top_p_sampling_from_probs(
        self, probs: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
    ) -> torch.Tensor:
        """Combined top-k and top-p sampling using FlashInfer."""
        if not self.available or self.ops is None:
            raise RuntimeError("FlashInfer not available for combined sampling")
        return self.ops.sampling.top_k_top_p_sampling_from_probs(  # type: ignore
            probs, top_k=top_k, top_p=top_p
        )


# Note: Backend selection should be done explicitly by each backend's code
# Each backend (flashinfer, metal, etc.) should instantiate their own ops class
