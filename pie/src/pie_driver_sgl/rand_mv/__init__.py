"""Sglang-driver-local copy of `pie_kernels.rand_mv` — dispatches to a
Triton/CUDA backend on NVIDIA, Metal on Apple Silicon, otherwise raises.

This is a deliberate fork of `pie_kernels.rand_mv`. Keeping a private copy
under `pie_driver_sgl` lets the sglang adapter path tweak shapes /
JIT-compile flags / tensor-layout assumptions for sglang's
`QKVParallelLinear` outputs without affecting the native flashinfer
driver's adapter math (which has its own copy under `pie_kernels`).

`RAND_MV_AVAILABLE` reports whether either backend is usable; callers
should gate invocations on it rather than catching `RuntimeError`.
"""

from __future__ import annotations

import torch

RAND_MV_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()

if not RAND_MV_AVAILABLE:
    def batched_randn_matmul(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_matmul requires CUDA or MPS; "
            "neither backend is available on this platform."
        )

    def batched_randn_generate(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_generate requires CUDA or MPS; "
            "neither backend is available on this platform."
        )

    def run_tests():
        raise RuntimeError(
            "rand_mv tests require CUDA or MPS."
        )

elif torch.cuda.is_available():
    from .cuda.rand_mv_new import (  # noqa: F401
        batched_randn_matmul,
        batched_randn_matmul_sectioned,
        batched_randn_matmul_sectioned_per_input,
        batched_randn_generate,
        run_tests,
    )

else:
    # MPS path
    from .metal.rand_mv import (  # noqa: F401
        batched_randn_matmul,
        batched_randn_generate,
    )

    def run_tests():
        raise RuntimeError(
            "rand_mv CUDA/Triton reference tests require CUDA. "
            "The Metal implementation has its own tests under tests/pie_kernels_metal/."
        )
