# Pie Backend (vLLM) — sibling backend to pie_backend that uses vllm's model
# definitions and kernels under the hood while preserving pie_backend's RPC
# surface. vLLM is an optional dependency: install via `pie-server[vllm]`.

from __future__ import annotations

import os


# vllm's FlashInfer backend imports trigger a strict version check between
# `flashinfer` (vllm pulls this) and `flashinfer-jit-cache` (pie pins this).
# Pie's pin lags vllm's, but the runtime is compatible. Bypass the check until
# the pins align in pyproject.toml.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


_VLLM_INSTALL_HINT = (
    "pie_backend_vllm requires vLLM. Install with `pip install pie-server[vllm]` "
    "or `uv sync --extra vllm`."
)


def _require_vllm():
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        raise ImportError(_VLLM_INSTALL_HINT) from e
