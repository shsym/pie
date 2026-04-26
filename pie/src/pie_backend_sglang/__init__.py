# Pie Backend (SGLang) — sibling backend to pie_backend that uses sglang's
# model definitions, attention kernels, and KV cache plumbing under the hood
# while preserving pie_backend's RPC surface. SGLang is an optional dependency:
# install via `pie-server[sglang]`.

from __future__ import annotations


_SGLANG_INSTALL_HINT = (
    "pie_backend_sglang requires SGLang. Install with `pip install pie-server[sglang]` "
    "or `uv sync --extra sglang`."
)


def _require_sglang():
    try:
        import sglang  # noqa: F401
    except ImportError as e:
        raise ImportError(_SGLANG_INSTALL_HINT) from e
