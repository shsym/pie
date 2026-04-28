# Pie Backend (vLLM) — `vllm` driver. Uses vllm's model definitions and
# kernels under pie's RPC surface. vLLM is an optional dependency: install
# via `pie-server[vllm]`.

from __future__ import annotations

import os


# Defensive: align flashinfer-python / flashinfer-cubin / flashinfer-jit-cache
# pin alignment is enforced in pyproject.toml. This bypass exists for early
# dev environments that haven't re-synced.
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


from pie.drivers import DriverSpec, register_driver

from .config import VllmDriverConfig

register_driver(DriverSpec(
    name="vllm",
    config_cls=VllmDriverConfig,
    worker_module="pie_backend_vllm.worker",
    extras=("vllm",),
))
