# Pie Driver (SGLang) — `sglang` driver. Uses sglang's model definitions,
# attention kernels, and KV cache plumbing under pie's RPC surface. SGLang
# is an optional dependency: install via `pie-server[sglang]`.

from __future__ import annotations


_SGLANG_INSTALL_HINT = (
    "pie_driver_sgl requires SGLang. Install with `pip install pie-server[sglang]` "
    "or `uv sync --extra sglang`."
)


def _require_sglang():
    try:
        import sglang  # noqa: F401
    except ImportError as e:
        raise ImportError(_SGLANG_INSTALL_HINT) from e


from pie.drivers import DriverSpec, register_driver

from .config import SGLangDriverConfig

register_driver(DriverSpec(
    name="sglang",
    config_cls=SGLangDriverConfig,
    worker_module="pie_driver_sgl.worker",
    extras=("sglang",),
))
