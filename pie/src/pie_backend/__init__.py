# Pie Backend Python Package — `native` driver.
from __future__ import annotations

from pie.drivers import DriverSpec, register_driver

from .config import NativeDriverConfig

register_driver(DriverSpec(
    name="native",
    config_cls=NativeDriverConfig,
    worker_module="pie_backend.worker",
    extras=("cu126", "cu128", "metal"),
))
