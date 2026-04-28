"""Dummy driver — registers `type = "dummy"` for tests and benchmarking.

Skips weight loading and runs random-token generation through the same
pie_backend machinery. Useful for verifying scheduler/runtime plumbing
without needing a real GPU model load.
"""

from __future__ import annotations

from pie.drivers import DriverSpec, register_driver

from .config import DummyDriverConfig

register_driver(DriverSpec(
    name="dummy",
    config_cls=DummyDriverConfig,
    worker_module="pie_backend_dummy.worker",
    extras=(),
))
