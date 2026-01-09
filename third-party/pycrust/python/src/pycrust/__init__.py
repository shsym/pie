"""
PyCrust - High-performance RPC framework for Rust-Python communication

This package provides a clean API for building RPC workers
that communicate with Rust clients over shared memory.

Example (without validation):
    from pycrust import RpcEndpoint

    endpoint = RpcEndpoint("calculator")

    def add(a: int, b: int) -> int:
        return a + b

    endpoint.register("add", add)
    endpoint.listen()

Example (with Pydantic validation - requires pydantic):
    from pycrust import RpcEndpoint
    from pydantic import BaseModel

    class AddArgs(BaseModel):
        a: int
        b: int

    endpoint = RpcEndpoint("calculator")

    def add(a: int, b: int) -> int:
        return a + b

    endpoint.register("add", add, request_model=AddArgs)
    endpoint.listen()
"""

__version__ = "0.1.0"

from .endpoint import RpcEndpoint

# Optional imports that require pydantic
try:
    from .decorators import rpc_method, auto_register
    from .validation import validate_args, ValidationError

    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

    # Provide stub implementations that raise helpful errors
    def rpc_method(*args, **kwargs):
        raise ImportError(
            "rpc_method requires pydantic. Install with: pip install pydantic"
        )

    def auto_register(*args, **kwargs):
        raise ImportError(
            "auto_register requires pydantic. Install with: pip install pydantic"
        )

    def validate_args(*args, **kwargs):
        raise ImportError(
            "validate_args requires pydantic. Install with: pip install pydantic"
        )

    class ValidationError(Exception):
        """Stub for ValidationError when pydantic is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ValidationError requires pydantic. Install with: pip install pydantic"
            )


__all__ = [
    "__version__",
    "RpcEndpoint",
    "rpc_method",
    "auto_register",
    "validate_args",
    "ValidationError",
]
