"""
PyCrust FFI - Python worker for direct FFI-based RPC.

This module provides a simple worker registration system for
handling RPC calls from Rust via direct FFI.
"""

import msgpack
from typing import Callable, Any

# Status codes (must match Rust)
STATUS_OK = 0
STATUS_METHOD_NOT_FOUND = 1
STATUS_INVALID_PARAMS = 2
STATUS_INTERNAL_ERROR = 3


class Worker:
    """Worker that handles RPC calls from Rust.

    Example:
        worker = Worker()

        @worker.register("add")
        def add(a: int, b: int) -> int:
            return a + b

        # Pass worker.dispatch to Rust FfiClient
        client = FfiClient(worker.dispatch)
    """

    def __init__(self):
        self._methods: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """Register a method handler.

        Args:
            name: Method name to register

        Returns:
            Decorator function
        """

        def decorator(fn: Callable) -> Callable:
            self._methods[name] = fn
            return fn

        return decorator

    def dispatch(self, method: str, payload: bytes) -> tuple[int, bytes]:
        """Dispatch an RPC call.

        Args:
            method: Method name to call
            payload: MessagePack-encoded arguments

        Returns:
            Tuple of (status_code, response_bytes)
        """
        fn = self._methods.get(method)
        if fn is None:
            return (
                STATUS_METHOD_NOT_FOUND,
                msgpack.packb(f"Method not found: {method}"),
            )

        try:
            args = msgpack.unpackb(payload)

            # Handle different argument formats
            if isinstance(args, dict):
                result = fn(**args)
            elif isinstance(args, (list, tuple)):
                result = fn(*args)
            else:
                result = fn(args)

            return (STATUS_OK, msgpack.packb(result))

        except TypeError as e:
            return (STATUS_INVALID_PARAMS, msgpack.packb(str(e)))
        except Exception as e:
            return (STATUS_INTERNAL_ERROR, msgpack.packb(str(e)))
