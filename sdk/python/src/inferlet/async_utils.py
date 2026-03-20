"""
Async utilities for WASI pollable futures.
"""

from typing import Protocol, TypeVar, runtime_checkable

from wit_world.imports.poll import Pollable

T = TypeVar("T")


@runtime_checkable
class WasiFuture(Protocol[T]):
    """Generic interface for WASI async operations."""

    def pollable(self) -> Pollable: ...
    def get(self) -> T | None: ...


def await_future(future: WasiFuture[T], error_message: str) -> T:
    """
    Awaits a WASI future by blocking on its pollable.

    Args:
        future: The WASI future to await
        error_message: Error message if result is None

    Returns:
        The resolved value

    Raises:
        RuntimeError: If the future returns None
    """
    pollable = future.pollable()
    pollable.block()

    result = future.get()
    if result is None:
        raise RuntimeError(error_message)
    return result
