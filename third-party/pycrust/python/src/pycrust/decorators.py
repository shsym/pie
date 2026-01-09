"""Convenience decorators for RPC methods."""

from functools import wraps
from typing import Any, Callable, Optional, Type, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from .endpoint import RpcEndpoint


def rpc_method(
    name: Optional[str] = None,
    request_model: Optional[Type[BaseModel]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Standalone decorator for marking RPC methods.

    This decorator adds metadata to functions that can be used
    when registering them with an RpcEndpoint via auto_register.

    Args:
        name: Optional method name (defaults to function name)
        request_model: Optional Pydantic model for validation

    Returns:
        Decorated function with RPC metadata

    Example:
        @rpc_method()
        def my_handler(x: int) -> int:
            return x * 2

        @rpc_method(name="custom", request_model=MyModel)
        def validated_handler(a: int, b: str) -> dict:
            return {"a": a, "b": b}
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        # Attach metadata
        wrapper._rpc_name = name or func.__name__  # type: ignore[attr-defined]
        wrapper._rpc_request_model = request_model  # type: ignore[attr-defined]
        wrapper._is_rpc_method = True  # type: ignore[attr-defined]

        return wrapper

    return decorator


def auto_register(
    endpoint: "RpcEndpoint",
) -> Callable[[type], type]:
    """
    Class decorator to auto-register all @rpc_method decorated methods.

    This decorator scans a class for methods marked with @rpc_method
    and automatically registers them with the provided endpoint.

    Args:
        endpoint: The RpcEndpoint to register methods with

    Returns:
        Class decorator

    Example:
        endpoint = RpcEndpoint("my_service")

        @auto_register(endpoint)
        class MyService:
            @rpc_method()
            def add(self, a: int, b: int) -> int:
                return a + b

            @rpc_method(name="multiply")
            def mul(self, x: int, y: int) -> int:
                return x * y

        # Methods are now registered and endpoint.listen() can be called
    """

    def decorator(cls: type) -> type:
        instance = cls()

        for attr_name in dir(instance):
            if attr_name.startswith("_"):
                continue

            attr = getattr(instance, attr_name)
            if callable(attr) and getattr(attr, "_is_rpc_method", False):
                name = getattr(attr, "_rpc_name", attr_name)
                model = getattr(attr, "_rpc_request_model", None)
                endpoint._methods[name] = attr
                endpoint._schemas[name] = model

        return cls

    return decorator
