"""Tests for decorators."""

import pytest
from pydantic import BaseModel

from pycrust import RpcEndpoint, rpc_method, auto_register


class RequestModel(BaseModel):
    value: int


def test_rpc_method_decorator():
    """Test the @rpc_method decorator adds metadata."""

    @rpc_method()
    def my_handler(x: int) -> int:
        return x * 2

    assert hasattr(my_handler, "_is_rpc_method")
    assert my_handler._is_rpc_method is True
    assert my_handler._rpc_name == "my_handler"
    assert my_handler._rpc_request_model is None

    # Function should still work normally
    assert my_handler(5) == 10


def test_rpc_method_with_custom_name():
    """Test @rpc_method with custom name."""

    @rpc_method(name="custom_handler")
    def my_handler(x: int) -> int:
        return x * 2

    assert my_handler._rpc_name == "custom_handler"


def test_rpc_method_with_request_model():
    """Test @rpc_method with request model."""

    @rpc_method(request_model=RequestModel)
    def my_handler(value: int) -> int:
        return value * 2

    assert my_handler._rpc_request_model is RequestModel


def test_auto_register_class():
    """Test @auto_register class decorator."""
    endpoint = RpcEndpoint("test_service")

    @auto_register(endpoint)
    class MyService:
        @rpc_method()
        def add(self, a: int, b: int) -> int:
            return a + b

        @rpc_method(name="multiply")
        def mul(self, x: int, y: int) -> int:
            return x * y

        def not_rpc_method(self) -> None:
            pass

    # Check that RPC methods are registered
    assert "add" in endpoint._methods
    assert "multiply" in endpoint._methods
    assert "mul" not in endpoint._methods  # Uses custom name
    assert "not_rpc_method" not in endpoint._methods

    # Test calling registered methods
    assert endpoint._dispatch("add", {"a": 5, "b": 3}) == 8
    assert endpoint._dispatch("multiply", {"x": 4, "y": 7}) == 28


def test_auto_register_with_request_model():
    """Test @auto_register with request models."""
    endpoint = RpcEndpoint("test_service")

    @auto_register(endpoint)
    class Calculator:
        @rpc_method(request_model=RequestModel)
        def double(self, value: int) -> int:
            return value * 2

    assert endpoint._schemas["double"] is RequestModel
    # Validation happens during dispatch
    assert endpoint._dispatch("double", {"value": 21}) == 42


def test_auto_register_private_methods_ignored():
    """Test that private methods are not registered."""
    endpoint = RpcEndpoint("test_service")

    @auto_register(endpoint)
    class Service:
        @rpc_method()
        def public_method(self) -> str:
            return "public"

        def _private_method(self) -> str:
            return "private"

        def __dunder_method__(self) -> str:
            return "dunder"

    assert "public_method" in endpoint._methods
    assert "_private_method" not in endpoint._methods
    assert "__dunder_method__" not in endpoint._methods
