#!/usr/bin/env python3
"""
Example PyCrust worker demonstrating various use cases.

To run:
    cd /third-party/pycrust
    pip install -e python/
    cd crates/pycrust-worker && maturin develop
    python examples/python_worker.py
"""

from pydantic import BaseModel
from pycrust import RpcEndpoint


# Define request models with Pydantic validation
class AddRequest(BaseModel):
    a: int
    b: int


class MultiplyRequest(BaseModel):
    x: float
    y: float


class ProcessDataRequest(BaseModel):
    data: list[int]
    operation: str  # "sum", "mean", "max", "min"


# Define handler functions
def ping() -> str:
    """Simple ping-pong method."""
    return "pong"


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(x: float, y: float) -> float:
    """Multiply two floats."""
    return x * y


def process_data(data: list, operation: str) -> dict:
    """Process a list of numbers with the given operation."""
    if not data:
        return {"result": 0, "operation": operation, "count": 0}

    if operation == "sum":
        result = sum(data)
    elif operation == "mean":
        result = sum(data) / len(data)
    elif operation == "max":
        result = max(data)
    elif operation == "min":
        result = min(data)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    return {
        "result": result,
        "operation": operation,
        "count": len(data),
    }


def echo(message: str) -> str:
    """Echo the message back."""
    return f"Echo: {message}"


def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number (for benchmarking)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    # Create the RPC endpoint
    rpc = RpcEndpoint("calculator")

    # Register methods
    rpc.register("ping", ping)
    rpc.register("add", add, request_model=AddRequest)
    rpc.register("multiply", multiply, request_model=MultiplyRequest)
    rpc.register("process_data", process_data, request_model=ProcessDataRequest)
    rpc.register("echo", echo)
    rpc.register("fibonacci", fibonacci)

    print("Starting calculator service...")
    print("Available methods:", list(rpc._methods.keys()))
    rpc.listen()
