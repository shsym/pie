#!/usr/bin/env python3
"""
PyCrust Benchmark Worker - Real-world workload simulation.

This worker provides handlers for benchmarking the full IPC round-trip
from Rust to Python and back.

To run:
    cd /third-party/pycrust
    pip install -e python/
    cd crates/pycrust-worker && maturin develop --release
    python examples/benchmark_worker.py
"""

from typing import Optional
from pydantic import BaseModel
from pycrust import RpcEndpoint


# ==============================================================================
# Request/Response Models (simulating real ML inference workloads)
# ==============================================================================


class NoopRequest(BaseModel):
    """Empty request for measuring baseline latency."""

    pass


class AddRequest(BaseModel):
    """Simple arithmetic request."""

    a: int
    b: int


class ForwardPassRequest(BaseModel):
    """Simulated ML forward pass request."""

    input_tokens: list[int]
    input_token_positions: list[int]
    adapter: Optional[int] = None
    adapter_seed: Optional[int] = None
    mask: list[list[int]]
    kv_page_ptrs: list[int] = []
    kv_page_last_len: int = 0
    output_token_indices: list[int] = []


class HandshakeRequest(BaseModel):
    """Service handshake request."""

    version: str


class BatchProcessRequest(BaseModel):
    """Batch processing request."""

    items: list[dict]
    operation: str  # "sum", "mean", "transform"


class EchoRequest(BaseModel):
    """Echo request with variable payload size."""

    data: bytes | list[int]


# ==============================================================================
# Handler Functions
# ==============================================================================


def noop() -> None:
    """Minimal handler for baseline latency measurement."""
    pass


def add(a: int, b: int) -> int:
    """Simple arithmetic for low-overhead measurement."""
    return a + b


def forward_pass(
    input_tokens: list[int],
    input_token_positions: list[int],
    adapter: Optional[int],
    adapter_seed: Optional[int],
    mask: list[list[int]],
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    output_token_indices: list[int],
) -> dict:
    """
    Simulated ML forward pass.

    In a real scenario, this would invoke PyTorch inference.
    Here we simulate the data processing overhead.
    """
    num_tokens = len(input_tokens)
    num_output = len(output_token_indices) if output_token_indices else 1

    # Simulate output generation
    output_tokens = [(tok + 1) % 32000 for tok in input_tokens[:num_output]]

    # Simulate probability distributions
    dists = [
        ([100, 200, 300, 400, 500], [0.4, 0.3, 0.15, 0.1, 0.05])
        for _ in range(num_output)
    ]

    return {
        "tokens": output_tokens,
        "dists": dists,
        "num_processed": num_tokens,
    }


def handshake(version: str) -> dict:
    """
    Service handshake returning model metadata.

    This simulates returning tokenizer and model configuration.
    """
    return {
        "version": "1.0.0",
        "model_name": "benchmark-model",
        "model_traits": ["text-generation", "benchmark"],
        "model_description": "A benchmark model for PyCrust testing",
        "prompt_template": "{prompt}",
        "prompt_template_type": "raw",
        "prompt_stop_tokens": ["<|end|>"],
        "kv_page_size": 256,
        "max_batch_tokens": 4096,
        "resources": {"0": 16384, "1": 8192},
        "tokenizer_num_vocab": 32000,
        "tokenizer_special_tokens": {"<pad>": 0, "<eos>": 1, "<bos>": 2},
        "tokenizer_split_regex": r"\s+",
        "tokenizer_escape_non_printable": True,
    }


def batch_process(items: list[dict], operation: str) -> dict:
    """
    Process a batch of items.

    Simulates batch processing workloads.
    """
    if operation == "sum":
        total = sum(item.get("value", 0) for item in items)
        return {"result": total, "count": len(items)}
    elif operation == "mean":
        values = [item.get("value", 0) for item in items]
        avg = sum(values) / len(values) if values else 0
        return {"result": avg, "count": len(items)}
    elif operation == "transform":
        transformed = [
            {"id": item.get("id", i), "value": item.get("value", 0) * 2}
            for i, item in enumerate(items)
        ]
        return {"items": transformed, "count": len(items)}
    else:
        raise ValueError(f"Unknown operation: {operation}")


def echo(data: bytes | list[int]) -> dict:
    """
    Echo data back.

    Used for measuring serialization overhead with varying payload sizes.
    """
    if isinstance(data, bytes):
        return {"data": data, "size": len(data)}
    else:
        return {"data": data, "size": len(data)}


def compute_heavy(iterations: int) -> dict:
    """
    CPU-intensive computation.

    Used for measuring overhead when handler does actual work.
    """
    result = 0
    for i in range(iterations):
        result += i * i
    return {"result": result, "iterations": iterations}


def nested_data(depth: int, width: int) -> dict:
    """
    Generate deeply nested data structure.

    Used for measuring serialization overhead with complex structures.
    """

    def build_nested(d: int, w: int) -> dict:
        if d <= 0:
            return {"leaf": True, "values": list(range(w))}
        return {f"level_{d}_{i}": build_nested(d - 1, w) for i in range(w)}

    return build_nested(depth, width)


# ==============================================================================
# Main Entry Point
# ==============================================================================


if __name__ == "__main__":
    # Create the RPC endpoint
    rpc = RpcEndpoint("benchmark_v4")

    # Register all benchmark handlers
    rpc.register("noop", noop)
    rpc.register("add", add, request_model=AddRequest)
    rpc.register("forward_pass", forward_pass, request_model=ForwardPassRequest)
    rpc.register("handshake", handshake, request_model=HandshakeRequest)
    rpc.register("batch_process", batch_process, request_model=BatchProcessRequest)
    rpc.register("echo", echo)
    rpc.register("compute_heavy", compute_heavy)
    rpc.register("nested_data", nested_data)

    print("=" * 60)
    print(" PyCrust Benchmark Worker")
    print("=" * 60)
    print(f"\nService name: benchmark")
    print(f"Registered methods: {list(rpc._methods.keys())}")
    print("\nWaiting for connections...")
    print("Press Ctrl+C to stop.\n")

    rpc.listen()
