#!/usr/bin/env python3
"""
PyCrust Latency Breakdown - Instrument each component.

This script measures:
1. Pure Python function call overhead
2. Pydantic validation overhead
3. MessagePack serialization/deserialization
4. pythonize conversion (measured via native extension)
"""

import time
import msgpack
from pydantic import BaseModel
from pycrust import RpcEndpoint


class AddRequest(BaseModel):
    a: int
    b: int


def noop():
    pass


def add(a: int, b: int) -> int:
    return a + b


def measure_python_call_overhead(iterations: int = 100000):
    """Measure pure Python function call overhead."""
    start = time.perf_counter()
    for _ in range(iterations):
        noop()
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000  # Convert to µs


def measure_python_call_with_args(iterations: int = 100000):
    """Measure Python function call with kwargs unpacking."""
    args = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        add(**args)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_pydantic_validation(iterations: int = 100000):
    """Measure Pydantic validation overhead."""
    data = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        validated = AddRequest(**data)
        _ = validated.model_dump()
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_msgpack_round_trip(iterations: int = 100000):
    """Measure MessagePack serialization round-trip."""
    data = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        packed = msgpack.packb(data)
        _ = msgpack.unpackb(packed)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_msgpack_serialize(iterations: int = 100000):
    """Measure MessagePack serialization only."""
    data = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        _ = msgpack.packb(data)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_msgpack_deserialize(iterations: int = 100000):
    """Measure MessagePack deserialization only."""
    packed = msgpack.packb({"a": 1, "b": 2})
    start = time.perf_counter()
    for _ in range(iterations):
        _ = msgpack.unpackb(packed)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_endpoint_dispatch(iterations: int = 100000):
    """Measure full endpoint dispatch (without IPC)."""
    endpoint = RpcEndpoint("test")
    endpoint.register("add", add)

    args = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        _ = endpoint._dispatch("add", args)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_endpoint_dispatch_with_validation(iterations: int = 100000):
    """Measure endpoint dispatch with Pydantic validation."""
    endpoint = RpcEndpoint("test")
    endpoint.register("add", add, request_model=AddRequest)

    args = {"a": 1, "b": 2}
    start = time.perf_counter()
    for _ in range(iterations):
        _ = endpoint._dispatch("add", args)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


def measure_complex_payload_msgpack(iterations: int = 10000):
    """Measure MessagePack with complex forward_pass-like payload."""
    data = {
        "input_tokens": list(range(128)),
        "input_token_positions": list(range(128)),
        "adapter": 1,
        "adapter_seed": 42,
        "mask": [[1] * 128 for _ in range(128)],
        "kv_page_ptrs": [0x1000, 0x2000, 0x3000],
        "kv_page_last_len": 64,
        "output_token_indices": [127],
    }

    start = time.perf_counter()
    for _ in range(iterations):
        packed = msgpack.packb(data)
        _ = msgpack.unpackb(packed)
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1_000_000


if __name__ == "__main__":
    print("=" * 60)
    print(" PyCrust Python-Side Latency Breakdown")
    print("=" * 60)

    iterations = 100000

    print(f"\nMeasuring with {iterations} iterations each...\n")

    print("Component Latencies (µs):")
    print("-" * 40)

    # Basic Python overhead
    py_noop = measure_python_call_overhead(iterations)
    print(f"  Python noop() call:           {py_noop:.3f} µs")

    py_args = measure_python_call_with_args(iterations)
    print(f"  Python add(**kwargs):         {py_args:.3f} µs")

    # Pydantic
    pydantic = measure_pydantic_validation(iterations)
    print(f"  Pydantic validation:          {pydantic:.3f} µs")

    # MessagePack
    print()
    msgpack_ser = measure_msgpack_serialize(iterations)
    print(f"  MessagePack serialize:        {msgpack_ser:.3f} µs")

    msgpack_de = measure_msgpack_deserialize(iterations)
    print(f"  MessagePack deserialize:      {msgpack_de:.3f} µs")

    msgpack_rt = measure_msgpack_round_trip(iterations)
    print(f"  MessagePack round-trip:       {msgpack_rt:.3f} µs")

    # Complex payload
    print()
    complex_msgpack = measure_complex_payload_msgpack(10000)
    print(f"  Complex payload msgpack RT:   {complex_msgpack:.3f} µs")

    # Endpoint dispatch
    print()
    dispatch = measure_endpoint_dispatch(iterations)
    print(f"  Endpoint dispatch (no val):   {dispatch:.3f} µs")

    dispatch_val = measure_endpoint_dispatch_with_validation(iterations)
    print(f"  Endpoint dispatch (w/ val):   {dispatch_val:.3f} µs")

    print()
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    print()
    print("Estimated Python-side overhead per RPC call:")
    print(f"  - Without validation: ~{dispatch:.1f} µs")
    print(f"  - With validation:    ~{dispatch_val:.1f} µs")
    print()
    print("Remaining latency is from:")
    print("  - iceoryx2 IPC (should be <1µs for shared memory)")
    print("  - Polling sleep loops (50-100µs per side)")
    print("  - Rust MessagePack serialization")
    print("  - pythonize conversion")
    print("  - Channel communication (mpsc)")
