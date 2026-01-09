#!/usr/bin/env python3
"""
Performance benchmark for PyCrust.

This script measures the latency and throughput of the PyCrust RPC framework.

Usage:
    1. Start the Python worker in one terminal:
       python examples/python_worker.py

    2. Run this benchmark in another terminal:
       python examples/benchmark.py

Note: This requires the pycrust-client Rust binary to be built as a Python extension
or a separate benchmark client. For now, this benchmarks the Python SDK dispatch only.
"""

import time
import statistics
from pydantic import BaseModel
from pycrust import RpcEndpoint


class AddArgs(BaseModel):
    a: int
    b: int


def benchmark_dispatch_latency(iterations: int = 10000) -> dict:
    """Benchmark the dispatch latency without IPC (pure Python overhead)."""
    endpoint = RpcEndpoint("benchmark")

    @endpoint.register()
    def noop() -> None:
        pass

    @endpoint.register()
    def add(a: int, b: int) -> int:
        return a + b

    @endpoint.register(request_model=AddArgs)
    def add_validated(a: int, b: int) -> int:
        return a + b

    results = {}

    # Benchmark noop (minimal overhead)
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        endpoint._dispatch("noop", {})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)  # Convert to microseconds

    results["noop"] = {
        "mean_us": statistics.mean(latencies),
        "median_us": statistics.median(latencies),
        "p99_us": statistics.quantiles(latencies, n=100)[98],
        "min_us": min(latencies),
        "max_us": max(latencies),
    }

    # Benchmark add (with arguments, no validation)
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        endpoint._dispatch("add", {"a": 10, "b": 20})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)

    results["add"] = {
        "mean_us": statistics.mean(latencies),
        "median_us": statistics.median(latencies),
        "p99_us": statistics.quantiles(latencies, n=100)[98],
        "min_us": min(latencies),
        "max_us": max(latencies),
    }

    # Benchmark add with Pydantic validation
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        endpoint._dispatch("add_validated", {"a": 10, "b": 20})
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1000)

    results["add_validated"] = {
        "mean_us": statistics.mean(latencies),
        "median_us": statistics.median(latencies),
        "p99_us": statistics.quantiles(latencies, n=100)[98],
        "min_us": min(latencies),
        "max_us": max(latencies),
    }

    return results


def benchmark_throughput(duration_seconds: float = 5.0) -> dict:
    """Benchmark throughput (calls per second)."""
    endpoint = RpcEndpoint("benchmark")

    @endpoint.register()
    def add(a: int, b: int) -> int:
        return a + b

    results = {}

    # Measure throughput
    count = 0
    start = time.perf_counter()
    end_time = start + duration_seconds

    while time.perf_counter() < end_time:
        endpoint._dispatch("add", {"a": 10, "b": 20})
        count += 1

    elapsed = time.perf_counter() - start
    results["throughput"] = {
        "calls": count,
        "duration_s": elapsed,
        "calls_per_second": count / elapsed,
    }

    return results


def print_results(title: str, results: dict) -> None:
    """Pretty print benchmark results."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n  {name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                if "us" in metric:
                    print(f"    {metric}: {value:.2f}")
                else:
                    print(f"    {metric}: {value:.2f}")
            else:
                print(f"    {metric}: {value}")


def main():
    print("PyCrust Performance Benchmark")
    print("=" * 60)
    print("\nNote: This benchmarks the Python SDK dispatch only.")
    print("Full IPC benchmarks require the Rust client.\n")

    # Warmup
    print("Warming up...")
    benchmark_dispatch_latency(iterations=1000)

    # Run latency benchmark
    print("Running latency benchmark (10,000 iterations)...")
    latency_results = benchmark_dispatch_latency(iterations=10000)
    print_results("Dispatch Latency (microseconds)", latency_results)

    # Run throughput benchmark
    print("\nRunning throughput benchmark (5 seconds)...")
    throughput_results = benchmark_throughput(duration_seconds=5.0)
    print_results("Dispatch Throughput", throughput_results)

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"\n  Median noop latency: {latency_results['noop']['median_us']:.2f} us")
    print(f"  Median add latency: {latency_results['add']['median_us']:.2f} us")
    print(
        f"  Median validated latency: {latency_results['add_validated']['median_us']:.2f} us"
    )
    print(
        f"  Throughput: {throughput_results['throughput']['calls_per_second']:.0f} calls/s"
    )
    print()


if __name__ == "__main__":
    main()
