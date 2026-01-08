#!/usr/bin/env python3
"""
Benchmark comparing GLOO broadcast_object_list vs IPC (multiprocessing.Queue).

This script measures the latency of broadcasting metadata structures between
processes using two different approaches:
1. GLOO: PyTorch distributed broadcast_object_list
2. IPC: multiprocessing.Queue

Run with: python test/benchmark_ipc_vs_gloo.py
"""

from __future__ import annotations

import os
import time
import pickle
import statistics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Queue


# Test payload (simulates typical metadata broadcast in runtime)
def create_test_payload(include_tensor_refs: bool = True):
    """Create a realistic metadata payload similar to what runtime broadcasts."""
    payload = {
        "type": "STEP",
        "inputs": {
            "token_ids": [1, 2, 3, 4, 5, 100, 200, 300],
            "position_ids": [0, 1, 2, 3, 4, 5, 6, 7],
            "qo_indptr": [0, 8],
            "kv_page_indices": [0, 1, 2, 3],
            "kv_page_indptr": [0, 4],
            "kv_last_page_lens": [8],
            "single_token_inference_mode": False,
            "adapter_indices": [],
            "adapter_seeds": None,
            "total_pages_cpu": 4,
        },
        "sampling_metadata": {
            "indices_for_logits": [7],
            "sampler_groups": {1: [0]},
            "sampler_params": [{"temperature": 0.7, "top_k": 50, "top_p": 0.95}],
        },
    }

    if include_tensor_refs:
        # Add tensor placeholders (similar to what broadcast_struct uses)
        payload["inputs"]["__TENSOR_0__"] = {
            "__TENSOR__": 0,
            "shape": (8,),
            "dtype": torch.long,
        }

    return payload


def benchmark_gloo_worker(
    rank: int, world_size: int, port: int, iterations: int, results_queue: Queue
):
    """Worker for GLOO benchmark."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Initialize with GLOO backend
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    payload = create_test_payload()
    latencies = []

    # Warmup
    for _ in range(10):
        if rank == 0:
            data = [payload]
        else:
            data = [None]
        dist.broadcast_object_list(data, src=0)

    # Barrier to sync before timing
    dist.barrier()

    # Benchmark
    for _ in range(iterations):
        if rank == 0:
            data = [payload]
            start = time.perf_counter_ns()
            dist.broadcast_object_list(data, src=0)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1000)  # Convert to μs
        else:
            data = [None]
            dist.broadcast_object_list(data, src=0)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        results_queue.put(latencies)


def benchmark_ipc_worker(
    rank: int,
    world_size: int,
    queues: list[Queue],
    iterations: int,
    results_queue: Queue,
):
    """Worker for IPC (Queue) benchmark."""
    payload = create_test_payload()
    latencies = []

    # Warmup
    for _ in range(10):
        if rank == 0:
            for q in queues:
                q.put(payload)
        else:
            queues[rank - 1].get()

    # Benchmark
    for _ in range(iterations):
        if rank == 0:
            start = time.perf_counter_ns()
            for q in queues:
                q.put(payload)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1000)  # Convert to μs
        else:
            queues[rank - 1].get()

    if rank == 0:
        results_queue.put(latencies)


def run_gloo_benchmark(world_size: int, iterations: int) -> list[float]:
    """Run GLOO benchmark and return latencies in μs."""
    import random

    port = 29500 + random.randint(0, 1000)
    results_queue = Queue()

    mp.spawn(
        benchmark_gloo_worker,
        args=(world_size, port, iterations, results_queue),
        nprocs=world_size,
        join=True,
    )

    return results_queue.get()


def run_ipc_benchmark(world_size: int, iterations: int) -> list[float]:
    """Run IPC benchmark and return latencies in μs."""
    results_queue = Queue()
    queues = [Queue() for _ in range(world_size - 1)]

    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_ipc_worker,
            args=(rank, world_size, queues, iterations, results_queue),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return results_queue.get()


def print_stats(name: str, latencies: list[float]):
    """Print benchmark statistics."""
    mean = statistics.mean(latencies)
    median = statistics.median(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    throughput = 1_000_000 / mean  # broadcasts/sec

    print(f"\n{name}:")
    print(f"  Mean:       {mean:>10.2f} μs")
    print(f"  Median:     {median:>10.2f} μs")
    print(f"  Std Dev:    {stdev:>10.2f} μs")
    print(f"  P95:        {p95:>10.2f} μs")
    print(f"  P99:        {p99:>10.2f} μs")
    print(f"  Throughput: {throughput:>10.0f} broadcasts/sec")

    return mean


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GLOO vs IPC")
    parser.add_argument("--world-size", type=int, default=2, help="Number of processes")
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Number of iterations"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("GLOO vs IPC Benchmark")
    print("=" * 60)
    print(f"World size: {args.world_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Payload size: {len(pickle.dumps(create_test_payload()))} bytes")

    mp.set_start_method("spawn", force=True)

    # Run GLOO benchmark
    print("\n[1/2] Running GLOO benchmark...")
    gloo_latencies = run_gloo_benchmark(args.world_size, args.iterations)
    gloo_mean = print_stats("GLOO (broadcast_object_list)", gloo_latencies)

    # Run IPC benchmark
    print("\n[2/2] Running IPC benchmark...")
    ipc_latencies = run_ipc_benchmark(args.world_size, args.iterations)
    ipc_mean = print_stats("IPC (multiprocessing.Queue)", ipc_latencies)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = gloo_mean / ipc_mean
    if speedup > 1:
        print(f"IPC is {speedup:.2f}x faster than GLOO")
    else:
        print(f"GLOO is {1/speedup:.2f}x faster than IPC")

    print("\n✓ Benchmark complete")


if __name__ == "__main__":
    main()
