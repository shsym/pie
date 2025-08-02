import asyncio
import time
import argparse
import random
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance
import statistics

from test_utils import append_log

async def launch_and_handle(client, args, program_hash, idx):
    instance_args = [
        "--message", f"{idx}-th instance launched!!"
    ]

    start_time_ns = time.monotonic_ns()
    instance = await client.launch_instance(program_hash, arguments=instance_args)
    elapsed_time_ns = time.monotonic_ns() - start_time_ns

    # Convert nanoseconds to microseconds and return
    return elapsed_time_ns / 1000.0


async def main(args):
    program_name = 'bench_spawn_time'
    program_path = Path(f"../example-apps/target/wasm32-wasip2/release/{program_name}.wasm")

    if not program_path.exists():
        print(f"Error: Program file not found at path: {program_path}")
        return

    async with PieClient(args.server_uri) as client:
        with open(program_path, "rb") as f:
            program_bytes = f.read()
        program_hash = blake3(program_bytes).hexdigest()

        if not await client.program_exists(program_hash):
            print("Program not found on server, uploading...")
            await client.upload_program(program_bytes)
            print("Upload complete.")

        print(f"Starting benchmark with {args.num_instances} total inferlet instances...")


        tasks = [launch_and_handle(client, args, program_hash, idx) for idx in range(args.num_instances)]
        results_in_us = await asyncio.gather(*tasks)

        mean_latency = statistics.mean(results_in_us)
        median_latency = statistics.median(results_in_us)
        stdev_latency = statistics.stdev(results_in_us) if len(results_in_us) > 1 else 0.0

        print("\n--- ✅ Benchmark Complete ---")
        print(f"mean latency:   {mean_latency:.2f} μs")
        print(f"median latency: {median_latency:.2f} μs")
        print(f"stdev latency:  {stdev_latency:.2f} μs")

        append_log('./logs/microbench_spawn_time.json', {
            'mean_latency': mean_latency,
            'median_latency': median_latency,
            'stdev_latency': stdev_latency,
            'args': vars(args),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Server & Program Configuration')
    server_group.add_argument("--server-uri", type=str, default="ws://127.0.0.1:8080", help="WebSocket URI for the Pie server.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-instances", type=int, default=1000, help="Total number of concurrent instances to launch.")

    args = parser.parse_args()
    asyncio.run(main(args))
