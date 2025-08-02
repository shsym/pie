import asyncio
import time
import argparse
import random
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance
import json
from test_utils import append_log


async def launch_and_handle(client, args, program_hash, idx):
    instance_args = [
        "--index", str(idx),
        "--layer", str(args.layer),
    ]

    instance = await client.launch_instance(program_hash, arguments=instance_args)

    return True


async def main(args):
    program_name = "bench_execution_latency"
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
        results = await asyncio.gather(*tasks)

        # wait for a bit, so that the instances are ready
        await asyncio.sleep(0.5)

        aggregator = await client.launch_instance(program_hash, arguments=[
            "--aggregate-size", str(args.num_instances),
        ])

        event, result = await aggregator.recv()
        result = json.loads(result)

        print("\n--- ✅ Benchmark Complete ---")
        print(f"mean latency:   {result['mean']:.2f} μs")
        print(f"median latency: {result['median']:.2f} μs")
        print(f"stdev latency:  {result['std_dev']:.2f} μs")

        print(f"Received result: {result}")
        # --- 4. Print Results ---

        append_log('./logs/microbench_execution_latency.json', {
            'mean_latency': result['mean'],
            'median_latency': result['median'],
            'stdev_latency': result['std_dev'],
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
    wasm_args_group = parser.add_argument_group('WASM Program Arguments')
    wasm_args_group.add_argument("--layer", type=str, default="inference", help="control or inference")

    args = parser.parse_args()
    asyncio.run(main(args))
