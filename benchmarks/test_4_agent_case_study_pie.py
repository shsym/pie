#!/usr/bin/env python3
import asyncio
import time
import argparse
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance  # Assuming pie.py is in the environment

from test_utils import append_log


async def launch_and_handle(client, args, program_hash):
    instance_args = [
        "--tokens-between-calls", str(args.tokens_between_calls),
        "--function-call-delay", str(args.function_call_delay),
    ]
    if args.use_prefix_cache:
        instance_args.append("--use-prefix-cache")
    if args.drop_tool_cache:
        instance_args.append("--drop-tool-cache")
    if args.concurrent_calls:
        instance_args.append("--concurrent-calls")

    instance = await client.launch_instance(program_hash, arguments=instance_args)

    while True:
        event, message = await instance.recv()
        if event == "terminated":
            if args.verbose:
                print(f"Instance {instance.instance_id} finished. Reason: {message}")
            break
        else:
            if args.verbose:
                print(f"Instance {instance.instance_id} received message '{message}'")

    return True


async def main(args):
    program_name = args.program_name
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

        start_time = time.monotonic()

        tasks = [launch_and_handle(client, args, program_hash) for _ in range(args.num_instances)]
        results = await asyncio.gather(*tasks)

        total_time = time.monotonic() - start_time
        throughput = args.num_instances / total_time if total_time > 0 else 0

        print("\n--- ✅ Benchmark Complete ---")
        print(f"Total Time Taken:       {total_time * 1000:.2f} milliseconds")
        print(f"Throughput:             {throughput:.2f} requests/second")
        print("--------------------------")

        append_log('./logs/test_4_agent_case_study_pie.json', {
            'total_time': total_time,
            'throughput': throughput,
            'args': vars(args),
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--server-uri", type=str, default="ws://127.0.0.1:8080", help="PIE server URI")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--program-name", type=str, default="agent_react_bench", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=128, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", type=bool, default=False, help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("-t", "--tokens-between-calls", type=int, default=32, help="Argument for WASM: Max tokens for each Thought/Action step.")
    wasm_args_group.add_argument("-d", "--function-call-delay", type=int, default=100, help="Argument for WASM: Simulated delay in milliseconds for tool execution.")
    wasm_args_group.add_argument("--use-prefix-cache", action='store_true', help="Argument for WASM: Enable caching of the initial system prompt.")
    wasm_args_group.add_argument("--drop-tool-cache", action='store_true', help="Argument for WASM: Drop the first tool's KV cache after use.")
    wasm_args_group.add_argument("--concurrent-calls", action='store_true', help="Argument for WASM: Simulate concurrent execution of function calls.")

    args = parser.parse_args()
    asyncio.run(main(args))

#
# --- ✅ Benchmark Complete ---
# Total Time Taken:       9296.52 milliseconds
# Throughput:             13.77 requests/second
# --------------------------

# Prefix cache on
# --- ✅ Benchmark Complete ---
# Total Time Taken:       6439.26 milliseconds
# Throughput:             19.88 requests/second
# --------------------------

# Prefix cache + tool drop
# --- ✅ Benchmark Complete ---
# Total Time Taken:       6065.97 milliseconds
# Throughput:             21.10 requests/second
# --------------------------

# Prefix cache + tool drop + concurrent calls
# --- ✅ Benchmark Complete ---
# Total Time Taken:       3331.73 milliseconds
# Throughput:             38.42 requests/second
# --------------------------
