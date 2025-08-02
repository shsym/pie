import asyncio
import time
import argparse
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance

from test_utils import append_log

ebnf_sql_grammar = """
    ?start: select_statement

    ?select_statement: "SELECT " column_list " FROM " table_name

    ?column_list: column_name ("," column_name)*

    ?table_name: identifier

    ?column_name: identifier

    ?identifier: /[a-zA-Z_][a-zA-Z0-9_]*/
"""

ebnf_json_grammar = """
    ?start: value
    ?value: object
            | array
            | string
            | SIGNED_NUMBER      -> number
            | "true"             -> true
            | "false"            -> false
            | "null"             -> null
    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value
    string : ESCAPED_STRING
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


async def launch_and_handle(client, args, program_hash):
    instance_args = [
        "--prompt", args.prompt,
        "--grammar", ebnf_json_grammar,
        "--max-tokens", str(args.max_tokens),
    ]
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

        append_log('./logs/test_12_ebnf_pie.json', {
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
    benchmark_group.add_argument("--program-name", type=str, default="constrained_decoding", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-instances", type=int, default=128, help="Total number of concurrent instances to launch.")
    benchmark_group.add_argument("--verbose", type=bool, default=False, help="Enable verbose output for debugging.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("--prompt", type=str, default="Generate a long random json data", help="Base prompt to send to the WASM program.")
    wasm_args_group.add_argument("--max-tokens", type=int, default=32, help="Argument for WASM program: max tokens to generate.")

    args = parser.parse_args()
    asyncio.run(main(args))

#
# --- ✅ Benchmark Complete ---
# Total Time Taken:       6372.99 milliseconds
# Throughput:             20.08 requests/second
# --------------------------