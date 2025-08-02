import asyncio
import time
import argparse
import random
from pathlib import Path
from blake3 import blake3
from pie import PieClient, Instance

from test_utils import append_log

AGENT_ROLES = [
    "idea_generator",
    "plot_developer",
    "character_creator",
    "dialogue_writer",
]


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

        # Check if the program exists on the server; upload if not
        if not await client.program_exists(program_hash):
            await client.upload_program(program_bytes)

        print(f"Starting benchmark with {args.num_pipelines * len(AGENT_ROLES)} total inferlet instances...")
        start_time = time.monotonic()

        pipelines = []
        all_pipeline_instance_futures = []

        for i in range(args.num_pipelines):
            pipeline_instance_futures = {}
            for role in AGENT_ROLES:
                instance_args = [
                    role,
                    "--group-id", str(i),
                    "--tokens-per-step", str(args.tokens_per_step),
                ]
                pipeline_instance_futures[role] = client.launch_instance(program_hash, arguments=instance_args)

            all_pipeline_instance_futures.append(pipeline_instance_futures)

        for pipeline_futures_dict in all_pipeline_instance_futures:
            resolved_instances = await asyncio.gather(*pipeline_futures_dict.values())

            pipeline_instances = {}
            for role, instance in zip(pipeline_futures_dict.keys(), resolved_instances):
                pipeline_instances[role] = instance

            pipelines.append(pipeline_instances)

        async def send_initial_prompt(pipeline_idx: int):
            first_agent_instance = pipelines[pipeline_idx]["idea_generator"]
            genres = ['a haunted spaceship', 'a detective who can talk to ghosts', 'a romance in a city that floats']
            prompt = random.choice(genres)
            await first_agent_instance.send(prompt)

        async def await_final_story(pipeline_idx: int):
            last_agent_instance = pipelines[pipeline_idx]["dialogue_writer"]
            event, message = await last_agent_instance.recv()
            if event == "message":
                return message
            else:
                print(f"Pipeline {pipeline_idx} terminated unexpectedly with event: {event}, msg: {message}")
                return None

        # Create and run all tasks concurrently
        sender_tasks = [send_initial_prompt(i) for i in range(args.num_pipelines)]
        receiver_tasks = [await_final_story(i) for i in range(args.num_pipelines)]

        await asyncio.gather(*sender_tasks)  # Ensure all initial prompts are sent
        await asyncio.gather(*receiver_tasks)  # Wait for all final stories

        total_time = time.monotonic() - start_time
        throughput = args.num_pipelines / total_time

        print("\n--- ✅ Benchmark Complete ---")
        print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
        print(f"Throughput:       {throughput:.2f} requests/second")
        print("--------------------------")

        append_log('./logs/test_3_agent_swarm_pie.json', {
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
    benchmark_group.add_argument("--program-name", type=str, default="agent_swarm", help="Name of the WASM program file (without .wasm extension).")
    benchmark_group.add_argument("--num-pipelines", type=int, default=32, help="Total number of concurrent story-writing pipelines to run.")

    wasm_args_group = parser.add_argument_group('Inferlet Arguments')
    wasm_args_group.add_argument("--tokens-per-step", type=int, default=96, help="Argument for WASM program: max tokens each agent generates.")

    args = parser.parse_args()
    asyncio.run(main(args))

# --- ✅ Benchmark Complete ---
# Total Time Taken: 6147.32 milliseconds
# Throughput:       5.21 requests/second
# --------------------------