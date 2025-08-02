#!/usr/bin/env python3
import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from tqdm import tqdm
from test_utils import get_call_generate, append_log  # Assuming test_utils.py is in the same directory

BOS_TOKEN = "<|begin_of_text|>"
EOT_ID = "<|eot_id|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = "<|eot_id|>"

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
PROMPT_DIVIDE_TEMPLATE = (
    "Your task is to analyze the given problem and decide whether it can be solved directly or needs to be divided "
    "into smaller subproblems. If the problem is simple and can be solved immediately, provide the solution wrapped "
    "in <leaf>final answer</leaf>. If not, divide the problem into exactly two independent subtasks such that solving "
    "these subtasks and combining their solutions will lead to the solution of the original problem. Present the "
    "subtasks wrapped in <branch>subtask1</branch> and <branch>subtask2</branch>. Be concise and ensure the subtasks "
    "are distinct and solvable. Problem: {task}"
)
PROMPT_SOLVE = "Now, please solve the problem. Reason step-by-step. Make your response short."
PROMPT_MERGE = "Now, please merge the two solutions into one. Make your response short."


def parse_subtasks(response: str) -> Tuple[str, str]:
    pattern = r"<branch>(.*?)</branch>"
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) == 2:
        return matches[0].strip(), matches[1].strip()
    raise ValueError("Expected exactly two <branch> tags, but none or a different number were found.")


def divide_and_conquer(session: str, task: str, depth: int, call_generate: callable, args: argparse.Namespace) -> str:

    if depth == args.max_depth:
        prompt_text = f"{PROMPT_SOLVE} {task}"
        prompt = session + USER_HEADER + prompt_text + EOT_ID + ASSISTANT_HEADER
        response = call_generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature, stop=[ASSISTANT_SUFFIX])
        return response.strip()

    divide_prompt_text = PROMPT_DIVIDE_TEMPLATE.format(task=task)
    divide_prompt_session = session + USER_HEADER + divide_prompt_text + EOT_ID + ASSISTANT_HEADER
    response = call_generate(divide_prompt_session, max_tokens=args.max_tokens, temperature=args.temperature, stop=[ASSISTANT_SUFFIX])

    try:
        subtask1, subtask2 = parse_subtasks(response)
    except ValueError:
        subtask1, subtask2 = task, task
        return "parse error"

    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(divide_and_conquer, divide_prompt_session, subtask1, depth + 1, call_generate, args)
        future2 = executor.submit(divide_and_conquer, divide_prompt_session, subtask2, depth + 1, call_generate, args)
        solution1 = future1.result()
        solution2 = future2.result()

    merge_prompt_text = f"Subtask 1 solution: {solution1}\nSubtask 2 solution: {solution2}\n{PROMPT_MERGE}"
    merge_prompt_session = divide_prompt_session + USER_HEADER + merge_prompt_text + EOT_ID + ASSISTANT_HEADER
    merged_response = call_generate(merge_prompt_session, max_tokens=args.max_tokens, temperature=args.temperature, stop=[ASSISTANT_SUFFIX])
    return merged_response.strip()


def run_dac_instance(task: str, call_generate: callable, args: argparse.Namespace) -> float:
    start_time = time.monotonic()
    initial_session = f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"
    divide_and_conquer(initial_session, task, 0, call_generate, args)
    latency = time.monotonic() - start_time
    return latency


def generate_multiplication_prompts(num_prompts: int) -> List[str]:
    return [
        f"What is the product of {random.randint(1, 1_000_000)} and {random.randint(1, 1_000_000)}?"
        for _ in range(num_prompts)
    ]

def main(args: argparse.Namespace):

    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    prompts = generate_multiplication_prompts(args.num_requests)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(run_dac_instance, p, call_generate, args) for p in prompts]
        # just for the progress bar
        for future in tqdm(as_completed(futures), total=args.num_requests):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_8_rot_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A benchmark script for a recursive, divide-and-conquer workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Server Configuration ---
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="The backend to use for the LLM.")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    # --- Benchmark Configuration ---
    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-requests", type=int, default=32, help="Total number of divide-and-conquer instances to run.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=32, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--max-depth", type=int, default=5, help="Maximum recursion depth for the divide-and-conquer agent.")
    benchmark_group.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation.")

    args = parser.parse_args()
    main(args)

# --- ✅ Benchmark Complete ---
# Total Time Taken: 2013.42 milliseconds
# Throughput:       15.89 requests/second
# --------------------------