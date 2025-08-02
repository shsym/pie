import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm
from test_utils import get_call_generate, append_log  # Assumed to exist

SYSTEM_PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful, and honest assistant that excels at mathematical reasoning. Please follow the user's instructions precisely.<|eot_id|>"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT_ID = "<|eot_id|>"

PROPOSE_PROMPT_TEMPLATE = "Please generate a high-level plan for solving the following question. First, just state the method you will use. Do not do the actual calculation. Keep your response concise and within 80 words. Question: {question}"
PROMPT_EXECUTE = "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the plan. Give me the final answer. Make your response short."
PROMPT_REFLECT = "Okay. Now, evaluate your own solution and give it a score on a scale of 1 to 5. Please rigorously check the correctness of the calculations and the final answer."


def run_tree_search_instance(question: str, call_generate: callable, args: argparse.Namespace):
    start_time = time.monotonic()
    final_solutions = []

    with ThreadPoolExecutor(max_workers=args.internal_concurrency) as executor:
        propose_prompt_base = (
                SYSTEM_PROMPT
                + USER_HEADER
                + PROPOSE_PROMPT_TEMPLATE.format(question=question)
                + EOT_ID
                + ASSISTANT_HEADER
        )

        plan_futures = [executor.submit(call_generate, propose_prompt_base, max_tokens=args.max_tokens, temperature=args.temperature, n=1, stop=[EOT_ID]) for _ in range(args.num_branches)]

        plan_forks = []
        for future in as_completed(plan_futures):
            completion = future.result()[0]
            plan_forks.append(propose_prompt_base + completion + EOT_ID)

        execute_futures = []
        for plan_context in plan_forks:
            execute_prompt_base = (
                    plan_context
                    + USER_HEADER
                    + PROMPT_EXECUTE
                    + EOT_ID
                    + ASSISTANT_HEADER
            )
            for _ in range(args.num_branches):
                future = executor.submit(call_generate, execute_prompt_base, max_tokens=args.max_tokens, temperature=args.temperature, n=1, stop=[EOT_ID])
                execute_futures.append((future, execute_prompt_base))

        execute_forks = []
        for future, base_prompt in execute_futures:
            completion = future.result()[0]
            execute_forks.append(base_prompt + completion + EOT_ID)

        reflect_futures = []
        for execute_context in execute_forks:
            reflect_prompt_base = (
                    execute_context
                    + USER_HEADER
                    + PROMPT_REFLECT
                    + EOT_ID
                    + ASSISTANT_HEADER
            )
            for _ in range(args.num_branches):
                future = executor.submit(call_generate, reflect_prompt_base, max_tokens=args.max_tokens, temperature=args.temperature, n=1, stop=[EOT_ID])
                reflect_futures.append((future, reflect_prompt_base))

        for future, base_prompt in reflect_futures:
            completion = future.result()[0]
            final_solutions.append(base_prompt + completion + EOT_ID)

    latency = time.monotonic() - start_time
    return latency


def generate_math_prompts(num_prompts: int) -> List[str]:
    prompts = []
    for _ in range(num_prompts):
        num1 = random.randint(1_000_000, 1_000_000_000)
        num2 = random.randint(1_000_000, 1_000_000_000)
        q = f"What is the sum of {num1} and {num2}?"
        prompts.append(q)
    return prompts


def main(args: argparse.Namespace):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    prompts = generate_math_prompts(args.num_requests)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    # This outer executor runs multiple ToT instances in parallel
    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(run_tree_search_instance, p, call_generate, args) for p in prompts]
        for future in tqdm(futures, total=len(prompts)):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_7_tot_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="The backend to use for the LLM.")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("-n", "--num-requests", type=int, default=64, help="Total number of ToT instances to run.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=64, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--internal-concurrency", type=int, default=8, help="Inner concurrency: Number of parallel branches within a single instance.")
    benchmark_group.add_argument("-b", "--num-branches", type=int, default=2, help="Number of branches to explore at each level.")
    benchmark_group.add_argument("-t", "--max-tokens", type=int, default=64, help="Max new tokens to generate at each step.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation. Use a low value for determinism.")

    args = parser.parse_args()
    main(args)

# --- ✅ Benchmark Complete ---
# Total Time Taken: 10327.14 milliseconds
# Throughput:       6.20 requests/second
# --------------------------