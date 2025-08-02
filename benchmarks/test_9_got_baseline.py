import argparse
import concurrent.futures
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Callable, Optional

from tqdm import tqdm
from test_utils import get_call_generate, append_log

BOS_TOKEN = "<|begin_of_text|>"
EOT_ID = "<|eot_id|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = "<|eot_id|>"

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
PROMPT_PROPOSAL_TEMPLATE = "Could you suggest a method or approach to solve the following question? Please provide a high-level plan without doing the actual calculation. Keep it concise, around 80 words. Question: {question}"
PROMPT_AGGREGATE = "Please compare the following solution with the one you just provided and aggregate their ideas into a single, improved solution:\n"

ProposalResult = Tuple[str, str]
AggregationResult = Tuple[str, str]


def generate_proposal(base_prompt: str, proposal_prompt: str, max_tokens: int, call_generate: Callable, args: argparse.Namespace) -> ProposalResult:
    full_prompt = f"{base_prompt}{USER_HEADER}{proposal_prompt}{EOT_ID}{ASSISTANT_HEADER}"
    response = call_generate(full_prompt, max_tokens=max_tokens, temperature=args.temperature, stop=[EOT_ID], n=1)[0]
    return response.strip(), full_prompt + response


def generate_aggregation(agg_context: str, prev_text: str, max_tokens: int, call_generate: Callable, args: argparse.Namespace) -> AggregationResult:
    full_prompt = f"{agg_context}{ASSISTANT_SUFFIX}{USER_HEADER}{PROMPT_AGGREGATE}{prev_text}{EOT_ID}{ASSISTANT_HEADER}"
    response = call_generate(full_prompt, max_tokens=max_tokens, temperature=args.temperature, stop=[EOT_ID], n=1)[0]
    return response.strip(), full_prompt + response


def run_hierarchical_instance(question: str, call_generate: Callable, args: argparse.Namespace) -> float:
    start_time = time.monotonic()
    base_prompt = f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"
    proposal_prompt = PROMPT_PROPOSAL_TEMPLATE.format(question=question)
    proposal_max_tokens = [4, 32, 16, 8, 4, 16, 3, 32]  # From original example

    with ThreadPoolExecutor(max_workers=args.internal_concurrency) as executor:
        proposal_futures = [
            executor.submit(generate_proposal, base_prompt, proposal_prompt, mt, call_generate, args)
            for mt in proposal_max_tokens
        ]

        first_agg_futures = []
        pending_proposal: Optional[ProposalResult] = None
        for future in concurrent.futures.as_completed(proposal_futures):
            current_proposal = future.result()
            if pending_proposal is None:
                pending_proposal = current_proposal
            else:
                prev_text, _ = pending_proposal
                _, current_context = current_proposal
                agg_future = executor.submit(generate_aggregation, current_context, prev_text, args.max_tokens, call_generate, args)
                first_agg_futures.append(agg_future)
                pending_proposal = None  # Reset

        second_agg_futures = []
        pending_aggregation: Optional[AggregationResult] = None
        for future in concurrent.futures.as_completed(first_agg_futures):
            current_agg = future.result()
            if pending_aggregation is None:
                pending_aggregation = current_agg
            else:
                prev_text, _ = pending_aggregation
                _, current_context = current_agg
                final_agg_future = executor.submit(generate_aggregation, current_context, prev_text, args.max_tokens, call_generate, args)
                second_agg_futures.append(final_agg_future)
                pending_aggregation = None  # Reset

        for future in concurrent.futures.as_completed(second_agg_futures):
            _ = future.result()  # Ensure all final tasks are complete

    return time.monotonic() - start_time


def generate_example_prompts(num_prompts: int) -> List[str]:
    return [
        f"Explain the concept of recursion in programming with a simple analogy. Add random numbers for variety: {random.randint(1, 1_000_000)}."
        for _ in range(num_prompts)
    ]


def main(args: argparse.Namespace):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    prompts = generate_example_prompts(args.num_requests)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()  # This executor manages the top-level parallelism of the benchmark (running multiple instances)
    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(run_hierarchical_instance, p, call_generate, args) for p in prompts]
        # just for the progress bar
        for future in tqdm(as_completed(futures), total=args.num_requests):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_9_got_baseline.json', {
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
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of hierarchical instances to run.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--internal-concurrency", type=int, default=8, help="Max workers for the thread pool inside each instance (for proposals, etc.).")
    benchmark_group.add_argument("--max-tokens", type=int, default=64, help="Maximum number of tokens for aggregation steps.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation.")

    args = parser.parse_args()
    main(args)

#
# --- ✅ Benchmark Complete ---
# Total Time Taken: 13832.41 milliseconds
# Throughput:       9.25 requests/second
# --------------------------