#!/usr/bin/env python3
import argparse
import random
import time
from typing import List

import sglang as sgl
from sglang import RuntimeEndpoint

# --- Prompt Templates ---
PROMPT_PLAN = "Please generate a high-level plan for solving the following question. As the first step, just say what method and idea you will use to solve the question. You can reorganize the information in the question. Do not do the actual calculation. Keep your response concise and within 80 words. Question: {question}"
PROMPT_EXECUTE = "The plan looks good! Now, use real numbers and do the calculation. Please solve the question step-by-step according to the high-level plan. Give me the final answer. Make your response short."
PROMPT_REFLECT = "Okay. Now you evaluate your own solution and give it a score on a scale of 1 to 5. Please do rigorous check of the correctness."


# --- Tree Search Prompting Functions ---

def propose_plan(s, question: str, n: int, temperature: float, max_tokens: int):
    """Generates `n` initial plans to solve the question."""
    s += sgl.user(PROMPT_PLAN.format(question=question))
    forks = s.fork(n)
    forks += sgl.assistant(sgl.gen("plan", max_tokens=max_tokens, temperature=temperature))
    return forks


def execute_plan(s, n: int, temperature: float, max_tokens: int):
    """Executes the plan, performing the actual calculations."""
    s += sgl.user(PROMPT_EXECUTE)
    forks = s.fork(n)
    forks += sgl.assistant(sgl.gen("answer", max_tokens=max_tokens, temperature=temperature))
    return forks


def reflect_solution(s, n: int, temperature: float, max_tokens: int):
    """Evaluates the generated solution and provides a score."""
    s += sgl.user(PROMPT_REFLECT)
    forks = s.fork(n)
    forks += sgl.assistant(sgl.gen("score", max_tokens=max_tokens, temperature=temperature))
    return forks


# --- Main SGLang Benchmark Function ---

@sgl.function
def tree_search(s, question, num_branches, temperature, max_tokens_plan, max_tokens_execute, max_tokens_reflect):
    """
    Defines the multi-step Tree-of-Thought generation process.
    1. Propose `n` plans.
    2. For each plan, execute it `n` times.
    3. For each execution, reflect on it `n` times.
    """
    # 1. Propose `n` plans from the initial state
    plan_forks = propose_plan(s, question, num_branches, temperature, max_tokens_plan)

    # 2. For each plan, fork to create `n` solution paths
    solution_states = []
    for plan_state in plan_forks:
        sol_forks = execute_plan(plan_state, num_branches, temperature, max_tokens_execute)
        solution_states.extend(sol_forks)

    # 3. For each solution, fork to create `n` reflections
    leaf_states = []
    for sol_state in solution_states:
        reflect_forks = reflect_solution(sol_state, num_branches, temperature, max_tokens_reflect)
        leaf_states.extend(reflect_forks)

    # Joining the top-level fork waits for all descendant generations to complete.
    plan_forks.join()

    # Return the final leaf states for potential inspection.
    return leaf_states


def generate_math_prompts(num_prompts: int) -> List[str]:
    """Generates a list of simple math questions."""
    return [
        f"What is the sum of {random.randint(1, 1_000_000)} and {random.randint(1, 1_000_000)}?"
        for _ in range(num_prompts)
    ]


def main(args: argparse.Namespace):
    """
    Main function to set up and run the Tree-of-Thought sglang benchmark.
    """
    # --- 1. Initialize Backend ---
    print("--- 1. Initializing SGLang Backend ---")
    backend = RuntimeEndpoint(f"{args.host}:{args.port}")
    print(f"✅ Successfully initialized SGLang client for endpoint: {args.host}:{args.port}")

    # --- 2. Generate Prompts ---
    print(f"\n--- 2. Generating Prompts ---")
    questions = generate_math_prompts(args.num_requests)
    print(f"Generated {len(questions)} unique math questions for the benchmark.")

    # Prepare arguments for run_batch. Each item corresponds to one full tree-search instance.
    batch_args = [
        {
            "question": q,
            "num_branches": args.num_branches,
            "temperature": args.temperature,
            "max_tokens_plan": args.max_tokens_plan,
            "max_tokens_execute": args.max_tokens_execute,
            "max_tokens_reflect": args.max_tokens_reflect,
        }
        for q in questions
    ]

    # --- 3. Run Benchmark ---
    print(f"\n--- 3. Running Benchmark ---")
    print(f"🚀 Starting benchmark with {args.num_requests} total requests (tree-search instances)...")
    start_time = time.monotonic()

    states = tree_search.run_batch(
        batch_args,
        backend=backend,
        progress_bar=True,
    )

    total_time = time.monotonic() - start_time

    # --- 4. Print Results ---
    print("\n--- ✅ Benchmark Complete ---")

    # Throughput is measured in "instances" per second, where each instance is a full tree search.
    successful_requests = len(states)
    throughput = successful_requests / total_time if total_time > 0 else 0
    avg_latency_per_instance = (total_time / successful_requests) * 1000 if successful_requests > 0 else 0

    print(f"\nTotal Time Taken:           {total_time:.2f} seconds")
    print(f"Total Successful Instances: {successful_requests}")
    print("-----------------------------------------")
    print(f"Throughput:                 {throughput:.2f} instances/second")
    print(f"Average Latency per Instance: {avg_latency_per_instance:.2f} ms")
    print("-----------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A benchmark script for multi-step, Tree-of-Thought style generation with SGLang.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Server Configuration ---
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the SGLang server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the SGLang server.")

    # --- Benchmark Configuration ---
    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-requests", type=int, default=64, help="Total number of tree-search instances to run.")
    benchmark_group.add_argument("--num-branches", type=int, default=2, help="Number of branches to explore at each stage (plan, execute, reflect).")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation. Use a low value for more deterministic results.")

    # --- Generation Arguments ---
    gen_group = parser.add_argument_group('Generation Arguments')
    gen_group.add_argument("--max-tokens-plan", type=int, default=64, help="Max new tokens for the planning step.")
    gen_group.add_argument("--max-tokens-execute", type=int, default=64, help="Max new tokens for the execution step.")
    gen_group.add_argument("--max-tokens-reflect", type=int, default=64, help="Max new tokens for the reflection step.")

    args = parser.parse_args()
    main(args)
