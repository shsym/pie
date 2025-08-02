import argparse
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable

from tqdm import tqdm

from test_utils import get_call_generate, append_log

SYSTEM_HEADER = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
EOT_ID = "<|eot_id|>"

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
PROMPT_PLAN_TEMPLATE = "Generate up to {max_points} key points that outline the answer to the following question: {question}. Each point should be a concise statement of a main idea, enclosed in <point> tags. Do not elaborate on the points yet."
PROMPT_ELABORATE_TEMPLATE = "Elaborate on the following point: {point}. Your response should be complete and only concerned with this point. Keep it concise."
POINT_PLACEHOLDER = PROMPT_ELABORATE_TEMPLATE

def run_plan_and_elaborate_instance(
        question: str,
        max_points: int,
        call_generate: Callable,
        args: argparse.Namespace
) -> float:
    start_time = time.monotonic()
    base_prompt = f"{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"

    plan_prompt_text = PROMPT_PLAN_TEMPLATE.format(max_points=max_points, question=question)
    full_plan_prompt = f"{base_prompt}{USER_HEADER}{plan_prompt_text}{EOT_ID}{ASSISTANT_HEADER}"
    plan_responses = call_generate(
        full_plan_prompt,
        max_tokens=args.max_tokens_plan,
        temperature=args.temperature,
        stop=[EOT_ID],
        n=1
    )
    points = re.findall(r"<point>(.*?)</point>", plan_responses[0], re.DOTALL)
    points = [p.strip() for p in points if p.strip()]

    if len(points) == 0:
        points = [POINT_PLACEHOLDER, POINT_PLACEHOLDER, POINT_PLACEHOLDER]

    if not points:
        return time.monotonic() - start_time

    with ThreadPoolExecutor() as sub_executor:
        elaboration_futures = []
        for point in points:
            elab_prompt_text = PROMPT_ELABORATE_TEMPLATE.format(point=point)
            full_elab_prompt = f"{base_prompt}{USER_HEADER}{elab_prompt_text}{EOT_ID}{ASSISTANT_HEADER}"
            future = sub_executor.submit(
                call_generate,
                full_elab_prompt,
                max_tokens=args.max_tokens_elab,
                temperature=args.temperature,
                stop=[EOT_ID],
                n=1
            )
            elaboration_futures.append(future)

        for future in as_completed(elaboration_futures):
            future.result()

    return time.monotonic() - start_time


def generate_questions(num_requests: int) -> List[dict]:
    question_pool = [
        "What are the defining characteristics of ancient Rome?",
        "Summarize the plot of 'Hamlet' in key stages.",
        "Explain the main benefits of using renewable energy sources.",
        "What were the primary causes of World War 1?",
        "Describe the process of photosynthesis.",
    ]
    return [{
        "question": random.choice(question_pool),
        "max_points": random.randint(3, 5)
    } for _ in range(num_requests)]


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    tasks = generate_questions(args.num_requests)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(run_plan_and_elaborate_instance, t["question"], t["max_points"], call_generate, args) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(tasks)):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_10_skot_baseline.json', {
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
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of plan-and-elaborate instances to run.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation.")
    benchmark_group.add_argument("--max-tokens-plan", type=int, default=80, help="Max tokens for the initial planning stage.")
    benchmark_group.add_argument("--max-tokens-elab", type=int, default=80, help="Max tokens for each elaboration stage.")

    args = parser.parse_args()
    main(args)

# --- ✅ Benchmark Complete ---
# Total Time Taken: 3997.12 milliseconds
# Throughput:       32.02 requests/second
# --------------------------