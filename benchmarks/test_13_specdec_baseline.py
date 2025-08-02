import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from test_utils import get_call_generate, append_log

BOS_TOKEN = "<|begin_of_text|>"
EOT_ID = "<|eot_id|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = " <|eot_id|>"


def send_request(user_prompt: str, call_generate: callable, args: argparse.Namespace):
    system_prompt = "You are a helpful, respectful and honest assistant."
    full_prompt = (
        f"{BOS_TOKEN}{SYSTEM_HEADER}{system_prompt}{EOT_ID}"
        f"{USER_HEADER}{user_prompt}{EOT_ID}{ASSISTANT_HEADER}"
    )

    start_time = time.monotonic()
    completion = call_generate(
        full_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=ASSISTANT_SUFFIX
    )
    latency = time.monotonic() - start_time

    print(f"Response: {completion}")
    return latency


def main(args: argparse.Namespace):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    prompts = []
    base_prompt = args.prompt
    for i in range(args.num_requests):
        user_prompt = f"{base_prompt} {i}."
        user_prompt = f"TASK ID: {random.randint(100000, 999999)}\n{user_prompt}"
        prompts.append(user_prompt)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(send_request, p, call_generate, args) for p in prompts]
        for future in tqdm(as_completed(futures), total=args.num_requests):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_13_specdec_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="The backend to use for the LLM (e.g., 'vllm').")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--prompt", type=str, default="Print 'hello' 100 times", help="The base prompt to send for each request.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of requests to send.")
    benchmark_group.add_argument("--max-tokens", type=int, default=64, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--temperature", type=float, default=0, help="The temperature for the generation.")

    args = parser.parse_args()
    main(args)

# vLLM with ngram speculation
# --- ✅ Benchmark Complete ---
# Total Time Taken: 1799.06 milliseconds
# Throughput:       71.15 requests/second
# --------------------------