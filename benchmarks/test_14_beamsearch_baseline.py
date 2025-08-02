import argparse
import time
import openai
import random
import textwrap
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from tqdm import tqdm

from test_utils import append_log


def send_request(user_prompt: str, client: openai.Client, args: argparse.Namespace) -> float:
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    start_time = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=args.model_path,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=0.0,
            extra_body={'use_beam_search': True, "best_of": args.beam_size}
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Request failed: {e}")
        return -1.0  # Indicate failure

    latency = time.monotonic() - start_time
    return latency


def main(args: argparse.Namespace):
    client = openai.Client(base_url=f"{args.host}:{args.port}/v1", api_key="None")

    prompts = [f"{args.prompt} #{i}" for i in range(args.num_requests)]
    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = {executor.submit(send_request, p, client, args): p for p in prompts}

        for future in tqdm(as_completed(futures), total=args.num_requests):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_14_beamsearch_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="The backend to use")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--prompt", type=str, default="Explain the meaning of this number", help="The base prompt to send for each request.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=64, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--num-requests", type=int, default=100, help="Total number of requests to send.")
    benchmark_group.add_argument("--max-tokens", type=int, default=32, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--beam-size", type=int, default=10, help="The beam size")

    args = parser.parse_args()
    main(args)
