import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from test_utils import get_call_generate, append_log

USER_PREFIX = "<|start_header_id|>user<|end_header_id|> "
USER_SUFFIX = " <|eot_id|>"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = " <|eot_id|>"
SYSTEM_PREFIX = "<|start_header_id|>system<|end_header_id|>\n\n"
SYSTEM_SUFFIX = "<|eot_id|>"

REACT_PROMPT_TEMPLATE = """
You are a helpful assistant that can use tools to answer questions. You have access to the following tools:

- `Search[query]`: Searches for information online.
- `Calculator[expression]`: Computes a mathematical expression.

To answer the user's question, you must break it down into a series of steps. For each step, you must first think about what to do, then output the action to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

After you perform an action, you will receive an observation with the result. You will repeat this process until you have the final answer.

Question: Who is the director of the movie that won the Best Picture Oscar in the year the James Webb Space Telescope was launched?
"""


def agent_main(task, call_generate, args):
    s = "<|begin_of_text|>"
    s += f"{SYSTEM_PREFIX}{REACT_PROMPT_TEMPLATE}{SYSTEM_SUFFIX}"
    s += f"{USER_PREFIX}{task}{USER_SUFFIX}"
    s += ASSISTANT_PREFIX

    for i in range(args.num_function_calls):

        completion = call_generate(
            s,
            max_tokens=args.tokens_between_calls,
            temperature=0.3,
            stop=ASSISTANT_SUFFIX
        )
        s += completion

        # Simulate network delay for sending/receiving the request
        # This is useful for simulating latency without having to deploy a remote LLM server
        # Should be set to zero for non-local testing
        if args.network_delay > 0:
            time.sleep(args.network_delay)

        if args.function_call_delay > 0:
            time.sleep(args.function_call_delay)

        s += f"\nObservation: Result from function call {i + 1} is available."

    return


def main(args):
    arguments = [
        {"task": f"TASK ID: {random.randint(100000, 999999)}\n{REACT_PROMPT_TEMPLATE}"}
        for _ in range(args.num_requests)
    ]

    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    states = [None] * args.num_requests

    def launch_agent(i):
        states[i] = agent_main(**arguments[i], call_generate=call_generate, args=args)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        list(
            tqdm(
                executor.map(launch_agent, range(args.num_requests)),
                total=args.num_requests,
            )
        )

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_1_agent_react_baseline.json', {
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
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of requests (conversations) to simulate.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--num-function-calls", type=int, default=8, help="Number of sequential function calls (Thought/Action/Observation cycles) per request.")
    benchmark_group.add_argument("--tokens-between-calls", type=int, default=16, help="Max tokens for the LLM to generate for each Thought/Action step.")
    benchmark_group.add_argument("--network-delay", type=float, default=0.0, help="Simulated network delay in seconds before each LLM call.")
    benchmark_group.add_argument("--function-call-delay", type=float, default=0.1, help="Simulated delay in seconds for local tool/function execution.")

    args = parser.parse_args()
    main(args)


