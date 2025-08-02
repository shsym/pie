#!/usr/bin/env python3
import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from test_utils import get_call_generate, append_log
from py_mini_racer import py_mini_racer

USER_PREFIX = "<|start_header_id|>user<|end_header_id|> "
USER_SUFFIX = " <|eot_id|>"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = " <|eot_id|>"
SYSTEM_PREFIX = "<|start_header_id|>system<|end_header_id|>\n\n"
SYSTEM_SUFFIX = "<|eot_id|>"


HARDCODED_RESPONSES = [
    """Thought: I need to find the first 10 prime numbers and sum them. First, I need a way to determine if a number is prime. I will write a helper function `isPrime` and define it for the subsequent steps.
```javascript
function isPrime(num) {
  if (num <= 1) return false;
  if (num <= 3) return true;
  if (num % 2 === 0 || num % 3 === 0) return false;
  for (let i = 5; i * i <= num; i = i + 6) {
    if (num % i === 0 || num % (i + 2) === 0) return false;
  }
  return true;
}
// The function is now defined for this execution context.
// Return a simple value to confirm completion.
"isPrime function created";
```""",
    """Thought: The `isPrime` function was defined successfully. Now I will use it to find the first 10 prime numbers and calculate their sum. I must include the `isPrime` definition again in this code block since the execution environment is stateless between turns.
```javascript
function isPrime(num) {
  if (num <= 1) return false;
  if (num <= 3) return true;
  if (num % 2 === 0 || num % 3 === 0) return false;
  for (let i = 5; i * i <= num; i = i + 6) {
    if (num % i === 0 || num % (i + 2) === 0) return false;
  }
  return true;
}

let primes = [];
let num = 2;
while (primes.length < 10) {
  if (isPrime(num)) {
    primes.push(num);
  }
  num++;
}
// Calculate and return the final sum.
primes.reduce((a, b) => a + b, 0);
```""",
    """Thought: I have successfully calculated the sum of the first 10 prime numbers. The result from the code execution was 129. I will now state the final answer clearly.

The final answer is that the sum of the first 10 prime numbers is 129."""
]

CODEACT_PROMPT_TEMPLATE = """
You are CodeACT, a highly intelligent AI assistant that can understand and execute JavaScript code to solve problems.
You will be given a task. To solve it, you must think step-by-step and produce JavaScript code in ```javascript ... ``` blocks.
You will receive the output of the code execution and repeat the process until you have the final answer.
"""


def extract_js_code(text: str) -> str:
    start_marker = "```javascript"
    end_marker = "```"
    start = text.find(start_marker)
    if start != -1:
        start += len(start_marker)
        end = text.find(end_marker, start)
        if end != -1:
            return text[start:end].strip()
    return None


def execute_js_code(code: str) -> str:
    ctx = py_mini_racer.MiniRacer()
    try:
        result = ctx.eval(code)
        return f"Output: {str(result)}"
    except Exception as e:
        return f"Execution Error: {e}"


def agent_main(task: str, call_generate, args):
    s = "<|begin_of_text|>"
    s += f"{SYSTEM_PREFIX}{CODEACT_PROMPT_TEMPLATE}{SYSTEM_SUFFIX}"
    s += f"{USER_PREFIX}{task}{USER_SUFFIX}"
    s += ASSISTANT_PREFIX

    num_code_turns = min(args.num_function_calls, len(HARDCODED_RESPONSES) - 1)

    for i in range(num_code_turns):
        if args.network_delay > 0:
            time.sleep(args.network_delay)


        _ = call_generate(
            s,
            max_tokens=args.tokens_between_calls,
            temperature=0.3,
            stop=ASSISTANT_SUFFIX
        )

        # Use the hardcoded response for this turn to control the logic.
        assistant_response = HARDCODED_RESPONSES[i]
        s += assistant_response
        s += ASSISTANT_SUFFIX

        js_code = extract_js_code(assistant_response)
        if js_code:
            result = execute_js_code(js_code)
            s += f"{SYSTEM_PREFIX}Code execution result:\n{result}{SYSTEM_SUFFIX}"
        else:
            s += f"{SYSTEM_PREFIX}No code was executed in this turn.{SYSTEM_SUFFIX}"

        s += ASSISTANT_PREFIX

    _ = call_generate(s, max_tokens=args.tokens_between_calls, temperature=0.0, stop=ASSISTANT_SUFFIX)
    s += HARDCODED_RESPONSES[num_code_turns]

    return s


def launch_agent_wrapper(params):
    try:
        return agent_main(**params)
    except Exception as e:
        return f"Error: {e}"


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    base_task = "Calculate the sum of the first 10 prime numbers."
    arguments = []
    for i in range(args.num_requests):
        task = f"Request {i}:  The primary goal is to {base_task}"
        arguments.append({"task": task, "args": args, "call_generate": call_generate})

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.num_max_workers) as executor:
        list(tqdm(executor.map(launch_agent_wrapper, arguments), total=args.num_requests, desc="Processing Requests"))

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_2_agent_codeact_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
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
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of concurrent conversations to simulate.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--prompt-length", type=int, default=20, help="Number of random words to add to the prompt.")
    benchmark_group.add_argument("--num-function-calls", type=int, default=2, help=f"Number of code execution cycles per request. Max: {len(HARDCODED_RESPONSES) - 1}.")
    benchmark_group.add_argument("--tokens-between-calls", type=int, default=32, help="Max tokens to ask the LLM to generate (to simulate realistic workload). The output is discarded.")
    benchmark_group.add_argument("--network-delay", type=float, default=0.0, help="Simulated non-model network delay in seconds to add before each turn.")

    args = parser.parse_args()

    max_calls = len(HARDCODED_RESPONSES) - 1
    if args.num_function_calls > max_calls:
        print(f"Warning: --num-function-calls is set to {args.num_function_calls}, but there are only {max_calls} code execution steps available. Clamping to max.")
        args.num_function_calls = max_calls

    main(args)
#
# --- ✅ Benchmark Complete ---
# Total Time Taken: 6353.81 milliseconds
# Throughput:       20.15 requests/second
# --------------------------