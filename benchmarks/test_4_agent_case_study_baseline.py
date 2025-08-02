#!/usr/bin/env python3
import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from test_utils import get_call_generate, append_log  # Assuming test_utils.py is in the same directory

TEMPLATE_INTRO = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant that can use tools to answer questions. You have access to the following tools:
"""

WEB_SEARCH_DOCS = """
- `WebSearch[query]`: This tool allows you to perform powerful, semantic searches across a vast corpus of indexed web pages, academic papers, and news articles. It is ideal for finding specific facts, dates, names, definitions, and general knowledge. The query should be a concise question or a set of keywords. For example, to find the capital of France, you would use `WebSearch[capital of France]`. The tool returns a snippet of the most relevant document. It is particularly effective for questions about recent events or information that might change over time. The search index is updated continuously, ensuring access to the most current information available. It can handle complex, natural language queries and is optimized for fact-finding missions. Avoid overly broad queries; be as specific as possible to get the best results. The underlying engine uses a combination of keyword matching and vector-based similarity search to identify the most relevant information, making it robust against variations in phrasing and terminology. This is your primary tool for gathering new information that is not already in your knowledge base. It is not suitable for mathematical calculations or for performing actions in the real world. It only retrieves information.
"""

CODE_INTERPRETER_DOCS = """
- `CodeInterpreter[code]`: This tool executes Python code in a sandboxed environment. It is perfect for complex calculations, data analysis, simulations, and any task that can be expressed as a program. The input `code` must be a valid Python script. The final line of the script should be an expression or a `print()` statement that produces the output. For example, `CodeInterpreter[print(sum([i*i for i in range(100)]))]`. You can use popular libraries like `pandas`, `numpy`, and `matplotlib`. The environment is stateful within a single turn, but the state is reset for each new call. This tool is essential for tasks requiring precise numerical computation, data manipulation, or algorithmic logic. Do not use it for simple arithmetic that can be done manually. The sandbox has no internet access; any required data must be passed directly into the `code` parameter. It is a powerful tool for any problem that requires logic, iteration, or complex mathematical formulas. Ensure the code is self-contained and produces a clear output, as this output will be the only thing returned to you as an observation. This is your go-to for quantitative reasoning.
"""

TEMPLATE_OUTRO = """
To answer the user's question, you must break it down into a series of steps. For each step, you must first think about what to do, then output the action to take. The format should be:

Thought: Your reasoning for the next action.
Action: The tool to use, in the format `ToolName[input]`.

After you perform an action, you will receive an observation with the result. You will repeat this process until you have the final answer.
<|eot_id|>
"""

PREDEFINED_SEQUENCES = [
    # 3x WebSearch calls
    (
        "\nThought: The user is asking a factual question about geography. I will use WebSearch to find the highest mountain in North America.\nAction: WebSearch[highest mountain in North America]",
        "\nObservation: Denali (formerly known as Mount McKinley) is the highest mountain peak in North America, with a summit elevation of 20,310 feet (6,190 m) above sea level.",
    ),
    (
        "\nThought: The user is asking a question about classic literature. I will use WebSearch to find the author of 'Pride and Prejudice'.\nAction: WebSearch[author of Pride and Prejudice]",
        "\nObservation: 'Pride and Prejudice' is a romantic novel of manners written by Jane Austen, published in 1813.",
    ),
    (
        "\nThought: The user wants to know a specific chemical formula. WebSearch is the appropriate tool for this.\nAction: WebSearch[chemical formula for caffeine]",
        "\nObservation: The chemical formula for caffeine is C8H10N4O2.",
    ),
    # 3x CodeInterpreter calls
    (
        "\nThought: Now I need to perform a calculation. I will calculate the sum of squares from 1 to 50.\nAction: CodeInterpreter[print(sum([i**2 for i in range(1, 51)]))]",
        "\nObservation: 42925",
    ),
    (
        "\nThought: I will perform another calculation. I'll find the 20th Fibonacci number.\nAction: CodeInterpreter[a, b = 0, 1\nfor _ in range(19):\n  a, b = b, a + b\nprint(a)]",
        "\nObservation: 6765",
    ),
    (
        "\nThought: One final calculation. I'll approximate pi using the Nilakantha series with 1000 terms.\nAction: CodeInterpreter[pi = 3.0\nsign = 1\nfor i in range(2, 2001, 2):\n  pi += sign * 4 / (i * (i + 1) * (i + 2))\n  sign *= -1\nprint(pi)]",
        "\nObservation: 3.1415921535897914",
    ),
]

TOOL_HEADER = "<|eot_id|><|start_header_id|>tool<|end_header_id|>"
ASSISTANT_HEADER = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
ASSISTANT_SUFFIX = "<|eot_id|>"


def agent_main(task_id, call_generate, args):
    system_prompt = (
        f"{TEMPLATE_INTRO}"
        f"{WEB_SEARCH_DOCS}"
        f"{CODE_INTERPRETER_DOCS}"
        f"{TEMPLATE_OUTRO}"
    )
    user_prompt = f"TASK ID: {task_id}\nPerform a series of research and calculation tasks."

    s = system_prompt
    s += f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}"
    s += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    for i in range(len(PREDEFINED_SEQUENCES)):
        if args.network_delay > 0:
            time.sleep(args.network_delay)

        call_generate(
            s,
            max_tokens=args.tokens_between_calls,
            temperature=0.0,
            stop=[ASSISTANT_SUFFIX]
        )

        s += PREDEFINED_SEQUENCES[i][0]

        time.sleep(args.function_call_delay)

        s += TOOL_HEADER
        s += PREDEFINED_SEQUENCES[i][1]
        s += ASSISTANT_HEADER


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    arguments = [
        {"task_id": random.randint(100000, 999999)}
        for _ in range(args.num_requests)
    ]

    def launch_agent(i):
        agent_main(**arguments[i], call_generate=call_generate, args=args)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        list(
            tqdm(
                executor.map(launch_agent, range(args.num_requests)),
                total=args.num_requests,
                desc="Processing Requests"
            )
        )

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_4_agent_case_study_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Server Configuration ---
    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="The backend to use for the LLM (e.g., 'vllm').")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of requests (conversations) to simulate.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--tokens-between-calls", type=int, default=32, help="Max tokens for the LLM to generate for each Thought/Action step.")
    benchmark_group.add_argument("--network-delay", type=float, default=0.0, help="Simulated network delay in seconds before each LLM call.")
    benchmark_group.add_argument("--function-call-delay", type=float, default=0.0, help="Simulated delay in seconds for local tool/function execution.")

    args = parser.parse_args()
    main(args)

#
# --- ✅ Benchmark Complete ---
# Total Time Taken: 24484.72 milliseconds
# Throughput:       5.23 requests/second
# --------------------------