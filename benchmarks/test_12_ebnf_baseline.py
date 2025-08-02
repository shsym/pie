#!/usr/bin/env python3
import argparse
import time
import openai
import random
import textwrap
import string  # <-- Import string module
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

from tqdm import tqdm

from test_utils import append_log

ebnf_grammar = """
root ::= city | description
city ::= "London" | "Paris" | "Berlin" | "Rome"
description ::= city " is " status
status ::= "the capital of " country
country ::= "England" | "France" | "Germany" | "Italy"
"""


def send_request(user_prompt: str, client: openai.Client, args: argparse.Namespace) -> float:
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    random_suffix = ""
    if args.disable_grammar_cache:
        random_letters = "".join(random.choices(string.ascii_lowercase, k=8))
        random_suffix = f"_{random_letters}"

    if args.backend == "vllm":
        grammar = f"""
?start: value
?value: basic_object
        | basic_array
        | basic_string
        | SIGNED_NUMBER      -> number
        | "true"             -> true
        | "false"            -> false
        | "null"             -> null
basic_array  : "[" [value ("," value)*] "]"
basic_object : "{" [pair (", " pair)*] "}"
pair   : basic_string ":" value
basic_string : ESCAPED_STRING
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
        """
        grammar = r"""
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
        """
        final_grammar = grammar.replace("basic_", "basic_" + random_suffix)

        extra_body = {
            "guided_grammar": final_grammar,
        }
    elif args.backend == "sglang":

        grammar = r"""
root ::= basic_array | basic_object
basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object
basic_integer ::= ("0" | "-"? [1-9] [0-9]*) ".0"?
basic_number ::= ("0" | "-"? [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
basic_string ::= (([\"] basic_string_1 [\"]))
basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1
escape ::= ["\\/bfnrt] | "u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]
basic_boolean ::= "true" | "false"
basic_null ::= "null"
basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]"
basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}"
ws ::= [ \n\t]*
        """
        final_grammar = grammar.replace("basic_", "basic_" + random_suffix)
        extra_body = {
            "ebnf": final_grammar,
        }

    start_time = time.monotonic()
    try:
        response = client.chat.completions.create(
            model=args.model_path,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Request failed: {e}")
        return -1.0

    latency = time.monotonic() - start_time
    return latency


def main(args: argparse.Namespace):
    client = openai.Client(base_url=f"{args.host}:{args.port}/v1", api_key="None")

    prompts = [f"{args.prompt} #{i}" for i in range(args.num_requests)]

    # --- 3. Run Benchmark ---
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

    append_log('./logs/test_12_ebnf_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="sglang", help="The backend to use")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="The path or name of the model to use.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--prompt", type=str, default="Generate a long random json data", help="The base prompt to send for each request.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of requests to send.")
    benchmark_group.add_argument("--max-tokens", type=int, default=32, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--temperature", type=float, default=0, help="The temperature for the generation.")
    benchmark_group.add_argument("--disable-grammar-cache", action="store_true", help="If set, disables grammar caching by making each grammar string unique.")

    args = parser.parse_args()
    main(args)


# vLLM 0.6.6
# --- ✅ Benchmark Complete ---
# Total Time Taken: 6994.02 milliseconds
# Throughput:       18.30 requests/second
# --------------------------

# vLLM 0.6.6 with grammar cache disabled
# --- ✅ Benchmark Complete ---
# Total Time Taken: 11922.65 milliseconds
# Throughput:       10.74 requests/second
# --------------------------

# SGLang
# --- ✅ Benchmark Complete ---
# Total Time Taken: 4229.01 milliseconds
# Throughput:       30.27 requests/second
# --------------------------
#
# SGLang with grammar cache disabled
# --- ✅ Benchmark Complete ---
# Total Time Taken: 6414.90 milliseconds
# Throughput:       19.95 requests/second
# --------------------------