import argparse
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm
from test_utils import get_call_generate, append_log  # Assuming test_utils.py is in the same directory

BOS_TOKEN = "<|begin_of_text|>"
EOT_ID = "<|eot_id|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
ASSISTANT_SUFFIX = " <|eot_id|>"

# Level 0: System Prompt
SYSTEM_PROMPT = "You are a helpful, friendly, and knowledgeable science tutor for students of all ages. Your goal is to explain complex biological concepts in a clear, accessible, and engaging manner, tailoring your language to the specified audience."
PROMPT_L1_PHOTO = "I'm curious about the fundamental process of photosynthesis. Could you provide a detailed overview of how plants create their own food using sunlight, water, and carbon dioxide?"
PROMPT_L1_RESP = "Now, could you explain the equally important process of cellular respiration? I'd like to understand how organisms, including plants and animals, break down glucose to release the energy needed for life."
PROMPT_L2_PHOTO_ELI5 = "That sounds complicated. Could you simplify it significantly for me? Please explain the core idea in a way that a curious 5-year-old child could easily grasp and remember. Use a simple analogy."
PROMPT_L2_PHOTO_HS = "Thank you. Now, could you provide a more technical explanation suitable for a high school biology student? I'm familiar with basic cell biology and chemistry, so please include relevant terminology like chloroplasts, chlorophyll, and light-dependent reactions."
PROMPT_L2_RESP_LOC = "I'm interested in the specific location within the cell where this process occurs. Can you describe the organelles involved and why their specific structures are uniquely suited for this essential energy-releasing function?"
PROMPT_L2_RESP_PROD = "Focusing on the outputs of this metabolic reaction, what are the primary products that result from this process? Please list and briefly describe the significance of each of these molecules for the cell."
PROMPT_L3_PHOTO_ELI5_CHEF = "To make it really fun, please begin your explanation with the exact phrase 'Plants are like little chefs...' and continue that cooking analogy to describe how they make their sugary food."
PROMPT_L3_PHOTO_ELI5_SUN = "Let's zoom in on the energy source for this recipe. Can you specifically detail the crucial role that sunlight plays in this process? Explain what the sun's energy does and why it's so important for the plant's 'kitchen'."
PROMPT_L3_PHOTO_HS_EQ = "For a more precise, scientific understanding, please provide the balanced chemical equation for the overall photosynthetic reaction. Also, briefly explain what each part of the equation represents in the context of the plant's metabolism."
PROMPT_L3_PHOTO_HS_ALGAE = "How does this process in terrestrial plants compare to what happens in aquatic organisms like algae or cyanobacteria? Are there any significant differences in the mechanism, pigments used, or the cellular location?"
PROMPT_L3_RESP_LOC_MITO = "Please elaborate specifically on the role of the mitochondria. Describe its inner and outer membranes and the matrix, and explain how this structure makes it the perfect 'powerhouse' of the cell during this process."
PROMPT_L3_RESP_LOC_PA = "Is this metabolic pathway entirely identical in both plant and animal cells? Please compare and contrast the process, highlighting any key similarities or differences in where or how cellular respiration occurs in these two major kingdoms."
PROMPT_L3_RESP_PROD_ATP = "One of the key products is usable energy. Could you explain in detail the role of adenosine triphosphate (ATP) as the main energy currency? How is it synthesized and then used by the cell to power its activities?"
PROMPT_L3_RESP_PROD_CO2 = "I understand that carbon dioxide is considered a waste product of this process. Can you elaborate on what exactly happens to this CO2? How does the organism expel it, and what is its ultimate fate in the larger ecosystem?"


def get_prefix_tree_prompts() -> list[str]:
    prompts = []
    base_system = f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"

    p1 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_PHOTO}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_PHOTO_ELI5}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_PHOTO_ELI5_CHEF}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p1)

    p2 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_PHOTO}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_PHOTO_ELI5}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_PHOTO_ELI5_SUN}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p2)

    p3 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_PHOTO}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_PHOTO_HS}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_PHOTO_HS_EQ}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p3)

    p4 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_PHOTO}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_PHOTO_HS}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_PHOTO_HS_ALGAE}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p4)

    p5 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_RESP}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_RESP_LOC}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_RESP_LOC_MITO}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p5)

    p6 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_RESP}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_RESP_LOC}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_RESP_LOC_PA}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p6)

    p7 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_RESP}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_RESP_PROD}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_RESP_PROD_ATP}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p7)

    p8 = (f"{base_system}"
          f"{USER_HEADER}{PROMPT_L1_RESP}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L2_RESP_PROD}{EOT_ID}"
          f"{USER_HEADER}{PROMPT_L3_RESP_PROD_CO2}{EOT_ID}"
          f"{ASSISTANT_HEADER}")
    prompts.append(p8)

    return prompts


def send_request(user_prompt: str, call_generate: callable, args: argparse.Namespace):
    system_prompt = "You are a helpful, respectful and honest assistant."

    # Manually format the full prompt string
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

    base_prompts = get_prefix_tree_prompts()

    prompts = []
    for i in range(args.num_requests):
        prompts.extend(base_prompts)

    print(f"Starting benchmark with {args.num_requests} total requests.")
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        futures = [executor.submit(send_request, p, call_generate, args) for p in prompts]

        # just for the progress bar
        for future in tqdm(as_completed(futures), total=args.num_requests):
            pass

    total_time = time.monotonic() - start_time
    throughput = args.num_requests / total_time

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_6_prefix_tree.json', {
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
    benchmark_group.add_argument("--num-requests", type=int, default=64, help="Total number of requests to send. Will cycle through the 8 base prompts.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=64, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--max-tokens", type=int, default=32, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for the generation. 0.0 for deterministic output.")

    args = parser.parse_args()
    main(args)

# vLLM
# --- ✅ Benchmark Complete ---
# Total Time Taken: 6375.72 milliseconds
# Throughput:       10.04 requests/second
# --------------------------


# vLLM with APC on
# --- ✅ Benchmark Complete ---
# Total Time Taken: 4919.75 milliseconds
# Throughput:       13.01 requests/second
# --------------------------


# SGLang with RadixCache
# --- ✅ Benchmark Complete ---
# Total Time Taken: 3638.94 milliseconds
# Throughput:       17.59 requests/second
# --------------------------