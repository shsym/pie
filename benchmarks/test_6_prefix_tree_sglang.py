#!/usr/bin/env python3
import argparse
import time
from typing import List

import sglang as sgl
from sglang import RuntimeEndpoint

# --- Llama-3.2 Instruct Chat Template Tokens ---
# Note: sglang handles the special tokens internally when using roles,
# but we define the prompt content here for clarity.

# --- Prompt Tree Components (from the Rust example) ---

# Level 0: System Prompt
SYSTEM_PROMPT = "You are a helpful, friendly, and knowledgeable science tutor for students of all ages. Your goal is to explain complex biological concepts in a clear, accessible, and engaging manner, tailoring your language to the specified audience."

# Level 1: Main Topics
PROMPT_L1_PHOTO = "I'm curious about the fundamental process of photosynthesis. Could you provide a detailed overview of how plants create their own food using sunlight, water, and carbon dioxide?"
PROMPT_L1_RESP = "Now, could you explain the equally important process of cellular respiration? I'd like to understand how organisms, including plants and animals, break down glucose to release the energy needed for life."

# Level 2: Sub-Topics
PROMPT_L2_PHOTO_ELI5 = "That sounds complicated. Could you simplify it significantly for me? Please explain the core idea in a way that a curious 5-year-old child could easily grasp and remember. Use a simple analogy."
PROMPT_L2_PHOTO_HS = "Thank you. Now, could you provide a more technical explanation suitable for a high school biology student? I'm familiar with basic cell biology and chemistry, so please include relevant terminology like chloroplasts, chlorophyll, and light-dependent reactions."
PROMPT_L2_RESP_LOC = "I'm interested in the specific location within the cell where this process occurs. Can you describe the organelles involved and why their specific structures are uniquely suited for this essential energy-releasing function?"
PROMPT_L2_RESP_PROD = "Focusing on the outputs of this metabolic reaction, what are the primary products that result from this process? Please list and briefly describe the significance of each of these molecules for the cell."

# Level 3: Leaf Prompts
PROMPT_L3_PHOTO_ELI5_CHEF = "To make it really fun, please begin your explanation with the exact phrase 'Plants are like little chefs...' and continue that cooking analogy to describe how they make their sugary food."
PROMPT_L3_PHOTO_ELI5_SUN = "Let's zoom in on the energy source for this recipe. Can you specifically detail the crucial role that sunlight plays in this process? Explain what the sun's energy does and why it's so important for the plant's 'kitchen'."
PROMPT_L3_PHOTO_HS_EQ = "For a more precise, scientific understanding, please provide the balanced chemical equation for the overall photosynthetic reaction. Also, briefly explain what each part of the equation represents in the context of the plant's metabolism."
PROMPT_L3_PHOTO_HS_ALGAE = "How does this process in terrestrial plants compare to what happens in aquatic organisms like algae or cyanobacteria? Are there any significant differences in the mechanism, pigments used, or the cellular location?"
PROMPT_L3_RESP_LOC_MITO = "Please elaborate specifically on the role of the mitochondria. Describe its inner and outer membranes and the matrix, and explain how this structure makes it the perfect 'powerhouse' of the cell during this process."
PROMPT_L3_RESP_LOC_PA = "Is this metabolic pathway entirely identical in both plant and animal cells? Please compare and contrast the process, highlighting any key similarities or differences in where or how cellular respiration occurs in these two major kingdoms."
PROMPT_L3_RESP_PROD_ATP = "One of the key products is usable energy. Could you explain in detail the role of adenosine triphosphate (ATP) as the main energy currency? How is it synthesized and then used by the cell to power its activities?"
PROMPT_L3_RESP_PROD_CO2 = "I understand that carbon dioxide is considered a waste product of this process. Can you elaborate on what exactly happens to this CO2? How does the organism expel it, and what is its ultimate fate in the larger ecosystem?"


@sgl.function
def science_tutor_tree(s, grp_prefix, max_tokens, temperature):
    """
    Defines the 8-prompt generation tree using sglang's native forking.
    """
    # Level 0: System Prompt
    s += sgl.system(grp_prefix+ SYSTEM_PROMPT)

    # Level 1 Forks: Photosynthesis vs. Respiration
    l1_forks = s.fork(2)
    l1_forks[0] += sgl.user(PROMPT_L1_PHOTO)
    l1_forks[1] += sgl.user(PROMPT_L1_RESP)

    # Level 2 Forks
    l2_photo_forks = l1_forks[0].fork(2)
    l2_photo_forks[0] += sgl.user(PROMPT_L2_PHOTO_ELI5)
    l2_photo_forks[1] += sgl.user(PROMPT_L2_PHOTO_HS)

    l2_resp_forks = l1_forks[1].fork(2)
    l2_resp_forks[0] += sgl.user(PROMPT_L2_RESP_LOC)
    l2_resp_forks[1] += sgl.user(PROMPT_L2_RESP_PROD)

    # Level 3 Forks (creating the 8 leaf nodes)
    l3_p_e_forks = l2_photo_forks[0].fork(2)
    l3_p_e_forks[0] += sgl.user(PROMPT_L3_PHOTO_ELI5_CHEF)
    l3_p_e_forks[1] += sgl.user(PROMPT_L3_PHOTO_ELI5_SUN)

    l3_p_h_forks = l2_photo_forks[1].fork(2)
    l3_p_h_forks[0] += sgl.user(PROMPT_L3_PHOTO_HS_EQ)
    l3_p_h_forks[1] += sgl.user(PROMPT_L3_PHOTO_HS_ALGAE)

    l3_r_l_forks = l2_resp_forks[0].fork(2)
    l3_r_l_forks[0] += sgl.user(PROMPT_L3_RESP_LOC_MITO)
    l3_r_l_forks[1] += sgl.user(PROMPT_L3_RESP_LOC_PA)

    l3_r_p_forks = l2_resp_forks[1].fork(2)
    l3_r_p_forks[0] += sgl.user(PROMPT_L3_RESP_PROD_ATP)
    l3_r_p_forks[1] += sgl.user(PROMPT_L3_RESP_PROD_CO2)

    # Combine all leaf states into a single list
    all_leaves = (
            l3_p_e_forks[0], l3_p_e_forks[1],
            l3_p_h_forks[0], l3_p_h_forks[1],
            l3_r_l_forks[0], l3_r_l_forks[1],
            l3_r_p_forks[0], l3_r_p_forks[1]
    )

    # Generate from all 8 leaves concurrently.
    # sglang leverages the shared prefixes automatically.
    for leaf in all_leaves:
        leaf += sgl.assistant(sgl.gen("generation", max_tokens=max_tokens, temperature=temperature))

    l3_p_e_forks.join()
    l3_p_h_forks.join()
    l3_r_l_forks.join()
    l3_r_p_forks.join()

    return all_leaves

def main(args: argparse.Namespace):
    """
    Main function to set up and run the sglang benchmark.
    """
    # --- 1. Initialize Backend ---
    print("--- 1. Initializing SGLang Backend ---")
    backend = RuntimeEndpoint(f"{args.host}:{args.port}")
    print(f"✅ Successfully initialized SGLang client for endpoint: {args.host}:{args.port}")


    # Prepare arguments for run_batch. Each item in the list corresponds to one full 8-prompt tree execution.
    batch_args = [
        {"max_tokens": args.max_tokens, "temperature": args.temperature, "grp_prefix": str(i)}
        for i in range(args.num_requests)
    ]
    print(f"\n--- 2. Preparing Benchmark ---")

    # --- 3. Run Benchmark ---
    print(f"\n--- 3. Running Benchmark ---")
    print(f"🚀 Starting benchmark...")

    start_time = time.monotonic()

    # Run the entire batch. SGLang's backend handles concurrent execution.
    states = science_tutor_tree.run_batch(
        batch_args,
        backend=backend,
        progress_bar=True,
    )

    for s in states:
        print(s.text())

    total_time = time.monotonic() - start_time

    # --- 4. Print Results ---
    print("\n--- ✅ Benchmark Complete ---")

    # Each state in the output corresponds to one full 8-prompt tree run.
    # We can inspect the generated text if needed, e.g., states[0][0].get_leaf_state("generation").text
    successful_runs = len(states)
    successful_requests = successful_runs

    throughput = successful_requests / total_time if total_time > 0 else 0
    # This latency is for an entire 8-prompt tree to complete.
    avg_latency_per_tree = (total_time / successful_runs) * 1000 if successful_runs > 0 else 0

    print(f"\nTotal Time Taken:      {total_time:.2f} seconds")
    print(f"Successful Prompts:    {successful_requests}")
    print("--------------------------")
    print(f"Throughput:            {throughput:.2f} prompts/second")
    print(f"Avg. Latency per Tree: {avg_latency_per_tree:.2f} ms (for 8 concurrent prompts)")
    print("--------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="The host of the SGLang server.")
    server_group.add_argument("--port", type=int, default=8000, help="The port of the SGLang server.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-requests", type=int, default=64, help="Total number of prompts to send. MUST be a multiple of 8.")
    benchmark_group.add_argument("--max-tokens", type=int, default=32, help="Maximum number of new tokens to generate per prompt.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for generation. 0.0 for deterministic output.")

    args = parser.parse_args()
    main(args)
