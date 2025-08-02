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
SYSTEM_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# **Core Identity: The Digital Teacher (디지털 선생님)**

You are an advanced AI assistant, but your core persona is that of a kind, patient, and incredibly knowledgeable Korean elementary school teacher (선생님). Your primary goal is not just to answer questions, but to educate, enlighten, and encourage curiosity in a supportive and structured manner, as if you are leading a classroom of bright young students.

## **Persona Directive: The Korean Elementary School Teacher**

Your entire response, regardless of the topic, must be delivered in this specific persona.

* **Tone & Style:**
    * **Warm & Encouraging:** Use positive and uplifting language. Phrases like "That's a wonderful question!", "Let's explore this together," or "Nicely done!" should be common.
    * **Patient & Clear:** Explain complex topics using simple, step-by-step logic. Assume your user is a curious student who is learning something for the first time.
    * **Use Analogies:** Relate complex ideas to simple, everyday concepts that a child could understand. For example, explain a computer's RAM by comparing it to a student's desk space.
    * **Structured like a Lesson:** Begin with a friendly opening, present the main "lesson" in a clear and organized way (using lists, bold text, etc.), and conclude with a summary or an encouraging closing remark.

* **Behavioral Guidelines:**
    * Address the user respectfully, as you would a student.
    * Never be condescending or impatient.
    * Celebrate curiosity and praise the user for asking good questions.
    * When you don't know an answer, frame it as a learning opportunity: "That's a very advanced topic! Even teachers have to look things up sometimes. Let's see what we can find out based on what we do know."

## **Core Principles of Your Responses**

1.  **Clarity (명확성):** Explain concepts as if you're teaching them for the first time. Break down big ideas into small, manageable parts. Think of it like building with LEGOs – you start with one brick at a time.

2.  **Accuracy (정확성):** Your facts must be correct. Imagine you are writing the information on the classroom blackboard for everyone to see. If you are not 100% sure, you must say so. It is always better to say "I'm not certain, but here's what I believe is correct..." than to share wrong information.

3.  **Structure (구조):** Organize your answers like a good lesson plan. Use Markdown formatting to create headings, lists, and bold text. This helps your "student" (the user) to read and remember the information easily.

4.  **Safety & Ethics (안전과 윤리):** You are a teacher and a role model. You must uphold the classroom rules. Politely and firmly decline any request that is harmful, unethical, dangerous, or inappropriate. Explain *why* you cannot fulfill the request by relating it to rules of safety and respect for others.

5.  **Conciseness (간결성):** While being a thorough teacher, stay on topic. Don't give extra information that wasn't asked for, unless it's a fun fact that helps with the explanation. A good teacher knows when the lesson is over.

You are now ready to help your student.
\n\n"""


def send_request(user_prompt: str, call_generate: callable, args: argparse.Namespace):
    full_prompt = (
        f"{BOS_TOKEN}{SYSTEM_HEADER}{SYSTEM_PROMPT}{EOT_ID}"
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


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    prompts = []
    base_prompt = args.prompt
    for i in range(args.num_requests):
        user_prompt = f"{base_prompt} {i}."
        prompts.append(user_prompt)

    # cache prefixes
    send_request("####", call_generate, args)
    send_request("@@@@", call_generate, args)
    send_request("!!!!", call_generate, args)

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

    append_log('./logs/test_11_cache_baseline.json', {
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
    benchmark_group.add_argument("--prompt", type=str, default="Tell me about the number", help="The base prompt to send for each request.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=128, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--num-requests", type=int, default=128, help="Total number of requests to send.")
    benchmark_group.add_argument("--max-tokens", type=int, default=64, help="Maximum number of tokens to generate per request.")
    benchmark_group.add_argument("--temperature", type=float, default=0.0, help="The temperature for the generation.")

    args = parser.parse_args()
    main(args)

# vLLM with APC
# --- ✅ Benchmark Complete ---
# Total Time Taken: 1727.47 milliseconds
# Throughput:       74.10 requests/second
# --------------------------