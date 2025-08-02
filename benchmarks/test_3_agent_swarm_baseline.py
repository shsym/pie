import argparse
import queue
import random
import time
from concurrent.futures import ThreadPoolExecutor
from test_utils import get_call_generate, append_log

ROLES = [
    {
        "name": "Story Idea Generator",
        "system_message": "You are an expert idea generator on a collaborative story-writing team. Your role is to create a compelling, one-sentence story concept.",
        "task_instruction": "Based on the user's request, generate a single, captivating sentence that establishes the core conflict or mystery of a story.",
        "section_header": "Concept",
        "prev_topic": None,
        "next_topic": "concept_to_plot"
    },
    {
        "name": "Plot Developer",
        "system_message": "You are a master storyteller on a collaborative writing team. Your role is to expand a story concept into a structured plot outline.",
        "task_instruction": "Read the provided story **Concept**. Your task is to write a brief plot outline with three distinct acts (Act 1: Setup, Act 2: Confrontation, Act 3: Resolution).",
        "section_header": "Plot Outline",
        "prev_topic": "concept_to_plot",
        "next_topic": "plot_to_chars"
    },
    {
        "name": "Character Creator",
        "system_message": "You are an expert character designer on a collaborative writing team. Your role is to create a memorable protagonist and antagonist.",
        "task_instruction": "Read the **Concept** and **Plot Outline**. Your task is to create a one-sentence description for a compelling protagonist and a formidable antagonist that fit the story.",
        "section_header": "Characters",
        "prev_topic": "plot_to_chars",
        "next_topic": "chars_to_dialogue"
    },
    {
        "name": "Dialogue Writer",
        "system_message": "You are a skilled dialogue writer on a collaborative writing team. Your role is to write a key piece of dialogue.",
        "task_instruction": "Read all the story elements. Your task is to write a single, impactful line of dialogue spoken by the protagonist during the story's climax.",
        "section_header": "Climax Dialogue",
        "prev_topic": "chars_to_dialogue",
        "next_topic": None
    },
]

TOPICS = ["concept_to_plot", "plot_to_chars", "chars_to_dialogue"]


def agent(
        name: str, system_message: str, task_instruction: str, section_header: str,
        prev_topic: str, next_topic: str, group_id: int,
        initial_prompt_q: queue.Queue, comm_queues: dict, final_story_q: queue.Queue,
        call_generate, args: argparse.Namespace
):
    if prev_topic is None:
        # The first agent gets the initial prompt
        initial_prompt = initial_prompt_q.get()
        user_prompt = f"{task_instruction}\n\nRequest: A story about {initial_prompt}."
        accumulated_story = ""
    else:
        # Subsequent agents receive the work from the previous agent
        accumulated_story = comm_queues[prev_topic].get()
        user_prompt = (
            f"**Previous Story Elements:**\n---\n{accumulated_story}\n---\n\n"
            f"**Your Specific Task:**\n{task_instruction}"
        )

    # Construct the context for the LLM call
    s = "<|begin_of_text|>"
    s += f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
    s += f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
    s += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    s += f"### {section_header}\n"  # Prime the model to start with the header

    contribution = call_generate(s, max_tokens=args.tokens_per_step, temperature=0.5, stop="<|eot_id|>")

    new_accumulated_story = f"{accumulated_story}\n### {section_header}\n{contribution}".strip()

    # Pass the work to the next agent or the final collector
    if next_topic is not None:
        comm_queues[next_topic].put(new_accumulated_story)
    else:
        final_story_q.put(new_accumulated_story)


def main(args):
    call_generate = get_call_generate(args.backend, args.host, args.port, args.model_path)

    pipeline_queues = [
        {
            "initial": queue.Queue(),
            "final": queue.Queue(),
            "comm": {topic: queue.Queue() for topic in TOPICS}
        }
        for _ in range(args.num_pipelines)
    ]

    print(f"Starting benchmark with {args.num_pipelines} total requests.")

    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.num_max_workers) as executor:
        for i in range(args.num_pipelines):
            for role_info in ROLES:
                executor.submit(
                    agent,
                    name=role_info["name"],
                    system_message=role_info["system_message"],
                    task_instruction=role_info["task_instruction"],
                    section_header=role_info["section_header"],
                    prev_topic=role_info["prev_topic"],
                    next_topic=role_info["next_topic"],
                    group_id=i,
                    initial_prompt_q=pipeline_queues[i]["initial"],
                    comm_queues=pipeline_queues[i]["comm"],
                    final_story_q=pipeline_queues[i]["final"],
                    call_generate=call_generate,
                    args=args
                )

        genres = ['a haunted spaceship', 'a detective who can talk to ghosts', 'a romance in a city that floats', 'a silent film star transported to the future']
        for i in range(args.num_pipelines):
            pipeline_queues[i]["initial"].put(random.choice(genres))

        final_stories = [pipeline_queues[i]["final"].get() for i in range(args.num_pipelines)]

    total_time = time.monotonic() - start_time
    total_agents = args.num_pipelines
    throughput = total_agents / total_time if total_time > 0 else 0

    print("\n--- ✅ Benchmark Complete ---")
    print(f"Total Time Taken: {total_time * 1000:.2f} milliseconds")
    print(f"Throughput:       {throughput:.2f} requests/second")
    print("--------------------------")

    append_log('./logs/test_3_agent_swarm_baseline.json', {
        'total_time': total_time,
        'throughput': throughput,
        'args': vars(args),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    server_group = parser.add_argument_group('Serving Engine Configuration')
    server_group.add_argument("--backend", type=str, default="vllm", help="Backend for text generation.")
    server_group.add_argument("--host", type=str, default="http://127.0.0.1", help="Host address of the LLM server.")
    server_group.add_argument("--port", type=int, default=8000, help="Port number of the LLM server.")
    server_group.add_argument("--model-path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model path or identifier.")

    benchmark_group = parser.add_argument_group('Benchmark Configuration')
    benchmark_group.add_argument("--num-pipelines", type=int, default=32, help="Number of story pipelines to run in parallel.")
    benchmark_group.add_argument("--num-max-workers", type=int, default=16, help="Maximum number of threadpool workers to use.")
    benchmark_group.add_argument("--tokens-per-step", type=int, default=96, help="Max new tokens each agent should generate.")

    args = parser.parse_args()
    main(args)

# --- ✅ Benchmark Complete ---
# Total Time Taken: 10652.75 milliseconds
# Throughput:       3.00 requests/second
# --------------------------