import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, List, Optional
from contextlib import AsyncExitStack

from blake3 import blake3
import numpy as np

# Assume pie is an installed library with this structure
# You may need to adjust imports based on the actual library
from pie import PieClient, Instance, compile_program
# Use the refactored imports
from countdown import CountdownTasksDataset, reward_function, SYSTEM_MESSAGE, USER_TEMPLATE, RESPONSE_PROMPT

# ==============================================================================
# Configuration
# ==============================================================================

# --- Server and Paths ---
# MODIFIED: Provide a list of websocket URIs for distributed clients.
SERVER_URIS = [
    "ws://127.0.0.1:8080",
    # "ws://127.0.0.1:8081", # <-- Add more client URIs here
]
SCRIPT_DIR = Path(__file__).resolve().parent
# NOTE: Update this path to your actual inferlet dependency location
INFERLET_DEPS = [
    'inferlet={ path = "/root/Workspace/pie/inferlet" }',
    'inferlet-macros={ path = "/root/Workspace/pie/inferlet-macros" }',
    'pico-args = "0.5.0"',
    'futures = "0.3.31"',
    'serde = { version = "1.0", features = ["derive"] }',
    'serde_json = "1.0"',
]
INFERLET_SRC_FILES = [
    "es-init.rs",
    "es-rollout.rs",
    "es-update.rs",
]

# --- ES Hyperparameters ---
ADAPTER_NAME = "evo-countdown-v1"
TRAINING_STEPS = 100
POPULATION_SIZE = 512  # Total number of seeds per step across all clients
TASKS_PER_SEED = 5  # Number of tasks to evaluate for each seed
NUM_ROLLOUT_WORKERS = 8  # Number of inferlets PER CLIENT
LORA_RANK = 8
LORA_ALPHA = 16.0
INITIAL_SIGMA = 0.0001
MU_FRACTION = 0.25
MAX_TOKENS_GEN = 512

# --- Dataset Config ---
# This would point to your actual dataset
DATA_PATH = "./Countdown-Tasks-3to4"


# ==============================================================================
# Inferlet Management & Execution
# ==============================================================================

def cache_paths_for_file(src_path: Path, base_dir: Path):
    """Computes cache filenames using the BLAKE3 hash of the file contents."""
    code_bytes = src_path.read_bytes()
    code_hash = blake3(code_bytes).hexdigest()
    wasm_path = base_dir / f"program-{code_hash}.wasm"
    hash_path = base_dir / f"program-{code_hash}.hash"
    return code_hash, wasm_path, hash_path


async def get_wasm_from_cache_or_compile(src_path: Path, deps: list[str], base_dir: Path) -> bytes:
    """Reuses cached WASM or compiles the Rust source file using the provided client."""
    if not src_path.exists():
        raise FileNotFoundError(f"Rust source file not found: {src_path}")

    code_hash, wasm_path, hash_path = cache_paths_for_file(src_path, base_dir)

    if wasm_path.exists():
        print(f"[Cache] Reusing cached wasm for {src_path.name} (hash {code_hash[:8]})")
        return wasm_path.read_bytes()

    print(f"[Compile] No cache for {src_path.name}...")
    program_bytes = await compile_program(str(src_path), deps)
    wasm_path.write_bytes(program_bytes)
    hash_path.write_text(code_hash + "\n")
    print(f"[Compile] Saved wasm to {wasm_path.name}")
    return program_bytes


async def launch_and_get_result(client: PieClient, program_hash: str, arguments: list[str], worker_id: Any = 0) -> Optional[str]:
    """Launches an inferlet and waits for it to terminate, returning the final message."""
    # MODIFIED: Use `worker_id` for more descriptive logging.
    print(f"🚀 Worker {worker_id}: Launching instance with hash {program_hash[:8]}...")
    instance = await client.launch_instance(program_hash, arguments=arguments)
    final_payload = None
    while True:
        event, message = await instance.recv()
        if event == "terminated":
            print(f"✅ Worker {worker_id}: Instance {instance.instance_id} finished. Reason: {message}")
            break
        elif event == "message":
            final_payload = message
        else:
            print(f"  [Worker {worker_id} / {instance.instance_id} / {event}] {message}")
    return final_payload


# ==============================================================================
# Main Training Script
# ==============================================================================

async def main():
    """The main entry point for the ES training process."""

    # --- 1. Connect and Compile Inferlets (Distributed) ---
    # MODIFIED: Use AsyncExitStack to manage multiple client connections.
    async with AsyncExitStack() as stack:
        clients = [
            await stack.enter_async_context(PieClient(uri))
            for uri in SERVER_URIS
        ]
        num_clients = len(clients)
        print(f"✅ Connected to {num_clients} Pie server(s).")

        print("\nCompiling and uploading inferlets to all clients...")
        program_hashes = {}
        # Use the first client to handle compilation to avoid redundant work.

        for src_filename in INFERLET_SRC_FILES:
            src_path = SCRIPT_DIR / src_filename
            # Compile once using one client.
            program_bytes = await get_wasm_from_cache_or_compile(src_path, INFERLET_DEPS, SCRIPT_DIR)
            program_hash = blake3(program_bytes).hexdigest()

            # Upload to all clients in parallel if the program doesn't exist.
            upload_tasks = []
            for client in clients:
                if not await client.program_exists(program_hash):
                    print(f"Uploading program {src_filename} ({program_hash[:8]})...")
                    upload_tasks.append(client.upload_program(program_bytes))

            await asyncio.gather(*upload_tasks)
            if upload_tasks:
                print(f"Program {src_filename} uploaded to all necessary clients.")

            program_hashes[src_path.stem] = program_hash

        # --- 2. Initialize the Adapter (Distributed) ---
        print("\n" + "=" * 50)
        print("Step 0: Initializing ES Adapter on all clients")
        print("=" * 50)
        init_args = [
            "--name", ADAPTER_NAME,
            "--rank", str(LORA_RANK),
            "--alpha", str(LORA_ALPHA),
            "--population-size", str(POPULATION_SIZE),
            "--mu-fraction", str(MU_FRACTION),
            "--initial-sigma", str(INITIAL_SIGMA),
        ]
        # MODIFIED: Run initialization on all clients in parallel.
        init_tasks = [
            launch_and_get_result(client, program_hashes["es-init"], init_args, worker_id=f"C{i}-Init")
            for i, client in enumerate(clients)
        ]
        await asyncio.gather(*init_tasks)
        print("✅ Adapter initialized on all clients.")

        # --- 3. Load Dataset ---
        dataset = CountdownTasksDataset(DATA_PATH, "train", 100)

        # --- 4. Start Training Loop ---
        print("\n" + "=" * 50)
        print(f"Starting ES Training Loop with {num_clients} clients and {NUM_ROLLOUT_WORKERS} workers per client...")
        print("=" * 50)

        for step in range(1, TRAINING_STEPS + 1):
            start_time = time.time()
            print(f"\n--- Step {step}/{TRAINING_STEPS} ---")

            # --- 4a. Rollout Phase (Distributed) ---
            print(f"Phase: Rollout across {num_clients} clients")
            seeds = np.random.randint(-2 ** 63, 2 ** 63 - 1, size=POPULATION_SIZE, dtype=np.int64)
            num_total_tasks = POPULATION_SIZE * TASKS_PER_SEED
            sample_indices = np.random.choice(len(dataset), size=num_total_tasks, replace=True)
            batch = [dataset[i] for i in sample_indices]
            prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MESSAGE}<|eot_id|>"

            # **MODIFIED: 1. Distribute work among CLIENTS**
            client_seed_chunks = np.array_split(seeds, num_clients)
            client_batch_chunks = np.array_split(np.array(batch), num_clients)

            rollout_tasks = []
            for client_idx, client in enumerate(clients):
                client_seeds = client_seed_chunks[client_idx]
                client_batch = client_batch_chunks[client_idx]

                if len(client_seeds) == 0: continue

                # **MODIFIED: 2. Distribute each client's work among its INFERLETS**
                worker_seed_chunks = np.array_split(client_seeds, NUM_ROLLOUT_WORKERS)
                worker_batch_chunks = np.array_split(client_batch, NUM_ROLLOUT_WORKERS)

                #print('len client seeds', len(client_seeds))

                for worker_idx in range(NUM_ROLLOUT_WORKERS):
                    worker_seeds = worker_seed_chunks[worker_idx]
                    worker_batch = worker_batch_chunks[worker_idx]

                    #print('len worker_seeds seeds', len(worker_seeds))

                    if len(worker_seeds) == 0: continue

                    worker_task_prompts = [
                        f"<|start_header_id|>user<|end_header_id|>\n\n{USER_TEMPLATE.format(numbers=item['nums'], target=item['target'])}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{RESPONSE_PROMPT}"
                        for item in worker_batch
                    ]
                    worker_tasks_json = json.dumps(worker_task_prompts)

                    rollout_args = [
                        "--name", ADAPTER_NAME,
                        "--prefix", prefix,
                        "--seeds", ",".join(map(str, worker_seeds)),
                        "--tasks-json", worker_tasks_json,
                        "--max-num-outputs", str(MAX_TOKENS_GEN),
                    ]

                    # Create a unique ID for logging: C for Client, W for Worker
                    descriptive_worker_id = f"C{client_idx}-W{worker_idx}"
                    #print(descriptive_worker_id)
                    rollout_tasks.append(
                        launch_and_get_result(client, program_hashes["es-rollout"], rollout_args, worker_id=descriptive_worker_id)
                    )

            # **MODIFIED: Run all rollout workers across all clients concurrently**
            worker_results_json = await asyncio.gather(*rollout_tasks)
            # print(worker_results_json)

            # **MODIFIED: Combine results from all workers**
            generated_texts = []
            for res_json in worker_results_json:
                if res_json:
                    generated_texts.extend(json.loads(res_json))
                else:
                    print("⚠️ Warning: A rollout worker did not return a result.")

            if len(generated_texts) != num_total_tasks:
                print(f"❌ Error: Mismatch in expected tasks ({num_total_tasks}) and generated texts ({len(generated_texts)}). Skipping step.")
                continue

            # --- 4b. Scoring Phase ---
            print("Phase: Scoring")
            scores = [
                reward_function(text, batch[i]["nums"], batch[i]["target"], end_token="<|eot_id|>")["reward"]
                for i, text in enumerate(generated_texts)
            ]

            # --- 4c. Aggregation Phase ---
            print("Phase: Aggregating Scores")
            aggregated_scores = [
                np.mean(scores[i * TASKS_PER_SEED: (i + 1) * TASKS_PER_SEED])
                for i in range(POPULATION_SIZE)
            ]
            print(f"Average reward for this step: {np.mean(scores):.4f}")

            # --- 4d. Update Phase (Distributed) ---
            print("Phase: Update")
            update_args = [
                "--name", ADAPTER_NAME,
                "--seeds", ",".join(map(str, seeds)),
                "--scores", ",".join(map(str, aggregated_scores)),
            ]
            #MODIFIED: Run update on all clients in parallel to keep them synchronized.
            update_tasks = [
                launch_and_get_result(client, program_hashes["es-update"], update_args, worker_id=f"C{i}-Update")
                for i, client in enumerate(clients)
            ]
            await asyncio.gather(*update_tasks)

            step_duration = time.time() - start_time
            print(f"Step {step} completed in {step_duration:.2f} seconds.")

        print("\nTraining finished!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
