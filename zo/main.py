import asyncio
# import csv  # W&B CHANGE: no longer needed (seed-score artifacts removed)
import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from contextlib import AsyncExitStack
import html  # W&B CHANGE: for wandb.Html logging
from collections import defaultdict  # >>> CHANGED

from blake3 import blake3
import numpy as np
import wandb  # Weights & Biases

# Assume pie is an installed library with this structure
# You may need to adjust imports based on the actual library
from pie import PieClient, Instance, compile_program
# Use the refactored imports
from countdown import CountdownTasksDataset, reward_function, SYSTEM_MESSAGE, USER_TEMPLATE, RESPONSE_PROMPT

# ==============================================================================
# Configuration
# ==============================================================================

# --- Server and Paths ---
SERVER_URIS = [
    "ws://127.0.0.1:8080",
    "ws://127.0.0.1:8081",
    "ws://127.0.0.1:8082",
    "ws://127.0.0.1:8083",
    "ws://127.0.0.1:8084",
    "ws://127.0.0.1:8085",
    "ws://127.0.0.1:8086",
    "ws://127.0.0.1:8087",
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
TRAINING_STEPS = 10000
POPULATION_SIZE = 128 * 8 #512 * 8  # Total number of seeds per step across all clients
TASKS_PER_SEED = 4        # Number of tasks to evaluate for each seed
NUM_ROLLOUT_WORKERS = 8#8   # Number of inferlets PER CLIENT
LORA_RANK = 8
LORA_ALPHA = 16.0
INITIAL_SIGMA = 0.005
MAX_SIGMA = 0.014
MU_FRACTION = 0.5
MAX_TOKENS_GEN = 512

# --- Dataset Config ---
DATA_PATH = "./Countdown-Tasks-3to4"

# --- W&B Config ---
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "pie-es-v4")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_MODE = os.getenv("WANDB_MODE")  # e.g., "offline"
WANDB_TAGS = ["es", "countdown", "lora"]

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

    # ---------------------------
    # W&B: initialize the run
    # ---------------------------
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{ADAPTER_NAME}-{int(time.time())}",
        job_type="training",
        tags=WANDB_TAGS,
        config={
            "adapter_name": ADAPTER_NAME,
            "training_steps": TRAINING_STEPS,
            "population_size": POPULATION_SIZE,
            "tasks_per_seed": TASKS_PER_SEED,
            "num_rollout_workers_per_client": NUM_ROLLOUT_WORKERS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "initial_sigma": INITIAL_SIGMA,
            "mu_fraction": MU_FRACTION,
            "max_sigma": MAX_SIGMA,  # W&B CHANGE: include max sigma
            "max_tokens_gen": MAX_TOKENS_GEN,
            "server_uris": SERVER_URIS,
            "data_path": DATA_PATH,
        },
        save_code=True,
        reinit=False,
        settings=wandb.Settings(start_method="thread", _disable_stats=False),
        mode=WANDB_MODE if WANDB_MODE else None,
    )
    # W&B CHANGE: simplify metric definitions and drive everything by `step`
    wandb.define_metric("step")
    wandb.define_metric("*", step_metric="step")

    try:
        # Optionally snapshot the code directory
        try:
            wandb.run.log_code(root=str(SCRIPT_DIR))
        except Exception:
            pass

        # --- 1. Connect and Compile Inferlets (Distributed) ---
        async with AsyncExitStack() as stack:
            clients = [
                await stack.enter_async_context(PieClient(uri))
                for uri in SERVER_URIS
            ]
            num_clients = len(clients)
            print(f"✅ Connected to {num_clients} Pie server(s).")
            wandb.config.update({"num_clients": num_clients}, allow_val_change=True)

            print("\nCompiling and uploading inferlets to all clients...")
            program_hashes = {}

            for src_filename in INFERLET_SRC_FILES:
                src_path = SCRIPT_DIR / src_filename
                program_bytes = await get_wasm_from_cache_or_compile(src_path, INFERLET_DEPS, SCRIPT_DIR)
                program_hash = blake3(program_bytes).hexdigest()

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
            init_tasks = [
                launch_and_get_result(client, program_hashes["es-init"], init_args, worker_id=f"C{i}-Init")
                for i, client in enumerate(clients)
            ]
            await asyncio.gather(*init_tasks)
            print("✅ Adapter initialized on all clients.")

            # --- 3. Load Dataset ---
            dataset = CountdownTasksDataset(DATA_PATH, "train", 100)
            wandb.config.update({"dataset_size_train_view": len(dataset)}, allow_val_change=True)

            # --- 4. Start Training Loop ---
            print("\n" + "=" * 50)
            print(f"Starting ES Training Loop with {num_clients} clients and {NUM_ROLLOUT_WORKERS} workers per client...")
            print("=" * 50)

            for step in range(1, TRAINING_STEPS + 1):
                start_time = time.time()
                print(f"\n--- Step {step}/{TRAINING_STEPS} ---")

                # --- 4a. Rollout Phase (Distributed) ---
                print(f"Phase: Rollout across {num_clients} clients")

                # >>> CHANGED: draw base seeds, repeat per task; sample tasks per seed
                base_seeds = np.random.randint(-2**63, 2**63 - 1, size=POPULATION_SIZE, dtype=np.int64)  # seeds per population member
                num_total_tasks = POPULATION_SIZE * TASKS_PER_SEED

                # sample TASKS_PER_SEED tasks for each seed, then flatten
                task_idx_matrix = np.random.choice(len(dataset), size=(POPULATION_SIZE, TASKS_PER_SEED), replace=True)
                all_tasks = [dataset[i] for i in task_idx_matrix.reshape(-1)]

                # repeat each seed TASKS_PER_SEED times to match tasks (len(seeds) == len(tasks))
                repeated_seeds = np.repeat(base_seeds, TASKS_PER_SEED)  # len == num_total_tasks

                # Distribute parallel arrays among clients
                client_seed_chunks = np.array_split(repeated_seeds, num_clients)
                client_batch_chunks = np.array_split(np.array(all_tasks, dtype=object), num_clients)

                rollout_futures = []  # >>> CHANGED: keep metadata to rebuild seed->score mapping
                rollout_meta = []

                for client_idx, client in enumerate(clients):
                    client_seeds = client_seed_chunks[client_idx]
                    client_batch = client_batch_chunks[client_idx]
                    if len(client_seeds) == 0:
                        continue

                    # Distribute each client's work among its inferlets
                    worker_seed_chunks = np.array_split(client_seeds, NUM_ROLLOUT_WORKERS)
                    worker_batch_chunks = np.array_split(client_batch, NUM_ROLLOUT_WORKERS)

                    for worker_idx in range(NUM_ROLLOUT_WORKERS):
                        worker_seeds = worker_seed_chunks[worker_idx]
                        worker_batch = worker_batch_chunks[worker_idx]
                        if len(worker_seeds) == 0:
                            continue

                        # build prompts for this worker's tasks
                        worker_tasks_json = json.dumps([item["prompt"] for item in worker_batch])

                        rollout_args = [
                            "--name", ADAPTER_NAME,
                            "--seeds", ",".join(map(str, worker_seeds)),  # >>> CHANGED: seeds list matches tasks
                            "--tasks-json", worker_tasks_json,
                            "--max-num-outputs", str(MAX_TOKENS_GEN),
                        ]
                        descriptive_worker_id = f"C{client_idx}-W{worker_idx}"
                        fut = launch_and_get_result(client, program_hashes["es-rollout"], rollout_args, worker_id=descriptive_worker_id)
                        rollout_futures.append(fut)
                        # store seeds + tasks for alignment after gather
                        rollout_meta.append({
                            "seeds": worker_seeds.tolist(),
                            "tasks": list(worker_batch),
                            "who": descriptive_worker_id,
                        })

                worker_results_json = await asyncio.gather(*rollout_futures)
                #print(worker_results_json)
                # Combine results + aligned seeds/tasks
                generated_texts = []
                generated_seeds = []  # >>> CHANGED
                generated_tasks = []  # >>> CHANGED
                mismatch = False

                for res_json, meta in zip(worker_results_json, rollout_meta):
                    if res_json:
                        texts = json.loads(res_json)
                        if len(texts) != len(meta["seeds"]) or len(texts) != len(meta["tasks"]):
                            print(f"⚠️ Warning: Worker {meta['who']} returned {len(texts)} outputs "
                                  f"but had {len(meta['seeds'])} seeds / {len(meta['tasks'])} tasks.")
                            mismatch = True
                        generated_texts.extend(texts)
                        generated_seeds.extend(meta["seeds"])
                        generated_tasks.extend(meta["tasks"])
                    else:
                        print(f"⚠️ Warning: A rollout worker ({meta['who']}) did not return a result.")
                        mismatch = True

                if mismatch or len(generated_texts) != num_total_tasks or len(generated_seeds) != num_total_tasks:
                    msg = (f"Mismatch in expected tasks ({num_total_tasks}) vs "
                           f"generated_texts ({len(generated_texts)}), seeds ({len(generated_seeds)}). Skipping step.")
                    print(f"❌ Error: {msg}")
                    try:
                        wandb.alert(title="Rollout mismatch", text=msg, level=wandb.AlertLevel.WARN)
                    except Exception:
                        pass
                    wandb.log({
                        "step": step,
                        "perf/step_duration_sec": float(time.time() - start_time),
                    }, step=step)
                    continue

                # --- Output length stats ---
                out_lens_chars = [len(t) for t in generated_texts]
                out_lens_ws_tokens = [len(t.split()) for t in generated_texts]
                avg_len_chars = float(np.mean(out_lens_chars))
                avg_len_ws_tokens = float(np.mean(out_lens_ws_tokens))

                # --- 4b. Scoring Phase ---
                print("Phase: Scoring")
                reward_infos = [
                    reward_function(text, generated_tasks[i]["nums"], generated_tasks[i]["target"], end_token="<|eot_id|>")  # >>> CHANGED: use aligned tasks
                    for i, text in enumerate(generated_texts)
                ]
                scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
                answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]
                format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]

                # --- 4c. Aggregation Phase ---
                print("Phase: Aggregating Scores")
                # >>> CHANGED: value-based grouping by actual seed values
                scores_by_seed: dict[int, list[float]] = defaultdict(list)
                for s, sc in zip(generated_seeds, scores):
                    scores_by_seed[int(s)].append(float(sc))

                # average per base seed, preserving base_seeds' order for update
                aggregated_scores = [float(np.mean(scores_by_seed[int(s)])) for s in base_seeds]  # len == POPULATION_SIZE

                step_mean = float(np.mean(scores))
                step_std = float(np.std(scores))
                step_max = float(np.max(scores))
                step_min = float(np.min(scores))
                success_rate_train = float(np.mean(answer_rewards)) if len(answer_rewards) else 0.0

                mean_population_score = float(np.mean(aggregated_scores))
                mu_k = max(1, int(np.ceil(MU_FRACTION * POPULATION_SIZE)))
                top_mu_mean = float(np.mean(sorted(aggregated_scores, reverse=True)[:mu_k]))

                # --- (New) Log a small qualitative HTML sample like script #2 ---
                num_to_log = min(5, len(generated_texts))
                if num_to_log > 0:
                    episode_html = "<br><hr><br>".join(
                        f"<pre>{html.escape(generated_texts[idx][:2048])}</pre>"
                        for idx in np.random.choice(len(generated_texts), size=num_to_log, replace=False)
                    )
                else:
                    episode_html = "<pre>No predictions</pre>"

                # --- 4d. Update Phase (Distributed) ---
                print("Phase: Update")
                update_args = [
                    "--name", ADAPTER_NAME,
                    "--seeds", ",".join(map(str, base_seeds)),          # >>> CHANGED: one score per unique/base seed
                    "--scores", ",".join(map(str, aggregated_scores)),  # aligned with base_seeds order
                    "--max-sigma", str(MAX_SIGMA),
                ]
                update_tasks = [
                    launch_and_get_result(client, program_hashes["es-update"], update_args, worker_id=f"C{i}-Update")
                    for i, client in enumerate(clients)
                ]
                await asyncio.gather(*update_tasks)

                step_duration = time.time() - start_time
                print(f"Step {step} completed in {step_duration:.2f} seconds.")

                # --- W&B: log step metrics ---
                wandb.log(
                    {
                        "step": step,
                        "mean_reward": step_mean,
                        "std_reward": step_std,
                        "success_rate/train": success_rate_train,
                        "duration_seconds": float(step_duration),
                        "perf/step_duration_sec": float(step_duration),
                        "num_finished_episodes": int(len(generated_texts)),
                        "mean_response_len": float(avg_len_ws_tokens),
                        "answer_reward/mean": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
                        "format_reward/mean": float(np.mean(format_rewards)) if format_rewards else 0.0,
                        "es/mean_population_score": mean_population_score,
                        "es/mean_fittest_score": top_mu_mean,
                        "episodes_text": wandb.Html(episode_html),
                    },
                    step=step,
                )

            print("\nTraining finished!")
            wandb.summary["final_mean_score"] = step_mean
            wandb.summary["final_max_score"] = step_max
            wandb.summary["final_min_score"] = step_min
            wandb.summary["final_success_rate_train"] = success_rate_train  # W&B CHANGE

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
