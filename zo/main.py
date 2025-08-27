import asyncio

# import csv  # W&B CHANGE: no longer needed (seed-score artifacts removed)
import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from contextlib import AsyncExitStack
import html  # W&B CHANGE: for wandb.Html logging
from collections import defaultdict, deque  # >>> CHANGED: add deque

from blake3 import blake3
import numpy as np
import wandb  # Weights & Biases

# progress bars
from tqdm.auto import tqdm

# Assume pie is an installed library with this structure
# You may need to adjust imports based on the actual library
from pie import PieClient, Instance, compile_program
# Use the refactored imports
from countdown import CountdownTasksDataset, reward_function, SYSTEM_MESSAGE, USER_TEMPLATE, RESPONSE_PROMPT

# ==============================================================================
# Configuration
# ==============================================================================

# --- Logging Verbosity ---
# Toggle to re-enable chatty per-worker prints if desired.
VERBOSE_WORKER_LOGS = False

# --- Server and Paths ---
SERVER_URIS = [
    "ws://127.0.0.1:8080",
    "ws://127.0.0.1:8081",
    "ws://127.0.0.1:8082",
    "ws://127.0.0.1:8083",
    # "ws://127.0.0.1:8084",
    # "ws://127.0.0.1:8085",
    # "ws://127.0.0.1:8086",
    # "ws://127.0.0.1:8087",
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
POPULATION_SIZE = 512 * 2  # Total seeds per step across all clients
TASKS_PER_SEED = 4  # Number of tasks to evaluate for each seed

# >>> CHANGED: remove NUM_ROLLOUT_WORKERS, add idiomatic NUM_ROLLOUTS_PER_WORKER
# Each inferlet handles this many (seed, task) requests concurrently.
NUM_ROLLOUTS_PER_WORKER = 8

LORA_RANK = 8
LORA_ALPHA = 16.0
INITIAL_SIGMA = 0.005
MAX_SIGMA = 0.014
MU_FRACTION = 0.5
MAX_TOKENS_GEN = 512  # tokens per request

# >>> NEW: Per-server token budgets (tokens available for generation across concurrent workers).
DEFAULT_MAX_TOKENS_PER_SERVER = 512 * (512)
# You can override specific servers here; any URI not present uses DEFAULT_MAX_TOKENS_PER_SERVER.
MAX_TOKENS_PER_SERVER = {
    uri: DEFAULT_MAX_TOKENS_PER_SERVER for uri in SERVER_URIS
    # Example overrides:
    # "ws://127.0.0.1:8080": 512 * 96,
    # "ws://127.0.0.1:8083": 512 * 32,
}

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
        tqdm.write(f"[Cache] Reusing cached wasm for {src_path.name} (hash {code_hash[:8]})")
        return wasm_path.read_bytes()

    tqdm.write(f"[Compile] No cache for {src_path.name}...")
    program_bytes = await compile_program(str(src_path), deps)
    wasm_path.write_bytes(program_bytes)
    hash_path.write_text(code_hash + "\n")
    tqdm.write(f"[Compile] Saved wasm to {wasm_path.name}")
    return program_bytes


async def launch_and_get_result(
        client: PieClient,
        program_hash: str,
        arguments: list[str],
        worker_id: Any = 0,
        verbose: bool = False,  # <<< added to quiet worker spam
) -> Optional[str]:
    """Launches an inferlet and waits for it to terminate, returning the final message."""
    if verbose:
        tqdm.write(f"🚀 Worker {worker_id}: Launching instance with hash {program_hash[:8]}...")
    instance = await client.launch_instance(program_hash, arguments=arguments)
    final_payload = None
    while True:
        event, message = await instance.recv()
        if event == "terminated":
            if verbose:
                tqdm.write(f"✅ Worker {worker_id}: Instance {instance.instance_id} finished. Reason: {message}")
            break
        elif event == "message":
            final_payload = message
        else:
            if verbose:
                tqdm.write(f"  [Worker {worker_id} / {instance.instance_id} / {event}] {message}")
    return final_payload


# ==============================================================================
# Adaptive Rollout Scheduling (Cross-Client)
# ==============================================================================

def _compute_client_capacity(server_uri: str) -> int:
    """Max concurrent workers for a client based on its token budget."""
    budget = MAX_TOKENS_PER_SERVER.get(server_uri, DEFAULT_MAX_TOKENS_PER_SERVER)
    denom = MAX_TOKENS_GEN * NUM_ROLLOUTS_PER_WORKER
    cap = max(1, budget // max(1, denom))
    return int(cap)


async def _rollout_batch_with_retry(
        client: PieClient,
        program_hash: str,
        adapter_name: str,
        seeds_slice: np.ndarray,
        tasks_slice: np.ndarray,
        who: str,
        max_retries: int = 1,
):
    """Run one inferlet on a batch; retry once on failure/mismatch. Returns list[str] or None."""
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        worker_tasks_json = json.dumps([item["prompt"] for item in tasks_slice])
        rollout_args = [
            "--name", adapter_name,
            "--seeds", ",".join(map(str, seeds_slice)),
            "--tasks-json", worker_tasks_json,
            "--max-num-outputs", str(MAX_TOKENS_GEN),
        ]
        res_json = await launch_and_get_result(
            client, program_hash, rollout_args, worker_id=f"{who}-try{attempt}", verbose=VERBOSE_WORKER_LOGS
        )
        if res_json:
            try:
                texts = json.loads(res_json)
            except Exception:
                texts = None
            if isinstance(texts, list) and len(texts) == len(seeds_slice):
                return texts  # success
        # else: retry
    # After retries, give up silently (no warnings per instruction)
    return None


async def _pop_batch_from_queue(
        q: deque,
        q_lock: asyncio.Lock,
        batch_size: int,
):
    """Atomically pop up to batch_size items from the global queue."""
    async with q_lock:
        if not q:
            return None, None
        seeds_list = []
        tasks_list = []
        for _ in range(min(batch_size, len(q))):
            s, t = q.popleft()
            seeds_list.append(int(s))
            tasks_list.append(t)
    # Return numpy arrays to match downstream expectations
    return np.array(seeds_list, dtype=np.int64), np.array(tasks_list, dtype=object)


async def _process_client_rollouts_cross_client(
        client: PieClient,
        server_uri: str,
        program_hash: str,
        adapter_name: str,
        global_queue: deque,
        queue_lock: asyncio.Lock,
        client_idx: int,
        pbar: Optional[tqdm] = None,  # <<< added shared progress bar handle
):
    """Adaptive cross-client scheduler using a shared global queue."""
    capacity = _compute_client_capacity(server_uri)
    tqdm.write(f"Client {client_idx} @ {server_uri}: capacity={capacity}, batch_size={NUM_ROLLOUTS_PER_WORKER}")

    generated_texts: list[str] = []
    generated_seeds: list[int] = []
    generated_tasks: list[dict] = []

    running_tasks = set()
    task_meta = {}
    issued = 0

    # Prime initial slots
    while len(running_tasks) < capacity:
        seeds_slice, tasks_slice = await _pop_batch_from_queue(global_queue, queue_lock, NUM_ROLLOUTS_PER_WORKER)
        if seeds_slice is None:  # queue empty
            break
        who = f"C{client_idx}-B{issued}"
        issued += 1
        t = asyncio.create_task(_rollout_batch_with_retry(
            client, program_hash, adapter_name, seeds_slice, tasks_slice, who
        ))
        running_tasks.add(t)
        task_meta[t] = (seeds_slice, tasks_slice, who)

    # Keep pipeline full while there is work or in-flight tasks
    while running_tasks:
        done, pending = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            seeds_slice, tasks_slice, who = task_meta.pop(t)
            texts = t.result()
            if texts is not None:
                generated_texts.extend(texts)
                generated_seeds.extend(list(map(int, seeds_slice)))
                generated_tasks.extend(list(tasks_slice))

            # >>> NEW: update global progress bar by batch size (attempted work)
            if pbar is not None:
                try:
                    pbar.update(len(seeds_slice))
                except Exception:
                    pass

            # Refill this slot from the global queue
            seeds_slice2, tasks_slice2 = await _pop_batch_from_queue(global_queue, queue_lock, NUM_ROLLOUTS_PER_WORKER)
            if seeds_slice2 is not None:
                who2 = f"C{client_idx}-B{issued}"
                issued += 1
                t2 = asyncio.create_task(_rollout_batch_with_retry(
                    client, program_hash, adapter_name, seeds_slice2, tasks_slice2, who2
                ))
                pending.add(t2)
                task_meta[t2] = (seeds_slice2, tasks_slice2, who2)
        running_tasks = set(pending)

    return generated_texts, generated_seeds, generated_tasks, capacity, server_uri


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
            "num_rollouts_per_worker": NUM_ROLLOUTS_PER_WORKER,
            "default_max_tokens_per_server": DEFAULT_MAX_TOKENS_PER_SERVER,
            "max_tokens_per_server": MAX_TOKENS_PER_SERVER,
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
            tqdm.write(f"✅ Connected to {num_clients} Pie server(s).")
            wandb.config.update({"num_clients": num_clients}, allow_val_change=True)

            tqdm.write("\nCompiling and uploading inferlets to all clients...")
            program_hashes = {}

            for src_filename in INFERLET_SRC_FILES:
                src_path = SCRIPT_DIR / src_filename
                program_bytes = await get_wasm_from_cache_or_compile(src_path, INFERLET_DEPS, SCRIPT_DIR)
                program_hash = blake3(program_bytes).hexdigest()

                upload_tasks = []
                for client in clients:
                    if not await client.program_exists(program_hash):
                        tqdm.write(f"Uploading program {src_filename} ({program_hash[:8]})...")
                        upload_tasks.append(client.upload_program(program_bytes))
                await asyncio.gather(*upload_tasks)
                if upload_tasks:
                    tqdm.write(f"Program {src_filename} uploaded to all necessary clients.")

                program_hashes[src_path.stem] = program_hash

            # --- 2. Initialize the Adapter (Distributed) ---
            tqdm.write("\n" + "=" * 50)
            tqdm.write("Step 0: Initializing ES Adapter on all clients")
            tqdm.write("=" * 50)
            init_args = [
                "--name", ADAPTER_NAME,
                "--rank", str(LORA_RANK),
                "--alpha", str(LORA_ALPHA),
                "--population-size", str(POPULATION_SIZE),
                "--mu-fraction", str(MU_FRACTION),
                "--initial-sigma", str(INITIAL_SIGMA),
            ]
            init_tasks = [
                launch_and_get_result(
                    client, program_hashes["es-init"], init_args, worker_id=f"C{i}-Init", verbose=VERBOSE_WORKER_LOGS
                )
                for i, client in enumerate(clients)
            ]
            await asyncio.gather(*init_tasks)
            tqdm.write("✅ Adapter initialized on all clients.")

            # --- 3. Load Dataset ---
            dataset = CountdownTasksDataset(DATA_PATH, "train", 100)
            wandb.config.update({"dataset_size_train_view": len(dataset)}, allow_val_change=True)

            # --- 4. Start Training Loop ---
            tqdm.write("\n" + "=" * 50)
            tqdm.write(f"Starting ES Training Loop with {num_clients} clients; adaptive cross-client concurrency.")
            tqdm.write("=" * 50)

            for step in range(1, TRAINING_STEPS + 1):
                start_time = time.time()
                tqdm.write(f"\n--- Step {step}/{TRAINING_STEPS} ---")

                # --- 4a. Rollout Phase (Distributed) ---
                tqdm.write(f"Phase: Rollout across {num_clients} clients (adaptive, shared queue)")

                # Draw base seeds, repeat per task; sample tasks per seed
                base_seeds = np.random.randint(-2 ** 63, 2 ** 63 - 1, size=POPULATION_SIZE, dtype=np.int64)
                num_total_tasks = POPULATION_SIZE * TASKS_PER_SEED

                task_idx_matrix = np.random.choice(len(dataset), size=(POPULATION_SIZE, TASKS_PER_SEED), replace=True)
                all_tasks = [dataset[i] for i in task_idx_matrix.reshape(-1)]
                repeated_seeds = np.repeat(base_seeds, TASKS_PER_SEED)  # len == num_total_tasks

                # --- NEW: Cross-client shared queue of (seed, task) pairs ---
                global_queue = deque((int(s), t) for s, t in zip(repeated_seeds.tolist(), all_tasks))
                queue_lock = asyncio.Lock()

                # Launch one manager per client; each pulls from the shared queue
                generated_texts: list[str] = []
                generated_seeds: list[int] = []
                generated_tasks: list[dict] = []
                cap_map = {}

                client_manager_tasks = []
                # >>> NEW: global per-step progress bar
                with tqdm(
                        total=num_total_tasks,
                        desc=f"Step {step} rollout",
                        dynamic_ncols=True,
                        leave=False,
                ) as pbar:
                    for client_idx, (client, server_uri) in enumerate(zip(clients, SERVER_URIS)):
                        t = asyncio.create_task(_process_client_rollouts_cross_client(
                            client=client,
                            server_uri=server_uri,
                            program_hash=program_hashes["es-rollout"],
                            adapter_name=ADAPTER_NAME,
                            global_queue=global_queue,
                            queue_lock=queue_lock,
                            client_idx=client_idx,
                            pbar=pbar,  # pass shared bar
                        ))
                        client_manager_tasks.append(t)

                    # Collect managers as they complete (no gather in rollout)
                    async for completed in _as_completed_iter(client_manager_tasks):
                        texts_i, seeds_i, tasks_i, cap_i, uri_i = await completed
                        generated_texts.extend(texts_i)
                        generated_seeds.extend(seeds_i)
                        generated_tasks.extend(tasks_i)
                        cap_map[uri_i] = cap_i

                # --- Output length stats ---
                out_lens_chars = [len(t) for t in generated_texts]
                out_lens_ws_tokens = [len(t.split()) for t in generated_texts]
                avg_len_chars = float(np.mean(out_lens_chars)) if out_lens_chars else 0.0
                avg_len_ws_tokens = float(np.mean(out_lens_ws_tokens)) if out_lens_ws_tokens else 0.0

                # --- 4b. Scoring Phase ---
                tqdm.write("Phase: Scoring")
                reward_infos = [
                    reward_function(text, generated_tasks[i]["nums"], generated_tasks[i]["target"], end_token="<|eot_id|>")
                    for i, text in enumerate(generated_texts)
                ]
                scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
                answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]
                format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]

                # --- 4c. Aggregation Phase ---
                tqdm.write("Phase: Aggregating Scores")
                scores_by_seed: dict[int, list[float]] = defaultdict(list)
                for s, sc in zip(generated_seeds, scores):
                    scores_by_seed[int(s)].append(float(sc))

                # average per base seed; if a seed has no data (after retries), fill with 0.0
                missing_seeds = 0
                aggregated_scores = []
                for s in base_seeds:
                    vals = scores_by_seed.get(int(s))
                    if vals:
                        aggregated_scores.append(float(np.mean(vals)))
                    else:
                        missing_seeds += 1
                        aggregated_scores.append(0.0)

                step_mean = float(np.mean(scores)) if scores else 0.0
                step_std = float(np.std(scores)) if scores else 0.0
                step_max = float(np.max(scores)) if scores else 0.0
                step_min = float(np.min(scores)) if scores else 0.0
                success_rate_train = float(np.mean(answer_rewards)) if answer_rewards else 0.0

                mean_population_score = float(np.mean(aggregated_scores)) if aggregated_scores else 0.0
                mu_k = max(1, int(np.ceil(MU_FRACTION * POPULATION_SIZE)))
                top_mu_mean = float(np.mean(sorted(aggregated_scores, reverse=True)[:mu_k])) if aggregated_scores else 0.0

                # Qualitative HTML sample
                num_to_log = min(5, len(generated_texts))
                if num_to_log > 0:
                    episode_html = "<br><hr><br>".join(
                        f"<pre>{html.escape(generated_texts[idx][:2048])}</pre>"
                        for idx in np.random.choice(len(generated_texts), size=num_to_log, replace=False)
                    )
                else:
                    episode_html = "<pre>No predictions</pre>"

                # --- 4d. Update Phase (Distributed) ---
                tqdm.write("Phase: Update")
                update_args = [
                    "--name", ADAPTER_NAME,
                    "--seeds", ",".join(map(str, base_seeds)),  # one score per base seed in order
                    "--scores", ",".join(map(str, aggregated_scores)),
                    "--max-sigma", str(MAX_SIGMA),
                ]
                update_tasks = [
                    launch_and_get_result(
                        client, program_hashes["es-update"], update_args, worker_id=f"C{i}-Update", verbose=VERBOSE_WORKER_LOGS
                    )
                    for i, client in enumerate(clients)
                ]
                await asyncio.gather(*update_tasks)

                step_duration = time.time() - start_time

                # --- NEW: concise, readable summary per step (doesn't break tqdm layout)
                tqdm.write(
                    "Step {s}: mean={m:.4f} std={sd:.4f} max={mx:.4f} min={mn:.4f} "
                    "succ={sr:.3f} episodes={n} len_tok≈{lt:.1f} dur={d:.1f}s".format(
                        s=step, m=step_mean, sd=step_std, mx=step_max, mn=step_min,
                        sr=success_rate_train, n=len(generated_texts), lt=avg_len_ws_tokens, d=step_duration
                    )
                )

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
                        "es/mean_population_score": mean_population_score,
                        "es/mean_fittest_score": top_mu_mean,
                        "episodes_text": wandb.Html(episode_html),
                        "rollout/missing_seeds": int(missing_seeds),
                        "rollout/max_concurrent_workers_per_client": cap_map,
                    },
                    step=step,
                )

            tqdm.write("\nTraining finished!")
            wandb.summary["final_mean_score"] = step_mean
            wandb.summary["final_max_score"] = step_max
            wandb.summary["final_min_score"] = step_min
            wandb.summary["final_success_rate_train"] = success_rate_train  # W&B CHANGE

    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user.")
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


async def _as_completed_iter(tasks):
    """Helper to iterate over tasks as they complete (avoids gather in rollout)."""
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            yield d


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user.")
