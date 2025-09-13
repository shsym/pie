import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from contextlib import AsyncExitStack
import html
from collections import defaultdict, deque
from dataclasses import dataclass, field

from blake3 import blake3
import numpy as np
import wandb
from tqdm.auto import tqdm

# Assume pie is an installed library
from pie import PieClient, Instance, Event
# Use the refactored dataset imports
from countdown import CountdownDataset
from openr1math import OpenR1MathDataset


# ==============================================================================
# 1. Configuration Class
# ==============================================================================

@dataclass
class TrainingConfig:
    """Configuration settings for the ES training run."""
    # --- Server and Paths ---
    SERVER_URIS: List[str] = field(default_factory=lambda: ["ws://127.0.0.1:8080"])
    SCRIPT_DIR: Path = Path(__file__).resolve().parent
    WASM_DIR: Path = SCRIPT_DIR / "inferlets" / "target" / "wasm32-wasip2" / "release"

    # --- Dataset Configuration ---
    DATASET_NAME: str = "countdown"  # "countdown" or "math"
    # Path for file-based datasets like Countdown
    DATA_PATH: str = "./Countdown-Tasks-3to4"
    DATASET_TEST_SIZE: int = 256

    # --- Inferlet WASM Paths ---
    INFERLET_WASM_PATHS: Dict[str, Path] = field(default_factory=dict)

    # --- ES Hyperparameters ---
    ADAPTER_NAME: str = "evo-countdown-v1"
    TRAINING_STEPS: int = 10000
    POPULATION_SIZE: int = 512
    TASKS_PER_SEED: int = 4
    NUM_ROLLOUTS_PER_WORKER: int = 1 # This is now the batch size
    LORA_RANK: int = 8
    LORA_ALPHA: float = 16.0
    INITIAL_SIGMA: float = 0.005
    MAX_SIGMA: float = 0.014
    MU_FRACTION: float = 0.5
    MAX_TOKENS_GEN: int = 1024
    SYSTEM_PROMPT: str = (
        "You are a helpful AI Assistant that provides well-reasoned and detailed responses. "
        "You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    )

    # --- Checkpointing Configuration ---
    INITIAL_CHECKPOINT_NAME: Optional[str] = None
    CHECKPOINT_EVERY_N_STEPS: int = 5

    # --- Evaluation Configuration ---
    EVAL_EVERY_N_STEPS: int = 2
    EVAL_TASKS_PER_WORKER: int = 1

    # --- W&B Config ---
    WANDB_PROJECT: str = os.getenv("WANDB_PROJECT", "pie-es-v5")
    WANDB_ENTITY: str = os.getenv("WANDB_ENTITY")
    WANDB_MODE: str = os.getenv("WANDB_MODE")
    WANDB_TAGS: List[str] = field(default_factory=lambda: ["es", "countdown", "lora"])

    # --- Logging ---
    VERBOSE_WORKER_LOGS: bool = True

    def __post_init__(self):
        """Set up dynamic paths and dictionaries after initialization."""
        self.INFERLET_WASM_PATHS = {
            "es-init": self.WASM_DIR / "es_init.wasm",
            "es-rollout": self.WASM_DIR / "es_rollout.wasm",
            "es-update": self.WASM_DIR / "es_update.wasm",
        }

        # Update adapter name based on dataset
        self.ADAPTER_NAME = f"evo-{self.DATASET_NAME}-v1"


# ==============================================================================
# 2. Utility and Helper Functions
# ==============================================================================

async def launch_and_get_result(
        client: PieClient,
        program_hash: str,
        arguments: List[str],
        worker_id: Any = 0,
        verbose: bool = False,
) -> Optional[str]:
    """Launches an inferlet and returns the final message."""
    if verbose:
        tqdm.write(f"🚀 Worker {worker_id}: Launching instance with hash {program_hash[:8]}...")
    instance = await client.launch_instance(program_hash, arguments=arguments)
    final_payload = None
    while True:
        event, message = await instance.recv()
        if event in (Event.Completed, Event.Message):
            final_payload = message
            if event == Event.Completed:
                if verbose:
                    tqdm.write(f"✅ Worker {worker_id}: Instance {instance.instance_id} finished.")
                break
        elif event in (Event.Aborted, Event.Exception, Event.ServerError, Event.OutOfResources):
            if verbose:
                tqdm.write(f"❌ Worker {worker_id} / {instance.instance_id} / {event}] {message}")
            break
    return final_payload


def _create_eval_wandb_html(generations: List[Dict]) -> wandb.Html:
    """Formats evaluation examples as an HTML table for wandb."""
    if not generations:
        return wandb.Html("<pre>No evaluation predictions to log.</pre>")

    table_rows = [
        '<tr><th style="text-align: left;">Prompt</th>'
        '<th style="text-align: left;">Generation</th>'
        '<th>Reward</th><th>Success</th></tr>'
    ]
    style = 'style="border: 1px solid #ddd; padding: 8px;"'
    for item in generations:
        task_content = item['task']
        # Generic prompt handling for different dataset structures
        if 'problem' in task_content:
            prompt_html = f"<pre>{html.escape(task_content['problem'])}</pre>"
        elif 'nums' in task_content and 'target' in task_content:
            prompt_html = f"<pre>nums = {task_content['nums']}\ntarget = {task_content['target']}</pre>"
        else:
            prompt_html = "<pre>N/A</pre>"

        text_html = f"<pre>{html.escape(item['text'])}</pre>"
        success_icon = "✅" if item['answer_reward'] == 1.0 else "❌"
        table_rows.append(
            f'<tr><td {style}>{prompt_html}</td><td {style}>{text_html}</td>'
            f'<td {style} align="center">{item["score"]:.3f}</td>'
            f'<td {style} align="center">{success_icon}</td></tr>'
        )
    return wandb.Html(f'<table style="width:100%; border-collapse: collapse;">{"".join(table_rows)}</table>')


# ==============================================================================
# 3. Core Logic: ESOrchestrator Class
# ==============================================================================

class ESOrchestrator:
    """Manages the state and execution of the ES training process."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.clients = []
        self.program_hashes = {}
        self.train_dataset = None
        self.eval_dataset = None
        self._exit_stack = AsyncExitStack()
        self.wandb_run = None

    async def setup(self):
        """Initialize resources: W&B, clients, WASM programs, and datasets."""
        self._initialize_wandb()

        tqdm.write("🔌 Connecting to Pie servers...")
        self.clients = [
            await self._exit_stack.enter_async_context(PieClient(uri))
            for uri in self.config.SERVER_URIS
        ]
        tqdm.write(f"✅ Connected to {len(self.clients)} Pie server(s).")
        wandb.config.update({"num_clients": len(self.clients)}, allow_val_change=True)

        await self._upload_inferlets()
        await self._initialize_adapter()
        self._load_datasets()

    async def train(self):
        """Runs the main training loop."""
        tqdm.write("\n" + "=" * 50)
        tqdm.write(f"🚀 Starting ES Training Loop with {len(self.clients)} client(s)")
        tqdm.write("=" * 50)

        tqdm.write("\n🔬 Running initial evaluation before training...")
        initial_eval_metrics = await self._run_evaluation(step=0)
        self.wandb_run.log(initial_eval_metrics, step=0)
        tqdm.write("✅ Initial evaluation complete.")

        for step in range(1, self.config.TRAINING_STEPS + 1):
            start_time = time.time()
            tqdm.write(f"\n--- Step {step}/{self.config.TRAINING_STEPS} ---")

            base_seeds = np.random.randint(-2 ** 63, 2 ** 63 - 1, size=self.config.POPULATION_SIZE, dtype=np.int64)
            rollout_results = await self._run_distributed_rollouts(base_seeds, self.train_dataset, self.config.NUM_ROLLOUTS_PER_WORKER)

            scores, metrics = self._score_and_aggregate(base_seeds, rollout_results)

            await self._run_update_phase(base_seeds, scores, step)

            step_duration = time.time() - start_time
            metrics["perf/step_duration_sec"] = step_duration
            metrics["step"] = step

            tqdm.write(
                f"Step {step}: mean_reward={metrics['mean_reward']:.4f} | "
                f"episodes={metrics['num_finished_episodes']} | duration={step_duration:.1f}s"
            )

            if step % self.config.EVAL_EVERY_N_STEPS == 0 or step == self.config.TRAINING_STEPS:
                eval_metrics = await self._run_evaluation(step)
                metrics.update(eval_metrics)

            self.wandb_run.log(metrics, step=step)

        tqdm.write("\n🎉 Training finished!")
        wandb.summary["final_mean_score"] = metrics["mean_reward"]

    async def teardown(self):
        """Clean up resources."""
        await self._exit_stack.aclose()
        if self.wandb_run:
            self.wandb_run.finish()
        tqdm.write("Resources cleaned up.")

    def _initialize_wandb(self):
        """Sets up the Weights & Biases run."""
        self.wandb_run = wandb.init(
            project=self.config.WANDB_PROJECT,
            entity=self.config.WANDB_ENTITY,
            name=f"{self.config.ADAPTER_NAME}-{int(time.time())}",
            tags=self.config.WANDB_TAGS,
            config=self.config.__dict__,
            save_code=True,
            mode=self.config.WANDB_MODE if self.config.WANDB_MODE else None,
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def _load_datasets(self):
        """Loads the training and evaluation datasets based on config."""
        tqdm.write(f"💿 Loading dataset: {self.config.DATASET_NAME}")
        if self.config.DATASET_NAME == "countdown":
            self.train_dataset = CountdownDataset(
                self.config.DATA_PATH, "train", self.config.DATASET_TEST_SIZE
            )
            self.eval_dataset = CountdownDataset(
                self.config.DATA_PATH, "test", self.config.DATASET_TEST_SIZE
            )
        elif self.config.DATASET_NAME == "math":
            self.train_dataset = OpenR1MathDataset(
                "train", self.config.DATASET_TEST_SIZE
            )
            self.eval_dataset = OpenR1MathDataset(
                "test", self.config.DATASET_TEST_SIZE
            )
        else:
            raise ValueError(f"Unknown dataset name: {self.config.DATASET_NAME}")

        wandb.config.update({
            "dataset_size_train_view": len(self.train_dataset),
            "dataset_size_eval": len(self.eval_dataset),
        }, allow_val_change=True)
        tqdm.write("✅ Datasets loaded.")

    async def _upload_inferlets(self):
        """Loads and uploads WASM binaries to all clients."""
        tqdm.write("\n📦 Loading and uploading inferlet WASM binaries...")
        for name, wasm_path in self.config.INFERLET_WASM_PATHS.items():
            if not wasm_path.exists():
                raise FileNotFoundError(f"WASM binary not found: {wasm_path}")

            program_bytes = wasm_path.read_bytes()
            program_hash = blake3(program_bytes).hexdigest()

            upload_tasks = [
                client.upload_program(program_bytes)
                for client in self.clients if not await client.program_exists(program_hash)
            ]
            if upload_tasks:
                tqdm.write(f"Uploading {wasm_path.name} ({program_hash[:8]})...")
                await asyncio.gather(*upload_tasks)

            self.program_hashes[name] = program_hash
        tqdm.write("✅ All inferlets are available on clients.")

    async def _initialize_adapter(self):
        """Initializes the ES adapter on all clients."""
        tqdm.write("\n⚙️  Initializing ES Adapter on all clients...")
        init_args = [
            "--name", self.config.ADAPTER_NAME, "--rank", str(self.config.LORA_RANK),
            "--alpha", str(self.config.LORA_ALPHA), "--population-size", str(self.config.POPULATION_SIZE),
            "--mu-fraction", str(self.config.MU_FRACTION), "--initial-sigma", str(self.config.INITIAL_SIGMA),
        ]

        if self.config.INITIAL_CHECKPOINT_NAME:
            tqdm.write(f"📂 Loading initial checkpoint: {self.config.INITIAL_CHECKPOINT_NAME}")
            init_args.extend(["--upload", self.config.INITIAL_CHECKPOINT_NAME])
        else:
            init_args.extend(["--upload", ""])

        init_tasks = [
            launch_and_get_result(client, self.program_hashes["es-init"], init_args, f"C{i}-Init", self.config.VERBOSE_WORKER_LOGS)
            for i, client in enumerate(self.clients)
        ]
        await asyncio.gather(*init_tasks)
        tqdm.write("✅ Adapter initialized on all clients.")

    async def _run_distributed_rollouts(self, base_seeds, dataset, batch_size, desc="rollout"):
        """
        Manages the full distributed rollout using a continuous, dynamic scheduling strategy.

        This implementation avoids distinct phases. Instead, it uses a central work queue
        and persistent client workers. Each worker processes batches concurrently up to a
        calculated capacity. When a batch finishes, the worker immediately pulls new work
        from the queue. If a batch is preempted (fails), its individual tasks are returned
        to the central queue to be retried by any available worker. This ensures maximum
        server utilization.
        """
        # --- 1. PREPARATION: Create all individual tasks and load them into a shared queue ---
        if desc == "evaluation":
            num_tasks = len(dataset)
            all_tasks = [dataset[i] for i in range(num_tasks)]
            seeds_to_run = base_seeds
        else: # Training
            num_tasks = self.config.POPULATION_SIZE * self.config.TASKS_PER_SEED
            task_indices = np.random.choice(len(dataset), size=(self.config.POPULATION_SIZE, self.config.TASKS_PER_SEED))
            all_tasks = [dataset[i] for i in task_indices.flatten()]
            seeds_to_run = np.repeat(base_seeds, self.config.TASKS_PER_SEED)

        # The work_queue contains individual (seed, task) tuples.
        work_queue = deque(zip(seeds_to_run, all_tasks))

        # Calculate the initial fair share of tasks for each worker.
        # This determines how many concurrent batches each worker will try to launch initially.
        initial_tasks_per_worker = np.ceil(num_tasks / len(self.clients)) if self.clients else 0

        queue_lock = asyncio.Lock()
        results = {"texts": [], "seeds": [], "tasks": [], "caps": {}}

        pbar = tqdm(total=num_tasks, desc=f"Step {desc}", dynamic_ncols=True, leave=False)
        try:
            # --- 2. LAUNCH PERSISTENT WORKERS ---
            # Each client gets one worker that runs until the shared work_queue is empty.
            client_tasks = []
            for i, (client, uri) in enumerate(zip(self.clients, self.config.SERVER_URIS)):
                task = asyncio.create_task(self._client_rollout_worker(
                    client, uri, self.program_hashes["es-rollout"],
                    work_queue, queue_lock, f"C{i}", pbar, batch_size,
                    initial_tasks_per_worker  # Pass the initial task count
                ))
                client_tasks.append(task)

            # --- 3. GATHER RESULTS ---
            # Wait for all workers to complete and collect their results.
            all_worker_results = await asyncio.gather(*client_tasks)
            for texts_i, seeds_i, tasks_i, cap_i, uri_i in all_worker_results:
                results["texts"].extend(texts_i)
                results["seeds"].extend(seeds_i)
                results["tasks"].extend(tasks_i)
                results["caps"][uri_i] = cap_i
        finally:
            pbar.close()

        return results

    def _score_and_aggregate(self, base_seeds, rollout_results):
        """Scores generations and aggregates them by seed using generic verifier."""
        reward_infos = [
            task["verifier"](text)
            for text, task in zip(rollout_results["texts"], rollout_results["tasks"])
        ]
        # Extract all reward components
        scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
        format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]
        answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

        scores_by_seed = defaultdict(list)
        for s, sc in zip(rollout_results["seeds"], scores):
            scores_by_seed[s].append(sc)

        aggregated_scores, missing_seeds = [], 0
        for s in base_seeds:
            vals = scores_by_seed.get(int(s))
            if vals:
                aggregated_scores.append(float(np.mean(vals)))
            else:
                aggregated_scores.append(0.0)
                missing_seeds += 1

        mu_k = max(1, int(np.ceil(self.config.MU_FRACTION * self.config.POPULATION_SIZE)))
        out_lens = [len(t.split()) for t in rollout_results["texts"]]

        metrics = {
            "mean_reward": float(np.mean(scores)) if scores else 0.0,
            "mean_format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "mean_answer_reward": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
            "std_reward": float(np.std(scores)) if scores else 0.0,
            "num_finished_episodes": len(rollout_results["texts"]),
            "mean_response_len": float(np.mean(out_lens)) if out_lens else 0.0,
            "es/mean_population_score": float(np.mean(aggregated_scores)),
            "es/mean_fittest_score": float(np.mean(sorted(aggregated_scores, reverse=True)[:mu_k])),
            "rollout/missing_seeds": missing_seeds,
            "rollout/max_concurrent_workers_per_client": rollout_results["caps"],
        }
        return aggregated_scores, metrics

    async def _run_update_phase(self, base_seeds, aggregated_scores, step: int):
        """Broadcasts the update command to all clients and handles checkpointing."""
        tqdm.write("Phase: Update")

        # Round scores to avoid overly long argument strings from float precision.
        scores_str = ",".join(f"{s:.6f}" for s in aggregated_scores)

        update_args = [
            "--name", self.config.ADAPTER_NAME,
            "--scores", scores_str,
            "--max-sigma", str(self.config.MAX_SIGMA),
        ]

        if step > 0 and step % self.config.CHECKPOINT_EVERY_N_STEPS == 0:
            checkpoint_name = f"{self.config.ADAPTER_NAME}-step-{step}"
            tqdm.write(f"💾 Saving checkpoint: {checkpoint_name}")
            update_args.extend(["--download", checkpoint_name])

        update_tasks = [
            launch_and_get_result(client, self.program_hashes["es-update"], update_args, f"C{i}-Update", self.config.VERBOSE_WORKER_LOGS)
            for i, client in enumerate(self.clients)
        ]
        await asyncio.gather(*update_tasks)

    async def _run_evaluation(self, step: int) -> Dict[str, Any]:
        """Runs evaluation on the central model parameters (seed=0)."""
        tqdm.write("\n" + "-" * 20 + f" Running Evaluation @ Step {step} " + "-" * 20)
        eval_start_time = time.time()

        num_eval_tasks = len(self.eval_dataset)
        eval_seeds = np.zeros(num_eval_tasks, dtype=np.int64)

        results = await self._run_distributed_rollouts(
            eval_seeds, self.eval_dataset, self.config.EVAL_TASKS_PER_WORKER, desc="evaluation"
        )

        reward_infos = [
            task["verifier"](text)
            for text, task in zip(results["texts"], results["tasks"])
        ]
        # Extract all reward components
        scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
        format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]
        answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

        tqdm.write(f"✅ Eval Complete: mean_reward={np.mean(scores):.4f}")

        num_to_log = min(10, len(results["texts"]))
        indices = np.random.choice(len(results["texts"]), size=num_to_log, replace=False)
        examples = [{
            "task": results["tasks"][i], "text": results["texts"][i],
            "score": scores[i], "answer_reward": answer_rewards[i]
        } for i in indices]

        metrics = {
            "eval/mean_reward": np.mean(scores) if scores else 0.0,
            "eval/mean_format_reward": np.mean(format_rewards) if format_rewards else 0.0,
            "eval/mean_answer_reward": np.mean(answer_rewards) if answer_rewards else 0.0,
            "eval/duration_seconds": time.time() - eval_start_time,
            "eval/examples": _create_eval_wandb_html(examples),
        }
        if step == self.config.TRAINING_STEPS:
            wandb.summary["final_eval_mean_reward"] = metrics["eval/mean_reward"]
        return metrics

    async def _run_batch(self, client, program_hash, seeds, tasks, who):
        """Helper to run a single batch."""
        rollouts = []
        for seed, task in zip(seeds, tasks):
            hasher = blake3(str(seed).encode('utf-8'))
            task_problem_str = task.get('problem', str(task))
            hasher.update(task_problem_str.encode('utf-8'))
            uid = hasher.hexdigest()
            rollouts.append({
                "uid": uid,
                "task": task_problem_str,
                "seed": int(seed)
            })
        rollouts_json = json.dumps(rollouts)

        args = [
            "--name", self.config.ADAPTER_NAME,
            "--rollouts", rollouts_json,
            "--max-num-outputs", str(self.config.MAX_TOKENS_GEN),
            "--system-prompt", self.config.SYSTEM_PROMPT,
        ]
        res_json = await launch_and_get_result(client, program_hash, args, who, self.config.VERBOSE_WORKER_LOGS)
        if res_json:
            try:
                texts = json.loads(res_json)
                if isinstance(texts, list) and len(texts) == len(seeds):
                    return texts
            except (json.JSONDecodeError, TypeError):
                tqdm.write(f"Warning: JSON decode failed for worker {who}.")
                pass
        # Return None to signal preemption/failure.
        return None

    async def _client_rollout_worker(
            self, client, server_uri, program_hash, work_queue, queue_lock, client_id, pbar, batch_size,
            initial_tasks_per_worker
    ):
        """
        The core logic for a single client worker. It continuously pulls work from the
        shared queue, runs batches concurrently up to its capacity, and returns
        results when all work is done.
        """
        # The worker's concurrency level ("capacity") is determined by its initial
        # fair share of the total workload. It will attempt to launch this many batches
        # at the start, creating the desired "overload" phase dynamically.
        capacity = int(np.ceil(initial_tasks_per_worker / batch_size))

        tqdm.write(f"Client {client_id} @ {server_uri}: capacity set to {capacity} batches based on initial fair share.")

        texts_out, seeds_out, tasks_out = [], [], []
        running_tasks = set()
        task_meta = {}
        batch_num = 0

        # Initially populate the worker's running tasks up to its capacity.
        async with queue_lock:
            for _ in range(capacity):
                if not work_queue: break
                batch_limit = min(batch_size, len(work_queue))
                new_work = [work_queue.popleft() for _ in range(batch_limit)]
                if new_work:
                    seeds, tasks = zip(*new_work)
                    who = f"{client_id}-B{batch_num}"
                    batch_num += 1
                    task = asyncio.create_task(self._run_batch(client, program_hash, list(seeds), list(tasks), who))
                    running_tasks.add(task)
                    task_meta[task] = (list(seeds), list(tasks))

        # Main loop: As tasks complete, process results and pull new work to maintain capacity.
        while running_tasks:
            done, pending = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                # This entire block is protected by a lock to prevent race conditions
                # on the shared `work_queue`.
                async with queue_lock:
                    seeds, tasks = task_meta.pop(task)
                    texts = task.result()

                    if texts:  # Batch succeeded
                        texts_out.extend(texts)
                        seeds_out.extend(seeds)
                        tasks_out.extend(tasks)
                        pbar.update(len(seeds))
                    else:  # Batch failed, add individual tasks back to the shared work_queue
                        tqdm.write(f"Worker {client_id}: Batch failed. Re-queuing {len(seeds)} tasks.")
                        for seed, single_task in zip(seeds, tasks):
                            work_queue.append((seed, single_task))

                    # Try to pull a new batch from the queue to replace the one that just finished.
                    if work_queue:
                        batch_limit = min(batch_size, len(work_queue))
                        new_work = [work_queue.popleft() for _ in range(batch_limit)]

                        if new_work:
                            new_seeds, new_tasks = zip(*new_work)
                            who = f"{client_id}-B{batch_num}"
                            batch_num += 1
                            new_task = asyncio.create_task(self._run_batch(client, program_hash, list(new_seeds), list(new_tasks), who))
                            # Add new task to the 'pending' set to be awaited in the next cycle.
                            pending.add(new_task)
                            task_meta[new_task] = (list(new_seeds), list(new_tasks))

            # ensuring all completed tasks in 'done' are processed before updating the set of running tasks.
            running_tasks = pending

        # When the loop finishes, all work is done for this worker.
        return texts_out, seeds_out, tasks_out, capacity, server_uri

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

async def main():
    """High-level entry point to configure and run the training orchestrator."""
    config = TrainingConfig()
    # Example: Switch to the math dataset
    # config.DATASET_NAME = "math"

    # Example: Resume from a checkpoint
    # config.INITIAL_CHECKPOINT_NAME = "evo-countdown-v1-step-10"
    orchestrator = ESOrchestrator(config)
    try:
        await orchestrator.setup()
        await orchestrator.train()
    except Exception as e:
        tqdm.write(f"\nAn unexpected error occurred: {e}")
        raise
    finally:
        await orchestrator.teardown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted by user.")