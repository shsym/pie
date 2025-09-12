# main.py
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
    NUM_ROLLOUTS_PER_WORKER: int = 8
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
    EVAL_TASKS_PER_WORKER: int = 16

    # --- Token Budgets ---
    DEFAULT_MAX_TOKENS_PER_SERVER: int = 512 * 100
    MAX_TOKENS_PER_SERVER: Dict[str, int] = field(default_factory=dict)

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
        self.MAX_TOKENS_PER_SERVER = {
            uri: self.DEFAULT_MAX_TOKENS_PER_SERVER for uri in self.SERVER_URIS
        }
        # Update adapter name based on dataset
        self.ADAPTER_NAME = f"evo-{self.DATASET_NAME}-v1"


# ==============================================================================
# 2. Utility and Helper Functions
# ==============================================================================

async def launch_and_get_result(
        client: PieClient,
        program_hash: str,
        arguments: list[str],
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


async def _as_completed_iter(tasks):
    """Helper to iterate over tasks as they complete."""
    pending = set(tasks)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            yield d


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
        """Manages the full distributed rollout and result collection phase."""
        if desc == "evaluation":
            num_tasks = len(dataset)
            all_tasks = [dataset[i] for i in range(num_tasks)]
            seeds_to_run = base_seeds
        else:  # Training
            num_tasks = self.config.POPULATION_SIZE * self.config.TASKS_PER_SEED
            task_indices = np.random.choice(len(dataset), size=(self.config.POPULATION_SIZE, self.config.TASKS_PER_SEED))
            all_tasks = [dataset[i] for i in task_indices.flatten()]
            seeds_to_run = np.repeat(base_seeds, self.config.TASKS_PER_SEED)

        work_queue = deque((int(s), t) for s, t in zip(seeds_to_run, all_tasks))
        queue_lock = asyncio.Lock()

        results = {"texts": [], "seeds": [], "tasks": [], "caps": {}}
        client_tasks = []
        with tqdm(total=num_tasks, desc=f"Step {desc}", dynamic_ncols=True, leave=False) as pbar:
            for i, (client, uri) in enumerate(zip(self.clients, self.config.SERVER_URIS)):
                task = asyncio.create_task(self._client_rollout_worker(
                    client, uri, self.program_hashes["es-rollout"],
                    work_queue, queue_lock, f"C{i}", pbar, batch_size
                ))
                client_tasks.append(task)

            async for completed in _as_completed_iter(client_tasks):
                texts_i, seeds_i, tasks_i, cap_i, uri_i = await completed
                results["texts"].extend(texts_i)
                results["seeds"].extend(seeds_i)
                results["tasks"].extend(tasks_i)
                results["caps"][uri_i] = cap_i
        return results

    # ▼▼▼ MODIFICATION START ▼▼▼
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

    # ▲▲▲ MODIFICATION END ▲▲▲

    async def _run_update_phase(self, base_seeds, aggregated_scores, step: int):
        """Broadcasts the update command to all clients and handles checkpointing."""
        tqdm.write("Phase: Update")
        update_args = [
            "--name", self.config.ADAPTER_NAME,
            "--seeds", ",".join(map(str, base_seeds)),
            "--scores", ",".join(map(str, aggregated_scores)),
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

    # ▲▲▲ MODIFICATION END ▲▲▲

    async def _client_rollout_worker(
            self, client, server_uri, program_hash, work_queue, queue_lock, client_id, pbar, batch_size
    ):
        """The core logic for a single client pulling from the shared work queue."""

        def _compute_capacity():
            budget = self.config.MAX_TOKENS_PER_SERVER.get(server_uri, self.config.DEFAULT_MAX_TOKENS_PER_SERVER)
            denom = self.config.MAX_TOKENS_GEN * batch_size
            return max(1, budget // max(1, denom))

        async def _pop_batch():
            async with queue_lock:
                if not work_queue: return None, None
                batch_limit = min(batch_size, len(work_queue))
                return zip(*[work_queue.popleft() for _ in range(batch_limit)])

        async def _run_batch_with_retry(seeds, tasks, who, max_retries=1):
            for attempt in range(max_retries + 1):
                # Use "problem" key which is consistent across new datasets
                tasks_json = json.dumps([item["problem"] for item in tasks])
                args = ["--name", self.config.ADAPTER_NAME, "--seeds", ",".join(map(str, seeds)),
                        "--tasks-json", tasks_json, "--max-num-outputs", str(self.config.MAX_TOKENS_GEN),
                        "--system-prompt", self.config.SYSTEM_PROMPT]
                res_json = await launch_and_get_result(client, program_hash, args, f"{who}-try{attempt + 1}", self.config.VERBOSE_WORKER_LOGS)
                if res_json:
                    try:
                        texts = json.loads(res_json)
                        if isinstance(texts, list) and len(texts) == len(seeds):
                            return texts
                    except (json.JSONDecodeError, TypeError):
                        pass

            return None

        capacity = _compute_capacity()
        tqdm.write(f"Client {client_id} @ {server_uri}: capacity={capacity}, batch_size={batch_size}")

        texts_out, seeds_out, tasks_out = [], [], []
        running_tasks = set()
        task_meta = {}
        batch_num = 0

        while len(running_tasks) < capacity:
            seeds, tasks = await _pop_batch()
            if not seeds: break

            who = f"{client_id}-B{batch_num}"
            batch_num += 1
            task = asyncio.create_task(_run_batch_with_retry(seeds, tasks, who))
            running_tasks.add(task)
            task_meta[task] = (seeds, tasks)

        while running_tasks:
            done, pending = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                seeds, tasks = task_meta.pop(task)
                texts = task.result()
                if texts:
                    texts_out.extend(texts)
                    seeds_out.extend(seeds)
                    tasks_out.extend(tasks)

                if pbar: pbar.update(len(seeds))

                new_seeds, new_tasks = await _pop_batch()
                if new_seeds:
                    who = f"{client_id}-B{batch_num}"
                    batch_num += 1
                    new_task = asyncio.create_task(_run_batch_with_retry(new_seeds, new_tasks, who))
                    pending.add(new_task)
                    task_meta[new_task] = (new_seeds, new_tasks)

            running_tasks = pending

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
