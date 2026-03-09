"""Simplified ES Training Script.

Takes advantage of the runtime's contention management — we simply
submit all requests upfront and let the runtime handle queuing,
suspension, and restoration internally.  No per-worker scheduling,
capacity tracking, or retry logic needed.

Supports two modes:
  - **Local** (``--local``): Spins up an in-process Pie server.
    Inferlets are fetched from the registry.
  - **Distributed** (default): Connects to remote server(s) via
    ``--servers`` URIs.  Inferlets are fetched from the registry.

Usage::

    # Local mode (single server, in-process)
    uv run python train.py --local --model meta-llama/Llama-3.2-1B-Instruct

    # Distributed mode (one or more remote servers)
    uv run python train.py --servers ws://gpu1:8080 ws://gpu2:8080

    # Override ES hyperparameters
    uv run python train.py --local --population-size 256 --tasks-per-seed 8
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from blake3 import blake3
from tqdm.auto import tqdm

from pie_client import PieClient, Event

from countdown import CountdownDataset
from openr1math import OpenR1MathDataset


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for the ES training run."""

    # --- Mode ---
    local: bool = False
    model: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda:0"
    gpu_mem_util: float = 0.8
    cpu_mem_budget: int = 0

    # --- Server (distributed mode) ---
    servers: list[str] = field(default_factory=lambda: ["ws://127.0.0.1:8080"])

    # --- Inferlet names (from registry) ---
    inferlet_names: dict[str, str] = field(
        default_factory=lambda: {
            "es-init": "es-init@0.1.5",
            "es-rollout": "es-rollout@0.1.5",
            "es-update": "es-update@0.1.5",
        }
    )

    # --- Dataset ---
    dataset: str = "math"  # "countdown" or "math"
    data_path: str = "./Countdown-Tasks-3to4"

    # --- ES Hyperparameters ---
    adapter_name: str = ""
    training_steps: int = 10000
    population_size: int = 512
    tasks_per_seed: int = 4
    lora_rank: int = 8
    lora_alpha: float = 16.0
    initial_sigma: float = 0.005
    max_sigma: float = 0.014
    mu_fraction: float = 0.5
    max_tokens_gen: int = 1024
    system_prompt: str = (
        "You are a helpful AI Assistant that provides well-reasoned and "
        "detailed responses. You first think about the reasoning process as "
        "an internal monologue and then provide the user with the answer. "
        "Respond in the following format: <think>\n...\n</think>\n"
        "<answer>\n...\n</answer>"
    )

    # --- Evaluation ---
    test_size: int = 100
    eval_every: int = 2

    # --- Checkpointing ---
    checkpoint_every: int = 5

    def __post_init__(self):
        if not self.adapter_name:
            self.adapter_name = f"evo-{self.dataset}-v1"


# =============================================================================
# Helpers
# =============================================================================


async def launch_and_collect(
    client: PieClient,
    inferlet: str,
    input_dict: dict,
) -> Optional[str]:
    """Launch an inferlet and block until it returns or errors."""
    process = await client.launch_process(inferlet, input=input_dict)
    result = None
    while True:
        event, message = await process.recv()
        if event in (Event.Return, Event.Message):
            result = message
            if event == Event.Return:
                break
        elif event == Event.Error:
            tqdm.write(f"⚠️  Process {process.process_id} failed: {message}")
            break
    return result


async def run_on_all_clients(
    clients: list[PieClient],
    inferlet: str,
    input_dict: dict,
) -> list[Optional[str]]:
    """Run the same inferlet call on every client in parallel."""
    return await asyncio.gather(
        *(launch_and_collect(c, inferlet, input_dict) for c in clients)
    )


# =============================================================================
# Rollout Distribution
# =============================================================================


def build_rollout_batches(
    seeds: np.ndarray,
    tasks: list[dict],
    adapter_name: str,
    max_tokens: int,
    system_prompt: str,
) -> list[dict]:
    """Build one rollout input dict per (seed, task) pair."""
    inputs = []
    for seed, task in zip(seeds, tasks):
        hasher = blake3(str(seed).encode())
        problem = task.get("problem", str(task))
        hasher.update(problem.encode())
        uid = hasher.hexdigest()

        inputs.append({
            "name": adapter_name,
            "rollouts": [{"uid": uid, "task": problem, "seed": int(seed)}],
            "max_num_outputs": max_tokens,
            "system_prompt": system_prompt,
        })
    return inputs


async def run_rollouts(
    clients: list[PieClient],
    inferlet: str,
    seeds: np.ndarray,
    tasks: list[dict],
    config: TrainingConfig,
    desc: str = "rollout",
) -> dict:
    """Submit all rollout requests across clients, wait for all results.

    Work is partitioned evenly across the clients.  Each client
    receives all of its requests upfront — the runtime's contention
    management handles queuing and scheduling internally.
    """
    inputs = build_rollout_batches(
        seeds, tasks, config.adapter_name,
        config.max_tokens_gen, config.system_prompt,
    )

    # Round-robin partition across clients
    per_client: list[list[tuple[int, dict, np.int64, dict]]] = [
        [] for _ in clients
    ]
    for i, (inp, seed, task) in enumerate(zip(inputs, seeds, tasks)):
        per_client[i % len(clients)].append((i, inp, seed, task))

    texts_out, seeds_out, tasks_out = [], [], []
    pbar = tqdm(total=len(inputs), desc=desc, dynamic_ncols=True, leave=False)

    async def client_worker(client, work_items):
        """Submit all items for this client and collect results."""
        coros = []
        metadata = []
        for _, inp, seed, task in work_items:
            coros.append(launch_and_collect(client, inferlet, inp))
            metadata.append((seed, task))

        results = await asyncio.gather(*coros)
        collected_texts, collected_seeds, collected_tasks = [], [], []
        for (seed, task), res_json in zip(metadata, results):
            if res_json:
                try:
                    texts = json.loads(res_json)
                    if isinstance(texts, list):
                        collected_texts.extend(texts)
                        collected_seeds.extend([seed] * len(texts))
                        collected_tasks.extend([task] * len(texts))
                        pbar.update(len(texts))
                except (json.JSONDecodeError, TypeError):
                    pbar.update(1)
            else:
                pbar.update(1)
        return collected_texts, collected_seeds, collected_tasks

    worker_results = await asyncio.gather(
        *(client_worker(c, items) for c, items in zip(clients, per_client))
    )
    pbar.close()

    for texts, seeds_list, tasks_list in worker_results:
        texts_out.extend(texts)
        seeds_out.extend(seeds_list)
        tasks_out.extend(tasks_list)

    return {"texts": texts_out, "seeds": seeds_out, "tasks": tasks_out}


# =============================================================================
# ES Trainer
# =============================================================================


class ESTrainer:
    """Manages ES training: adapter init, rollouts, scoring, updates."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.clients: list[PieClient] = []
        self.train_dataset = None
        self.eval_dataset = None
        self._exit_stack = AsyncExitStack()
        self._server = None  # Only used in local mode

    # -- Lifecycle ------------------------------------------------------------

    async def setup(self):
        """Connect to server(s), initialize adapter, load datasets."""
        if self.config.local:
            await self._setup_local()
        else:
            await self._setup_distributed()

        await self._init_adapter()
        self._load_datasets()

    async def teardown(self):
        """Clean up all resources."""
        await self._exit_stack.aclose()
        tqdm.write("Resources cleaned up.")

    async def _setup_local(self):
        """Spin up an in-process Pie server and connect."""
        from pie.server import Server
        from pie.config import Config, ModelConfig, AuthConfig

        device = (
            [d.strip() for d in self.config.device.split(",")]
            if "," in self.config.device
            else [self.config.device]
        )

        cfg = Config(
            port=0,
            auth=AuthConfig(enabled=False),
            models=[ModelConfig(
                hf_repo=self.config.model,
                device=device,
                gpu_mem_utilization=self.config.gpu_mem_util,
                cpu_mem_budget_in_gb=self.config.cpu_mem_budget,
            )],
        )

        tqdm.write(f"🚀 Starting local Pie server (model={self.config.model}, device={device})...")
        server = await self._exit_stack.enter_async_context(Server(cfg))
        self._server = server
        client = await server.connect()
        self.clients = [client]
        tqdm.write("✅ Local server ready.")

    async def _setup_distributed(self):
        """Connect to remote Pie server(s)."""
        tqdm.write("🔌 Connecting to Pie servers...")
        for uri in self.config.servers:
            client = await self._exit_stack.enter_async_context(PieClient(uri))
            await client.authenticate("main-user")
            self.clients.append(client)
        tqdm.write(f"✅ Connected to {len(self.clients)} server(s).")

    def _load_datasets(self):
        """Load training and evaluation datasets."""
        tqdm.write(f"💿 Loading dataset: {self.config.dataset}")
        if self.config.dataset == "countdown":
            self.train_dataset = CountdownDataset(
                self.config.data_path, "train", self.config.test_size
            )
            self.eval_dataset = CountdownDataset(
                self.config.data_path, "test", self.config.test_size
            )
        elif self.config.dataset == "math":
            self.train_dataset = OpenR1MathDataset("train", self.config.test_size)
            self.eval_dataset = OpenR1MathDataset("test", self.config.test_size)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        tqdm.write("✅ Datasets loaded.")

    async def _init_adapter(self):
        """Initialize ES adapter on all clients."""
        tqdm.write("⚙️  Initializing adapter on all clients...")
        init_input = {
            "name": self.config.adapter_name,
            "rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "population_size": self.config.population_size,
            "mu_fraction": self.config.mu_fraction,
            "initial_sigma": self.config.initial_sigma,
            "upload": "",
        }
        results = await run_on_all_clients(
            self.clients, self.config.inferlet_names["es-init"], init_input,
        )
        if any(r is None for r in results):
            failed = [i for i, r in enumerate(results) if r is None]
            raise RuntimeError(f"Adapter init failed on client(s): {failed}")
        tqdm.write("✅ Adapter initialized.")

    # -- Training loop --------------------------------------------------------

    async def train(self):
        """Main training loop."""
        tqdm.write(f"\n{'='*50}")
        tqdm.write(f"🚀 Starting ES Training ({len(self.clients)} client(s))")
        tqdm.write(f"{'='*50}")

        consecutive_failures = 0
        MAX_FAILURES = 3

        for step in range(1, self.config.training_steps + 1):
            t0 = time.time()
            tqdm.write(f"\n--- Step {step}/{self.config.training_steps} ---")

            # Generate seeds and sample tasks
            base_seeds = np.random.randint(
                -(2**63), 2**63 - 1,
                size=self.config.population_size, dtype=np.int64,
            )
            seeds = np.repeat(base_seeds, self.config.tasks_per_seed)
            task_indices = np.random.choice(
                len(self.train_dataset),
                size=self.config.population_size * self.config.tasks_per_seed,
            )
            tasks = [self.train_dataset[i] for i in task_indices]

            # Rollout phase
            results = await run_rollouts(
                self.clients,
                self.config.inferlet_names["es-rollout"],
                seeds, tasks, self.config,
                desc=f"Step {step}",
            )

            # Score and aggregate
            scores, metrics = self._score(base_seeds, results)

            # Update phase
            await self._update(base_seeds, scores, step)

            dt = time.time() - t0
            metrics["perf/step_duration_sec"] = dt
            n = metrics["num_finished_episodes"]
            tqdm.write(
                f"Step {step}: reward={metrics['mean_reward']:.4f} | "
                f"episodes={n} | {dt:.1f}s"
            )

            if n == 0:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    tqdm.write(f"\n❌ Aborting: {MAX_FAILURES} steps with 0 episodes.")
                    return
            else:
                consecutive_failures = 0

            # Evaluation
            if step % self.config.eval_every == 0 or step == self.config.training_steps:
                await self._evaluate(step)

        tqdm.write("\n🎉 Training finished!")

    # -- Scoring --------------------------------------------------------------

    def _score(self, base_seeds, rollout_results) -> tuple[list[float], dict]:
        """Score generations and aggregate by seed."""
        reward_infos = [
            task["verifier"](text)
            for text, task in zip(rollout_results["texts"], rollout_results["tasks"])
        ]
        scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
        format_rewards = [float(ri.get("format_reward", 0.0)) for ri in reward_infos]
        answer_rewards = [float(ri.get("answer_reward", 0.0)) for ri in reward_infos]

        by_seed = defaultdict(list)
        for s, sc in zip(rollout_results["seeds"], scores):
            by_seed[int(s)].append(sc)

        aggregated = []
        missing = 0
        for s in base_seeds:
            vals = by_seed.get(int(s))
            if vals:
                aggregated.append(float(np.mean(vals)))
            else:
                aggregated.append(0.0)
                missing += 1

        mu_k = max(1, int(np.ceil(self.config.mu_fraction * self.config.population_size)))
        out_lens = [len(t.split()) for t in rollout_results["texts"]]

        metrics = {
            "mean_reward": float(np.mean(scores)) if scores else 0.0,
            "mean_format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "mean_answer_reward": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
            "std_reward": float(np.std(scores)) if scores else 0.0,
            "num_finished_episodes": len(rollout_results["texts"]),
            "mean_response_len": float(np.mean(out_lens)) if out_lens else 0.0,
            "es/mean_population_score": float(np.mean(aggregated)),
            "es/mean_fittest_score": float(np.mean(sorted(aggregated, reverse=True)[:mu_k])),
            "rollout/missing_seeds": missing,
        }
        return aggregated, metrics

    # -- Update ---------------------------------------------------------------

    async def _update(self, base_seeds, scores, step: int):
        """Broadcast update to all clients."""
        tqdm.write("Phase: Update")
        update_input = {
            "name": self.config.adapter_name,
            "seeds": [int(s) for s in base_seeds],
            "scores": [float(s) for s in scores],
            "max_sigma": self.config.max_sigma,
        }
        if step > 0 and step % self.config.checkpoint_every == 0:
            ckpt = f"{self.config.adapter_name}-step-{step}"
            tqdm.write(f"💾 Checkpoint: {ckpt}")
            update_input["download"] = ckpt

        await run_on_all_clients(
            self.clients, self.config.inferlet_names["es-update"], update_input,
        )

    # -- Evaluation -----------------------------------------------------------

    async def _evaluate(self, step: int):
        """Run evaluation on the central model (seed=0)."""
        tqdm.write(f"\n{'─'*20} Eval @ Step {step} {'─'*20}")
        t0 = time.time()

        num_eval = len(self.eval_dataset)
        eval_seeds = np.zeros(num_eval, dtype=np.int64)
        eval_tasks = [self.eval_dataset[i] for i in range(num_eval)]

        results = await run_rollouts(
            self.clients,
            self.config.inferlet_names["es-rollout"],
            eval_seeds, eval_tasks, self.config,
            desc="eval",
        )

        reward_infos = [
            task["verifier"](text)
            for text, task in zip(results["texts"], results["tasks"])
        ]
        scores = [float(ri.get("reward", 0.0)) for ri in reward_infos]
        tqdm.write(f"✅ Eval: mean_reward={np.mean(scores):.4f} ({time.time()-t0:.1f}s)")


# =============================================================================
# CLI & Main
# =============================================================================


def parse_args() -> TrainingConfig:
    p = argparse.ArgumentParser(description="Simplified ES Training on Pie")

    # Mode
    p.add_argument("--local", action="store_true", help="Run with in-process Pie server")
    p.add_argument("--servers", nargs="+", default=["ws://127.0.0.1:8080"],
                   help="Remote server URIs (distributed mode)")

    # Model / device (local mode)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B", help="HuggingFace model ID")
    p.add_argument("--device", default="cuda:0", help="Device(s), comma-separated")
    p.add_argument("--gpu-mem-util", type=float, default=0.8)
    p.add_argument("--cpu-mem-budget", type=int, default=0)

    # Dataset
    p.add_argument("--dataset", default="math", choices=["math", "countdown"])
    p.add_argument("--data-path", default="./Countdown-Tasks-3to4")

    # ES hyperparameters
    p.add_argument("--population-size", type=int, default=512)
    p.add_argument("--tasks-per-seed", type=int, default=4)
    p.add_argument("--training-steps", type=int, default=10000)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--initial-sigma", type=float, default=0.005)
    p.add_argument("--max-sigma", type=float, default=0.014)
    p.add_argument("--mu-fraction", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=1024)

    # Evaluation
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=5)

    args = p.parse_args()

    return TrainingConfig(
        local=args.local,
        model=args.model,
        device=args.device,
        gpu_mem_util=args.gpu_mem_util,
        cpu_mem_budget=args.cpu_mem_budget,
        servers=args.servers,
        dataset=args.dataset,
        data_path=args.data_path,
        population_size=args.population_size,
        tasks_per_seed=args.tasks_per_seed,
        training_steps=args.training_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        initial_sigma=args.initial_sigma,
        max_sigma=args.max_sigma,
        mu_fraction=args.mu_fraction,
        max_tokens_gen=args.max_tokens,
        test_size=args.test_size,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )


async def main():
    config = parse_args()
    trainer = ESTrainer(config)
    try:
        await trainer.setup()
        await trainer.train()
    except Exception as e:
        tqdm.write(f"\n❌ Error: {e}")
        raise
    finally:
        await trainer.teardown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        tqdm.write("\nTraining interrupted.")
