"""Resolve a vllm model class for the requested HF repo, build a VllmConfig,
load weights, and surface the backend choice.

This module owns the "what kind of vllm model are we running" decisions so
`engine.py` and `worker.py` stay backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from pie_backend.config import RuntimeConfig


# Map pie's RuntimeConfig.activation_dtype (torch.dtype) to vllm's string form.
# vllm's EngineArgs wants strings; we work in torch.dtype internally.
_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
    """Find the local cache path for an HF repo's tokenizer/config.

    Uses `huggingface_hub.snapshot_download` with `local_files_only=True` —
    if vllm has already loaded weights we know the snapshot is on disk. If
    nothing is cached, fall back to scanning HF_HUB_CACHE for the repo's dir.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo, local_files_only=True)
    except Exception:
        pass

    try:
        from pathlib import Path
        from huggingface_hub.constants import HF_HUB_CACHE
        repo_dir = Path(HF_HUB_CACHE) / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
        if repo_dir.is_dir():
            snapshots = sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snapshots:
                return str(snapshots[0])
    except Exception:
        pass

    return None


@dataclass
class LoadedModel:
    """Bundle of everything `Engine.load` needs after vllm finishes loading."""

    model: torch.nn.Module
    vllm_config: Any           # vllm.config.VllmConfig
    attn_backend: Any          # resolved AttentionBackend class (or None — resolved lazily)
    model_config: Any          # vllm's ModelConfig — exposes vocab_size, num_layers, etc.
    arch_type: str             # HF architecture string, e.g. "LlamaForCausalLM"
    info: dict
    snapshot_dir: str | None


def _build_vllm_config(config: RuntimeConfig) -> Any:
    """Build a VllmConfig from pie's RuntimeConfig via the EngineArgs path.

    EngineArgs is vllm's canonical entry point: it does HF config resolution,
    architecture detection, sane defaults, dtype coercion, and produces a
    fully-validated VllmConfig.
    """
    from vllm.engine.arg_utils import EngineArgs

    dtype_str = _DTYPE_TO_STR.get(config.activation_dtype, "auto")

    # Attention backend: read from the RuntimeConfig if pie's CLI set it.
    # `vllm_attn_backend` flows from pie's ModelConfig; "AUTO" → vllm picks.
    attn_backend = getattr(config, "vllm_attn_backend", "AUTO")
    if attn_backend == "AUTO":
        attn_backend_arg = None
    else:
        attn_backend_arg = attn_backend

    engine_kwargs = dict(
        model=config.hf_repo,
        dtype=dtype_str,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_mem_utilization,
        block_size=config.kv_page_size,
        seed=config.random_seed,
        skip_tokenizer_init=True,        # pie owns tokenization
        enforce_eager=not config.use_cuda_graphs,
        download_dir=config.cache_dir,
    )
    if attn_backend_arg is not None:
        # vllm 0.19+ surfaces this via EngineArgs.attention_backend, which is
        # then routed into VllmConfig.attention_config.backend.
        engine_kwargs["attention_backend"] = attn_backend_arg

    args = EngineArgs(**engine_kwargs)
    return args.create_engine_config()


def _ensure_vllm_distributed(vllm_config: Any, rank: int, local_rank: int) -> None:
    """Bring vllm's parallel state up on top of pie's torch.distributed init.

    Pie's worker calls `torch.distributed.init_process_group` only when
    world_size > 1 (single-GPU path skips it for latency). vllm's parallel
    state machinery requires *some* process group, so for the single-rank
    case we bring up a tcp://localhost rendezvous of size 1 here.

    `init_distributed_environment` is idempotent — if dist is already
    initialized (multi-GPU path), it just records the world/rank.
    `ensure_model_parallel_initialized` then constructs vllm's TP/PP/DP
    groups that model layers consult during construction.
    """
    import tempfile, datetime
    from vllm.distributed import (
        init_distributed_environment,
        ensure_model_parallel_initialized,
    )

    parallel_config = vllm_config.parallel_config

    if not dist.is_initialized():
        # Single-rank fallback. FileStore avoids picking a port (no contention).
        store_path = tempfile.mktemp(prefix="pie_vllm_singlerank_")
        store = dist.FileStore(store_path, parallel_config.world_size)
        device_id = (
            torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else None
        )
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            store=store,
            rank=rank,
            world_size=parallel_config.world_size,
            timeout=datetime.timedelta(seconds=300),
            device_id=device_id,
        )

    init_distributed_environment(
        world_size=parallel_config.world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )


def load_vllm_model(
    config: RuntimeConfig,
    log_queue: object = None,
    compute_pg=None,
) -> LoadedModel:
    """Construct a vllm model on the local rank's device and load its weights."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    # Pin the device for this rank — vllm reads current_device() during
    # model construction.
    device_str = str(config.device)
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        local_rank = int(device_str.split(":")[1]) if ":" in device_str else 0
    else:
        local_rank = config.rank

    _log(f"Building VllmConfig for {config.hf_repo}", "DEBUG")
    vllm_config = _build_vllm_config(config)

    # vllm's parallel state and model construction both consult
    # `get_current_vllm_config()` — they must run inside `set_current_vllm_config`.
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader

    with set_current_vllm_config(vllm_config):
        _log("Bringing up vllm parallel state", "DEBUG")
        _ensure_vllm_distributed(vllm_config, rank=config.rank, local_rank=local_rank)

        _log(f"Loading model weights ({config.hf_repo})", "INFO")
        loader = get_model_loader(vllm_config.load_config)
        model = loader.load_model(
            vllm_config=vllm_config,
            model_config=vllm_config.model_config,
        )
        _log("Model weights loaded", "INFO")

    # Architecture string (first arch in the HF config). Used for telemetry
    # and as the `arch_type` reported back through pie's ready handshake.
    arches = list(vllm_config.model_config.hf_text_config.architectures)
    arch_type = arches[0] if arches else "Unknown"

    info = {
        "architecture": {"type": arch_type, "all": arches},
        "vocab_size": vllm_config.model_config.get_vocab_size(),
        "max_model_len": vllm_config.model_config.max_model_len,
        "num_hidden_layers": vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        ),
    }

    # Snapshot dir: pie's Rust runtime needs a local filesystem path that
    # contains tokenizer.json (and the HF config). vllm's `model_config.model`
    # is just the HF repo name like "Qwen/Qwen3-0.6B", so we resolve it to
    # the cached snapshot via huggingface_hub. (We avoid `pie_backend.hf_utils`
    # because it transitively imports `pie_kernels` which JIT-compiles CUDA
    # extensions; that's heavy and irrelevant on the vllm path.)
    snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo)
    if snapshot_dir is None:
        _log(f"Could not resolve HF snapshot dir for {config.hf_repo}", "WARN")

    # `attn_backend` is resolved lazily inside vllm's Attention layers — we
    # don't pin it here; metadata builder reads it from the layer at first
    # forward.
    attn_backend = None

    return LoadedModel(
        model=model,
        vllm_config=vllm_config,
        attn_backend=attn_backend,
        model_config=vllm_config.model_config,
        arch_type=arch_type,
        info=info,
        snapshot_dir=snapshot_dir,
    )
