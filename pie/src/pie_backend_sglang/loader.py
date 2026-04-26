"""Resolve an SGLang model + ModelRunner for the requested HF repo.

Builds an `sglang.srt.server_args.ServerArgs` and an `sglang.srt.configs.model_config.ModelConfig`
from pie's `RuntimeConfig`, brings up sglang's distributed environment on top of
pie's pre-initialized `torch.distributed`, instantiates `ModelRunner`, and
returns a `LoadedModel` bundle that `engine.py` consumes.

This is the analog of `pie_backend_vllm/loader.py`. The high-level shape is
identical; only the SGLang-specific constructor calls change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from pie_backend.config import RuntimeConfig


# Map pie's RuntimeConfig.activation_dtype (torch.dtype) to sglang's string form.
_DTYPE_TO_STR = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


def _resolve_hf_snapshot_dir(hf_repo: str) -> str | None:
    """Find the local cache path for an HF repo's tokenizer/config.

    Pie's Rust runtime needs a local filesystem path containing tokenizer.json.
    """
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(hf_repo, local_files_only=True)
    except Exception:
        pass

    try:
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
    """Bundle of everything `Engine.load` needs after sglang finishes loading."""

    runner: Any                # sglang.srt.model_executor.model_runner.ModelRunner
    server_args: Any           # sglang.srt.server_args.ServerArgs
    sglang_model_config: Any   # sglang.srt.configs.model_config.ModelConfig
    arch_type: str             # HF architecture string, e.g. "Qwen2ForCausalLM"
    info: dict
    snapshot_dir: str | None
    page_size: int             # sglang's chosen page size (== pie's kv_page_size)
    num_kv_layers: int


def _build_sglang_server_args(config: RuntimeConfig) -> Any:
    """Build an SGLang ServerArgs from pie's RuntimeConfig.

    We only set the fields pie cares about; everything else uses sglang's
    defaults. `__post_init__` handles validation and platform-specific
    normalization.
    """
    from sglang.srt.server_args import ServerArgs

    dtype_str = _DTYPE_TO_STR.get(config.activation_dtype, "auto")

    # SGLang's `device` is the *device kind* (cuda/cpu/...), not the index.
    # The index flows through `gpu_id` to ModelRunner.
    device_kind = "cuda" if str(config.device).startswith("cuda") else str(config.device)

    args = ServerArgs(
        model_path=config.hf_repo,
        # Pie owns tokenization (Rust runtime). Skip sglang's tokenizer init.
        skip_tokenizer_init=True,
        trust_remote_code=True,
        dtype=dtype_str,
        # pie's gpu_mem_utilization → sglang's mem_fraction_static
        mem_fraction_static=config.gpu_mem_utilization,
        page_size=config.kv_page_size,
        device=device_kind,
        tp_size=config.tensor_parallel_size,
        attention_backend=config.sglang_attn_backend,
        # We don't run sglang's HTTP server; these fields are inert here.
        host="127.0.0.1",
        port=30000,
        # SGLang uses cuda graphs by default; mirror pie's flag.
        disable_cuda_graph=not config.use_cuda_graphs,
        # Don't try to start a real RPC server; we drive ModelRunner directly.
        skip_server_warmup=True,
        random_seed=config.random_seed,
        download_dir=config.cache_dir,
        # Disable features pie doesn't need (and that often pull in extra deps).
        disable_radix_cache=True,
    )
    return args


def _ensure_sglang_distributed(server_args: Any, rank: int, gpu_id: int, tp_size: int) -> None:
    """Bring up sglang's parallel state on top of pie's torch.distributed init.

    SGLang's `init_distributed_environment` is idempotent: if `torch.distributed`
    is already up, it skips `init_process_group` and just records the world/rank.
    `initialize_model_parallel` then creates SGLang's TP/PP groups that model
    layers consult during construction.
    """
    import datetime
    import tempfile

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not dist.is_initialized():
        # Single-rank fallback (TP=1, no pre-init).
        store_path = tempfile.mktemp(prefix="pie_sglang_singlerank_")
        store = dist.FileStore(store_path, tp_size)
        device_id = (
            torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else None
        )
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            store=store,
            rank=rank,
            world_size=tp_size,
            timeout=datetime.timedelta(seconds=300),
            device_id=device_id,
        )

    init_distributed_environment(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        world_size=tp_size,
        rank=rank,
        local_rank=gpu_id,
        distributed_init_method="env://",
    )

    initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )


def load_sglang_model(
    config: RuntimeConfig,
    log_queue: object = None,
    compute_pg=None,
) -> LoadedModel:
    """Construct an SGLang ModelRunner on the local rank's device."""

    def _log(msg: str, level: str = "INFO"):
        if log_queue is not None:
            log_queue.put({"message": msg, "level": level})

    # Pin the device for this rank.
    device_str = str(config.device)
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
        gpu_id = int(device_str.split(":")[1]) if ":" in device_str else 0
    else:
        gpu_id = config.rank

    _log(f"Building SGLang ServerArgs for {config.hf_repo}", "DEBUG")
    server_args = _build_sglang_server_args(config)

    # ModelRunner.init_torch_distributed reads env vars MASTER_ADDR/MASTER_PORT.
    # Pie's worker._init_distributed sets them via FileStore; for the single-rank
    # path we still need the env vars to be set so sglang doesn't blow up.
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", str(config.rank))
    os.environ.setdefault("WORLD_SIZE", str(config.tensor_parallel_size))
    os.environ.setdefault("LOCAL_RANK", str(gpu_id))

    # Build SGLang's ModelConfig (separate from ServerArgs; reads HF config).
    from sglang.srt.configs.model_config import ModelConfig as SGLangModelConfig
    sglang_model_config = SGLangModelConfig.from_server_args(server_args)

    _log(f"Loading model weights via SGLang ({config.hf_repo})", "INFO")

    from sglang.srt.model_executor.model_runner import ModelRunner

    # ModelRunner internally:
    #   - calls init_torch_distributed (idempotent w/ pre-init)
    #   - loads the model
    #   - allocates KV cache via init_memory_pool
    #   - sets up the attention backend
    runner = ModelRunner(
        model_config=sglang_model_config,
        mem_fraction_static=server_args.mem_fraction_static or config.gpu_mem_utilization,
        gpu_id=gpu_id,
        tp_rank=config.rank,
        tp_size=config.tensor_parallel_size,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=int(os.environ.get("MASTER_PORT", "29500")) + 1,
        server_args=server_args,
    )

    _log("Model loaded via SGLang", "INFO")

    # Pull architecture string from HF config (matches vllm path).
    arches = list(sglang_model_config.hf_config.architectures)
    arch_type = arches[0] if arches else "Unknown"

    # Resolve the chosen page_size — sglang may override what we asked for
    # depending on the attention backend.
    chosen_page_size = int(runner.page_size)

    info = {
        "architecture": {"type": arch_type, "all": arches},
        "vocab_size": sglang_model_config.vocab_size,
        "max_model_len": sglang_model_config.context_len,
        "num_hidden_layers": runner.num_effective_layers,
    }

    snapshot_dir = _resolve_hf_snapshot_dir(config.hf_repo)
    if snapshot_dir is None:
        _log(f"Could not resolve HF snapshot dir for {config.hf_repo}", "WARN")

    return LoadedModel(
        runner=runner,
        server_args=server_args,
        sglang_model_config=sglang_model_config,
        arch_type=arch_type,
        info=info,
        snapshot_dir=snapshot_dir,
        page_size=chosen_page_size,
        num_kv_layers=runner.num_effective_layers,
    )
