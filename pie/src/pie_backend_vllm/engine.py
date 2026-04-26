"""vLLM-backed inference engine.

Mirrors `pie_backend.engine.Engine`'s public surface so that worker.py can use
either backend interchangeably. Internally, the model and kernels come from
vllm; the surrounding RPC, batching, telemetry, and adapter scaffolding are
imported directly from `pie_backend`.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_backend.config import RuntimeConfig
from pie_backend import telemetry

from . import _require_vllm


class VllmEngine:
    """Inference engine that delegates the forward pass to a vllm model.

    Public surface matches `pie_backend.engine.Engine`:
      - `Engine.load(config, ...)` classmethod
      - `engine.fire_batch(inputs, sampling_metadata) -> list`
      - `engine.kv_cache_at_layer`, `engine.kv_cache_at_layer_host`
      - `engine.adapters`, `engine.swap_pool_size`
      - `engine.config`, `engine.model_config`, `engine.arch_type`,
        `engine.snapshot_dir`
      - `engine.query`, `engine.init_adapter`, `engine.update_adapter`,
        `engine.load_adapter`, `engine.save_adapter`
    """

    config: RuntimeConfig
    forward_pass: object
    model_config: object
    kv_cache_at_layer: list[torch.Tensor]
    kv_cache_at_layer_host: list[torch.Tensor]
    swap_pool_size: int
    adapter_at_layer: list
    adapters: dict
    arch_type: str
    info: dict
    snapshot_dir: str | None

    def __init__(
        self,
        config: RuntimeConfig,
        driver_config,
        model_config,
        forward_pass,
        kv_cache_at_layer: list,
        adapter_at_layer: list,
        arch_type: str,
        info: dict,
        snapshot_dir: str | None = None,
        kv_cache_at_layer_host: list | None = None,
        swap_pool_size: int = 0,
    ):
        self.config = config
        self.driver_config = driver_config
        self.model_config = model_config
        self.forward_pass = forward_pass
        self.kv_cache_at_layer = kv_cache_at_layer
        self.kv_cache_at_layer_host = kv_cache_at_layer_host or []
        self.swap_pool_size = swap_pool_size
        self.adapter_at_layer = adapter_at_layer
        self.arch_type = arch_type
        self.info = info
        self.snapshot_dir = snapshot_dir
        self.adapters = {}

    @classmethod
    def load(
        cls,
        config: RuntimeConfig,
        driver_config,
        log_queue: object = None,
        compute_process_group=None,
    ) -> "VllmEngine":
        _require_vllm()

        from .forward_pass import VllmForwardPass
        from .kv_cache import allocate_and_bind_kv_cache, allocate_host_pool
        from .loader import load_vllm_model
        from .mask_strategies import install_mask_strategies

        def _log(msg: str, level: str = "INFO"):
            if log_queue is not None:
                log_queue.put({"message": msg, "level": level})

        if config.rank == 0:
            telemetry.init_telemetry(
                enabled=config.telemetry_enabled,
                service_name=config.telemetry_service_name,
                endpoint=config.telemetry_endpoint,
            )

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        _log("Loading vllm model", "DEBUG")
        loaded = load_vllm_model(
            config, driver_config, log_queue=log_queue, compute_pg=compute_process_group
        )
        _log("Loaded vllm model", "DEBUG")

        # Wrap each attn layer's impl in a mask-aware proxy. Idempotent;
        # an absent `pie_attn_extras` makes the proxy a single dict.get
        # plus a delegating call. See mask_strategies.py for the per-
        # backend strategies and the refusal policy on unsupported impls.
        install_mask_strategies(loaded.vllm_config)

        # `kv_page_size` is not on the lean RuntimeConfig — the shared RPC
        # worker reads it via `engine.capabilities().kv_page_size`, which
        # we already source from `vllm_config.cache_config.block_size` in
        # `capabilities()` below.

        kv_cache_at_layer = allocate_and_bind_kv_cache(loaded, config, driver_config)
        host_kv, pool_size = allocate_host_pool(kv_cache_at_layer, config.swap_budget_bytes)

        forward_pass = VllmForwardPass(
            model=loaded.model,
            vllm_config=loaded.vllm_config,
            attn_backend=loaded.attn_backend,
            runtime_config=config,
            model_config=loaded.model_config,
        )

        return cls(
            config=config,
            driver_config=driver_config,
            model_config=loaded.model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=[],
            arch_type=loaded.arch_type,
            info=loaded.info,
            snapshot_dir=loaded.snapshot_dir,
            kv_cache_at_layer_host=host_kv,
            swap_pool_size=pool_size,
        )

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> list:
        input_embeds = self.forward_pass.embed_inputs(inputs)

        hidden_states = self.forward_pass.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            # Only forward the mask when there's something to apply — the
            # patched Attention.forward checks for absence as the zero-overhead
            # signal. inputs["custom_mask"] is always populated (BRLE decoded
            # to all-True when no mask was supplied), so we gate on the
            # explicit `has_custom_mask` flag.
            custom_mask=inputs["custom_mask"] if inputs.get("has_custom_mask") else None,
            single_token_inference_mode=inputs["single_token_inference_mode"],
            total_pages_cpu=inputs.get("total_pages_cpu", 0),
        )

        return self.forward_pass.sample(hidden_states, sampling_metadata)

    # ------------------------------------------------------------------
    # Adapters — deferred. v1 raises clearly so workloads that need them
    # fail fast instead of silently producing wrong tokens.
    # ------------------------------------------------------------------

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not yet supported on the vllm backend. "
            "Use --backend native for adapter workloads."
        )

    def update_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the vllm backend.")

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the vllm backend.")

    def save_adapter(self, *args, **kwargs):
        return b""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def query(self, query: str) -> str:
        if query == "ping":
            return "pong"
        return "unknown query"

    def capabilities(self):
        """Report this backend's resolved capacities up to pie's runtime.

        Sources every value from vllm's resolved `VllmConfig` rather than
        echoing the user's `RuntimeConfig`. In particular `kv_page_size`
        comes from the attention backend's chosen block size, which may
        differ from what the user requested.

        Fails loudly if any expected value is missing — the runtime/Rust
        side relies on these being correct, so silent defaulting is unsafe.
        """
        from pie.capabilities import BackendCapabilities

        vc = self.forward_pass.vllm_config
        mc = vc.model_config
        cc = vc.cache_config

        if self.config.max_num_kv_pages is None:
            raise RuntimeError(
                "config.max_num_kv_pages was not set by the loader — KV cache "
                "allocation must run before capabilities() is called."
            )
        if not self.snapshot_dir:
            raise RuntimeError("snapshot_dir is empty; loader did not resolve it.")

        dtype_str = str(mc.dtype).removeprefix("torch.")
        if dtype_str.startswith("torch."):
            raise RuntimeError(
                f"Could not normalize activation dtype {mc.dtype!r}; expected a "
                "torch.dtype with a 'torch.' prefix."
            )

        # vllm expresses batch limits as max_num_seqs / max_num_batched_tokens.
        # Capabilities normalizes them to max_batch_size / max_batch_tokens
        # for pie's runtime side. If max_num_batched_tokens is None (vllm
        # default), use scheduler_config's resolved value.
        max_batch_size = int(self.driver_config.max_num_seqs)
        max_batch_tokens = self.driver_config.max_num_batched_tokens
        if max_batch_tokens is None:
            max_batch_tokens = int(vc.scheduler_config.max_num_batched_tokens)
        else:
            max_batch_tokens = int(max_batch_tokens)

        return BackendCapabilities(
            total_pages=int(self.config.max_num_kv_pages),
            kv_page_size=int(cc.block_size),
            swap_pool_size=int(self.swap_pool_size),
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            arch_name=self.arch_type,
            vocab_size=int(mc.get_vocab_size()),
            max_model_len=int(mc.max_model_len),
            activation_dtype=dtype_str,
            snapshot_dir=str(self.snapshot_dir),
        )
