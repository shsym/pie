"""SGLang-backed inference engine.

Mirrors `pie_backend.engine.Engine`'s public surface so worker.py can use
either backend interchangeably. Internally, the model and kernels come from
SGLang; the surrounding RPC, batching, telemetry, and dispatch scaffolding are
imported directly from `pie_backend`.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_backend.config import RuntimeConfig
from pie_backend import telemetry

from . import _require_sglang


class SGLangEngine:
    """Inference engine that delegates the forward pass to an SGLang ModelRunner.

    Public surface matches `pie_backend.engine.Engine`. See `VllmEngine` in the
    sibling vllm backend for the canonical reference.
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
        log_queue: object = None,
        compute_process_group=None,
    ) -> "SGLangEngine":
        _require_sglang()

        from .loader import load_sglang_model
        from .kv_cache import _rebind_pool_buffers
        from .forward_pass import SGLangForwardPass

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

        _log("Loading sglang model", "DEBUG")
        loaded = load_sglang_model(config, log_queue=log_queue, compute_pg=compute_process_group)
        _log("Loaded sglang model", "DEBUG")

        # Rebind sglang's k_buffer/v_buffer to pie-shaped storage so the swap
        # RPC handlers see the canonical (num_blocks, 2, page_size, h, d) layout.
        kv_cache_at_layer, num_blocks, page_size = _rebind_pool_buffers(loaded, config)
        config.max_num_kv_pages = num_blocks
        # Honor sglang's chosen page size (may differ from the TOML hint).
        config.kv_page_size = page_size

        forward_pass = SGLangForwardPass(
            runner=loaded.runner,
            runtime_config=config,
            page_size=page_size,
        )

        return cls(
            config=config,
            model_config=loaded.sglang_model_config,
            forward_pass=forward_pass,
            kv_cache_at_layer=kv_cache_at_layer,
            adapter_at_layer=[],
            arch_type=loaded.arch_type,
            info=loaded.info,
            snapshot_dir=loaded.snapshot_dir,
            kv_cache_at_layer_host=[],
            swap_pool_size=0,
        )

    @torch.inference_mode()
    def fire_batch(self, inputs: dict, sampling_metadata: dict) -> dict:
        # We collapse embed/transform: SGLang's forward() owns input embedding.
        # `embed_inputs` returns the inputs dict as a passthrough; transform()
        # builds the ForwardBatch and runs the model.
        passthrough = self.forward_pass.embed_inputs(inputs)

        gathered_logits = self.forward_pass.transform(
            input_embeds=passthrough,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            single_token_inference_mode=inputs["single_token_inference_mode"],
            total_pages_cpu=inputs.get("total_pages_cpu", 0),
        )

        return self.forward_pass.sample(gathered_logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Adapters — deferred. Same stance as vllm backend.
    # ------------------------------------------------------------------

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not yet supported on the sglang backend. "
            "Use --backend native for adapter workloads."
        )

    def update_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the sglang backend.")

    def load_adapter(self, *args, **kwargs):
        raise NotImplementedError("Adapters are not yet supported on the sglang backend.")

    def save_adapter(self, *args, **kwargs):
        return b""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def query(self, query: str) -> str:
        if query == "ping":
            return "pong"
        return "unknown query"


# Alias so worker code that imports `Engine` from this module works.
Engine = SGLangEngine
