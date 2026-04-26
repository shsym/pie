"""SGLang-backed inference engine.

Mirrors `pie_backend.engine.Engine`'s public surface so the worker can use
either driver interchangeably. Internally, the model and kernels come from
SGLang; the surrounding RPC, batching, telemetry, and dispatch scaffolding
are imported directly from `pie_backend`.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from pie_backend.config import RuntimeConfig
from pie_backend import telemetry

from . import _require_sglang
from .config import SGLangDriverConfig


class SGLangEngine:
    """Inference engine that delegates the forward pass to an SGLang ModelRunner.

    Public surface matches `pie_backend.engine.Engine`. See `VllmEngine` for
    the canonical sibling reference.
    """

    config: RuntimeConfig
    driver_config: SGLangDriverConfig
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
        driver_config: SGLangDriverConfig,
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
        driver_config: SGLangDriverConfig,
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
        loaded = load_sglang_model(
            config, driver_config, log_queue=log_queue, compute_pg=compute_process_group
        )
        _log("Loaded sglang model", "DEBUG")

        # Rebind sglang's k_buffer/v_buffer to pie-shaped storage so the swap
        # RPC handlers see the canonical (num_blocks, 2, page_size, h, d) layout.
        kv_cache_at_layer, num_blocks, page_size = _rebind_pool_buffers(loaded, config)
        config.max_num_kv_pages = num_blocks

        forward_pass = SGLangForwardPass(
            runner=loaded.runner,
            runtime_config=config,
            page_size=page_size,
        )

        return cls(
            config=config,
            driver_config=driver_config,
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
            custom_mask=inputs.get("custom_mask"),
        )

        return self.forward_pass.sample(gathered_logits, sampling_metadata)

    # ------------------------------------------------------------------
    # Adapters — deferred. Same stance as vllm backend.
    # ------------------------------------------------------------------

    def init_adapter(self, *args, **kwargs):
        raise NotImplementedError(
            "Adapters are not yet supported on the sglang backend. "
            "Use --driver native for adapter workloads."
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

    def capabilities(self):
        """Report this backend's resolved capacities up to pie's runtime.

        Sources values from the loaded SGLang ModelRunner / ServerArgs so
        the user's preferences are reconciled against what sglang's chosen
        attention kernel actually supports (page_size in particular may
        differ from what the user asked for).
        """
        from pie.capabilities import BackendCapabilities

        runner = self.forward_pass.runner
        sglang_mc = runner.model_config

        if self.config.max_num_kv_pages is None:
            raise RuntimeError(
                "config.max_num_kv_pages was not set by the loader — KV cache "
                "rebind must run before capabilities() is called."
            )
        if not self.snapshot_dir:
            raise RuntimeError("snapshot_dir is empty; loader did not resolve it.")

        dtype_str = str(self.config.activation_dtype).removeprefix("torch.")

        # SGLang derives `max_running_requests` and `max_total_num_tokens`
        # from `mem_fraction_static`. We surface those as pie's batch limits.
        max_batch_size = int(getattr(runner, "max_running_requests", 256))
        max_batch_tokens = int(getattr(runner, "max_total_num_tokens", 10240))

        return BackendCapabilities(
            total_pages=int(self.config.max_num_kv_pages),
            kv_page_size=int(runner.page_size),
            swap_pool_size=int(self.swap_pool_size),
            max_batch_tokens=max_batch_tokens,
            max_batch_size=max_batch_size,
            arch_name=self.arch_type,
            vocab_size=int(sglang_mc.vocab_size),
            max_model_len=int(sglang_mc.context_len),
            activation_dtype=dtype_str,
            snapshot_dir=str(self.snapshot_dir),
        )


# Alias so worker code that imports `Engine` from this module works.
Engine = SGLangEngine
