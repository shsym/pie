"""
Runtime orchestrator for PIE backend.

This module provides the main Runtime class that orchestrates inference
by delegating to specialized components:
- config.py: RuntimeConfig
- batching.py: BatchBuilder, BatchState
- loader.py: ModelLoader
"""

from __future__ import annotations

import base64
import random
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
from . import utils
import torch
import torch.distributed as dist
import queue
import threading

from .config import RuntimeConfig
from .batching import BatchBuilder, Batch
from .loader import ModelLoader
from .adapter import AdapterSubpass, CmaesAdapter
from .model import llama3, qwen2, qwen3, common, gpt_oss
from . import message
from . import hf_utils

# Re-export RuntimeConfig for backward compatibility
__all__ = ["Runtime", "RuntimeConfig"]


# Helper class for result tracking
class PendingResult:
    def __init__(self, total, metadata):
        self.total = total
        self.received = 0
        self.metadata = metadata
        self.resps = [None] * total

    def add_response(self, idx, resp):
        self.resps[idx] = resp
        self.received += 1
        return self.received == self.total


class Runtime:
    """
    Main runtime orchestrator for PIE inference.

    This class has been refactored to focus on protocol handling,
    delegating loading to ModelLoader and batch management to BatchBuilder.
    """

    config: RuntimeConfig

    # Model components (renamed from forward_pass to engine to avoid collision)
    engine: llama3.ForwardPass
    model_config: llama3.ModelConfig
    kv_cache_at_layer: list[torch.Tensor]

    # Adapter state
    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    adapters: dict

    # Batch management
    batch_builder: BatchBuilder
    batch: Batch | None

    # Logging
    log_queue: object | None

    def __init__(self, config: RuntimeConfig, log_queue: object | None = None):
        """
        Initialize the runtime.

        Args:
            config: Runtime configuration
            log_queue: Optional queue for sending logs back to controller
        """
        self.config = config
        self.log_queue = log_queue
        self.log_queue = log_queue
        self.adapters = {}
        self.batch = None

        # Async Execution - 2 Stage Pipeline
        # Stage 1: worker_thread (in server.py) receives requests and enqueues
        # Stage 2: execution_loop drains queue, builds batch, executes
        self.request_queue = queue.Queue()
        self.response_callback = None

        self.execution_thread = threading.Thread(
            target=self.execution_loop, daemon=True
        )
        self.execution_thread.start()

        # Initialize seeds
        msg = f"Initializing with random seed: {config.random_seed}"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_seed)

        # Load model weights using ModelLoader
        loader = ModelLoader(config)

        msg = "Loading model weights"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        weights, normalized_arch, self.info = loader.load()

        # Store snapshot_dir for tokenizer loading
        self.snapshot_dir = loader.snapshot_dir

        msg = "Loaded model weights"
        self.log_queue.put({"message": msg, "level": "DEBUG"})

        # Store architecture type
        self.type = self.info["architecture"]["type"]

        # Create model-specific components based on architecture
        match self.type:
            case "llama3" | "l4ma":
                # Create model config
                self.model_config = llama3.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST (needed for ForwardPass)
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = llama3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                # Create adapter cache
                self.adapter_at_layer = llama3.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = llama3.create_kv_cache(
                    self.model_config, config
                )

                # Warmup CUDA graphs
                self.engine.warmup_cuda_graphs(self.kv_cache_at_layer)

            case "qwen2":
                # Create model config
                self.model_config = qwen2.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = qwen2.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                # Create adapter cache
                self.adapter_at_layer = qwen2.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = qwen2.create_kv_cache(
                    self.model_config, config
                )

            case "qwen3":
                # Create model config
                self.model_config = qwen3.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = qwen3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )

                # Create adapter cache
                self.adapter_at_layer = qwen3.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = qwen3.create_kv_cache(
                    self.model_config, config
                )

            case "gptoss":
                # Create model config
                self.model_config = gpt_oss.ModelConfig.from_dict(normalized_arch)

                # Evaluate and store max_num_kv_pages in config FIRST
                config.max_num_kv_pages = self.model_config.eval_max_num_kv_pages(
                    config
                )

                # Create forward pass with weights
                self.engine = gpt_oss.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                # Create adapter cache
                self.adapter_at_layer = gpt_oss.create_adapter_cache(
                    self.model_config, config
                )
                # Create KV cache
                self.kv_cache_at_layer = gpt_oss.create_kv_cache(
                    self.model_config, config
                )

            case _:
                raise ValueError(f"Unsupported architecture type: {self.type}")

        # Initialize batch builder
        self.batch_builder = BatchBuilder(
            kv_page_size=config.kv_page_size,
            max_dist_size=config.max_dist_size,
            adapters=self.adapters,
        )

    # ========================================================================
    # Metadata Accessors
    # ========================================================================

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "name": self.config.hf_repo,
            "description": "",
            "version": "1.0.0",
        }

    def get_chat_template(self) -> dict:
        """Get chat template configuration from HuggingFace tokenizer_config.json."""
        if self.snapshot_dir is None:
            return {
                "template_type": "none",
                "template_content": "",
                "stop_tokens": [],
            }

        # Load tokenizer info from HuggingFace
        tokenizer_info = hf_utils.load_hf_tokenizer(self.snapshot_dir)
        chat_template = tokenizer_info.get("chat_template", "")

        # Determine stop tokens from special tokens
        special_tokens = tokenizer_info.get("special_tokens", {})
        stop_tokens = []
        # Common stop tokens across models
        for token_name in ["<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "</s>"]:
            if token_name in special_tokens:
                stop_tokens.append(token_name)

        return {
            "template_type": "minijinja" if chat_template else "none",
            "template_content": chat_template,
            "stop_tokens": stop_tokens,
        }

    def get_tokenizer(self) -> dict:
        """Get tokenizer configuration with merge table from HuggingFace."""
        if self.snapshot_dir is None:
            return {
                "type": "bpe",
                "num_vocab": self.engine.weights.get("embed_token").shape[0],
                "merge_table": {},
                "split_regex": "",
                "special_tokens": {},
                "escape_non_printable": False,
            }

        # Load tokenizer info from HuggingFace
        tokenizer_info = hf_utils.load_hf_tokenizer(self.snapshot_dir)

        return {
            "type": tokenizer_info.get("type", "bpe"),
            "num_vocab": tokenizer_info.get(
                "num_vocab", self.engine.weights.get("embed_token").shape[0]
            ),
            "merge_table": tokenizer_info.get("merge_table", {}),
            "split_regex": tokenizer_info.get("split_regex", ""),
            "special_tokens": tokenizer_info.get("special_tokens", {}),
            "escape_non_printable": tokenizer_info.get("escape_non_printable", False),
        }

    # ========================================================================
    # Service Protocol Implementation
    # ========================================================================
    def handshake(
        self, reqs: list[message.HandshakeRequest]
    ) -> list[message.HandshakeResponse]:
        """Handle handshake requests returning model and tokenizer info."""
        metadata = self.get_metadata()
        template = self.get_chat_template()
        tokenizer = self.get_tokenizer()

        responses = []
        for _ in reqs:
            resp = message.HandshakeResponse(
                version=metadata.get("version", "1.0.0"),
                model_name=metadata["name"],
                model_traits=[],  # TODO: populate traits
                model_description=metadata.get("description", ""),
                prompt_template=template["template_content"],
                prompt_template_type=template["template_type"],
                prompt_stop_tokens=template["stop_tokens"],
                kv_page_size=self.config.kv_page_size,
                max_batch_tokens=self.config.max_batch_tokens or 10240,
                resources={
                    0: self.config.max_num_kv_pages or 0,
                    1: self.config.max_num_embeds,
                    2: self.config.max_num_adapters,
                },
                tokenizer_num_vocab=tokenizer["num_vocab"],
                tokenizer_merge_table=tokenizer["merge_table"],
                tokenizer_special_tokens=tokenizer["special_tokens"],
                tokenizer_split_regex=tokenizer["split_regex"],
                tokenizer_escape_non_printable=tokenizer["escape_non_printable"],
            )
            responses.append(resp)
        return responses

    def query(self, reqs: list[message.QueryRequest]) -> list[message.QueryResponse]:
        """Handle query requests."""
        responses = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            responses.append(message.QueryResponse(value=value))
        return responses

    def forward_pass_handler(
        self, reqs: list[message.ForwardPassRequest]
    ) -> list[message.ForwardPassResponse]:
        """
        Handle batched forward pass inference requests.

        Note: This method was renamed from forward_pass to avoid collision
        with the forward_pass property. For backward compatibility, the
        server.py handler should call this method.
        """
        # Accumulate requests into the batch
        for req in reqs:
            self.batch_builder.add_request(req)

        # Execute the batch and return responses
        return self._execute_batch()

    def set_response_callback(self, callback):
        """Set callback for sending async responses."""
        self.response_callback = callback

    def forward_pass_handler_v2(
        self, reqs: list[message.ForwardPassRequest], metadata: tuple
    ) -> None:
        """
        Async handler for batched forward pass inference requests.
        Enqueues raw requests for the preparation thread.
        """
        self.request_queue.put((reqs, metadata))

    @torch.inference_mode()
    def execution_loop(self):
        """
        Unified execution loop: drains request queue, builds batches on GPU, executes.
        This runs in a single thread to avoid GIL contention.
        """
        while True:
            try:
                item = self.request_queue.get()
            except Exception:
                break

            if item is None:
                break

            # Start collecting items to process
            pending_items = [item]

            # Drain queue to accumulate more requests for better batching
            while True:
                try:
                    next_item = self.request_queue.get_nowait()
                    if next_item is None:
                        break
                    pending_items.append(next_item)
                except queue.Empty:
                    break

            # Create Result Objects
            pending_results = []
            for reqs, metadata in pending_items:
                pending_results.append(PendingResult(len(reqs), metadata))

            # Build and execute batches
            curr_cursor_per_item = [0] * len(pending_items)

            while True:
                batch_mapping = []  # (PendingResult, req_idx)
                batch_full = False

                for item_idx, (reqs, _) in enumerate(pending_items):
                    start_idx = curr_cursor_per_item[item_idx]
                    if start_idx >= len(reqs):
                        continue

                    for i, req in enumerate(reqs[start_idx:]):
                        self.batch_builder.add_request(req)
                        batch_mapping.append((pending_results[item_idx], start_idx + i))

                        current_batch = self.batch_builder.current_batch
                        if current_batch:
                            # Check 1: Max tokens
                            is_full_tokens = (
                                current_batch.total_tokens
                                >= self.config.max_batch_tokens
                            )

                            # Check 2: Max requests (batch size)
                            is_full_size = False
                            if self.config.max_batch_size:
                                is_full_size = (
                                    len(current_batch.requests)
                                    >= self.config.max_batch_size
                                )

                            if is_full_tokens or is_full_size:
                                batch_full = True
                                break

                    items_added = sum(
                        1
                        for (pr, _) in batch_mapping
                        if pr is pending_results[item_idx] and _ >= start_idx
                    )
                    curr_cursor_per_item[item_idx] += items_added

                    if batch_full:
                        break

                if not self.batch_builder.is_empty():
                    # Execute batch directly (on GPU)
                    responses = self._execute_batch()

                    # Send responses
                    if self.response_callback:
                        for resp, (pending_result, req_idx) in zip(
                            responses, batch_mapping
                        ):
                            if pending_result.add_response(req_idx, resp):
                                self.response_callback(
                                    *pending_result.metadata, pending_result.resps
                                )
                else:
                    break

                if not batch_full:
                    break

    @torch.inference_mode()
    def _execute_prepared_batch(
        self, inputs, sampling_metadata, batch
    ) -> list[message.ForwardPassResponse]:
        """
        Execute a batch that has already been prepared.
        """
        # Broadcast if needed
        if self.config.world_size > 1:
            msg = {
                "type": "STEP",
                "inputs": inputs,
                "sampling_metadata": sampling_metadata,
            }
            utils.broadcast_struct(msg, src=0, device=self.config.device)

        # Execute step
        sampling_results = self._run_step(inputs, sampling_metadata)

        # Package responses
        responses = batch.create_responses(sampling_results)

        return responses

    def embed_image(self, reqs: list[message.EmbedImageRequest]) -> None:
        """Handle image embedding requests."""
        # TODO: implement image embedding
        pass

    def initialize_adapter(self, reqs: list[message.InitializeAdapterRequest]) -> None:
        """Initialize adapter functionality."""
        for req in reqs:
            adapter_ptr = req.adapter_ptr
            # Prepare args for initialization
            args = {
                "adapter_ptr": adapter_ptr,
                "rank": req.rank,
                "alpha": req.alpha,
                "population_size": req.population_size,
                "mu_fraction": req.mu_fraction,
                "initial_sigma": req.initial_sigma,
            }

            if self.config.world_size > 1:
                # Broadcast INIT_ADAPTER command
                msg = {"type": "INIT_ADAPTER", "kwargs": args}
                utils.broadcast_struct(msg, src=0, device=self.config.device)

            self._initialize_adapter(**args)

    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]) -> None:
        """Update adapter parameters."""
        for req in reqs:
            args = {
                "adapter_ptr": req.adapter_ptr,
                "scores": req.scores,
                "seeds": req.seeds,
                "max_sigma": req.max_sigma,
            }

            if self.config.world_size > 1:
                # Broadcast UPDATE_ADAPTER command
                msg = {"type": "UPDATE_ADAPTER", "kwargs": args}
                utils.broadcast_struct(msg, src=0, device=self.config.device)

            self._update_adapter(**args)

    def upload_adapter(self, reqs: list[message.UploadAdapterRequest]) -> None:
        """Upload adapter weights."""
        for req in reqs:
            # Convert list[int] to bytes if necessary
            data = req.adapter_data
            if isinstance(data, list):
                data = bytes(data)

            args = {"adapter_ptr": req.adapter_ptr, "name": req.name, "data": data}

            if self.config.world_size > 1:
                # Broadcast UPLOAD_ADAPTER command
                # We do NOT include the data in the broadcast for now to avoid massive traffic
                # Each rank must load it? Or rank 0 loads and broadcasts weights?
                # The CmaesAdapter implementation handles broadcasting/sharding OF THE WEIGHTS
                # inside .upload() if we wanted, but currently it just loads from file.
                # However, since we are moving to in-memory, we might need rank 0 to broadcast.
                # BUT: The current architecture assumes the CLIENT sends the request to the server.
                # If we are using multi-GPU, we need consistent state.
                # For now, let's assume we pass the data down.
                msg = {"type": "UPLOAD_ADAPTER", "kwargs": args}
                utils.broadcast_struct(msg, src=0, device=self.config.device)

            self._upload_adapter(**args)

    def _upload_adapter(self, adapter_ptr: int, name: str, data: bytes) -> None:
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                # For multi-GPU, append rank to filename to ensure each rank loads its own shard
                # Current CmaesAdapter implementation expects rank-specific checkpoints
                # For multi-GPU, append rank to filename to ensure each rank loads its own shard
                # Current CmaesAdapter implementation expects rank-specific checkpoints
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"

                adapter.upload(name, data)

    def download_adapter(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        """Download adapter weights."""
        resps = []
        for req in reqs:
            args = {"adapter_ptr": req.adapter_ptr, "name": req.name}

            if self.config.world_size > 1:
                # Broadcast DOWNLOAD_ADAPTER command
                msg = {"type": "DOWNLOAD_ADAPTER", "kwargs": args}
                utils.broadcast_struct(msg, src=0, device=self.config.device)

            data = self._download_adapter(**args)

            # Pack response (only Rank 0 returns data to client)
            resp = message.DownloadAdapterResponse(adapter_data=data)
            resps.append(resp)

        return resps

    def _download_adapter(self, adapter_ptr: int, name: str) -> bytes:
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                # For multi-GPU, append rank to filename to save rank-specific shards
                if self.config.world_size > 1:
                    name = f"{name}_rank{self.config.rank}"

                return adapter.download(name)
        return b""

    # ========================================================================
    # Internal Adapter Methods
    # ========================================================================

    @torch.inference_mode()
    def _initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ):
        cfg = self.model_config

        # Check if adapter limits are exceeded
        if adapter_ptr >= self.config.max_num_adapters:
            raise ValueError(
                f"Adapter pointer {adapter_ptr} exceeds max_num_adapters {self.config.max_num_adapters}"
            )
        # print parameters
        # print(f"Initializing adapter {adapter_ptr} with rank {rank}, alpha {alpha}, population size {population_size}, mu fraction {mu_fraction}, initial sigma {initial_sigma}")
        # Calculate local shard sizes for distributed adapters
        world_size = self.config.world_size
        gpu_rank = self.config.rank

        local_num_q_heads = cfg.num_q_heads // world_size
        local_num_kv_heads = cfg.num_kv_heads // world_size

        # Local output features (sharded up-projection)
        local_out_features = [
            cfg.dim_head * local_num_q_heads,
            cfg.dim_head * local_num_kv_heads,
            cfg.dim_head * local_num_kv_heads,
        ]

        # Initialize adapter
        self.adapters[adapter_ptr] = CmaesAdapter(
            adapter_id=adapter_ptr,
            adapter_at_layer=self.adapter_at_layer,
            rank=rank,
            alpha=alpha,
            in_features=cfg.dim_hidden,
            out_features=local_out_features,
            num_layers=cfg.num_layers,
            population_size=population_size,
            mu_fraction=mu_fraction,
            initial_sigma=initial_sigma,
            min_sigma=1e-7,
            min_var=1e-8,
            max_var=1e4,
            device=self.config.device,
            dtype=self.config.activation_dtype,
            gpu_rank=gpu_rank,
            world_size=world_size,
        )

    @torch.inference_mode()
    def _update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        if adapter_ptr in self.adapters:
            adapter = self.adapters[adapter_ptr]
            if isinstance(adapter, CmaesAdapter):
                adapter.update(scores, seeds, max_sigma)

    # ========================================================================
    # Batch Execution
    # ========================================================================

    @torch.inference_mode()
    def worker_loop(self):
        """
        Worker loop for ranks > 0.
        Waits for control messages from rank 0 and executes commands.
        """
        import torch.distributed as dist
        import signal

        # Re-enable SIGTERM handling so this worker can be terminated
        # by the parent process during shutdown
        shutdown_requested = False

        def sigterm_handler(signum, frame):
            nonlocal shutdown_requested
            shutdown_requested = True

        signal.signal(signal.SIGTERM, sigterm_handler)
        signal.signal(signal.SIGINT, sigterm_handler)

        device = self.config.device

        while not shutdown_requested:
            # Receive control message
            try:
                msg = utils.broadcast_struct(None, src=0, device=device)
            except Exception:
                break

            if shutdown_requested:
                break

            if msg == "STOP":
                break

            if isinstance(msg, dict):
                msg_type = msg.get("type")

                if msg_type == "STEP":
                    # Execute inference step
                    inputs = msg["inputs"]
                    sampling_metadata = msg["sampling_metadata"]
                    try:
                        self._run_step(inputs, sampling_metadata)
                    except Exception as e:
                        # Log error but continue - don't let one error hang the whole system
                        print(f"Worker {self.config.rank} _run_step error: {e}")
                        # Sync CUDA to clear any pending operations
                        torch.cuda.synchronize()

                elif msg_type == "INIT_ADAPTER":
                    # Initialize adapter
                    kwargs = msg["kwargs"]
                    self._initialize_adapter(**kwargs)

                elif msg_type == "UPDATE_ADAPTER":
                    # Update adapter
                    kwargs = msg["kwargs"]
                    self._update_adapter(**kwargs)

                elif msg_type == "UPLOAD_ADAPTER":
                    # Upload adapter
                    kwargs = msg["kwargs"]
                    self._upload_adapter(**kwargs)

                elif msg_type == "DOWNLOAD_ADAPTER":
                    # Download adapter
                    kwargs = msg["kwargs"]
                    self._download_adapter(**kwargs)

            # Other message types can be added here

    def _run_step(self, inputs: dict, sampling_metadata: dict) -> list:
        """
        Execute a single inference step (Embed -> Transform -> Sample).

        Returns:
            Sampling results (only valid on Rank 0 usually, but we return whatever comes out)
        """
        if self.config.world_size > 1:
            torch.distributed.barrier()

        # 2. Embed inputs
        input_embeds = self.engine.embed_inputs(inputs)

        # 3. Use raw indices/seeds to create local AdapterSubpass (to avoid device mismatch)
        adapter_subpass = None
        if inputs.get("adapter_indices"):
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=inputs["adapter_indices"],
                adapter_extras=self.adapters,
                rand_seeds=inputs["adapter_seeds"],
                qo_indptr=inputs["qo_indptr"],
            )

        # 4. Run transformer forward pass
        hidden_states = self.engine.transform(
            input_embeds=input_embeds,
            position_ids=inputs["position_ids"],
            qo_indptr=inputs["qo_indptr"],
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=inputs["kv_page_indices"],
            kv_page_indptr=inputs["kv_page_indptr"],
            kv_last_page_lens=inputs["kv_last_page_lens"],
            custom_mask=inputs["custom_mask"],
            single_token_inference_mode=inputs["single_token_inference_mode"],
            adapter_subpass=adapter_subpass,
            total_pages_cpu=inputs.get("total_pages_cpu", 0),
        )

        # 4. Sampling Pass
        # Pass hidden_states (replicated or sharded? Transform returns replicated in current llama3.py)
        sampling_results = self.engine.sample(hidden_states, sampling_metadata)

        return sampling_results

    @torch.inference_mode()
    def _execute_batch(self) -> list[message.ForwardPassResponse]:
        """
        Execute the accumulated batch and return responses.
        """
        batch = self.batch_builder.build()
        device = self.config.device
        sampling_metadata = batch.get_sampling_metadata(
            device, self.config.activation_dtype
        )

        inputs = batch.get_model_inputs(device)

        # Broadcast if needed
        if self.config.world_size > 1:
            # Broadcast Step command
            msg = {
                "type": "STEP",
                "inputs": inputs,
                "sampling_metadata": sampling_metadata,
            }

            utils.broadcast_struct(msg, src=0, device=device)

        # Execute step
        sampling_results = self._run_step(inputs, sampling_metadata)

        # Package responses
        responses = batch.create_responses(sampling_results)

        return responses

    def shutdown(self):
        """
        Cleanup runtime resources.

        For Rank 0 in multi-GPU setup, this broadcasts the 'STOP' signal to workers.
        """
        if self.config.world_size > 1 and self.config.rank == 0:
            print("Broadcasting STOP signal to workers...")
            utils.broadcast_struct("STOP", src=0, device=self.config.device)
