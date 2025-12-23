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
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from .config import RuntimeConfig
from .batching import BatchBuilder, Batch
from .loader import ModelLoader
from .adapter import AdapterSubpass
from .model import llama3, qwen2, qwen3, common
from . import message

# Re-export RuntimeConfig for backward compatibility
__all__ = ["Runtime", "RuntimeConfig"]


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

    def __init__(self, config: RuntimeConfig):
        """
        Initialize the runtime.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self.adapters = {}
        self.batch = None

        # Load model weights using ModelLoader
        loader = ModelLoader(config)
        print("Loading model weights")
        weights, normalized_arch, self.info = loader.load()

        print("Loaded model weights")

        # Store architecture type
        self.type = self.info["architecture"]["type"]

        # Create model-specific components based on architecture
        match self.type:
            case "llama3" | "l4ma":
                # Create model config
                self.model_config = llama3.ModelConfig.from_dict(normalized_arch)
                
                # Create forward pass with weights
                self.engine = llama3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                
                # Create KV cache
                self.kv_cache_at_layer = llama3.create_kv_cache(
                    self.model_config, config
                )

            case "qwen2":
                # Create model config
                self.model_config = qwen2.ModelConfig.from_dict(normalized_arch)
                
                # Create forward pass with weights
                self.engine = qwen2.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                
                # Create KV cache
                self.kv_cache_at_layer = qwen2.create_kv_cache(
                    self.model_config, config
                )
            case "qwen3":
                # Create model config
                self.model_config = qwen3.ModelConfig.from_dict(normalized_arch)
                
                # Create forward pass with weights
                self.engine = qwen3.ForwardPass(
                    self.model_config,
                    config,
                    weights,
                )
                
                # Create KV cache
                self.kv_cache_at_layer = qwen3.create_kv_cache(
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

        # Initialize adapter states
        # Initialize adapter states
        self._init_adapter_states()


    # For backward compatibility, expose forward_pass as alias to engine
    @property
    def forward_pass(self) -> llama3.ForwardPass:
        """Backward compatibility alias for self.engine."""
        return self.engine

    def _init_adapter_states(self) -> None:
        """Initialize adapter memory tensors."""
        device = self.config.device

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.config.max_adapter_rank * 3,
                        self.model_config.dim_hidden,
                    ),
                    dtype=self.config.activation_dtype,
                    device=device,
                ),
                torch.zeros(
                    (
                        self.config.max_num_adapters,
                        self.model_config.dim_head
                        * (
                            self.model_config.num_q_heads
                            + self.model_config.num_kv_heads * 2
                        ),
                        self.config.max_adapter_rank,
                    ),
                    dtype=self.config.activation_dtype,
                    device=device,
                ),
            )
            for _ in range(self.model_config.num_layers)
        ]

    # ========================================================================
    # Metadata Accessors
    # ========================================================================

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "name": self.info.get("name", self.config.model),
            "description": self.info.get("description", ""),
            "version": self.info.get("version", "1.0.0"),
        }

    def get_chat_template(self) -> dict:
        """Get chat template configuration."""
        template = self.info.get("template", {})
        return {
            "template_type": template.get("type", "none"),
            "template_content": template.get("content", ""),
            "stop_tokens": template.get("stop_tokens", []),
        }

    def get_tokenizer(self) -> dict:
        """Get tokenizer configuration with merge table."""
        tokenizer_info = self.info.get("tokenizer", {})
        model_dir = Path(self.config.cache_dir) / "models" / self.config.model
        #print("model_dir", model_dir)
        # Get vocab file path
        vocab_filename = tokenizer_info.get("vocab") or tokenizer_info.get(
            "vocabulary_file"
        )
        if not vocab_filename:
            # Return minimal tokenizer info if no vocab file
            return {
                "type": tokenizer_info.get("type", "bpe"),
                "num_vocab": tokenizer_info.get(
                    "vocab_size", self.engine.weights.get("embed_token").shape[0]
                ),
                "merge_table": {},
                "split_regex": tokenizer_info.get("split_regex", ""),
                "special_tokens": tokenizer_info.get("special_tokens", {}),
                "escape_non_printable": tokenizer_info.get(
                    "escape_non_printable", False
                ),
            }

        vocab_file_path = model_dir / vocab_filename
        merge_rules: dict[int, bytes] = {}

        if vocab_file_path.exists():
            with open(vocab_file_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    b64_token, rank_str = parts
                    try:
                        decoded_token = base64.b64decode(b64_token)
                        rank = int(rank_str)
                        merge_rules[rank] = decoded_token
                    except (ValueError, TypeError):
                        continue

        return {
            "type": tokenizer_info.get("type", "bpe"),
            "num_vocab": tokenizer_info.get(
                "vocab_size", self.engine.weights.get("embed_token").shape[0]
            ),
            "merge_table": merge_rules,
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

    def heartbeat(
        self, reqs: list[message.HeartbeatRequest]
    ) -> list[message.HeartbeatResponse]:
        """Handle heartbeat keepalive requests."""
        return [message.HeartbeatResponse() for _ in reqs]

    def query(
        self, reqs: list[message.QueryRequest]
    ) -> list[message.QueryResponse]:
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

    def embed_image(self, reqs: list[message.EmbedImageRequest]) -> None:
        """Handle image embedding requests."""
        # TODO: implement image embedding
        pass

    def initialize_adapter(
        self, reqs: list[message.InitializeAdapterRequest]
    ) -> None:
        """Initialize adapter functionality."""
        for req in reqs:
            self._initialize_adapter(
                adapter_ptr=req.adapter_ptr,
                rank=req.rank,
                alpha=req.alpha,
                population_size=req.population_size,
                mu_fraction=req.mu_fraction,
                initial_sigma=req.initial_sigma,
            )

    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]) -> None:
        """Update adapter parameters."""
        for req in reqs:
            self._update_adapter(
                adapter_ptr=req.adapter_ptr,
                scores=req.scores,
                seeds=req.seeds,
                max_sigma=req.max_sigma,
            )

    def upload_adapter(self, reqs: list[message.UploadAdapterRequest]) -> None:
        """Upload adapter weights."""
        # TODO: implement adapter upload
        pass

    def download_adapter(
        self, reqs: list[message.DownloadAdapterRequest]
    ) -> list[message.DownloadAdapterResponse]:
        """Download adapter weights."""
        # TODO: implement adapter download
        return [message.DownloadAdapterResponse(adapter_data=b"") for _ in reqs]

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
        raise NotImplementedError

    @torch.inference_mode()
    def _update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ):
        raise NotImplementedError

    # ========================================================================
    # Batch Execution
    # ========================================================================

    @torch.inference_mode()
    def worker_loop(self):
        """
        Worker loop for ranks > 0.
        Waits for inputs from rank 0 and executes the model.
        """
        print(f"Worker {self.config.rank} started")
        device = self.config.device
        
        while True:
            # Wait for inputs
            objects = [None, None]
            dist.broadcast_object_list(objects, src=0)
            inputs, sampling_metadata = objects
            
            if inputs == "STOP":
                break
                
            if inputs is None:
                continue

            # Move inputs to device
            inputs = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            sampling_metadata = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in sampling_metadata.items()
            }
            
            # Execute step
            self._run_step(inputs, sampling_metadata)
            
        print(f"Worker {self.config.rank} finished")

    def _run_step(self, inputs: dict, sampling_metadata: dict) -> list:
        """
        Execute a single inference step (Embed -> Transform -> Sample).
        
        Returns:
            Sampling results (only valid on Rank 0 usually, but we return whatever comes out)
        """
        # 2. Embed inputs
        input_embeds = self.engine.embed_inputs(inputs)
        
        # 3. Run transformer forward pass
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
            adapter_subpass=inputs["adapter_subpass"],
        )
        
        # 4. Sampling Pass
        # Pass hidden_states (replicated or sharded? Transform returns replicated in current llama3.py)
        sampling_results = self.engine.sample(hidden_states, sampling_metadata)
        
        return sampling_results

    @torch.inference_mode()
    def _execute_batch(self) -> list[message.ForwardPassResponse]:
        """
        Execute the accumulated batch and return responses.
        
        Returns:
            List of ForwardPassResponse for each request in the batch
        """
        t_start = time.perf_counter()  
        batch = self.batch_builder.build()
        t_build = time.perf_counter()
        device = self.config.device#[self.config.rank]

        # 1. Prepare inputs using batch method
        inputs = batch.get_model_inputs(device)
        t_inputs = time.perf_counter()

        # Handle empty batch
        if not inputs["token_ids"]:
             if self.config.world_size > 1:
                  dist.broadcast_object_list([None, None], src=0)
             return []

        # Prepare sampling metadata
        sampling_metadata = batch.get_sampling_metadata(device, self.config.activation_dtype)

        # Broadcast if needed
        if self.config.world_size > 1:
            # Move to CPU for safe pickling/broadcasting
            cpu_inputs = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()
            }
            cpu_sampling_metadata = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v) 
                for k, v in sampling_metadata.items()
            }
            dist.broadcast_object_list([cpu_inputs, cpu_sampling_metadata], src=0)

        # Execute step
        sampling_results = self._run_step(inputs, sampling_metadata)
        
        t_forward_sample = time.perf_counter() # merged timing for simplicity

        # Package responses
        responses = batch.create_responses(sampling_results)
        t_package = time.perf_counter()

        print(f"Batch execution: {(t_package - t_start) * 1000:.2f} ms")
        print(f"  - Build: {(t_build - t_start) * 1000:.2f} ms")
        print(f"  - Inputs: {(t_inputs - t_build) * 1000:.2f} ms")
        print(f"  - Forward+Sample: {(t_forward_sample - t_inputs) * 1000:.2f} ms")
        print(f"  - Package: {(t_package - t_forward_sample) * 1000:.2f} ms")

        return responses


