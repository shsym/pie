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
from typing import TYPE_CHECKING

import numpy as np
import torch

from .config import RuntimeConfig
from .batching import BatchBuilder, BatchState, ResponsePackager
from .loader import ModelLoader
from .adapter import AdapterSubpass
from .model import llama3, qwen2, qwen3
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
    batch: BatchState | None

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
    def _execute_batch(self) -> list[message.ForwardPassResponse]:
        """
        Execute the accumulated batch and return responses.
        
        Returns:
            List of ForwardPassResponse for each request in the batch
        """
        batch = self.batch_builder.build()
        device = self.config.device#[self.config.rank]

        if not batch.token_ids:
            return []

        # 1. Prepare adapter subpass if needed
        adapter_subpass = None
        if batch.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(
                batch.adapter_seeds, device=device, dtype=torch.long
            )
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self.adapter_at_layer,
                adapter_indices=batch.adapter_indices,
                adapter_extras=self.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=batch.qo_indptr,
            )

        # 2. Prepare attention mask
        batched_attention_mask = (
            np.concatenate(batch.attention_masks)
            if batch.attention_masks
            else np.array([], dtype=np.bool_)
        )

        # 3. Prepare input tensors
        token_ids_tensor = torch.as_tensor(
            batch.token_ids, device=device, dtype=torch.int32
        )
        input_embeds = self.engine.embed_tokens(token_ids=token_ids_tensor)

        # 4. Run transformer forward pass
        hidden_states = self.engine.transform(
            input_embeds=input_embeds,
            position_ids=torch.as_tensor(
                batch.position_ids, device=device, dtype=torch.int32
            ),
            qo_indptr=torch.as_tensor(batch.qo_indptr, device=device, dtype=torch.int32),
            kv_cache_at_layer=self.kv_cache_at_layer,
            kv_page_indices=torch.as_tensor(
                batch.kv_page_indices, device=device, dtype=torch.int32
            ),
            kv_page_indptr=torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            kv_last_page_lens=torch.as_tensor(
                batch.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            custom_mask=torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            single_token_inference_mode=batch.single_token_mode,
            adapter_subpass=adapter_subpass,
        )

        # 5. Compute logits and sample
        return self._sample_and_format_response(hidden_states, batch, device)

    def _sample_and_format_response(
        self,
        hidden_states: torch.Tensor,
        batch: BatchState,
        device: torch.device,
    ) -> list[message.ForwardPassResponse]:
        """
        Sample from logits and format responses.
        
        Args:
            hidden_states: Output from transformer
            batch: The batch state with request info
            device: Target device
            
        Returns:
            List of ForwardPassResponse for each request
        """
        # Get logits for positions that need them
        if batch.indices_for_logits:
            logits_indices = torch.as_tensor(
                batch.indices_for_logits, device=device, dtype=torch.long
            )
            selected_hidden = hidden_states[logits_indices]
            logits = self.engine.lm_head(selected_hidden)
        else:
            logits = None

        if logits is None:
            # No logits requested, return empty responses
            return [
                message.ForwardPassResponse(tokens=[], dists=[])
                for _ in batch.requests
            ]

        # Apply temperature scaling
        temperatures = torch.tensor(
            [p["temperature"] for p in batch.sampler_params],
            device=device,
            dtype=logits.dtype,
        ).unsqueeze(1)
        scaled_logits = logits / temperatures.clamp(min=1e-6)

        # Group requests by sampler type
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(batch.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        # Sample tokens
        probs = torch.softmax(scaled_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Build responses for each request
        responses: list[message.ForwardPassResponse] = []
        cursor = 0

        for req in batch.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists: list[tuple[list[int], list[float]]] = []
            request_tokens: list[int] = []

            # Iterate through the slice of results belonging to this request
            for i in range(cursor, cursor + num_outputs):
                if i < len(batch.sampler_types):
                    if batch.sampler_types[i] == 0:
                        # Distribution request - return top-k
                        top_k = batch.sampler_params[i].get("top_k", 10)
                        token_probs = probs[i]
                        top_probs, top_indices = torch.topk(token_probs, k=min(top_k, token_probs.shape[0]))
                        request_dists.append(
                            (top_indices.cpu().tolist(), top_probs.cpu().tolist())
                        )
                    else:
                        # Sampling request
                        request_tokens.append(sampled_tokens[i].item())

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses
