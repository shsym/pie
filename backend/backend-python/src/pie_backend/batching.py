"""
Batch management for PIE backend inference.

This module encapsulates the complex logic of converting requests into
tensors for batched inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from flashinfer.sampling import (
            sampling_from_probs,
            top_p_sampling_from_probs,
            top_k_sampling_from_probs,
            min_p_sampling_from_probs,
            top_k_top_p_sampling_from_probs,
        )
        
from . import message

if TYPE_CHECKING:
    from .runtime import Runtime


@dataclass
class BatchState:
    """
    Holds the accumulated state for a specific inference step.
    
    This replaces the original SyncBatch dataclass with cleaner naming
    and explicit tracking of the original requests for response mapping.
    """

    # Input tokens and positions
    token_ids: list[int] = field(default_factory=list)
    position_ids: list[int] = field(default_factory=list)

    # KV Cache Layout
    kv_page_indices: list[int] = field(default_factory=list)
    kv_page_indptr: list[int] = field(default_factory=lambda: [0])
    kv_last_page_lens: list[int] = field(default_factory=list)

    # Query/Output indirection pointers
    qo_indptr: list[int] = field(default_factory=lambda: [0])

    # Attention masks (one per request, flattened later)
    attention_masks: list[np.ndarray] = field(default_factory=list)

    # Adapters
    adapter_indices: list[int] = field(default_factory=list)
    adapter_seeds: list[int] = field(default_factory=list)
    adapter_subpass_needed: bool = False

    # Output mapping for logits and embeddings
    indices_for_logits: list[int] = field(default_factory=list)
    indices_for_embed_storage: list[int] = field(default_factory=list)
    embed_storage_pointers: list[int] = field(default_factory=list)

    # Sampler configuration
    sampler_types: list[int] = field(default_factory=list)
    sampler_params: list[dict] = field(default_factory=list)

    # Metadata for tracking and reconstruction
    requests: list[message.ForwardPassRequest] = field(default_factory=list)
    total_tokens: int = 0
    single_token_mode: bool = True


class BatchBuilder:
    """
    Responsible for converting incoming ForwardPassRequests into a BatchState.
    
    This encapsulates the complex batch construction logic that was previously
    scattered across the Runtime class.
    """

    def __init__(self, kv_page_size: int, max_dist_size: int, adapters: dict):
        """
        Initialize the BatchBuilder.
        
        Args:
            kv_page_size: Size of each KV cache page
            max_dist_size: Maximum distribution size for sampling
            adapters: Dictionary of available adapters
        """
        self.kv_page_size = kv_page_size
        self.max_dist_size = max_dist_size
        self.adapters = adapters
        self.current_batch: BatchState | None = None

    def reset(self) -> None:
        """Reset the builder for a new batch."""
        self.current_batch = BatchState()

    def add_request(self, req: message.ForwardPassRequest) -> None:
        """
        Process a single request and append to the current batch state.
        
        Args:
            req: Forward pass request to add to the batch
        """
        if self.current_batch is None:
            self.reset()

        batch = self.current_batch
        batch.requests.append(req)

        # 1. Handle adapter information
        if req.adapter is not None and req.adapter in self.adapters:
            seed = req.adapter_seed if req.adapter_seed is not None else 0
            batch.adapter_seeds.extend([seed] * len(req.input_tokens))
            batch.adapter_indices.append(req.adapter)
            batch.adapter_subpass_needed = True

        # 2. Handle KV cache pages
        batch.kv_page_indices.extend(req.kv_page_ptrs)
        batch.kv_page_indptr.append(len(batch.kv_page_indices))
        batch.kv_last_page_lens.append(req.kv_page_last_len or 0)

        # 3. Handle output mappings for embeddings
        if len(req.output_embed_indices) != len(req.output_embed_ptrs):
            raise ValueError(
                f"Mismatch between output_embed_indices length ({len(req.output_embed_indices)}) "
                f"and output_embed_ptrs length ({len(req.output_embed_ptrs)})"
            )
        for token_idx, storage_ptr in zip(req.output_embed_indices, req.output_embed_ptrs):
            batch.indices_for_embed_storage.append(token_idx + batch.total_tokens)
            batch.embed_storage_pointers.append(storage_ptr)

        # 4. Handle output mappings for tokens requiring logits
        for token_idx in req.output_token_indices:
            batch.indices_for_logits.append(token_idx + batch.total_tokens)

        # 5. Extract sampler configurations
        for sampler_config in req.output_token_samplers:
            params = {}
            sampler_idx = sampler_config["sampler"]
            batch.sampler_types.append(sampler_idx)

            if sampler_idx == 0:
                # Distribution sampler
                params["top_k"] = min(
                    sampler_config.get("top_k", self.max_dist_size),
                    self.max_dist_size,
                )
            else:
                # Other samplers
                params["top_k"] = sampler_config.get("top_k", 0)
                params["top_p"] = sampler_config.get("top_p", 1.0)
                params["min_p"] = sampler_config.get("min_p", 0.0)

            params["temperature"] = sampler_config.get("temperature", 1.0)
            batch.sampler_params.append(params)

        # 6. Handle input tokens and positions
        batch.token_ids.extend(req.input_tokens)
        batch.position_ids.extend(req.input_token_positions)
        batch.total_tokens += len(req.input_tokens)
        batch.qo_indptr.append(batch.total_tokens)

        if len(req.input_tokens) > 1:
            batch.single_token_mode = False

        # 7. Generate attention mask
        attention_mask = _generate_mask_for_request(
            input_tokens=req.input_tokens,
            mask=req.mask,
            kv_page_ptrs=req.kv_page_ptrs,
            kv_page_last_len=req.kv_page_last_len,
            kv_page_size=self.kv_page_size,
        )
        batch.attention_masks.append(attention_mask)

    def build(self) -> BatchState:
        """
        Return the prepared batch and reset the builder.
        
        Returns:
            The completed BatchState ready for inference
        """
        batch = self.current_batch
        self.reset()
        return batch

    def is_empty(self) -> bool:
        """Check if the current batch is empty."""
        return self.current_batch is None or len(self.current_batch.token_ids) == 0


def _generate_mask_for_request(
    input_tokens: list[int],
    mask: list[list[int]],
    kv_page_ptrs: list[int],
    kv_page_last_len: int,
    kv_page_size: int,
) -> np.ndarray:
    """
    Generate the custom attention mask for a single request.
    
    Args:
        input_tokens: List of input token IDs
        mask: BRLE-encoded attention masks for each token
        kv_page_ptrs: KV cache page pointers
        kv_page_last_len: Length of the last KV cache page
        kv_page_size: Size of each KV cache page
        
    Returns:
        Flattened boolean attention mask array
    """
    if len(mask) != len(input_tokens):
        raise ValueError(
            f"Mismatch between number of masks ({len(mask)}) and "
            f"input tokens ({len(input_tokens)})."
        )

    # Ensure we have at least one page for proper computation
    if len(kv_page_ptrs) >= 1:
        sequence_length = kv_page_size * (len(kv_page_ptrs) - 1) + kv_page_last_len
    else:
        sequence_length = kv_page_last_len

    # Validate sequence_length is sufficient for input tokens
    input_token_count = len(input_tokens)
    if sequence_length < input_token_count:
        raise ValueError(
            f"Insufficient sequence length ({sequence_length}) for input tokens "
            f"({input_token_count}). Sequence length must be at least equal to "
            f"the number of input tokens."
        )

    context_length = sequence_length - input_token_count

    request_attention_mask = np.zeros(
        (len(input_tokens), sequence_length), dtype=np.bool_
    )
    for i, brle_buffer in enumerate(mask):
        decoded_mask = _decode_brle(brle_buffer)
        expected_len = context_length + i + 1
        if len(decoded_mask) != expected_len:
            raise ValueError(
                f"Decoded mask for token {i} has length {len(decoded_mask)}, "
                f"but expected {expected_len}"
            )
        request_attention_mask[i, :expected_len] = decoded_mask

    return request_attention_mask.flatten()


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decode a Binary Run-Length Encoded buffer into a boolean numpy array.
    
    The format assumes alternating runs of True and False, starting with True
    (in attention masking, True means attend).
    
    Args:
        brle_buffer: List of run lengths
        
    Returns:
        Decoded boolean array
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # In attention masking, True means attend.
    for run_len in brle_buffer:
        if run_len > 0:
            decoded_array[current_pos : current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array


class ResponsePackager:
    """
    Packages model outputs into responses for each original request.
    
    This class handles the post-inference stages:
    - Embedding storage
    - Logits computation via LM head
    - Temperature scaling
    - Probability computation (softmax)
    - Grouped sampling operations
    - Response distribution back to original requests
    """

    def __init__(self, runtime: Any, batch: BatchState):
        """
        Initialize the response packager.
        
        Args:
            runtime: The Runtime instance providing model and device access.
                     Expected attributes:
                     - config.device[config.rank]: torch.device
                     - config.activation_dtype: torch.dtype
                     - engine.lm_head(x): logits projection
                     - embeds: dict for embedding storage (optional)
                     - ops.sampling: sampling operations (optional, uses fallback)
            batch: The BatchState containing request metadata and sampler configs
        """
        self.runtime = runtime
        self.batch = batch
        
        # Resolve device and dtype from runtime.config
        self.device = runtime.config.device#[runtime.config.rank]
        self.dtype = runtime.config.activation_dtype
        self.logits_dtype = getattr(runtime, "logits_dtype", self.dtype)

    def finalize(self) -> dict[str, Any]:
        """
        Finalize batch preparation, creating tensors and the adapter subpass.
        
        This method prepares the input dictionary for the model forward pass.
        
        Returns:
            Dictionary containing all tensors needed for inference
        """
        device = self.device
        batch = self.batch

        # Setup adapter subpass if needed
        adapter_subpass = None
        if batch.adapter_subpass_needed:
            from .adapters import ensure_adapter_available
            adapter_subpass_class = ensure_adapter_available()
            seeds_tensor = torch.as_tensor(
                batch.adapter_seeds, device=device, dtype=torch.long
            )
            adapter_subpass = adapter_subpass_class(
                adapter_at_layer=self.runtime.adapter_at_layer,
                adapter_indices=batch.adapter_indices,
                adapter_extras=self.runtime.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=batch.qo_indptr,
            )

        # Create batched attention mask
        batched_attention_mask = (
            np.concatenate(batch.attention_masks)
            if batch.attention_masks
            else np.array([], dtype=np.bool_)
        )

        # Create token tensor and get embeddings
        token_ids_tensor = torch.as_tensor(
            batch.token_ids, device=device, dtype=torch.int32
        )
        embed_tokens = self.runtime.lm.model.embed_tokens  # type: ignore[attr-defined]
        input_embeds = embed_tokens(token_ids_tensor)  # type: ignore[operator]

        return {
            "input_embeds": input_embeds,
            "position_ids": torch.as_tensor(
                batch.position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                batch.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                batch.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                batch.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                batch.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": batch.single_token_mode,
            "adapter_subpass": adapter_subpass,
        }

    def package_responses(
        self, output_embeds: torch.Tensor
    ) -> list[message.ForwardPassResponse]:
        """
        Package the model outputs into responses for each original request.
        
        This method performs:
        1. Embedding storage for specified indices
        2. Logits computation via LM head
        3. Temperature scaling
        4. Softmax probability computation
        5. Grouped sampling operations
        6. Response distribution back to original requests
        
        Args:
            output_embeds: Output embeddings from the model forward pass
            
        Returns:
            List of ForwardPassResponse objects, one per original request
        """
        batch = self.batch
        device = self.device

        # Stage 1: Store specified embeddings (if embed storage is available)
        if batch.indices_for_embed_storage and hasattr(self.runtime, 'embeds'):
            embeddings_to_store = output_embeds[batch.indices_for_embed_storage]
            for i, ptr in enumerate(batch.embed_storage_pointers):
                self.runtime.embeds[ptr].copy_(
                    embeddings_to_store[i], non_blocking=True
                )

        # Early return if no logits needed
        if not batch.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in batch.requests
            ]

        # Stage 2: Compute logits via LM head
        logits_input = output_embeds[batch.indices_for_logits]
        if logits_input.dtype != self.logits_dtype:
            logits_input = logits_input.to(self.logits_dtype)

        logits = self.runtime.engine.lm_head(logits_input)

        # Promote logits to handler dtype for numerically stable softmax
        if logits.dtype != self.logits_dtype:
            logits = logits.to(dtype=self.logits_dtype)

        # Stage 3: Apply temperature scaling
        temperatures = torch.tensor(
            [p["temperature"] for p in batch.sampler_params],
            device=device,
            dtype=self.logits_dtype,
        ).unsqueeze(1)
        scaled_logits = logits / torch.clamp(temperatures, min=1e-6)

        # Stage 4: Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1)

        if not torch.isfinite(probs).all():
            raise RuntimeError("Non-finite probabilities produced by LM head")

        # Stage 5: Group requests by sampler type for efficient batch processing
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(batch.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        num_logit_requests = len(batch.indices_for_logits)
        final_dists: list[tuple[list[int], list[float]] | None] = [None] * num_logit_requests
        final_tokens_tensor = torch.empty(
            num_logit_requests, dtype=torch.long, device=device
        )

        # Stage 6: Execute sampling for each group
        for sampler_idx, indices in sampler_groups.items():
            if not indices:
                continue

            indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            group_probs = probs.index_select(0, indices_tensor)

            if sampler_idx == 0:
                # Distribution mode: return top-k token IDs and probabilities
                self._process_distributions(
                    indices, group_probs, final_dists
                )
            else:
                # Sampling mode: sample tokens using the specified sampler
                sampled = self._execute_sampler(
                    sampler_idx, indices, group_probs
                )
                if sampled.dtype != torch.long:
                    sampled = sampled.to(torch.long)
                final_tokens_tensor.scatter_(0, indices_tensor, sampled)

        # Stage 7: Distribute results back to individual responses
        return self._build_responses(final_dists, final_tokens_tensor)

    def _process_distributions(
        self,
        indices: list[int],
        group_probs: torch.Tensor,
        final_dists: list[tuple[list[int], list[float]] | None],
    ) -> None:
        """
        Process distribution requests, computing top-k values and indices.
        
        Args:
            indices: Indices of requests in this group
            group_probs: Probabilities for this group
            final_dists: Output list to populate with (token_ids, probs) tuples
        """
        batch = self.batch
        group_top_k = [batch.sampler_params[i]["top_k"] for i in indices]
        max_k = max(group_top_k) if group_top_k else 0

        if max_k > 0:
            topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)
            for i, original_idx in enumerate(indices):
                k = batch.sampler_params[original_idx]["top_k"]
                ids = topk_inds[i, :k].tolist()
                vals = topk_vals[i, :k].tolist()
                final_dists[original_idx] = (ids, vals)

    def _execute_sampler(
        self,
        sampler_idx: int,
        indices: list[int],
        group_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute the appropriate sampling operation.
        
        Sampler indices:
            1: sampling_from_probs (basic sampling)
            2: top_p_sampling_from_probs
            3: top_k_sampling_from_probs
            4: min_p_sampling_from_probs
            5: top_k_top_p_sampling_from_probs (combined)
        
        Args:
            sampler_idx: Index identifying the sampler type
            indices: Indices of requests in this group
            group_probs: Probabilities for this group
            
        Returns:
            Tensor of sampled token indices
        """
        
        
        batch = self.batch
        device = self.device

        if sampler_idx == 1:
            # Basic sampling from probabilities
            return sampling_from_probs(group_probs)

        elif sampler_idx == 2:
            # Top-p (nucleus) sampling
            top_p_vals = torch.tensor(
                [batch.sampler_params[i]["top_p"] for i in indices],
                device=device,
                dtype=self.dtype,
            )
            return top_p_sampling_from_probs(group_probs, top_p=top_p_vals)

        elif sampler_idx == 3:
            # Top-k sampling
            top_k_vals = torch.tensor(
                [batch.sampler_params[i]["top_k"] for i in indices],
                device=device,
                dtype=torch.long,
            )
            return top_k_sampling_from_probs(group_probs, top_k=top_k_vals)

        elif sampler_idx == 4:
            # Min-p sampling
            min_p_vals = torch.tensor(
                [batch.sampler_params[i]["min_p"] for i in indices],
                device=device,
                dtype=self.dtype,
            )
            return min_p_sampling_from_probs(group_probs, min_p=min_p_vals)

        elif sampler_idx == 5:
            # Combined top-k + top-p sampling
            top_k_vals = torch.tensor(
                [batch.sampler_params[i]["top_k"] for i in indices],
                device=device,
                dtype=torch.long,
            )
            top_p_vals = torch.tensor(
                [batch.sampler_params[i]["top_p"] for i in indices],
                device=device,
                dtype=self.dtype,
            )
            return top_k_top_p_sampling_from_probs(
                group_probs, top_k=top_k_vals, top_p=top_p_vals
            )

        else:
            raise ValueError(f"Unknown sampler index: {sampler_idx}")

    def _build_responses(
        self,
        final_dists: list[tuple[list[int], list[float]] | None],
        final_tokens_tensor: torch.Tensor,
    ) -> list[message.ForwardPassResponse]:
        """
        Build response objects by distributing results back to original requests.
        
        Args:
            final_dists: List of distribution results (token_ids, probs) or None
            final_tokens_tensor: Tensor of sampled tokens
            
        Returns:
            List of ForwardPassResponse objects
        """
        batch = self.batch
        responses = []
        cursor = 0

        for req in batch.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            for i in range(cursor, cursor + num_outputs):
                if batch.sampler_types[i] == 0:
                    # Distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:
                    # Sampling request
                    request_tokens.append(final_tokens_tensor[i].item())

            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses
