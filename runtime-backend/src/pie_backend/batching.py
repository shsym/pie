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

from . import message
from .adapter import AdapterSubpass

if TYPE_CHECKING:
    from .runtime import Runtime


@dataclass
class Batch:
    """
    Holds the accumulated state for a specific inference step and handles packaging.
    
    This consolidates the state storage (formerly BatchState) and packaging logic
    (formerly ResponsePackager) into a single unified class.
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

    def get_model_inputs(self, device: torch.device) -> dict[str, Any]:
        """
        Finalize batch preparation and create input tensors for the model.
        
        Args:
            device: The torch device to create tensors on.
            
        Returns:
            Dictionary containing input tensors for the model engine.
        """
        # Create batched attention mask
        batched_attention_mask = (
            np.concatenate(self.attention_masks)
            if self.attention_masks
            else np.array([], dtype=np.bool_)
        )

        return {
            "token_ids": torch.as_tensor(self.token_ids, device=device, dtype=torch.long),
            "position_ids": torch.as_tensor(
                self.position_ids, device=device, dtype=torch.int32
            ),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lens, device=device, dtype=torch.int32
            ),
            "custom_mask": torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ),
            "single_token_inference_mode": self.single_token_mode,
            "single_token_inference_mode": self.single_token_mode,
            "adapter_indices": self.adapter_indices if self.adapter_subpass_needed else [],
            "adapter_seeds": torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long) if self.adapter_subpass_needed else None,
        }

    def get_sampling_metadata(self, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        """
        Prepare the metadata required for the SamplingPass.
        
        Args:
            device: Torch device.
            dtype: Torch dtype for temperatures.
            
        Returns:
            Dictionary containing sampling metadata.
        """
        # Return empty if no logits needed
        if not self.indices_for_logits:
            return {
                "indices_for_logits": None
            }

        indices_for_logits = self.indices_for_logits
        
        temperatures = torch.tensor(
            [p["temperature"] for p in self.sampler_params],
            device=device,
            dtype=dtype,
        ).clamp(min=1e-6).unsqueeze(1)

        # Group samplers
        sampler_groups: dict[int, list[int]] = {}
        for i, sampler_idx in enumerate(self.sampler_types):
            if sampler_idx not in sampler_groups:
                sampler_groups[sampler_idx] = []
            sampler_groups[sampler_idx].append(i)

        return {
            "indices_for_logits": indices_for_logits,
            "temperatures": temperatures,
            "sampler_groups": sampler_groups,
            "sampler_params": self.sampler_params,
        }

    def create_responses(
        self, sampling_results: dict[str, Any]
    ) -> list[message.ForwardPassResponse]:
        """
        Package the sampling results into responses for each original request.
        
        Args:
            sampling_results: Dictionary containing 'tokens' and 'dists'.
            
        Returns:
            List of responses in the order of requests.
        """
        # Early return if no logits needed
        if not self.indices_for_logits:
            return [
                message.ForwardPassResponse(dists=[], tokens=[])
                for _ in self.requests
            ]

        final_dists = sampling_results["dists"]
        final_tokens_list = sampling_results["tokens"]
        
        responses = []
        cursor = 0

        for req in self.requests:
            output_token_indices = req.output_token_indices or []
            num_outputs = len(output_token_indices)
            request_dists = []
            request_tokens = []

            for i in range(cursor, cursor + num_outputs):
                if self.sampler_types[i] == 0:
                    # Distribution request
                    if final_dists[i] is not None:
                        request_dists.append(final_dists[i])
                else:
                    # Sampling request
                    request_tokens.append(final_tokens_list[i])


            responses.append(
                message.ForwardPassResponse(dists=request_dists, tokens=request_tokens)
            )
            cursor += num_outputs

        return responses


class BatchBuilder:
    """
    Responsible for converting incoming ForwardPassRequests into a Batch.
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
        self.current_batch: Batch | None = None

    def reset(self) -> None:
        """Reset the builder for a new batch."""
        self.current_batch = Batch()

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

    def build(self) -> Batch:
        """
        Return the prepared batch and reset the builder.
        
        Returns:
            The completed Batch ready for inference
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