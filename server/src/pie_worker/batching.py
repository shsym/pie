"""
Batch management for PIE backend inference.

This module provides the Batch dataclass that holds inference batch state
and handles tensor creation and response packaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import message


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

        self.adapter_subpass_needed = False
        # Create batched attention mask
        batched_attention_mask = (
            np.concatenate(self.attention_masks)
            if self.attention_masks
            else np.array([], dtype=np.bool_)
        )

        return {
            "token_ids": torch.as_tensor(
                self.token_ids, device=device, dtype=torch.long
            ).contiguous(),
            "position_ids": torch.as_tensor(
                self.position_ids, device=device, dtype=torch.int32
            ).contiguous(),
            "qo_indptr": torch.as_tensor(
                self.qo_indptr, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_page_indices": torch.as_tensor(
                self.kv_page_indices, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_page_indptr": torch.as_tensor(
                self.kv_page_indptr, device=device, dtype=torch.int32
            ).contiguous(),
            "kv_last_page_lens": torch.as_tensor(
                self.kv_last_page_lens, device=device, dtype=torch.int32
            ).contiguous(),
            "custom_mask": torch.as_tensor(
                batched_attention_mask, device=device, dtype=torch.bool
            ).contiguous(),
            "single_token_inference_mode": self.single_token_mode,
            "adapter_indices": (
                self.adapter_indices if self.adapter_subpass_needed else []
            ),
            "adapter_seeds": (
                torch.as_tensor(self.adapter_seeds, device=device, dtype=torch.long)
                if self.adapter_subpass_needed
                else None
            ),
            "total_pages_cpu": self.kv_page_indptr[-1],
        }

    def get_sampling_metadata(
        self, device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
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
            return {"indices_for_logits": None}

        indices_for_logits = self.indices_for_logits

        temperatures = (
            torch.tensor(
                [p["temperature"] for p in self.sampler_params],
                device=device,
                dtype=dtype,
            )
            .clamp(min=1e-6)
            .unsqueeze(1)
        )

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
                message.ForwardPassResponse(dists=[], tokens=[]) for _ in self.requests
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


def _decode_brle(brle_buffer) -> np.ndarray:
    """
    Decode a Binary Run-Length Encoded buffer into a boolean numpy array.

    The format assumes alternating runs of True and False, starting with True
    (in attention masking, True means attend).

    Args:
        brle_buffer: List of run lengths or bytes (u32 little-endian encoded)

    Returns:
        Decoded boolean array
    """
    # Handle bytes input (from FFI - u32 little-endian encoded)
    if isinstance(brle_buffer, bytes):
        brle_buffer = np.frombuffer(brle_buffer, dtype=np.uint32).tolist()

    if not brle_buffer:
        return np.array([], dtype=bool)

    # Hybrid approach: Iterative loop is faster for small buffers (most decoding steps)
    # NumPy vectorization is 10x faster for large buffers (complex prefills)
    if len(brle_buffer) < 16:
        total_size = sum(brle_buffer)
        decoded_array = np.empty(total_size, dtype=bool)
        current_pos = 0
        value = True
        for run_len in brle_buffer:
            if run_len > 0:
                decoded_array[current_pos : current_pos + run_len] = value
            current_pos += run_len
            value = not value
        return decoded_array
    else:
        pattern = np.empty(len(brle_buffer), dtype=bool)
        pattern[::2] = True
        pattern[1::2] = False
        return np.repeat(pattern, brle_buffer)
