"""
Common modeling components for PIE backend.
"""

from __future__ import annotations

import torch
from typing import Callable, Any

from flashinfer.sampling import (
    sampling_from_probs,
    top_p_sampling_from_probs,
    top_k_sampling_from_probs,
    min_p_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
)

def sample_common(
    hidden_states: torch.Tensor,
    sampling_metadata: dict,
    lm_head_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """
    Execute the sampling pass.
    
    Args:
        hidden_states: The output hidden states from the model.
        sampling_metadata: Dictionary containing prepared sampling metadata.
        lm_head_fn: Function to compute logits from hidden states.
        device: Torch device.
        dtype: Torch dtype for intermediate calculations.
        
    Returns:
        Dictionary containing 'tokens' (list[int]) and 'dists' (list[tuple]).
    """
    if not sampling_metadata.get("indices_for_logits"):
        return {"tokens": [], "dists": []}

    indices_for_logits = sampling_metadata["indices_for_logits"]
    
    # Stage 1: Compute logits via LM head
    logits_input = hidden_states[indices_for_logits]
    logits = lm_head_fn(logits_input)

    # Stage 2: Apply temperature scaling
    temperatures = sampling_metadata["temperatures"]
    scaled_logits = logits / torch.clamp(temperatures, min=1e-6)

    # Stage 3: Compute probabilities
    probs = torch.softmax(scaled_logits, dim=-1)

    # Stage 4: Execute sampling for each group
    num_logit_requests = len(indices_for_logits)
    final_dists = [None] * num_logit_requests
    final_tokens_tensor = torch.empty(
        num_logit_requests, dtype=torch.long, device=device
    )
    
    sampler_groups = sampling_metadata["sampler_groups"]
    sampler_params = sampling_metadata["sampler_params"]

    for sampler_idx, indices in sampler_groups.items():
        if not indices:
            continue

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_probs = probs.index_select(0, indices_tensor)

        if sampler_idx == 0:
            # Distribution mode
            _process_distributions(
                indices, group_probs, final_dists, sampler_params
            )
        else:
            #print(sampler_idx, indices, group_probs, sampler_params, device, dtype)
            # Sampling mode
            sampled = _execute_sampler(
                sampler_idx, indices, group_probs, sampler_params, device, dtype
            )
            if sampled.dtype != torch.long:
                sampled = sampled.to(torch.long)

            # check if sampled exceeds 10000000
            # if sampled.max() > 10000000:
            #     print("client-sent token id {} is out of range".format(sampled))
            final_tokens_tensor.scatter_(0, indices_tensor, sampled)

    # Stage 5: Combine results
    final_tokens_list = final_tokens_tensor.tolist()
    
    return {
        "tokens": final_tokens_list,
        "dists": final_dists 
    }

def _process_distributions(
    indices: list[int],
    group_probs: torch.Tensor,
    final_dists: list[tuple[list[int], list[float]] | None],
    sampler_params: list[dict],
) -> None:
    """Process distribution requests."""
    # Note: sampler_params is a list corresponding to the WHOLE batch logit requests
    # We need to map `indices` (which are indices into the logit batch) back to param access.
    
    group_top_k = [sampler_params[i]["top_k"] for i in indices]
    max_k = max(group_top_k) if group_top_k else 0

    if max_k > 0:
        topk_vals, topk_inds = torch.topk(group_probs, k=max_k, sorted=True)
        
        topk_vals_list = topk_vals.tolist()
        topk_inds_list = topk_inds.tolist()
        
        for i, original_idx in enumerate(indices):
            k = sampler_params[original_idx]["top_k"]
            ids = topk_inds_list[i][:k]
            vals = topk_vals_list[i][:k]
            final_dists[original_idx] = (ids, vals)

def _execute_sampler(
    sampler_idx: int,
    indices: list[int],
    group_probs: torch.Tensor,
    sampler_params: list[dict],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Execute the appropriate sampling operation."""
    
    # Gather params for this group
    group_params = [sampler_params[i] for i in indices]

    if sampler_idx == 1:
        return sampling_from_probs(group_probs)

    elif sampler_idx == 2:
        top_p_vals = torch.tensor(
            [p["top_p"] for p in group_params],
            device=device,
            dtype=dtype,
        )
        return top_p_sampling_from_probs(group_probs, top_p=top_p_vals)

    elif sampler_idx == 3:
        top_k_vals = torch.tensor(
            [p["top_k"] for p in group_params],
            device=device,
            dtype=torch.long,
        )
        return top_k_sampling_from_probs(group_probs, top_k=top_k_vals)

    elif sampler_idx == 4:
        min_p_vals = torch.tensor(
            [p["min_p"] for p in group_params],
            device=device,
            dtype=dtype,
        )
        return min_p_sampling_from_probs(group_probs, min_p=min_p_vals)

    elif sampler_idx == 5:
        top_k_vals = torch.tensor(
            [p["top_k"] for p in group_params],
            device=device,
            dtype=torch.long,
        )
        top_p_vals = torch.tensor(
            [p["top_p"] for p in group_params],
            device=device,
            dtype=dtype,
        )
        return top_k_top_p_sampling_from_probs(
            group_probs, top_k=top_k_vals, top_p=top_p_vals
        )

    else:
        raise ValueError(f"Unknown sampler index: {sampler_idx}")
