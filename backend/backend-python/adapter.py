from __future__ import annotations

import time
import torch
import torch.nn as nn
import math


def run_length_encode(data: list[int]) -> list[tuple[int, int]]:
    """
    Perform run-length encoding on a list of integers.

    Args:
        data (List[int]): The input list of integers.

    Returns:
        List[Tuple[int, int]]: A list of (value, count) pairs.
    """
    if not data:
        return []

    encoded = []
    current_value = data[0]
    count = 1

    for num in data[1:]:
        if num == current_value:
            count += 1
        else:
            encoded.append((current_value, count))
            current_value = num
            count = 1

    # Append the last run
    encoded.append((current_value, count))

    return encoded


class AdapterSubpass:

    def __init__(
        self,
        adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]],
        adapter_indices: list[int],
        adapter_extras: dict[int, Adapter],
        rand_seeds: torch.Tensor,
        qo_indptr: list[int],
    ):
        self.adapter_at_layer = adapter_at_layer
        self.adapter_indices = adapter_indices
        self.adapter_extras = adapter_extras
        self.rand_seeds = rand_seeds
        self.adapter_indices_rle = run_length_encode(self.adapter_indices)
        self.qo_indptr = qo_indptr

    def execute(
        self,
        layer_idx: int,
        xs: torch.Tensor,
        q_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
    ):
        i = 0
        for adapter_index, count in self.adapter_indices_rle:

            x_start = self.qo_indptr[i]
            x_end = self.qo_indptr[i + count]
            x = xs[x_start:x_end]

            rand_seeds = self.rand_seeds[i : i + count]
            inject_noise = rand_seeds.any().item()

            assert x.shape[0] == rand_seeds.shape[0], "Batch size must match seeds."

            # DOWN noise uses 3 equal chunks of size `rank` each (Q/K/V).
            Wd = self.adapter_at_layer[layer_idx][0][adapter_index]
            Wu = self.adapter_at_layer[layer_idx][1][
                adapter_index
            ]  # (rank, d_q+d_k+d_v)
            adapter_info = self.adapter_extras[adapter_index]

            rank = adapter_info.rank
            out_indptr = adapter_info.out_features_indptr  # built from [d_q, d_k, d_v]

            qkv_down = x @ Wd
            d_q, d_k, d_v = torch.split(qkv_down, [rank, rank, rank], dim=-1)

            Wu_q = Wu[:, out_indptr[0] : out_indptr[1]]  # (rank, d_q)
            Wu_k = Wu[:, out_indptr[1] : out_indptr[2]]  # (rank, d_k)
            Wu_v = Wu[:, out_indptr[2] : out_indptr[3]]  # (rank, d_v)

            u_q = d_q @ Wu_q
            u_k = d_k @ Wu_k
            u_v = d_v @ Wu_v

            # ===== 3) Combine mean + noise =====
            scaling = adapter_info.alpha / float(rank)

            q_final = scaling * u_q
            k_final = scaling * u_k
            v_final = scaling * u_v

            q_state[x_start:x_end].add_(q_final)
            k_state[x_start:x_end].add_(k_final)
            v_state[x_start:x_end].add_(v_final)

            i += count


class Adapter:
    """
    Abstract base class defining the interface for a LoRA-style adapter.
    """

    adapter_id: int
    rank: int
    alpha: float
    out_features: list[int]
    out_features_indptr: list[int]

    def __init__(
        self, adapter_id: int, rank: int, alpha: float, out_features: list[int]
    ):
        self.adapter_id = adapter_id
        self.rank = rank
        self.alpha = alpha
        self.out_features = out_features

        self.out_features_indptr = [0]
        for i in range(len(out_features)):
            self.out_features_indptr.append(
                self.out_features_indptr[-1] + out_features[i]
            )
