"""Dummy driver config — minimal knobs for the random-token driver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DummyDriverConfig:
    """Knobs for the `dummy` driver. Most fields mirror NativeDriverConfig
    so the dummy can stand in for native in tests."""

    # Memory fraction for KV cache (small allocations even in dummy mode
    # so the scheduler has real block IDs to play with).
    gpu_mem_utilization: float = 0.2

    max_batch_tokens: int = 4096
    max_batch_size: int = 128
    max_dist_size: int = 32
    max_num_embeds: int = 128
    max_num_adapters: int = 16
    max_adapter_rank: int = 8
    kv_page_size: int = 16
    weight_dtype: str = "bfloat16"
    cpu_mem_budget_in_gb: int = 0
