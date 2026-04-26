"""vLLM driver config — typed view of a `vllm.engine.arg_utils.EngineArgs`
subset. Field names mirror EngineArgs exactly so values flow verbatim:

    [model.X.driver.vllm]
    attention_backend = "FLASHINFER"      → EngineArgs.attention_backend
    enforce_eager = false                 → EngineArgs.enforce_eager
    gpu_memory_utilization = 0.85         → EngineArgs.gpu_memory_utilization
    ...

Adding a new vllm knob: add a same-named field here, splat into EngineArgs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VllmDriverConfig:
    """vLLM-specific knobs, expressed in vllm's vocabulary."""

    # Attention backend selection (FLASHINFER / FLASH_ATTN / TRITON_ATTN /
    # FLEX_ATTENTION / etc.). None = let vllm auto-pick per platform.
    attention_backend: str | None = None

    # If True, vllm runs eager (no torch.compile, no CUDA graphs).
    enforce_eager: bool = False

    # Fraction of free GPU memory to use for KV cache + activations.
    gpu_memory_utilization: float = 0.9

    # Max concurrent sequences in a batch.
    max_num_seqs: int = 256

    # Max tokens (across all sequences) in a batch. None = vllm's default.
    max_num_batched_tokens: int | None = None

    # KV cache block size override. None = vllm picks based on attention
    # backend's allowed sizes (FlashInfer: 16/32/64; FlashAttention: 16/32).
    block_size: int | None = None
