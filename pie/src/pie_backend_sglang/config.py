"""SGLang driver config — typed view of the `ServerArgs` subset pie cares
about. Field names mirror sglang's `ServerArgs` so values flow verbatim.

    [model.X.driver.sglang]
    attention_backend = "triton"        → ServerArgs.attention_backend
    mem_fraction_static = 0.85          → ServerArgs.mem_fraction_static
    page_size = 16                      → ServerArgs.page_size
    ...

Adding a new sglang knob: add a same-named field here, splat into ServerArgs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SGLangDriverConfig:
    """SGLang-specific knobs, expressed in sglang's vocabulary."""

    # Attention backend selection (triton / flashinfer / flex_attention /
    # fa3 / fa4 / aiter / wave / torch_native / ...). "triton" is pie's
    # default — it has the cleanest custom-mask support and works on any
    # NVIDIA SM 7.5+. See sglang/srt/layers/attention/attention_registry.py
    # for the full list.
    attention_backend: str = "triton"

    # Fraction of free GPU memory to reserve for KV cache + activations.
    # Mirrors sglang's `mem_fraction_static`.
    mem_fraction_static: float = 0.85

    # KV cache page size override. None = sglang picks based on the chosen
    # attention backend's allowed sizes.
    page_size: int | None = 16

    # If True, sglang runs eager (no torch.compile, no CUDA graphs).
    # Mirrors sglang's `disable_cuda_graph`.
    disable_cuda_graph: bool = False

    # KV cache element dtype. "auto" inherits the model's activation dtype.
    kv_cache_dtype: str = "auto"

    # Trust user-supplied remote code in HF repos (needed for some models).
    trust_remote_code: bool = True

    # Optional explicit context length cap. None = read from HF config.
    context_length: int | None = None

    # Disable sglang's radix prefix cache. Pie owns prefix sharing via its
    # own scheduler; running sglang's on top is wasted work.
    disable_radix_cache: bool = True
