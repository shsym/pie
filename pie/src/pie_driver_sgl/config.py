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
    # Mirrors sglang's `mem_fraction_static`. Default lower than sglang's
    # standalone 0.88 because pie's KV-rebind allocates a parallel tensor
    # in pie's canonical layout (see kv_cache.py); without slack we OOM at
    # first fire_batch.
    mem_fraction_static: float = 0.65

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

    # Universal pie knob (not an sglang ServerArgs field). Sized in GiB; sets
    # the pinned host KV pool that backs D2H/H2D swap. 0 disables swap. The
    # worker forwards this into `RuntimeConfig.swap_budget_bytes`; the loader
    # filters it out when splatting into `ServerArgs`.
    cpu_mem_budget_in_gb: int = 0

    # ---- CMA-ES adapter (zero-order training) ----
    # When True, allocates per-layer adapter storage and class-swaps
    # `QKVParallelLinear` on every decoder layer for an adapter-aware
    # wrapper that adds the noisy DOWN/UP-projection contribution to
    # Q/K/V at forward time. Required for ZO/CMA-ES training inferlets
    # (e.g., sdk/demo/zo-training/). Disabled by default — enabling
    # forces `disable_cuda_graph=True` since the v1 wrapper isn't graph-
    # capture-friendly (the sub pass-or-not branch needs a record-with /
    # record-without-adapter pair to be CUDA-graph safe; deferred).
    enable_adapter: bool = False
    # Per-engine adapter slot capacity and max LoRA rank. Mirror of the
    # fields on `pie_backend.config.NativeRuntimeConfig`; placed here
    # rather than on the base RuntimeConfig so the change is sglang-
    # scoped (matches the user's "no cross-cutting native changes for
    # now" guidance).
    max_num_adapters: int = 32
    max_adapter_rank: int = 8

    # ---- Speculative decoding (NGRAM, backend-supplied drafts) ----
    # When True, the engine maintains an n-gram trie of recently-accepted
    # tokens and proposes linear draft continuations to the runtime as
    # `next_spec_tokens` in TokensWithSpeculation responses. The inferlet
    # opts in by calling `output_speculative_tokens(true)`; otherwise the
    # drafts are dropped at response packaging.
    spec_ngram_enabled: bool = False
    # Number of drafts proposed per accepted iteration.
    spec_ngram_num_drafts: int = 4
    # Maximum trie depth — longer ngrams are not stored.
    spec_ngram_max_depth: int = 18
    # Trie capacity in tokens (approximate node budget).
    spec_ngram_capacity: int = 1_000_000
