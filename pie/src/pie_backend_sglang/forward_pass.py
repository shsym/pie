"""Adapter that exposes an SGLang ModelRunner under pie_backend's ForwardPass contract.

`pie_backend.engine.Engine` calls three methods on its `forward_pass` object:
`embed_inputs(inputs) -> hidden`, `transform(...) -> hidden`, and
`sample(hidden, sampling_metadata) -> dict`. We satisfy that contract here
while delegating the actual compute to SGLang's ModelRunner.

Pie supports multiple sampling positions per request (best-of-n, distribution
mode in prefill, multi-step parallel generation). SGLang's default
LogitsProcessor gathers logits to *one* position per request — the last
extend-token. To preserve pie's contract we replace the model's
LogitsProcessor with a passthrough that returns the full per-token hidden
state tensor; we then apply pie's per-output `indices_for_logits` gather and
the LM head ourselves inside `sample_common`.
"""

from __future__ import annotations

from typing import Any

import torch

from pie_backend.config import RuntimeConfig
from pie_backend.model.common import sample_common

from .forward_batch import build_sglang_forward_batch


class _HiddenCapture(torch.nn.Module):
    """Drop-in replacement for `runner.model.logits_processor`.

    SGLang models assign their LogitsProcessor as a `nn.Module` child, so we
    inherit from `torch.nn.Module` to satisfy `__setattr__`'s typecheck.

    SGLang models call:
        self.logits_processor(input_ids, hidden_states, lm_head, forward_batch, aux=None)
    expecting a `LogitsProcessorOutput`. We sidestep the LM head + per-request
    gather and just stash the full per-token hidden states for our caller.
    """

    def __init__(self):
        super().__init__()
        self.captured: torch.Tensor | None = None

    def forward(self, input_ids, hidden_states, lm_head, forward_batch, aux=None):
        # Stash and return a sentinel LogitsProcessorOutput that carries the
        # full per-token hidden states. SGLang's model.forward returns this
        # directly — we'll pull `.hidden_states` out in `transform()`.
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        self.captured = hidden_states
        # `next_token_logits=None` is allowed and signals that no logits are
        # available (multi-item scoring path). The caller doesn't read it.
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )


class SGLangForwardPass:
    """Thin shim around an SGLang ModelRunner.

    Contract:
      - `embed_inputs(inputs)`: passthrough — SGLang's `forward()` does its
        own input embedding.
      - `transform(...)`: builds a ForwardBatch from pie's inputs, invokes
        `runner.forward()` with the LM-head-stage hijacked, returns full
        per-token hidden states `(num_query_tokens, hidden_dim)`.
      - `sample(hidden, sampling_metadata)`: gathers the indices pie
        requested, applies the LM head, runs pie's sampler.
    """

    def __init__(
        self,
        *,
        runner: Any,
        runtime_config: RuntimeConfig,
        page_size: int,
    ):
        self.runner = runner
        self.runtime_config = runtime_config
        self.page_size = page_size
        self.device = torch.device(runtime_config.device)

        # Install the hidden-state capture once; keep a handle so transform()
        # can read what was captured.
        self._capture = _HiddenCapture()
        runner.model.logits_processor = self._capture

        # Resolve the LM head once. Used by sample() as `lm_head_fn`.
        # ParallelLMHead with TP > 1 needs an all-gather, but for v1 we only
        # support TP=1; the matmul-against-weight path matches sglang's own
        # `_compute_lm_head` for that case.
        self._lm_head_module = runner.model.lm_head

    # ------------------------------------------------------------------
    # Pie contract methods
    # ------------------------------------------------------------------

    def embed_inputs(self, inputs: dict) -> dict:
        # SGLang's forward() owns input embedding; nothing to do here.
        return inputs

    def transform(
        self,
        *,
        input_embeds: dict,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        single_token_inference_mode: bool,
        total_pages_cpu: int = 0,
        custom_mask: torch.Tensor | None = None,
        adapter_subpass=None,
    ) -> torch.Tensor:
        inputs = input_embeds
        fb = build_sglang_forward_batch(
            runner=self.runner,
            inputs={
                "token_ids": inputs["token_ids"],
                "position_ids": position_ids,
                "qo_indptr": qo_indptr,
                "kv_page_indices": kv_page_indices,
                "kv_page_indptr": kv_page_indptr,
                "kv_last_page_lens": kv_last_page_lens,
            },
            page_size=self.page_size,
            device=self.device,
        )

        # Run the model. Our `_HiddenCapture` runs in place of the
        # LogitsProcessor and stashes full per-token hidden states.
        self._capture.captured = None
        self.runner.forward(fb)

        hidden_states = self._capture.captured
        if hidden_states is None:
            raise RuntimeError(
                "pie_backend_sglang: hidden states were not captured. The model "
                "may not have called its logits_processor (unexpected forward path)."
            )
        return hidden_states

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _lm_head_fn(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the LM head to the gathered per-output hidden states.

        Mirrors sglang's `LogitsProcessor._compute_lm_head` (TP=1 path) at
        layers/logits_processor.py:891-913.
        """
        lm_head = self._lm_head_module
        if hasattr(lm_head, "weight"):
            return torch.matmul(
                hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
            )
        # Fallback to module call for unusual lm_heads (LoRA-wrapped, GGUF).
        return lm_head(hidden_states)

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        if not sampling_metadata or sampling_metadata.get("indices_for_logits") is None:
            return {"tokens": [], "dists": [], "spec_tokens": [], "spec_positions": []}

        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self._lm_head_fn,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )
