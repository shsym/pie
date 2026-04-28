"""Per-attention-backend strategy for routing pie's BRLE-decoded mask into
SGLang's attention kernels.

Pie's runtime always emits a `custom_mask` (a flat bool tensor) regardless
of whether the inferlet asks for non-causal patterns — for purely-causal
workloads it encodes the upper-triangular pattern. Each strategy below
arranges for the chosen `AttentionBackend` to honor that mask, either by
monkey-patching its metadata builder or by stamping a field onto each
`ForwardBatch` before sglang's `init_forward_metadata` reads it.

Dispatch is by `type(attn_backend).__name__`; the module imports nothing
from sglang's attention package so loading is cheap.

Coverage:
  ✓ Triton / AITER / Wave   — first-class `custom_mask` + `mask_indptr`.
  ✓ FlashInfer              — same kernel pie native uses; routed via
                              `forward_batch.cross_attention_custom_mask`.
  ✓ TorchFlex               — captured-tensor mask_mod for FlexAttention.
  ✓ TorchNative             — `attn_mask=` param to SDPA.
  ✗ FA3 / FA4 / MLA / NSA / dual_chunk / trtllm_mha / intel_amx
                              refused at engine init via
                              `_UnsupportedBackendError`. Pie's runtime
                              always emits a mask; silently dropping it
                              would produce wrong tokens for inferlets
                              that depend on non-causal attention.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _CustomMaskStrategy:
    """Per-backend strategy hook.

    Subclasses install their monkey-patch in `__init__` (one-shot) and read
    the live mask state from `self._mask` / `self._indptr` at execution
    time. `set()` is called per fire_batch by `SGLangForwardPass.transform`
    before invoking `runner.forward(fb)`; `clear()` resets state in the
    `finally` block so a stale mask can't leak into the next batch.
    """

    def __init__(self, attn_backend: Any):
        self._attn_backend = attn_backend
        self._mask: torch.Tensor | None = None
        self._indptr: torch.Tensor | None = None

    def set(self, mask: torch.Tensor, indptr: torch.Tensor) -> None:
        self._mask = mask
        self._indptr = indptr

    def clear(self) -> None:
        self._mask = None
        self._indptr = None

    def apply_to_forward_batch(self, fb: Any) -> None:
        """Strategies that route through ForwardBatch fields override this.
        Monkey-patch strategies leave it as a no-op."""
        pass


# ---------------------------------------------------------------------------
# Triton / AITER / Wave — first-class `custom_mask` + `mask_indptr` fields
# ---------------------------------------------------------------------------


class _TritonStyleStrategy(_CustomMaskStrategy):
    """Backends whose `ForwardMetadata` exposes `custom_mask` and
    `mask_indptr`, with an extend kernel gated on `USE_CUSTOM_MASK` /
    `SKIP_PREFIX_CUSTOM_MASK`.

    1. Wrap `init_forward_metadata` to inject our mask + indptr after
       sglang builds its own metadata.
    2. Wrap `extend_attention_fwd` to force `skip_prefix_custom_mask=False`
       so the prefix portion also honors pie's mask. Pie's mask spans the
       full sequence; sglang's default skips the prefix because
       speculative decoding is the only upstream consumer of `custom_mask`.
    """

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)

        orig_init = attn_backend.init_forward_metadata

        def _hooked_init(forward_batch):
            orig_init(forward_batch)
            md = attn_backend.forward_metadata
            if self._mask is not None and md is not None:
                md.custom_mask = self._mask
                md.mask_indptr = self._indptr

        attn_backend.init_forward_metadata = _hooked_init

        orig_fwd = attn_backend.extend_attention_fwd
        def _hooked_fwd(*args, **kwargs):
            kwargs.setdefault("skip_prefix_custom_mask", False)
            return orig_fwd(*args, **kwargs)
        attn_backend.extend_attention_fwd = _hooked_fwd


# ---------------------------------------------------------------------------
# FlashInfer — `forward_batch.cross_attention_custom_mask` passthrough
# ---------------------------------------------------------------------------


class _FlashInferStrategy(_CustomMaskStrategy):
    """SGLang's flashinfer adapter, when `spec_info is None` (normal extend),
    threads `forward_batch.cross_attention_custom_mask` straight into
    `BatchPrefillWithPagedKVCacheWrapper.begin_forward(custom_mask=...)`
    — the same kernel pie's native backend uses today (both pin
    `flashinfer-python==0.6.8.post1`). No monkey-patching: just stamp the
    mask onto the FB before sglang's `init_forward_metadata` reads it.
    """

    def apply_to_forward_batch(self, fb: Any) -> None:
        if self._mask is not None:
            fb.cross_attention_custom_mask = self._mask


# ---------------------------------------------------------------------------
# TorchFlexAttention — mask_mod callable
# ---------------------------------------------------------------------------


class _FlexStrategy(_CustomMaskStrategy):
    """Replace sglang's hardcoded `_causal_mask` block-mask construction with
    one that reads pie's flat mask buffer. FlexAttention compiles the
    `mask_mod` closure into the kernel via captured tensors.

    SGLang builds one `BlockMask` per request (extend mode) or per query
    (decode mode), so we do the same — each block_mask gets a mask_mod
    closure tied to that request's `mask_indptr` offset.
    """

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)
        from torch.nn.attention.flex_attention import create_block_mask
        self._create_block_mask = create_block_mask

        orig_init = type(attn_backend).init_forward_metadata
        strategy = self

        def _hooked_init(forward_batch):
            if strategy._mask is None:
                # Pie always emits a mask, but be defensive.
                orig_init(attn_backend, forward_batch)
                return
            torch.cuda.empty_cache()
            attn_backend.extend_block_masks = []
            attn_backend.decode_block_masks = []
            target = (
                attn_backend.extend_block_masks
                if forward_batch.forward_mode.is_extend()
                else attn_backend.decode_block_masks
            )
            seq_q_per_request = (
                lambda kv: kv if forward_batch.forward_mode.is_extend() else 1
            )
            for r in range(int(forward_batch.batch_size)):
                seq_kv = int(forward_batch.seq_lens_cpu[r])
                target.append(strategy._make_block_mask(seq_q_per_request(seq_kv), seq_kv, r))

        attn_backend.init_forward_metadata = _hooked_init

    def _make_block_mask(self, seq_len_q: int, seq_len_kv: int, req_idx: int):
        """Build a per-request `BlockMask` whose mask_mod reads pie's buffer."""
        base = int(self._indptr[req_idx].item())
        flat = self._mask
        stride = seq_len_kv

        def _mod(b, h, q_idx, kv_idx, _base=base, _stride=stride, _flat=flat):
            return _flat[_base + q_idx * _stride + kv_idx]

        return self._create_block_mask(
            _mod, None, None, seq_len_q, seq_len_kv,
            device=self._attn_backend.device, _compile=False,
        )


# ---------------------------------------------------------------------------
# TorchNative — SDPA's `attn_mask` parameter
# ---------------------------------------------------------------------------


class _TorchNativeStrategy(_CustomMaskStrategy):
    """Replace sglang's per-request SDPA loop with one that passes our
    `attn_mask=` slice (and disables `is_causal=`, since the mask already
    encodes that)."""

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)
        from torch.nn.functional import scaled_dot_product_attention
        self._sdpa = scaled_dot_product_attention
        attn_backend._run_sdpa_forward_extend = self._run_extend
        attn_backend._run_sdpa_forward_decode = self._run_decode

    def _per_request_mask(self, r: int, seq_len_q: int, seq_len_kv: int):
        """Slice pie's flat mask to a (seq_len_q, seq_len_kv) bool view."""
        base = int(self._indptr[r].item())
        return self._mask[base : base + seq_len_q * seq_len_kv].view(
            seq_len_q, seq_len_kv
        )

    def _gather_kv(self, k_cache, v_cache, req_to_token, req_pool_idx, seq_kv,
                   q_dim: int, ref_dtype):
        """Gather + dtype-coerce K/V for one request."""
        tokens = req_to_token[req_pool_idx, :seq_kv]
        k = k_cache[tokens].movedim(0, q_dim - 2)
        v = v_cache[tokens].movedim(0, q_dim - 2)
        if k.dtype != ref_dtype:
            k = k.to(ref_dtype)
            v = v.to(ref_dtype)
        return k, v

    def _attn_args(self, r: int, seq_q: int, seq_kv: int, causal: bool):
        """Return `(attn_mask, is_causal)` for SDPA — pie's slice when
        we have a mask, else fall back to sglang's causal default."""
        if self._mask is None:
            return None, causal
        return self._per_request_mask(r, seq_q, seq_kv).unsqueeze(0).unsqueeze(0), False

    def _run_extend(
        self, query, output, k_cache, v_cache, req_to_token,
        req_pool_indices, seq_lens, extend_prefix_lens, extend_seq_lens,
        scaling=None, enable_gqa=False, causal=False,
    ):
        # We pass an (ext_q, seq_kv) mask straight to SDPA — no query
        # padding (sglang's reference padded with `torch.empty` and threw
        # away the dummy rows; with our explicit mask we don't need that).
        query = query.movedim(0, query.dim() - 2)
        start_q = 0
        for r in range(seq_lens.shape[0]):
            ext_q = int(extend_seq_lens[r])
            seq_kv = int(seq_lens[r])
            end_q = start_q + ext_q

            per_req_q = query[:, start_q:end_q, :]
            k, v = self._gather_kv(
                k_cache, v_cache, req_to_token, req_pool_indices[r],
                seq_kv, query.dim(), per_req_q.dtype,
            )
            # Pie's mask for request r is (ext_q × seq_kv) — one row per
            # extend query token, full kv-length wide.
            attn_mask, is_causal_arg = self._attn_args(r, ext_q, seq_kv, causal)

            out = self._sdpa(
                per_req_q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                attn_mask=attn_mask, enable_gqa=enable_gqa, scale=scaling,
                is_causal=is_causal_arg,
            ).squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = out
            start_q = end_q
        return output

    def _run_decode(
        self, query, output, k_cache, v_cache, req_to_token,
        req_pool_indices, seq_lens, scaling=None, enable_gqa=False, causal=False,
    ):
        query = query.movedim(0, query.dim() - 2)
        for r in range(seq_lens.shape[0]):
            seq_kv = int(seq_lens[r])
            per_req_q = query[:, r:r + 1, :]
            k, v = self._gather_kv(
                k_cache, v_cache, req_to_token, req_pool_indices[r],
                seq_kv, query.dim(), per_req_q.dtype,
            )
            # In decode the single query token corresponds to the LAST row of
            # this request's mask (q_idx = 0 within the row).
            attn_mask, is_causal_arg = self._attn_args(r, 1, seq_kv, causal)

            out = self._sdpa(
                per_req_q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                attn_mask=attn_mask, enable_gqa=enable_gqa, scale=scaling,
                is_causal=is_causal_arg,
            ).squeeze(0).movedim(query.dim() - 2, 0)
            output[r:r + 1, :, :] = out
        return output


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


# `type(attn_backend).__name__` → strategy class. Adding a new backend means
# appending one entry. Class names are listed instead of imported eagerly so
# this module stays cheap.
_STRATEGY_BY_NAME: dict[str, type[_CustomMaskStrategy]] = {
    "TritonAttnBackend": _TritonStyleStrategy,
    "AiterAttnBackend": _TritonStyleStrategy,
    "WaveAttnBackend": _TritonStyleStrategy,
    "FlashInferAttnBackend": _FlashInferStrategy,
    "TorchFlexAttnBackend": _FlexStrategy,
    "TorchNativeAttnBackend": _TorchNativeStrategy,
}


# Backends explicitly known to NOT support arbitrary custom masks. We refuse
# at engine init rather than silently producing wrong tokens for inferlets
# that depend on non-causal attention.
_UNSUPPORTED_BACKENDS: dict[str, str] = {
    "FlashAttentionBackend": (
        "FlashAttention v3/v4 only consults `custom_mask` for spec-decoding "
        "mask extraction; arbitrary patterns aren't a kernel parameter."
    ),
    "FlashInferMLAAttnBackend": "DeepSeek MLA — no custom_mask support.",
    "FlashMLABackend": "DeepSeek MLA — no custom_mask support.",
    "CutlassMLABackend": "DeepSeek MLA via CUTLASS — no custom_mask support.",
    "TRTLLMMLABackend": "DeepSeek MLA via TensorRT-LLM — no custom_mask support.",
    "TRTLLMHAAttnBackend": (
        "TensorRT-LLM's flashinfer.{prefill,decode}.trtllm_batch_* "
        "kernels are causal-only."
    ),
    "IntelAMXAttnBackend": (
        "Intel AMX dispatches to fused C++ ops without a mask parameter."
    ),
    "NativeSparseAttnBackend": (
        "DeepSeek-V3.2 native sparse attention; mask is fixed by sparsity."
    ),
    "DualChunkFlashAttentionBackend": (
        "Qwen 1M dual-chunk attention — mask is fixed by the chunking."
    ),
}


class _UnsupportedBackendError(RuntimeError):
    """Raised when sglang's resolved attention backend can't honor pie's
    custom_mask. Listed early so engine load fails before any inferlet."""


def make_mask_strategy(attn_backend: Any) -> _CustomMaskStrategy:
    """Pick the right strategy for an attention backend instance, or raise.

    Dispatches on the class name to avoid pulling in every sglang
    attention-backend module at import time.
    """
    name = type(attn_backend).__name__
    if cls := _STRATEGY_BY_NAME.get(name):
        return cls(attn_backend)

    supported = sorted(_STRATEGY_BY_NAME)
    if reason := _UNSUPPORTED_BACKENDS.get(name):
        msg = (
            f"pie_backend_sglang refuses to load with attention_backend "
            f"resolved to {name!r}: {reason} Pie's runtime always emits a "
            f"custom_mask buffer; silently ignoring it would produce wrong "
            f"tokens for inferlets that rely on non-causal attention. Set "
            f"`[model.X.driver.sglang] attention_backend = \"triton\"` (or "
            f"one of {supported}) instead."
        )
    else:
        # Unknown class — refuse, but tell the user how to register it.
        msg = (
            f"pie_backend_sglang doesn't know how to route custom_mask "
            f"through SGLang attention backend {name!r}. Either add a "
            f"strategy in pie_backend_sglang/mask_hooks.py or pick one of "
            f"the verified backends: {supported}."
        )
    raise _UnsupportedBackendError(msg)
