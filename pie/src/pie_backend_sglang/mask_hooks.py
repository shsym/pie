"""Per-attention-backend strategy for routing pie's BRLE-decoded mask into
SGLang's attention kernels.

Pie's runtime always emits a `custom_mask` (a flat bool tensor) regardless
of whether the inferlet asks for non-causal patterns — for purely causal
workloads it's effectively the upper-triangular pattern. Each strategy below
installs a one-shot monkeypatch on the chosen `AttentionBackend` so the
kernel honors that mask. The patch reads `strategy.set(mask, indptr)` state
that pie's `SGLangForwardPass.transform` updates per fire_batch.

We dispatch on `type(attn_backend).__name__` rather than imports so the
strategy module stays import-light; each backend's module only loads when
its strategy is constructed.

Coverage matrix (sglang upstream as of this branch):

  ✓ Triton                  — first-class custom_mask + mask_indptr fields.
  ✓ AITER, Wave             — same ForwardMetadata shape as Triton.
  ✓ FlashInfer              — same kernel pie's *native* uses; routes via
                              `forward_batch.cross_attention_custom_mask`.
  ✓ TorchFlex               — captured-tensor mask_mod via FlexAttention.
  ✓ TorchNative             — attn_mask param to scaled_dot_product_attention.
  ✗ FA3 / FA4               — refused at engine init; kernel has no
                              arbitrary-mask path.
  ✗ trtllm_mha / intel_amx  — refused; fused kernels without exposed mask.
  ✗ MLA / NSA / dual_chunk  — refused; specialized kernels.

Backends that can't honor a custom mask are rejected at engine init via
`_UnsupportedBackendError` — pie's runtime always sends a mask, so silently
ignoring it would produce wrong tokens for inferlets that rely on
non-causal attention (Jacobi, tree decoding, attention sink, etc.). The
user must pick a supported `attention_backend`.
"""

from __future__ import annotations

from typing import Any

import torch


class _CustomMaskStrategy:
    """Base class for per-backend mask hooks.

    Subclasses install their monkeypatch in `__init__` (one-shot). Every
    fire_batch, `SGLangForwardPass.transform` calls `set(mask, indptr,
    seq_lens_np)` before invoking `runner.forward(...)`; the patched kernel
    reads from the strategy at execution time.
    """

    supports_custom_mask: bool = False

    def __init__(self, attn_backend: Any):
        self._attn_backend = attn_backend
        self._mask: torch.Tensor | None = None
        self._indptr: torch.Tensor | None = None
        self._seq_lens_np = None

    def set(self, mask: torch.Tensor, indptr: torch.Tensor, seq_lens_np) -> None:
        self._mask = mask
        self._indptr = indptr
        self._seq_lens_np = seq_lens_np

    def clear(self) -> None:
        self._mask = None
        self._indptr = None
        self._seq_lens_np = None

    def apply_to_forward_batch(self, fb: Any) -> None:
        """Optional hook for strategies that route the mask via fields on
        the ForwardBatch (e.g. FlashInfer's `cross_attention_custom_mask`).

        Strategies that work via monkey-patched `init_forward_metadata`
        leave this as a no-op.
        """
        pass


# ---------------------------------------------------------------------------
# Triton / AITER / Wave — identical ForwardMetadata shape & kernel signature
# ---------------------------------------------------------------------------


class _TritonStyleStrategy(_CustomMaskStrategy):
    """Hook for backends whose `ForwardMetadata` exposes `custom_mask` and
    `mask_indptr` fields, with a Triton extend kernel that gates on
    `USE_CUSTOM_MASK` and `SKIP_PREFIX_CUSTOM_MASK`.

    We:
      1. Wrap `init_forward_metadata` to inject our mask + indptr after
         sglang builds the rest of the metadata.
      2. Wrap the extend kernel to force `skip_prefix_custom_mask=False`
         so the prefix portion also applies pie's mask. Pie's mask covers
         the full sequence (prefix + extend), and pie's per-token row of
         `seq_len` bools encodes the causal pattern when no inferlet
         override is present.

    Re-entrancy: this is single-threaded (one engine, one fire_batch at a
    time per group leader). No locking needed.
    """

    supports_custom_mask = True

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)

        # 1) Wrap init_forward_metadata. Method type is `bound method`; we
        # replace the attribute with a closure that calls the original then
        # patches the metadata. Don't replace the underlying class method —
        # that would affect every other backend instance in the process
        # (only one in pie, but defensive).
        orig_init = attn_backend.init_forward_metadata

        def _hooked_init(forward_batch):
            orig_init(forward_batch)
            if self._mask is not None and attn_backend.forward_metadata is not None:
                attn_backend.forward_metadata.custom_mask = self._mask
                attn_backend.forward_metadata.mask_indptr = self._indptr

        attn_backend.init_forward_metadata = _hooked_init

        # 2) Wrap extend_attention_fwd. The kernel's `skip_prefix_custom_mask`
        # default is True (only spec-decoding wants prefix-skipped masks).
        # Pie's mask spans the full seq_len so we always want False.
        orig_fwd = attn_backend.extend_attention_fwd

        def _hooked_fwd(*args, **kwargs):
            kwargs.setdefault("skip_prefix_custom_mask", False)
            return orig_fwd(*args, **kwargs)

        attn_backend.extend_attention_fwd = _hooked_fwd


# ---------------------------------------------------------------------------
# TorchFlexAttention — mask_mod callable
# ---------------------------------------------------------------------------


class _FlexStrategy(_CustomMaskStrategy):
    """Hook for `TorchFlexAttnBackend`.

    SGLang's flex backend builds a `BlockMask` via `create_block_mask(...)`
    inside `init_forward_metadata`, hardcoding `_causal_mask` /
    `_decode_mask` callables. We replace those callables with closures over
    pie's flat mask buffer so FlexAttention reads the per-position bool from
    our tensor inside the compiled kernel.

    NB: FlexAttention's `mask_mod(b, h, q_idx, kv_idx) -> bool` runs inside
    a `torch.compile`'d graph. Captured tensors are fine — the mask buffer
    becomes a graph input.
    """

    supports_custom_mask = True

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)

        strategy = self
        # SGLang creates one block_mask *per request* in its loop, so it
        # passes per-request `seq_len_q` and `seq_len_kv` to create_block_mask.
        # We need to know which request the mask corresponds to. SGLang
        # doesn't pass `b` indexing, so we drive the index ourselves: each
        # call to `_pie_extend_mask_mod` gets a fresh closure tied to a
        # specific request's offset.

        # Replace `_causal_mask` / `_decode_mask` with builders that close
        # over pie's mask. SGLang calls them as `self._causal_mask(b, h, q,
        # kv)`, but we want a per-request closure. So instead we override
        # the helper that creates block masks and substitute the mask_mod.
        # Simpler approach: replace `init_forward_metadata` with a thin
        # version that builds block_masks using our mask_mod factory.

        from torch.nn.attention.flex_attention import create_block_mask

        def _hooked_init(forward_batch):
            mask = strategy._mask
            indptr = strategy._indptr
            seq_lens_np = strategy._seq_lens_np

            # When pie hasn't supplied a mask (shouldn't happen — pie always
            # emits one), fall back to sglang's original behavior.
            if mask is None:
                # Re-bind the original method via the class, since we replaced
                # it on the instance.
                type(attn_backend).init_forward_metadata(attn_backend, forward_batch)
                return

            # FlexAttention is single-batch (per-request) in sglang's flex
            # backend, so we build one BlockMask per request, each with its
            # own mask_mod closure that reads pie's buffer.
            torch.cuda.empty_cache()
            attn_backend.extend_block_masks = []
            attn_backend.decode_block_masks = []

            if forward_batch.forward_mode.is_extend():
                for r in range(int(forward_batch.batch_size)):
                    seq_len_kv = int(seq_lens_np[r])
                    seq_len_q = seq_len_kv  # sglang uses kv-length for flex extend
                    base = int(indptr[r].item())
                    row_stride = seq_len_kv
                    flat = mask  # captured; lives on device

                    def _mod(b, h, q_idx, kv_idx,
                             _base=base, _stride=row_stride, _flat=flat):
                        return _flat[_base + q_idx * _stride + kv_idx]

                    attn_backend.extend_block_masks.append(
                        create_block_mask(
                            _mod, None, None, seq_len_q, seq_len_kv,
                            device=attn_backend.device, _compile=False,
                        )
                    )
            elif forward_batch.forward_mode.is_decode():
                for r in range(int(forward_batch.batch_size)):
                    seq_len_kv = int(seq_lens_np[r])
                    seq_len_q = 1
                    base = int(indptr[r].item())
                    row_stride = seq_len_kv
                    flat = mask

                    def _mod(b, h, q_idx, kv_idx,
                             _base=base, _stride=row_stride, _flat=flat):
                        return _flat[_base + q_idx * _stride + kv_idx]

                    attn_backend.decode_block_masks.append(
                        create_block_mask(
                            _mod, None, None, seq_len_q, seq_len_kv,
                            device=attn_backend.device, _compile=False,
                        )
                    )

        attn_backend.init_forward_metadata = _hooked_init


# ---------------------------------------------------------------------------
# TorchNative — SDPA's attn_mask parameter
# ---------------------------------------------------------------------------


class _TorchNativeStrategy(_CustomMaskStrategy):
    """Hook for `TorchNativeAttnBackend`.

    SGLang calls `scaled_dot_product_attention(..., is_causal=causal)` with
    no `attn_mask`. We replace `_run_sdpa_forward_extend` and
    `_run_sdpa_forward_decode` with versions that build a per-request
    `attn_mask` from pie's flat buffer and pass it to SDPA (and disable
    `is_causal` since the mask already encodes that).
    """

    supports_custom_mask = True

    def __init__(self, attn_backend: Any):
        super().__init__(attn_backend)

        strategy = self
        from torch.nn.functional import scaled_dot_product_attention

        def _per_request_mask(r: int, seq_len_q: int, seq_len_kv: int):
            """Slice pie's flat mask to a (seq_len_q, seq_len_kv) bool view."""
            base = int(strategy._indptr[r].item())
            return strategy._mask[base : base + seq_len_q * seq_len_kv].view(
                seq_len_q, seq_len_kv
            )

        def _hooked_extend(query, output, k_cache, v_cache, req_to_token,
                           req_pool_indices, seq_lens, extend_prefix_lens,
                           extend_seq_lens, scaling=None, enable_gqa=False,
                           causal=False):
            # Per-sequence loop mirrors sglang's reference, but we pass our
            # mask instead of `is_causal`.
            query = query.movedim(0, query.dim() - 2)
            start_q = 0
            for r in range(seq_lens.shape[0]):
                ext_q = int(extend_seq_lens[r])
                pre_q = int(extend_prefix_lens[r])
                seq_kv = int(seq_lens[r])
                end_q = start_q + ext_q

                per_req_q = query[:, start_q:end_q, :]
                # Pad query to full seq_kv length so SDPA's per-query masks
                # align with key indices [0, seq_kv).
                per_req_q_full = torch.empty(
                    (per_req_q.shape[0], seq_kv, per_req_q.shape[2]),
                    dtype=per_req_q.dtype, device=per_req_q.device,
                )
                per_req_q_full[:, pre_q:, :] = per_req_q

                req_pool_idx = req_pool_indices[r]
                tokens = req_to_token[req_pool_idx, :seq_kv]
                k = k_cache[tokens].movedim(0, query.dim() - 2)
                v = v_cache[tokens].movedim(0, query.dim() - 2)
                if not (per_req_q.dtype == k.dtype == v.dtype):
                    k = k.to(per_req_q.dtype)
                    v = v.to(per_req_q.dtype)

                if strategy._mask is not None:
                    attn_mask = _per_request_mask(r, seq_kv, seq_kv).unsqueeze(0).unsqueeze(0)
                    is_causal_arg = False
                else:
                    attn_mask = None
                    is_causal_arg = causal

                out_full = scaled_dot_product_attention(
                    per_req_q_full.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=is_causal_arg,
                ).squeeze(0).movedim(query.dim() - 2, 0)
                output[start_q:end_q, :, :] = out_full[pre_q:, :, :]
                start_q = end_q
            return output

        def _hooked_decode(query, output, k_cache, v_cache, req_to_token,
                           req_pool_indices, seq_lens, scaling=None,
                           enable_gqa=False, causal=False):
            query = query.movedim(0, query.dim() - 2)
            start_q = 0
            for r in range(seq_lens.shape[0]):
                seq_kv = int(seq_lens[r])
                end_q = start_q + 1
                per_req_q = query[:, start_q:end_q, :]

                req_pool_idx = req_pool_indices[r]
                tokens = req_to_token[req_pool_idx, :seq_kv]
                k = k_cache[tokens].movedim(0, query.dim() - 2)
                v = v_cache[tokens].movedim(0, query.dim() - 2)
                if not (per_req_q.dtype == k.dtype == v.dtype):
                    k = k.to(per_req_q.dtype)
                    v = v.to(per_req_q.dtype)

                if strategy._mask is not None:
                    # In decode the query has 1 token; its row of the mask
                    # corresponds to the *last* query position in the request.
                    attn_mask = _per_request_mask(r, 1, seq_kv).unsqueeze(0).unsqueeze(0)
                    is_causal_arg = False
                else:
                    attn_mask = None
                    is_causal_arg = causal

                out = scaled_dot_product_attention(
                    per_req_q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=is_causal_arg,
                ).squeeze(0).movedim(query.dim() - 2, 0)
                output[start_q:end_q, :, :] = out
                start_q = end_q
            return output

        attn_backend._run_sdpa_forward_extend = _hooked_extend
        attn_backend._run_sdpa_forward_decode = _hooked_decode


# ---------------------------------------------------------------------------
# FlashInfer — route via `forward_batch.cross_attention_custom_mask`
# ---------------------------------------------------------------------------


class _FlashInferStrategy(_CustomMaskStrategy):
    """Hook for `FlashInferAttnBackend`.

    SGLang's flashinfer adapter, when `spec_info is None` (normal extend),
    pulls `forward_batch.cross_attention_custom_mask` through to
    `BatchPrefillWithPagedKVCacheWrapper.begin_forward(custom_mask=...)`
    — see flashinfer_backend.py:514, 1431, 1490. The kernel layout for
    that mask is per-request rectangular `query_len × seq_len` flat,
    which is exactly what pie's BRLE decoder produces. So instead of
    monkeypatching anything, we just stamp the mask onto each
    ForwardBatch in `apply_to_forward_batch` and let sglang's path do
    its thing.

    NB: this is the same kernel pie's *native* backend uses today (both
    pin `flashinfer-python==0.6.8.post1`). Performance ceiling matches
    native; the only overhead vs Triton is FlashInfer's per-step plan().
    """

    supports_custom_mask = True

    def apply_to_forward_batch(self, fb: Any) -> None:
        # Pie always emits a mask; for purely-causal workloads it
        # encodes the upper-triangular pattern and the kernel is happy.
        if self._mask is not None:
            fb.cross_attention_custom_mask = self._mask


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


# Map from `type(attn_backend).__name__` to strategy class. Adding a new
# backend means appending one entry here. Strategy class names are listed
# instead of imported eagerly so the module stays cheap to load.
_STRATEGY_BY_NAME: dict[str, type[_CustomMaskStrategy]] = {
    # NVIDIA / AMD with first-class custom_mask + mask_indptr in
    # ForwardMetadata (same Triton-style kernel signature):
    "TritonAttnBackend": _TritonStyleStrategy,
    "AiterAttnBackend": _TritonStyleStrategy,
    "WaveAttnBackend": _TritonStyleStrategy,
    # FlashInfer — same kernel pie native uses; mask routes via
    # `forward_batch.cross_attention_custom_mask` in normal extend mode:
    "FlashInferAttnBackend": _FlashInferStrategy,
    # PyTorch FlexAttention — mask_mod callable:
    "TorchFlexAttnBackend": _FlexStrategy,
    # Pure PyTorch SDPA fallback:
    "TorchNativeAttnBackend": _TorchNativeStrategy,
}


# Backends explicitly known to NOT support arbitrary custom masks, with
# the canonical reason. We refuse to load on these rather than silently
# producing wrong tokens for inferlets that depend on the mask.
_UNSUPPORTED_BACKENDS: dict[str, str] = {
    "FlashAttentionBackend": (
        "FlashAttention v3/v4's kernel only consults `custom_mask` for "
        "speculative-decoding mask extraction; arbitrary attention "
        "patterns aren't a parameter of the kernel."
    ),
    "FlashInferMLAAttnBackend": (
        "DeepSeek MLA kernels don't take a custom_mask parameter."
    ),
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
        "DeepSeek-V3.2 native sparse attention; mask is fixed by the "
        "sparsity pattern."
    ),
    "DualChunkFlashAttentionBackend": (
        "Qwen 1M dual-chunk attention — mask is fixed by the chunking scheme."
    ),
}


class _UnsupportedBackendError(RuntimeError):
    """Raised when the user picks an sglang attention backend that can't
    honor pie's custom_mask. Listed early so engine load fails before any
    inferlet runs."""


def make_mask_strategy(attn_backend: Any) -> _CustomMaskStrategy:
    """Pick the right strategy for an attention backend instance.

    Dispatches on the class name to avoid pulling in every sglang
    attention-backend module at import time. Raises
    `_UnsupportedBackendError` for backends that can't honor pie's
    custom_mask — pie's runtime always emits one, and silently dropping
    it would produce wrong tokens for inferlets that rely on non-causal
    patterns (Jacobi, tree decoding, attention sink, etc.).
    """
    name = type(attn_backend).__name__

    cls = _STRATEGY_BY_NAME.get(name)
    if cls is not None:
        return cls(attn_backend)

    reason = _UNSUPPORTED_BACKENDS.get(name)
    supported = sorted(_STRATEGY_BY_NAME)
    if reason is not None:
        msg = (
            f"pie_backend_sglang refuses to load with attention_backend "
            f"resolved to {name!r}: {reason} "
            f"Pie's runtime always emits a custom_mask buffer; silently "
            f"ignoring it would produce wrong tokens for inferlets that "
            f"rely on non-causal attention. Set "
            f"`[model.X.driver.sglang] attention_backend = \"triton\"` "
            f"(or one of {supported}) instead."
        )
    else:
        # Unknown class name — be conservative and refuse, but tell the
        # user how to add it. New sglang attention backends land
        # occasionally; add the class name to either _STRATEGY_BY_NAME or
        # _UNSUPPORTED_BACKENDS once you've checked the kernel path.
        msg = (
            f"pie_backend_sglang doesn't know how to route custom_mask "
            f"through SGLang attention backend {name!r}. Either add a "
            f"strategy in pie_backend_sglang/mask_hooks.py or pick one of "
            f"the verified backends: {supported}."
        )
    raise _UnsupportedBackendError(msg)
