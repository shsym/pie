"""Per-backend strategies for routing pie's BRLE-decoded mask into vllm's
attention kernels.

Pairs with `mask_compute.py`: this file owns *dispatch* (which backend
gets which kernel call); `mask_compute.py` owns *kernels and data layout*
(`PieAttnExtras`, the SDPA gather, the FlashInfer wrapper helpers).

Architecture
------------
Vllm's V1 attention separates layer-level bookkeeping (`Attention.forward`)
from kernel dispatch (`AttentionImpl.forward`). Per backend, vllm ships an
impl: `FlashInferImpl`, `FlashAttentionImpl`, `TritonAttentionImpl`, ...

We use **composition**, not inheritance:

  * `PieMaskedAttentionImplProxy` wraps the original impl. Its `forward`
    consults `ForwardContext.additional_kwargs["pie_attn_extras"]`; absent
    means pass-through to the wrapped impl, present means dispatch to a
    per-backend strategy.

  * `_*Strategy` classes hold kernel-specific logic and any per-strategy
    flags (`pie_uses_flashinfer_wrapper` etc.). One strategy class per
    supported backend; one strategy *instance* per attn layer.

  * `install_mask_strategies(vllm_config)` looks up the strategy by impl
    class name, instantiates it, then replaces `attn_layer.impl` with a
    proxy that closes over the strategy. Idempotent.

Why composition over the subclass + `__class__` swap pattern
------------------------------------------------------------
Vllm's AttentionImpl chain declares `__slots__ = ()` end-to-end. Subclassing
+ `instance.__class__ = MaskedSubclass` works *if* the subclass also has
matching slots, but: (1) CPython's layout check rejects mixin-first MROs
even with matching empty slots; (2) any future vllm impl that omits
`__slots__ = ()` breaks our install path; (3) the subclass approach can't
hold per-instance strategy state (slots block instance attrs). Composition
sidesteps all three.

Refusal policy
--------------
Impls without a registered strategy cause `NotImplementedError` at engine
init. Pie's runtime always emits a custom_mask buffer; silently ignoring
it on an unverified backend would risk wrong tokens for inferlets that
depend on non-causal attention. Add a strategy + register in
`_STRATEGY_BY_NAME` to support a new backend.
"""

from __future__ import annotations

from typing import Any

from .mask_compute import PieAttnExtras


# ----------------------------------------------------------------------------
# Strategy base
# ----------------------------------------------------------------------------


class _Strategy:
    """Per-backend masked-attention compute path.

    Strategies are stateless w.r.t. per-batch data — the mask + plumbing
    arrive via the `extras` argument. A strategy may carry per-engine
    state (a FlashInfer workspace pointer, etc.) but per-batch state
    flows through `PieAttnExtras` only.
    """

    # Strategies that consume the FlashInfer prefill wrapper signal it here
    # so `forward_pass.transform()` knows to pre-plan one this batch.
    pie_uses_flashinfer_wrapper: bool = False

    def run(
        self,
        impl: Any,
        layer: Any,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        extras: PieAttnExtras,
    ) -> None:
        """Run masked attention. Subclasses override.

        `impl` is the wrapped AttentionImpl, in case the strategy needs to
        call `impl.do_kv_cache_update` or read `impl.scale` etc.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Per-backend strategies
# ----------------------------------------------------------------------------


class _FlashInferStrategy(_Strategy):
    """Run the pre-planned FlashInfer prefill wrapper with `custom_mask=`.

    The wrapper itself is built and planned in `forward_pass.transform()`
    once per batch and stashed on `extras.flashinfer_wrapper`. The strategy
    just runs it.
    """

    pie_uses_flashinfer_wrapper = True

    def run(self, impl, layer, query, key, value, kv_cache, attn_metadata,
            output, extras):
        if extras.flashinfer_wrapper is None:
            raise RuntimeError(
                "pie_backend_vllm: FlashInfer strategy reached run() but "
                "extras.flashinfer_wrapper is None. forward_pass.transform() "
                "should have planned the wrapper for this batch."
            )
        extras.flashinfer_wrapper.run(query, kv_cache, out=output)


# ----------------------------------------------------------------------------
# Strategy registry — dispatched by impl class NAME (resilient to vllm
# moving the import path; fails gracefully if vllm renames the class).
# ----------------------------------------------------------------------------


_STRATEGY_BY_NAME: dict[str, type[_Strategy]] = {
    "FlashInferImpl": _FlashInferStrategy,
}


# Map of impl class names → reason for explicit refusal. Listed rather
# than allowing auto-fallback because silent wrong-logit risk is
# unacceptable for an inference engine. To support a backend listed here,
# add a strategy class + register in `_STRATEGY_BY_NAME` and remove the
# entry below.
_MLA_REASON = (
    "MLA — compressed KV with projection-on-read; the standard "
    "(num_blocks, page, kv_heads, head_dim) gather assumption is invalid."
)
_FA_REASON = (
    "FlashAttention V1's `custom_mask` path is wired only for spec-decode "
    "extraction, not arbitrary patterns; passing pie's BRLE mask would be "
    "silently dropped."
)
_TRITON_REASON = (
    "Triton attention kernel is causal+sliding-window only; no kernel "
    "parameter for arbitrary masks."
)

_UNSUPPORTED_IMPLS: dict[str, str] = {
    # ---- MLA family (DeepSeek-style compressed KV) ----
    "MLACommonImpl":           _MLA_REASON,
    "FlashInferMLAImpl":       _MLA_REASON,
    "FlashInferMLASparseImpl": _MLA_REASON,
    "FlashAttnMLAImpl":        _MLA_REASON,
    "FlashMLAImpl":            _MLA_REASON,
    "FlashMLASparseImpl":      _MLA_REASON,
    "CutlassMLAImpl":          _MLA_REASON,
    "TritonMLAImpl":           _MLA_REASON,
    "AiterMLAImpl":            _MLA_REASON,
    "AiterTritonMLAImpl":      _MLA_REASON,
    "ROCMAiterMLASparseImpl":  _MLA_REASON,
    "XPUMLASparseImpl":        _MLA_REASON,
    "SparseMLAAttentionImpl":  _MLA_REASON,

    # ---- FlashAttention V1 / variants ----
    "FlashAttentionImpl":           _FA_REASON,
    "FlashAttentionDiffKVImpl":     _FA_REASON,
    "AiterFlashAttentionImpl":      _FA_REASON,
    "RocmAttentionImpl":            _FA_REASON,
    "RocmAiterUnifiedAttentionImpl": _FA_REASON,

    # ---- Triton (no arbitrary mask) ----
    "TritonAttentionImpl": _TRITON_REASON,

    # ---- FlexAttention — could be supported via mask_mod, not yet wired ----
    "FlexAttentionImpl": (
        "PyTorch FlexAttention. Could support arbitrary masks via mask_mod "
        "but no strategy is registered yet. Add `_FlexStrategy` + register "
        "if you need it; until then, switch to `FLASHINFER`."
    ),

    # ---- Speculative-decoding tree attention (custom semantics) ----
    "TreeAttentionImpl": (
        "Tree attention used for speculative decoding; mask is determined "
        "by the spec tree, not by inferlet input."
    ),

    # ---- Quantized + CPU paths ----
    "TurboQuantAttentionImpl": (
        "Quantized attention; pie's mask path assumes unquantized K/V."
    ),
    "CPUAttentionBackendImpl": (
        "CPU attention backend isn't on the production path; no strategy."
    ),
}


def _resolve_strategy(impl_cls: type) -> _Strategy:
    """Return a strategy instance for `impl_cls`, or raise."""
    name = impl_cls.__name__
    if cls := _STRATEGY_BY_NAME.get(name):
        return cls()
    if reason := _UNSUPPORTED_IMPLS.get(name):
        raise NotImplementedError(
            f"pie_backend_vllm: refusing to load with attention impl "
            f"{name!r}: {reason} Pie's runtime always emits a custom_mask "
            f"buffer; ignoring it would silently produce wrong tokens for "
            f"inferlets that depend on non-causal attention. Pick a "
            f"verified backend instead: {sorted(_STRATEGY_BY_NAME)}."
        )
    raise NotImplementedError(
        f"pie_backend_vllm: no mask strategy registered for impl {name!r}. "
        f"Either add a strategy in pie_backend_vllm/mask_strategies.py "
        f"(see _FlashInferStrategy as a template) or pick a verified "
        f"backend: {sorted(_STRATEGY_BY_NAME)}."
    )


# ----------------------------------------------------------------------------
# Proxy — wraps the impl, intercepts forward, delegates everything else
# ----------------------------------------------------------------------------


class PieMaskedAttentionImplProxy:
    """Wraps a vllm AttentionImpl, replacing `forward` with mask-aware
    dispatch and delegating every other attribute access to the wrapped
    instance.

    Read paths like `impl.scale`, `impl.kv_sharing_target_layer_name`,
    `impl.do_kv_cache_update(...)` continue to work via `__getattr__`.

    Detection: pie's own code looks for `_pie_proxy_marker` (class attr,
    leading underscore = internal contract) rather than `isinstance(...,
    PieMaskedAttentionImplProxy)` because the proxy fakes its `__class__`
    for vllm's benefit (see the property below) and isinstance against
    the proxy class returns False.
    """

    # Sentinel class attribute pie's own install/discovery code reads.
    # Not for external use; underscore signals the internal contract.
    _pie_proxy_marker = True

    __slots__ = ("_impl", "_strategy", "_dispatch_super")

    def __init__(self, impl: Any, strategy: _Strategy):
        # Bypass __setattr__ via object.__setattr__; the slots are hard.
        object.__setattr__(self, "_impl", impl)
        object.__setattr__(self, "_strategy", strategy)
        # Captured once at install time — the wrapped impl's forward, in
        # case of further subclassing in vllm's chain.
        object.__setattr__(self, "_dispatch_super", impl.forward)

    # vllm internals (e.g., `get_per_layer_parameters`) call
    # `isinstance(layer.impl, FlashInferImpl)` to pull layer config.
    # Pretend to be the wrapped impl for those checks. `type(proxy)` still
    # returns `PieMaskedAttentionImplProxy`, but `isinstance(proxy, X)`
    # returns whatever `isinstance(impl, X)` would have.
    @property
    def __class__(self):  # type: ignore[override]
        return type(self._impl)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        *args,
        **kwargs,
    ):
        from ._vllm_compat import get_attention_context, get_forward_context

        extras = get_forward_context().additional_kwargs.get("pie_attn_extras")
        if extras is None:
            # Zero-overhead pass-through: vllm's normal flow runs unchanged.
            return self._dispatch_super(
                layer, query, key, value, kv_cache, attn_metadata, output,
                *args, **kwargs,
            )

        # Most V1 backends have `forward_includes_kv_cache_update=False`, in
        # which case vllm's `unified_kv_cache_update` ran before us and KV is
        # already written. Backends with `=True` write KV inside their
        # forward; since we skip super().forward, we do it ourselves.
        impl = self._impl
        if getattr(layer.attn_backend, "forward_includes_kv_cache_update", False):
            _, _, _, layer_slot_mapping = get_attention_context(layer.layer_name)
            if (
                impl.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
                and layer_slot_mapping is not None
                and hasattr(impl, "do_kv_cache_update")
            ):
                impl.do_kv_cache_update(layer, key, value, kv_cache, layer_slot_mapping)

        self._strategy.run(
            impl, layer, query, key, value, kv_cache, attn_metadata, output,
            extras,
        )

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal attribute lookup fails,
        # so methods/attrs defined directly on the proxy class (forward,
        # _impl, _strategy, _dispatch_super) bypass this entirely.
        return getattr(self._impl, name)


# ----------------------------------------------------------------------------
# Installer
# ----------------------------------------------------------------------------


def install_mask_strategies(vllm_config) -> None:
    """Wrap each attention layer's impl in a `PieMaskedAttentionImplProxy`.

    Idempotent — re-wrapping a proxy is a no-op.

    Refuses (NotImplementedError) on impls without a registered strategy:
    silent fallback risks wrong logits for inferlets that depend on
    non-causal attention patterns.
    """
    from ._vllm_compat import AttentionLayerBase

    fc = vllm_config.compilation_config.static_forward_context
    for _name, layer in fc.items():
        if not isinstance(layer, AttentionLayerBase):
            continue

        impl = layer.impl
        if getattr(impl, "_pie_proxy_marker", False):
            continue  # already wrapped

        strategy = _resolve_strategy(type(impl))
        layer.impl = PieMaskedAttentionImplProxy(impl, strategy)


def first_attention_strategy(vllm_config) -> _Strategy | None:
    """Return the first attention layer's strategy, or None if no layers
    are wrapped. Used by `forward_pass.transform()` to discover whether
    the FlashInfer wrapper needs pre-planning this batch.

    Uses `_pie_proxy_marker` rather than `isinstance(impl, Proxy)` because
    the proxy fakes its `__class__` for vllm's benefit (see the property
    on `PieMaskedAttentionImplProxy`).
    """
    from ._vllm_compat import AttentionLayerBase

    fc = vllm_config.compilation_config.static_forward_context
    for _name, layer in fc.items():
        if not isinstance(layer, AttentionLayerBase):
            continue
        impl = layer.impl
        if getattr(impl, "_pie_proxy_marker", False):
            return impl._strategy
    return None
