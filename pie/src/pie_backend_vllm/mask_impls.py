"""Mask-aware `AttentionImpl` subclasses, installed in place of vllm's default.

Architecture
------------
Vllm's V1 attention separates layer-level bookkeeping (`Attention.forward`)
from kernel dispatch (`AttentionImpl.forward`). Per backend, vllm ships an
impl: `FlashInferImpl`, `FlashAttentionImpl`, `TritonAttentionImpl`, ...

We extend each impl by *direct subclassing* — no mixin. Each subclass:

  * overrides `forward()` with a thin call into `_dispatch_masked_forward`,
    which checks for `pie_attn_extras` on the forward context;
  * overrides `_compute_with_mask()` to run the kernel-specific masked path.

`install_mask_aware_impls(vllm_config)` walks every attention layer and
re-types its impl in place via `instance.__class__ = MaskedSubclass`. The
subclass shares the impl's `__slots__ = ()` layout so the assignment is
ABI-compatible. Direct inheritance (not a mixin) is required for `__class__`
assignment to work — CPython's layout check rejects the mixin-first MRO
even when both classes declare empty slots.

Refusal policy
--------------
Impls without a registered subclass cause a NotImplementedError at engine
init. Pie's runtime always emits a custom_mask buffer; silently ignoring it
on an unverified backend would risk wrong tokens for inferlets that depend
on non-causal attention. Add a fast-path subclass + register in
`_explicit_subclasses()` to support a new backend.

Caveats
-------
* Backends with `forward_includes_kv_cache_update=True` integrate KV write
  into their forward; we replicate that explicitly. All real V1 backends
  today are `False`, so vllm's `unified_kv_cache_update` runs before us.
"""

from __future__ import annotations

from typing import Callable

from .mask_compute import PieAttnExtras, sdpa_gather_path


# ----------------------------------------------------------------------------
# The dispatch helper — every masked subclass calls this from its forward()
# ----------------------------------------------------------------------------


def _dispatch_masked_forward(
    impl,
    super_forward: Callable,
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
    """Fast path: pass-through. Slow path: KV write (if backend needs it)
    + per-backend `_compute_with_mask`.

    `super_forward` is the impl's original forward (bound to `impl`). Each
    subclass passes `super().forward` here so we don't need to know the impl
    class at this site.
    """
    from ._vllm_compat import get_attention_context, get_forward_context

    extras = get_forward_context().additional_kwargs.get("pie_attn_extras")
    if extras is None:
        return super_forward(
            layer, query, key, value, kv_cache, attn_metadata, output,
            *args, **kwargs,
        )

    # Most V1 backends have `forward_includes_kv_cache_update=False`, in which
    # case vllm's `unified_kv_cache_update` ran before us and KV is already
    # written. Backends with `=True` write KV inside their forward; since we
    # skip super().forward, we do it ourselves.
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

    impl._compute_with_mask(layer, query, kv_cache, extras, output)


# ----------------------------------------------------------------------------
# Per-backend fast paths
# ----------------------------------------------------------------------------


def _explicit_subclasses() -> dict[type, type]:
    """Build the registry of hand-written fast-path subclasses.

    Lazy at install time so vllm's per-backend modules aren't imported
    eagerly (they pull in heavy deps).
    """
    registry: dict[type, type] = {}

    from ._vllm_compat import get_flashinfer_impl

    try:
        FlashInferImpl = get_flashinfer_impl()
    except ImportError:
        FlashInferImpl = None  # type: ignore

    if FlashInferImpl is not None:

        class PieMaskedFlashInferImpl(FlashInferImpl):
            """FlashInfer fast path: run the wrapper pre-planned in transform().

            Setting `PIE_VLLM_MASK_FORCE_SDPA=1` falls back to the SDPA gather
            path. This is a debug knob for A/B-testing the universal fallback
            against the fast path on the same vllm backend; not a runtime
            production toggle.
            """

            __slots__ = ()
            _pie_uses_flashinfer_wrapper = True

            def forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, *args, **kwargs,
            ):
                return _dispatch_masked_forward(
                    self, super().forward,
                    layer, query, key, value, kv_cache, attn_metadata, output,
                    *args, **kwargs,
                )

            def _compute_with_mask(self, layer, query, kv_cache, extras, output):
                import os

                if os.environ.get("PIE_VLLM_MASK_FORCE_SDPA"):
                    sdpa_gather_path(
                        layer=layer, query=query, kv_cache=kv_cache,
                        extras=extras, output=output,
                    )
                    return
                if extras.flashinfer_wrapper is None:
                    raise RuntimeError(
                        "pie_backend_vllm: PieMaskedFlashInferImpl reached "
                        "_compute_with_mask but extras.flashinfer_wrapper is "
                        "None. forward_pass.transform() should have planned "
                        "the wrapper for this batch."
                    )
                extras.flashinfer_wrapper.run(query, kv_cache, out=output)

        registry[FlashInferImpl] = PieMaskedFlashInferImpl

    return registry


# ----------------------------------------------------------------------------
# Unsupported backends — refuse at engine init with a descriptive error.
# ----------------------------------------------------------------------------


# Map of impl class names → reason. Listed explicitly rather than auto-
# falling-back to SDPA because silent wrong-logit risk is unacceptable for
# an inference engine. New backends require a hand-written subclass +
# entry in `_explicit_subclasses()`.
_UNSUPPORTED_IMPLS: dict[str, str] = {
    "MLAImpl": "MLA — compressed KV / projection-on-read; standard gather invalid.",
    "MLACommonImpl": "MLA — compressed KV / projection-on-read; standard gather invalid.",
    "FlashInferMLAImpl": "DeepSeek MLA via FlashInfer — no custom_mask path.",
    "TritonMLAImpl": "DeepSeek MLA via Triton — no custom_mask path.",
    "FlashMLAImpl": "DeepSeek MLA — no custom_mask path.",
    "CutlassMLAImpl": "DeepSeek MLA via CUTLASS — no custom_mask path.",
}


def _verified_impl_names(explicit: dict[type, type]) -> list[str]:
    return sorted(impl_cls.__name__ for impl_cls in explicit)


# ----------------------------------------------------------------------------
# Installer
# ----------------------------------------------------------------------------


def _is_pie_masked(cls: type) -> bool:
    """A class is mask-aware if its name follows our convention."""
    return cls.__name__.startswith("PieMasked")


def _resolve_masked_class(
    impl_cls: type,
    *,
    explicit: dict[type, type],
) -> type:
    if _is_pie_masked(impl_cls):
        return impl_cls
    if impl_cls in explicit:
        return explicit[impl_cls]

    name = impl_cls.__name__
    verified = _verified_impl_names(explicit)
    if reason := _UNSUPPORTED_IMPLS.get(name):
        raise NotImplementedError(
            f"pie_backend_vllm: refusing to load with attention impl "
            f"{name!r}: {reason} Pie's runtime always emits a custom_mask "
            f"buffer; ignoring it would silently produce wrong tokens for "
            f"inferlets that depend on non-causal attention. Pick a "
            f"verified backend instead: {verified}."
        )
    raise NotImplementedError(
        f"pie_backend_vllm: no mask strategy registered for impl {name!r}. "
        f"Either add a subclass in pie_backend_vllm/mask_impls.py "
        f"(see PieMaskedFlashInferImpl as a template) or pick a verified "
        f"backend: {verified}."
    )


def install_mask_aware_impls(vllm_config) -> None:
    """Re-type each attention layer's impl to a mask-aware subclass.

    Idempotent — re-typing to the same class is a no-op.

    Refuses (NotImplementedError) on impls without a registered mask
    strategy: silent fallback risks wrong logits for inferlets that
    depend on non-causal attention patterns.
    """
    from ._vllm_compat import AttentionLayerBase

    explicit = _explicit_subclasses()

    fc = vllm_config.compilation_config.static_forward_context
    for _name, layer in fc.items():
        if not isinstance(layer, AttentionLayerBase):
            continue

        impl = layer.impl
        masked_cls = _resolve_masked_class(type(impl), explicit=explicit)
        if type(impl) is masked_cls:
            continue
        try:
            impl.__class__ = masked_cls
        except TypeError as exc:
            raise RuntimeError(
                f"pie_backend_vllm: cannot install mask-aware subclass on "
                f"{type(impl).__name__} — likely a __slots__ layout mismatch. "
                "Verify the subclass uses direct inheritance (not a mixin) "
                "and declares `__slots__ = ()`."
            ) from exc
