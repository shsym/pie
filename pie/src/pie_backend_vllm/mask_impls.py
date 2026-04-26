"""Mask-aware `AttentionImpl` subclasses, installed in place of vllm's default.

Architecture
------------
Vllm's V1 attention separates layer-level bookkeeping (`Attention.forward`)
from kernel dispatch (`AttentionImpl.forward`). Per backend, vllm ships an
impl: `FlashInferImpl`, `FlashAttentionImpl`, `TritonAttentionImpl`, ...

We extend each impl by *direct subclassing* — no mixin. Each subclass:

  * overrides `forward()` with a thin call into `_dispatch_masked_forward`,
    which checks for `pie_attn_extras` on the forward context;
  * overrides `_compute_with_mask()` for a backend-specific fast path
    (FlashInfer wrapper, FlexAttention block-mask, etc.).

The default `_compute_with_mask` falls through to the universal SDPA gather,
so a subclass that overrides nothing is automatically correct (just slower).

`install_mask_aware_impls(vllm_config)` walks every attention layer and
re-types its impl in place via `instance.__class__ = MaskedSubclass`. The
subclass shares the impl's `__slots__ = ()` layout so the assignment is
ABI-compatible. Note that direct inheritance (not a mixin) is required for
`__class__` assignment to work — CPython's layout check rejects the
mixin-first MRO even when both classes declare empty slots.

Auto-synth
----------
Unknown impls get an auto-generated subclass via `_make_auto_subclass()` so
that any attention backend Pie hasn't seen before still works correctly,
just at SDPA speed. Add a hand-written subclass + register in
`_explicit_subclasses()` to expose a fast path.

Caveats
-------
* Backends with `forward_includes_kv_cache_update=True` integrate KV write
  into their forward; we replicate that explicitly. All real V1 backends
  today are `False`, so vllm's `unified_kv_cache_update` runs before us.
* MLA backends (compressed KV, projection-on-read) violate the "K/V live
  in the cache as-is" assumption of the SDPA gather; auto-synth would
  silently produce wrong logits, so we refuse them explicitly.
"""

from __future__ import annotations

from typing import Any, Callable

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
    from vllm.forward_context import get_forward_context

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
        from vllm.model_executor.layers.attention.attention import (
            get_attention_context,
        )

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


def _default_compute_with_mask(
    self,
    layer: Any,
    query,
    kv_cache,
    extras: PieAttnExtras,
    output,
) -> None:
    """Universal fallback installed on every masked subclass that doesn't
    override it. Works on any HW PyTorch supports."""
    sdpa_gather_path(
        layer=layer, query=query, kv_cache=kv_cache,
        extras=extras, output=output,
    )


# ----------------------------------------------------------------------------
# Per-backend fast paths
# ----------------------------------------------------------------------------


def _explicit_subclasses() -> dict[type, type]:
    """Build the registry of hand-written fast-path subclasses.

    Lazy at install time so vllm's per-backend modules aren't imported
    eagerly (they pull in heavy deps).
    """
    registry: dict[type, type] = {}

    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl
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
# Auto-synthesis for unknown impls
# ----------------------------------------------------------------------------


def _make_auto_subclass(impl_cls: type) -> type:
    """Build a default masked subclass for `impl_cls` — SDPA fallback only.

    Defined inside a function so `super()` inside the synthesized `forward`
    binds to the closure-time `__class__` cell.
    """

    class Auto(impl_cls):  # type: ignore[valid-type, misc]
        __slots__ = ()

        def forward(
            self, layer, query, key, value, kv_cache, attn_metadata, output,
            *args, **kwargs,
        ):
            return _dispatch_masked_forward(
                self, super().forward,
                layer, query, key, value, kv_cache, attn_metadata, output,
                *args, **kwargs,
            )

        _compute_with_mask = _default_compute_with_mask

    Auto.__name__ = f"PieMasked{impl_cls.__name__}"
    Auto.__qualname__ = Auto.__name__
    return Auto


# ----------------------------------------------------------------------------
# MLA refusal — auto-synth would gather garbage from compressed KV.
# ----------------------------------------------------------------------------


_MLA_BLOCKLIST_NAMES = (
    "MLAImpl",
    "MLACommonImpl",
    "FlashInferMLAImpl",
    "TritonMLAImpl",
    "FlashMLAImpl",
    "CutlassMLAImpl",
)


def _is_mla(impl_cls: type) -> bool:
    if impl_cls.__name__ in _MLA_BLOCKLIST_NAMES:
        return True
    for base in impl_cls.__mro__:
        if base.__name__ in _MLA_BLOCKLIST_NAMES:
            return True
    return False


# ----------------------------------------------------------------------------
# Installer
# ----------------------------------------------------------------------------


_AUTO_SYNTH_CACHE: dict[type, type] = {}


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
    if impl_cls in _AUTO_SYNTH_CACHE:
        return _AUTO_SYNTH_CACHE[impl_cls]
    if _is_mla(impl_cls):
        raise NotImplementedError(
            f"pie_backend_vllm: {impl_cls.__name__} is an MLA-style backend "
            "(compressed KV / projection-on-read). The SDPA gather fallback "
            "would read garbage from the cache. Custom masks on MLA require "
            "a dedicated _compute_with_mask implementation."
        )
    new_cls = _make_auto_subclass(impl_cls)
    _AUTO_SYNTH_CACHE[impl_cls] = new_cls
    return new_cls


def install_mask_aware_impls(vllm_config) -> None:
    """Re-type each attention layer's impl to a mask-aware subclass.

    Under compile mode (enforce_eager=False), we keep `use_direct_call=False`
    so vllm routes attention through the `torch.ops.vllm.unified_*` custom
    ops. Custom ops are opaque to Dynamo, which is what lets it trace through
    the model without diving into FlashInfer's wrappers (Dynamo can't follow
    them). Our patched `impl.forward` lives behind that opaque boundary too.

    Under eager mode (enforce_eager=True), `do_not_compile=True` short-
    circuits the @support_torch_compile path and Dynamo never runs — so we
    can flip `use_direct_call=True` to skip ~15 µs × 2 ops × N layers of
    dispatch overhead per step. The direct-call path invokes the same
    `unified_*` *functions* without the torch dispatch layer.

    Idempotent — re-typing to the same class is a no-op.
    """
    from vllm.config.compilation import CompilationMode
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    explicit = _explicit_subclasses()

    # Only safe to skip the custom-op dispatch when vllm isn't compiling.
    # With compile active, Dynamo would trace through our patched
    # impl.forward and fail on FlashInfer's wrappers.
    is_eager = vllm_config.compilation_config.mode == CompilationMode.NONE

    fc = vllm_config.compilation_config.static_forward_context
    for _name, layer in fc.items():
        if not isinstance(layer, AttentionLayerBase):
            continue

        if is_eager and hasattr(layer, "use_direct_call"):
            layer.use_direct_call = True

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
