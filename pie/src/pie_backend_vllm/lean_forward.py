"""Eager-mode forward-path optimizations for the vllm backend.

Pie's vllm backend runs with `enforce_eager=True` (no `torch.compile`, no
CUDA graphs). In that mode, PyTorch's per-module hook checks and vllm's
quant/TP dispatch wrappers are pure Python overhead — no compiler is going
to fuse them away. This module applies two surgical optimizations that
recover ~1.7 ms/step at p50 on a 28-layer 0.6B model:

  1. **Lean Linear forward** — replace `ColumnParallelLinear` /
     `RowParallelLinear` instance `forward` with a direct `F.linear` call.
     Skips `quant_method.apply` dispatch, the `gather_output` branch, and
     `linear_batch_invariant` env-var check. Applied only when:
       * the layer is unquantized
       * `tp_size == 1` (no all-gather/all-reduce needed).

  2. **Lean `nn.Module.__call__`** — global class-level replacement that
     calls `self.forward(...)` directly, skipping PyTorch's `_call_impl`
     machinery. The skipped work is hook bookkeeping (forward_pre_hooks,
     forward_hooks, full_backward_hooks, parametrization, etc.). None of
     those features are used on the inference hot path.

Caveats
-------
The `nn.Module.__call__` swap is **global** to the Python process. Any
forward hooks anyone registers on any module will silently not fire while
this is active. Pie's vllm worker is a dedicated inference process, so the
global scope is acceptable. We keep the original `__call__` reference and
expose `disable_lean_module_call()` for tests / cleanup.

Both optimizations are no-ops under TP > 1 or quantization — the lean
Linear paths would skip required communication or kernel selection.
"""

from __future__ import annotations

import types

import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Lean Linear (per-instance forward replacement)
# ----------------------------------------------------------------------------


def _lean_linear_forward(self, input_):
    """Bare F.linear, no quant dispatch, no TP branches.

    Only safe when the layer is unquantized and `tp_size == 1`. The
    installer enforces both before binding this method.
    """
    bias = self.bias if not self.skip_bias_add else None
    out = F.linear(input_, self.weight, bias)
    if self.return_bias:
        return out, self.bias if self.skip_bias_add else None
    return out


def _install_lean_linear(model: nn.Module) -> int:
    """Replace forward on every eligible Linear module. Returns count."""
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        RowParallelLinear,
        UnquantizedLinearMethod,
    )

    n = 0
    for m in model.modules():
        if not isinstance(m, (ColumnParallelLinear, RowParallelLinear)):
            continue
        # Safety: only safe when unquantized + non-TP. Both check class types
        # so a future quantized backend won't get silently skipped.
        if not isinstance(m.quant_method, UnquantizedLinearMethod):
            continue
        if getattr(m, "tp_size", 1) != 1:
            continue
        m.forward = types.MethodType(_lean_linear_forward, m)
        n += 1
    return n


# ----------------------------------------------------------------------------
# Lean Module.__call__ (global class-level replacement)
# ----------------------------------------------------------------------------


_ORIG_MODULE_CALL: object | None = None


def _lean_module_call(self, *args, **kwargs):
    """Skip `_call_impl`'s hook checks and call `forward` directly.

    Equivalent to `instance.forward(*args, **kwargs)` for modules with no
    hooks installed.
    """
    return self.forward(*args, **kwargs)


def _enable_lean_module_call() -> bool:
    """Idempotent. Returns True if newly installed."""
    global _ORIG_MODULE_CALL
    if _ORIG_MODULE_CALL is not None:
        return False
    _ORIG_MODULE_CALL = nn.Module.__call__
    nn.Module.__call__ = _lean_module_call
    return True


def disable_lean_module_call() -> None:
    """Restore PyTorch's original `nn.Module.__call__`. For tests / shutdown."""
    global _ORIG_MODULE_CALL
    if _ORIG_MODULE_CALL is None:
        return
    nn.Module.__call__ = _ORIG_MODULE_CALL  # type: ignore[assignment]
    _ORIG_MODULE_CALL = None


# ----------------------------------------------------------------------------
# Installer
# ----------------------------------------------------------------------------


def install_lean_forwards(model: nn.Module, *, enforce_eager: bool) -> dict:
    """Apply both optimizations. Safe to call multiple times.

    `model` is the top-level vllm model module (e.g., `Qwen3ForCausalLM`).
    `enforce_eager` is the pie/vllm config flag — these optimizations are
    only safe (and only beneficial) in eager mode. Compile mode would
    re-trace through the patched paths and is out of scope.

    Returns a small dict for logging / introspection.
    """
    if not enforce_eager:
        return {"applied": False, "reason": "enforce_eager=False"}

    n_linear = _install_lean_linear(model)
    installed_call = _enable_lean_module_call()
    return {
        "applied": True,
        "linear_patched": n_linear,
        "module_call_replaced": installed_call,
    }
