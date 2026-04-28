"""CMA-ES adapter injection for the sglang driver.

We follow sglang's own LoRA pattern: at engine load time, walk
`runner.model.layers[*].self_attn.qkv_proj` and class-swap each one for a
`QKVAdapterWrapper` that:
  1. delegates to the original `QKVParallelLinear` for the base projection,
  2. if a per-batch `AdapterSubpass` is active (set by the forward path
     via `SubpassSlot`), invokes `subpass.execute(layer_idx, hidden, q, k, v)`
     to add the noisy DOWN/UP-projection contribution in-place.

The wrapper returns the same `(qkv, bias)` tuple shape sglang's caller
code expects (see sglang/python/sglang/srt/models/llama.py:201), so no
changes to sglang's model files are needed.

`SubpassSlot` is the single shared mailbox that connects
`SGLangForwardPass.transform()` (writer) to all wrapper instances
(readers). Holding only a non-tensor pointer keeps it CUDA-graph-safe —
the graph captures the wrapper's `forward()` body, which dereferences
`self._slot.current` per call; only the tensor data inside the subpass
participates in the captured kernels.

v1 scope:
  * Llama-family only (any architecture whose attention exposes
    `self_attn.qkv_proj` as `QKVParallelLinear` works automatically).
  * TP=1 only — matches the constraint already enforced for the LM head
    in `forward_pass.py:82-84`. Multi-rank shard offsets in
    `AdapterSubpass.execute` work, but installing the wrapper on a
    `QKVParallelLinear` configured for TP>1 hasn't been verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class SubpassSlot:
    """One-element mailbox connecting `forward_pass.transform` to wrappers.

    The forward path sets `current` to the active `AdapterSubpass` before
    `runner.forward()` and clears it in `finally`. Wrapper instances read
    it on every forward call. Single-process / single-engine, so no
    locking is needed.
    """

    current: object | None = None


class QKVAdapterWrapper(nn.Module):
    """Wraps an sglang `QKVParallelLinear`. Pass-through unless an
    `AdapterSubpass` is active in `slot`, in which case the adapter's
    in-place noisy contribution is added to Q / K / V before returning.

    `qkv_proj.forward(x)` returns `(qkv: Tensor[B, q+k+v], bias)`. We
    split the output into views (no copy) and hand them to
    `subpass.execute(layer_idx, x, q, k, v)`, which writes back through
    the views; the contiguous `qkv` tensor returned to sglang has the
    adapter contribution included.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        layer_idx: int,
        slot: SubpassSlot,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.layer_idx = layer_idx
        self._slot = slot
        # `q_size` / `kv_size` come from the parent `LlamaAttention`;
        # `QKVParallelLinear` exposes equivalent info via `output_sizes`
        # (a 3-element list [q, k, v] of LOCAL output dims after TP).
        # Reading from base_layer keeps this wrapper self-contained.
        sizes = list(base_layer.output_sizes)
        if len(sizes) != 3:
            raise ValueError(
                f"QKVAdapterWrapper expects QKVParallelLinear with 3 output "
                f"sections (q,k,v); got output_sizes={sizes!r}"
            )
        self._q_size, self._k_size, self._v_size = sizes

    def forward(self, hidden_states: torch.Tensor):
        qkv, bias = self.base_layer(hidden_states)
        subpass = self._slot.current
        if subpass is not None:
            q, k, v = qkv.split(
                [self._q_size, self._k_size, self._v_size], dim=-1,
            )
            # `execute` writes through the q/k/v views via the rand_mv
            # `out=` / `beta=1.0` path, leaving `qkv` updated in-place.
            subpass.execute(self.layer_idx, hidden_states, q, k, v)
        return qkv, bias


def install_adapter_wrappers(
    runner, slot: SubpassSlot,
) -> int:
    """Walk `runner.model.layers[*].self_attn.qkv_proj` and class-swap each
    `QKVParallelLinear` for a `QKVAdapterWrapper`. Returns the number of
    wrappers installed (== number of decoder layers for Llama-family).

    Idempotent: re-running on a model that already has wrappers detects
    them and skips, so calling it twice is safe.
    """
    from sglang.srt.utils.common import replace_submodule

    layers = getattr(runner.model, "layers", None)
    if layers is None:
        # Some sglang archs nest the layer list one level deeper
        # (e.g., `runner.model.model.layers`). Probe.
        inner = getattr(runner.model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "install_adapter_wrappers: could not locate model.layers on "
            f"{type(runner.model).__name__}; sglang model layout differs "
            "from what we expect."
        )

    n_installed = 0
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        qkv = getattr(attn, "qkv_proj", None)
        if qkv is None:
            continue
        if isinstance(qkv, QKVAdapterWrapper):
            continue  # idempotent

        wrapper = QKVAdapterWrapper(qkv, layer_idx=layer_idx, slot=slot)

        # Build the dotted path so replace_submodule can locate the parent.
        # We didn't track the prefix during the walk, so do it the simple
        # way: look up the layer's attribute name on its parent. For
        # Llama, layers are an nn.ModuleList named `layers`, so the path
        # is `layers.{layer_idx}.self_attn.qkv_proj` (or
        # `model.layers.{layer_idx}.self_attn.qkv_proj` for nested archs).
        path = _resolve_qkv_path(runner.model, layer_idx)
        replace_submodule(runner.model, path, wrapper)
        n_installed += 1

    return n_installed


def _resolve_qkv_path(root: nn.Module, layer_idx: int) -> str:
    """Best-effort resolution of the `qkv_proj` dotted path for layer
    `layer_idx`. Tries the two layouts we know about (`layers.N...` and
    `model.layers.N...`) and returns whichever matches an existing
    submodule. Raises if neither hits.
    """
    candidates = [
        f"layers.{layer_idx}.self_attn.qkv_proj",
        f"model.layers.{layer_idx}.self_attn.qkv_proj",
    ]
    for cand in candidates:
        try:
            root.get_submodule(cand)
            return cand
        except AttributeError:
            continue
    raise RuntimeError(
        f"_resolve_qkv_path: layer {layer_idx} qkv_proj not at any of "
        f"{candidates!r}; the sglang model's layer naming differs."
    )
