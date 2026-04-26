"""Adapter that exposes a vllm model under pie_backend's ForwardPass contract.

`pie_backend.engine.Engine` calls three methods on its `forward_pass` object:
`embed_inputs(inputs) -> hidden`, `transform(...) -> hidden`, and
`sample(hidden, sampling_metadata) -> list`. We satisfy that contract here
while delegating the actual compute to vllm.
"""

from __future__ import annotations

from typing import Any

import torch

from pie_backend.config import RuntimeConfig
from pie_backend.model.common import sample_common

from .attn_metadata import build_common_metadata


class VllmForwardPass:
    """Thin shim around a vllm model.

    `embed_inputs` runs the model's input-embedding layer.

    `transform` builds vllm-style attention metadata from pie's batch dicts,
    enters `set_forward_context`, and runs the vllm model's forward.

    `sample` reuses pie_backend's `sample_common`, calling the vllm model's
    `compute_logits` as the LM head.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        vllm_config: Any,
        attn_backend: Any,
        runtime_config: RuntimeConfig,
        model_config: Any,
    ):
        self.model = model
        self.vllm_config = vllm_config
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.device = torch.device(runtime_config.device)

        # Lazy: built on first transform() call so we know the resolved
        # backend (set during model construction inside set_current_vllm_config).
        self._builder = None
        self._kv_spec = None
        self._layer_names: list[str] = []
        # Per-model attention shape (uniform across layers for plain LLMs;
        # mixed-attention architectures would need per-layer plans, out of
        # scope today). Captured during _ensure_metadata_builder().
        self._num_qo_heads: int | None = None
        self._num_kv_heads: int | None = None
        self._head_dim_qk: int | None = None
        # True when the installed mask-aware impl declares it consumes the
        # FlashInfer prefill wrapper. Set during _ensure_metadata_builder.
        self._impl_uses_flashinfer_wrapper: bool = False
        # Mask plumbing: lazily allocated FlashInfer wrapper, planned once
        # per batch in transform() when the FlashInfer fast path is active.
        self._mask_wrapper: Any | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _ensure_metadata_builder(self) -> None:
        """Construct the per-backend AttentionMetadataBuilder once.

        We call this lazily (not in __init__) because vllm's `current_vllm_config`
        context must be active and the model fully constructed before
        `get_kv_cache_spec` is sound.
        """
        if self._builder is not None:
            return

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.model_executor.models.utils import extract_layer_index

        fc = self.vllm_config.compilation_config.static_forward_context
        attn_layers = [
            (name, layer) for name, layer in fc.items()
            if isinstance(layer, AttentionLayerBase)
        ]
        # Sort by extracted index so layer ordering is stable & matches
        # pie's swap RPC indexing.
        attn_layers.sort(key=lambda x: extract_layer_index(x[0]))

        self._layer_names = [name for name, _ in attn_layers]
        first_layer = attn_layers[0][1]
        backend = first_layer.attn_backend

        # Capture attention shape for FlashInfer plan(). Vllm's Attention
        # carries `num_heads` (post-TP) and `num_kv_heads`; head_size is
        # uniform across Q/K for non-MLA architectures.
        self._num_qo_heads = int(first_layer.num_heads)
        self._num_kv_heads = int(first_layer.num_kv_heads)
        self._head_dim_qk = int(first_layer.head_size)

        # The installed mask-aware impl flags whether it needs the
        # FlashInfer prefill wrapper pre-planned. We look at the first
        # layer; mixed-impl models would need a per-layer flag, out of
        # scope today.
        self._impl_uses_flashinfer_wrapper = bool(
            getattr(type(first_layer.impl), "_pie_uses_flashinfer_wrapper", False)
        )

        with set_current_vllm_config(self.vllm_config):
            self._kv_spec = first_layer.get_kv_cache_spec(self.vllm_config)
            builder_cls = backend.get_builder_cls()
            self._builder = builder_cls(
                kv_cache_spec=self._kv_spec,
                layer_names=self._layer_names,
                vllm_config=self.vllm_config,
                device=self.device,
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def embed_inputs(self, inputs: dict) -> torch.Tensor:
        """Run the model's input-embedding layer."""
        token_ids = inputs["token_ids"].to(self.device, non_blocking=True)
        return self.model.embed_input_ids(token_ids)

    def transform(
        self,
        *,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        total_pages_cpu: int = 0,
    ) -> torch.Tensor:
        """Run the model's transformer trunk inside `set_forward_context`."""
        from vllm.forward_context import set_forward_context

        self._ensure_metadata_builder()

        page_size = self._kv_spec.block_size

        common = build_common_metadata(
            qo_indptr=qo_indptr,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            page_size=page_size,
            device=self.device,
        )

        # Backend-specific metadata. `common_prefix_len=0` disables cascade
        # attention (we don't use it from pie's side).
        backend_metadata = self._builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
        )

        # Same slot_mapping for every layer (no cross-attention or shared KV).
        slot_mapping_dict = {
            name: common.slot_mapping for name in self._layer_names
        }

        # vllm's RoPE kernel expects int64 positions; pie uses int32.
        positions = position_ids.to(self.device, dtype=torch.int64, non_blocking=True)

        # Mask plumbing: only when there's something to apply. Mask-aware
        # `AttentionImpl` subclasses (installed at engine load) read
        # `pie_attn_extras` from additional_kwargs; an absent key is the
        # zero-overhead signal — those subclasses just call super().forward.
        pie_attn_extras = None
        if custom_mask is not None:
            from .mask_compute import (
                PieAttnExtras, make_flashinfer_wrapper, plan_flashinfer_wrapper,
            )

            pie_attn_extras = PieAttnExtras.build(
                custom_mask=custom_mask,
                query_start_loc=common.query_start_loc,
                seq_lens=common.seq_lens,
                block_table=common.block_table_tensor,
                page_size=page_size,
                device=self.device,
            )

            # Pre-plan FlashInfer wrapper ONCE for this batch IF the active
            # impl uses it. Each masked subclass declares this via the
            # `_pie_uses_flashinfer_wrapper` class attribute. Native pie
            # follows the same shape: plan() per batch, run() per layer.
            if self._impl_uses_flashinfer_wrapper:
                if self._mask_wrapper is None:
                    self._mask_wrapper = make_flashinfer_wrapper(
                        128 * 1024 * 1024, self.device,
                    )
                plan_flashinfer_wrapper(
                    self._mask_wrapper,
                    extras=pie_attn_extras,
                    kv_page_indices=kv_page_indices,
                    kv_page_indptr=kv_page_indptr,
                    kv_last_page_lens=kv_last_page_lens,
                    num_qo_heads=self._num_qo_heads,
                    num_kv_heads=self._num_kv_heads,
                    head_dim_qk=self._head_dim_qk,
                    q_data_type=input_embeds.dtype,
                )
                pie_attn_extras.flashinfer_wrapper = self._mask_wrapper

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=common.num_actual_tokens,
            slot_mapping=slot_mapping_dict,
        ):
            # `set_forward_context` takes its `additional_kwargs` from the
            # platform hook, not from a kwarg, so inject ours after entering.
            if pie_attn_extras is not None:
                from vllm.forward_context import get_forward_context

                get_forward_context().additional_kwargs["pie_attn_extras"] = pie_attn_extras

            hidden_states = self.model.forward(
                input_ids=None,
                positions=positions,
                inputs_embeds=input_embeds,
            )

        return hidden_states

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: dict) -> dict:
        """Sample via pie_backend's sample_common; vllm's LM head is the lm_head_fn."""
        return sample_common(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
            lm_head_fn=self.model.compute_logits,
            device=self.device,
            dtype=self.runtime_config.activation_dtype,
        )
