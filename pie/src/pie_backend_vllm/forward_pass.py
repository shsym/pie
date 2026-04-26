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

        with set_forward_context(
            attn_metadata=backend_metadata,
            vllm_config=self.vllm_config,
            num_tokens=common.num_actual_tokens,
            slot_mapping=slot_mapping_dict,
        ):
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
