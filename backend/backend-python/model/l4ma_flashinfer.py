"""FlashInfer-backed runtime implementation for the L4MA architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from common import L4maArch
from common_model.l4ma_runtime import L4maBackend, L4maForwardContext, RuntimeInputs

try:  # pragma: no cover - optional dependency guard
    import flashinfer as ops  # type: ignore[import]

    # FlashInfer wrapper types - using object as fallback for type checking
    # Union type not available when FlashInfer is missing
    FlashInferWrapper = object  # type: ignore[misc]
except ImportError:  # pragma: no cover - optional dependency guard
    ops = None
    FlashInferWrapper = object  # type: ignore[misc]


def _infer_page_size(kv_cache_at_layer) -> int:
    if not kv_cache_at_layer:
        raise ValueError("kv_cache_at_layer must contain at least one tensor")
    first_layer = kv_cache_at_layer[0]
    if first_layer.ndim < 3:
        raise ValueError("Unexpected KV cache tensor shape; expected >= 3 dimensions")
    return int(first_layer.shape[2])


@dataclass(frozen=True)
class FlashInferRuntimeMetadata:
    """Metadata describing the prepared FlashInfer execution state."""

    page_size: int
    batch_indices: torch.Tensor
    batch_positions: torch.Tensor


class _FlashInferForwardContext(L4maForwardContext):
    """FlashInfer-backed forward context implementation."""

    def __init__(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
        wrapper: FlashInferWrapper,  # type: ignore[name-defined]
        kv_layout: str,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        metadata: FlashInferRuntimeMetadata,
    ) -> None:
        self._config = config
        self._inputs = inputs
        self.wrapper = wrapper
        self._kv_layout = kv_layout
        self._batch_indices = batch_indices
        self._batch_positions = batch_positions
        self._metadata = metadata

    @property
    def batch_indices(self) -> torch.Tensor:
        """Get the batch indices tensor."""
        return self._batch_indices

    @property
    def batch_positions(self) -> torch.Tensor:
        """Get the batch positions tensor."""
        return self._batch_positions

    @property
    def metadata(self) -> FlashInferRuntimeMetadata:
        """Get the runtime metadata."""
        return self._metadata

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        """Apply RoPE encoding to query and key states."""
        if ops is None:
            raise RuntimeError("FlashInfer not available")
        ops.apply_llama31_rope_pos_ids_inplace(  # type: ignore
            q=query_states,
            k=key_states,
            pos_ids=position_ids,
        )

    def append_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> None:
        """Append key and value states to the KV cache."""
        _ = layer_idx  # Parameter not currently used
        if ops is None:
            raise RuntimeError("FlashInfer not available")
        ops.append_paged_kv_cache(  # type: ignore
            append_key=key_states,
            append_value=value_states,
            batch_indices=self._batch_indices,
            positions=self._batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=self._inputs.kv_page_indices,
            kv_indptr=self._inputs.kv_page_indptr,
            kv_last_page_len=self._inputs.kv_last_page_lens,
            kv_layout=self._kv_layout,
        )

    def run_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Run attention computation using FlashInfer."""
        _ = layer_idx  # Parameter not currently used
        attn_output = self.wrapper.run(query_states, kv_cache_layer)
        return attn_output.reshape(attn_output.size(0), -1)


class FlashInferL4maBackend(L4maBackend):
    """Default FlashInfer implementation of the L4MA runtime backend."""

    @staticmethod
    def is_available() -> bool:
        """Return True if the FlashInfer runtime dependency is present."""
        return ops is not None

    def __init__(
        self, workspace_size_bytes: int = 128 * 1024 * 1024, kv_layout: str = "NHD"
    ) -> None:
        if ops is None:
            raise RuntimeError(
                "flashinfer is not available. "
                "Install flashinfer-python to use FlashInferL4maBackend."
            )

        self.workspace_size_bytes = workspace_size_bytes
        self.kv_layout = kv_layout

        self._workspace_buffer: Optional[torch.Tensor] = None
        # Type ignore: FlashInfer types are optional dependencies not available in CI
        self._decode_wrapper: Optional[  # type: ignore[name-defined]
            ops.BatchDecodeWithPagedKVCacheWrapper  # type: ignore[name-defined]
        ] = None
        self._prefill_wrapper: Optional[  # type: ignore[name-defined]
            ops.BatchPrefillWithPagedKVCacheWrapper  # type: ignore[name-defined]
        ] = None

    def _ensure_workspace(self, device: torch.device | str) -> None:
        tensor_device = torch.device(device)
        if (
            self._workspace_buffer is not None
            and self._workspace_buffer.device == tensor_device
        ):
            return

        self._workspace_buffer = torch.empty(
            self.workspace_size_bytes,
            dtype=torch.uint8,
            device=tensor_device,
        )
        if ops is None:
            raise RuntimeError("FlashInfer not available")
        self._decode_wrapper = ops.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_buffer, self.kv_layout
        )
        self._prefill_wrapper = ops.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace_buffer, self.kv_layout
        )

    def create_forward_context(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
    ) -> L4maForwardContext:
        """Create a forward context for FlashInfer execution."""
        self._ensure_workspace(config.device)

        if ops is None:
            raise RuntimeError("FlashInfer not available")

        page_size = _infer_page_size(inputs.kv_cache_at_layer)
        seq_lens = ops.get_seq_lens(
            inputs.kv_page_indptr,
            inputs.kv_last_page_lens,
            page_size,
        )
        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=inputs.qo_indptr,
            seq_lens=seq_lens,
            nnz=inputs.num_tokens,
        )

        if inputs.single_token_inference_mode:
            wrapper = self._decode_wrapper
            assert wrapper is not None
            wrapper.plan(
                indptr=inputs.kv_page_indptr,
                indices=inputs.kv_page_indices,
                last_page_len=inputs.kv_last_page_lens,
                num_qo_heads=config.num_query_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_size,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=config.dtype,
            )
        else:
            wrapper = self._prefill_wrapper
            assert wrapper is not None
            wrapper.plan(
                qo_indptr=inputs.qo_indptr,
                paged_kv_indptr=inputs.kv_page_indptr,
                paged_kv_indices=inputs.kv_page_indices,
                paged_kv_last_page_len=inputs.kv_last_page_lens,
                num_qo_heads=config.num_query_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim_qk=config.head_size,
                page_size=page_size,
                custom_mask=inputs.custom_mask,
                q_data_type=config.dtype,
            )

        metadata = FlashInferRuntimeMetadata(
            page_size=page_size,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
        )

        return _FlashInferForwardContext(
            config=config,
            inputs=inputs,
            wrapper=wrapper,
            kv_layout=self.kv_layout,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            metadata=metadata,
        )


__all__ = [
    "FlashInferL4maBackend",
    "FlashInferRuntimeMetadata",
]
