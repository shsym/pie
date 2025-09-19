"""Factory helpers for constructing backend models and fusion maps."""

from __future__ import annotations

from config.common import ModelInfo


def create_model_and_fusion_map(model_info: ModelInfo):
    """Instantiate a model and its fusion map based on the architecture."""
    arch_type = model_info.architecture.type.lower()

    if arch_type == "l4ma":
        from model.l4ma import L4maForCausalLM, create_fusion_map as create_l4ma_fusion_map
        from model.l4ma_runtime import FlashInferL4maBackend

        if not FlashInferL4maBackend.is_available():
            raise RuntimeError(
                "FlashInfer backend is not available; cannot instantiate L4MA model."
            )

        backend = FlashInferL4maBackend()
        model = L4maForCausalLM(model_info.architecture, backend=backend)
        fusion_map = create_l4ma_fusion_map(model)
        return model, fusion_map

    if arch_type == "qwen3":
        from model.qwen3 import Qwen3ForCausalLM, create_fusion_map as create_qwen3_fusion_map

        model = Qwen3ForCausalLM(model_info.architecture)
        fusion_map = create_qwen3_fusion_map(model)
        return model, fusion_map

    raise ValueError(f"Unsupported architecture type: {model_info.architecture.type}")


__all__ = ["create_model_and_fusion_map"]
