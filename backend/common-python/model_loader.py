"""Shared model loading utilities for PIE backends."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Tuple

import torch
import ztensor
from tqdm import tqdm

from config.common import ModelInfo


CreateModelFn = Callable[[ModelInfo], Tuple[torch.nn.Module, dict]]


def load_model(
    config: dict, create_model_fn: CreateModelFn
) -> tuple[torch.nn.Module, ModelInfo]:
    """Load a model using the provided factory function and fusion metadata."""

    model_name = config["model"]
    cache_dir = config["cache_dir"]
    model_path = Path(cache_dir) / "models"
    metadata_path = model_path / f"{model_name}.toml"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")

    model_device = config["device"]
    model_dtype = getattr(torch, config["dtype"])
    model_info = ModelInfo.load_from_file(str(metadata_path), model_device, model_dtype)

    try:
        model, fusion_map = create_model_fn(model_info)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to instantiate model for architecture {model_info.architecture.type}: {exc}"
        ) from exc

    source_to_fusion_target = {
        source: target
        for target, details in fusion_map.items()
        for source in details["sources"]
    }

    pending_fusion_tensors = {}
    model_state_keys = set(model.state_dict().keys())
    loaded_keys = set()

    try:
        for param_file in model_info.parameters:
            weights_path = model_path / model_name / param_file
            with ztensor.Reader(str(weights_path)) as reader:
                tensor_names = reader.get_tensor_names()
                pbar_desc = (
                    f"Loading {param_file[:30]}..."
                    if len(param_file) > 30
                    else f"Loading {param_file}"
                )
                for name in tqdm(tensor_names, desc=pbar_desc, unit="tensors"):
                    if name in source_to_fusion_target:
                        pending_fusion_tensors[name] = reader.read_tensor(
                            name, to="torch"
                        )
                        continue

                    if name in model_state_keys and name not in loaded_keys:
                        tensor_data = reader.read_tensor(name, to="torch")
                        if tensor_data is None:
                            print(
                                f"    Warning: Could not read tensor '{name}'. Skipping."
                            )
                            continue
                        param = model.state_dict()[name]

                        if tensor_data.shape != param.shape:
                            print(
                                f"    Warning: Shape mismatch for tensor '{name}'. Skipping."
                            )
                            continue
                        # Ensure dtype/device compatibility using a single conversion
                        if hasattr(tensor_data, "to"):
                            needs_conversion = False
                            conversion_kwargs = {}

                            if hasattr(tensor_data, "dtype"):
                                tensor_dtype = getattr(tensor_data, "dtype")
                                if tensor_dtype != param.dtype:
                                    conversion_kwargs["dtype"] = param.dtype
                                    needs_conversion = True

                            if hasattr(tensor_data, "device"):
                                tensor_device = getattr(tensor_data, "device")
                                if tensor_device != param.device:
                                    conversion_kwargs["device"] = param.device
                                    needs_conversion = True

                            if needs_conversion:
                                to_method = getattr(tensor_data, "to")
                                tensor_data = to_method(**conversion_kwargs)

                        if name == "model.layers.0.input_layernorm.weight":
                            print("[ModelLoaderDebug] copying", name)
                            dtype_val = getattr(tensor_data, "dtype", "unknown")
                            device_val = getattr(tensor_data, "device", "unknown")
                            min_val = (
                                float(tensor_data.min())
                                if hasattr(tensor_data, "min")
                                else "unknown"
                            )
                            max_val = (
                                float(tensor_data.max())
                                if hasattr(tensor_data, "max")
                                else "unknown"
                            )
                            print(
                                "[ModelLoaderDebug] source",
                                name,
                                "dtype=",
                                dtype_val,
                                "device=",
                                device_val,
                                "min=",
                                min_val,
                                "max=",
                                max_val,
                            )
                            print(
                                "[ModelLoaderDebug] target before copy",
                                name,
                                "dtype=",
                                param.dtype,
                                "device=",
                                param.device,
                            )
                        with torch.no_grad():
                            param.copy_(tensor_data, non_blocking=True)
                        if name == "model.layers.0.input_layernorm.weight":
                            print(
                                "[ModelLoaderDebug] target after copy",
                                name,
                                "min=",
                                float(param.min()),
                                "max=",
                                float(param.max()),
                            )
                        loaded_keys.add(name)

        for target_name, details in fusion_map.items():
            source_names = details["sources"]
            if all(s in pending_fusion_tensors for s in source_names):
                tensors_to_fuse = [pending_fusion_tensors[s] for s in source_names]
                fused_tensor = torch.cat(tensors_to_fuse, dim=details["dim"])
                param = model.state_dict()[target_name]
                if fused_tensor.shape != param.shape:
                    print(
                        f"    Warning: Shape mismatch for fused tensor '{target_name}'. Skipping."
                    )
                    continue
                with torch.no_grad():
                    param.copy_(fused_tensor, non_blocking=True)
                loaded_keys.add(target_name)

        if "lm_head.weight" in model_state_keys and "lm_head.weight" not in loaded_keys:
            if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                embed_tokens = getattr(model.model, "embed_tokens")
                if hasattr(embed_tokens, "weight"):
                    target_tensor = model.state_dict()["lm_head.weight"]
                    source_tensor = getattr(embed_tokens, "weight").to(
                        dtype=target_tensor.dtype, device=target_tensor.device
                    )
                    target_tensor.copy_(source_tensor, non_blocking=True)
                    loaded_keys.add("lm_head.weight")

        missing_keys = model_state_keys - loaded_keys
        if missing_keys:
            print("\nWarning: Some model weights were not found in any parameter file:")
            for key in sorted(list(missing_keys)):
                print(f"  - {key}")
        else:
            print("\nSuccessfully loaded all expected model weights.")

        model.eval()
        return model, model_info

    except ztensor.ZTensorError as exc:
        print(
            f"Fatal Error: Failed to read a ztensor file. Error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        print(
            f"An unexpected fatal error occurred during model loading: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


__all__ = ["load_model"]
