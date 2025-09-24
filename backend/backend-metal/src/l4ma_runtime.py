"""Metal-backed runtime implementation skeleton for the L4MA architecture."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import torch

# Ensure backend-python and common_python modules are importable so we can reuse shared interfaces
BACKEND_PYTHON_PATH = Path(__file__).resolve().parents[2] / "backend-python"
COMMON_PYTHON_PATH = Path(__file__).resolve().parents[2] / "common_python"
if str(BACKEND_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(BACKEND_PYTHON_PATH))
if str(COMMON_PYTHON_PATH) not in sys.path:
    sys.path.insert(0, str(COMMON_PYTHON_PATH))

from common_model.l4ma_runtime import L4maBackend, L4maForwardContext, RuntimeInputs
from config.l4ma import L4maArch
from debug_utils import is_tensor_debug_enabled, is_capture_debug_enabled

try:  # pragma: no cover - optional dependency guard
    from metal_backend import MetalBackend
except ImportError:  # pragma: no cover - optional dependency guard
    MetalBackend = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MetalRuntimeMetadata:
    """Metadata describing the Metal runtime configuration."""

    page_size: int
    device: str


class _MetalForwardContext(L4maForwardContext):
    """Forward context that currently falls back to Torch operations.

    The intent is to progressively replace these implementations with true Metal
    kernels as they become available. The structure mirrors the FlashInfer
    context but keeps the surface torch-based so the model can execute while the
    Metal backend matures.
    """

    def __init__(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
        backend: MetalBackend | None,
        metadata: MetalRuntimeMetadata,
    ) -> None:
        self._config = config
        self._inputs = inputs
        self._backend = backend
        self._metadata = metadata

        capture_env = os.environ.get("METAL_CAPTURE_BOTH_PATHS", "0").lower()
        self._capture_both_paths = capture_env in {"1", "true", "yes", "on"}
        force_env = os.environ.get("METAL_FORCE_REFERENCE_OUTPUT", "0").lower()
        self._force_reference_output = force_env in {"1", "true", "yes", "on"}


        self._capture_output_dir: Path | None = None
        self._capture_counter = 0
        if self._capture_both_paths:
            capture_dir_env = os.environ.get("METAL_CAPTURE_OUTPUT_DIR")
            if capture_dir_env:
                self._capture_output_dir = Path(capture_dir_env).expanduser()
                self._capture_output_dir.mkdir(parents=True, exist_ok=True)

        # Derive batch indices/positions in a backend-agnostic way.
        self._batch_indices = self._compute_batch_indices(inputs.qo_indptr, inputs.num_tokens, device=config.device)
        self._batch_positions = self._compute_batch_positions(inputs.qo_indptr, device=config.device)

    @property
    def batch_indices(self) -> torch.Tensor:
        return self._batch_indices

    @property
    def batch_positions(self) -> torch.Tensor:
        return self._batch_positions

    @property
    def metadata(self) -> MetalRuntimeMetadata:
        return self._metadata

    @staticmethod
    def _compute_batch_indices(qo_indptr: torch.Tensor, num_tokens: int, device: str) -> torch.Tensor:
        batch_indices = torch.empty(num_tokens, dtype=torch.int32, device=device)
        for batch_idx in range(qo_indptr.numel() - 1):
            start = int(qo_indptr[batch_idx].item())
            end = int(qo_indptr[batch_idx + 1].item())
            batch_indices[start:end] = batch_idx
        return batch_indices

    @staticmethod
    def _compute_batch_positions(qo_indptr: torch.Tensor, device: str) -> torch.Tensor:
        positions = []
        for batch_idx in range(qo_indptr.numel() - 1):
            start = int(qo_indptr[batch_idx].item())
            end = int(qo_indptr[batch_idx + 1].item())
            seq_len = end - start
            positions.append(torch.arange(seq_len, dtype=torch.int32, device=device))
        if positions:
            return torch.cat(positions)
        return torch.empty(0, dtype=torch.int32, device=device)

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        # TODO: Replace with Metal RoPE kernel invocation. For now, use torch implementation.
        self._apply_rope_torch(query_states, position_ids, self._config)
        self._apply_rope_torch(key_states, position_ids, self._config)

    @staticmethod
    def _apply_rope_torch(tensor: torch.Tensor, position_ids: torch.Tensor, config: L4maArch) -> None:
        if tensor.ndim != 3:
            raise ValueError("Expected tensor shape [tokens, heads, head_size] for RoPE application")

        head_size = config.head_size
        half = head_size // 2
        sinusoids = _build_rope_sinusoids(position_ids, half, config.rope_theta, device=tensor.device, dtype=tensor.dtype)
        cos, sin = sinusoids

        # Expand sinusoids to match tensor shape [batch, heads, half_head_size]
        cos = cos.unsqueeze(1)  # [batch, 1, half_head_size]
        sin = sin.unsqueeze(1)  # [batch, 1, half_head_size]

        tensor_left = tensor[..., :half]
        tensor_right = tensor[..., half:]

        rotated_left = tensor_left * cos - tensor_right * sin
        rotated_right = tensor_left * sin + tensor_right * cos

        tensor[..., :half] = rotated_left
        tensor[..., half:] = rotated_right

    def append_kv_cache(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> None:
        # TODO: Implement Metal-aware KV cache writes. Currently falls back to torch copy.
        positions = self._batch_positions
        page_size = self._metadata.page_size
        kv_indices = self._inputs.kv_page_indices

        if key_states.numel() and is_tensor_debug_enabled():
            k_min, k_max = key_states.aminmax()
            k_nan = torch.isnan(key_states).any().item()
            k_inf = torch.isinf(key_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=kv_append_keys",
                "dtype=",
                key_states.dtype,
                "min=",
                float(k_min),
                "max=",
                float(k_max),
                "has_nan=",
                bool(k_nan),
                "has_inf=",
                bool(k_inf),
            )

        if value_states.numel() and is_tensor_debug_enabled():
            v_min, v_max = value_states.aminmax()
            v_nan = torch.isnan(value_states).any().item()
            v_inf = torch.isinf(value_states).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=kv_append_values",
                "dtype=",
                value_states.dtype,
                "min=",
                float(v_min),
                "max=",
                float(v_max),
                "has_nan=",
                bool(v_nan),
                "has_inf=",
                bool(v_inf),
            )

        num_pages = kv_indices.numel()
        for token_idx in range(key_states.size(0)):
            seq_pos = int(positions[token_idx].item())
            page_slot = seq_pos // page_size
            if page_slot >= num_pages:
                page_slot = num_pages - 1
            offset = seq_pos % page_size
            cache_page = int(kv_indices[page_slot].item())

            kv_cache_layer[cache_page, 0, offset].copy_(key_states[token_idx])
            kv_cache_layer[cache_page, 1, offset].copy_(value_states[token_idx])

    def run_attention(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        if self._backend is None:
            raise RuntimeError("Metal backend is not available")

        try:
            original_dtype = query_states.dtype

            query_cpu = query_states.detach().to(device="cpu", dtype=torch.float32)
            cache_cpu = kv_cache_layer.detach().to(device="cpu", dtype=torch.float32)

            query_np = query_cpu.numpy()
            cache_np = cache_cpu.numpy()
            kv_indices_np = self._inputs.kv_page_indices.detach().cpu().numpy()
            kv_indptr_np = self._inputs.kv_page_indptr.detach().cpu().numpy()
            kv_last_len_np = self._inputs.kv_last_page_lens.detach().cpu().numpy()

            # Debug instrumentation: log input statistics for the first few invocations
            if layer_idx == 0 and query_states.size(0) <= 32:
                print("[MetalInputDebug] query shape=", query_states.shape,
                      "min=", float(query_states.min()),
                      "max=", float(query_states.max()),
                      "has_nan=", bool(torch.isnan(query_states).any()))
                print("[MetalInputDebug] kv_cache shape=", kv_cache_layer.shape,
                      "min=", float(kv_cache_layer.min()),
                      "max=", float(kv_cache_layer.max()),
                      "has_nan=", bool(torch.isnan(kv_cache_layer).any()))
                print("[MetalInputDebug] kv_indices=", kv_indices_np,
                      "kv_indptr=", kv_indptr_np,
                      "kv_last_page_lens=", kv_last_len_np)

            result = self._backend.run_attention_with_kv_cache(
                query_np,
                cache_np,
                kv_page_indices=kv_indices_np,
                kv_page_indptr=kv_indptr_np,
                kv_last_page_lens=kv_last_len_np,
            )
        except Exception as exc:
            raise RuntimeError(f"Metal attention execution failed: {exc}") from exc

        if not hasattr(result, "output") or result.output is None:
            raise RuntimeError("Metal attention execution returned no output")

        output = torch.from_numpy(result.output).to(device=query_states.device, dtype=torch.float32)
        if output.dtype != original_dtype:
            output = output.to(original_dtype)
        if not torch.isfinite(output).all():
            raise RuntimeError("Metal attention execution produced non-finite values")

        if layer_idx == 0 and output.size(0) <= 32:
            print("[MetalOutputDebug] output shape=", output.shape,
                  "min=", float(output.min()),
                  "max=", float(output.max()),
                  "has_nan=", bool(torch.isnan(output).any()))

        debug_compare = os.environ.get("METAL_DEBUG_COMPARE", "0") == "1"
        compute_reference = self._capture_both_paths or debug_compare or self._force_reference_output
        ref_output: torch.Tensor | None = None

        if compute_reference:
            ref_output = self._compute_torch_reference(layer_idx, query_states, kv_cache_layer)

        if self._capture_both_paths and ref_output is not None:
            self._record_attention_snapshot(layer_idx, output, ref_output)

        if debug_compare and ref_output is not None:
            diff = torch.max(torch.abs(output.to(torch.float32) - ref_output.to(torch.float32))).item()
            print("[MetalCompareDebug] max |metal - torch| =", diff)

        if self._force_reference_output and ref_output is not None:
            if ref_output.dtype != original_dtype:
                ref_output = ref_output.to(original_dtype)
            output = ref_output.to(device=query_states.device)
            if is_capture_debug_enabled():
                print(
                    "[MetalCaptureDebug]",
                    f"layer={layer_idx}",
                    "using_reference_output=1",
                )

        return output.view(query_states.size(0), -1)

    def _record_attention_snapshot(
        self,
        layer_idx: int,
        metal_output: torch.Tensor,
        reference_output: torch.Tensor,
    ) -> None:
        metal_cpu = metal_output.detach().to(device="cpu", dtype=torch.float32)
        reference_cpu = reference_output.detach().to(device="cpu", dtype=torch.float32)

        if metal_cpu.numel() == 0:
            return

        diff_cpu = metal_cpu - reference_cpu
        abs_diff_cpu = diff_cpu.abs()
        token_norms = abs_diff_cpu.max(dim=1).values
        metal_min, metal_max = float(metal_cpu.min().item()), float(metal_cpu.max().item())
        ref_min, ref_max = float(reference_cpu.min().item()), float(reference_cpu.max().item())
        diff_max = float(abs_diff_cpu.max().item())
        diff_mean = float(abs_diff_cpu.mean().item()) if abs_diff_cpu.numel() else 0.0

        worst_token_idx = int(token_norms.argmax().item()) if token_norms.numel() else -1
        worst_token = int(self._inputs.batch_token_indices[worst_token_idx].item()) if worst_token_idx >= 0 and hasattr(self._inputs, "batch_token_indices") else worst_token_idx

        if is_capture_debug_enabled():
            print(
                "[MetalCaptureDebug]",
                f"layer={layer_idx}",
                f"tokens={metal_cpu.size(0)}",
                f"metal_min={metal_min:.6f}",
                f"metal_max={metal_max:.6f}",
                f"ref_min={ref_min:.6f}",
                f"ref_max={ref_max:.6f}",
                f"diff_max={diff_max:.6f}",
                f"diff_mean={diff_mean:.6f}",
                f"worst_token={worst_token}",
            )

        if self._capture_output_dir:
            capture_path = self._capture_output_dir / (
                f"layer_{layer_idx:02d}_capture_{self._capture_counter:03d}.pt"
            )
            torch.save(
                {
                    "layer_idx": layer_idx,
                    "metal": metal_cpu,
                    "reference": reference_cpu,
                    "diff_max": diff_max,
                    "diff_mean": diff_mean,
                    "worst_token_idx": worst_token_idx,
                    "token_norms": token_norms,
                },
                capture_path,
            )
        self._capture_counter += 1

    def _compute_torch_reference(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        import torch.nn.functional as F

        device = query_states.device
        dtype = query_states.dtype

        kv_indices = self._inputs.kv_page_indices
        kv_last_page_lens = self._inputs.kv_last_page_lens
        page_size = self._metadata.page_size

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        total_pages = kv_indices.numel()
        last_len = page_size
        if kv_last_page_lens.numel() > 0:
            last_len = int(kv_last_page_lens[-1].item()) or page_size

        for idx in range(total_pages):
            page_ptr = int(kv_indices[idx].item())
            page_tensor = kv_cache_layer[page_ptr]
            length = page_size if idx < total_pages - 1 else last_len
            keys.append(page_tensor[0, :length])
            values.append(page_tensor[1, :length])

        key_tensor = torch.cat(keys, dim=0)
        value_tensor = torch.cat(values, dim=0)

        if key_tensor.numel() and is_tensor_debug_enabled():
            key_tensor_min, key_tensor_max = key_tensor.aminmax()
            key_tensor_nan = torch.isnan(key_tensor).any().item()
            key_tensor_inf = torch.isinf(key_tensor).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=torch_ref_key_tensor",
                "dtype=",
                key_tensor.dtype,
                "min=",
                float(key_tensor_min),
                "max=",
                float(key_tensor_max),
                "has_nan=",
                bool(key_tensor_nan),
                "has_inf=",
                bool(key_tensor_inf),
            )

        if value_tensor.numel() and is_tensor_debug_enabled():
            value_tensor_min, value_tensor_max = value_tensor.aminmax()
            value_tensor_nan = torch.isnan(value_tensor).any().item()
            value_tensor_inf = torch.isinf(value_tensor).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=torch_ref_value_tensor",
                "dtype=",
                value_tensor.dtype,
                "min=",
                float(value_tensor_min),
                "max=",
                float(value_tensor_max),
                "has_nan=",
                bool(value_tensor_nan),
                "has_inf=",
                bool(value_tensor_inf),
            )

        num_q_heads = query_states.size(1)
        num_kv_heads = key_tensor.size(1)
        if num_kv_heads != num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            key_tensor = key_tensor.repeat_interleave(repeat_factor, dim=1)
            value_tensor = value_tensor.repeat_interleave(repeat_factor, dim=1)

        q = query_states.to(torch.float32)
        k = key_tensor.to(torch.float32)
        v = value_tensor.to(torch.float32)

        attn_out = F.scaled_dot_product_attention(
            q.permute(1, 0, 2),
            k.permute(1, 0, 2),
            v.permute(1, 0, 2),
            is_causal=True,
        )

        if attn_out.numel() and is_tensor_debug_enabled():
            ref_min, ref_max = attn_out.aminmax()
            ref_nan = torch.isnan(attn_out).any().item()
            ref_inf = torch.isinf(attn_out).any().item()
            print(
                "[MetalTensorDebug]",
                f"layer={layer_idx}",
                "stage=torch_ref_output",
                "dtype=",
                attn_out.dtype,
                "min=",
                float(ref_min),
                "max=",
                float(ref_max),
                "has_nan=",
                bool(ref_nan),
                "has_inf=",
                bool(ref_inf),
            )

        return attn_out.permute(1, 0, 2).reshape(q.size(0), -1).to(device=device, dtype=dtype)


def _build_rope_sinusoids(position_ids: torch.Tensor, half_head: int, rope_theta: float, *, device: torch.device | str, dtype: torch.dtype):
    float_dtype = torch.float32
    positions = position_ids.to(device=device, dtype=float_dtype)
    indices = torch.arange(half_head, device=device, dtype=float_dtype)
    frequencies = 1.0 / (rope_theta ** (indices / half_head))
    angles = torch.einsum("b,h->bh", positions, frequencies)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


class MetalL4maBackend(L4maBackend):
    """Metal runtime backend that mirrors the FlashInfer interface."""

    @staticmethod
    def is_available() -> bool:
        return MetalBackend is not None

    def __init__(self, metal_backend: Optional[MetalBackend] = None) -> None:
        if metal_backend is None and MetalBackend is not None:
            metal_backend = MetalBackend()
            if not metal_backend.initialize():
                metal_backend = None

        self._backend = metal_backend

        if self._backend is None:
            raise RuntimeError(
                "Metal backend is not available; install the Metal debug framework to use MetalL4maBackend."
            )

    def create_forward_context(
        self,
        *,
        config: L4maArch,
        inputs: RuntimeInputs,
    ) -> L4maForwardContext:
        page_size = int(inputs.kv_cache_at_layer[0].shape[2]) if inputs.kv_cache_at_layer else config.head_size
        metadata = MetalRuntimeMetadata(
            page_size=page_size,
            device=str(config.device),
        )

        return _MetalForwardContext(
            config=config,
            inputs=inputs,
            backend=self._backend,
            metadata=metadata,
        )


__all__ = [
    "MetalL4maBackend",
    "MetalRuntimeMetadata",
]
