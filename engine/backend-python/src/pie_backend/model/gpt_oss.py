"""GPT-OSS Large Language Model Architecture.

This model implements the GPT-OSS architecture with:
- Mixture of Experts (MoE) using FlashInfer's fused MXFP4 kernel
- Attention sinks with sliding window (even layers) and full attention (odd layers)
- YaRN-style RoPE scaling

Note: This model requires CUDA and FlashInfer. Apple Silicon is not supported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn.functional as fun
import torch.distributed as dist

from . import ModelConfig as ModelConfigBase
from .gpt_oss_utils import (
    ALIGNMENT,
    TUNE_MAX_NUM_TOKENS,
    pad_to_multiple,
    prepare_gptoss_moe_gate_up,
    prepare_gptoss_moe_down,
)
from ..config import RuntimeConfig
from ..adapter import AdapterSubpass
from ..utils import get_available_memory
from ..loader import Schema, Source, WeightStore

# GPT-OSS requires CUDA-only FlashInfer features
import flashinfer as ops  # type: ignore[import]
from flashinfer.attention import BatchAttentionWithAttentionSinkWrapper  # type: ignore[import]


# =============================================================================
# GPT-OSS WEIGHT SCHEMA
# =============================================================================

def _create_moe_gate_up_transform(hidden_size: int, intermediate_size: int, num_experts: int):
    """Create a transform function for MoE gate_up weights with pre-set config."""
    padded_hidden = pad_to_multiple(hidden_size, ALIGNMENT)
    padded_intermediate = pad_to_multiple(intermediate_size, ALIGNMENT)
    
    def transform_fn(tensors: list[torch.Tensor], kwargs: dict) -> dict:
        config = {
            **kwargs,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "padded_hidden_size": padded_hidden,
            "padded_intermediate_size": padded_intermediate,
            "num_experts": num_experts,
        }
        return prepare_gptoss_moe_gate_up(tensors, config)
    
    return transform_fn


def _create_moe_down_transform(hidden_size: int, intermediate_size: int, num_experts: int):
    """Create a transform function for MoE down weights with pre-set config."""
    padded_hidden = pad_to_multiple(hidden_size, ALIGNMENT)
    padded_intermediate = pad_to_multiple(intermediate_size, ALIGNMENT)
    
    def transform_fn(tensors: list[torch.Tensor], kwargs: dict) -> dict:
        config = {
            **kwargs,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "padded_hidden_size": padded_hidden,
            "padded_intermediate_size": padded_intermediate,
            "num_experts": num_experts,
        }
        return prepare_gptoss_moe_down(tensors, config)
    
    return transform_fn


def create_gpt_oss_schema(model_config: "ModelConfig") -> Schema:
    """
    Create the weight loading schema for GPT-OSS.
    
    This is a factory function because MoE transforms need model dimensions.
    """
    hidden_size = model_config.dim_hidden
    intermediate_size = model_config.dim_mlp
    num_experts = model_config.num_experts
    
    gate_up_transform = _create_moe_gate_up_transform(
        hidden_size, intermediate_size, num_experts
    )
    down_transform = _create_moe_down_transform(
        hidden_size, intermediate_size, num_experts
    )
    
    schema = (
        Schema("gpt_oss")
        # Embedding (no sharding, no quantization for embeddings)
        .define(
            "embed_token",
            Source("model.embed_tokens.weight"),
        )
        # LM head (separate, not weight-tied in GPT-OSS)
        .define(
            "lm_head",
            Source("lm_head.weight"),
        )
        # Per-layer layer norms
        .define(
            "layers.*.norm_attn",
            Source("model.layers.*.input_layernorm.weight"),
        )
        .define(
            "layers.*.norm_mlp",
            Source("model.layers.*.post_attention_layernorm.weight"),
        )
        # Fused QKV projection weights
        .define(
            "layers.*.proj_qkv.weight",
            Source.fuse(
                [
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ],
                dim=0,
            ),
        )
        # Fused QKV projection biases
        .define(
            "layers.*.proj_qkv.bias",
            Source.fuse(
                [
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ],
                dim=0,
            ),
        )
        # Output projection
        .define(
            "layers.*.proj_o",
            Source("model.layers.*.self_attn.o_proj.weight"),
        )
        # Attention sinks (converted to float32)
        .define(
            "layers.*.attn_sinks",
            Source("model.layers.*.self_attn.sinks").dtype(torch.float32),
        )
        # Router weights and bias
        .define(
            "layers.*.router.weight",
            Source("model.layers.*.mlp.router.weight"),
        )
        .define(
            "layers.*.router.bias",
            Source("model.layers.*.mlp.router.bias"),
        )
        # MoE gate_up weights (complex transform)
        .define(
            "layers.*.moe.gemm1_weights",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.gate_up_proj_blocks",
                    "model.layers.*.mlp.experts.gate_up_proj_scales",
                    "model.layers.*.mlp.experts.gate_up_proj_bias",
                ]
            ).transform(gate_up_transform, output_type="weights"),
        )
        .define(
            "layers.*.moe.gemm1_scales",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.gate_up_proj_blocks",
                    "model.layers.*.mlp.experts.gate_up_proj_scales",
                    "model.layers.*.mlp.experts.gate_up_proj_bias",
                ]
            ).transform(gate_up_transform, output_type="scales"),
        )
        .define(
            "layers.*.moe.gemm1_bias",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.gate_up_proj_blocks",
                    "model.layers.*.mlp.experts.gate_up_proj_scales",
                    "model.layers.*.mlp.experts.gate_up_proj_bias",
                ]
            ).transform(gate_up_transform, output_type="bias"),
        )
        # MoE down weights (complex transform)
        .define(
            "layers.*.moe.gemm2_weights",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.down_proj_blocks",
                    "model.layers.*.mlp.experts.down_proj_scales",
                    "model.layers.*.mlp.experts.down_proj_bias",
                ]
            ).transform(down_transform, output_type="weights"),
        )
        .define(
            "layers.*.moe.gemm2_scales",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.down_proj_blocks",
                    "model.layers.*.mlp.experts.down_proj_scales",
                    "model.layers.*.mlp.experts.down_proj_bias",
                ]
            ).transform(down_transform, output_type="scales"),
        )
        .define(
            "layers.*.moe.gemm2_bias",
            Source.gather(
                [
                    "model.layers.*.mlp.experts.down_proj_blocks",
                    "model.layers.*.mlp.experts.down_proj_scales",
                    "model.layers.*.mlp.experts.down_proj_bias",
                ]
            ).transform(down_transform, output_type="bias"),
        )
        # Final layer norm
        .define(
            "norm_last",
            Source("model.norm.weight"),
        )
    )
    
    return schema


# =============================================================================
# MODEL CONFIG
# =============================================================================

@dataclass
class ModelConfig(ModelConfigBase):
    """
    GPT-OSS-specific model architecture configuration.
    
    Inherits from the abstract ModelConfig base class and defines
    all architecture-specific parameters for GPT-OSS models.
    """

    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    num_vocabs: int

    dim_head: int
    dim_hidden: int
    dim_mlp: int

    rms_norm_eps: float

    # MoE configuration
    num_experts: int
    experts_per_token: int

    # YaRN RoPE configuration
    rope_theta: float
    rope_scaling_factor: float
    rope_ntk_alpha: float
    rope_ntk_beta: float

    # Model specific parameters
    initial_context_length: int
    sliding_window: int
    swiglu_alpha: float
    swiglu_beta: float
    swiglu_limit: float

    @staticmethod
    def from_dict(spec: dict) -> "ModelConfig":
        moe = spec.get("moe", {})
        rope = spec.get("rope", {})
        
        return ModelConfig(
            num_layers=int(spec["num_layers"]),
            num_q_heads=int(spec["num_query_heads"]),
            num_kv_heads=int(spec["num_key_value_heads"]),
            dim_head=int(spec["head_size"]),
            dim_hidden=int(spec["hidden_size"]),
            dim_mlp=int(spec["intermediate_size"]),
            num_vocabs=int(spec["vocab_size"]),
            rms_norm_eps=float(spec.get("rms_norm_eps", 1e-6)),
            # MoE
            num_experts=int(moe.get("num_experts", spec.get("num_experts", 8))),
            experts_per_token=int(moe.get("experts_per_token", spec.get("experts_per_token", 2))),
            # YaRN RoPE
            rope_theta=float(rope.get("theta", spec.get("rope_theta", 10000.0))),
            rope_scaling_factor=float(rope.get("scaling_factor", spec.get("rope_scaling_factor", 1.0))),
            rope_ntk_alpha=float(rope.get("ntk_alpha", spec.get("rope_ntk_alpha", 1.0))),
            rope_ntk_beta=float(rope.get("ntk_beta", spec.get("rope_ntk_beta", 32.0))),
            # Model specific
            initial_context_length=int(spec.get("initial_context_length", 4096)),
            sliding_window=int(spec.get("sliding_window", 4096)),
            swiglu_alpha=float(spec.get("swiglu_alpha", 1.0)),
            swiglu_beta=float(spec.get("swiglu_beta", 1.0)),
            swiglu_limit=float(spec.get("swiglu_limit", 20.0)),
        )

    def eval_max_num_kv_pages(self, runtime_config: RuntimeConfig) -> int:
        """Evaluate the maximum number of KV pages based on available memory."""
        available_bytes = get_available_memory(
            devices=runtime_config.devices,
            rank=runtime_config.rank,
        )
        usable_bytes = available_bytes * runtime_config.gpu_mem_utilization
        element_size_bytes = torch.empty((), dtype=runtime_config.activation_dtype).element_size()
        total_bytes_per_page = (
            element_size_bytes
            * 2
            * runtime_config.kv_page_size
            * self.num_kv_heads
            * self.dim_head
            * self.num_layers
        )

        max_num_pages = int(usable_bytes // total_bytes_per_page)
        return max_num_pages


# =============================================================================
# FORWARD PASS
# =============================================================================

class ForwardPass:
    """
    GPT-OSS forward pass implementation.
    
    Key differences from Llama3/Qwen2:
    - Uses BatchAttentionWithAttentionSinkWrapper for attention
    - Even layers use sliding window, odd layers use full attention
    - MoE layer with FlashInfer's trtllm_fp4_block_scale_moe
    - YaRN-style RoPE with pre-computed cos/sin cache
    """

    def __init__(
        self,
        model_config: ModelConfig,
        runtime_config: RuntimeConfig,
        weights: WeightStore,
    ):
        """Initialize the forward pass with weights and attention wrappers."""
        self.model_config = model_config
        self.runtime_config = runtime_config
        self.weights = weights
        
        # Compute padded dimensions for MoE
        self.padded_hidden_size = pad_to_multiple(model_config.dim_hidden, ALIGNMENT)
        self.padded_intermediate_size = pad_to_multiple(model_config.dim_mlp, ALIGNMENT)
        
        # Pre-compute YaRN RoPE cos/sin cache
        self._rope_cos_sin_cache = self._compute_rope_cache()
        
        # Create workspace buffers for FlashInfer wrappers
        workspace_window = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=runtime_config.device
        )
        workspace_full = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=runtime_config.device
        )
        
        # Wrapper for even layers (sliding window attention)
        self.wrapper_window = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace_window,
            kv_layout="NHD",
            window_left=model_config.sliding_window - 1,
            q_data_type=runtime_config.activation_dtype,
            kv_data_type=runtime_config.activation_dtype,
            head_dim_qk=model_config.dim_head,
            head_dim_vo=model_config.dim_head,
        )
        
        # Wrapper for odd layers (full attention)
        self.wrapper_full = BatchAttentionWithAttentionSinkWrapper(
            float_workspace_buffer=workspace_full,
            kv_layout="NHD",
            window_left=-1,
            q_data_type=runtime_config.activation_dtype,
            kv_data_type=runtime_config.activation_dtype,
            head_dim_qk=model_config.dim_head,
            head_dim_vo=model_config.dim_head,
        )
        
        # Pre-compute MoE activation parameters
        num_experts = model_config.num_experts
        device = runtime_config.device
        
        self._output1_scale = torch.full((num_experts,), 1.0, device=device)
        self._output1_scale_gate = torch.full((num_experts,), 1.0, device=device)
        self._output2_scale = torch.full((num_experts,), 1.0, device=device)
        self._gemm1_alpha = torch.full(
            (num_experts,), model_config.swiglu_alpha, device=device, dtype=torch.float32
        )
        self._gemm1_beta = torch.full(
            (num_experts,), model_config.swiglu_beta, device=device, dtype=torch.float32
        )
        self._gemm1_clamp_limit = torch.full(
            (num_experts,), model_config.swiglu_limit, device=device, dtype=torch.float32
        )

    def _compute_rope_cache(self) -> torch.Tensor:
        """Pre-compute YaRN RoPE cos/sin cache for all positions."""
        cfg = self.model_config
        device = self.runtime_config.device
        head_dim = cfg.dim_head
        max_position_id = 131072  # Max sequence length
        
        # Compute base frequencies
        freq = cfg.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float, device=device)
            / head_dim
        )
        
        if cfg.rope_scaling_factor > 1.0:
            # YaRN concentration
            concentration = 0.1 * math.log(cfg.rope_scaling_factor) + 1.0
            
            d_half = head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(cfg.initial_context_length / (cfg.rope_ntk_beta * 2 * math.pi))
                / math.log(cfg.rope_theta)
            )
            high = (
                d_half
                * math.log(cfg.initial_context_length / (cfg.rope_ntk_alpha * 2 * math.pi))
                / math.log(cfg.rope_theta)
            )
            
            interpolation = 1.0 / (cfg.rope_scaling_factor * freq)
            extrapolation = 1.0 / freq
            
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        
        # Compute positions and frequencies
        position_ids = torch.arange(max_position_id, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", position_ids, inv_freq)
        
        # Compute cos/sin with concentration scaling
        cos_cache = freqs.cos() * concentration
        sin_cache = freqs.sin() * concentration
        
        # Concatenate for FlashInfer format: [max_pos, head_dim]
        cos_sin_cache = torch.cat([cos_cache, sin_cache], dim=-1)
        
        return cos_sin_cache.to(torch.float32)

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs into hidden states."""
        return fun.embedding(token_ids, self.weights.get("embed_token"))

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        # Apply final layer norm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[self.model_config.dim_hidden],
            weight=self.weights.get("norm_last"),
            eps=self.model_config.rms_norm_eps,
        )
        # Project to vocab
        return fun.linear(normed, self.weights.get("lm_head"))

    def attention(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_ids: torch.Tensor,
        kv_cache_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: Optional[AdapterSubpass],
        wrapper: Any,
    ) -> torch.Tensor:
        """Execute the attention block for a single layer."""
        cfg = self.model_config
        n = hidden_states.size(0)
        
        # Save for residual
        residual = hidden_states
        
        # 1. Input RMSNorm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[cfg.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_attn"),
            eps=cfg.rms_norm_eps,
        )
        
        # 2. QKV projection with bias
        qkv_proj = fun.linear(
            normed,
            weight=self.weights.get(f"layers.{layer_idx}.proj_qkv.weight"),
            bias=self.weights.get(f"layers.{layer_idx}.proj_qkv.bias"),
        )
        
        # Split Q, K, V
        q_size = cfg.num_q_heads * cfg.dim_head
        kv_size = cfg.num_kv_heads * cfg.dim_head
        q, k, v = torch.split(qkv_proj, [q_size, kv_size, kv_size], dim=-1)
        
        # Apply adapters if provided
        if adapter_subpass is not None:
            adapter_subpass.execute(
                layer_idx, normed, q_state=q, k_state=k, v_state=v
            )
        del normed
        
        # Reshape for attention
        q = q.view(n, cfg.num_q_heads, cfg.dim_head)
        k = k.view(n, cfg.num_kv_heads, cfg.dim_head)
        v = v.view(n, cfg.num_kv_heads, cfg.dim_head)
        
        # Apply YaRN RoPE
        ops.apply_rope_with_cos_sin_cache_inplace(
            positions=position_ids.to(torch.int32),
            query=q,
            key=k,
            head_size=cfg.dim_head,
            cos_sin_cache=self._rope_cos_sin_cache,
            is_neox=True,
        )
        
        # Append to KV cache
        ops.append_paged_kv_cache(
            append_key=k,
            append_value=v,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_layer,
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )
        
        # Compute attention with sinks
        sinks = self.weights.get(f"layers.{layer_idx}.attn_sinks")
        scaling = cfg.dim_head ** -0.5
        attn_output = wrapper.run(q, kv_cache_layer, sinks, scaling)
        attn_output = attn_output.reshape(n, -1)
        
        # Output projection
        attn_proj = fun.linear(
            attn_output,
            weight=self.weights.get(f"layers.{layer_idx}.proj_o"),
            bias=None,
        )
        
        # Residual
        return residual + attn_proj

    def moe(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Execute the MoE MLP block for a single layer."""
        cfg = self.model_config
        
        # Save for residual
        residual = hidden_states
        
        # 1. MLP RMSNorm
        normed = fun.rms_norm(
            hidden_states,
            normalized_shape=[cfg.dim_hidden],
            weight=self.weights.get(f"layers.{layer_idx}.norm_mlp"),
            eps=cfg.rms_norm_eps,
        )
        
        # 2. Router logits
        router_logits = fun.linear(
            normed.reshape(-1, cfg.dim_hidden),
            weight=self.weights.get(f"layers.{layer_idx}.router.weight"),
            bias=self.weights.get(f"layers.{layer_idx}.router.bias"),
        )
        
        # 3. Prepare input for MoE kernel
        hidden_bf16 = normed.to(torch.bfloat16)
        
        # Pad hidden states if needed
        if cfg.dim_hidden != self.padded_hidden_size:
            num_tokens = hidden_bf16.shape[0]
            padded = torch.zeros(
                (num_tokens, self.padded_hidden_size),
                dtype=hidden_bf16.dtype,
                device=hidden_bf16.device,
            )
            padded[:, :cfg.dim_hidden] = hidden_bf16
            hidden_bf16 = padded
        
        # 4. FlashInfer fused MoE kernel
        output = ops.trtllm_fp4_block_scale_moe(
            routing_logits=router_logits.to(torch.bfloat16),
            routing_bias=None,
            hidden_states=hidden_bf16,
            hidden_states_scale=None,
            gemm1_weights=self.weights.get(f"layers.{layer_idx}.moe.gemm1_weights"),
            gemm1_weights_scale=self.weights.get(f"layers.{layer_idx}.moe.gemm1_scales"),
            gemm1_bias=self.weights.get(f"layers.{layer_idx}.moe.gemm1_bias"),
            gemm1_alpha=self._gemm1_alpha,
            gemm1_beta=self._gemm1_beta,
            gemm1_clamp_limit=self._gemm1_clamp_limit,
            gemm2_weights=self.weights.get(f"layers.{layer_idx}.moe.gemm2_weights"),
            gemm2_weights_scale=self.weights.get(f"layers.{layer_idx}.moe.gemm2_scales"),
            gemm2_bias=self.weights.get(f"layers.{layer_idx}.moe.gemm2_bias"),
            output1_scale_scalar=self._output1_scale,
            output1_scale_gate_scalar=self._output1_scale_gate,
            output2_scale_scalar=self._output2_scale,
            num_experts=cfg.num_experts,
            top_k=cfg.experts_per_token,
            n_group=None,
            topk_group=None,
            intermediate_size=self.padded_intermediate_size,
            local_expert_offset=0,
            local_num_experts=cfg.num_experts,
            routed_scaling_factor=None,
            tile_tokens_dim=None,
            routing_method_type=1,  # Renormalize (TopK -> Softmax)
            gated_act_type=0,  # SwiGlu
            do_finalize=True,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )
        
        output = output[0]
        
        # Strip padding
        if cfg.dim_hidden != self.padded_hidden_size:
            output = output[:, :cfg.dim_hidden]
        
        output = output.to(hidden_states.dtype)
        
        # Residual
        return residual + output

    def transform(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        custom_mask: torch.Tensor | None,
        single_token_inference_mode: bool,
        adapter_subpass: Optional[AdapterSubpass],
    ) -> torch.Tensor:
        """Main transformation pipeline through all layers."""
        cfg = self.model_config
        
        hidden_states = input_embeds
        n, _ = hidden_states.size()
        page_size = int(kv_cache_at_layer[0].shape[2])
        
        seq_lens = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size)
        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=seq_lens,
            nnz=n,
        )
        
        # Plan both wrappers (sliding window and full attention)
        # custom_mask is not used with attention sink wrapper
        _ = custom_mask
        _ = single_token_inference_mode
        
        self.wrapper_window.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_lens,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.dim_head,
            page_size,
            causal=True,
            window_left=cfg.sliding_window - 1,
            q_data_type=self.runtime_config.activation_dtype,
            kv_data_type=self.runtime_config.activation_dtype,
            non_blocking=True,
        )
        
        self.wrapper_full.plan(
            qo_indptr,
            kv_page_indptr,
            kv_page_indices,
            kv_last_page_lens,
            cfg.num_q_heads,
            cfg.num_kv_heads,
            cfg.dim_head,
            page_size,
            causal=True,
            window_left=-1,
            q_data_type=self.runtime_config.activation_dtype,
            kv_data_type=self.runtime_config.activation_dtype,
            non_blocking=True,
        )
        
        for layer_idx in range(cfg.num_layers):
            # Select wrapper: even layers use sliding window, odd use full
            wrapper = self.wrapper_window if layer_idx % 2 == 0 else self.wrapper_full
            
            # 1. Attention block
            hidden_states = self.attention(
                hidden_states=hidden_states,
                layer_idx=layer_idx,
                position_ids=position_ids,
                kv_cache_layer=kv_cache_at_layer[layer_idx],
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
                wrapper=wrapper,
            )
            
            # 2. MoE MLP block
            hidden_states = self.moe(hidden_states, layer_idx)
        
        return hidden_states


def create_kv_cache(model_config: ModelConfig, runtime_config: RuntimeConfig) -> list[torch.Tensor]:
    """Create KV cache tensors for all layers."""
    
    max_num_kv_pages = model_config.eval_max_num_kv_pages(runtime_config)
    
    return [
        torch.zeros(
            (
                max_num_kv_pages,
                2,
                runtime_config.kv_page_size,
                model_config.num_kv_heads,
                model_config.dim_head,
            ),
            dtype=runtime_config.activation_dtype,
            device=runtime_config.device,
        )
        for _ in range(model_config.num_layers)
    ]
