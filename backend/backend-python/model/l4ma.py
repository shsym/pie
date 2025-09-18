"""Llama-Like Large Language Model Architecture (L4MA)"""

from __future__ import annotations

import torch
from torch import nn

import flashinfer as ops
from adapter import AdapterSubpass

from config.l4ma import L4maArch

# Import debug framework checkpoint decorator
try:
    from debug_framework.decorators.checkpoint_decorator import checkpoint_validation
    CHECKPOINT_DECORATOR_AVAILABLE = True
except ImportError:
    CHECKPOINT_DECORATOR_AVAILABLE = False
    # Fallback no-op decorator
    def checkpoint_validation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

VERSION = "0.1.0"


def create_fusion_map(model: nn.Module):
    """
    Analyzes the model and creates a map for fusing weights.

    Returns:
        A dictionary mapping {fused_tensor_name: {"sources": [source_names], "dim": cat_dim}}.
    """
    fusion_map = {}
    for name, module in model.named_modules():
        # --- Rule for L4maAttention QKV Fusion ---
        if isinstance(module, L4maAttention):
            # Handle weights
            target_w = f"{name}.qkv_proj.weight"
            sources_w = [
                f"{name}.q_proj.weight",
                f"{name}.k_proj.weight",
                f"{name}.v_proj.weight",
            ]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0}

            # Handle biases if they exist
            if module.qkv_proj.bias is not None:
                target_b = f"{name}.qkv_proj.bias"
                sources_b = [
                    f"{name}.q_proj.bias",
                    f"{name}.k_proj.bias",
                    f"{name}.v_proj.bias",
                ]
                fusion_map[target_b] = {"sources": sources_b, "dim": 0}

        # --- Rule for L4maMlp Gate/Up Fusion ---
        elif isinstance(module, L4maMlp):
            target_w = f"{name}.gate_up_proj.weight"
            sources_w = [f"{name}.gate_proj.weight", f"{name}.up_proj.weight"]
            fusion_map[target_w] = {"sources": sources_w, "dim": 0}

    return fusion_map


class L4maMlp(nn.Module):
    """TODO: Add class docstring."""

    def __init__(self, config: L4maArch):
        """TODO: Add method docstring."""
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,  # Double the output dimension
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # Gate/Up projection for Metal GEMM kernel comparison
        gate_up_proj_out = self._gate_up_projection(x)
        gate_proj, up_proj = gate_up_proj_out.chunk(2, dim=-1)

        # SiLU activation for Metal activation kernel comparison
        interim = self._silu_activation(gate_proj, up_proj)

        # Down projection for Metal GEMM kernel comparison
        down_proj = self._down_projection(interim)
        return down_proj

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def _gate_up_projection(self, x):
        """Gate/Up projection for Metal GEMM kernel comparison."""
        return self.gate_up_proj(x)

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_activation",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def _silu_activation(self, gate_proj, up_proj):
        """SiLU activation for Metal activation kernel comparison."""
        return self.act_fn(gate_proj) * up_proj

    @checkpoint_validation(
        checkpoint_name="l4ma_mlp_down_proj",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def _down_projection(self, interim):
        """Down projection for Metal GEMM kernel comparison."""
        return self.down_proj(interim)


class L4maAttention(nn.Module):
    """TODO: Add class docstring."""

    def __init__(self, config: L4maArch, layer_idx: int):
        """TODO: Add method docstring."""
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Define the output sizes for Q, K, and V for clarity
        self.q_size = config.num_query_heads * config.head_size
        self.k_size = config.num_key_value_heads * config.head_size
        self.v_size = config.num_key_value_heads * config.head_size

        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + self.k_size + self.v_size,
            bias=config.use_qkv_bias,
            device=config.device,
            dtype=config.dtype,
        )

        self.o_proj = nn.Linear(
            config.num_query_heads * config.head_size,
            config.hidden_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(
        self,
        wrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """TODO: Add method docstring."""

        n, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(
            qkv_states, [self.q_size, self.k_size, self.v_size], dim=-1
        )

        # apply adapters if provided
        if adapter_subpass is not None:
            adapter_subpass.execute(
                self.layer_idx,
                hidden_states,
                q_state=query_states,
                k_state=key_states,
                v_state=value_states,
            )

        # Reshape and continue as before
        query_states = query_states.view(
            n, self.config.num_query_heads, self.config.head_size
        )
        key_states = key_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )
        value_states = value_states.view(
            n, self.config.num_key_value_heads, self.config.head_size
        )

        # print(position_ids)
        ops.apply_llama31_rope_pos_ids_inplace(
            q=query_states, k=key_states, pos_ids=position_ids
        )

        # Ensure query_states matches the configured dtype for FlashInfer plan
        if query_states.dtype != self.config.dtype:
            print("warn: query dtype does not match config dtype!")
            query_states = query_states.to(self.config.dtype)

        ops.append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=batch_indices,
            positions=batch_positions,
            paged_kv_cache=kv_cache_at_layer[self.layer_idx],
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout="NHD",
        )

        # Execute attention computation and capture for Metal backend comparison
        attn_output = self._attention_computation(wrapper, query_states, kv_cache_at_layer)

        attn_output = self.o_proj(attn_output)

        return attn_output

    @checkpoint_validation(
        checkpoint_name="l4ma_attention_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True,
        capture_inputs={
            'query_states': 'query_states',
            'kv_cache_at_layer': 'kv_cache_at_layer'
        }
    )
    def _attention_computation(
        self,
        wrapper,
        query_states: torch.Tensor,
        kv_cache_at_layer: torch.Tensor
    ) -> torch.Tensor:
        # Capture input tensors for Metal kernel verification
        # Query states are directly passed
        # Key/Value states need to be extracted from the KV cache

        attn_output = wrapper.run(query_states, kv_cache_at_layer[self.layer_idx])
        return attn_output.reshape(attn_output.size(0), -1)


class L4maDecoderLayer(nn.Module):
    """TODO: Add class docstring."""

    def __init__(self, config: L4maArch, layer_idx: int):
        """TODO: Add method docstring."""
        super().__init__()

        self.self_attn = L4maAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

    @checkpoint_validation(
        checkpoint_name="l4ma_decoder_layer_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def forward(
        self,
        wrapper,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer: torch.Tensor,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_positions: torch.Tensor,
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """TODO: Add method docstring."""
        residual = hidden_states

        # Input normalization for Metal RMSNorm kernel comparison
        hidden_states = self._input_normalization(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            wrapper=wrapper,
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            batch_indices=batch_indices,
            batch_positions=batch_positions,
            adapter_subpass=adapter_subpass,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # Post-attention normalization for Metal RMSNorm kernel comparison
        hidden_states = self._post_attention_normalization(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states

    @checkpoint_validation(
        checkpoint_name="l4ma_input_norm",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def _input_normalization(self, hidden_states):
        """Input RMSNorm for Metal normalization kernel comparison."""
        return self.input_layernorm(hidden_states)

    @checkpoint_validation(
        checkpoint_name="l4ma_post_attention_norm",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def _post_attention_normalization(self, hidden_states):
        """Post-attention RMSNorm for Metal normalization kernel comparison."""
        return self.post_attention_layernorm(hidden_states)


class L4maModel(nn.Module):
    """TODO: Add class docstring."""

    def __init__(self, config: L4maArch):
        """TODO: Add method docstring."""
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
            device=config.device,
            dtype=config.dtype,
        )
        self.layers = nn.ModuleList(
            [
                L4maDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            device=config.device,
            dtype=config.dtype,
        )

        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=config.device
        )
        self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )

    @checkpoint_validation(
        checkpoint_name="l4ma_model_forward",
        capture_tensors=True,
        include_metadata=True,
        tolerance=1e-5,
        backend_comparison=None,
        performance_monitoring=True
    )
    def forward(
        self,
        # input
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        qo_indptr: torch.Tensor,
        # kv cache
        kv_cache_at_layer: list[torch.Tensor],
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        # mask
        custom_mask: torch.Tensor,
        single_token_inference_mode: bool,
        # subpasses
        adapter_subpass: AdapterSubpass | None,
    ) -> torch.Tensor:
        """TODO: Add method docstring."""
        hidden_states = input_embeds
        n, _ = hidden_states.size()

        page_size = kv_cache_at_layer[0].shape[2]

        batch_indices, batch_positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n,
        )

        # check if its decoding (qo_indptr is )
        if single_token_inference_mode:
            self.wrapper_decode.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=self.config.num_query_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_size,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=self.config.dtype,
            )
            wrapper = self.wrapper_decode
        else:
            self.wrapper_append.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=self.config.num_query_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim_qk=self.config.head_size,
                page_size=page_size,
                custom_mask=custom_mask,
                q_data_type=self.config.dtype,
            )
            wrapper = self.wrapper_append

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                wrapper=wrapper,
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                batch_indices=batch_indices,
                batch_positions=batch_positions,
                adapter_subpass=adapter_subpass,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class L4maForCausalLM(nn.Module):
    """TODO: Add class docstring."""

    def __init__(self, config: L4maArch):
        """TODO: Add method docstring."""
        super().__init__()
        self.config = config
        self.model = L4maModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=config.device,
            dtype=config.dtype,
        )

    def forward(self):
        """
        Should not be called. Method 'forward' is abstract in class
        'torch.nn.modules.module' so must be overridden in child class.
        """
        raise NotImplementedError("Should not be called")
