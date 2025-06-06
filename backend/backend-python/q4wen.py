# Q4wen: Qwen-like Large Language Model Architecture with FlashInfer
from __future__ import annotations

import torch
from torch import nn

import flashinfer as ops
from config import NUM_TOKENS_IN_BLOCK


class Q4wenMlp(nn.Module):
    """MLP layer for Qwen models - identical to Llama MLP structure"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Q4wenAttention(nn.Module):
    """FlashInfer-based attention for Qwen models"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Qwen3 uses attention_bias for all projections
        attention_bias = getattr(config, 'attention_bias', False)
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)  # o_proj typically doesn't use bias
        
        # Qwen3 uses QK normalization - critical for correct attention!
        self.q_norm = nn.RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.k_norm = nn.RMSNorm(self.head_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))
        
        # Extract RoPE configuration from config
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.rope_scaling = getattr(config, 'rope_scaling', None)
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)

    def forward(
            self,
            wrapper,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache_at_layer: torch.Tensor,
            kv_page_indices: torch.Tensor,
            kv_page_indptr: torch.Tensor,
            kv_last_page_lens: torch.Tensor,
            qo_indptr: torch.Tensor,
    ) -> torch.Tensor:
        # Handle both 2D (nnz, hidden_size) and 3D (batch_size, seq_len, hidden_size) inputs
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_size)  # Flatten to (nnz, hidden_size)
            nnz = batch_size * seq_len
            # Also flatten position_ids if it's 2D
            if position_ids.dim() == 2:
                position_ids = position_ids.view(-1)
        else:
            nnz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(nnz, self.num_heads, self.head_dim)
        key_states = key_states.view(nnz, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(nnz, self.num_key_value_heads, self.head_dim)

        # Apply QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Apply RoPE with proper config
        ops.apply_rope_pos_ids_inplace(
            q=query_states, 
            k=key_states, 
            pos_ids=position_ids,
            rope_theta=self.rope_theta
        )

        # ...existing KV cache code...
        batch_indices, positions = ops.get_batch_indices_positions(
            qo_indptr,
            ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, NUM_TOKENS_IN_BLOCK),
            nnz
        )
        
        ops.append_paged_kv_cache(
            key_states,
            value_states,
            batch_indices,
            positions,
            kv_cache_at_layer[self.layer_idx],
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens
        )

        attn_output = wrapper.run(query_states, kv_cache_at_layer[self.layer_idx])
        attn_output = attn_output.reshape(nnz, -1)

        attn_output = self.o_proj(attn_output)
        
        return attn_output


class Q4wenDecoderLayer(nn.Module):
    """Qwen decoder layer with FlashInfer attention"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Q4wenAttention(config, layer_idx)
        self.mlp = Q4wenMlp(config)
        
        # Qwen uses RMSNorm like Llama
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(
            self,
            wrapper,
            hidden_states: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache_at_layer: torch.Tensor,
            kv_page_indices: torch.Tensor,
            kv_page_indptr: torch.Tensor,
            kv_last_page_lens: torch.Tensor,
            qo_indptr: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            wrapper=wrapper,
            hidden_states=hidden_states,
            position_ids=position_ids,
            kv_cache_at_layer=kv_cache_at_layer,
            kv_page_indices=kv_page_indices,
            kv_page_indptr=kv_page_indptr,
            kv_last_page_lens=kv_last_page_lens,
            qo_indptr=qo_indptr,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Q4wenModel(nn.Module):
    """FlashInfer-based Qwen model implementation"""
    def __init__(self, config):
        super().__init__()

        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size
        self.config = config
        
        # Indicate that this model uses FlashInfer for attention
        self.use_flashinfer = True
        
        # Create model components - transformers will handle device placement
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Q4wenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

        # FlashInfer workspace and wrappers - will be initialized when moved to device
        self.workspace_buffer = None
        self.wrapper_decode = None
        self.wrapper_append = None
        
        # FlashInfer buffers will be initialized lazily when needed
        
    def _init_flashinfer_buffers(self, device):
        """Initialize FlashInfer buffers when model is moved to device"""
        if self.workspace_buffer is None or self.workspace_buffer.device != device:
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            self.wrapper_decode = ops.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.wrapper_append = ops.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

    def forward(
            self,
            input_embeds: torch.Tensor,
            position_ids: torch.Tensor,
            kv_cache_at_layer: torch.Tensor,
            kv_page_indices: torch.Tensor,
            kv_page_indptr: torch.Tensor,
            kv_last_page_lens: torch.Tensor,
            qo_indptr: torch.Tensor,
            custom_mask: torch.Tensor,
            single_token_inference_mode: bool=False,
    ) -> torch.Tensor:
        # Initialize FlashInfer buffers if needed
        self._init_flashinfer_buffers(input_embeds.device)
        
        hidden_states = input_embeds
        
        # Plan FlashInfer kernels based on mode
        if single_token_inference_mode:
            self.wrapper_decode.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads),
                page_size=NUM_TOKENS_IN_BLOCK,
                pos_encoding_mode="NONE",  # RoPE is applied manually
                q_data_type=torch.bfloat16
            )
            wrapper = self.wrapper_decode
        else:
            self.wrapper_append.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_page_indptr,
                paged_kv_indices=kv_page_indices,
                paged_kv_last_page_len=kv_last_page_lens,
                num_qo_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim_qk=getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads),
                page_size=NUM_TOKENS_IN_BLOCK,
                custom_mask=custom_mask,
                q_data_type=torch.bfloat16
            )
            wrapper = self.wrapper_append

        # Forward through decoder layers
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                wrapper=wrapper,
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                qo_indptr=qo_indptr,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)
        return hidden_states
