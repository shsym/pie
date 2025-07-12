# Llama-Like Large Language Model Architecture (L4MA)
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

import flashinfer as ops

from config import L4maConfig

VERSION = "0.1.0"


class L4maMlp(nn.Module):
    def __init__(self, config: L4maConfig):
        super().__init__()
        self.config = config
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=config.device, dtype=config.dtype)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, device=config.device, dtype=config.dtype)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False, device=config.device, dtype=config.dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        
        
        interim = self.act_fn(gate_proj) * up_proj
        
        print(f"interim shape: {interim.shape}, mean: {interim.mean().item()}")
        
        down_proj = self.down_proj(interim)
        return down_proj


class L4maAttention(nn.Module):

    def __init__(self, config: L4maConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(config.hidden_size, config.num_query_heads * config.head_size, bias=config.use_qkv_bias, device=config.device, dtype=config.dtype)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_size, bias=config.use_qkv_bias, device=config.device, dtype=config.dtype)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_size, bias=config.use_qkv_bias, device=config.device, dtype=config.dtype)
        self.o_proj = nn.Linear(config.num_query_heads * config.head_size, config.hidden_size, bias=False, device=config.device, dtype=config.dtype)

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
        
        
        # print the mean of qkv matirces
        # q_mean = self.q_proj.weight.mean().item()
        # k_mean = self.k_proj.weight.mean().item()
        # v_mean = self.v_proj.weight.mean().item()
        # print(f"Q Mean: {q_mean}, K Mean: {k_mean}, V Mean: {v_mean}")
        
        n, _ = hidden_states.size()
        page_size = kv_cache_at_layer[0].shape[2]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # print the mean of qkv matirces
        # q_mean = query_states.mean().item()
        # k_mean = key_states.mean().item()
        # v_mean = value_states.mean().item()
        # print(f"Q Mean: {q_mean}, K Mean: {k_mean}, V Mean: {v_mean}")

        query_states = query_states.view(n, self.config.num_query_heads, self.config.head_size)
        key_states = key_states.view(n, self.config.num_key_value_heads, self.config.head_size)
        value_states = value_states.view(n, self.config.num_key_value_heads, self.config.head_size)

        # print(position_ids)
        ops.apply_llama31_rope_pos_ids_inplace(q=query_states, k=key_states, pos_ids=position_ids)

        # print the mean of qkv matirces after applying rope
        # q_mean = query_states.mean().item()
        # k_mean = key_states.mean().item()
        # print(f"Q Mean after rope: {q_mean}, K Mean after rope: {k_mean}")

        batch_indices, positions = ops.get_batch_indices_positions(
            append_indptr=qo_indptr,
            seq_lens=ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, page_size),
            nnz=n
        )
        
        #print(f"batch_indices: {batch_indices}, positions: {positions}")

        ops.append_paged_kv_cache(
            append_key=key_states,
            append_value=value_states,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=kv_cache_at_layer[self.layer_idx],
            kv_indices=kv_page_indices,
            kv_indptr=kv_page_indptr,
            kv_last_page_len=kv_last_page_lens,
            kv_layout='NHD'
        )

        attn_output = wrapper.run(query_states, kv_cache_at_layer[self.layer_idx])
        attn_output = attn_output.reshape(n, -1)
        
        # for i in range(4):
        #     tgt = attn_output[i]
        #     print(f"attn_output[{i}]: {tgt.mean().item()}")
        #     print(f"attn_output[{i}]: {tgt[:8].tolist()})")
        
        attn_output = self.o_proj(attn_output)
        
        # print(f"attn_output shape: {attn_output.shape}, mean: {attn_output.mean().item()}")
       
        return attn_output


class L4maDecoderLayer(nn.Module):
    def __init__(self, config: L4maConfig, layer_idx: int):
        super().__init__()

        self.self_attn = L4maAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype)

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
        
        #print(f"post_attention_layernorm shape after self-attention: {hidden_states.shape}, mean: {hidden_states.mean().item()}")

        
        hidden_states = residual + hidden_states

        return hidden_states


class L4maModel(nn.Module):
    def __init__(self, config: L4maConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0, device=config.device, dtype=config.dtype)
        self.layers = nn.ModuleList(
            [L4maDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=config.device, dtype=config.dtype)

        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=config.device)
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
            single_token_inference_mode: bool = False,
    ) -> torch.Tensor:
        # attention_mask = proc_mask(attention_mask, batch.dtype())
        hidden_states = input_embeds
        
        #print(f"mean of input_embeds: {hidden_states.mean().item()}")
        
        page_size = kv_cache_at_layer[0].shape[2]

        # check if its decoding (qo_indptr is )
        if single_token_inference_mode:
            self.wrapper_decode.plan(
                indptr=kv_page_indptr,
                indices=kv_page_indices,
                last_page_len=kv_last_page_lens,
                num_qo_heads=self.config.num_query_heads,
                num_kv_heads=self.config.num_key_value_heads,
                head_dim=self.config.hidden_size // self.config.num_query_heads,
                page_size=page_size,
                pos_encoding_mode="NONE",
                q_data_type=torch.bfloat16
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
                head_dim_qk=self.config.hidden_size // self.config.num_query_heads,
                page_size=page_size,
                custom_mask=custom_mask,
                q_data_type=torch.bfloat16
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
                qo_indptr=qo_indptr,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


class L4maForCausalLM(nn.Module):

    def __init__(self, config: L4maConfig):
        super().__init__()
        self.config = config
        self.model = L4maModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=config.device, dtype=config.dtype)
