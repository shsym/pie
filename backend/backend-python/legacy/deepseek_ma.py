# DeepSeek-V3-style Multi-head-Latent-Attention Architecture (DeepSeek-MA)
from __future__ import annotations

import math
import torch
from torch import nn

import flashinfer as ops                       # flashinfer.mla & flashinfer.page are pulled in here
from config import NUM_TOKENS_IN_BLOCK


# --------------------------------------------------------------------------- #
# MLP (identical to Llama-path - DeepSeek keeps the same SILU-Gated FFN)      #
# --------------------------------------------------------------------------- #
class DsmaMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size,  bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
#  Multi-head Latent Attention (MLA)                                          #
# --------------------------------------------------------------------------- #
class DsmaAttention(nn.Module):
    r"""
    Implements DeepSeek-V3 MLA using FlashInfer's paged-KV kernels.

    Notation  (per DeepSeek paper):
        H_q   - # query/output heads                    (config.num_attention_heads)
        H_l   - # latent (compressed)  heads            (config.num_latent_heads)
        D_ckv - head dim for *content* (compressed KV)  (config.head_dim_ckv  , e.g. 512)
        D_kpe - head dim for rotary PE sub-space        (config.head_dim_kpe  , e.g.  64)

    Matrix-Absorption trick is assumed (W_UQ absorbed into W_UK and W_UV absorbed into W_O):
        - we project Q into two parts:  q_nope  (D_ckv)  and q_pe (D_kpe)
        - we project K/V into latent space:    ckv      (D_ckv)  and kpe (D_kpe)
        - wrapper mixes latent→full heads and returns [nnz, H_q, D_ckv]
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads        = config.num_attention_heads          # H_q
        self.num_latent_heads = config.num_latent_heads             # H_l

        self.D_ckv = config.head_dim_ckv                            # 512
        self.D_kpe = config.head_dim_kpe                            #  64
        self.hidden_size = config.hidden_size                       # (D_ckv + D_kpe) * H_q

        # --- projections ------------------------------------------------------------------ #
        self.q_proj_ckv = nn.Linear(self.hidden_size, self.num_heads        * self.D_ckv, bias=config.use_qkv_bias)
        self.q_proj_kpe = nn.Linear(self.hidden_size, self.num_heads        * self.D_kpe, bias=config.use_qkv_bias)

        self.k_proj_ckv = nn.Linear(self.hidden_size, self.num_latent_heads * self.D_ckv, bias=config.use_qkv_bias)
        self.k_proj_kpe = nn.Linear(self.hidden_size, self.num_latent_heads * self.D_kpe, bias=config.use_qkv_bias)

        # Up-projection after attention (W_O already absorbs W_UV)
        self.o_proj = nn.Linear(self.num_heads * self.D_ckv, self.hidden_size, bias=False)

        # Soft-max scaling (dimension *before* absorption, cf. DS-V3 report §3.2)
        self.sm_scale = 1.0 / math.sqrt(self.D_ckv + self.D_kpe)

    # ----------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ----------------------------------------------------------------------- #
    def forward(
        self,
        wrapper,                                # flashinfer.mla.BatchMLAPagedAttentionWrapper
        hidden_states:       torch.Tensor,      # [nnz, hidden_size]
        position_ids:        torch.Tensor,      # [nnz]
        kv_cache_at_layer,                      # (ckv_cache, kpe_cache) paged-KV tensors
        kv_page_indices:     torch.Tensor,
        kv_page_indptr:      torch.Tensor,
        kv_last_page_lens:   torch.Tensor,
        qo_indptr:           torch.Tensor,
    ) -> torch.Tensor:
        nnz, _ = hidden_states.shape

        # ---- (1) Q/K projections --------------------------------------------------------- #
        q_ckv = self.q_proj_ckv(hidden_states).view(nnz, self.num_heads,        self.D_ckv)
        q_pe  = self.q_proj_kpe(hidden_states).view(nnz, self.num_heads,        self.D_kpe)

        ckv   = self.k_proj_ckv(hidden_states).view(nnz, self.num_latent_heads, self.D_ckv)
        kpe   = self.k_proj_kpe(hidden_states).view(nnz, self.num_latent_heads, self.D_kpe)

        # ---- (2) Rotary on PE sub-space -------------------------------------------------- #
        ops.apply_rope_pos_ids_inplace(q=q_pe, k=kpe, pos_ids=position_ids)     # PE dims only 

        # ---- (3) KV-cache bookkeeping ---------------------------------------------------- #
        batch_idx, positions = ops.get_batch_indices_positions(
            qo_indptr,
            ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, NUM_TOKENS_IN_BLOCK),
            nnz,
        )

        # paged-KV append (MLA-specific)
        ops.append_paged_mla_kv_cache(
            ckv, kpe,
            batch_idx,
            positions,
            kv_cache_at_layer[self.layer_idx][0],      # ckv cache tensor
            kv_cache_at_layer[self.layer_idx][1],      # kpe cache tensor
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
        )

        # ---- (4) MLA attention ----------------------------------------------------------- #
        attn_out = wrapper.run(
            q_ckv,                                    # q_nope
            q_pe,                                     # q_pe
            kv_cache_at_layer[self.layer_idx][0],     # ckv
            kv_cache_at_layer[self.layer_idx][1],     # kpe
        )                                             # → [nnz, H_q, D_ckv]
        attn_out = attn_out.reshape(nnz, -1)          # flatten heads

        # ---- (5) Output projection ------------------------------------------------------- #
        return self.o_proj(attn_out)


# --------------------------------------------------------------------------- #
# Decoder Layer                                                              #
# --------------------------------------------------------------------------- #
class DsmaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DsmaAttention(config, layer_idx)

        self.mlp   = DsmaMlp(config)
        self.norm1 = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        wrapper,
        hidden_states:     torch.Tensor,
        position_ids:      torch.Tensor,
        kv_cache_at_layer,
        kv_page_indices:   torch.Tensor,
        kv_page_indptr:    torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr:         torch.Tensor,
    ) -> torch.Tensor:

        # --- MLA self-attention ----------------------------------------------------------- #
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        hidden_states = self.self_attn(
            wrapper               = wrapper,
            hidden_states         = hidden_states,
            position_ids          = position_ids,
            kv_cache_at_layer     = kv_cache_at_layer,
            kv_page_indices       = kv_page_indices,
            kv_page_indptr        = kv_page_indptr,
            kv_last_page_lens     = kv_last_page_lens,
            qo_indptr             = qo_indptr,
        )
        hidden_states = residual + hidden_states

        # --- Feed-Forward ----------------------------------------------------------------- #
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# --------------------------------------------------------------------------- #
# Full Model                                                                  #
# --------------------------------------------------------------------------- #
class DsmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config       = config
        self.padding_idx  = config.pad_token_id
        self.vocab_size   = config.vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.layers       = nn.ModuleList([DsmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm         = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 128 MB scratch (same as Llama path)
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

        # A single MLA wrapper works for both decode & incremental-prefill
        self.wrapper_mla = ops.mla.BatchMLAPagedAttentionWrapper(self.workspace_buffer, backend="auto")

    # ----------------------------------------------------------------------- #
    def forward(
        self,
        input_embeds:      torch.Tensor,      # [nnz, hidden]
        position_ids:      torch.Tensor,      # [nnz]
        kv_cache_at_layer,                    # sequence of (ckv,kpe) tensors
        kv_page_indices:   torch.Tensor,
        kv_page_indptr:    torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr:         torch.Tensor,
        single_token_inference_mode: bool = False,
    ) -> torch.Tensor:

        hidden_states = input_embeds

        # ------------------------------------------------------------------- #
        # Plan MLA kernel (works for both decode & prefill)                   #
        # ------------------------------------------------------------------- #
        kv_len_arr = ops.get_seq_lens(kv_page_indptr, kv_last_page_lens, NUM_TOKENS_IN_BLOCK)
        causal = True  # DeepSeek uses causal MLA

        self.wrapper_mla.plan(
            qo_indptr,                     # queries per request
            kv_page_indptr,
            kv_page_indices,
            kv_len_arr,
            self.config.num_attention_heads,
            self.config.head_dim_ckv,
            self.config.head_dim_kpe,
            NUM_TOKENS_IN_BLOCK,
            causal,
            1.0 / math.sqrt(self.config.head_dim_ckv + self.config.head_dim_kpe),
            torch.bfloat16,   # q dtype
            torch.bfloat16,   # kv dtype
        )

        # ------------------------------------------------------------------- #
        # Decoder stack                                                      #
        # ------------------------------------------------------------------- #
        for layer in self.layers:
            hidden_states = layer(
                wrapper             = self.wrapper_mla,
                hidden_states       = hidden_states,
                position_ids        = position_ids,
                kv_cache_at_layer   = kv_cache_at_layer,
                kv_page_indices     = kv_page_indices,
                kv_page_indptr      = kv_page_indptr,
                kv_last_page_lens   = kv_last_page_lens,
                qo_indptr           = qo_indptr,
            )

        return self.norm(hidden_states)
