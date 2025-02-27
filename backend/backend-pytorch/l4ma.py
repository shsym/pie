# Llama-Like Large Language Model Architecture (L4MA)
from __future__ import annotations

import math

import torch
from torch import nn

import ops


class L4maMlp(nn.Module):
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


class L4maAttention(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_ptr: torch.Tensor,  # KV storage pointer
            new_q_lut: torch.Tensor,  # Query LUT
            new_kv_lut: torch.Tensor,  # New KV LUT
            all_kv_lut: torch.Tensor,  # All KV LUT
            mask: torch.Tensor,  # Attention mask
            cmd_groups: torch.Tensor,  # Command groups
            rope_cache: tuple[torch.Tensor, torch.Tensor],  # Cos and Sin cache

    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope_cache

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        ops.fill_kv_block_storage(kv_ptr[self.layer_idx], key_states, value_states, new_kv_lut)

        attn_output = ops.qkv_attention(
            q=query_states,
            kv=kv_ptr[self.layer_idx],
            q_lut=new_q_lut,
            kv_lut=all_kv_lut,
            mask=mask,
            reduce_grp=cmd_groups,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


class L4maDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = L4maAttention(config, layer_idx)

        self.mlp = L4maMlp(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_ptr: torch.Tensor,  # KV storage pointer
            new_q_lut: torch.Tensor,  # Query LUT
            new_kv_lut: torch.Tensor,  # New KV LUT
            all_kv_lut: torch.Tensor,  # All KV LUT
            mask: torch.Tensor,  # Attention mask
            cmd_groups: torch.Tensor,  # Command groups
            rope_cache: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_ptr=kv_ptr,
            new_q_lut=new_q_lut,
            new_kv_lut=new_kv_lut,
            all_kv_lut=all_kv_lut,
            mask=mask,
            cmd_groups=cmd_groups,
            rope_cache=rope_cache,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class L4maModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        # super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [L4maDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            input_embeds: torch.Tensor,
            kv_ptr: torch.Tensor,  # KV storage pointer
            new_q_lut: torch.Tensor,  # Query LUT
            new_kv_lut: torch.Tensor,  # New KV LUT
            all_kv_lut: torch.Tensor,  # All KV LUT
            mask: torch.Tensor,  # Attention mask
            cmd_groups: torch.Tensor,  # Command groups
            rope_cache: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        # attention_mask = proc_mask(attention_mask, batch.dtype())
        hidden_states = input_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                kv_ptr=kv_ptr,
                new_q_lut=new_q_lut,
                new_kv_lut=new_kv_lut,
                all_kv_lut=all_kv_lut,
                mask=mask,
                cmd_groups=cmd_groups,
                rope_cache=rope_cache
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


def get_image_position_ids(offset: int, patch_h: int, patch_w: int) -> list[tuple[int, int, int]]:
    output_ids = []
    for i in range(patch_h * patch_w):
        output_ids.append((
            offset,
            offset + i // patch_w,
            offset + i % patch_w
        ))
    return output_ids


def get_video_position_ids(offset, patch_t, patch_h, patch_w, time_scale) -> list[tuple[int, int, int]]:
    output_ids = []

    for i in range(patch_t * patch_h * patch_w):
        patch_t_i = i // (patch_h * patch_w)
        patch_hw_i = i % (patch_h * patch_w)

        output_ids.append((
            offset + patch_t_i * time_scale,
            offset + patch_hw_i // patch_w,
            offset + patch_hw_i % patch_w
        ))

    return output_ids


def _compute_default_rope_parameters(
        base: int, dim: int, device: torch.device
) -> torch.Tensor:
    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq


def _compute_llama3_parameters(
        base: int,
        dim: int,
        factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
        device: torch.device,
) -> torch.Tensor:
    inv_freq = _compute_default_rope_parameters(base, dim, device)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama


class L4maRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.dim = config.hidden_size // config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        # print( config.rope_scaling)
        if config.rope_scaling is not None and config.rope_scaling["rope_type"] == "llama3":
            print("Using LLAMA3 parameters")
            inv_freq = _compute_llama3_parameters(
                base=self.base,
                dim=self.dim,
                factor=config.rope_scaling["factor"],
                low_freq_factor=config.rope_scaling["low_freq_factor"],
                high_freq_factor=config.rope_scaling["high_freq_factor"],
                old_context_len=config.rope_scaling["original_max_position_embeddings"],
                device=device,
            )

        else:
            inv_freq = _compute_default_rope_parameters(self.base, self.dim, device)
            # inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device
        )

    def _set_cos_sin_cache(self, seq_len, device):
        self.max_seq_len_cached = seq_len
        # print(self.max_seq_len_cached, device, self.inv_freq.dtype)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.float)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().float(), persistent=False)
        self.register_buffer("sin_cached", emb.sin().float(), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device)

        return (
            self.cos_cached[:seq_len, ...].to(x.dtype),
            self.sin_cached[:seq_len, ...].to(x.dtype),
        )


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bs, num_heads, seq_len, head_dim]
    # cos, sin: [bs, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# token-level attention buffer
class AttentionStorage:

    def __init__(self, num_layers: int, num_blocks: int, num_heads: int, block_size: int, head_dim: int, device: str, dtype=torch.bfloat16):
        self.ptr = torch.empty((num_layers, num_blocks, num_heads, block_size * 2, head_dim), device=device, dtype=dtype)

        self.dtype = dtype
        self.device = device

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim


class VectorStorage:

    def __init__(self, num_vectors: int, embed_dim: int, device: str, dtype=torch.bfloat16):
        self.ptr = torch.empty((num_vectors, embed_dim), device=device, dtype=dtype)

        self.dtype = dtype
        self.device = device

        self.num_vectors = num_vectors
        self.embed_dim = embed_dim
