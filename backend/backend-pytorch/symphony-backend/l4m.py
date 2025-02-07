# Llama-Like Large Language Model (L4M)
from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
from sortedcontainers import SortedList
from safetensors import safe_open
from safetensors.torch import save_file

from blocks import KvBlockStorage, KvBlockId


@dataclasses.dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    attention_bias: bool
    vocab_size: int


@dataclasses.dataclass
class Input:
    input_embeds: torch.Tensor
    position_embeds: torch.Tensor
    attention_mask: torch.Tensor

    block_storage: KvBlockStorage
    target_blocks: torch.LongTensor  # (N)
    context_blocks: torch.LongTensor  # (N, #blocks-in-bundle)
    reduction_groups: torch.LongTensor  # (num-cmd, #max-num-bundles-in-cmd)

    def batch_size(self) -> int:
        return self.input_embeds.size(0)


class RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 attention_bias: bool,
                 layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
            attention_mask: torch.Tensor | None = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,

    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if buffer is not None:
            buffer.sink(self.layer_idx, buffer_sink_ids, key_states, value_states)
            key_states, value_states = buffer.cache(self.layer_idx, repeat=self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        #
        # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights + attention_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


class DecoderLayer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 attention_bias: bool,
                 rms_norm_eps: float,
                 layer_idx: int
                 ):
        super().__init__()
        self.self_attn = Attention(hidden_size,
                                   num_attention_heads,
                                   num_key_value_heads,
                                   attention_bias, layer_idx)

        self.mlp = Mlp(hidden_size, intermediate_size)
        self.input_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(hidden_size, eps=rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],  # necessary, but kept here for BC
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            # **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        # super().__init__(config)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            inputs_embeds: torch.Tensor,
            position_embeds: torch.Tensor,
            attention_mask: torch.Tensor,

            block_storage: KvBlockStorage,
            target_blocks: torch.LongTensor,  # (N)
            context_blocks: torch.LongTensor,  # (N, #blocks-in-bundle)
            reduction_groups: torch.LongTensor,  # (num-cmd, #max-num-bundles-in-cmd)

            # buffer: AttentionBuffer,
            # buffer_sink_ids: list[int],
    ) -> torch.Tensor:
        attention_mask = proc_mask(attention_mask, inputs_embeds.dtype)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeds,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states


def get_relocation_map(free_ids: SortedList, allocated_ids: SortedList) -> tuple[list[int], list[int]]:
    free_ids = list(reversed(free_ids))
    allocated_ids = list(allocated_ids)

    relocation_ids = allocated_ids[-len(free_ids):]
    src = []
    dst = []
    while len(free_ids) > 0 and len(relocation_ids) > 0:

        if free_ids[-1] > relocation_ids[-1]:
            break

        src.append(relocation_ids.pop())
        dst.append(free_ids.pop())

    return src, dst


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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


# mask = 1 (true) for tokens that are masked
# mask = 0 (false) for tokens that are not masked
def proc_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    float_mask = mask.to(dtype)

    return float_mask.masked_fill(mask.to(torch.bool), torch.finfo(dtype).min)


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


class RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        self.dim = config.hidden_size // config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        # print( config.rope_scaling)
        if config.rope_scaling is not None and config.rope_scaling["rope_type"] == "llama3":
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
class AttentionBuffer:
    num_batch: int
    capacity: int
    num_layers: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: str

    free_indices: SortedList
    used_indices: SortedList

    k: list[torch.Tensor]
    v: list[torch.Tensor]

    #

    def __init__(self, num_batch: int, capacity: int, num_layers: int, num_heads: int, head_dim: int,
                 dtype=torch.float,
                 device: str = "cuda"):
        self.num_batch = num_batch
        self.capacity = capacity
        self.free_indices = SortedList(range(capacity))
        self.used_indices = SortedList()

        self.k = [torch.empty((num_batch, num_heads, capacity, head_dim), dtype=dtype, device=device) for _ in
                  range(num_layers)]
        self.v = [torch.empty((num_batch, num_heads, capacity, head_dim), dtype=dtype, device=device) for _ in
                  range(num_layers)]

        self.dtype = dtype
        self.device = device

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __repr__(self):
        # print used indices series
        return ",".join([str(i) for i in self.used_indices])

    def allocate(self, size: int) -> list[int]:

        if len(self.free_indices) < size:
            raise RuntimeError("Out of buffer capacity")

        # free indices are sorted in ascending order
        allocated = [self.free_indices.pop(0) for _ in range(size)]
        self.used_indices.update(allocated)

        return allocated

    def free(self, indices: list[int]):
        for i in indices:
            self.used_indices.remove(i)
            self.free_indices.add(i)

    def clear(self):
        self.free_indices.clear()
        self.used_indices.clear()
        self.free_indices.update(range(self.capacity))

    def size(self):
        if len(self.used_indices) == 0:
            return 0
        else:
            return self.used_indices[-1] + 1

    def memory_consumption(self):
        return self.size() * self.num_layers * self.num_batch * self.num_heads * self.head_dim * 2 * 4

    def optimize(self):
        src, dst = get_relocation_map(self.free_indices, self.used_indices)

        for i in range(len(src)):
            for j in range(len(self.k)):
                self.k[j][:, :, dst[i], :].copy_(self.k[j][:, :, src[i], :], non_blocking=True)
                self.v[j][:, :, dst[i], :].copy_(self.v[j][:, :, src[i], :], non_blocking=True)

            self.free_indices.add(src[i])
            self.free_indices.remove(dst[i])
            self.used_indices.add(dst[i])
            self.used_indices.remove(src[i])

    def sink(self, layer_id: int, indices: list[int], k: torch.Tensor, v: torch.Tensor):
        # shape of k and v: (1, num_heads, size,  head_dim)
        num_batch, num_heads, size, head_dim = k.shape
        # assert num_batch == 1

        # self.k[layer_id][:, indices, :].copy_(k.squeeze(0), non_blocking=True)
        # self.v[layer_id][:, indices, :].copy_(v.squeeze(0), non_blocking=True)

        # print(k.shape)
        # print(v.shape)

        for i, j in enumerate(indices):
            self.k[layer_id][:, :, j, :].copy_(k[:, :, i, :], non_blocking=True)
            self.v[layer_id][:, :, j, :].copy_(v[:, :, i, :], non_blocking=True)

    def excerpt(self, indices: list[int]) -> AttentionBuffer:

        res = AttentionBuffer(
            num_batch=self.num_batch,
            capacity=len(indices),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device)

        res.allocate(len(indices))
        for i in range(len(self.k)):
            res.k[i][:, :, :, :].copy_(self.k[i][:, :, indices, :], non_blocking=True)
            res.v[i][:, :, :, :].copy_(self.v[i][:, :, indices, :], non_blocking=True)

        return res

    def save(self, path: str):
        # make sure the buffer is optimized before saving
        self.optimize()

        tensors = {}
        for i in range(len(self.k)):
            tensors[f"k_{i}"] = self.k[i][:, :, :self.size(), :].contiguous()
            tensors[f"v_{i}"] = self.v[i][:, :, :self.size(), :].contiguous()

        save_file(tensors, filename=path)

    def load(self, path: str):

        # clear the buffer before loading
        self.clear()

        tensors = {}
        num_batch, num_heads, size, head_dim = 0, 0, 0, 0
        with safe_open(path, framework="pt", device=self.device) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
                if size == 0:
                    num_batch, num_heads, size, head_dim = tensors[k].shape

        assert num_batch == self.num_batch or num_batch == 1
        assert num_heads == self.num_heads
        assert head_dim == self.k[0].shape[3]
        assert size <= self.capacity

        # print(f"Loading buffer of size {size}, head_dim {head_dim}, num_heads {num_heads}")

        self.allocate(size)

        for i in range(len(self.k)):
            self.k[i][:, :, :self.size(), :].copy_(tensors[f"k_{i}"], non_blocking=True)
            self.v[i][:, :, :self.size(), :].copy_(tensors[f"v_{i}"], non_blocking=True)

            # print(torch.sum(self.k[i][:, :self.size(), :]))
            # print(torch.sum(tensors[f"k_{i}"]))
            # print('----')

        # see if the values has been changed
        # for i in range(len(self.k)):
        #     assert torch.allclose(self.k[i][:, :self.size(), :], tensors[f"k_{i}"].to(self.device), atol=1e-5)

    def cache(self, layer_id: int, repeat: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k[layer_id][:, :, :self.size(), :]
        v = self.v[layer_id][:, :, :self.size(), :]

        if repeat > 1:
            k = repeat_kv(k, repeat)
            v = repeat_kv(v, repeat)

        return k, v

    def __len__(self) -> int:
        return self.size()
