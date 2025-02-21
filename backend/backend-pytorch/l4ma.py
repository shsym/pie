# Llama-Like Large Language Model Architecture (L4MA)
from __future__ import annotations

import math

import torch
from torch import nn

import ops
from engine import Batch


class Config:
    ...


class L4maRmsNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        L4maRmsNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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
            batch:Batch
            # position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
            # attention_mask: torch.Tensor | None = None,
            # buffer: AttentionBuffer | None = None,
            # buffer_sink_ids: list[int] | None = None,
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = batch.rope_cache

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        ops.copy_kv_block(batch.block_storage_ptr, key_states, value_states, batch.ctx_block_ptrs)

        attn_output = ops.qkv_attention(
            query_states,
            batch.block_storage_ptr,
            batch_size=batch.input_embeds.shape[0],
            num_grps=batch.num_requests,
            max_grp_size=batch.max_chunk_count(),
            num_blocks_per_batch=batch.chunk_size(),
            q_lut=batch.q_lut,
            kv_lut=batch.kv_lut,
            mask_lut=batch.mask_lut,
            reduce_grp_lut=batch.reduce_grp_lut,
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
        self.input_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            batch:Batch
            # attention_mask: torch.Tensor,
            # position_embeddings: tuple[torch.Tensor, torch.Tensor],  # necessary, but kept here for BC
            # buffer: AttentionBuffer | None = None,
            # buffer_sink_ids: list[int] | None = None,
            # **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            batch=batch
            # attention_mask=attention_mask,
            # position_embeddings=position_embeddings,
            # buffer=buffer,
            # buffer_sink_ids=buffer_sink_ids,
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
        self.norm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            batch:Batch,
    ) -> torch.Tensor:
        #attention_mask = proc_mask(attention_mask, batch.dtype())
        hidden_states = batch.input_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                batch
                # attention_mask=attention_mask,
                # position_embeddings=position_embeds,
                # buffer=buffer,
                # buffer_sink_ids=buffer_sink_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states




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


class L4maRotaryEmbedding(nn.Module):
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


def get_rope_index(
        config,
        input_ids: torch.LongTensor,
        image_grid_thw: torch.LongTensor | None,
        video_grid_thw: torch.LongTensor | None,
        second_per_grid_ts: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # print("input_ids: ", input_ids)
    # print("image_grid_thw: ", image_grid_thw)
    # print("video_grid_thw: ", video_grid_thw)
    # print("second_per_grid_ts: ", second_per_grid_ts)
    # print("attention_mask: ", attention_mask)

    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = config.vision_config.spatial_merge_size
    image_token_id = config.image_token_id
    video_token_id = config.video_token_id
    vision_start_token_id = config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0

        for i, input_ids in enumerate(total_input_ids):
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        # print(position_ids)

        return position_ids, mrope_position_deltas
    else:

        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return position_ids, mrope_position_deltas


# token-level attention buffer
class AttentionStorage:

    def __init__(self, num_layers: int, num_blocks: int, num_heads: int, block_size: int, head_dim: int, device: str, dtype=torch.bfloat16):
        self.ptr = torch.empty((num_layers, num_blocks, num_heads, block_size * 2, head_dim), device=device, dtype=dtype)

        self.dtype = dtype
        self.device = device

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim


class VectorStorage:

    def __init__(self, num_embeds: int, embed_dim: int, device: str, dtype=torch.bfloat16):
        self.ptr = torch.empty((num_embeds, embed_dim), device=device, dtype=dtype)

        self.dtype = dtype
        self.device = device

        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
