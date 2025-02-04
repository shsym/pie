# Llama-Like Large Language Model Architecture (L4MA)
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sortedcontainers import SortedList
from safetensors import safe_open
from safetensors.torch import save_file


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

#
# "rope_scaling": {
#     "factor": 32.0,
#     "high_freq_factor": 4.0,
#     "low_freq_factor": 1.0,
#     "original_max_position_embeddings": 8192,
#     "rope_type": "llama3"
#   },

#
# "rope_scaling": {
#     "type": "mrope",
#     "mrope_section": [
#       16,
#       24,
#       24
#     ]
#   },
#
# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(self, config: LlamaConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings
#
#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
#
#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq
#
#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
#             self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len
#
#         if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
#             # This .to() is needed if the model has been moved to a device after being initialized (because
#             # the buffer is automatically moved, but not the original copy)
#             self.original_inv_freq = self.original_inv_freq.to(device)
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len
#
#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_ids, device=x.device)
#
#         # Core RoPE block
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#
#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling
#
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
#
#
#
#
# class Qwen2_5_VLRotaryEmbedding(nn.Module):
#     def __init__(self, config: Qwen2_5_VLConfig, device=None):
#         super().__init__()
#         # BC: "rope_type" was originally "type"
#         if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
#             self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
#         else:
#             self.rope_type = "default"
#         self.max_seq_len_cached = config.max_position_embeddings
#         self.original_max_seq_len = config.max_position_embeddings
#
#         self.config = config
#         self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
#
#         inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         self.original_inv_freq = self.inv_freq
#
#     def _dynamic_frequency_update(self, position_ids, device):
#         """
#         dynamic RoPE layers should recompute `inv_freq` in the following situations:
#         1 - growing beyond the cached sequence length (allow scaling)
#         2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
#         """
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_seq_len_cached:  # growth
#             inv_freq, self.attention_scaling = self.rope_init_fn(
#                 self.config, device, seq_len=seq_len, **self.rope_kwargs
#             )
#             self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
#             self.max_seq_len_cached = seq_len
#
#         if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
#             self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
#             self.max_seq_len_cached = self.original_max_seq_len
#
#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         if "dynamic" in self.rope_type:
#             self._dynamic_frequency_update(position_ids, device=x.device)
#
#         # Core RoPE block. In contrast to other models, Qwen2_5_VL has different position ids for thw grids
#         # So we expand the inv_freq to shape (3, ...)
#         inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
#         position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
#         # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#
#         # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
#         cos = cos * self.attention_scaling
#         sin = sin * self.attention_scaling
#
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
