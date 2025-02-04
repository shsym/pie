import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from qwen_vit import Qwen2_5_VisionTransformerPretrainedModel

from common import Qwen2_5_VLPreTrainedModel
from l4ma import L4maRmsNorm, L4maMlp, AttentionBuffer, proc_mask, L4maRotaryEmbedding, apply_rotary_pos_emb, L4maAttention

#
# class Qwen2_5_VLAttention(nn.Module):
#
#     def __init__(self, config: Qwen2_5_VLConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.is_causal = True
#         self.attention_dropout = config.attention_dropout
#         self.rope_scaling = config.rope_scaling
#
#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )
#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
#         self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
#
#         # self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
#
#     def forward(
#             self,
#             hidden_states: torch.Tensor,
#             attention_mask: Optional[torch.Tensor] = None,
#             # position_ids: Optional[torch.LongTensor] = None,
#             # past_key_value: Optional[Cache] = None,
#             # output_attentions: bool = False,
#             # use_cache: bool = False,
#             # cache_position: Optional[torch.LongTensor] = None,
#             position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#
#             buffer: AttentionBuffer | None = None,
#             buffer_sink_ids: list[int] | None = None,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()
#
#         query_states = self.q_proj(hidden_states)
#         key_states = self.k_proj(hidden_states)
#         value_states = self.v_proj(hidden_states)
#
#         query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
#
#         cos, sin = position_embeddings
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
#
#         if buffer is not None:
#             buffer.sink(self.layer_idx, buffer_sink_ids, key_states, value_states)
#             key_states, value_states = buffer.cache(self.layer_idx, repeat=self.num_key_value_groups)
#
#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
#
#         if attention_mask is not None:  # no matter the length, we just slice it
#             # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + attention_mask
#
#         # Fix precision issues in Qwen2-VL float16 inference
#         # Replace inf values with zeros in attention weights to prevent NaN propagation
#         if query_states.dtype == torch.float16:
#             attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
#
#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)
#
#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )
#
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, -1)
#
#         attn_output = self.o_proj(attn_output)
#
#         return attn_output


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = L4maAttention(config, layer_idx, use_bias=True)

        self.mlp = L4maMlp(config)
        self.input_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
            # **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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


class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = L4maRmsNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb_new = L4maRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            inputs_embeds: torch.FloatTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.Tensor | None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        attention_mask = proc_mask(attention_mask, inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # (bsz, dim)
        # cos_ref, sin_ref = position_embeddings
        cos, sin = self.rotary_emb_new(hidden_states, position_ids.max().item() + 1)

        dim = cos.shape[-1]
        bsz = hidden_states.shape[0]
        # (3, bsz, dim)
        # print("cos: ", cos.shape)
        # print("sin: ", sin.shape)
        cos_ex = cos[None, None, :, :].expand(3, bsz, -1, -1)
        sin_ex = sin[None, None, :, :].expand(3, bsz, -1, -1)

        # print("position_ids: ", position_ids.max())
        cos = torch.gather(cos_ex, dim=2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, dim))
        sin = torch.gather(sin_ex, dim=2, index=position_ids.unsqueeze(-1).expand(-1, -1, -1, dim))

        mrope_section = self.config.rope_scaling["mrope_section"] * 2

        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            1
        )

        position_embeddings = (cos, sin)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                buffer=buffer,
                buffer_sink_ids=buffer_sink_ids,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
        )


@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None


class Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            buffer: AttentionBuffer | None = None,
            buffer_sink_ids: list[int] | None = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        # print all the inputs
        # print("input_ids: ", input_ids)
        # print("attention_mask: ", attention_mask)
        # print("position_ids: ", position_ids)
        # print("past_key_values: ", past_key_values)
        # print("inputs_embeds: ", inputs_embeds)
        # print("use_cache: ", use_cache)
        # print("output_attentions: ", output_attentions)
        # print("output_hidden_states: ", output_hidden_states)
        # print("return_dict: ", return_dict)
        # print("pixel_values: ", pixel_values)
        # print("pixel_values_videos: ", pixel_values_videos)
        # print("image_grid_thw: ", image_grid_thw)
        # print("video_grid_thw: ", video_grid_thw)
        # print("rope_deltas: ", rope_deltas)
        # print("cache_position: ", cache_position)
        # print("second_per_grid_ts: ", second_per_grid_ts)

        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        return Qwen2_5_VLCausalLMOutputWithPast(
            logits=logits
        )
