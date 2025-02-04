import torch
import torch.nn as nn
from transformers import PreTrainedModel

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from qwen_vit import Qwen2_5_VisionTransformerPretrainedModel

from l4ma import AttentionBuffer, L4maRotaryEmbedding, L4maModel


class Qwen2_5_VLForConditionalGeneration(PreTrainedModel):
    config_class = Qwen2_5_VLConfig

    def __init__(self, config):

        config.use_qkv_bias = True

        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = L4maModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = L4maRotaryEmbedding(config=config)

    def forward(
            self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
            buffer: AttentionBuffer,
            buffer_sink_ids: list[int],

            pixel_values: torch.Tensor | None = None,
            pixel_values_videos: torch.FloatTensor | None = None,
            image_grid_thw: torch.LongTensor | None = None,
            video_grid_thw: torch.LongTensor | None = None,

    ) -> torch.Tensor:

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

        # prepare position embeds
        # create position embeddings to be shared across the decoder layers
        # position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # (bsz, dim)
        # cos_ref, sin_ref = position_embeddings
        cos, sin = self.rotary_emb(inputs_embeds, position_ids.max().item() + 1)

        dim = cos.shape[-1]
        bsz = inputs_embeds.shape[0]
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

        final_hidden_states = self.model(
            inputs_embeds=inputs_embeds,
            position_embeds=position_embeddings,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )

        logits = self.lm_head(final_hidden_states)

        return logits
