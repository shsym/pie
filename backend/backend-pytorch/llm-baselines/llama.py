import torch
from torch import nn
from transformers import PreTrainedModel, LlamaConfig

from l4ma import AttentionBuffer, L4maRotaryEmbedding, L4maModel


class LlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        config.use_qkv_bias = False

        super().__init__(config)
        self.model = L4maModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = L4maRotaryEmbedding(config=config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: torch.LongTensor,
            position_ids: torch.LongTensor,
            attention_mask: torch.BoolTensor,
            buffer: AttentionBuffer,
            buffer_sink_ids: list[int],
    ) -> torch.Tensor:
        # print("input_ids", input_ids.shape)
        # print("position_ids", position_ids.shape)
        # print("attention_mask", attention_mask.shape)
        # print("buffer_sink_ids", buffer_sink_ids)

        input_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(input_embeds, position_ids.max().item() + 1)
        position_embeds = (cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1))

        final_hidden_states = self.model(
            inputs_embeds=input_embeds,
            position_embeds=position_embeds,
            attention_mask=attention_mask,
            buffer=buffer,
            buffer_sink_ids=buffer_sink_ids,
        )

        logits = self.lm_head(final_hidden_states)

        return logits
