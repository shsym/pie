import torch
from torch import nn
from transformers import PreTrainedModel, LlamaConfig

from l4ma import  L4maModel


class LlamaForCausalLM(PreTrainedModel):
    config_class = LlamaConfig

    def __init__(self, config):
        config.use_qkv_bias = False

        super().__init__(config)
        self.model = L4maModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    
