# deepseek.py
import torch
from torch import nn
from transformers import PreTrainedModel, LlamaConfig   # Re-use the config class

from deepseek_ma import DsmaModel                       # the MLA backbone


class DeepSeekForCausalLM(PreTrainedModel):
    """
    Lightweight wrapper that matches the structure of your existing
    LlamaForCausalLM class.  No custom `forward` is required because the
    surrounding code calls `.model.forward(...)` and then `.lm_head(...)`
    explicitly.
    """
    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig):
        config.use_qkv_bias = False          # keep parity with your Llama path
        super().__init__(config)

        self.model = DsmaModel(config)       # DeepSeek backbone
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # ----- embedding helpers (HF API) ------------------------------------ #
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings