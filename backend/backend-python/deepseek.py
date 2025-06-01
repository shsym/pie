# deepseek.py
import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig

from deepseek_ma import DsmaModel                       # the MLA backbone


class DeepSeekForCausalLM(PreTrainedModel):
    """
    Lightweight wrapper that matches the structure of your existing
    LlamaForCausalLM class.  No custom `forward` is required because the
    surrounding code calls `.model.forward(...)` and then `.lm_head(...)`
    explicitly.
    """
    config_class = AutoConfig

    def __init__(self, config):
        # Add missing attributes to config if they don't exist
        if not hasattr(config, 'use_qkv_bias'):
            config.use_qkv_bias = False
        if not hasattr(config, 'num_latent_heads'):
            # For DeepSeek-V3, num_latent_heads is typically num_attention_heads // 4
            config.num_latent_heads = getattr(config, 'num_attention_heads', 32) // 4
        if not hasattr(config, 'head_dim_ckv'):
            # Typical values for DeepSeek-V3
            config.head_dim_ckv = 512
        if not hasattr(config, 'head_dim_kpe'):
            config.head_dim_kpe = 64
        if not hasattr(config, 'rms_norm_eps'):
            config.rms_norm_eps = getattr(config, 'layer_norm_eps', 1e-6)
            
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

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Custom from_pretrained that handles DeepSeek-V3 models properly
        """
        from transformers import AutoConfig
        
        # Load the config first and modify it to remove problematic settings
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Handle quantization issues by removing unsupported quantization configs
        if hasattr(config, 'quantization_config') and config.quantization_config is not None:
            config.quantization_config = None
            print("Warning: Removed quantization config to avoid compatibility issues")
        
        # Use the parent class from_pretrained with modified config
        kwargs['config'] = config
        return super().from_pretrained(model_name_or_path, **kwargs)