# qwen.py
import torch
import warnings
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig


class QwenModelWrapper(nn.Module):
    """
    Wrapper for the Qwen3 model.model to provide the same forward interface
    as expected by the symphony driver.py
    """
    
    def __init__(self, qwen_model):
        super().__init__()
        self.qwen_model = qwen_model
        self.embed_tokens = qwen_model.embed_tokens
        
    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer,  # This will be ignored for standard Qwen3 models
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: torch.Tensor = None,
        single_token_inference_mode: bool = False,
        **kwargs
    ):
        """
        Forward pass that adapts the symphony driver interface to work with
        standard Qwen3 models. 
        
        Note: This is a simplified implementation that doesn't use the custom
        KV cache parameters. For full compatibility with symphony's paged attention,
        you would need to implement the custom KV cache handling.
        """
        # For now, we'll use the standard Qwen3 forward pass
        # In a full implementation, you'd need to handle the custom KV cache
        
        # Convert position_ids to the right shape if needed
        batch_size = 1  # Assuming single batch for now
        seq_len = input_embeds.shape[0]
        
        # Create attention mask if custom_mask is provided
        attention_mask = None
        if custom_mask is not None:
            # Convert custom_mask to attention_mask format expected by Qwen3
            attention_mask = custom_mask.unsqueeze(0)  # Add batch dimension
        
        # Reshape inputs for standard model
        input_embeds_reshaped = input_embeds.unsqueeze(0)  # Add batch dimension
        position_ids_reshaped = position_ids.unsqueeze(0)  # Add batch dimension
        
        # Call the standard Qwen3 model
        outputs = self.qwen_model(
            inputs_embeds=input_embeds_reshaped,
            position_ids=position_ids_reshaped,
            attention_mask=attention_mask,
            use_cache=False,  # Disable built-in caching since symphony handles it
            return_dict=True
        )
        
        # Return last hidden states, removing batch dimension
        return outputs.last_hidden_state.squeeze(0)


class QwenForCausalLM:
    """
    Wrapper for Qwen3 models that provides the same interface as DeepSeekForCausalLM
    but uses the standard transformers implementation under the hood.
    
    This is designed to work with models like:
    - deepseek-ai/DeepSeek-R1-0528-Qwen3-8B (which is actually a Qwen3 model)
    - Qwen/Qwen3-8B-Instruct
    - Any other Qwen3-based models
    """

    def __init__(self, transformers_model):
        """
        Initialize with a pre-loaded transformers model
        """
        self.transformers_model = transformers_model
        self.model = QwenModelWrapper(transformers_model.model)
        self.lm_head = transformers_model.lm_head
        self.config = transformers_model.config
        self.vocab_size = transformers_model.config.vocab_size
        
        # Ensure config has all required attributes for driver.py
        self._ensure_config_compatibility()

    def _ensure_config_compatibility(self):
        """
        Ensure the config has all attributes expected by driver.py
        """
        # Map Qwen3 config attributes to expected names if needed
        if not hasattr(self.config, 'num_key_value_heads'):
            # Qwen3 might use different attribute names
            if hasattr(self.config, 'num_key_value_heads'):
                pass  # Already has the right name
            elif hasattr(self.config, 'num_kv_heads'):
                self.config.num_key_value_heads = self.config.num_kv_heads
            else:
                # Default to num_attention_heads if not specified
                self.config.num_key_value_heads = getattr(self.config, 'num_attention_heads', 32)
        
        # Ensure other required attributes exist
        if not hasattr(self.config, 'hidden_size'):
            self.config.hidden_size = getattr(self.config, 'd_model', 4096)
            
        if not hasattr(self.config, 'num_attention_heads'):
            self.config.num_attention_heads = getattr(self.config, 'n_head', 32)
            
        if not hasattr(self.config, 'num_hidden_layers'):
            self.config.num_hidden_layers = getattr(self.config, 'n_layer', 24)

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Load a Qwen3 model using standard transformers AutoModelForCausalLM
        """
        print(f"Loading Qwen3 model: {model_name_or_path}")
        
        # Ensure trust_remote_code is set
        kwargs['trust_remote_code'] = True
        
        # Suppress the rope_scaling warning about unrecognized keys
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized keys in `rope_scaling`")
            
            # Load model using standard transformers
            transformers_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                **kwargs
            )
        
        print(f"Model loaded successfully: {type(transformers_model)}")
        return cls(transformers_model)

    # ----- embedding helpers (HF API) ------------------------------------ #
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
