# qwen.py
import torch
import warnings
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig


class QwenModelWrapper(nn.Module):
    """
    Wrapper for the Qwen3 model.model to provide the same forward interface
    as expected by the symphony driver.py
    
    This implementation uses a simplified approach that bridges the Symphony driver
    interface with the standard transformers model, avoiding complex FlashInfer integration
    while still maintaining compatibility with Symphony's expectations.
    """

    def __init__(self, qwen_model):
        super().__init__()
        self.qwen_model = qwen_model
        self.embed_tokens = qwen_model.embed_tokens
        self.config = qwen_model.config
        
        # Track state for debugging and cache management
        self._last_position = None
        self._sequence_id = None
        self._past_key_values = None  # Store transformers KV cache
        self._total_sequence_length = 0  # Track total sequence length

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache_at_layer,
        kv_page_indices: torch.Tensor,
        kv_page_indptr: torch.Tensor,
        kv_last_page_lens: torch.Tensor,
        qo_indptr: torch.Tensor,
        custom_mask: torch.Tensor = None,
        single_token_inference_mode: bool = False,
        **kwargs
    ):
        """
        Forward pass that provides Symphony driver compatibility.
        
        This implementation uses the transformers model's native KV cache
        to maintain state between single-token inference calls.
        """
        seq_len = input_embeds.shape[0]
        print(f"QwenModelWrapper forward: single_token_mode={single_token_inference_mode}, "
              f"seq_len={seq_len}, total_seq_len={self._total_sequence_length}, has_cache={self._past_key_values is not None}")
        
        # Convert to transformers format (add batch dimension)
        input_embeds_batched = input_embeds.unsqueeze(0)  # [seq_len, hidden] -> [1, seq_len, hidden]
        position_ids_batched = position_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        
        # Handle attention mask for proper context awareness
        attention_mask = None
        if single_token_inference_mode and self._past_key_values is not None:
            # In single-token mode, create attention mask that includes all previous tokens
            # The past_key_values already contains the cached context
            past_length = self._past_key_values[0][0].shape[2]  # Get the cached sequence length
            total_length = past_length + seq_len
            attention_mask = torch.ones((1, total_length), dtype=torch.long, device=input_embeds.device)
            print(f"QwenModelWrapper: Single-token mode WITH CACHE, past_length={past_length}, total_length={total_length}")
        elif single_token_inference_mode and self._past_key_values is None:
            print("QwenModelWrapper: Single-token mode but NO CACHE available")
        elif not single_token_inference_mode:
            # In multi-token mode (prefill), we reset everything
            print("QwenModelWrapper: Multi-token mode, resetting cache")
            self._past_key_values = None
            self._total_sequence_length = 0
        
        # Use transformers model with caching
        try:
            outputs = self.qwen_model(
                inputs_embeds=input_embeds_batched,
                position_ids=position_ids_batched,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,  # Use cached KV
                use_cache=True,  # Enable caching for state persistence
                return_dict=True
            )
            
            # Update cache only if we're actually using it
            if single_token_inference_mode or not single_token_inference_mode:
                self._past_key_values = outputs.past_key_values
                self._total_sequence_length += seq_len
            
            # Return the hidden states without batch dimension
            hidden_states = outputs.last_hidden_state.squeeze(0)  # [1, seq_len, hidden] -> [seq_len, hidden]
            
            print(f"QwenModelWrapper forward successful: output_shape={hidden_states.shape}, "
                  f"new_total_seq_len={self._total_sequence_length}, has_cache={self._past_key_values is not None}")
            return hidden_states
            
        except Exception as e:
            print(f"QwenModelWrapper forward failed: {e}")
            print(f"  input_embeds.shape: {input_embeds.shape}")
            print(f"  position_ids.shape: {position_ids.shape}")
            print(f"  single_token_inference_mode: {single_token_inference_mode}")
            print(f"  total_sequence_length: {self._total_sequence_length}")
            print(f"  past_key_values: {type(self._past_key_values)}")
            raise


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
            if hasattr(self.config, 'num_kv_heads'):
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

        # Debug print the configuration
        print(f"Model config validation:")
        print(f"  vocab_size: {getattr(self.config, 'vocab_size', 'NOT SET')}")
        print(f"  hidden_size: {self.config.hidden_size}")
        print(f"  num_attention_heads: {self.config.num_attention_heads}")
        print(f"  num_key_value_heads: {self.config.num_key_value_heads}")
        print(f"  num_hidden_layers: {self.config.num_hidden_layers}")

        # Check for potential EOS token issues
        if hasattr(self.config, 'eos_token_id'):
            print(f"  eos_token_id: {self.config.eos_token_id}")
        if hasattr(self.config, 'pad_token_id'):
            print(f"  pad_token_id: {self.config.pad_token_id}")
        if hasattr(self.config, 'bos_token_id'):
            print(f"  bos_token_id: {self.config.bos_token_id}")

        # Validate vocabulary size matches the model
        expected_vocab_size = self.lm_head.out_features
        config_vocab_size = getattr(self.config, 'vocab_size', expected_vocab_size)
        if expected_vocab_size != config_vocab_size:
            print(f"WARNING: Vocabulary size mismatch!")
            print(f"  lm_head.out_features: {expected_vocab_size}")
            print(f"  config.vocab_size: {config_vocab_size}")
            # Update config to match actual model
            self.config.vocab_size = expected_vocab_size
            self.vocab_size = expected_vocab_size

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
