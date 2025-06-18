import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from q4wen import Q4wenModel


class QwenForCausalLM(PreTrainedModel, GenerationMixin):
    """
    Qwen model for causal language modeling using FlashInfer for efficient attention.
    This class follows the same pattern as LlamaForCausalLM.
    """
    config_class = Qwen3Config

    def __init__(self, config):
        # Ensure config compatibility for FlashInfer
        if not hasattr(config, 'num_key_value_heads'):
            if hasattr(config, 'num_kv_heads'):
                config.num_key_value_heads = config.num_kv_heads
            else:
                config.num_key_value_heads = getattr(config, 'num_attention_heads', 32)

        super().__init__(config)
        self.model = Q4wenModel(config)
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

    # == Test only ==
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        """
        Forward pass for QwenForCausalLM.
        
        NOTE: This method is only used by test scripts and standalone usage.
        In production, the Symphony driver calls self.model.forward() directly,
        bypassing this wrapper method entirely.
        """
        # Convert input_ids to embeddings if needed
        if input_ids is not None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        else:
            inputs_embeds = kwargs.get('inputs_embeds')

        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device

        # Create simple position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Check if we have proper FlashInfer parameters from the driver
        has_flashinfer_params = all(param is not None for param in [
            kwargs.get('kv_cache_at_layer'),
            kwargs.get('kv_page_indices'),
            kwargs.get('kv_page_indptr'),
            kwargs.get('kv_last_page_lens'),
            kwargs.get('qo_indptr')
        ])

        if has_flashinfer_params:
            # Production path: use driver-provided FlashInfer parameters
            outputs = self.model(
                input_embeds=inputs_embeds,
                position_ids=position_ids,
                **kwargs
            )
        else:
            # Testing/fallback path: create minimal dummy FlashInfer parameters
            
            from config import NUM_TOKENS_IN_BLOCK
            
            # Calculate number of pages needed
            total_tokens = batch_size * seq_len
            num_pages = (total_tokens + NUM_TOKENS_IN_BLOCK - 1) // NUM_TOKENS_IN_BLOCK
            
            # Create dummy KV cache (simplified single-page setup)
            head_dim = getattr(self.config, 'head_dim', self.config.hidden_size // self.config.num_attention_heads)
            kv_cache_at_layer = [
                torch.zeros(
                    (num_pages, 2, NUM_TOKENS_IN_BLOCK, self.config.num_key_value_heads, head_dim),
                    dtype=inputs_embeds.dtype,
                    device=device
                ) for _ in range(self.config.num_hidden_layers)
            ]
            
            # Create dummy page indices and pointers for simple sequential layout
            kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            kv_page_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * (num_pages // batch_size)
            kv_last_page_lens = torch.full((batch_size,), seq_len % NUM_TOKENS_IN_BLOCK or NUM_TOKENS_IN_BLOCK, dtype=torch.int32, device=device)
            qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
            
            # Create causal mask (lower triangular)
            custom_mask = torch.tril(torch.ones(total_tokens, total_tokens, dtype=torch.bool, device=device))
            
            outputs = self.model(
                input_embeds=inputs_embeds,
                position_ids=position_ids,
                kv_cache_at_layer=kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                qo_indptr=qo_indptr,
                custom_mask=custom_mask,
                single_token_inference_mode=False,
            )

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs
        logits = self.lm_head(hidden_states)

        return type('ModelOutput', (), {
            'logits': logits,
            'past_key_values': getattr(outputs, 'past_key_values', None),
            'hidden_states': getattr(outputs, 'hidden_states', None),
            'attentions': getattr(outputs, 'attentions', None),
        })()
