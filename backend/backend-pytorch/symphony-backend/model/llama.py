import copy
import math
from typing import List, Optional, Tuple, Union, override

import torch
from torch import nn
from transformers import BitsAndBytesConfig

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig

from symphony.engine import TaskBatch
from symphony.model import Model
from symphony.ops import create_rope_cache, rope, copy_kv_block, qkv_attention, qkv_attention_baseline, rope_baseline, copy_kv_block_baseline


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
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


class LlamaMLP(nn.Module):
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


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # self.rotary_emb = LlamaRotaryEmbedding(
        #     self.head_dim,
        #     max_position_embeddings=self.max_position_embeddings,
        #     base=self.rope_theta,
        # )

    def forward(
            self,
            kv_cache: torch.Tensor,
            rope_cache: torch.Tensor,
            batch: TaskBatch,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # prompt_len = len(batch.tasks[0].token_ids)
        # h_checksum = hidden_states[:, :prompt_len, :].sum()
        # q_checksum = q[:, :prompt_len, :].sum()
        # k_checksum = k[:, :prompt_len, :].sum()
        # v_checksum = v[:, :prompt_len, :].sum()
        # print('CHECKSUM', prompt_len, h_checksum, q_checksum, k_checksum, v_checksum)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        q = rope(rope_cache, q, batch.position_offsets)
        k = rope(rope_cache, k, batch.position_offsets)

        # q_checksum = q[:, :, :prompt_len, :].sum()
        # k_checksum = k[:, :, :prompt_len, :].sum()
        #
        # print(q.shape, k.shape)
        #
        # print('QK CHECKSUM', q_len, q_checksum, k_checksum)
        # exit()

        copy_kv_block(kv_cache, k, v, batch.kv_drain_addr_lut)
        # print( batch.kv_drain_addr_lut)

        attn_output = qkv_attention(
            q,
            kv_cache,
            batch_size=batch.batch_size(),
            num_grps=batch.num_tasks(),
            max_grp_size=batch.max_grp_size(),
            num_blocks_per_batch=batch.num_blocks_per_batch(),
            q_lut=batch.q_lut,
            kv_lut=batch.kv_lut,
            mask_lut=batch.mask_lut,
            reduce_grp_lut=batch.reduce_grp_lut,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            kv_cache: torch.Tensor,
            rope_cache: torch.Tensor,
            batch: TaskBatch,
            hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            kv_cache,
            rope_cache,
            batch,
            hidden_states,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"


class LlamaModel(LlamaPreTrainedModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
            self,
            kv_cache: list[torch.Tensor],
            rope_cache: torch.Tensor,
            task_batch: TaskBatch
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(task_batch.token_ids)
        #prompt_len = len(task_batch.tasks[0].token_ids)

        # print('INITIAL', prompt_len, hidden_states[:, :prompt_len, :].sum())
        # decoder layers
        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                kv_cache[i],
                rope_cache,
                task_batch,
                hidden_states,
            )

            # print('LAYER', i, prompt_len, hidden_states[:, :prompt_len, :].sum())

        hidden_states = self.norm(hidden_states)

        return hidden_states


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rope_cache = None


class Llama(Model):
    rope_cache: torch.Tensor | None

    def __init__(self, model_path, device):
        super().__init__(model_path)
        self.llama = LlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True,
                                                   llm_int8_threshold=200.0),
            torch_dtype=torch.float16,
            device_map=device)

        self.rope_cache = None

    @override
    def forward(self, kv_cache: list[torch.Tensor], batch: TaskBatch) -> torch.Tensor:
        if self.rope_cache is None:
            head_dim = self.llama.config.hidden_size // self.llama.config.num_attention_heads
            self.rope_cache = create_rope_cache(8192, head_dim, kv_cache[0].dtype, kv_cache[0].device)

        hidden_states = self.llama.model.forward(
            kv_cache,
            self.rope_cache,
            batch,
        )

        logits = self.llama.lm_head(hidden_states).float()
        return logits

    @override
    def kv_shape(self) -> Tuple[int, int, int]:
        head_dim = self.llama.config.hidden_size // self.llama.config.num_attention_heads

        return self.llama.config.num_hidden_layers, self.llama.config.num_key_value_heads, head_dim
