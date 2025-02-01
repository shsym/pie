# Small, dummy LLM model for testing purposes


import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, override

import torch
from torch import nn

from symphony.engine import TaskBatch
from symphony.model import Model
from symphony.ops import create_rope_cache, rope, copy_kv_block, qkv_attention


@dataclass
class DummyConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    num_heads: int = 8
    num_key_value_heads: int = 4
    num_hidden_layers: int = 6


class DummyAttention(nn.Module):

    def __init__(self, config: DummyConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

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

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        rope(rope_cache, k, batch.position_offsets)
        rope(rope_cache, v, batch.position_offsets)
        copy_kv_block(kv_cache, k, v, batch.kv_drain_addr_lut)

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


class DummyModel(Model):
    config: DummyConfig

    def __init__(self, config):
        super().__init__("dummy")

        self.config = config

        self.vocab_size = config.vocab_size
        self.head_dim = config.hidden_size // config.num_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, 0)
        self.layers = nn.ModuleList([DummyAttention(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_cache = None

    @override
    def forward(
            self,
            kv_cache: list[torch.Tensor],
            tasks: TaskBatch
    ) -> torch.Tensor:

        if self.rope_cache is None:
            self.rope_cache = create_rope_cache(8192, self.head_dim, kv_cache[0].dtype, kv_cache[0].device)

        hidden_states = self.embed_tokens(tasks.token_ids)

        for i, attention in enumerate(self.layers):
            hidden_states = attention(
                kv_cache[i],
                self.rope_cache,
                tasks,
                hidden_states,
            )

        logits = self.lm_head(hidden_states).float()

        return logits

    @override
    def kv_shape(self) -> Tuple[int, int, int]:
        return self.config.num_hidden_layers, self.config.num_key_value_heads, self.head_dim
