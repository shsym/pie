from __future__ import annotations

import random

import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn

from common import ceil_div


@triton.jit
def rope_kernel(
    # Shape: (max_pos, head_dim)
    rope_cache_ptr,
    # Shape: (batch_size, num_head, block_size, head_dim)
    x_ptr,
    # Shape: (batch_size, )
    start_pos_ptr,
    rope_stride_pos, rope_stride_dim,
    x_stride_bch, x_stride_hd, x_stride_blk, x_stride_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(start_pos_ptr + batch_idx)

    # Shape: (block_size, head_dim/2)
    cache_block_ptr = tl.make_block_ptr(
        base=rope_cache_ptr,
        shape=(MAX_POS, HEAD_DIM),
        strides=(rope_stride_pos, rope_stride_dim),
        offsets=(pos, 0),
        block_shape=(BLOCK_SIZE, HEAD_DIM // 2),
        order=(0, 1)
    )

    cos = tl.load(cache_block_ptr)
    sin = tl.load(tl.advance(cache_block_ptr, (0, HEAD_DIM // 2)))

    # Shape: (block_size, head_dim/2)
    x1_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * x_stride_bch + head_idx * x_stride_hd,
        shape=(BLOCK_SIZE, HEAD_DIM),
        strides=(x_stride_blk, x_stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, HEAD_DIM // 2),
        order=(0, 1)
    )
    x2_ptr = tl.advance(x1_ptr, (0, HEAD_DIM // 2))

    x1 = tl.load(x1_ptr)
    x2 = tl.load(x2_ptr)

    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin

    tl.store(x1_ptr, x1_rotated)
    tl.store(x2_ptr, x2_rotated)


@triton.jit
def rope_kernel_scatter_gather(
    # Shape: (max_pos, head_dim)
    rope_table_ptr,
    # Shape: (batch_size, num_head, block_size, head_dim)
    x_ptr,
    # Shape: (batch_size, block_size)
    rope_idxs_ptr,
    x_stride_bch, x_stride_hd,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    rope_idx_ptr = rope_idxs_ptr + batch_idx * BLOCK_SIZE
    x_ptr = x_ptr + batch_idx * x_stride_bch + head_idx * x_stride_hd

    # Triton will likely unroll and pipeline the operations in the loop body
    for _ in tl.static_range(BLOCK_SIZE):
        rope_idx = tl.load(rope_idx_ptr)
        cache_block_ptr = rope_table_ptr + rope_idx * HEAD_DIM
        cos = tl.load(cache_block_ptr + tl.arange(0, HEAD_DIM // 2))
        sin = tl.load(cache_block_ptr + HEAD_DIM // 2 + tl.arange(0, HEAD_DIM // 2))

        x1_ptr = x_ptr + tl.arange(0, HEAD_DIM // 2)
        x2_ptr = x1_ptr + HEAD_DIM // 2

        x1 = tl.load(x1_ptr)
        x2 = tl.load(x2_ptr)

        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x2 * cos + x1 * sin
        
        tl.store(x1_ptr, x1_rotated)
        tl.store(x2_ptr, x2_rotated)

        rope_idx_ptr += 1
        x_ptr += HEAD_DIM


@triton.jit
def rope_kernel_pre_indexed(
    # Shape: (batch_size, block_size, head_dim)
    rope_cache_ptr,
    # Shape: (batch_size, num_head, block_size, head_dim)
    x_ptr,
    rope_stride_bch, rope_stride_blk, rope_stride_dim,
    x_stride_bch, x_stride_hd, x_stride_blk, x_stride_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Shape: (block_size, head_dim/2)
    cache_block_ptr = tl.make_block_ptr(
        base=rope_cache_ptr + batch_idx * rope_stride_bch,
        shape=(BLOCK_SIZE, HEAD_DIM),
        strides=(rope_stride_blk, rope_stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, HEAD_DIM // 2),
        order=(0, 1)
    )

    cos = tl.load(cache_block_ptr)
    sin = tl.load(tl.advance(cache_block_ptr, (0, HEAD_DIM // 2)))

    # Shape: (block_size, head_dim/2)
    x1_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * x_stride_bch + head_idx * x_stride_hd,
        shape=(BLOCK_SIZE, HEAD_DIM),
        strides=(x_stride_blk, x_stride_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, HEAD_DIM // 2),
        order=(0, 1)
    )
    x2_ptr = tl.advance(x1_ptr, (0, HEAD_DIM // 2))

    x1 = tl.load(x1_ptr)
    x2 = tl.load(x2_ptr)

    x1_rotated = x1 * cos - x2 * sin
    x2_rotated = x2 * cos + x1 * sin

    tl.store(x1_ptr, x1_rotated)
    tl.store(x2_ptr, x2_rotated)


@triton.jit
def fill_kv_block_storage_kernel(dst_kv_ptr,  # destination block table float(I1, H, 2S, D)
                                 src_k_ptr,  # source block table float(I2, H, S, D)
                                 src_v_ptr,  # source block table float(I2, H, S, D)
                                 dst_lut,  # int(I2,)
                                 stride_dst_i, stride_dst_h, stride_dst_2s, stride_dst_d,  # strides for destination block table
                                 stride_src_i, stride_src_h, stride_src_s, stride_src_d,  # strides for source block table
                                 BLOCK_SIZE: tl.constexpr,
                                 BLOCK_DIM: tl.constexpr
                                 ):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    src_idx = batch_idx
    dst_idx = tl.load(dst_lut + batch_idx)

    dst_kv_block_ptr = tl.make_block_ptr(
        base=dst_kv_ptr + dst_idx * stride_dst_i + head_idx * stride_dst_h,
        shape=(2 * BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_dst_2s, stride_dst_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    src_k_block_ptr = tl.make_block_ptr(
        base=src_k_ptr + src_idx * stride_src_i + head_idx * stride_src_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_src_s, stride_src_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    src_v_block_ptr = tl.make_block_ptr(
        base=src_v_ptr + src_idx * stride_src_i + head_idx * stride_src_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_src_s, stride_src_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    tl.store(dst_kv_block_ptr, tl.load(src_k_block_ptr))
    tl.store(tl.advance(dst_kv_block_ptr, (BLOCK_SIZE, 0)), tl.load(src_v_block_ptr))


@triton.jit
def reduce_y_slices_kernel(
        # Unreduced attention results
        # Shape: (num_chunk_in_batch, num_head, num_tok_per_blk, head_dim)
        y_ptr,  # unreduced y float(I1, H, S, D)
        # Attention stats tensor (required for reducing y)
        # Shape: (num_chunk_in_batch, num_head, 2 * num_tok_per_blk)
        attn_stats_ptr,
        # Shape: (batch_size, num_chunk_per_req)
        reduce_grp_ptr,
        # Output reduced attention results
        # Shape: (batch_size, num_head, num_tok_per_blk, head_dim)
        y_reduced_ptr,
        # Strides to access the data in the tensors
        stride_y_i, stride_y_h, stride_y_s, stride_y_d,
        stride_attn_i, stride_attn_h, stride_attn_2s,
        stride_y_reduced_i, stride_y_reduced_h, stride_y_reduced_s, stride_y_reduced_d,
        # Number of tokens contained by a block
        NUM_TOK_PER_BLK: tl.constexpr,
        # The hidden dimension of the model (head_dim)
        HEAD_DIM: tl.constexpr,
        # Number of chunks in a request
        NUM_CHUNK_PER_REQ: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    y_reduced = tl.zeros([NUM_TOK_PER_BLK, HEAD_DIM], dtype=tl.float32)
    attn_max_reduced = tl.zeros([NUM_TOK_PER_BLK], dtype=tl.float32)
    attn_sum_reduced = tl.zeros([NUM_TOK_PER_BLK], dtype=tl.float32)

    # Reduce attention results across chunks (flash decoding)
    for i in range(NUM_CHUNK_PER_REQ):
        block_idx = tl.load(reduce_grp_ptr + req_idx * NUM_CHUNK_PER_REQ + i)

        # When requests in the batch have unequal length, shorter requests
        # will be padded and set the padding block index to -1. We should
        # skip the padding blocks.
        if block_idx >= 0:

            y_block_ptr = tl.make_block_ptr(
                base=y_ptr + block_idx * stride_y_i + head_idx * stride_y_h,
                shape=(NUM_TOK_PER_BLK, HEAD_DIM),
                strides=(stride_y_s, stride_y_d),
                offsets=(0, 0),
                block_shape=(NUM_TOK_PER_BLK, HEAD_DIM),
                order=(0, 1)
            )

            attn_block_ptr = attn_stats_ptr + block_idx * stride_attn_i + head_idx * stride_attn_h

            if i == 0:
                y_reduced = tl.load(y_block_ptr).to(tl.float32)
                attn_max_reduced = tl.load(attn_block_ptr + tl.arange(0, NUM_TOK_PER_BLK))
                attn_sum_reduced = tl.load(attn_block_ptr + tl.arange(NUM_TOK_PER_BLK, NUM_TOK_PER_BLK * 2))

            else:
                y = tl.load(y_block_ptr)
                attn_max = tl.load(attn_block_ptr + tl.arange(0, NUM_TOK_PER_BLK))
                attn_sum = tl.load(attn_block_ptr + tl.arange(NUM_TOK_PER_BLK, NUM_TOK_PER_BLK * 2))

                # re-normalize the softmax
                attn_max_diff = tl.maximum(attn_max_reduced, attn_max)
                alpha = tl.math.exp2(attn_max_reduced - attn_max_diff)
                beta = tl.math.exp2(attn_max - attn_max_diff)

                y_reduced = y_reduced * alpha[:, None] + y * beta[:, None]
                attn_sum_reduced = attn_sum_reduced * alpha + attn_sum * beta
                attn_max_reduced = attn_max_diff

    y_reduced = y_reduced / attn_sum_reduced[:, None]

    y_reduced_block_ptr = tl.make_block_ptr(
        base=y_reduced_ptr + req_idx * stride_y_reduced_i + head_idx * stride_y_reduced_h,
        shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        strides=(stride_y_reduced_s, stride_y_reduced_d),
        offsets=(0, 0),
        block_shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        order=(0, 1)
    )

    tl.store(y_reduced_block_ptr, y_reduced.to(y_ptr.dtype.element_ty))


def rope_(
        # Shape: (max_pos, head_dim)
        rope_cache: torch.Tensor,
        # Shape: (batch_size, num_head, block_size, head_dim)
        x: torch.Tensor,
        # Shape: (batch_size, )
        start_pos: torch.Tensor | list[int],
):
    if isinstance(start_pos, list):
        start_pos = torch.tensor(start_pos, dtype=torch.int32, device=x.device)

    max_pos, head_dim = rope_cache.shape
    batch_size, num_head, block_size, head_dim_ = x.shape
    batch_size_, = start_pos.shape

    assert head_dim == head_dim_
    assert batch_size == batch_size_

    grid = (batch_size, num_head)

    rope_kernel[grid](
        rope_cache,
        x,
        start_pos,
        rope_cache.stride(0), rope_cache.stride(1),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        block_size, head_dim, max_pos
    )


def rope_scatter_gather_(
        # Shape: (max_pos, head_dim)
        rope_table: torch.Tensor,
        # Shape: (batch_size, num_head, block_size, head_dim)
        x: torch.Tensor,
        # Shape: (batch_size, block_size)
        rope_idxs: torch.Tensor,
):
    max_pos, head_dim = rope_table.shape
    batch_size, num_head, block_size, head_dim_ = x.shape
    batch_size_, block_size_ = rope_idxs.shape

    assert head_dim == head_dim_
    assert batch_size == batch_size_
    assert block_size == block_size_

    grid = (batch_size, num_head)

    rope_kernel_scatter_gather[grid](
        rope_table,
        x,
        rope_idxs,
        x.stride(0), x.stride(1),
        block_size, head_dim,
    )


def rope_pre_indexed_(
        # Shape: (batch_size, block_size, head_dim)
        rope_cache: torch.Tensor,
        # Shape: (batch_size, num_head, block_size, head_dim)
        x: torch.Tensor,
):
    batch_size, block_size, head_dim = rope_cache.shape
    batch_size_, num_head, block_size_, head_dim_ = x.shape

    assert block_size == block_size_
    assert head_dim == head_dim_
    assert batch_size == batch_size_

    grid = (batch_size, num_head)

    rope_kernel_pre_indexed[grid](
        rope_cache,
        x,
        rope_cache.stride(0), rope_cache.stride(1), rope_cache.stride(2),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        block_size, head_dim
    )


# pytorch-only inefficient implementation of rope for testing purposes
def rope_baseline(
        # Shape: (max_pos, head_dim)
        rope_cache: torch.Tensor,
        # Shape: (batch_size, num_head, block_size, head_dim)
        x: torch.Tensor,
        # Shape: (batch_size, )
        start_pos: torch.Tensor,
):
    # Split out the cos and sin part in the cache
    _, head_dim = rope_cache.shape
    cos_cache = rope_cache[..., : head_dim // 2]
    sin_cache = rope_cache[..., head_dim // 2 :]

    # Shape: (max_pos, head_dim)
    cos_cache = torch.cat([cos_cache, cos_cache], dim=-1)
    sin_cache = torch.cat([sin_cache, sin_cache], dim=-1)

    _, _, block_size, _ = x.shape

    # Shape: (batch_size, block_size)
    x_pos = start_pos.unsqueeze(1) + torch.arange(0, block_size, device=x.device).unsqueeze(0)

    # Shape: (max_pos, head_dim)
    cos = cos_cache[x_pos].unsqueeze(1)
    sin = sin_cache[x_pos].unsqueeze(1)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    # Shape: (batch_size, num_head, block_size, head_dim)
    x_inverted = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (x_inverted * sin)

    return x_rotated


def rope_baseline_no_cache(
        # Shape: (batch_size, num_head, block_size, head_dim)
        x: torch.Tensor,
        # Shape: (batch_size, )
        start_pos: torch.Tensor,
        base: int = 50000,
) -> torch.Tensor:
    _batch_size, _num_head, block_size, head_dim = x.shape

    # Shape: (head_dim/2, )
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))

    max_pos = torch.max(start_pos).item() + block_size

    # Shape: (max_pos, )
    all_pos = torch.arange(max_pos, device=x.device, dtype=inv_freq.dtype)

    # Shape: (max_pos, head_dim)
    theta = torch.outer(all_pos, inv_freq)
    theta = torch.cat((theta, theta), dim=-1)

    # Shape: (max_pos, head_dim)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # Shape: (batch_size, block_size)
    x_pos = start_pos.unsqueeze(1) + torch.arange(0, block_size, device=x.device).unsqueeze(0)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    # Shape: (batch_size, 1, block_size, head_dim)
    cos = cos[x_pos].unsqueeze(1)
    sin = sin[x_pos].unsqueeze(1)

    # Shape: (batch_size, num_head, block_size, head_dim)
    x_inverted = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (x_inverted * sin)

    return x_rotated


def create_rope_cache(
        max_pos: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        base: int = 50000
) -> torch.Tensor:
    # Shape: (head_dim/2, )
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device) / head_dim))

    # Shape: (max_pos, )
    all_pos = torch.arange(max_pos, device=device)

    # Shape: (max_pos, head_dim/2)
    theta = torch.outer(all_pos, inv_freq)

    # Shape: (max_pos, head_dim/2)
    cos, sin = torch.cos(theta), torch.sin(theta)

    # Shape: (max_pos, head_dim)
    return torch.cat((cos, sin), dim=-1).to(dtype)


# previousely copy_kv_block
def fill_kv_block_storage(
        dst_kv: torch.Tensor,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        dst_lut: torch.Tensor,
):
    # print(src_k.shape)
    # print(dst_kv.shape)
    _, num_head, block_size, block_dim = src_k.shape
    _, num_head_, block_size2, block_dim_ = dst_kv.shape

    assert block_dim == block_dim_
    assert block_size == block_size2 // 2
    assert num_head == num_head_

    grid = (dst_lut.shape[0], num_head)

    fill_kv_block_storage_kernel[grid](
        dst_kv,
        src_k,
        src_v,
        dst_lut,
        dst_kv.stride(0), dst_kv.stride(1), dst_kv.stride(2), dst_kv.stride(3),
        src_k.stride(0), src_k.stride(1), src_k.stride(2), src_k.stride(3),
        block_size, block_dim
    )


def fill_kv_block_storage_baseline(
        dst_kv: torch.Tensor,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        dst_lut: torch.Tensor,
):
    _, num_head, block_size, block_dim = src_k.shape
    _, num_head_, block_size2, block_dim_ = dst_kv.shape

    assert block_dim == block_dim_
    assert block_size == block_size2 // 2
    assert num_head == num_head_

    for i in range(dst_lut.shape[0]):
        dst_idx = dst_lut[i]
        dst_kv[dst_idx, :, :block_size, :] = src_k[i]
        dst_kv[dst_idx, :, block_size:, :] = src_v[i]


# The "everything" kernel

@triton.jit
def qkv_attention_kernel(
        # Input query tensors
        # Shape: (batch_size, num_head, num_tok_per_blk, head_dim)
        q_table_ptr,
        # Input KV cache table tensors
        # Shape: (num_kv_block, num_kv_head, 2*num_tok_per_blk, head_dim)
        kv_table_ptr,
        # Output tensor
        # Shape: (batch_size, num_head, num_tok_per_blk, head_dim)
        y_ptr,
        # Attention stats tensor (required for reducing y)
        # Shape: (num_chunk_in_batch, num_head, 2 * num_tok_per_blk)
        attn_stats_ptr,
        # Indicies into `q_table`
        # For each chunk give the index of the query tensor in `q_table`
        # Shape: (num_chunk_in_batch, 1)
        q_idxs_ptr,
        # Indicies into `kv_table`
        # For each chunk and for each block in the chunk, give the index of the key and
        # value tensor in `kv_table`
        # Shape: (num_chunk_in_batch, num_blk_per_chunk)
        kv_idxs_ptr,
        # Attention mask
        # Shape: (num_chunk_in_batch, num_tok_per_blk, num_blk_per_chunk * num_tok_per_blk)
        mask_ptr,
        # Strides to access the data in the tensors
        stride_q_i, stride_q_h, stride_q_s, stride_q_d,
        stride_kv_i, stride_kv_h, stride_kv_2s, stride_kv_d,
        stride_mask_i, stride_mask_j, stride_mask_k,
        stride_y_i, stride_y_h, stride_y_s, stride_y_d,
        stride_attn_i, stride_attn_h, stride_attn_2s,
        normalize_at_the_end,
        # Scaling factor for the softmax
        sm_scale: tl.constexpr,
        # Number of tokens contained by a block
        NUM_TOK_PER_BLK: tl.constexpr,
        # The hidden dimension of the model (head_dim)
        HEAD_DIM: tl.constexpr,
        # Number of blocks contained by a chunk
        NUM_BLK_PER_CHUNK: tl.constexpr,
        # Needed for Grouped-Query Attention (https://arxiv.org/pdf/2305.13245)
        # If NUM_HEAD_GROUPS is 1, then this is a standard multi-head attention.
        NUM_HEAD_GROUPS: tl.constexpr = 1,  # 1 means no GQA.
        # Whether to use tf32 in computing
        ALLOW_TF32: tl.constexpr = True,
):
    glb_chunk_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    head_kv_idx = head_q_idx // NUM_HEAD_GROUPS

    q_idx = tl.load(q_idxs_ptr + glb_chunk_idx)

    q_block_ptr = tl.make_block_ptr(
        base=q_table_ptr + q_idx * stride_q_i + head_q_idx * stride_q_h,
        shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        strides=(stride_q_s, stride_q_d),
        offsets=(0, 0),
        block_shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        order=(0, 1)
    )

    mask_block_ptr = tl.make_block_ptr(
        base=mask_ptr + glb_chunk_idx * stride_mask_i,
        shape=(NUM_TOK_PER_BLK, NUM_TOK_PER_BLK * NUM_BLK_PER_CHUNK),
        strides=(stride_mask_j, stride_mask_k),
        offsets=(0, 0),
        block_shape=(NUM_TOK_PER_BLK, NUM_TOK_PER_BLK),
        order=(0, 1)
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr + glb_chunk_idx * stride_y_i + head_q_idx * stride_y_h,
        shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        strides=(stride_y_s, stride_y_d),
        offsets=(0, 0),
        block_shape=(NUM_TOK_PER_BLK, HEAD_DIM),
        order=(0, 1)
    )

    attn_block_ptr = attn_stats_ptr + glb_chunk_idx * stride_attn_i + head_q_idx * stride_attn_h

    qk_scale = sm_scale * 1.4426950408889634

    # Shape: (block_size, head_dim)
    q_block = tl.load(q_block_ptr)
    q_block = (q_block * qk_scale).to(kv_table_ptr.dtype.element_ty)

    attn_max_reduced = tl.zeros([NUM_TOK_PER_BLK], dtype=tl.float32) - 1000000.0  # float("inf")
    attn_sum_reduced = tl.zeros([NUM_TOK_PER_BLK], dtype=tl.float32)
    y_block = tl.zeros([NUM_TOK_PER_BLK, HEAD_DIM], dtype=tl.float32)

    for block_idx in range(NUM_BLK_PER_CHUNK):
        kv_idx = tl.load(kv_idxs_ptr + glb_chunk_idx * NUM_BLK_PER_CHUNK + block_idx)

        k_block_ptr = tl.make_block_ptr(
            base=kv_table_ptr + kv_idx * stride_kv_i + head_kv_idx * stride_kv_h,
            shape=(HEAD_DIM, NUM_TOK_PER_BLK),
            strides=(stride_kv_d, stride_kv_2s),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, NUM_TOK_PER_BLK),
            order=(1, 0)
        )

        v_block_ptr = tl.make_block_ptr(
            base=kv_table_ptr + kv_idx * stride_kv_i + head_kv_idx * stride_kv_h,
            shape=(NUM_TOK_PER_BLK, HEAD_DIM),
            strides=(stride_kv_2s, stride_kv_d),
            offsets=(NUM_TOK_PER_BLK, 0),
            block_shape=(NUM_TOK_PER_BLK, HEAD_DIM),
            order=(0, 1)
        )

        # Shape: (head_dim, block_size)
        k_block = tl.load(k_block_ptr)
        # Shape: (block_size, head_dim)
        v_block = tl.load(v_block_ptr)
        # Shape: (block_size, block_size)
        mask = tl.load(mask_block_ptr)

        # Shape: (block_size, block_size)
        attn = tl.dot(q_block, k_block, allow_tf32=ALLOW_TF32)

        # Looks like Triton uses int8 for torch bool types. I added cast to satisfy triton compiler.
        # Check if this cast is causing any performance issues.
        attn += tl.where(mask.cast(tl.int1), 0.0, -1000000.0)

        # Flash attention
        attn_max = tl.maximum(attn_max_reduced, tl.max(attn, axis=1))
        alpha = tl.math.exp2(attn_max_reduced - attn_max)
        attn = tl.math.exp2(attn - attn_max[:, None])

        # Shape: (block_size, head_dim)
        y_block = alpha[:, None] * y_block + tl.dot(attn.to(kv_table_ptr.dtype.element_ty), v_block, allow_tf32=ALLOW_TF32)

        attn_sum_reduced = attn_sum_reduced * alpha + tl.sum(attn, 1)
        attn_max_reduced = attn_max

        # Advance the mask ptr
        mask_block_ptr = tl.advance(mask_block_ptr, (0, NUM_TOK_PER_BLK))

    if normalize_at_the_end:
        y_block = y_block / attn_sum_reduced[:, None]
    tl.store(y_block_ptr, y_block.to(y_ptr.dtype.element_ty))

    tl.store(attn_block_ptr + tl.arange(0, NUM_TOK_PER_BLK), attn_max_reduced)
    tl.store(attn_block_ptr + tl.arange(NUM_TOK_PER_BLK, NUM_TOK_PER_BLK * 2), attn_sum_reduced)


def qkv_attention(
        # Shape: (batch_size, num_head, num_tok_per_blk, head_dim)
        q_table: torch.Tensor,
        # Shape: (num_kv_block, num_kv_head, 2*num_tok_per_blk, head_dim)
        kv_table: torch.Tensor,
        # Shape: (num_chunk_in_batch, 1)
        q_idxs: torch.Tensor,
        # Shape: (num_chunk_in_batch, num_blk_per_chunk)
        kv_idxs: torch.Tensor,
        # Shape: (num_chunk_in_batch, num_tok_per_blk, num_blk_per_chunk * num_tok_per_blk)
        mask: torch.Tensor,
        # Shape: (batch_size, num_chunk_per_req)
        reduce_grp: torch.Tensor,
        # Whether to use tf32 in computing
        allow_tf32: bool = True,
) -> torch.Tensor:
    batch_size, num_head, num_tok_per_blk, head_dim = q_table.shape
    _, num_kv_head, num_tok_per_blk2, head_dim_ = kv_table.shape

    assert num_tok_per_blk * 2 == num_tok_per_blk2
    assert head_dim == head_dim_
    assert num_head % num_kv_head == 0

    num_chunk_in_batch = q_idxs.shape[0]
    num_chunk_per_req = reduce_grp.shape[1]
    num_blk_per_chunk = kv_idxs.shape[1]

    # Check if it is a GQA
    num_gqa_groups = num_head // num_kv_head

    grid = (num_chunk_in_batch, num_head)

    y = torch.empty(
        (num_chunk_in_batch, num_head, num_tok_per_blk, head_dim),
        device=q_table.device, dtype=q_table.dtype
    )
    attn_stats = torch.empty(
        (num_chunk_in_batch, num_head, 2 * num_tok_per_blk),
        device=q_table.device, dtype=torch.float32
    )

    # Check if we need to reduce y. If not, we do a normalization in this kernel.
    need_reduce = num_chunk_per_req > 1

    qkv_attention_kernel[grid](
        q_table,
        kv_table,
        y,
        attn_stats,
        q_idxs,
        kv_idxs,
        mask,
        q_table.stride(0), q_table.stride(1), q_table.stride(2), q_table.stride(3),
        kv_table.stride(0), kv_table.stride(1), kv_table.stride(2), kv_table.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        attn_stats.stride(0), attn_stats.stride(1), attn_stats.stride(2),
        not need_reduce,
        1.0 / (head_dim ** 0.5),
        num_tok_per_blk, head_dim, num_blk_per_chunk, num_gqa_groups,
        allow_tf32
    )

    # reduce y if necessary
    if need_reduce:
        y_reduced = torch.empty_like(q_table)

        grid = (batch_size, num_head)

        reduce_y_slices_kernel[grid](
            y,
            attn_stats,
            reduce_grp,
            y_reduced,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            attn_stats.stride(0), attn_stats.stride(1), attn_stats.stride(2),
            y_reduced.stride(0), y_reduced.stride(1), y_reduced.stride(2), y_reduced.stride(3),
            num_tok_per_blk, head_dim, num_chunk_per_req
        )

        y = y_reduced

    return y


def qkv_attention_baseline(
        # Shape: (batch_size, num_head, block_size, head_dim)
        q_table: torch.Tensor,
        # Shape: (num_kv_block, num_kv_head, 2*block_size, head_dim)
        kv_table: torch.Tensor,
        # Shape: (batch_size, 1)
        q_idxs: torch.Tensor,
        # Shape: (batch_size, num_blk_per_bch)
        kv_idxs: torch.Tensor,
        # Shape: (batch_size, block_size, num_blk_per_bch * block_size)
        mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_head, block_size, head_dim = q_table.shape
    _, num_kv_head, block_size2, head_dim_ = kv_table.shape
    num_blk_per_bch = kv_idxs.shape[1]

    assert block_size * 2 == block_size2
    assert head_dim == head_dim_
    assert num_head % num_kv_head == 0

    # Shape: (batch_size, num_head, block_size, head_dim)
    q = q_table[q_idxs].squeeze(1)

    # Shape: (batch_size, num_blk_per_bch, num_kv_head, 2*block_size, head_dim)
    kv = kv_table[kv_idxs]

    # Shape: (batch_size, num_blk_per_bch, num_kv_head, block_size, head_dim)
    k = kv[:, :, :, :block_size, :]
    v = kv[:, :, :, block_size:, :]

    # Shape: (batch_size, num_kv_head, seq_len, head_dim)
    k = k.transpose(1, 2).reshape(batch_size, num_kv_head, num_blk_per_bch * block_size, head_dim)
    v = v.transpose(1, 2).reshape(batch_size, num_kv_head, num_blk_per_bch * block_size, head_dim)

    # Repeat KV heads for GQA
    # Shape: (batch_size, num_head, seq_len, head_dim)
    k = torch.repeat_interleave(k, dim=1, repeats=num_head // num_kv_head)
    v = torch.repeat_interleave(v, dim=1, repeats=num_head // num_kv_head)

    # Shape: (batch_size, num_head, block_size, seq_len)
    attn = torch.einsum('nhqd,nhkd->nhqk', q, k) / (head_dim ** 0.5)

    m = torch.where(mask, 0.0, torch.finfo(attn.dtype).min).to(attn.dtype).unsqueeze(1)
    attn = attn + m
    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)

    # Shape: (batch_size, num_head, block_size, head_dim)
    return torch.einsum('nhqk,nhkd->nhqd', attn, v)
