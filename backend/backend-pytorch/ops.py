from __future__ import annotations

import random

import numpy as np
import torch
import triton
import triton.language as tl
from torch import nn

from common import ceil_div


@triton.jit
def rope_kernel(rope_ptr,  # (N, D)
                x_ptr,  # (I, H, S, D)
                pos_ptr,  # (I, ) stores the position of each block
                rope_stride_i, rope_stride_d,
                x_stride_i, x_stride_h, x_stride_s, x_stride_d,
                BLOCK_SIZE: tl.constexpr,
                BLOCK_DIM: tl.constexpr,
                MAX_SEQ_LEN: tl.constexpr,
                ):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    rope_idx = tl.load(pos_ptr + batch_idx)

    # print("rope_idx", rope_idx)
    rope_block_ptr = tl.make_block_ptr(
        base=rope_ptr,
        shape=(MAX_SEQ_LEN, BLOCK_DIM),
        strides=(rope_stride_i, rope_stride_d),
        offsets=(rope_idx, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM // 2),
        order=(0, 1)
    )

    cos_block = tl.load(rope_block_ptr)
    sin_block = tl.load(tl.advance(rope_block_ptr, (0, BLOCK_DIM // 2)))

    # load halves
    x_half_block_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * x_stride_i + head_idx * x_stride_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(x_stride_s, x_stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM // 2),
        order=(0, 1)
    )

    y_half_block_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * x_stride_i + head_idx * x_stride_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(x_stride_s, x_stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM // 2),
        order=(0, 1)
    )

    x_block_a = tl.load(x_half_block_ptr)
    x_block_b = tl.load(tl.advance(x_half_block_ptr, (0, BLOCK_DIM // 2)))

    new_x_block_a = x_block_a * cos_block - x_block_b * sin_block
    new_x_block_b = x_block_b * cos_block + x_block_a * sin_block

    tl.store(y_half_block_ptr, new_x_block_a)
    tl.store(tl.advance(y_half_block_ptr, (0, BLOCK_DIM // 2)), new_x_block_b)


@triton.jit
def flip_kernel(x_ptr,  # (I, H, S, D)
                k_stride_i, k_stride_h, k_stride_s, k_stride_d,
                BLOCK_SIZE: tl.constexpr,
                BLOCK_DIM: tl.constexpr,
                ):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # load halves
    x_half_block_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * k_stride_i + head_idx * k_stride_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(k_stride_s, k_stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM // 2),
        order=(0, 1)
    )

    y_half_block_ptr = tl.make_block_ptr(
        base=x_ptr + batch_idx * k_stride_i + head_idx * k_stride_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(k_stride_s, k_stride_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM // 2),
        order=(0, 1)
    )

    x_block_a = tl.load(x_half_block_ptr)
    x_block_b = tl.load(tl.advance(x_half_block_ptr, (0, BLOCK_DIM // 2)))

    tl.store(y_half_block_ptr, x_block_b)
    tl.store(tl.advance(y_half_block_ptr, (0, (BLOCK_DIM // 2))), x_block_a)


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
        y_ptr,  # unreduced y float(I1, H, S, D)
        attn_ptr,  # attn stats float(I1, H, 2S)
        grp_lut_ptr,  # reduce group lookup table. all ys in the same group are reduced. int(I2, NUM_BLOCKS_IN_GROUP)
        y_reduced_ptr,  # reduced y float(I2, H, S, D)
        stride_y_i, stride_y_h, stride_y_s, stride_y_d,
        stride_attn_i, stride_attn_h, stride_attn_2s,
        stride_y_reduced_i, stride_y_reduced_h, stride_y_reduced_s, stride_y_reduced_d,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        NUM_BLOCKS_IN_GROUP: tl.constexpr,
):
    grp_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    y_reduced = tl.zeros([BLOCK_SIZE, BLOCK_DIM], dtype=tl.float32)
    attn_max_reduced = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    attn_sum_reduced = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for i in range(NUM_BLOCKS_IN_GROUP):
        # get block index
        block_idx = tl.load(grp_lut_ptr + grp_idx * NUM_BLOCKS_IN_GROUP + i)

        # skip if block_idx is -1
        if block_idx >= 0:

            # get block ptr
            y_block_ptr = tl.make_block_ptr(
                base=y_ptr + block_idx * stride_y_i + head_idx * stride_y_h,
                shape=(BLOCK_SIZE, BLOCK_DIM),
                strides=(stride_y_s, stride_y_d),
                offsets=(0, 0),
                block_shape=(BLOCK_SIZE, BLOCK_DIM),
                order=(0, 1)
            )

            attn_block_ptr = attn_ptr + block_idx * stride_attn_i + head_idx * stride_attn_h

            if i == 0:
                y_reduced = tl.load(y_block_ptr).to(tl.float32)
                attn_max_reduced = tl.load(attn_block_ptr + tl.arange(0, BLOCK_SIZE))
                attn_sum_reduced = tl.load(attn_block_ptr + tl.arange(BLOCK_SIZE, BLOCK_SIZE * 2))

            else:

                y = tl.load(y_block_ptr)
                attn_max = tl.load(attn_block_ptr + tl.arange(0, BLOCK_SIZE))
                attn_sum = tl.load(attn_block_ptr + tl.arange(BLOCK_SIZE, BLOCK_SIZE * 2))

                # re-normalize the softmax
                attn_max_diff = tl.maximum(attn_max_reduced, attn_max)
                alpha = tl.math.exp2(attn_max_reduced - attn_max_diff)
                beta = tl.math.exp2(attn_max - attn_max_diff)

                y_reduced = y_reduced * alpha[:, None] + y * beta[:, None]
                attn_sum_reduced = attn_sum_reduced * alpha + attn_sum * beta
                attn_max_reduced = attn_max_diff

    y_reduced = y_reduced / attn_sum_reduced[:, None]

    y_reduced_block_ptr = tl.make_block_ptr(
        base=y_reduced_ptr + grp_idx * stride_y_reduced_i + head_idx * stride_y_reduced_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_y_reduced_s, stride_y_reduced_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    tl.store(y_reduced_block_ptr, y_reduced.to(y_ptr.dtype.element_ty))


def flip(x: torch.Tensor):
    batch_size, num_head, block_size, block_dim = x.shape

    grid = (batch_size, num_head)

    flip_kernel[grid](
        x,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        block_size, block_dim
    )

    return x


def flip_baseline(x: torch.Tensor):
    _, num_head, block_size, block_dim = x.shape
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    x = torch.cat((x2, x1), dim=-1)
    return x


def rope(
        rope_cache: torch.Tensor,
        x: torch.Tensor,
        pos: torch.Tensor | list[int],
):
    if isinstance(pos, list):
        pos = torch.tensor(pos, dtype=torch.int32, device=x.device)

    max_seq_len, block_dim = rope_cache.shape
    batch_size, num_head, block_size, block_dim_ = x.shape
    batch_size_, = pos.shape

    assert block_dim == block_dim_
    assert batch_size == batch_size_

    grid = (batch_size, num_head)

    rope_kernel[grid](
        rope_cache,
        x,
        pos,
        rope_cache.stride(0), rope_cache.stride(1),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        block_size, block_dim, max_seq_len
    )

    return x


# pytorch-only inefficient implementation of rope for testing purposes
def rope_baseline(
        rope_cache: torch.Tensor,
        x: torch.Tensor,
        pos: torch.Tensor | list[int],
):
    rope_cache = rope_cache.view(-1, 2, rope_cache.size(-1) // 2)
    cos_cache, sin_cache = rope_cache[:, 0, :], rope_cache[:, 1, :]
    cos_cache = cos_cache.squeeze(1)
    sin_cache = sin_cache.squeeze(1)
    cos_cache = torch.cat([cos_cache, cos_cache], dim=-1)
    sin_cache = torch.cat([sin_cache, sin_cache], dim=-1)

    pp = pos.unsqueeze(1) + torch.arange(0, x.size(-2), device=x.device).unsqueeze(0)
    cos_cache = cos_cache[pp].unsqueeze(1)
    sin_cache = sin_cache[pp].unsqueeze(1)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)

    y = (x * cos_cache) + (rotated_x * sin_cache)

    return y


def rope_baseline_no_cache(
        x: torch.Tensor,
        pos: torch.Tensor | list[int],
):
    bsz, num_head, block_size, head_dim = x.shape

    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))

    t = torch.arange(8000, device=x.device, dtype=inv_freq.dtype)

    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos_cache = emb.cos()
    sin_cache = emb.sin()

    pp = pos.unsqueeze(1) + torch.arange(0, block_size, device=x.device).unsqueeze(0)

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    cos_cache = cos_cache[pp].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin_cache = sin_cache[pp].unsqueeze(1)  # [bs, 1, seq_len, dim]
    y = (x * cos_cache) + (x_rotated * sin_cache)

    return y


def create_rope_cache(max_seq_len: int, block_dim: int, dtype: torch.dtype, device: torch.device, base: int = 50000) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, block_dim, 2, device=device) / block_dim))

    # (seq_len, )
    seq_idx = torch.arange(max_seq_len, device=device)

    # (seq_len, block_dim/2)
    idx_theta = torch.outer(seq_idx, theta)

    # (seq_len, block_dim/2)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    return torch.cat((cos, sin), dim=-1).to(dtype)


# previousely copy_kv_block
def fill_kv_block_storage(
        dst_kv: torch.Tensor,
        src_k: torch.Tensor,
        src_v: torch.Tensor,
        dst_lut: torch.Tensor,
):
    #print(src_k.shape)
    #print(dst_kv.shape)
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

        # <LEGEND>
        # NUM_REQS: Number of requests (fill cmd) in the batch.
        # NUM_ROWS (N): Number of rows in the batch. Why NUM_REQ != NUM_ROWS? Because each request can have multiple rows.
        # NUM_HEAD (H): Number of heads in the model. (config.num_attention_heads)
        # BLOCK_SIZE: Number of tokens in a block. (config.block_size)
        # BLOCK_DIM (D): The hidden dimension of the model (head_dim)
        # NUM_BLOCKS_PER_ROW: CHUNK_SIZE * BLOCK_SIZE => Number of blocks in a row. (config.chunk_size)
        # NUM_TOTAL_BLOCKS_IN_STORAGE (CAP): Total number of blocks in the storage.

        # Q tensor (a single block)
        q_ptr,  # q float(NUM_REQS, H, BLOCK_SIZE, D)

        # KV storage
        kv_ptr,  # kv float(CAP, H, 2 * BLOCK_SIZE, D)

        # Output tensor
        y_ptr,  # y float(N, H, BLOCK_SIZE, D)

        # Attention stats tensor (required for reducing y)
        attn_ptr,  # attn stats float(N, H, 2*BLOCK_SIZE)

        # Lookup tables to map N -> NUM_REQS (i.e., all entries are < NUM_REQS)
        q_lut_ptr,  # q lookup int(N, 1)

        # Lookup tables to map N -> list of block ids in a  (i.e., all entries are < NUM_TOTAL_BLOCKS_IN_STORAGE)
        kv_lut_ptr,  # kv lookup int(N, NUM_BLOCKS_PER_ROW)

        # Attention mask.
        mask_ptr,  # mask lookup bool(N, BLOCK_SIZE, NUM_BLOCKS_PER_ROW * BLOCK_SIZE)

        # Strides to access the data in the tensors
        stride_q_i, stride_q_h, stride_q_s, stride_q_d,
        stride_kv_i, stride_kv_h, stride_kv_2s, stride_kv_d,
        stride_mask_i, stride_mask_j, stride_mask_k,
        stride_y_i, stride_y_h, stride_y_s, stride_y_d,
        stride_attn_i, stride_attn_h, stride_attn_2s,
        normalize_at_the_end,

        # Scaling factor for the softmax
        sm_scale: tl.constexpr,

        # Number of tokens in a block
        BLOCK_SIZE: tl.constexpr,

        # The hidden dimension of the model (head_dim)
        BLOCK_DIM: tl.constexpr,

        # This value is referred to as "chunk_size" in the driver code
        NUM_BLOCKS_PER_ROW: tl.constexpr,

        # Needed for Grouped-Query Attention (https://arxiv.org/pdf/2305.13245)
        # If NUM_HEAD_GROUPS is 1, then this is a standard multi-head attention.
        NUM_HEAD_GROUPS: tl.constexpr = 1  # 1 means no GQA.
):
    batch_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    head_kv_idx = head_q_idx // NUM_HEAD_GROUPS

    # read q_idx from q_lut
    q_idx = tl.load(q_lut_ptr + batch_idx)

    # shape (BLOCK_SIZE, D)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + q_idx * stride_q_i + head_q_idx * stride_q_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_q_s, stride_q_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    # shape (BLOCK_SIZE, BLOCK_SIZE)
    mask_block_ptr = tl.make_block_ptr(
        base=mask_ptr + batch_idx * stride_mask_i,
        shape=(BLOCK_SIZE, BLOCK_SIZE),
        strides=(stride_mask_j, stride_mask_k),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_SIZE),
        order=(0, 1)
    )

    y_block_ptr = tl.make_block_ptr(
        base=y_ptr + batch_idx * stride_y_i + head_q_idx * stride_y_h,
        shape=(BLOCK_SIZE, BLOCK_DIM),
        strides=(stride_y_s, stride_y_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE, BLOCK_DIM),
        order=(0, 1)
    )

    attn_block_ptr = attn_ptr + batch_idx * stride_attn_i + head_q_idx * stride_attn_h

    qk_scale = sm_scale * 1.44269504
    q_block = tl.load(q_block_ptr)
    q_block = (q_block * qk_scale).to(kv_ptr.dtype.element_ty)

    attn_max_reduced = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - 1000000.0  # float("inf")
    attn_sum_reduced = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    y_block = tl.zeros([BLOCK_SIZE, BLOCK_DIM], dtype=tl.float32)

    for block_idx in range(NUM_BLOCKS_PER_ROW):
        kv_idx = tl.load(kv_lut_ptr + batch_idx * NUM_BLOCKS_PER_ROW + block_idx)

        k_block_ptr = tl.make_block_ptr(
            base=kv_ptr + kv_idx * stride_kv_i + head_kv_idx * stride_kv_h,
            shape=(BLOCK_DIM, BLOCK_SIZE),
            strides=(stride_kv_d, stride_kv_2s),
            offsets=(0, 0),
            block_shape=(BLOCK_DIM, BLOCK_SIZE),
            order=(1, 0)
        )

        v_block_ptr = tl.make_block_ptr(
            base=kv_ptr + kv_idx * stride_kv_i + head_kv_idx * stride_kv_h,
            shape=(BLOCK_SIZE, BLOCK_DIM),
            strides=(stride_kv_2s, stride_kv_d),
            offsets=(BLOCK_SIZE, 0),
            block_shape=(BLOCK_SIZE, BLOCK_DIM),
            order=(0, 1)
        )

        k_block = tl.load(k_block_ptr)
        v_block = tl.load(v_block_ptr)
        mask = tl.load(mask_block_ptr)

        attn = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
        attn += tl.dot(q_block, k_block, allow_tf32=True)

        # Looks like Triton uses int8 for torch bool types. I added cast to satisfy triton compiler.
        # Check if this cast is causing any performance issues.
        attn += tl.where(mask.cast(tl.int1), 0.0, -1000000.0)

        attn_max = tl.maximum(attn_max_reduced, tl.max(attn, axis=1))
        alpha = tl.math.exp2(attn_max_reduced - attn_max)
        attn = tl.math.exp2(attn - attn_max[:, None])

        y_block = alpha[:, None] * y_block + tl.dot(attn.to(kv_ptr.dtype.element_ty), v_block, allow_tf32=True)

        attn_sum_reduced = attn_sum_reduced * alpha + tl.sum(attn, 1)
        attn_max_reduced = attn_max

        # advance the mask ptr
        mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_SIZE))

    if normalize_at_the_end:
        y_block = y_block / attn_sum_reduced[:, None]
    tl.store(y_block_ptr, y_block.to(y_ptr.dtype.element_ty))

    tl.store(attn_block_ptr + tl.arange(0, BLOCK_SIZE), attn_max_reduced)
    tl.store(attn_block_ptr + tl.arange(BLOCK_SIZE, BLOCK_SIZE * 2), attn_sum_reduced)


def qkv_attention(
        q: torch.Tensor,
        kv: torch.Tensor,
        q_lut: torch.Tensor,
        kv_lut: torch.Tensor,
        mask: torch.Tensor,
        reduce_grp: torch.Tensor,

) -> torch.Tensor:
    num_reqs, num_head, block_size, block_dim = q.shape
    _, num_head_kv, block_size2, block_dim_ = kv.shape

    assert block_size * 2 == block_size2
    assert block_dim == block_dim_

    num_rows = q_lut.shape[0]
    max_grp_size = reduce_grp.shape[1]
    num_blocks_per_row = kv_lut.shape[1]

    #print(q.shape)
    #print(kv.shape)

    # print(mask)

    # print('num_reqs', num_reqs)
    # print('num_head', num_head)
    # print('block_size', block_size)
    # print('block_dim', block_dim)
    # print('num_rows', num_rows)
    # print('max_grp_size', max_grp_size)
    # print('num_blocks_per_row', num_blocks_per_row)

    # check if it is a GQA
    num_gqa_groups = num_head // num_head_kv

    grid = (num_rows, num_head)

    y = torch.empty((num_rows, num_head, block_size, block_dim), device=q.device, dtype=q.dtype)
    attn_stats = torch.zeros((num_rows, num_head, 2 * block_size), device=q.device, dtype=torch.float32)

    # check if we need to reduce y. If not, we do a normalization in this kernel.
    need_reduce = max_grp_size > 1

    qkv_attention_kernel[grid](
        q,
        kv,
        y,
        attn_stats,
        q_lut,
        kv_lut,
        mask,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        attn_stats.stride(0), attn_stats.stride(1), attn_stats.stride(2),
        not need_reduce,
        1.0 / (block_dim ** 0.5),
        block_size, block_dim, num_blocks_per_row, num_gqa_groups
    )

    # reduce y if necessary
    if need_reduce:
        #print('reducing y')
        y_reduced = q  # torch.empty_like(q)

        grid = (num_reqs, num_head)

        reduce_y_slices_kernel[grid](
            y,
            attn_stats,
            reduce_grp,
            y_reduced,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            attn_stats.stride(0), attn_stats.stride(1), attn_stats.stride(2),
            y_reduced.stride(0), y_reduced.stride(1), y_reduced.stride(2), y_reduced.stride(3),
            block_size, block_dim, max_grp_size
        )

        y = y_reduced

    return y


def qkv_attention_baseline(
        q: torch.Tensor,
        kv: torch.Tensor,
        q_lut: torch.Tensor,
        kv_lut: torch.Tensor,
        mask: torch.Tensor,

) -> torch.Tensor:
    num_rows, num_head, block_size, block_dim = q.shape
    _, num_head_kv, block_size2, block_dim_ = kv.shape

    assert block_size * 2 == block_size2
    assert block_dim == block_dim_

    q = q[q_lut].squeeze(1)
    kv = kv[kv_lut]  # (batch_size, num_blocks_per_batch, num_head_kv, 2*block_size,  block_dim)

    k = kv[:, :, :, :block_size, :]
    v = kv[:, :, :, block_size:, :]

    num_blocks_per_row = kv_lut.shape[1]

    # mask = mask[mask_lut].view(num_rows, 1, num_blocks_per_batch * block_size, block_size).transpose(-1, -2)

    k = k.transpose(1, 2).reshape(num_rows, num_head_kv, num_blocks_per_row * block_size, block_dim)
    v = v.transpose(1, 2).reshape(num_rows, num_head_kv, num_blocks_per_row * block_size, block_dim)

    # GQA
    k = torch.repeat_interleave(k, dim=1, repeats=num_head // num_head_kv)
    v = torch.repeat_interleave(v, dim=1, repeats=num_head // num_head_kv)

    attn = torch.einsum('nhqd,nhkd->nhqk', q, k) / (block_dim ** 0.5)

    m = torch.where(mask, 0.0, torch.finfo(attn.dtype).min).to(attn.dtype).unsqueeze(1)

    attn = attn + m

    attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(attn.dtype)

    y = torch.einsum('nhqk,nhkd->nhqd', attn, v)

    return y


def construct_input(reqs: list[tuple[list[int], np.ndarray]], chunk_size: int, block_size: int, device: torch.device) -> dict:
    # mask (num_reqs, num_blocks_per_row, block_size)
    num_chunks_per_req = [ceil_div(len(req[0]), chunk_size) for req in reqs]

    q_lut = np.zeros((sum(num_chunks_per_req), 1), dtype=np.int32)
    kv_lut = np.zeros((sum(num_chunks_per_req), chunk_size), dtype=np.int32)
    reduce_grps = np.zeros((len(reqs), max(num_chunks_per_req)), dtype=np.int32)  # (num_reqs, max num_blocks_per_row)
    masks = np.zeros((sum(num_chunks_per_req), block_size, chunk_size * block_size), dtype=np.bool_)
    # print(reduce_grps)
    k = 0
    for i, req in enumerate(reqs):

        ctx_ids, mask = req
        # mask: (len(ctx_ids) * block_size, block_size)

        num_chunks = ceil_div(len(ctx_ids), chunk_size)

        for j in range(num_chunks):
            start = j * chunk_size
            end = min(start + chunk_size, len(ctx_ids))

            q_lut[k] = i
            kv_lut[k, :end - start] = ctx_ids[start:end]
            masks[k, :, :(end - start) * block_size] = mask[:, start * block_size: end * block_size]

            # if all items in the chunk are False, then it will cause NaN in softmax. Check:
            if not masks[k].any():
                raise ValueError('All items in the chunk are False. This will cause NaN in softmax.')

            reduce_grps[i, j] = k

            k += 1

    return {
        'q_lut': torch.as_tensor(q_lut, dtype=torch.long, device=device),
        'kv_lut': torch.as_tensor(kv_lut, dtype=torch.long, device=device),
        'reduce_grps': torch.as_tensor(reduce_grps, dtype=torch.long, device=device),
        'masks': torch.as_tensor(masks, dtype=torch.bool, device=device)
    }


def construct_input_baseline(reqs: list[tuple[list[int], np.ndarray]], block_size: int, device: torch.device) -> dict:
    # just pad them to the same size

    max_num_blocks = max(len(req[0]) for req in reqs)

    kv_lut = np.zeros((len(reqs), max_num_blocks), dtype=np.int32)
    q_lut = np.zeros((len(reqs), 1), dtype=np.int32)
    masks = np.zeros((len(reqs), block_size, max_num_blocks * block_size), dtype=np.bool_)

    for i, req in enumerate(reqs):
        ctx_ids, mask = req
        q_lut[i] = i
        kv_lut[i, :len(ctx_ids)] = ctx_ids
        masks[i, :, :len(ctx_ids) * block_size] = mask

    return {
        'q_lut': torch.as_tensor(q_lut, device=device),
        'kv_lut': torch.as_tensor(kv_lut, device=device),
        'mask': torch.as_tensor(masks, device=device)
    }


@torch.inference_mode()
def test_qkv_attention():
    device = torch.device('cuda')

    #### CREATE A DUMMY MODEL ####

    # create a dummy model
    num_heads = 9
    head_dim = 32
    hidden_size = head_dim * num_heads
    num_key_value_heads = num_heads // 1

    # create a rope cache
    # rope_cache = create_rope_cache(8192, head_dim, torch.float32, device)

    q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False, device=device)
    # k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device)
    # v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False, device=device)

    ###############################

    #### CREATE A DUMMY KV CACHE ####
    NUM_TOTAL_BLOCKS = 128
    BLOCK_SIZE = 32

    # create a dummy kv cache
    kv_cache_table = torch.randn(NUM_TOTAL_BLOCKS, num_key_value_heads, BLOCK_SIZE * 2, head_dim, device=device)

    ###############################

    #### CREATE A DUMMY TASK BATCH ####
    CHUNK_SIZE = 3
    NUM_REQS = 5
    tasks = []

    for _ in range(NUM_REQS):
        # pick random number between 1 and 10
        num_blocks = 10 #random.randint(1, 8)

        # select `num_total_blocks` amount of random block ids (does not need to be unique)
        ctx_ids = random.choices(range(NUM_TOTAL_BLOCKS), k=num_blocks)

        # create a random mask (num_total_blocks * block_size, block_size) using numpy
        mask = np.random.choice([0, 1], size=(BLOCK_SIZE, num_blocks * BLOCK_SIZE), p=[0.1, 0.9])

        # ensure the first block is always True
        mask[:, 0] = 1

        # create a full true mask
        # mask = np.ones((BLOCK_SIZE, num_blocks * BLOCK_SIZE), dtype=np.bool_)

        tasks.append((ctx_ids, mask))

    inp_baseline = construct_input_baseline(tasks, block_size=BLOCK_SIZE, device=device)
    inp = construct_input(tasks, chunk_size=CHUNK_SIZE, block_size=BLOCK_SIZE, device=device)

    # simulate the previous state
    hidden_states = torch.randn(NUM_REQS, BLOCK_SIZE, hidden_size, device=device)

    q = q_proj(hidden_states)
    # k = k_proj(hidden_states)
    # v = v_proj(hidden_states)

    q = q.view(NUM_REQS, BLOCK_SIZE, num_heads, head_dim).transpose(1, 2)
    # k = k.view(NUM_REQS, BLOCK_SIZE, num_key_value_heads, head_dim).transpose(1, 2)
    # v = v.view(NUM_REQS, BLOCK_SIZE, num_key_value_heads, head_dim).transpose(1, 2)

    # rope(rope_cache, q, batch.position_offsets)
    # rope(rope_cache, k, batch.position_offsets)

    # attention
    y1 = qkv_attention_baseline(
        q,
        kv_cache_table,
        inp_baseline['q_lut'],
        inp_baseline['kv_lut'],
        inp_baseline['mask']
    )

    # print(y1[0])

    y2 = qkv_attention(
        q,
        kv_cache_table,
        inp['q_lut'],
        inp['kv_lut'],
        inp['masks'],
        inp['reduce_grps']
    )

    # print(y2[0])
    # print('baseline:')
    # print(y1.unsqueeze(0).unsqueeze(0))
    #
    # print('triton:')
    # print(y2.unsqueeze(0).unsqueeze(0))

    # shape
    print('y1, y2', y1.shape, y2.shape)

    print('y1, y2', torch.abs(y1 - y2).sum())

    print('done')

    ...


def test_rope():
    head_dim = 64
    batch_size = 100
    device = 'cuda'

    k = torch.randn((batch_size, 32, 8, head_dim), device=device)
    position_offsets = torch.tensor(list(range(batch_size)), dtype=torch.int32, device=device)
    rope_cache = create_rope_cache(8192, head_dim, torch.float32, device)

    k_pos1 = rope_baseline(rope_cache, k, position_offsets)
    k_pos2 = rope_baseline_no_cache(k, position_offsets)

    k_pos_triton = rope(rope_cache, k, position_offsets)

    # print the difference
    print('kpos1, kpos2', torch.abs(k_pos1 - k_pos2).sum())
    assert torch.allclose(k_pos2, k_pos1, atol=1e-3)

    print('kpos1, kpostriton', torch.abs(k_pos1 - k_pos_triton).sum())
    assert torch.allclose(k_pos1, k_pos_triton, atol=1e-3)
    # assert torch.allclose(v_pos1, v)


if __name__ == '__main__':
    # test_sliced_attention()
    # test_rope()
    test_qkv_attention()
