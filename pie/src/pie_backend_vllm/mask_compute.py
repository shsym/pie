"""Custom-attention-mask compute primitives for `pie_backend_vllm`.

Module split (with `mask_strategies.py`):

  * **mask_compute.py** (this file) — *data + kernels*: `PieAttnExtras`
    construction from pie's BRLE mask, the universal SDPA gather, and
    the FlashInfer prefill-wrapper helpers. No vllm dispatch logic.

  * **mask_strategies.py** — *per-backend routing*: a proxy that wraps
    each `AttentionImpl` and dispatches to a strategy class which calls
    the helpers here.

KV layouts we recognize:
  FlashAttn V1:                   (2, num_blocks, block_size, num_kv_heads, head_dim)
  FlashInfer V1 (NHD, default):   (num_blocks, 2, block_size, num_kv_heads, head_dim)
Detected via `kv_cache.shape[0] == 2`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Per-batch extras built once and stashed on the forward context
# ----------------------------------------------------------------------------


@dataclass
class PieAttnExtras:
    """Everything the mask-aware impl needs to apply pie's BRLE mask.

    All tensors live on the compute device. CPU shadows are kept for the
    things we index from Python (per-request loop bounds).
    """

    # Flat per-token mask: pie's decoder writes a fixed-width span of
    # `seq_lens[req(token)]` cells per query token, of which only the first
    # `position_ids[token] + 1` cells carry real bits (the rest are zero pad).
    # Length = sum_over_tokens(seq_lens[req(token)]).
    custom_mask: torch.Tensor          # bool, [total_bits]
    # Cumulative bit offset per query token under pie's padded layout.
    # `token_acc_seq_lens[k+1] - token_acc_seq_lens[k] == seq_lens[req(k)]`.
    token_acc_seq_lens_cpu: torch.Tensor  # int32 on cpu

    # Echo of vllm's CommonAttentionMetadata for our local use.
    query_start_loc: torch.Tensor      # int32, [batch + 1]
    query_start_loc_cpu: torch.Tensor  # int32, cpu
    seq_lens: torch.Tensor             # int32, [batch]
    seq_lens_cpu: torch.Tensor         # int32, cpu
    block_table: torch.Tensor          # int32, [batch, max_blocks]
    page_size: int

    # Pre-flattened gather indices for the SDPA fallback. Per request, indexed
    # via `gather_starts_cpu`. Indexing K/V with two int64 tensors avoids the
    # 64 MB-per-layer reshape copy that the original code triggered.
    gather_block_idx: torch.Tensor     # int64, [sum(kv_len_per_req)]
    gather_offset:    torch.Tensor     # int64, same shape
    gather_starts_cpu: torch.Tensor    # int64, [batch+1]; per-request slice into above

    # Optional FlashInfer prefill wrapper, pre-planned ONCE per batch in
    # `forward_pass.transform()`. Only set when the active impl uses the
    # FlashInfer fast path.
    flashinfer_wrapper: Any | None = None

    @staticmethod
    def build(
        *,
        custom_mask: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        page_size: int,
        device: torch.device,
    ) -> "PieAttnExtras":
        # Match pie's BRLE decoder layout exactly: each query token's bit span
        # is fixed-width = `seq_lens[req(token)]`. The decoder writes True bits
        # into the first `position_ids[k] + 1` cells of that span and leaves
        # the rest zero-padded — but the *offsets* are seq-len-padded.
        # See `pie_backend.batching.Batch.__init__` (token_acc_seq_lens_np).
        qsl_cpu = query_start_loc.detach().cpu()
        seq_lens_cpu = seq_lens.detach().cpu()

        qo_lens = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int64)
        seq_lens_per_token = torch.repeat_interleave(seq_lens_cpu.to(torch.int64), qo_lens)

        token_acc_cpu = torch.zeros(seq_lens_per_token.numel() + 1, dtype=torch.int32)
        token_acc_cpu[1:] = torch.cumsum(seq_lens_per_token, dim=0).to(torch.int32)

        # Gather indices for the SDPA fallback. For each request r and each
        # kv position p in [0, seq_lens[r]):
        #   block_idx[g] = block_table[r, p // page]   (int64)
        #   offset[g]    = p % page                     (int64)
        page = int(page_size)
        seq_int = seq_lens_cpu.to(torch.int64)
        gather_starts_cpu = torch.zeros(seq_int.numel() + 1, dtype=torch.int64)
        gather_starts_cpu[1:] = torch.cumsum(seq_int, dim=0)
        total_kv = int(gather_starts_cpu[-1].item())

        if total_kv > 0:
            req_id = torch.repeat_interleave(
                torch.arange(seq_int.numel(), dtype=torch.int64), seq_int
            )
            pos_in_req = torch.arange(total_kv, dtype=torch.int64)
            pos_in_req = pos_in_req - gather_starts_cpu[req_id]
            page_idx = pos_in_req // page
            offset_cpu = pos_in_req % page
            bt_cpu = block_table.detach().cpu().to(torch.int64)
            block_idx_cpu = bt_cpu[req_id, page_idx]
            gather_block_idx = block_idx_cpu.to(device, non_blocking=True)
            gather_offset = offset_cpu.to(device, non_blocking=True)
        else:
            gather_block_idx = torch.empty(0, dtype=torch.int64, device=device)
            gather_offset = torch.empty(0, dtype=torch.int64, device=device)

        return PieAttnExtras(
            custom_mask=custom_mask.to(device, dtype=torch.bool, non_blocking=True),
            token_acc_seq_lens_cpu=token_acc_cpu,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            page_size=page,
            gather_block_idx=gather_block_idx,
            gather_offset=gather_offset,
            gather_starts_cpu=gather_starts_cpu,
        )


# ----------------------------------------------------------------------------
# KV layout helpers
# ----------------------------------------------------------------------------


def is_flashinfer_layout(kv_cache: torch.Tensor) -> bool:
    """FlashInfer NHD: (num_blocks, 2, block_size, num_kv_heads, head_dim)."""
    return kv_cache.dim() == 5 and kv_cache.shape[1] == 2


def is_flashattn_layout(kv_cache: torch.Tensor) -> bool:
    """FlashAttn: (2, num_blocks, block_size, num_kv_heads, head_dim)."""
    return kv_cache.dim() == 5 and kv_cache.shape[0] == 2


def split_kv(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (K, V) views with shape (num_blocks, page, num_kv_heads, head_dim)."""
    if is_flashinfer_layout(kv_cache):
        return kv_cache[:, 0], kv_cache[:, 1]
    if is_flashattn_layout(kv_cache):
        return kv_cache[0], kv_cache[1]
    raise RuntimeError(
        f"pie_backend_vllm: unsupported KV layout {tuple(kv_cache.shape)}. "
        "Only NHD (num_blocks, 2, ...) and FlashAttn (2, num_blocks, ...) "
        "are wired today."
    )


# ----------------------------------------------------------------------------
# Universal SDPA gather path
# ----------------------------------------------------------------------------


def sdpa_gather_path(
    *,
    layer: Any,
    query: torch.Tensor,        # [num_q_tokens, num_heads, head_dim]
    kv_cache: torch.Tensor,
    extras: PieAttnExtras,
    output: torch.Tensor,       # [num_q_tokens, num_heads, head_dim_v], pre-allocated
) -> None:
    """Per-request SDPA on gathered K/V. Writes into `output` in place.

    For each request r: gather K, V from paged storage, slice query rows,
    materialize a [q_len, kv_len] mask via a single `view`, run SDPA with
    `enable_gqa=True`. Works on any HW PyTorch supports.
    """
    # Index `(num_blocks, page, kv_h, hd)` directly with two int64 tensors.
    # Reshaping `kv_cache[:, 0]` to flat would force a 64 MB copy of the
    # whole cache *every layer* — see git history for the diagnosis.
    K_all, V_all = split_kv(kv_cache)

    qsl_cpu = extras.query_start_loc_cpu.tolist()
    seq_lens_cpu = extras.seq_lens_cpu.tolist()
    token_acc_cpu = extras.token_acc_seq_lens_cpu.tolist()
    gather_starts_cpu = extras.gather_starts_cpu.tolist()

    flat_mask = extras.custom_mask
    block_idx_all = extras.gather_block_idx
    offset_all = extras.gather_offset

    num_heads = query.shape[1]
    head_dim_q = query.shape[2]
    num_kv_heads = V_all.shape[2]
    sm_scale = float(getattr(layer.impl, "scale", 1.0 / (head_dim_q ** 0.5)))

    batch = len(seq_lens_cpu)
    for r in range(batch):
        q_start = qsl_cpu[r]
        q_end = qsl_cpu[r + 1]
        q_len = q_end - q_start
        if q_len == 0:
            continue
        kv_len = seq_lens_cpu[r]
        if kv_len == 0:
            output[q_start:q_end].zero_()
            continue

        g_start = gather_starts_cpu[r]
        g_end = gather_starts_cpu[r + 1]
        block_idx = block_idx_all[g_start:g_end]
        offset    = offset_all[g_start:g_end]

        K_r = K_all[block_idx, offset]   # [kv_len, num_kv, hd_k]
        V_r = V_all[block_idx, offset]   # [kv_len, num_kv, hd_v]

        Q_r = query[q_start:q_end].unsqueeze(0).transpose(1, 2).contiguous()
        K_r = K_r.unsqueeze(0).transpose(1, 2).contiguous()
        V_r = V_r.unsqueeze(0).transpose(1, 2).contiguous()

        # Pie's BRLE decoder lays out per-token spans at fixed width = kv_len
        # within a request, so the contiguous slice [token_acc[q_start],
        # token_acc[q_end]) reshapes directly to (q_len, kv_len).
        mask_start = token_acc_cpu[q_start]
        mask_end = token_acc_cpu[q_end]
        sdpa_mask = flat_mask[mask_start:mask_end].view(q_len, kv_len)
        sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)

        try:
            out_r = F.scaled_dot_product_attention(
                Q_r, K_r, V_r, attn_mask=sdpa_mask, enable_gqa=True,
                scale=sm_scale,
            )
        except TypeError:
            rep = num_heads // num_kv_heads
            K_rep = K_r.repeat_interleave(rep, dim=1)
            V_rep = V_r.repeat_interleave(rep, dim=1)
            out_r = F.scaled_dot_product_attention(
                Q_r, K_rep, V_rep, attn_mask=sdpa_mask, scale=sm_scale,
            )
        out_r = out_r.squeeze(0).transpose(0, 1).contiguous()
        output[q_start:q_end].copy_(out_r)


# ----------------------------------------------------------------------------
# FlashInfer prefill wrapper helpers
# ----------------------------------------------------------------------------


def make_flashinfer_wrapper(workspace_bytes: int, device: torch.device) -> Any:
    """Allocate a single FlashInfer prefill wrapper + workspace.

    The workspace is reused across all batches and layers — the wrapper just
    stores plan-state and forwards to its kernels.
    """
    import flashinfer  # type: ignore[import-not-found]

    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    # Keep a reference so the workspace tensor isn't GC'd; FlashInfer holds
    # only an internal pointer.
    wrapper._pie_workspace = workspace
    return wrapper


def plan_flashinfer_wrapper(
    wrapper: Any,
    *,
    extras: PieAttnExtras,
    kv_page_indices: torch.Tensor,    # int32 CSR, flat
    kv_page_indptr: torch.Tensor,     # int32 CSR, [batch+1]
    kv_last_page_lens: torch.Tensor,  # int32, [batch]
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    q_data_type: torch.dtype,
) -> None:
    """Call wrapper.plan() ONCE for this batch with pie's flat custom_mask.

    Native pie does the same thing — plan once per batch in transform(), then
    each layer just calls run().
    """
    wrapper.plan(
        qo_indptr=extras.query_start_loc.to(torch.int32),
        paged_kv_indptr=kv_page_indptr.to(torch.int32),
        paged_kv_indices=kv_page_indices.to(torch.int32),
        paged_kv_last_page_len=kv_last_page_lens.to(torch.int32),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        page_size=extras.page_size,
        custom_mask=extras.custom_mask,
        q_data_type=q_data_type,
    )
