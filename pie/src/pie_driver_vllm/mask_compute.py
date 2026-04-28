"""Custom-attention-mask compute primitives for `pie_driver_vllm`.

Module split (with `mask_strategies.py`):

  * **mask_compute.py** (this file) — *data + kernels*: `PieAttnExtras`
    construction from pie's BRLE mask, plus the FlashInfer prefill-
    wrapper helpers. No vllm dispatch logic.

  * **mask_strategies.py** — *per-backend routing*: a proxy that wraps
    each `AttentionImpl` and dispatches to a strategy class which calls
    the helpers here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


# ----------------------------------------------------------------------------
# Per-batch extras built once and stashed on the forward context
# ----------------------------------------------------------------------------


@dataclass
class PieAttnExtras:
    """Everything the mask-aware impl needs to apply pie's BRLE mask.

    All tensors live on the compute device; CPU shadows are kept for the
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
        # Match pie's BRLE decoder layout exactly: each query token's bit
        # span is fixed-width = `seq_lens[req(token)]`. The decoder writes
        # True bits into the first `position_ids[k] + 1` cells of that
        # span and leaves the rest zero-padded — but the *offsets* are
        # seq-len-padded. See `pie_driver.batching.Batch.__init__`
        # (token_acc_seq_lens_np).
        qsl_cpu = query_start_loc.detach().cpu()
        seq_lens_cpu = seq_lens.detach().cpu()

        qo_lens = (qsl_cpu[1:] - qsl_cpu[:-1]).to(torch.int64)
        seq_lens_per_token = torch.repeat_interleave(
            seq_lens_cpu.to(torch.int64), qo_lens
        )

        token_acc_cpu = torch.zeros(seq_lens_per_token.numel() + 1, dtype=torch.int32)
        token_acc_cpu[1:] = torch.cumsum(seq_lens_per_token, dim=0).to(torch.int32)

        return PieAttnExtras(
            custom_mask=custom_mask.to(device, dtype=torch.bool, non_blocking=True),
            token_acc_seq_lens_cpu=token_acc_cpu,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            block_table=block_table,
            page_size=int(page_size),
        )


# ----------------------------------------------------------------------------
# FlashInfer prefill wrapper helpers
# ----------------------------------------------------------------------------


def make_flashinfer_wrapper(workspace_bytes: int, device: torch.device) -> Any:
    """Allocate a single FlashInfer prefill wrapper + workspace.

    The workspace is reused across all batches and layers — the wrapper
    just stores plan-state and forwards to its kernels.
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

    Native pie does the same thing — plan once per batch in transform(),
    then each layer just calls run().
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
