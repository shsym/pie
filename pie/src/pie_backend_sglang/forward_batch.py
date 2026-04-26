"""Translate pie's CSR-form batch metadata into an SGLang `ForwardBatch`.

Pie emits per-batch metadata in CSR form:
  - `qo_indptr[r..r+1]`     → query token range for request r
  - `kv_page_indices`       → flat list of block IDs across all requests
  - `kv_page_indptr[r..r+1]`→ block range for request r
  - `kv_last_page_lens[r]`  → valid token count in request r's last block

SGLang's `ForwardBatch` (and the attention backends downstream of it) consume:
  - `req_pool_indices: (batch,)`  — slot in `req_to_token_pool` for each request
  - `seq_lens: (batch,)`           — total sequence length per request
  - `extend_seq_lens, extend_prefix_lens` — split for prefill
  - `out_cache_loc: (num_query_tokens,)` — flat token slot for each query token
  - `req_to_token_pool.req_to_token[req_pool_idx, :seq_len]` — slot per token

We bypass SGLang's `req_to_token_pool.alloc()` and write the table directly:
for each request, expand its CSR block list into per-token slots
`block_id * page_size + offset`, where `block_id` is owned by pie's Rust
scheduler and `offset` is the within-block position.

`out_cache_loc` similarly maps each query token to its destination slot. The
shape matches pie's `slot_mapping` in the vllm path.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np
import torch


@numba.njit(cache=True, parallel=False)
def _build_req_to_token_rows(
    kv_page_indices: np.ndarray,    # int32 (total_pages,)
    kv_page_indptr: np.ndarray,     # int32 (batch+1,)
    seq_lens: np.ndarray,           # int32 (batch,)
    page_size: int,
    out: np.ndarray,                # int32 (batch, max_seq_len)
):
    """Expand CSR block IDs into a `(batch, max_seq_len)` per-token slot table.

    Each row r of `out` is filled with `seq_lens[r]` valid slot indices; the
    rest is left as zeros (SGLang's kernels guard via `seq_lens`).
    """
    batch = kv_page_indptr.shape[0] - 1
    for r in range(batch):
        s_r = seq_lens[r]
        page_base = kv_page_indptr[r]
        for i in range(s_r):
            page_idx = i // page_size
            offset = i % page_size
            block_id = kv_page_indices[page_base + page_idx]
            out[r, i] = block_id * page_size + offset


@numba.njit(cache=True, parallel=False)
def _build_out_cache_loc(
    qo_indptr: np.ndarray,           # int32 (batch+1,)
    kv_page_indices: np.ndarray,     # int32 (total_pages,)
    kv_page_indptr: np.ndarray,      # int32 (batch+1,)
    seq_lens: np.ndarray,            # int32 (batch,)
    page_size: int,
    out: np.ndarray,                 # int64 (num_query_tokens,)
):
    """For each query token, its destination flat slot in the KV pool.

    Mirrors `attn_metadata._build_slot_mapping` in the vllm backend.
    """
    batch = qo_indptr.shape[0] - 1
    for r in range(batch):
        q_start = qo_indptr[r]
        q_end = qo_indptr[r + 1]
        q_len = q_end - q_start
        s_r = seq_lens[r]
        page_base = kv_page_indptr[r]
        first_pos = s_r - q_len
        for i in range(q_len):
            p = first_pos + i
            page_idx = p // page_size
            offset = p % page_size
            block_id = kv_page_indices[page_base + page_idx]
            out[q_start + i] = block_id * page_size + offset


def build_sglang_forward_batch(
    *,
    runner: Any,                     # sglang.srt.model_executor.model_runner.ModelRunner
    inputs: dict,
    page_size: int,
    device: torch.device,
):
    """Build a `ForwardBatch` from pie's `inputs` dict.

    Returns the populated ForwardBatch object ready for `runner.forward(...)`.
    """
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        ForwardMode,
    )

    # ---- Pull pie's CSR metadata to numpy (cheap, already on CPU as int32) ----

    qo_indptr_np = inputs["qo_indptr"].cpu().to(torch.int32).numpy()
    kv_idx_np = inputs["kv_page_indices"].cpu().to(torch.int32).numpy()
    kv_indptr_np = inputs["kv_page_indptr"].cpu().to(torch.int32).numpy()
    kv_last_np = inputs["kv_last_page_lens"].cpu().to(torch.int32).numpy()

    batch_size = qo_indptr_np.shape[0] - 1
    num_query_tokens = int(qo_indptr_np[-1])

    # Per-request total seq_len = (num_pages - 1) * page_size + last_page_len
    pages_per_req = (kv_indptr_np[1:] - kv_indptr_np[:-1]).astype(np.int32)
    seq_lens_np = ((pages_per_req - 1) * page_size + kv_last_np).astype(np.int32)
    extend_seq_lens_np = (qo_indptr_np[1:] - qo_indptr_np[:-1]).astype(np.int32)
    extend_prefix_lens_np = (seq_lens_np - extend_seq_lens_np).astype(np.int32)
    extend_num_tokens = int(extend_seq_lens_np.sum())

    # ---- Allocate / write req_to_token_pool.req_to_token rows ----

    # We claim the first `batch_size` slots. SGLang's pool initializes
    # free_slots = list(range(size)); we ignore that bookkeeping and just
    # write rows directly. Pie's runtime owns block IDs; sglang's pool only
    # owns the lookup table.
    req_pool_indices_np = np.arange(batch_size, dtype=np.int32)
    req_to_token = runner.req_to_token_pool.req_to_token

    if batch_size > 0:
        max_seq_len = int(seq_lens_np.max())
        rows_np = np.zeros((batch_size, max_seq_len), dtype=np.int32)
        _build_req_to_token_rows(
            kv_idx_np, kv_indptr_np, seq_lens_np, page_size, rows_np
        )
        rows = torch.from_numpy(rows_np).to(device, non_blocking=True)
        # Write into the pool's table at our chosen slots.
        req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(device)
        # `req_to_token[req_pool_indices_t, :max_seq_len] = rows`
        req_to_token[req_pool_indices_t, :max_seq_len] = rows
    else:
        max_seq_len = 0

    # ---- out_cache_loc (per query token destination slot) ----

    if num_query_tokens > 0:
        out_cache_np = np.zeros(num_query_tokens, dtype=np.int64)
        _build_out_cache_loc(
            qo_indptr_np, kv_idx_np, kv_indptr_np, seq_lens_np,
            page_size, out_cache_np,
        )
        out_cache_loc = torch.from_numpy(out_cache_np).to(device, non_blocking=True)
    else:
        out_cache_loc = torch.empty(0, dtype=torch.int64, device=device)

    # ---- Tensors for ForwardBatch ----

    seq_lens_t = torch.from_numpy(seq_lens_np).to(torch.int64).to(device, non_blocking=True)
    seq_lens_cpu_t = torch.from_numpy(seq_lens_np).to(torch.int64)
    req_pool_indices_t = torch.from_numpy(req_pool_indices_np).to(torch.int64).to(device)
    extend_seq_lens_t = torch.from_numpy(extend_seq_lens_np).to(device, non_blocking=True)
    extend_prefix_lens_t = torch.from_numpy(extend_prefix_lens_np).to(device, non_blocking=True)

    # We keep pie's caller-supplied position_ids verbatim (matches vllm path).
    positions = inputs["position_ids"].to(device=device, dtype=torch.int64, non_blocking=True)

    # We always run the EXTEND path. Pie issues prefill+decode through the
    # same code path and SGLang's decode kernel doesn't support custom_mask
    # nor pie's per-token positions cleanly. The EXTEND kernel handles
    # length-1 query tokens fine; per-request `extend_seq_lens=1, prefix_lens=N`.
    forward_mode = ForwardMode.EXTEND

    # Lazily fill ForwardBatch — we bypass init_new() because we're not
    # constructing through ScheduleBatch/ModelWorkerBatch. We populate just
    # what the attention backend needs.
    fb = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=batch_size,
        input_ids=inputs["token_ids"].to(device=device, dtype=torch.int32, non_blocking=True),
        req_pool_indices=req_pool_indices_t,
        seq_lens=seq_lens_t,
        out_cache_loc=out_cache_loc,
        seq_lens_sum=int(seq_lens_np.sum()),
        seq_lens_cpu=seq_lens_cpu_t,
        positions=positions,
        extend_num_tokens=extend_num_tokens,
        extend_seq_lens=extend_seq_lens_t,
        extend_prefix_lens=extend_prefix_lens_t,
        extend_prefix_lens_cpu=extend_prefix_lens_np.tolist(),
        extend_seq_lens_cpu=extend_seq_lens_np.tolist(),
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool=runner.token_to_kv_pool,
        attn_backend=runner.attn_backend,
    )
    return fb
