"""KV cache rebinding for the SGLang backend.

SGLang allocates `MHATokenToKVPool.k_buffer[layer]` and `.v_buffer[layer]`
during ModelRunner construction. Each is shape `(num_blocks*page_size, h, d)`
— flat token-slot indexed.

Pie's swap RPCs in `pie_backend.worker._handle_copy_d2h/d2d/h2d/h2h` operate
on `engine.kv_cache_at_layer[layer]` indexed by *block ID* — they expect each
layer's tensor to look like `(num_blocks, 2, page_size, h, d)`.

To bridge the two, we allocate one `(2, num_blocks, page_size, h, d)` tensor
per layer ourselves, replace SGLang's `k_buffer[i]` and `v_buffer[i]` with
contiguous views into dim 0 of that tensor, and expose a permuted block-major
view as `kv_cache_at_layer[i]`. The permuted view is non-contiguous but
PyTorch's advanced indexing (used by pie's swap handlers) reads it correctly.

For v1 we skip the host pool: pie's vllm backend does the same, since the
swap handlers' bounds check would index an empty list. Set
`cpu_mem_budget_in_gb = 0` in pie's TOML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pie_backend.config import RuntimeConfig

if TYPE_CHECKING:
    from .loader import LoadedModel


def _rebind_pool_buffers(
    loaded: "LoadedModel",
    config: RuntimeConfig,
) -> tuple[list[torch.Tensor], int, int]:
    """Replace SGLang's per-layer k/v buffers with views into a pie-shaped tensor.

    Returns:
        (kv_cache_at_layer, num_blocks, page_size) — one tensor per layer in
        layer-index order, plus the resolved block count and page size.
    """
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = loaded.runner.token_to_kv_pool
    if not isinstance(pool, MHATokenToKVPool):
        raise NotImplementedError(
            f"pie_backend_sglang v1 only supports MHATokenToKVPool; got "
            f"{type(pool).__name__}. MLA / hybrid models are not yet supported."
        )

    page_size = int(pool.page_size)
    head_num = int(pool.head_num)
    head_dim = int(pool.head_dim)
    v_head_dim = int(pool.v_head_dim)
    if head_dim != v_head_dim:
        raise NotImplementedError(
            f"pie_backend_sglang v1 requires k_head_dim == v_head_dim "
            f"(got {head_dim} vs {v_head_dim})."
        )

    # SGLang allocates `size + page_size` token slots; the trailing `page_size`
    # is a sentinel page (slot 0..page_size-1 are reserved for "padded" tokens
    # in some kernels). We treat (size + page_size) // page_size as the total
    # number of pages our scheduler can hand out, and trust SGLang's kernels
    # to leave the sentinel alone (block 0 is reserved by pie too).
    total_token_slots = pool.k_buffer[0].shape[0]
    num_blocks = total_token_slots // page_size
    if num_blocks * page_size != total_token_slots:
        # Truncate to a whole number of pages. The leftover slots (if any) are
        # unused.
        total_token_slots = num_blocks * page_size

    num_layers = len(pool.k_buffer)
    dtype = pool.k_buffer[0].dtype
    device = pool.k_buffer[0].device

    kv_cache_at_layer: list[torch.Tensor] = []

    # Free sglang's pre-allocated buffers up front. Without this we hold
    # both the original `(size+page_size, h, d)` tensors AND our parallel
    # `(2, num_blocks, page_size, h, d)` tensors for the duration of
    # rebind, doubling KV memory and OOMing under tight `mem_fraction_static`.
    # Replacing the list entries below would only drop refs after the
    # caching allocator has already failed to grow.
    pool.k_buffer = [None] * num_layers
    pool.v_buffer = [None] * num_layers
    torch.cuda.empty_cache()

    for layer_idx in range(num_layers):
        # Allocate the pie-shaped storage. Layout: K block then V block per layer.
        # `tensor[0]` is K (contiguous, shape (num_blocks, page_size, h, d));
        # `tensor[1]` is V.
        layer_tensor = torch.zeros(
            (2, num_blocks, page_size, head_num, head_dim),
            dtype=dtype,
            device=device,
        )

        # Views into the new storage that match SGLang's expected
        # `(num_blocks*page_size, h, d)` flat-token shape.
        k_view = layer_tensor[0].reshape(num_blocks * page_size, head_num, head_dim)
        v_view = layer_tensor[1].reshape(num_blocks * page_size, head_num, head_dim)

        # If sglang's allocated buffer is larger (sentinel page), copy any
        # initialized state across so we don't lose it. In practice everything
        # is freshly zeroed at this point, so we can just replace.
        pool.k_buffer[layer_idx] = k_view
        pool.v_buffer[layer_idx] = v_view

        # Pie's view: (num_blocks, 2, page_size, h, d), block-indexed.
        pie_view = layer_tensor.permute(1, 0, 2, 3, 4)
        kv_cache_at_layer.append(pie_view)

    # Refresh pool's `data_ptrs` / `data_strides` if they exist, since they
    # cache pointers into the old tensors. These are only used by the optional
    # `enable_kv_cache_copy` Triton kernel, which we leave disabled — but rather
    # than rely on that, recompute them so a future SGLang change can't break
    # us silently.
    if hasattr(pool, "data_ptrs"):
        pool.data_ptrs = torch.tensor(
            [x.data_ptr() for x in pool.k_buffer + pool.v_buffer],
            dtype=torch.int64,
            device=device,
        )

    return kv_cache_at_layer, num_blocks, page_size
