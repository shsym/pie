"""KV cache rebinding for the SGLang backend.

SGLang allocates `MHATokenToKVPool.k_buffer[layer]` and `.v_buffer[layer]`
during ModelRunner construction — each shape `(num_blocks*page_size, h, d)`,
flat token-slot indexed.

Pie's swap RPCs in `pie_backend.worker._handle_copy_*` operate on
`engine.kv_cache_at_layer[layer]` indexed by *block ID*, expecting each
layer's tensor to look like `(num_blocks, 2, page_size, h, d)`.

To bridge: allocate one `(2, num_blocks, page_size, h, d)` tensor per layer
ourselves, replace SGLang's `k_buffer[i]` / `v_buffer[i]` with contiguous
views into dim 0, and expose a permuted `(num_blocks, 2, page_size, h, d)`
view for pie. The permuted view is non-contiguous but PyTorch's advanced
indexing — used by pie's swap handlers — reads it correctly.

For v1 we skip the host pool: pie's vllm sibling does the same since the
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
) -> tuple[list[torch.Tensor], int]:
    """Replace sglang's k/v buffers with views into pie-shaped storage.

    Returns `(kv_cache_at_layer, num_blocks)`. Page size is left to the
    caller to read from `loaded.runner.page_size`.
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
    if head_dim != int(pool.v_head_dim):
        raise NotImplementedError(
            f"pie_backend_sglang v1 requires k_head_dim == v_head_dim "
            f"(got {head_dim} vs {pool.v_head_dim})."
        )

    # SGLang allocates `size + page_size` slots; the trailing page is a
    # sentinel for "padded tokens" in some kernels. We round down to a
    # whole number of pages and trust sglang to leave the sentinel alone.
    total_token_slots = pool.k_buffer[0].shape[0]
    num_blocks = total_token_slots // page_size

    num_layers = len(pool.k_buffer)
    dtype = pool.k_buffer[0].dtype
    device = pool.k_buffer[0].device

    # Free sglang's pre-allocated buffers up front. Without this we hold
    # both sglang's `(size+page_size, h, d)` tensors AND our parallel
    # `(2, num_blocks, page_size, h, d)` tensors during rebind, doubling
    # KV memory and OOMing under tight `mem_fraction_static`.
    pool.k_buffer = [None] * num_layers
    pool.v_buffer = [None] * num_layers
    torch.cuda.empty_cache()

    kv_cache_at_layer: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        # Layout: K block then V block per layer. `tensor[0]` is K and
        # `tensor[1]` is V — both contiguous of shape (num_blocks, page_size, h, d).
        layer_tensor = torch.zeros(
            (2, num_blocks, page_size, head_num, head_dim),
            dtype=dtype, device=device,
        )
        flat = (num_blocks * page_size, head_num, head_dim)
        pool.k_buffer[layer_idx] = layer_tensor[0].reshape(flat)
        pool.v_buffer[layer_idx] = layer_tensor[1].reshape(flat)
        # Pie's view: (num_blocks, 2, page_size, h, d), block-indexed.
        kv_cache_at_layer.append(layer_tensor.permute(1, 0, 2, 3, 4))

    # Refresh `data_ptrs` if it exists — only consulted by sglang's optional
    # `enable_kv_cache_copy` Triton kernel (we leave that disabled), but
    # recomputing keeps a future sglang change from breaking us silently.
    if hasattr(pool, "data_ptrs"):
        pool.data_ptrs = torch.tensor(
            [x.data_ptr() for x in pool.k_buffer + pool.v_buffer],
            dtype=torch.int64, device=device,
        )

    return kv_cache_at_layer, num_blocks
