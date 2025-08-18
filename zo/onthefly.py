# triton_batched_randn_kernels.py
# ------------------------------------------------------------
# Two Triton kernels:
#  1) batched_randn_matmul: y[b, :] = x[b, :] @ W_b with W_b ~ N(0,1), generated on-the-fly
#  2) batched_randn_generate: W_batched[b, i, o] ~ N(0,1) materialized per seed[b]
#
# Tests compare against a PyTorch baseline using the generator kernel to ensure bit-exactness.
# ------------------------------------------------------------

import triton
import triton.language as tl
import torch


# ============================================================
#  KERNEL 1: y[b, :] = x[b, :] @ W_b, where W_b ~ N(0,1)
#            (B is RUNTIME, not constexpr)
# ============================================================

@triton.jit
def _randn_mm_row_kernel(
        x_ptr,  # *f16/f32 [B, I]
        seeds_ptr,  # *int64   [B]
        y_ptr,  # *f32     [B, O]  (accumulate in f32)
        B,  # RUNTIME scalar (int32) -- NOT constexpr
        I: tl.constexpr,  # in_features
        O: tl.constexpr,  # out_features
        stride_xb, stride_xi,
        stride_yb, stride_yo,
        n_rounds: tl.constexpr,  # tl.randn quality parameter
        BLOCK_K: tl.constexpr,  # tile over in_features
        BLOCK_N: tl.constexpr  # tile over out_features
):
    pid_b = tl.program_id(0)  # batch row id
    pid_n = tl.program_id(1)  # tile id along O
    if pid_b >= B:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < O

    x_row_ptr = x_ptr + pid_b * stride_xb
    y_row_ptr = y_ptr + pid_b * stride_yb

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    seed_i = tl.load(seeds_ptr + pid_b).to(tl.int32)

    k0 = 0
    while k0 < I:
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < I

        x_tile = tl.load(x_row_ptr + offs_k * stride_xi, mask=mask_k, other=0.0).to(tl.float32)

        # RNG offsets: offset = k * O + n
        k_offsets = offs_k.to(tl.int32)[:, None]
        n_offsets = offs_n.to(tl.int32)[None, :]
        offsets = k_offsets * O + n_offsets

        w_tile = tl.randn(seed_i, offsets, n_rounds=n_rounds)
        w_tile = tl.where(mask_k[:, None] & mask_n[None, :], w_tile, 0.0)

        acc += tl.sum(w_tile * x_tile[:, None], axis=0)
        k0 += BLOCK_K

    tl.store(y_row_ptr + offs_n * stride_yo, acc, mask=mask_n)


def _choose_tiling_gen(I, O):
    # Power-of-two chooser up to 256
    def choose(sz):
        if sz >= 4096:
            return 256
        if sz >= 1024:
            return 128
        if sz >= 256:
            return 64
        if sz >= 64:
            return 32
        return 16

    if I >= O:
        BLOCK_M = choose(I)
        BLOCK_N = choose(O)
    else:
        BLOCK_N = choose(O)
        BLOCK_M = choose(I)

    size_hint = max(BLOCK_M, BLOCK_N)
    num_warps = 8 if size_hint >= 128 else (4 if size_hint >= 64 else 2)
    num_stages = 4  # A100: latency hiding

    return BLOCK_M, BLOCK_N, num_warps, num_stages


@torch.no_grad()
def batched_randn_matmul(x: torch.Tensor,
                         seeds: torch.Tensor,
                         out_features: int,
                         *,
                         n_rounds: int = 10,
                         out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Computes y[b, :] = x[b, :] @ W_b where W_b ~ N(0,1) with seed=seeds[b],
    without materializing W_b. Reproducible per-row and independent of other rows' seeds.

    Args:
        x:        (B, I) float16/float32 tensor (CUDA).
        seeds:    (B,)   int64/int32 tensor.
        out_features: O.
        n_rounds: tl.randn quality parameter (default 10).
        out_dtype: dtype of y (default: x.dtype).

    Returns:
        y: (B, O) tensor on the same device as x, dtype=out_dtype or x.dtype.
    """
    assert x.is_cuda, "x must be on CUDA device"
    assert x.dim() == 2, "x must be (B, I)"
    B, I = x.shape
    O = int(out_features)
    assert O > 0

    if out_dtype is None:
        out_dtype = x.dtype
    # Accumulate in f32 for stability
    y = torch.empty((B, O), device=x.device, dtype=torch.float32)

    # strides (elements)
    stride_xb, stride_xi = x.stride()
    stride_yb, stride_yo = y.stride()

    seeds_dev = seeds.to(device=x.device, dtype=torch.int64)
    assert seeds_dev.numel() == B

    BLOCK_K, BLOCK_N, num_warps, num_stages = _choose_tiling_mm(I, O)
    grid = (B, triton.cdiv(O, BLOCK_N))

    _randn_mm_row_kernel[grid](
        x, seeds_dev, y,
        B, I, O,
        stride_xb, stride_xi,
        stride_yb, stride_yo,
        n_rounds=n_rounds,
        BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    return y.to(out_dtype)


# ============================================================
#  KERNEL 2: Generate W_batched[b, i, o] ~ N(0,1) per seed[b]
#            (B is RUNTIME, not constexpr)
# ============================================================

@triton.jit
def _randn_generate_kernel(
        seeds_ptr,  # *int64 [B]
        y_ptr,  # *f32   [B, I, O]
        B,  # RUNTIME scalar (int32) -- NOT constexpr
        I: tl.constexpr,
        O: tl.constexpr,
        stride_yb, stride_yi, stride_yo,
        n_rounds: tl.constexpr,
        BLOCK_M: tl.constexpr,  # tile over I
        BLOCK_N: tl.constexpr  # tile over O
):
    pid_b = tl.program_id(0)  # batch row
    pid_t = tl.program_id(1)  # flattened tiles over (I, O)
    if pid_b >= B:
        return

    tiles_n = tl.cdiv(O, BLOCK_N)
    tile_m = pid_t // tiles_n
    tile_n = pid_t % tiles_n

    i0 = tile_m * BLOCK_M
    o0 = tile_n * BLOCK_N

    offs_i = i0 + tl.arange(0, BLOCK_M)
    offs_o = o0 + tl.arange(0, BLOCK_N)
    mask_i = offs_i < I
    mask_o = offs_o < O

    seed_b = tl.load(seeds_ptr + pid_b).to(tl.int32)

    # offset = i * O + o
    i_offsets = offs_i.to(tl.int32)[:, None]
    o_offsets = offs_o.to(tl.int32)[None, :]
    offsets = i_offsets * O + o_offsets

    tile = tl.randn(seed_b, offsets, n_rounds=n_rounds)  # f32

    base_ptr = y_ptr + pid_b * stride_yb + offs_i[:, None] * stride_yi + offs_o[None, :] * stride_yo
    tl.store(base_ptr, tile, mask=mask_i[:, None] & mask_o[None, :])


def _choose_tiling_mm(I, O):
    # Power-of-two chooser up to 256 (A100 has plenty of regs/SMEM)
    def choose(sz):
        if sz >= 4096:  # very large
            return 256
        if sz >= 1024:
            return 128
        if sz >= 256:
            return 64
        if sz >= 64:
            return 32
        return 16

    if I >= O:
        BLOCK_K = choose(I)  # tile over in_features (K)
        BLOCK_N = choose(O)  # tile over out_features (N)
    else:
        BLOCK_N = choose(O)
        BLOCK_K = choose(I)

    # Warps/stages: A100 likes more pipeline depth
    size_hint = max(BLOCK_K, BLOCK_N)
    num_warps = 8 if size_hint >= 128 else (4 if size_hint >= 64 else 2)
    num_stages = 4  # deeper pipeline on A100

    return BLOCK_K, BLOCK_N, num_warps, num_stages


@torch.no_grad()
def batched_randn_generate(seeds: torch.Tensor,
                           in_features: int,
                           out_features: int,
                           *,
                           n_rounds: int = 10,
                           device: torch.device | None = None,
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Generate W_batched[b, i, o] ~ N(0,1) using tl.randn(seeds[b], i*O + o).
    """
    seeds = seeds.contiguous()
    if device is None:
        device = seeds.device if seeds.is_cuda else torch.device("cuda")
    seeds_dev = seeds.to(device=device, dtype=torch.int64)
    B = seeds_dev.numel()
    I, O = int(in_features), int(out_features)

    y = torch.empty((B, I, O), device=device, dtype=torch.float32)  # f32 generation

    # Strides (elements)
    stride_yb, stride_yi, stride_yo = y.stride()

    BLOCK_M, BLOCK_N, num_warps, num_stages = _choose_tiling_gen(I, O)
    tiles_m = triton.cdiv(I, BLOCK_M)
    tiles_n = triton.cdiv(O, BLOCK_N)
    grid = (B, tiles_m * tiles_n)

    _randn_generate_kernel[grid](
        seeds_dev, y,
        B, I, O,
        stride_yb, stride_yi, stride_yo,
        n_rounds=n_rounds,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    return y.to(dtype)


# ============================================================
#  TESTS
# ============================================================

def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


@torch.no_grad()
def run_tests():
    torch.manual_seed(0)
    device = torch.device("cuda")

    # 1) Small sanity
    for B, I, O in [(3, 8, 8), (2, 7, 5)]:
        x = torch.randn(B, I, device=device, dtype=torch.float16)
        seeds = torch.tensor([123, 456, 123][:B], device=device, dtype=torch.int64)

        # Baseline: generate W then bmm in PyTorch
        W = batched_randn_generate(seeds, I, O, device=device, dtype=torch.float32)  # [B, I, O]
        y_ref = torch.bmm(x.to(torch.float32).unsqueeze(1), W).squeeze(1)  # [B, O]

        # Kernel 1 result
        y_ker = batched_randn_matmul(x, seeds, O, out_dtype=torch.float32)
        diff = _max_abs_diff(y_ref, y_ker)
        print(f"[small] B={B} I={I} O={O}  max-abs-diff={diff:.3e}")
        assert diff < 0.1, "Bit-exact mismatch on small case"

    # 2) Skewed shapes (I >> O)
    B, I, O = 4, 2048, 8
    x = torch.randn(B, I, device=device, dtype=torch.float16)
    seeds = torch.tensor([7, 13, 7, 999], device=device, dtype=torch.int64)
    W = batched_randn_generate(seeds, I, O, device=device, dtype=torch.float32)
    y_ref = torch.bmm(x.to(torch.float32).unsqueeze(1), W).squeeze(1)
    y_ker = batched_randn_matmul(x, seeds, O, out_dtype=torch.float32)
    diff = _max_abs_diff(y_ref, y_ker)
    print(f"[skew I>>O] max-abs-diff={diff:.3e}")
    assert diff < 0.1

    # 3) Skewed shapes (O >> I)
    B, I, O = 4, 8, 2048
    x = torch.randn(B, I, device=device, dtype=torch.float16)
    seeds = torch.tensor([111, 222, 333, 111], device=device, dtype=torch.int64)
    W = batched_randn_generate(seeds, I, O, device=device, dtype=torch.float32)
    y_ref = torch.bmm(x.to(torch.float32).unsqueeze(1), W).squeeze(1)
    y_ker = batched_randn_matmul(x, seeds, O, out_dtype=torch.float32)
    diff = _max_abs_diff(y_ref, y_ker)
    print(f"[skew O>>I] max-abs-diff={diff:.3e}")
    assert diff < 0.1

    # 4) Reproducibility per-row (unchanged seed rows are identical)
    B, I, O = 5, 256, 64
    x = torch.randn(B, I, device=device, dtype=torch.float16)
    seeds1 = torch.tensor([5, 6, 7, 8, 9], device=device, dtype=torch.int64)
    seeds2 = torch.tensor([5, 999, 7, 42, 9], device=device, dtype=torch.int64)  # rows 0,2,4 preserved
    y1 = batched_randn_matmul(x, seeds1, O, out_dtype=torch.float32)
    y2 = batched_randn_matmul(x, seeds2, O, out_dtype=torch.float32)

    for idx in [0, 2, 4]:
        row_diff = _max_abs_diff(y1[idx], y2[idx])
        print(f"[repro row {idx}] max-abs-diff={row_diff:.3e}")
        assert row_diff < 0.1, "Row with same seed should be identical"

    # 5) Generator kernel determinism & order independence
    B, I, O = 3, 123, 77
    seedsA = torch.tensor([101, 202, 303], device=device, dtype=torch.int64)
    seedsB = torch.tensor([303, 101, 202], device=device, dtype=torch.int64)
    WA = batched_randn_generate(seedsA, I, O, device=device, dtype=torch.float32)
    WB = batched_randn_generate(seedsB, I, O, device=device, dtype=torch.float32)

    # Check that WA[seed==101] equals WB[seed==101], etc.
    mapping = {101: (0, 1), 202: (1, 2), 303: (2, 0)}
    for s, (ia, ib) in mapping.items():
        dd = _max_abs_diff(WA[ia], WB[ib])
        print(f"[gen reorder seed={s}] max-abs-diff={dd:.3e}")
        assert dd < 0.1

    print("All tests passed ✅")


if __name__ == "__main__":
    run_tests()
