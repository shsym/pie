"""CUDA implementation of batched random-matmul / generate.

Drop-in replacement for the Triton `rand_mv` module (see baseline.py):

    batched_randn_matmul(x, seeds, S, *, n_rounds=10, out_dtype=None,
                         col_offset=0, global_cols=None) -> Tensor
    batched_randn_generate(seeds, S, *, n_rounds=10, device=None,
                           dtype=torch.float32, col_offset=0,
                           global_cols=None) -> Tensor
    run_tests()
    RAND_MV_AVAILABLE

`n_rounds` honors the Philox round count. Default 10 matches the Triton
baseline; passing n_rounds=7 opts into the BM7 variant (≈15% faster on
compute-bound shapes, passes every statistical check in stats.py).

Advanced: the four underlying variants are exposed as attributes for
research / A-B testing:

    rand_mv_cuda.BM      - Box-Muller, 10 rounds (this module's default)
    rand_mv_cuda.BM7     - Box-Muller,  7 rounds (fastest)
    rand_mv_cuda.PROBIT  - normcdfinvf, 10 rounds
    rand_mv_cuda.ZIG     - Ziggurat,   10 rounds

See stats.py for the statistical harness and bench.py for timings.
"""
from __future__ import annotations
import os
import types
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

RAND_MV_AVAILABLE = torch.cuda.is_available()


if not RAND_MV_AVAILABLE:
    def batched_randn_matmul(*args, **kwargs):
        raise RuntimeError("rand_mv_cuda requires CUDA.")

    def batched_randn_generate(*args, **kwargs):
        raise RuntimeError("rand_mv_cuda requires CUDA.")

    def run_tests():
        raise RuntimeError("rand_mv_cuda tests require CUDA.")

else:
    from torch.utils.cpp_extension import load

    _ext_cache: dict[tuple[int, int], object] = {}

    def _get_ext(rng_method: int, rounds: int):
        key = (rng_method, rounds)
        if key in _ext_cache:
            return _ext_cache[key]
        ext = load(
            name=f"rand_mv_cuda_v9_m{rng_method}_r{rounds}",
            sources=[os.path.join(_HERE, "cuda_kernels.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-gencode=arch=compute_86,code=sm_86",
                "-std=c++17",
                f"-DRNG_METHOD={rng_method}",
                f"-DPHILOX_ROUNDS={rounds}",
            ],
            extra_cflags=[
                "-O3", "-std=c++17",
                f"-DRNG_METHOD={rng_method}",
                f"-DPHILOX_ROUNDS={rounds}",
            ],
            verbose=False,
        )
        _ext_cache[key] = ext
        return ext

    def _bind(ext):
        """Wrap a compiled extension with the public matmul/generate API."""

        @torch.no_grad()
        def matmul(x, seeds, S, *, n_rounds=10, out_dtype=None,
                   col_offset=0, global_cols=None,
                   seed_offset=0, out=None, alpha=1.0, beta=0.0,
                   W_mean=None):
            """Compute y = alpha * x @ (W_mean + S*N(seeds+seed_offset)) + beta*out.

            When W_mean is provided, the deterministic mean projection (x @
            W_mean) is fused into the noise kernel, saving a separate cublas
            call. W_mean must have the same shape and dtype as S.
            `out` (if provided) may be float32, float16, or bfloat16. When
            out is None we always allocate float32 internally.
            """
            assert x.is_cuda and S.is_cuda
            assert x.dim() == 2 and S.dim() == 2
            B, I = x.shape
            I_S, O = S.shape
            assert I_S == I
            if out_dtype is None:
                out_dtype = x.dtype
            if global_cols is None:
                global_cols = O
            seeds_dev = seeds.to(device=x.device, dtype=torch.int64)
            y = ext.batched_randn_matmul(
                x, seeds_dev, S, col_offset, global_cols,
                int(seed_offset), out, float(alpha), float(beta), W_mean,
            )
            return y.to(out_dtype) if (out is None and out_dtype != torch.float32) else y

        @torch.no_grad()
        def matmul_sectioned(x, seeds, S, *,
                             section_widths, section_offsets,
                             col_offset=0, global_cols=None,
                             out=None, alpha=1.0, beta=0.0, out_dtype=None,
                             W_mean=None):
            """Fused N-section batched_randn_matmul. Same alpha/beta semantics
            as matmul: y = alpha * x @ (...) + beta*out."""
            assert x.is_cuda and S.is_cuda
            assert x.dim() == 2 and S.dim() == 2
            assert len(section_widths) == len(section_offsets)
            assert sum(section_widths) == S.shape[1]
            if global_cols is None:
                global_cols = int(S.shape[1])
            seeds_dev = seeds.to(device=x.device, dtype=torch.int64)
            y = ext.batched_randn_matmul_sectioned(
                x, seeds_dev, S,
                list(section_widths), list(section_offsets),
                col_offset, global_cols,
                out, float(alpha), float(beta), W_mean,
            )
            if out is None and out_dtype is not None and out_dtype != torch.float32:
                y = y.to(out_dtype)
            return y

        @torch.no_grad()
        def generate(seeds, S, *, n_rounds=10, device=None,
                     dtype=torch.float32, col_offset=0, global_cols=None):
            if device is None:
                device = S.device if S.is_cuda else torch.device("cuda")
            seeds_dev = seeds.to(device=device, dtype=torch.int64)
            S_dev = S.to(device=device)
            assert S_dev.dim() == 2
            if global_cols is None:
                global_cols = int(S_dev.size(1))
            y = ext.batched_randn_generate(seeds_dev, S_dev, col_offset, global_cols)
            return y.to(dtype) if dtype != torch.float32 else y

        return types.SimpleNamespace(
            batched_randn_matmul=matmul,
            batched_randn_matmul_sectioned=matmul_sectioned,
            batched_randn_generate=generate,
        )

    # Only BM7 (the default) is eager — the rest compile on first access so
    # that `import rand_mv_cuda` in a fresh env doesn't build every variant.
    class _LazyVariant:
        def __init__(self, rng_method: int, rounds: int):
            self._rng_method = rng_method
            self._rounds = rounds
            self._mod = None
        def _resolve(self):
            if self._mod is None:
                self._mod = _bind(_get_ext(self._rng_method, self._rounds))
            return self._mod
        def __getattr__(self, name):
            return getattr(self._resolve(), name)

    BM7    = _bind(_get_ext(0,  7))
    BM     = _LazyVariant(0, 10)
    PROBIT = _LazyVariant(1, 10)
    ZIG    = _LazyVariant(2, 10)

    # Public top-level API — `n_rounds` routes to the matching variant so the
    # function is a true drop-in for the Triton version. Anything <= 7 uses
    # BM7 (fastest); anything >= 8 uses BM (matches Triton's default of 10).
    def batched_randn_matmul(x, seeds, S, *, n_rounds=10, out_dtype=None,
                             col_offset=0, global_cols=None,
                             seed_offset=0, out=None, alpha=1.0, beta=0.0,
                             W_mean=None):
        impl = BM7 if n_rounds <= 7 else BM
        return impl.batched_randn_matmul(
            x, seeds, S,
            n_rounds=n_rounds, out_dtype=out_dtype,
            col_offset=col_offset, global_cols=global_cols,
            seed_offset=seed_offset, out=out, alpha=alpha, beta=beta,
            W_mean=W_mean,
        )

    def batched_randn_matmul_sectioned(x, seeds, S, *,
                                       section_widths, section_offsets,
                                       n_rounds=10, out_dtype=None,
                                       col_offset=0, global_cols=None,
                                       out=None, alpha=1.0, beta=0.0,
                                       W_mean=None):
        impl = BM7 if n_rounds <= 7 else BM
        return impl.batched_randn_matmul_sectioned(
            x, seeds, S,
            section_widths=section_widths, section_offsets=section_offsets,
            col_offset=col_offset, global_cols=global_cols,
            out=out, alpha=alpha, beta=beta, out_dtype=out_dtype,
            W_mean=W_mean,
        )

    def batched_randn_generate(seeds, S, *, n_rounds=10, device=None,
                               dtype=torch.float32, col_offset=0, global_cols=None):
        impl = BM7 if n_rounds <= 7 else BM
        return impl.batched_randn_generate(
            seeds, S,
            n_rounds=n_rounds, device=device, dtype=dtype,
            col_offset=col_offset, global_cols=global_cols,
        )

    # ==================================================================
    #  Tests (mirror baseline.py's run_tests() but driven by our impls).
    # ==================================================================

    def _max_abs_diff(a, b):
        return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()

    @torch.no_grad()
    def run_tests():
        """Compare our matmul against bmm(x, generate(seed, S)) for self-consistency,
        plus zero-seed, reproducibility, and sharded-output checks."""
        torch.manual_seed(0)
        device = torch.device("cuda")

        def do_case(B, I, O, dtype_x=torch.float16, dtype_S=torch.float32):
            x = torch.randn(B, I, device=device, dtype=dtype_x)
            S = torch.ones(I, O, device=device, dtype=dtype_S)
            seeds = torch.randint(1, 1 << 30, (B,), device=device, dtype=torch.int64)

            W = batched_randn_generate(seeds, S, device=device, dtype=torch.float32)
            y_ref = torch.bmm(x.to(torch.float32).unsqueeze(1), W).squeeze(1)
            y_ker = batched_randn_matmul(x, seeds, S, out_dtype=torch.float32)
            diff = _max_abs_diff(y_ref, y_ker)
            print(f"B={B} I={I} O={O} | max-abs-diff={diff:.3e}")
            assert diff < 0.1, "matmul vs bmm(generate) mismatch"

            if B >= 2:
                seeds_with_zero = seeds.clone()
                seeds_with_zero[1] = 0
                W_z = batched_randn_generate(seeds_with_zero, S, device=device,
                                             dtype=torch.float32)
                assert torch.all(W_z[1] == 0), "gen: zero seed didn't zero batch"
                assert torch.all(W_z[0] == W[0]), "gen: zero seed disturbed other batch"

                y_z = batched_randn_matmul(x, seeds_with_zero, S, out_dtype=torch.float32)
                assert torch.all(y_z[1] == 0), "matmul: zero seed didn't zero batch"
                assert _max_abs_diff(y_z[0], y_ref[0]) < 0.1
                print(f"  zero-seed OK")

            if B >= 3:
                seeds2 = seeds.clone()
                seeds2[1] = seeds2[1] + 12345
                y1 = batched_randn_matmul(x, seeds,  S, out_dtype=torch.float32)
                y2 = batched_randn_matmul(x, seeds2, S, out_dtype=torch.float32)
                assert _max_abs_diff(y1[0], y2[0]) < 0.1
                assert _max_abs_diff(y1[2], y2[2]) < 0.1
                print(f"  repro OK")

            if O % 2 == 0:
                half = O // 2
                y_left = batched_randn_matmul(
                    x, seeds, S[:, :half],
                    out_dtype=torch.float32, col_offset=0, global_cols=O,
                )
                y_right = batched_randn_matmul(
                    x, seeds, S[:, half:],
                    out_dtype=torch.float32, col_offset=half, global_cols=O,
                )
                d_l = _max_abs_diff(y_left,  y_ref[:, :half])
                d_r = _max_abs_diff(y_right, y_ref[:, half:])
                print(f"  sharding {O} -> {half}+{half} | L={d_l:.2e} R={d_r:.2e}")
                assert d_l < 0.1 and d_r < 0.1

        do_case(B=3,  I=8,    O=8)
        do_case(B=2,  I=7,    O=5)
        do_case(B=4,  I=2048, O=8)
        do_case(B=4,  I=8,    O=2048)
        do_case(B=3,  I=129,  O=65, dtype_x=torch.float16, dtype_S=torch.float16)
        print("All tests passed ✅")


if __name__ == "__main__":
    run_tests()
