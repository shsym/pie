#pragma once

#include "prelude/device.cuh"

#include "linear/fp8.cuh"

namespace pie::linear {

__device__ __constant__ float kNvfp4Lut[16] = {
     0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f,
};

template <class T, int kRowsT>
__global__ void matmul_nvfp4(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    float tensor_scale,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kGroup = 16;
    const int token = blockIdx.x;

    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int groups_per_row = k / kGroup;

    const int words_per_row = k / 8;
    const unsigned* __restrict__ w32 =
        reinterpret_cast<const unsigned*>(codes);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int g = lane_id; g < groups_per_row; g += 32) {
        float xv[kGroup];
#pragma unroll
        for (int j = 0; j < kGroup; ++j)
            xv[j] = Elem<T>::to_f32(x[g * kGroup + j]);

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            float part = 0.f;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                const unsigned word =
                    w32[static_cast<long long>(row_of[r]) * words_per_row
                        + g * 2 + q];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned byte = (word >> (8 * b)) & 0xFFu;
                    const int at = q * 8 + b * 2;
                    part = fmaf(kNvfp4Lut[byte & 0xFu], xv[at], part);
                    part = fmaf(kNvfp4Lut[byte >> 4], xv[at + 1], part);
                }
            }

            const u8 sb = scales[static_cast<long long>(row_of[r])
                                 * groups_per_row + g];
            acc[r] = fmaf(part, e4m3_to_f32(sb), acc[r]);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], off);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int row = row0 + r;
            if (row < n)
                out[static_cast<long long>(token) * n + row] =
                    Elem<T>::from_f32(acc[r] * tensor_scale);
        }
    }
}

}
