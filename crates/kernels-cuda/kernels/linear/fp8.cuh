#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

__device__ __forceinline__ float e4m3_to_f32(u8 byte) {
    const int exp = (byte >> 3) & 0xF;
    const int mant = byte & 0x7;
    float mag;
    if (exp == 0) {

        mag = static_cast<float>(mant) * 0.001953125f;
    } else if (exp == 0xF && mant == 0x7) {
        mag = __int_as_float(0x7fc00000);
    } else {

        mag = (1.f + static_cast<float>(mant) * 0.125f)
            * __int_as_float((exp - 7 + 127) << 23);
    }
    return (byte & 0x80) ? -mag : mag;
}

template <class T, int kRowsT>
__global__ void matmul_fp8_row(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    const int token = blockIdx.x;

    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const float* __restrict__ sf = reinterpret_cast<const float*>(scales);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) row_of[r] = min(row0 + r, n - 1);

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int j = lane_id; j < k; j += 32) {
        const float xv = Elem<T>::to_f32(x[j]);
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const u8 code =
                codes[static_cast<long long>(row_of[r]) * k + j];
            acc[r] = fmaf(e4m3_to_f32(code), xv, acc[r]);
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
                    Elem<T>::from_f32(acc[r] * sf[row]);
        }
    }
}

template <class T, int kRowsT>
__global__ void matmul_fp8_tile(
    const T* __restrict__ act,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    T* __restrict__ out,
    int n,
    int k,
    const u32* __restrict__ win)
{
    constexpr int kRows = kRowsT;
    constexpr int kTile = 128;
    const int token = blockIdx.x;
    if (win != nullptr && token >= static_cast<int>(win[0])) return;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row0 = (blockIdx.y * (blockDim.x >> 5) + warp_in_block) * kRows;
    if (row0 >= n) return;

    const int ktiles = (k + kTile - 1) / kTile;
    const float* __restrict__ sf = reinterpret_cast<const float*>(scales);
    const T* x = act + static_cast<long long>(token) * k;

    int row_of[kRows];
    const float* band_of[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        row_of[r] = min(row0 + r, n - 1);
        band_of[r] = sf + static_cast<long long>(row_of[r] / kTile) * ktiles;
    }

    float acc[kRows];
#pragma unroll
    for (int r = 0; r < kRows; ++r) acc[r] = 0.f;

    for (int kt = 0; kt < ktiles; ++kt) {
        const int base = kt * kTile;
        const int lim = min(kTile, k - base);

        float part[kRows];
#pragma unroll
        for (int r = 0; r < kRows; ++r) part[r] = 0.f;

        for (int j = lane_id; j < lim; j += 32) {
            const float xv = Elem<T>::to_f32(x[base + j]);
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const u8 code =
                    codes[static_cast<long long>(row_of[r]) * k + base + j];
                part[r] = fmaf(e4m3_to_f32(code), xv, part[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r)
            acc[r] = fmaf(part[r], band_of[r][kt], acc[r]);
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
                    Elem<T>::from_f32(acc[r]);
        }
    }
}

}
