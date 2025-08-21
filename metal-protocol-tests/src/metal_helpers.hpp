// Shared helpers for Metal protocol tests
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <iostream>
#include <vector>

// Pull in bfloat16_t alias from Metal GEMM header (uint16_t)
#include "metal_gemm.hpp"

// bf16 conversions
static inline bfloat16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<bfloat16_t>((bits + 0x8000u) >> 16);
}

static inline float bf16_to_float(bfloat16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Minimal IEEE fp16 conversions (sufficient for tests)
static inline uint16_t float_to_half(float f) {
    union { float f; uint32_t i; } u;
    u.f = f;
    if (f == 0.0f) return static_cast<uint16_t>(u.i >> 16); // preserve sign of zero
    if (!std::isfinite(f)) {
        if (std::isnan(f)) return 0x7e00; // qNaN
        return static_cast<uint16_t>((u.i >> 16) | 0x7c00); // inf with sign
    }
    uint32_t sign = (u.i >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((u.i >> 23) & 0xff) - 127 + 15;
    uint32_t mantissa = (u.i >> 13) & 0x3ffu;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mantissa);
}

static inline float half_to_float(uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u) >> 10;
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t f;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f = sign;
        } else {
            int shift = 0;
            while ((h_sig & 0x0400u) == 0) { h_sig <<= 1; ++shift; }
            h_sig &= 0x03FFu;
            uint32_t exp = 127 - 15 - static_cast<uint32_t>(shift);
            uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
            f = sign | (exp << 23) | mant;
        }
    } else if (h_exp == 0x1Fu) {
        uint32_t exp = 0xFFu;
        uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
        f = sign | (exp << 23) | mant;
    } else {
        uint32_t exp = static_cast<uint32_t>(h_exp) - 15 + 127;
        uint32_t mant = static_cast<uint32_t>(h_sig) << 13;
        f = sign | (exp << 23) | mant;
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Debug helpers for quick stats (only meaningful for bf16 vectors)
template<typename T>
static inline void print_vec_stats(const std::string& name, const std::vector<T>& vec) {
    if (vec.empty()) {
        std::cout << "📊 " << name << " is empty." << std::endl;
        return;
    }
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum_val = 0.0;
    size_t non_zero_count = 0;
    for (const auto& val_bf16 : vec) {
        float val = bf16_to_float(val_bf16);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
        if (std::abs(val) > 1e-9f) non_zero_count++;
    }
    std::cout << "📊 " << name << " range: [" << min_val << ", " << max_val
              << "], avg=" << sum_val / vec.size()
              << ", non_zero=" << non_zero_count << "/" << vec.size()
              << " (" << (100.0 * non_zero_count / vec.size()) << "%)" << std::endl;
}

template<typename T>
static inline void print_vec_stats(const std::string& name, const std::vector<T>& vec, size_t n) {
    if (vec.empty()) {
        std::cout << "📊 " << name << " is empty." << std::endl;
        return;
    }
    size_t count = std::min(n, vec.size());
    if (count == 0) {
        print_vec_stats(name, vec);
        return;
    }
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum_val = 0.0;
    size_t non_zero_count = 0;
    for (size_t i = 0; i < count; ++i) {
        float val = bf16_to_float(vec[i]);
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
        if (std::abs(val) > 1e-9f) non_zero_count++;
    }
    std::cout << "📊 " << name << " range (first " << count << "): [" << min_val << ", " << max_val
              << "], avg=" << sum_val / count
              << ", non_zero=" << non_zero_count << "/" << count
              << ", total_size=" << vec.size() << std::endl;
}
