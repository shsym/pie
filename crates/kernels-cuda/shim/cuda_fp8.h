
#pragma once

#if !defined(__CUDA_ARCH__)
#error "cuda_fp8.h (pie shim) is device text: it is compiled by NVRTC for one \
architecture, and every conversion in it is an sm_89 instruction. There is no \
host half, because the host half of this crate is Rust."
#elif __CUDA_ARCH__ < 890
#error "cuda_fp8.h (pie shim) implements E4M3/E5M2 with cvt.rn.satfinite.e4m3x2.f32 \
and cvt.rn.f16x2.e4m3x2, which need sm_89 or newer. A software path is a \
deliberate absence: nvcc emits the same hardware cvt on this box, so an \
emulation could not be gated against it, and ungated arithmetic is the failure \
this shim exists to prevent."
#endif


typedef unsigned char __nv_fp8_storage_t;

typedef unsigned short __nv_fp8x2_storage_t;

typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_fp8_interpretation_t {
    __NV_E4M3 = 0,
    __NV_E5M2 = 1
} __nv_fp8_interpretation_t;

typedef enum __nv_saturation_t { __NV_SATFINITE = 1 } __nv_saturation_t;


__device__ __forceinline__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(
    const float2 x, const __nv_saturation_t saturate,
    const __nv_fp8_interpretation_t fp8_interpretation) {
    (void)saturate;
    __nv_fp8x2_storage_t storage;
    if (fp8_interpretation == __NV_E5M2) {
        asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    } else {
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
            : "=h"(storage)
            : "f"(x.y), "f"(x.x));
    }
    return storage;
}

__device__ __forceinline__ __nv_fp8_storage_t
__nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate,
                      const __nv_fp8_interpretation_t fp8_interpretation) {
    const float2 pair = make_float2(x, 0.0f);
    return (__nv_fp8_storage_t)__nv_cvt_float2_to_fp8x2(pair, saturate,
                                                        fp8_interpretation);
}

#if defined(__CUDA_FP16_TYPES_EXIST__)

__device__ __forceinline__ __half_raw
__nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x,
                        const __nv_fp8_interpretation_t fp8_interpretation) {
    const unsigned short packed = (unsigned short)x;
    unsigned int pair;
    if (fp8_interpretation == __NV_E5M2) {
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(pair) : "h"(packed));
    } else {
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(packed));
    }
    __half_raw res;
    res.x = (unsigned short)(pair & 0xFFFFu);
    return res;
}

#endif


__device__ __forceinline__ float __pie_fp8_halfbits_to_float(const unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

__device__ __forceinline__ __nv_fp8_storage_t
__pie_fp8_from_float(const float f, const __nv_fp8_interpretation_t interp) {
    return __nv_cvt_float_to_fp8(f, __NV_SATFINITE, interp);
}


struct __nv_fp8_e4m3 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e4m3() = default;

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const float f) {
        __x = __pie_fp8_from_float(f, __NV_E4M3);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __half f) {
        __x = __pie_fp8_from_float(
            __pie_fp8_halfbits_to_float(static_cast<__half_raw>(f).x), __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E4M3));
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8_e4m3(const __nv_bfloat16 f) {
        const unsigned int bits = ((unsigned int)static_cast<__nv_bfloat16_raw>(f).x) << 16;
        __x = __pie_fp8_from_float(__int_as_float((int)bits), __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    explicit __device__ __forceinline__ operator float() const {
        const unsigned short packed = (unsigned short)__x;
        unsigned int pair;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(packed));
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};

struct __nv_fp8_e5m2 {
    __nv_fp8_storage_t __x;

    __nv_fp8_e5m2() = default;

    explicit __device__ __forceinline__ __nv_fp8_e5m2(const float f) {
        __x = __pie_fp8_from_float(f, __NV_E5M2);
    }

#if defined(__CUDA_FP16_TYPES_EXIST__)
    explicit __device__ __forceinline__ __nv_fp8_e5m2(const __half f) {
        __x = __pie_fp8_from_float(
            __pie_fp8_halfbits_to_float(static_cast<__half_raw>(f).x), __NV_E5M2);
    }

    explicit __device__ __forceinline__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp8_to_halfraw(__x, __NV_E5M2));
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
    explicit __device__ __forceinline__ __nv_fp8_e5m2(const __nv_bfloat16 f) {
        const unsigned int bits = ((unsigned int)static_cast<__nv_bfloat16_raw>(f).x) << 16;
        __x = __pie_fp8_from_float(__int_as_float((int)bits), __NV_E5M2);
    }

    explicit __device__ __forceinline__ operator __nv_bfloat16() const {
        unsigned short bits;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(float(*this)));
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return static_cast<__nv_bfloat16>(raw);
    }
#endif

    explicit __device__ __forceinline__ operator float() const {
        const unsigned short packed = (unsigned short)__x;
        unsigned int pair;
        asm("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(pair) : "h"(packed));
        return __pie_fp8_halfbits_to_float((unsigned short)(pair & 0xFFFFu));
    }
};


struct __nv_fp8x2_e4m3 {
    __nv_fp8x2_storage_t __x;

    __nv_fp8x2_e4m3() = default;

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(__nv_fp8x2_storage_t bits) : __x(bits) {}

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const float2 f)
        : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3)) {}

#if defined(__CUDA_FP16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __half2 v) {
        const __half2_raw raw = static_cast<__half2_raw>(v);
        float2 f;
        f.x = __pie_fp8_halfbits_to_float(raw.x);
        f.y = __pie_fp8_halfbits_to_float(raw.y);
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }

    explicit __device__ __forceinline__ operator __half2() const {
        unsigned int pair;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"(__x));
        __half2_raw raw;
        raw.x = (unsigned short)(pair & 0xFFFFu);
        raw.y = (unsigned short)(pair >> 16);
        return static_cast<__half2>(raw);
    }
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)

    explicit __device__ __forceinline__ __nv_fp8x2_e4m3(const __nv_bfloat162 v) {
        const __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(v);
        float2 f;
        f.x = __int_as_float((int)(((unsigned int)raw.x) << 16));
        f.y = __int_as_float((int)(((unsigned int)raw.y) << 16));
        __x = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
#endif
};

struct __nv_fp8x2_e5m2 {
    __nv_fp8x2_storage_t __x;
};

struct __nv_fp8x4_e4m3 {
    __nv_fp8x4_storage_t __x;
};

struct __nv_fp8x4_e5m2 {
    __nv_fp8x4_storage_t __x;
};
