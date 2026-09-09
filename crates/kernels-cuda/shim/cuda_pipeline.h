
#pragma once

__device__ __forceinline__ void __pipeline_memcpy_async(void* dst, const void* src,
                                                        unsigned long n) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    if (n == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 16;" ::"r"(s), "l"(src));
    } else if (n == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, 8;" ::"r"(s), "l"(src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, 4;" ::"r"(s), "l"(src));
    }
}

__device__ __forceinline__ void __pipeline_commit() {
    asm volatile("cp.async.commit_group;" ::);
}

template <int N>
__device__ __forceinline__ void pie_pipeline_wait_prior() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
#define __pipeline_wait_prior(N) pie_pipeline_wait_prior<(N)>()
