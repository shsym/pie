#pragma once

#include <Metal/Metal.h>
#include <cstdint>
#include <cstring>

namespace metal {

// Type definitions
using bfloat16_t = uint16_t;
using float16_t = uint16_t;  // Use uint16_t to represent F16 bits

// Forward declarations
class DTypeConverter {
public:
    // BF16 <-> F32 conversions
    static float bf16_to_f32(bfloat16_t bf16);
    static bfloat16_t f32_to_bf16(float f32);
    
    // F32 <-> F16 conversions using bit manipulation
    static float16_t f32_to_f16(float f32);
    static float f16_to_f32(float16_t f16);
    
    // Full conversion chain: BF16 -> F32 -> F16
    static float16_t bf16_to_f16(bfloat16_t bf16);
    
    // Full conversion chain: F16 -> F32 -> BF16
    static bfloat16_t f16_to_bf16(float16_t f16);
    
    // Batch conversion utilities
    static void convert_bf16_to_f16_batch(const bfloat16_t* input, float16_t* output, size_t count);
    static void convert_f16_to_bf16_batch(const float16_t* input, bfloat16_t* output, size_t count);
    
    // Check for native BF16 support (fallback implementation)
    static bool check_native_bf16_support(id<MTLDevice> device);
    
    // Conversion overhead measurement
    static double measure_conversion_overhead_us(size_t element_count);
};

} // namespace metal