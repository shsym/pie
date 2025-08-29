#include "metal_dtype_conversion.hpp"
#include <chrono>
#include <iostream>
#include <vector>

namespace metal {

// BF16 to F32 conversion
float DTypeConverter::bf16_to_f32(bfloat16_t bf16) {
    // BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
    // F32 format: 1 sign bit, 8 exponent bits, 23 mantissa bits
    // Simply left-shift by 16 to convert BF16 to F32
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

// F32 to BF16 conversion with rounding
bfloat16_t DTypeConverter::f32_to_bf16(float f32) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f32, sizeof(f32_bits));
    
    // Round to nearest even (banker's rounding)
    uint32_t rounding_bias = 0x7FFF + ((f32_bits >> 16) & 1);
    uint32_t rounded = f32_bits + rounding_bias;
    
    // Extract top 16 bits for BF16
    return static_cast<bfloat16_t>(rounded >> 16);
}

// F32 to F16 conversion using IEEE 754 bit manipulation
float16_t DTypeConverter::f32_to_f16(float f32) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f32, sizeof(f32_bits));
    
    // Extract components
    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exp = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa = f32_bits & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 0xFF) {  // Infinity or NaN
        return static_cast<float16_t>((sign << 15) | 0x7C00 | (mantissa ? 0x200 : 0));
    }
    
    if (exp == 0) {  // Zero or subnormal
        return static_cast<float16_t>(sign << 15);
    }
    
    // Adjust exponent for F16 bias (15 vs 127)
    int32_t f16_exp = static_cast<int32_t>(exp) - 127 + 15;
    
    // Handle overflow
    if (f16_exp >= 31) {
        return static_cast<float16_t>((sign << 15) | 0x7C00);  // Infinity
    }
    
    // Handle underflow
    if (f16_exp <= 0) {
        return static_cast<float16_t>(sign << 15);  // Zero
    }
    
    // Normal case
    uint32_t f16_mantissa = mantissa >> 13;  // Truncate to 10 bits
    return static_cast<float16_t>((sign << 15) | (f16_exp << 10) | f16_mantissa);
}

// F16 to F32 conversion using IEEE 754 bit manipulation
float DTypeConverter::f16_to_f32(float16_t f16) {
    // Extract components
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mantissa = f16 & 0x3FF;
    
    uint32_t f32_bits;
    
    if (exp == 0x1F) {  // Infinity or NaN
        f32_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else if (exp == 0) {  // Zero or subnormal
        f32_bits = sign << 31;  // Zero
    } else {  // Normal case
        uint32_t f32_exp = exp - 15 + 127;  // Adjust bias
        f32_bits = (sign << 31) | (f32_exp << 23) | (mantissa << 13);
    }
    
    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
}

// Full conversion chain: BF16 -> F32 -> F16
float16_t DTypeConverter::bf16_to_f16(bfloat16_t bf16) {
    float f32_intermediate = bf16_to_f32(bf16);
    return f32_to_f16(f32_intermediate);
}

// Full conversion chain: F16 -> F32 -> BF16
bfloat16_t DTypeConverter::f16_to_bf16(float16_t f16) {
    float f32_intermediate = f16_to_f32(f16);
    return f32_to_bf16(f32_intermediate);
}

// Batch conversion: BF16 -> F16 (for input data)
void DTypeConverter::convert_bf16_to_f16_batch(const bfloat16_t* input, float16_t* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = bf16_to_f16(input[i]);
    }
}

// Batch conversion: F16 -> BF16 (for output data)
void DTypeConverter::convert_f16_to_bf16_batch(const float16_t* input, bfloat16_t* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = f16_to_bf16(input[i]);
    }
}

// Check for native BF16 support (fallback implementation)
bool DTypeConverter::check_native_bf16_support(id<MTLDevice> device) {
    // Simple fallback - assume no native BF16 support for now
    // Real implementation would check device capabilities
    return false;
}

// Measure conversion overhead
double DTypeConverter::measure_conversion_overhead_us(size_t element_count) {
    // Create test data
    std::vector<bfloat16_t> bf16_data(element_count);
    std::vector<float16_t> f16_data(element_count);
    std::vector<bfloat16_t> bf16_result(element_count);
    
    // Initialize with test pattern
    for (size_t i = 0; i < element_count; ++i) {
        bf16_data[i] = f32_to_bf16(static_cast<float>(i) * 0.1f);
    }
    
    // Measure BF16 -> F16 conversion
    auto start = std::chrono::high_resolution_clock::now();
    convert_bf16_to_f16_batch(bf16_data.data(), f16_data.data(), element_count);
    auto mid = std::chrono::high_resolution_clock::now();
    convert_f16_to_bf16_batch(f16_data.data(), bf16_result.data(), element_count);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto total_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(total_duration.count()) / 1000.0; // Convert to microseconds
}

} // namespace metal