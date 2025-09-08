#pragma once

#include "metal_dtype_conversion.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <limits>

namespace metal_test {

// Data type validation and conversion utilities
class DTypeValidator {
public:
    DTypeValidator() = default;
    ~DTypeValidator() = default;

    // Validate and convert CUDA artifact data to Metal format
    std::vector<float> convert_cuda_to_metal_f32(const std::vector<float>& cuda_data, const std::string& cuda_dtype);

    // Convert Metal output to comparison format
    std::vector<float> convert_metal_to_comparison(const void* metal_data, size_t element_count, const std::string& metal_dtype);

    // Convert from BF16 to F32 for comparison
    std::vector<float> convert_bf16_to_f32(const std::vector<uint16_t>& bf16_data);
    std::vector<float> convert_bf16_to_f32(const metal::bfloat16_t* bf16_data, size_t count);

    // Convert from F16 to F32 for comparison
    std::vector<float> convert_f16_to_f32(const std::vector<uint16_t>& f16_data);
    std::vector<float> convert_f16_to_f32(const metal::float16_t* f16_data, size_t count);

    // Validate conversion accuracy
    struct ConversionValidationResult {
        bool passed;
        double max_conversion_error;
        double mean_conversion_error;
        size_t failed_conversions;
        std::string error_message;

        ConversionValidationResult() : passed(false), max_conversion_error(0.0), mean_conversion_error(0.0), failed_conversions(0) {}
    };

    ConversionValidationResult validate_bf16_conversion_accuracy(const std::vector<float>& original_f32);
    ConversionValidationResult validate_f16_conversion_accuracy(const std::vector<float>& original_f32);

    // Check for invalid floating point values
    struct FloatValidationResult {
        bool has_nan;
        bool has_inf;
        bool has_subnormal;
        size_t nan_count;
        size_t inf_count;
        size_t subnormal_count;

        FloatValidationResult() : has_nan(false), has_inf(false), has_subnormal(false),
                                 nan_count(0), inf_count(0), subnormal_count(0) {}
    };

    FloatValidationResult validate_float_values(const std::vector<float>& data);

    // Utility methods
    static bool is_supported_dtype(const std::string& dtype);
    static std::string get_metal_dtype_name(const std::string& cuda_dtype);

    // Static conversion methods using centralized converter
    static std::vector<float> static_bf16_to_f32(const metal::bfloat16_t* data, size_t count);
    static std::vector<float> static_f16_to_f32(const metal::float16_t* data, size_t count);
    static std::vector<metal::bfloat16_t> static_f32_to_bf16(const std::vector<float>& f32_data);
    static std::vector<metal::float16_t> static_f32_to_f16(const std::vector<float>& f32_data);

private:
    // Internal methods
    bool is_valid_float_value(float value, FloatValidationResult& result) const;
};

} // namespace metal_test