#include "dtype_validator.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

namespace metal_test {

std::vector<float> DTypeValidator::convert_cuda_to_metal_f32(const std::vector<float>& cuda_data, const std::string& cuda_dtype) {
    if (cuda_dtype == "float32") {
        // Direct copy for F32
        return cuda_data;
    } else if (cuda_dtype == "bfloat16") {
        std::cerr << "Warning: CUDA artifact labeled as bfloat16 but contains float32 data" << std::endl;
        return cuda_data;
    } else if (cuda_dtype == "float16") {
        std::cerr << "Warning: CUDA artifact labeled as float16 but contains float32 data" << std::endl;
        return cuda_data;
    } else {
        std::cerr << "Unsupported CUDA dtype: " << cuda_dtype << std::endl;
        return cuda_data; // Return as-is, might fail later
    }
}

std::vector<float> DTypeValidator::convert_metal_to_comparison(const void* metal_data, size_t element_count, const std::string& metal_dtype) {
    if (metal_dtype == "float32" || metal_dtype == "f32") {
        const float* f32_data = static_cast<const float*>(metal_data);
        return std::vector<float>(f32_data, f32_data + element_count);

    } else if (metal_dtype == "bfloat16" || metal_dtype == "bf16") {
        const metal::bfloat16_t* bf16_data = static_cast<const metal::bfloat16_t*>(metal_data);
        return convert_bf16_to_f32(bf16_data, element_count);

    } else if (metal_dtype == "float16" || metal_dtype == "f16") {
        const metal::float16_t* f16_data = static_cast<const metal::float16_t*>(metal_data);
        return convert_f16_to_f32(f16_data, element_count);

    } else {
        std::cerr << "Unsupported Metal dtype: " << metal_dtype << std::endl;
        return std::vector<float>(element_count, 0.0f); // Return zeros as fallback
    }
}

std::vector<float> DTypeValidator::convert_bf16_to_f32(const std::vector<uint16_t>& bf16_data) {
    std::vector<float> f32_data;
    f32_data.reserve(bf16_data.size());

    for (uint16_t bf16_val : bf16_data) {
        metal::bfloat16_t bf16 = static_cast<metal::bfloat16_t>(bf16_val);
        float f32 = metal::DTypeConverter::bf16_to_f32(bf16);
        f32_data.push_back(f32);
    }

    return f32_data;
}

std::vector<float> DTypeValidator::convert_bf16_to_f32(const metal::bfloat16_t* bf16_data, size_t count) {
    std::vector<float> f32_data;
    f32_data.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        float f32 = metal::DTypeConverter::bf16_to_f32(bf16_data[i]);
        f32_data.push_back(f32);
    }

    return f32_data;
}

std::vector<float> DTypeValidator::convert_f16_to_f32(const std::vector<uint16_t>& f16_data) {
    std::vector<float> f32_data;
    f32_data.reserve(f16_data.size());

    for (uint16_t f16_val : f16_data) {
        metal::float16_t f16 = static_cast<metal::float16_t>(f16_val);
        float f32 = metal::DTypeConverter::f16_to_f32(f16);
        f32_data.push_back(f32);
    }

    return f32_data;
}

std::vector<float> DTypeValidator::convert_f16_to_f32(const metal::float16_t* f16_data, size_t count) {
    std::vector<float> f32_data;
    f32_data.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        float f32 = metal::DTypeConverter::f16_to_f32(f16_data[i]);
        f32_data.push_back(f32);
    }

    return f32_data;
}

DTypeValidator::ConversionValidationResult DTypeValidator::validate_bf16_conversion_accuracy(const std::vector<float>& original_f32) {
    ConversionValidationResult result;

    if (original_f32.empty()) {
        result.passed = true;
        return result;
    }

    double sum_error = 0.0;
    result.max_conversion_error = 0.0;
    result.failed_conversions = 0;

    for (const float& f32_val : original_f32) {
        // Convert F32 -> BF16 -> F32 and measure error
        metal::bfloat16_t bf16 = metal::DTypeConverter::f32_to_bf16(f32_val);
        float recovered_f32 = metal::DTypeConverter::bf16_to_f32(bf16);

        double conversion_error = std::abs(static_cast<double>(f32_val) - static_cast<double>(recovered_f32));
        sum_error += conversion_error;
        result.max_conversion_error = std::max(result.max_conversion_error, conversion_error);

        // Check if conversion is reasonable (BF16 has ~3-4 decimal digits of precision)
        if (conversion_error > 1e-2 * std::abs(f32_val) && std::abs(f32_val) > 1e-6) {
            result.failed_conversions++;
        }
    }

    result.mean_conversion_error = sum_error / original_f32.size();

    // Consider conversion acceptable if < 1% of values have excessive error
    result.passed = (result.failed_conversions < original_f32.size() / 100) && (result.max_conversion_error < 1.0);

    if (!result.passed) {
        result.error_message = "BF16 conversion accuracy below threshold. Max error: " +
                              std::to_string(result.max_conversion_error) +
                              ", Failed conversions: " + std::to_string(result.failed_conversions);
    }

    return result;
}

DTypeValidator::ConversionValidationResult DTypeValidator::validate_f16_conversion_accuracy(const std::vector<float>& original_f32) {
    ConversionValidationResult result;

    if (original_f32.empty()) {
        result.passed = true;
        return result;
    }

    double sum_error = 0.0;
    result.max_conversion_error = 0.0;
    result.failed_conversions = 0;

    for (const float& f32_val : original_f32) {
        // Convert F32 -> F16 -> F32 and measure error
        metal::float16_t f16 = metal::DTypeConverter::f32_to_f16(f32_val);
        float recovered_f32 = metal::DTypeConverter::f16_to_f32(f16);

        double conversion_error = std::abs(static_cast<double>(f32_val) - static_cast<double>(recovered_f32));
        sum_error += conversion_error;
        result.max_conversion_error = std::max(result.max_conversion_error, conversion_error);

        // Check if conversion is reasonable (F16 has ~3-4 decimal digits of precision)
        if (conversion_error > 1e-3 * std::abs(f32_val) && std::abs(f32_val) > 1e-7) {
            result.failed_conversions++;
        }
    }

    result.mean_conversion_error = sum_error / original_f32.size();

    // Consider conversion acceptable if < 1% of values have excessive error
    result.passed = (result.failed_conversions < original_f32.size() / 100) && (result.max_conversion_error < 1.0);

    if (!result.passed) {
        result.error_message = "F16 conversion accuracy below threshold. Max error: " +
                              std::to_string(result.max_conversion_error) +
                              ", Failed conversions: " + std::to_string(result.failed_conversions);
    }

    return result;
}

DTypeValidator::FloatValidationResult DTypeValidator::validate_float_values(const std::vector<float>& data) {
    FloatValidationResult result;

    for (const float& value : data) {
        is_valid_float_value(value, result);
    }

    return result;
}

bool DTypeValidator::is_supported_dtype(const std::string& dtype) {
    return (dtype == "float32" || dtype == "f32" ||
            dtype == "bfloat16" || dtype == "bf16" ||
            dtype == "float16" || dtype == "f16");
}

std::string DTypeValidator::get_metal_dtype_name(const std::string& cuda_dtype) {
    if (cuda_dtype == "float32") return "f32";
    if (cuda_dtype == "bfloat16") return "bf16";
    if (cuda_dtype == "float16") return "f16";
    return cuda_dtype; // Return as-is if unknown
}

// Static methods
std::vector<float> DTypeValidator::static_bf16_to_f32(const metal::bfloat16_t* data, size_t count) {
    std::vector<float> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.push_back(metal::DTypeConverter::bf16_to_f32(data[i]));
    }

    return result;
}

std::vector<float> DTypeValidator::static_f16_to_f32(const metal::float16_t* data, size_t count) {
    std::vector<float> result;
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        result.push_back(metal::DTypeConverter::f16_to_f32(data[i]));
    }

    return result;
}

std::vector<metal::bfloat16_t> DTypeValidator::static_f32_to_bf16(const std::vector<float>& f32_data) {
    std::vector<metal::bfloat16_t> result;
    result.reserve(f32_data.size());

    for (float f32_val : f32_data) {
        result.push_back(metal::DTypeConverter::f32_to_bf16(f32_val));
    }

    return result;
}

std::vector<metal::float16_t> DTypeValidator::static_f32_to_f16(const std::vector<float>& f32_data) {
    std::vector<metal::float16_t> result;
    result.reserve(f32_data.size());

    for (float f32_val : f32_data) {
        result.push_back(metal::DTypeConverter::f32_to_f16(f32_val));
    }

    return result;
}

// Private methods
bool DTypeValidator::is_valid_float_value(float value, FloatValidationResult& result) const {
    if (std::isnan(value)) {
        result.has_nan = true;
        result.nan_count++;
        return false;
    }

    if (std::isinf(value)) {
        result.has_inf = true;
        result.inf_count++;
        return false;
    }

    // Check for subnormal values
    if (value != 0.0f && std::abs(value) < std::numeric_limits<float>::min()) {
        result.has_subnormal = true;
        result.subnormal_count++;
        // Subnormal values are not necessarily invalid, just flagged
    }

    return true;
}

} // namespace metal_test