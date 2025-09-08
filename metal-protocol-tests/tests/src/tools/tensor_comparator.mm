#include "tensor_comparator.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <numeric>

namespace metal_test {

TensorComparator::TensorComparator(const ComparisonTolerances& tolerances)
    : tolerances_(tolerances) {
}

ComparisonResult TensorComparator::compare(const TensorData& expected, const TensorData& actual,
                                         const std::string& tensor_name) const {
    ComparisonResult result;

    // Check shape compatibility
    if (!are_shapes_compatible(expected.get_shape(), actual.get_shape())) {
        std::cerr << "Shape mismatch for tensor '" << tensor_name << "':" << std::endl;
        std::cerr << "  Expected shape: [";
        for (size_t i = 0; i < expected.get_shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << expected.get_shape()[i];
        }
        std::cerr << "]" << std::endl;
        std::cerr << "  Actual shape: [";
        for (size_t i = 0; i < actual.get_shape().size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << actual.get_shape()[i];
        }
        std::cerr << "]" << std::endl;
        return result; // passed = false
    }

    return perform_comparison(expected.get_data(), actual.get_data(),
                            expected.get_shape(), tensor_name);
}

ComparisonResult TensorComparator::compare_raw(const std::vector<float>& expected,
                                             const std::vector<float>& actual,
                                             const std::vector<size_t>& shape,
                                             const std::string& tensor_name) const {
    // Verify size consistency
    size_t expected_elements = compute_total_elements(shape);
    if (expected.size() != expected_elements || actual.size() != expected_elements) {
        std::cerr << "Size mismatch for tensor '" << tensor_name << "':" << std::endl;
        std::cerr << "  Expected elements from shape: " << expected_elements << std::endl;
        std::cerr << "  Expected vector size: " << expected.size() << std::endl;
        std::cerr << "  Actual vector size: " << actual.size() << std::endl;
        return ComparisonResult(); // passed = false
    }

    return perform_comparison(expected, actual, shape, tensor_name);
}

// Static utility methods
bool TensorComparator::are_shapes_compatible(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }

    for (size_t i = 0; i < shape1.size(); ++i) {
        if (shape1[i] != shape2[i]) {
            return false;
        }
    }

    return true;
}

size_t TensorComparator::compute_total_elements(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    size_t total = 1;
    for (size_t dim : shape) {
        total *= dim;
    }
    return total;
}

std::vector<size_t> TensorComparator::compute_indices_from_flat(size_t flat_index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());

    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        indices[i] = flat_index % shape[i];
        flat_index /= shape[i];
    }

    return indices;
}

std::string TensorComparator::format_indices(const std::vector<size_t>& indices) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < indices.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << indices[i];
    }
    oss << "]";
    return oss.str();
}

// Private methods
bool TensorComparator::is_value_match(float expected, float actual) const {
    // Handle special floating point values
    if (std::isnan(expected) && std::isnan(actual)) {
        return true;
    }
    if (std::isnan(expected) || std::isnan(actual)) {
        return false;
    }
    if (std::isinf(expected) && std::isinf(actual)) {
        return (std::signbit(expected) == std::signbit(actual));
    }
    if (std::isinf(expected) || std::isinf(actual)) {
        return false;
    }

    // Compute absolute and relative errors
    double abs_error = std::abs(static_cast<double>(expected) - static_cast<double>(actual));
    double abs_expected = std::abs(static_cast<double>(expected));
    double rel_error = (abs_expected > 1e-15) ? (abs_error / abs_expected) : abs_error;

    // Check tolerances
    return (abs_error <= tolerances_.absolute_tolerance) || (rel_error <= tolerances_.relative_tolerance);
}

ComparisonResult TensorComparator::perform_comparison(const std::vector<float>& expected,
                                                    const std::vector<float>& actual,
                                                    const std::vector<size_t>& shape,
                                                    const std::string& tensor_name) const {
    ComparisonResult result;
    result.total_elements = expected.size();

    if (expected.size() != actual.size()) {
        std::cerr << "Vector size mismatch for tensor '" << tensor_name << "'" << std::endl;
        return result; // passed = false
    }

    // Compute error statistics
    compute_error_statistics(expected, actual, result);

    // Find first mismatch
    find_first_mismatch(expected, actual, shape, result);

    // Determine if comparison passed
    result.passed = (result.mean_absolute_error <= tolerances_.max_allowed_mae) &&
                    (result.root_mean_square_error <= tolerances_.max_allowed_rmse) &&
                    (result.mismatched_elements == 0);

    return result;
}

void TensorComparator::compute_error_statistics(const std::vector<float>& expected,
                                              const std::vector<float>& actual,
                                              ComparisonResult& result) const {
    if (expected.empty()) {
        return;
    }

    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    result.max_absolute_error = 0.0;
    result.mismatched_elements = 0;

    for (size_t i = 0; i < expected.size(); ++i) {
        double abs_error = std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
        double sq_error = abs_error * abs_error;

        sum_abs_error += abs_error;
        sum_sq_error += sq_error;
        result.max_absolute_error = std::max(result.max_absolute_error, abs_error);

        if (!is_value_match(expected[i], actual[i])) {
            result.mismatched_elements++;
        }
    }

    result.mean_absolute_error = sum_abs_error / expected.size();
    result.root_mean_square_error = std::sqrt(sum_sq_error / expected.size());
}

void TensorComparator::find_first_mismatch(const std::vector<float>& expected,
                                         const std::vector<float>& actual,
                                         const std::vector<size_t>& shape,
                                         ComparisonResult& result) const {
    result.has_first_mismatch = false;

    for (size_t i = 0; i < expected.size(); ++i) {
        if (!is_value_match(expected[i], actual[i])) {
            result.has_first_mismatch = true;
            result.first_mismatch_indices = compute_indices_from_flat(i, shape);
            result.expected_value = expected[i];
            result.actual_value = actual[i];
            result.first_mismatch_error = std::abs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
            break;
        }
    }
}

} // namespace metal_test