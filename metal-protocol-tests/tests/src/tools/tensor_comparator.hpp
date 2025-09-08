#pragma once

#include "artifact_reader.hpp"
#include <vector>
#include <string>
#include <memory>

namespace metal_test {

// Comparison result for individual tensors
struct ComparisonResult {
    bool passed;
    double max_absolute_error;
    double mean_absolute_error;
    double root_mean_square_error;
    size_t total_elements;
    size_t mismatched_elements;

    // First mismatch details
    bool has_first_mismatch;
    std::vector<size_t> first_mismatch_indices;
    float expected_value;
    float actual_value;
    double first_mismatch_error;

    // Constructor
    ComparisonResult()
        : passed(false), max_absolute_error(0.0), mean_absolute_error(0.0),
          root_mean_square_error(0.0), total_elements(0), mismatched_elements(0),
          has_first_mismatch(false), expected_value(0.0f), actual_value(0.0f),
          first_mismatch_error(0.0) {}
};

// Comparison tolerances
struct ComparisonTolerances {
    double absolute_tolerance;
    double relative_tolerance;
    double max_allowed_mae;
    double max_allowed_rmse;

    // Default tolerances for floating point comparison
    ComparisonTolerances()
        : absolute_tolerance(1e-5), relative_tolerance(1e-4),
          max_allowed_mae(1e-4), max_allowed_rmse(1e-3) {}

    ComparisonTolerances(double abs_tol, double rel_tol, double mae_tol, double rmse_tol)
        : absolute_tolerance(abs_tol), relative_tolerance(rel_tol),
          max_allowed_mae(mae_tol), max_allowed_rmse(rmse_tol) {}
};

// Main tensor comparator class
class TensorComparator {
public:
    TensorComparator(const ComparisonTolerances& tolerances = ComparisonTolerances());
    ~TensorComparator() = default;

    // Compare two tensors
    ComparisonResult compare(const TensorData& expected, const TensorData& actual,
                           const std::string& tensor_name = "") const;

    // Compare raw float arrays with shapes
    ComparisonResult compare_raw(const std::vector<float>& expected,
                               const std::vector<float>& actual,
                               const std::vector<size_t>& shape,
                               const std::string& tensor_name = "") const;

    // Utility methods
    void set_tolerances(const ComparisonTolerances& tolerances) { tolerances_ = tolerances; }
    const ComparisonTolerances& get_tolerances() const { return tolerances_; }

    // Static utility methods
    static bool are_shapes_compatible(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2);
    static size_t compute_total_elements(const std::vector<size_t>& shape);
    static std::vector<size_t> compute_indices_from_flat(size_t flat_index, const std::vector<size_t>& shape);
    static std::string format_indices(const std::vector<size_t>& indices);

private:
    ComparisonTolerances tolerances_;

    // Internal comparison methods
    bool is_value_match(float expected, float actual) const;
    ComparisonResult perform_comparison(const std::vector<float>& expected,
                                      const std::vector<float>& actual,
                                      const std::vector<size_t>& shape,
                                      const std::string& tensor_name) const;

    // Statistics computation
    void compute_error_statistics(const std::vector<float>& expected,
                                const std::vector<float>& actual,
                                ComparisonResult& result) const;

    void find_first_mismatch(const std::vector<float>& expected,
                           const std::vector<float>& actual,
                           const std::vector<size_t>& shape,
                           ComparisonResult& result) const;
};

} // namespace metal_test