#pragma once

#include "tensor_comparator.hpp"
#include <string>
#include <vector>
#include <memory>
#include <fstream>

// Include JSON header
#include <nlohmann/json.hpp>

namespace metal_test {

// Result for a single layer step
struct LayerStepResult {
    std::string layer_name;
    std::string step_name;
    ComparisonResult comparison;
    bool executed;
    std::string error_message;

    LayerStepResult(const std::string& layer, const std::string& step)
        : layer_name(layer), step_name(step), executed(false) {}
};

// Result for an entire layer
struct LayerResult {
    std::string layer_name;
    std::vector<LayerStepResult> step_results;
    bool all_passed;
    size_t first_failed_step_index;

    LayerResult(const std::string& layer)
        : layer_name(layer), all_passed(true), first_failed_step_index(SIZE_MAX) {}
};

// Overall validation result
struct ValidationResult {
    std::vector<LayerResult> layer_results;
    LayerStepResult final_result;
    bool all_passed;
    size_t first_failed_layer_index;
    size_t total_comparisons;
    size_t passed_comparisons;
    size_t failed_comparisons;

    ValidationResult()
        : final_result("final", "logits"), all_passed(true),
          first_failed_layer_index(SIZE_MAX), total_comparisons(0),
          passed_comparisons(0), failed_comparisons(0) {}
};

// Validation reporter class
class ValidationReporter {
public:
    ValidationReporter(bool verbose = false, bool use_colors = true);
    ~ValidationReporter() = default;

    // Add layer and step results
    void add_layer_result(const LayerResult& layer_result);
    void set_final_result(const LayerStepResult& final_result);

    // Generate reports
    void print_console_report(const ValidationResult& result) const;
    void print_summary(const ValidationResult& result) const;
    void print_detailed_report(const ValidationResult& result) const;

    // Export detailed JSON report
    bool export_json_report(const ValidationResult& result, const std::string& output_file) const;

    // Create validation result from individual components
    ValidationResult create_result(const std::vector<LayerResult>& layers, const LayerStepResult& final) const;

    // Utility methods
    void set_verbose(bool verbose) { verbose_ = verbose; }
    void set_use_colors(bool use_colors) { use_colors_ = use_colors; }

private:
    bool verbose_;
    bool use_colors_;

    // Console output formatting
    std::string format_pass_fail(bool passed) const;
    std::string format_layer_header(const std::string& layer_name) const;
    std::string format_step_result(const LayerStepResult& step_result) const;
    std::string format_error_statistics(const ComparisonResult& result) const;
    std::string format_first_mismatch(const ComparisonResult& result) const;

    // Color codes
    std::string get_color_code(const std::string& color) const;
    std::string get_reset_code() const;

    // JSON export helpers
    nlohmann::json layer_result_to_json(const LayerResult& layer_result) const;
    nlohmann::json step_result_to_json(const LayerStepResult& step_result) const;
    nlohmann::json comparison_result_to_json(const ComparisonResult& comparison) const;

    // Statistics computation
    void compute_validation_statistics(ValidationResult& result) const;
    void find_first_failure(ValidationResult& result) const;
};

// Helper class for progress tracking during validation
class ValidationProgress {
public:
    ValidationProgress(size_t total_steps);
    ~ValidationProgress() = default;

    void update(size_t completed_steps, const std::string& current_operation = "");
    void complete();
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    size_t total_steps_;
    size_t completed_steps_;
    bool verbose_;
    bool completed_;

    void print_progress() const;
};

} // namespace metal_test