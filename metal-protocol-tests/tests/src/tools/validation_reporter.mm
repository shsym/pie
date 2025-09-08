#include "validation_reporter.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>

// Use header-only JSON library
#include <nlohmann/json.hpp>

namespace metal_test {

ValidationReporter::ValidationReporter(bool verbose, bool use_colors)
    : verbose_(verbose), use_colors_(use_colors) {
}

void ValidationReporter::add_layer_result(const LayerResult& layer_result) {
    // This would be used if we were accumulating results, but we'll use create_result instead
}

void ValidationReporter::set_final_result(const LayerStepResult& final_result) {
    // This would be used if we were accumulating results, but we'll use create_result instead
}

ValidationResult ValidationReporter::create_result(const std::vector<LayerResult>& layers,
                                                 const LayerStepResult& final) const {
    ValidationResult result;
    result.layer_results = layers;
    result.final_result = final;

    compute_validation_statistics(result);
    find_first_failure(result);

    return result;
}

void ValidationReporter::print_console_report(const ValidationResult& result) const {
    print_summary(result);

    if (verbose_ || !result.all_passed) {
        std::cout << std::endl;
        print_detailed_report(result);
    }
}

void ValidationReporter::print_summary(const ValidationResult& result) const {
    std::cout << std::endl;
    std::cout << get_color_code("bold") << "=== Metal Protocol Test: CUDA Artifact Validation Summary ==="
              << get_reset_code() << std::endl;

    std::cout << "Total Comparisons: " << result.total_comparisons << std::endl;
    std::cout << "Passed: " << get_color_code("green") << result.passed_comparisons << get_reset_code();
    std::cout << ", Failed: " << get_color_code("red") << result.failed_comparisons << get_reset_code() << std::endl;

    std::cout << "Overall Result: " << format_pass_fail(result.all_passed) << std::endl;

    if (!result.all_passed && result.first_failed_layer_index != SIZE_MAX) {
        const auto& first_failed_layer = result.layer_results[result.first_failed_layer_index];
        std::cout << get_color_code("yellow") << "First failure detected at: "
                  << first_failed_layer.layer_name;

        if (first_failed_layer.first_failed_step_index != SIZE_MAX) {
            const auto& first_failed_step = first_failed_layer.step_results[first_failed_layer.first_failed_step_index];
            std::cout << ", step: " << first_failed_step.step_name;
        }
        std::cout << get_reset_code() << std::endl;
    }
}

void ValidationReporter::print_detailed_report(const ValidationResult& result) const {
    std::cout << get_color_code("bold") << "=== Detailed Validation Report ===" << get_reset_code() << std::endl;

    // Print layer results
    for (const auto& layer_result : result.layer_results) {
        std::cout << format_layer_header(layer_result.layer_name) << std::endl;

        for (const auto& step_result : layer_result.step_results) {
            std::cout << format_step_result(step_result) << std::endl;

            if (verbose_ || !step_result.comparison.passed) {
                if (step_result.executed && !step_result.comparison.passed) {
                    std::cout << format_error_statistics(step_result.comparison);
                    if (step_result.comparison.has_first_mismatch) {
                        std::cout << format_first_mismatch(step_result.comparison);
                    }
                    std::cout << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }

    // Print final result
    if (result.final_result.executed) {
        std::cout << format_layer_header("Final") << std::endl;
        std::cout << format_step_result(result.final_result) << std::endl;

        if (verbose_ || !result.final_result.comparison.passed) {
            if (!result.final_result.comparison.passed) {
                std::cout << format_error_statistics(result.final_result.comparison);
                if (result.final_result.comparison.has_first_mismatch) {
                    std::cout << format_first_mismatch(result.final_result.comparison);
                }
            }
        }
        std::cout << std::endl;
    }
}

bool ValidationReporter::export_json_report(const ValidationResult& result, const std::string& output_file) const {
    try {
        nlohmann::json report;

        // Overall statistics
        report["summary"] = {
            {"all_passed", result.all_passed},
            {"total_comparisons", result.total_comparisons},
            {"passed_comparisons", result.passed_comparisons},
            {"failed_comparisons", result.failed_comparisons}
        };

        // First failure information
        if (!result.all_passed && result.first_failed_layer_index != SIZE_MAX) {
            const auto& first_failed_layer = result.layer_results[result.first_failed_layer_index];
            report["first_failure"] = {
                {"layer_name", first_failed_layer.layer_name},
                {"layer_index", result.first_failed_layer_index}
            };

            if (first_failed_layer.first_failed_step_index != SIZE_MAX) {
                const auto& first_failed_step = first_failed_layer.step_results[first_failed_layer.first_failed_step_index];
                report["first_failure"]["step_name"] = first_failed_step.step_name;
                report["first_failure"]["step_index"] = first_failed_layer.first_failed_step_index;
            }
        }

        // Layer results
        nlohmann::json layers = nlohmann::json::array();
        for (const auto& layer_result : result.layer_results) {
            layers.push_back(layer_result_to_json(layer_result));
        }
        report["layers"] = layers;

        // Final result
        if (result.final_result.executed) {
            report["final"] = step_result_to_json(result.final_result);
        }

        // Write to file
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open output file: " << output_file << std::endl;
            return false;
        }

        file << report.dump(2);  // Pretty print with 2-space indent
        file.close();

        std::cout << "Detailed JSON report exported to: " << output_file << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error exporting JSON report: " << e.what() << std::endl;
        return false;
    }
}

// Private methods - formatting
std::string ValidationReporter::format_pass_fail(bool passed) const {
    if (passed) {
        return get_color_code("green") + "✅ PASS" + get_reset_code();
    } else {
        return get_color_code("red") + "❌ FAIL" + get_reset_code();
    }
}

std::string ValidationReporter::format_layer_header(const std::string& layer_name) const {
    return get_color_code("cyan") + layer_name + ":" + get_reset_code();
}

std::string ValidationReporter::format_step_result(const LayerStepResult& step_result) const {
    std::ostringstream oss;

    oss << "  " << std::setw(20) << std::left << step_result.step_name << ": ";

    if (!step_result.executed) {
        oss << get_color_code("yellow") << "SKIPPED";
        if (!step_result.error_message.empty()) {
            oss << " (" << step_result.error_message << ")";
        }
        oss << get_reset_code();
    } else {
        oss << format_pass_fail(step_result.comparison.passed);

        if (step_result.comparison.passed) {
            oss << " (MAE: " << std::scientific << std::setprecision(2)
                << step_result.comparison.mean_absolute_error << ")";
        } else {
            oss << " (MAE: " << std::scientific << std::setprecision(2)
                << step_result.comparison.mean_absolute_error
                << ", Max: " << step_result.comparison.max_absolute_error << ")";
        }
    }

    return oss.str();
}

std::string ValidationReporter::format_error_statistics(const ComparisonResult& result) const {
    std::ostringstream oss;

    oss << "    Error Statistics:" << std::endl;
    oss << "      Mean Absolute Error: " << std::scientific << std::setprecision(4)
        << result.mean_absolute_error << std::endl;
    oss << "      Max Absolute Error:  " << std::scientific << std::setprecision(4)
        << result.max_absolute_error << std::endl;
    oss << "      Root Mean Square Error: " << std::scientific << std::setprecision(4)
        << result.root_mean_square_error << std::endl;
    oss << "      Mismatched Elements: " << result.mismatched_elements
        << " / " << result.total_elements
        << " (" << std::fixed << std::setprecision(2)
        << (100.0 * result.mismatched_elements / result.total_elements) << "%)" << std::endl;

    return oss.str();
}

std::string ValidationReporter::format_first_mismatch(const ComparisonResult& result) const {
    std::ostringstream oss;

    oss << "    " << get_color_code("red") << "First Mismatch:" << get_reset_code() << std::endl;
    oss << "      Index: " << TensorComparator::format_indices(result.first_mismatch_indices) << std::endl;
    oss << "      Expected: " << std::fixed << std::setprecision(6) << result.expected_value << std::endl;
    oss << "      Actual:   " << std::fixed << std::setprecision(6) << result.actual_value << std::endl;
    oss << "      Error:    " << std::scientific << std::setprecision(4) << result.first_mismatch_error << std::endl;

    return oss.str();
}

std::string ValidationReporter::get_color_code(const std::string& color) const {
    if (!use_colors_) {
        return "";
    }

    if (color == "red") return "\033[31m";
    if (color == "green") return "\033[32m";
    if (color == "yellow") return "\033[33m";
    if (color == "cyan") return "\033[36m";
    if (color == "bold") return "\033[1m";

    return "";
}

std::string ValidationReporter::get_reset_code() const {
    return use_colors_ ? "\033[0m" : "";
}

// JSON export helpers
nlohmann::json ValidationReporter::layer_result_to_json(const LayerResult& layer_result) const {
    nlohmann::json layer_json;

    layer_json["layer_name"] = layer_result.layer_name;
    layer_json["all_passed"] = layer_result.all_passed;

    if (layer_result.first_failed_step_index != SIZE_MAX) {
        layer_json["first_failed_step_index"] = layer_result.first_failed_step_index;
    }

    nlohmann::json steps = nlohmann::json::array();
    for (const auto& step_result : layer_result.step_results) {
        steps.push_back(step_result_to_json(step_result));
    }
    layer_json["steps"] = steps;

    return layer_json;
}

nlohmann::json ValidationReporter::step_result_to_json(const LayerStepResult& step_result) const {
    nlohmann::json step_json;

    step_json["step_name"] = step_result.step_name;
    step_json["executed"] = step_result.executed;

    if (!step_result.error_message.empty()) {
        step_json["error_message"] = step_result.error_message;
    }

    if (step_result.executed) {
        step_json["comparison"] = comparison_result_to_json(step_result.comparison);
    }

    return step_json;
}

nlohmann::json ValidationReporter::comparison_result_to_json(const ComparisonResult& comparison) const {
    nlohmann::json comp_json;

    comp_json["passed"] = comparison.passed;
    comp_json["max_absolute_error"] = comparison.max_absolute_error;
    comp_json["mean_absolute_error"] = comparison.mean_absolute_error;
    comp_json["root_mean_square_error"] = comparison.root_mean_square_error;
    comp_json["total_elements"] = comparison.total_elements;
    comp_json["mismatched_elements"] = comparison.mismatched_elements;

    if (comparison.has_first_mismatch) {
        comp_json["first_mismatch"] = {
            {"indices", comparison.first_mismatch_indices},
            {"expected_value", comparison.expected_value},
            {"actual_value", comparison.actual_value},
            {"error", comparison.first_mismatch_error}
        };
    }

    return comp_json;
}

// Statistics computation
void ValidationReporter::compute_validation_statistics(ValidationResult& result) const {
    result.total_comparisons = 0;
    result.passed_comparisons = 0;
    result.failed_comparisons = 0;

    // Count layer step results
    for (const auto& layer_result : result.layer_results) {
        for (const auto& step_result : layer_result.step_results) {
            if (step_result.executed) {
                result.total_comparisons++;
                if (step_result.comparison.passed) {
                    result.passed_comparisons++;
                } else {
                    result.failed_comparisons++;
                }
            }
        }
    }

    // Count final result
    if (result.final_result.executed) {
        result.total_comparisons++;
        if (result.final_result.comparison.passed) {
            result.passed_comparisons++;
        } else {
            result.failed_comparisons++;
        }
    }

    // Determine overall pass/fail
    result.all_passed = (result.failed_comparisons == 0) && (result.total_comparisons > 0);
}

void ValidationReporter::find_first_failure(ValidationResult& result) const {
    result.first_failed_layer_index = SIZE_MAX;

    for (size_t layer_idx = 0; layer_idx < result.layer_results.size(); ++layer_idx) {
        if (!result.layer_results[layer_idx].all_passed) {
            result.first_failed_layer_index = layer_idx;
            break;
        }
    }
}

// ValidationProgress implementation
ValidationProgress::ValidationProgress(size_t total_steps)
    : total_steps_(total_steps), completed_steps_(0), verbose_(false), completed_(false) {
}

void ValidationProgress::update(size_t completed_steps, const std::string& current_operation) {
    completed_steps_ = std::min(completed_steps, total_steps_);

    if (verbose_) {
        print_progress();
        if (!current_operation.empty()) {
            std::cout << " - " << current_operation;
        }
        std::cout << std::endl;
    }
}

void ValidationProgress::complete() {
    completed_steps_ = total_steps_;
    completed_ = true;

    if (verbose_) {
        std::cout << "Validation completed." << std::endl;
    }
}

void ValidationProgress::print_progress() const {
    if (total_steps_ == 0) {
        std::cout << "Progress: 0%";
        return;
    }

    double percentage = (100.0 * completed_steps_) / total_steps_;
    std::cout << "Progress: " << std::fixed << std::setprecision(1)
              << percentage << "% (" << completed_steps_ << "/" << total_steps_ << ")";
}

} // namespace metal_test