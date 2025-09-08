#include "artifact_reader.hpp"
#include "tensor_comparator.hpp"
#include "dtype_validator.hpp"
#include "validation_reporter.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <getopt.h>

// Use header-only JSON library
#include <nlohmann/json.hpp>

namespace metal_test {

// Configuration for validation run
struct ValidationConfig {
    std::string cuda_artifacts_path;
    std::string test_case_id = "real_model_forward_pass";
    bool verbose = false;
    bool export_json = false;
    std::string json_output_file = "validation_report.json";
    ComparisonTolerances tolerances;

    ValidationConfig() {
        // Default tolerances - more relaxed for cross-platform comparison
        tolerances.absolute_tolerance = 1e-4;
        tolerances.relative_tolerance = 1e-3;
        tolerances.max_allowed_mae = 1e-3;
        tolerances.max_allowed_rmse = 1e-2;
    }
};

// Main validation class
class ForwardPassValidator {
public:
    ForwardPassValidator(const ValidationConfig& config)
        : config_(config), comparator_(config.tolerances), reporter_(config.verbose) {
    }

    bool run_validation() {
        std::cout << "=== Metal Protocol Test: CUDA Artifact Validation ===" << std::endl;
        std::cout << "Loading CUDA artifacts from: " << config_.cuda_artifacts_path << std::endl;

        // Initialize artifact reader
        ArtifactReader reader(config_.cuda_artifacts_path);
        if (!reader.is_valid()) {
            std::cerr << "Failed to initialize artifact reader" << std::endl;
            return false;
        }

        auto collection_metadata = reader.get_collection_metadata();
        std::cout << "Model config: " << collection_metadata.num_layers << " layers, "
                  << "hidden_size=" << collection_metadata.hidden_size << ", "
                  << "seq_len=" << collection_metadata.sequence_length << std::endl;

        // Set up validation progress tracking
        size_t total_steps = collection_metadata.num_layers * collection_metadata.steps_per_layer.size() +
                           collection_metadata.final_steps.size();
        ValidationProgress progress(total_steps);
        progress.set_verbose(config_.verbose);

        // Validate each layer
        std::vector<LayerResult> layer_results;
        for (int layer_idx = 0; layer_idx < collection_metadata.num_layers; ++layer_idx) {
            std::string layer_name = "layer_" + std::to_string(layer_idx);

            LayerResult layer_result(layer_name);
            validate_layer(reader, layer_name, collection_metadata.steps_per_layer, layer_result, progress);
            layer_results.push_back(std::move(layer_result));
        }

        // Validate final steps (logits)
        LayerStepResult final_result("final", "logits");
        validate_final_step(reader, "logits", final_result, progress);

        // Create and display results
        ValidationResult validation_result = reporter_.create_result(layer_results, final_result);
        reporter_.print_console_report(validation_result);

        // Export JSON report if requested
        if (config_.export_json) {
            reporter_.export_json_report(validation_result, config_.json_output_file);
        }

        return validation_result.all_passed;
    }

private:
    ValidationConfig config_;
    TensorComparator comparator_;
    ValidationReporter reporter_;

    void validate_layer(ArtifactReader& reader, const std::string& layer_name,
                       const std::vector<std::string>& steps, LayerResult& layer_result,
                       ValidationProgress& progress) {

        layer_result.all_passed = true;
        layer_result.first_failed_step_index = SIZE_MAX;

        for (size_t step_idx = 0; step_idx < steps.size(); ++step_idx) {
            const std::string& step_name = steps[step_idx];
            LayerStepResult step_result(layer_name, step_name);

            progress.update(0, "Validating " + layer_name + "::" + step_name);

            // Check if artifact exists
            if (!reader.has_artifact(layer_name, step_name)) {
                step_result.executed = false;
                step_result.error_message = "Artifact not found";
                layer_result.step_results.push_back(std::move(step_result));
                continue;
            }

            // Load CUDA reference tensor
            auto cuda_tensor = reader.load_tensor(layer_name, step_name);
            if (!cuda_tensor) {
                step_result.executed = false;
                step_result.error_message = "Failed to load CUDA artifact";
                layer_result.step_results.push_back(std::move(step_result));
                continue;
            }

            // Validate tensor data
            DTypeValidator dtype_validator;
            auto float_validation = dtype_validator.validate_float_values(cuda_tensor->get_data());
            if (float_validation.has_nan || float_validation.has_inf) {
                step_result.executed = false;
                step_result.error_message = "CUDA tensor contains invalid values (NaN/Inf)";
                layer_result.step_results.push_back(std::move(step_result));
                continue;
            }

            // For now, simulate Metal tensor output (placeholder)
            // In a real implementation, this would execute the Metal operation and get results
            std::vector<float> simulated_metal_output = simulate_metal_operation(layer_name, step_name, cuda_tensor->get_shape());

            // Compare tensors
            step_result.comparison = comparator_.compare_raw(
                cuda_tensor->get_data(),
                simulated_metal_output,
                cuda_tensor->get_shape(),
                layer_name + "::" + step_name
            );
            step_result.executed = true;

            // Update layer result
            if (!step_result.comparison.passed) {
                layer_result.all_passed = false;
                if (layer_result.first_failed_step_index == SIZE_MAX) {
                    layer_result.first_failed_step_index = step_idx;
                }
            }

            layer_result.step_results.push_back(std::move(step_result));
        }
    }

    void validate_final_step(ArtifactReader& reader, const std::string& step_name,
                           LayerStepResult& step_result, ValidationProgress& progress) {

        progress.update(0, "Validating final::" + step_name);

        // Check if final artifact exists
        if (!reader.has_final_artifact(step_name)) {
            step_result.executed = false;
            step_result.error_message = "Final artifact not found";
            return;
        }

        // Load CUDA reference tensor
        auto cuda_tensor = reader.load_final_tensor(step_name);
        if (!cuda_tensor) {
            step_result.executed = false;
            step_result.error_message = "Failed to load final CUDA artifact";
            return;
        }

        // Validate tensor data
        DTypeValidator dtype_validator;
        auto float_validation = dtype_validator.validate_float_values(cuda_tensor->get_data());
        if (float_validation.has_nan || float_validation.has_inf) {
            step_result.executed = false;
            step_result.error_message = "Final CUDA tensor contains invalid values (NaN/Inf)";
            return;
        }

        // Simulate Metal final output (placeholder)
        std::vector<float> simulated_metal_output = simulate_metal_operation("final", step_name, cuda_tensor->get_shape());

        // Compare tensors
        step_result.comparison = comparator_.compare_raw(
            cuda_tensor->get_data(),
            simulated_metal_output,
            cuda_tensor->get_shape(),
            "final::" + step_name
        );
        step_result.executed = true;
    }

    // Placeholder method - in real implementation this would execute Metal operations
    std::vector<float> simulate_metal_operation(const std::string& layer_name, const std::string& step_name,
                                              const std::vector<size_t>& shape) {

        size_t total_elements = TensorComparator::compute_total_elements(shape);
        std::vector<float> output(total_elements);

        // Simulate some realistic-looking output with small errors
        // This is just for demonstration - real implementation would run Metal kernels
        for (size_t i = 0; i < total_elements; ++i) {
            float base_value = 0.1f * (std::hash<std::string>{}(layer_name + step_name) % 1000) / 1000.0f;
            float element_variation = 0.01f * (i % 100) / 100.0f;

            // Add small numerical error to simulate cross-platform differences
            float error = 1e-5f * ((i * 7 + 13) % 200 - 100) / 100.0f;

            output[i] = base_value + element_variation + error;
        }

        return output;
    }
};

// Command line parsing
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -p, --path PATH          CUDA artifacts path (required)" << std::endl;
    std::cout << "  -c, --case CASE_ID       Test case ID (default: real_model_forward_pass)" << std::endl;
    std::cout << "  -v, --verbose            Enable verbose output" << std::endl;
    std::cout << "  -j, --json [FILE]        Export JSON report (default: validation_report.json)" << std::endl;
    std::cout << "  --abs-tol TOLERANCE      Absolute tolerance (default: 1e-4)" << std::endl;
    std::cout << "  --rel-tol TOLERANCE      Relative tolerance (default: 1e-3)" << std::endl;
    std::cout << "  --mae-tol TOLERANCE      Max allowed MAE (default: 1e-3)" << std::endl;
    std::cout << "  --rmse-tol TOLERANCE     Max allowed RMSE (default: 1e-2)" << std::endl;
    std::cout << "  -h, --help               Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " -p /path/to/cuda-protocol-tests/tests/artifacts/forward_pass_integration/real_model_forward_pass" << std::endl;
    std::cout << "  " << program_name << " -p /path/to/artifacts -v -j detailed_report.json" << std::endl;
}

bool parse_command_line(int argc, char* argv[], ValidationConfig& config) {
    static struct option long_options[] = {
        {"path",     required_argument, 0, 'p'},
        {"case",     required_argument, 0, 'c'},
        {"verbose",  no_argument,       0, 'v'},
        {"json",     optional_argument, 0, 'j'},
        {"abs-tol",  required_argument, 0, 1001},
        {"rel-tol",  required_argument, 0, 1002},
        {"mae-tol",  required_argument, 0, 1003},
        {"rmse-tol", required_argument, 0, 1004},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "p:c:vj::h", long_options, &option_index)) != -1) {
        switch (c) {
            case 'p':
                config.cuda_artifacts_path = optarg;
                break;
            case 'c':
                config.test_case_id = optarg;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'j':
                config.export_json = true;
                if (optarg) {
                    config.json_output_file = optarg;
                }
                break;
            case 1001:
                config.tolerances.absolute_tolerance = std::stod(optarg);
                break;
            case 1002:
                config.tolerances.relative_tolerance = std::stod(optarg);
                break;
            case 1003:
                config.tolerances.max_allowed_mae = std::stod(optarg);
                break;
            case 1004:
                config.tolerances.max_allowed_rmse = std::stod(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return false;
            case '?':
                print_usage(argv[0]);
                return false;
            default:
                break;
        }
    }

    // Validate required arguments
    if (config.cuda_artifacts_path.empty()) {
        std::cerr << "Error: CUDA artifacts path is required (-p/--path)" << std::endl;
        print_usage(argv[0]);
        return false;
    }

    return true;
}

} // namespace metal_test

int main(int argc, char* argv[]) {
    metal_test::ValidationConfig config;

    // Parse command line arguments
    if (!metal_test::parse_command_line(argc, argv, config)) {
        return 1;
    }

    // Run validation
    try {
        metal_test::ForwardPassValidator validator(config);
        bool success = validator.run_validation();

        return success ? 0 : 1;

    } catch (const std::exception& e) {
        std::cerr << "Validation failed with exception: " << e.what() << std::endl;
        return 1;
    }
}