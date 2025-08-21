// Per-op Metal wrapper for Softmax
#include <cstdint>
#include <iostream>
#include <vector>
#include <random>
#include <sstream>

#include "ops.hpp"
#include "artifacts.hpp"
#include "metal_softmax.hpp"

namespace ops {

void run_softmax_metal(const std::string& case_id, const SoftmaxConfig& cfg, uint64_t seed) {
    using T = float;  // Use float to match CUDA FlashInfer OnlineSoftmax

    const int batch_size = cfg.batch_size;
    const int vocab_size = cfg.vocab_size;
    const float temperature = cfg.temperature;

    std::cout << "Running Metal Softmax: batch_size=" << batch_size << ", vocab_size=" << vocab_size
              << ", temperature=" << temperature << std::endl;

    // Generate same test data as CUDA version
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);  // Reasonable logit range

    const size_t logits_size = static_cast<size_t>(batch_size) * vocab_size;
    std::vector<T> h_input_logits(logits_size);
    std::vector<T> h_output(logits_size, 0);

    for (auto& v : h_input_logits) v = dist(rng);

    // Call Metal softmax implementation
    int result = metal_softmax_float(
        h_input_logits.data(),
        h_output.data(),
        batch_size,
        vocab_size,
        temperature
    );

    if (result != 0) {
        std::cerr << "Metal softmax failed with error: " << result << std::endl;
        return;
    }

    // Generate artifacts for comparison with CUDA
    if (artifacts::op_enabled("softmax")) {
        auto dir = artifacts::ensure_dir_for_case("softmax", case_id + "_metal");

        artifacts::write_vector_bin(dir, "input_logits", h_input_logits);
        artifacts::write_vector_bin(dir, "output", h_output);

        std::ostringstream meta;
        meta << "\"version\": \"1\",\n"
             << "\"op\": \"softmax\",\n"
             << "\"backend\": \"metal\",\n"
             << "\"case_id\": " << artifacts::json_escape(case_id) << ",\n"
             << "\"config\": {\"batch_size\": " << batch_size
             << ", \"vocab_size\": " << vocab_size
             << ", \"temperature\": " << temperature << "},\n"
             << "\"dtype_map\": {\"input_logits\": \"fp32\", \"output\": \"fp32\"},\n"
             << "\"shape_map\": {\"input_logits\": [" << batch_size << ", " << vocab_size
             << "], \"output\": [" << batch_size << ", " << vocab_size << "]}";
        artifacts::write_meta_json(dir, meta.str());
    }

    std::cout << "Metal Softmax completed successfully" << std::endl;
}

} // namespace ops
