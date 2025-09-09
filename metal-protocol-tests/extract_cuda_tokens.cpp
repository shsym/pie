#include "../backend/backend-cuda/src/bpe.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    try {
        std::string text = "The weather today is sunny";

        // Find tokenizer path (same logic as CUDA test)
        std::string tokenizer_path;
        const char* model_path_env = std::getenv("PIE_MODEL_PATH");
        if (model_path_env) {
            std::string model_path_str(model_path_env);
            size_t last_slash = model_path_str.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string model_dir = model_path_str.substr(0, last_slash);
                tokenizer_path = model_dir + "/llama-3.2.vocab";
            }
        }

        if (tokenizer_path.empty()) {
            std::cerr << "Could not find tokenizer path" << std::endl;
            return 1;
        }

        std::cout << "Using tokenizer: " << tokenizer_path << std::endl;
        std::cout << "Input text: \"" << text << "\"" << std::endl;

        // Load tokenizer (simplified - would need full BPE initialization)
        std::cout << "Text length: " << text.length() << " characters" << std::endl;
        std::cout << "Expected ~5 tokens based on CUDA artifacts" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}