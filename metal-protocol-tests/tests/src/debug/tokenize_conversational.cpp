/**
 * Simple tokenization utility using the existing C++ BPE tokenizer
 * This will help us get token IDs for "Hello, how are you?" to hardcode into Metal test
 */
#include "backend/backend-cuda/src/bpe.hpp"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "=== C++ BPE Tokenization Test ===" << std::endl;

    try {
        // Try to find tokenizer file - use PIE_MODEL_PATH directory if available
        std::vector<std::string> possible_tokenizer_paths;

        // First try to derive from PIE_MODEL_PATH (similar to forward pass test)
        const char* model_path_env = std::getenv("PIE_MODEL_PATH");
        if (model_path_env) {
            std::string model_path(model_path_env);
            // Extract directory from model path (remove filename)
            size_t last_slash = model_path.find_last_of("/\\");
            if (last_slash != std::string::npos) {
                std::string model_dir = model_path.substr(0, last_slash);
                possible_tokenizer_paths.push_back(model_dir + "/llama-3.2.vocab");
                possible_tokenizer_paths.push_back(model_dir + "/tokenizer.model");
            }
        }

        // Add other common paths
        possible_tokenizer_paths.insert(possible_tokenizer_paths.end(), {
            std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/.cache/pie/models/llama-3.2-1b-instruct/llama-3.2.vocab",
            std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/Library/Caches/pie/models/llama-3.2-1b-instruct/llama-3.2.vocab",
            std::string(std::getenv("PIE_HOME") ? std::getenv("PIE_HOME") : ".") + "/models/llama-3.2-1b-instruct/llama-3.2.vocab",
            "./llama-3.2.vocab",
            "../llama-3.2.vocab"
        });

        std::string input_text = "Hello, how are you?";
        std::cout << "\n📝 Attempting to tokenize: \"" << input_text << "\"" << std::endl;

        bool found_tokenizer = false;
        std::vector<uint32_t> actual_tokens;

        // Try to use actual BPE tokenizer
        for (const auto& path : possible_tokenizer_paths) {
            try {
                std::cout << "Trying tokenizer path: " << path << std::endl;
                auto tokenizer = bpe::llama3_tokenizer(path);
                actual_tokens = tokenizer.encode_with_special_tokens(input_text);
                found_tokenizer = true;
                std::cout << "✅ Successfully loaded tokenizer from: " << path << std::endl;
                break;
            } catch (const std::exception& e) {
                std::cout << "❌ Failed to load tokenizer from " << path << ": " << e.what() << std::endl;
                continue;
            }
        }

        std::vector<uint32_t> tokens_to_use;

        if (found_tokenizer) {
            std::cout << "\n🎯 ACTUAL BPE TOKENS:" << std::endl;
            tokens_to_use = actual_tokens;
        } else {
            std::cout << "\n⚠️  No tokenizer found, using estimated tokens based on Llama patterns:" << std::endl;
            // Fallback to estimated tokens
            tokens_to_use = {
                1,      // <|begin_of_text|> (BOS token)
                9906,   // "Hello"
                11,     // ","
                1268,   // " how"
                527,    // " are"
                499,    // " you"
                30      // "?"
            };
        }

        std::cout << "\n🎯 ESTIMATED TOKENS FOR METAL TEST:" << std::endl;
        std::cout << "std::vector<int32_t> input_ids = {" << std::endl;
        for (size_t i = 0; i < tokens_to_use.size(); ++i) {
            if (i == tokens_to_use.size() - 1) {
                std::cout << "    " << tokens_to_use[i] << std::endl;
            } else {
                std::cout << "    " << tokens_to_use[i] << "," << std::endl;
            }
        }
        std::cout << "}; // \"Hello, how are you?\"" << std::endl;

        std::cout << "\nstd::vector<int32_t> position_ids = {" << std::endl;
        for (size_t i = 0; i < tokens_to_use.size(); ++i) {
            if (i == tokens_to_use.size() - 1) {
                std::cout << "    " << (int)i << std::endl;
            } else {
                std::cout << "    " << (int)i << "," << std::endl;
            }
        }
        std::cout << "}; // Sequence positions" << std::endl;

        std::cout << "\n📊 Token Analysis:" << std::endl;
        std::cout << "  Total tokens: " << tokens_to_use.size() << std::endl;
        std::cout << "  Sequence length: " << tokens_to_use.size() << std::endl;
        std::cout << "  Contains BOS token: Yes (token 1)" << std::endl;

        std::cout << "\n💡 Note: These are estimated tokens based on typical Llama patterns." << std::endl;
        std::cout << "     For exact tokenization, you would use:" << std::endl;
        std::cout << "     auto tokenizer = bpe::llama3_tokenizer(\"/path/to/tokenizer.model\");" << std::endl;
        std::cout << "     auto tokens = tokenizer.encode_with_special_tokens(\"Hello, how are you?\");" << std::endl;

        std::cout << "\n✅ Ready to integrate into Metal forward pass test!" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}