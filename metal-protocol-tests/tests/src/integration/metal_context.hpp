#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "metal_l4ma.hpp"
#include "metal_buffer.hpp"
#include "metal_kv_cache.hpp"
#include "bpe.hpp"

/**
 * @brief Metal Context Management Class
 *
 * This class mirrors the functionality of the CUDA CudaContext class,
 * providing proper KV cache management, token tracking, and autoregressive generation
 * for Metal-based transformer models.
 *
 * Key features mirrored from CUDA implementation:
 * - Proper stateful KV cache page management using MetalL4maKVCache
 * - Token and position ID tracking across generation steps
 * - Modular design for reusable generation logic
 * - Compatible with Metal buffer management and MetalL4maForCausalLM
 */
class MetalGenerationContext {
public:
    /**
     * @brief Distribution result from model forward pass
     */
    struct Distribution {
        std::vector<uint32_t> token_ids;
        std::vector<float> probabilities;
    };

    /**
     * @brief Sampler interface for token selection
     */
    class Sampler {
    public:
        virtual ~Sampler() = default;
        virtual uint32_t sample(const std::vector<uint32_t>& token_ids,
                               const std::vector<float>& probabilities) = 0;
    };

    /**
     * @brief Greedy sampler - always selects highest probability token
     */
    class GreedySampler : public Sampler {
    public:
        uint32_t sample(const std::vector<uint32_t>& token_ids,
                       const std::vector<float>& probabilities) override;
    };

    /**
     * @brief Stop condition interface for generation control
     */
    class StopCondition {
    public:
        virtual ~StopCondition() = default;
        virtual bool should_stop(const std::vector<uint32_t>& generated_tokens) = 0;
    };

    /**
     * @brief Length-based stop condition
     */
    class LengthStopCondition : public StopCondition {
    private:
        size_t max_length;
    public:
        explicit LengthStopCondition(size_t max_len) : max_length(max_len) {}
        bool should_stop(const std::vector<uint32_t>& generated_tokens) override {
            return generated_tokens.size() >= max_length;
        }
    };

    /**
     * @brief Token-based stop condition (e.g., for EOS tokens)
     */
    class TokenStopCondition : public StopCondition {
    private:
        std::vector<uint32_t> stop_tokens;
    public:
        explicit TokenStopCondition(const std::vector<uint32_t>& tokens) : stop_tokens(tokens) {}
        bool should_stop(const std::vector<uint32_t>& generated_tokens) override {
            if (generated_tokens.empty()) return false;
            uint32_t last_token = generated_tokens.back();
            return std::find(stop_tokens.begin(), stop_tokens.end(), last_token) != stop_tokens.end();
        }
    };

private:
    // Model and tokenizer
    std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> model_;
    std::string tokenizer_path_;

    // Token state management (mirrors CUDA Context)
    std::vector<uint32_t> token_ids_;           // Committed tokens (in KV cache)
    std::vector<uint32_t> token_ids_pending_;   // Tokens to be processed
    std::vector<uint32_t> position_ids_;        // Position IDs for all committed tokens

    // KV cache page management (mirrors CUDA Context but uses Metal KV cache)
    std::unique_ptr<MetalL4maKVCache<bfloat16_t>> kv_cache_;
    std::vector<uint32_t> kv_page_ids_;         // Allocated KV page IDs (global unique IDs)
    std::vector<uint32_t> kv_buffer_indices_;   // Buffer indices (0, 1, 2, ...) for Metal kernel
    uint32_t kv_page_last_len_;                 // Number of tokens in last page
    uint32_t kv_page_size_;                     // Tokens per KV page

    // Metal buffer management
    std::unique_ptr<MetalL4maBuffer<bfloat16_t>> buffer_;
    std::unique_ptr<PersistentMemoryPool> memory_pool_;

    // Configuration
    L4maConfig config_;
    bool verbose_;

public:
    /**
     * @brief Constructor
     * @param model Unique pointer to initialized Metal model
     * @param tokenizer_path Path to BPE tokenizer file
     * @param kv_page_size Number of tokens per KV cache page (default: 16)
     */
    MetalGenerationContext(std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> model,
                          const std::string& tokenizer_path,
                          uint32_t kv_page_size = 16);

    /**
     * @brief Destructor - handles KV page cleanup
     */
    ~MetalGenerationContext();

    // --- Token Management (mirrors CUDA Context) ---

    /**
     * @brief Fill context with text input (tokenizes and adds to pending)
     * @param text Input text to tokenize and add
     */
    void fill(const std::string& text);

    /**
     * @brief Fill context with pre-tokenized input
     * @param token_ids Vector of token IDs to add to pending buffer
     */
    void fill_tokens(const std::vector<uint32_t>& token_ids);

    /**
     * @brief Add a single token to pending buffer
     * @param token_id Token ID to add
     */
    void fill_token(uint32_t token_id);

    /**
     * @brief Get current committed token sequence
     * @return Vector of committed token IDs
     */
    const std::vector<uint32_t>& get_token_ids() const { return token_ids_; }

    /**
     * @brief Get current text representation of committed tokens
     * @return Decoded text string
     */
    std::string get_text() const;

    // --- KV Cache Management (mirrors CUDA Context) ---

    /**
     * @brief Get current KV page IDs
     * @return Vector of allocated page IDs
     */
    const std::vector<uint32_t>& get_kv_page_ids() const { return kv_page_ids_; }

    /**
     * @brief Get number of tokens in last KV page
     * @return Number of tokens in last page
     */
    uint32_t get_kv_page_last_len() const { return kv_page_last_len_; }

    /**
     * @brief Flush pending tokens to KV cache (all but last token)
     * Mirrors CUDA Context::flush() behavior - processes all but the last pending token,
     * leaving one token as seed for next generation step
     */
    void flush();

    // --- Generation (mirrors CUDA Context) ---

    /**
     * @brief Perform single autoregressive decode step
     * Takes the last pending token, runs forward pass, returns distribution
     * @return Distribution over next possible tokens
     */
    Distribution decode_step();

    /**
     * @brief Generate text with custom sampler and stop condition
     * @param sampler Token sampling strategy
     * @param stop_condition When to halt generation
     * @return Generated text string
     */
    std::string generate(Sampler& sampler, StopCondition& stop_condition);

    /**
     * @brief Generate text until stop string or max tokens
     * @param stop_str String that signals end of generation
     * @param max_tokens Maximum number of tokens to generate
     * @return Generated text string
     */
    std::string generate_until(const std::string& stop_str, size_t max_tokens);

    // --- Debugging and Introspection ---

    /**
     * @brief Enable/disable verbose logging
     */
    void set_verbose(bool verbose) { verbose_ = verbose; }

    /**
     * @brief Get current context state for debugging
     */
    void print_state() const;

    /**
     * @brief Validate internal state consistency
     * @return true if state is valid, false otherwise
     */
    bool validate_state() const;

    /**
     * @brief Reset context to fresh state (clear all tokens and KV cache)
     * Useful for starting new generation sequences without creating new context
     */
    void reset();

private:
    // --- Internal KV Cache Management (mirrors CUDA Context) ---

    /**
     * @brief Grow KV cache to accommodate additional tokens
     * @param num_tokens Number of additional tokens to allocate space for
     */
    void grow_kv_pages(size_t num_tokens);

    /**
     * @brief Shrink KV cache by deallocating unused pages
     * @param num_tokens Number of tokens to remove capacity for
     */
    void shrink_kv_pages(size_t num_tokens);

    /**
     * @brief Adjust KV cache size (positive = grow, negative = shrink)
     * @param num_tokens Change in token capacity (signed)
     */
    void adjust_kv_pages(int64_t num_tokens);

    /**
     * @brief Calculate total number of tokens that can fit in current KV cache
     * @return Total token capacity across all allocated pages
     */
    size_t calculate_total_kv_capacity() const;

    /**
     * @brief Calculate number of committed tokens in KV cache
     * @return Number of tokens currently stored in KV cache
     */
    size_t calculate_committed_tokens() const;

    // --- Internal Helpers ---

    /**
     * @brief Update position IDs for new tokens
     * @param num_new_tokens Number of new tokens to create position IDs for
     * @return Vector of position IDs for new tokens
     */
    std::vector<uint32_t> generate_position_ids(size_t num_new_tokens);

    /**
     * @brief Setup Metal buffer for forward pass
     * @param token_ids Token IDs to process
     * @param position_ids Position IDs for tokens
     * @param output_indices Which tokens to return outputs for (empty = none)
     */
    void setup_buffer_for_forward_pass(const std::vector<uint32_t>& token_ids,
                                      const std::vector<uint32_t>& position_ids,
                                      const std::vector<uint32_t>& output_indices = {});

    /**
     * @brief Log state transition for debugging
     * @param operation Description of operation being performed
     */
    void log_state_transition(const std::string& operation) const;
};