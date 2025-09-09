#include "metal_context.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <set>
#include <atomic>

// --- Constructor / Destructor ---

MetalGenerationContext::MetalGenerationContext(std::unique_ptr<MetalL4maForCausalLM<bfloat16_t>> model,
                                              const std::string& tokenizer_path,
                                              uint32_t kv_page_size)
    : model_(std::move(model))
    , tokenizer_path_(tokenizer_path)
    , kv_page_last_len_(0)
    , kv_page_size_(kv_page_size)
    , verbose_(false) {

    if (!model_) {
        throw std::invalid_argument("Model cannot be null");
    }

    try {
        auto test_tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load tokenizer: " + std::string(e.what()));
    }

    // Get model config
    config_ = model_->get_config();

    // Initialize Metal singleton context
    auto& metal_ctx = MetalContext::getInstance();
    std::cout << "MetalGenerationContext: Checking Metal initialization..." << std::endl;
    if (!metal_ctx.isInitialized()) {
        std::cout << "MetalGenerationContext: Metal not initialized, calling initialize()..." << std::endl;
        if (!metal_ctx.initialize()) {
            throw std::runtime_error("Failed to initialize Metal context");
        }
    } else {
        std::cout << "MetalGenerationContext: Metal already initialized" << std::endl;
    }

    // Initialize KV cache with ALL THREE required parameters
    int32_t num_kv_pages = 1024;  // Max 1024 pages
    kv_cache_ = std::make_unique<MetalL4maKVCache<bfloat16_t>>(
        config_, num_kv_pages, kv_page_size_
    );

    // Calculate workspace size for buffer
    size_t max_tokens = 512;
    size_t batch_size = 32;
    size_t max_kv_seqlens = 2048;
    size_t dist_size = 50000;

    size_t workspace_size = MetalL4maBuffer<bfloat16_t>::get_workspace_size(
        config_, max_tokens, batch_size, max_kv_seqlens, dist_size
    );

    // Initialize buffer with ALL FOUR required parameters
    buffer_ = std::make_unique<MetalL4maBuffer<bfloat16_t>>(
        config_, kv_page_size_, dist_size, workspace_size
    );

    // Initialize memory pool for efficient zero-copy operations
    memory_pool_ = std::make_unique<PersistentMemoryPool>(workspace_size * 2);
    memory_pool_->initialize();

    // Initialize empty state
    token_ids_.clear();
    token_ids_pending_.clear();
    position_ids_.clear();
    kv_page_ids_.clear();
    kv_buffer_indices_.clear();
}

MetalGenerationContext::~MetalGenerationContext() {
    // KV cache and buffer cleanup is handled by their destructors
}

// --- Samplers ---

uint32_t MetalGenerationContext::GreedySampler::sample(const std::vector<uint32_t>& token_ids,
                                                       const std::vector<float>& probabilities) {
    if (token_ids.empty() || probabilities.empty()) {
        throw std::invalid_argument("Empty token_ids or probabilities in GreedySampler");
    }

    // Find index of maximum probability
    auto max_it = std::max_element(probabilities.begin(), probabilities.end());
    size_t max_index = std::distance(probabilities.begin(), max_it);
    return token_ids[max_index];
}

// --- Token Management ---

void MetalGenerationContext::fill(const std::string& text) {
    auto tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
    std::set<std::string> allowed_special;
    std::vector<uint32_t> new_token_ids = tokenizer.encode(text, allowed_special);

    if (verbose_) {
        std::cout << "    Input text: \"" << text << "\"" << std::endl;
        std::cout << "    Tokenized to " << new_token_ids.size() << " tokens: [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), new_token_ids.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << new_token_ids[i];
        }
        if (new_token_ids.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }

    fill_tokens(new_token_ids);
}

void MetalGenerationContext::fill_tokens(const std::vector<uint32_t>& token_ids) {
    if (verbose_) {
        log_state_transition("fill_tokens(count=" + std::to_string(token_ids.size()) + ")");
    }

    token_ids_pending_.insert(token_ids_pending_.end(), token_ids.begin(), token_ids.end());
}

void MetalGenerationContext::fill_token(uint32_t token_id) {
    if (verbose_) {
        log_state_transition("fill_token(" + std::to_string(token_id) + ")");
    }

    token_ids_pending_.push_back(token_id);
}

std::string MetalGenerationContext::get_text() const {
    try {
        auto tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
        return tokenizer.decode(token_ids_);
    } catch (const std::exception& e) {
        return "[decode error: " + std::string(e.what()) + "]";
    }
}

// --- KV Cache Management ---

void MetalGenerationContext::grow_kv_pages(size_t num_tokens) {
    adjust_kv_pages(static_cast<int64_t>(num_tokens));
}

void MetalGenerationContext::shrink_kv_pages(size_t num_tokens) {
    adjust_kv_pages(-static_cast<int64_t>(num_tokens));
}

void MetalGenerationContext::adjust_kv_pages(int64_t num_tokens) {
    if (num_tokens == 0) {
        return;
    }

    if (verbose_) {
        log_state_transition("adjust_kv_pages(" + std::to_string(num_tokens) + ")");
    }

    size_t current_tokens = calculate_committed_tokens();

    // Calculate new total tokens after adjustment
    int64_t new_total_tokens_signed = static_cast<int64_t>(current_tokens) + num_tokens;
    if (new_total_tokens_signed < 0) {
        throw std::runtime_error("Token count adjustment would result in negative tokens");
    }
    size_t new_total_tokens = static_cast<size_t>(new_total_tokens_signed);

    // Calculate required pages
    size_t current_pages = kv_page_ids_.size();
    size_t required_pages = (new_total_tokens + kv_page_size_ - 1) / kv_page_size_; // Ceiling division

    if (required_pages > current_pages) {
        // Grow: Allocate new pages through Metal KV cache
        size_t new_pages_needed = required_pages - current_pages;

        // Allocate pages through the Metal KV cache system
        // CRITICAL FIX: Use global counter to prevent page ID reuse across contexts
        static std::atomic<uint32_t> global_page_counter{1000};

        // Note: Metal KV cache manages actual allocation internally
        // We track both global page IDs (for isolation) and buffer indices (for Metal kernel)
        std::cout << "    🔍 [PAGE ALLOC] Allocating " << new_pages_needed << " new pages" << std::endl;
        for (size_t i = 0; i < new_pages_needed; ++i) {
            uint32_t page_id = global_page_counter.fetch_add(1);
            kv_page_ids_.push_back(page_id);

            // Buffer indices start from 0 and increment sequentially
            uint32_t buffer_index = static_cast<uint32_t>(kv_buffer_indices_.size());
            kv_buffer_indices_.push_back(buffer_index);
            std::cout << "      Page " << i << ": page_id=" << page_id << " -> buffer_index=" << buffer_index << std::endl;
        }

        // Ensure KV cache can handle the new size
        if (kv_cache_) {
            // The Metal KV cache will expand as needed during forward pass
            // We just need to track the logical size here
        }

    } else if (required_pages < current_pages) {
        // Shrink: Remove unused pages
        size_t pages_to_remove = current_pages - required_pages;

        // Remove pages from our tracking (both page IDs and buffer indices)
        for (size_t i = 0; i < pages_to_remove; ++i) {
            kv_page_ids_.pop_back();
            kv_buffer_indices_.pop_back();
        }

        // Note: Metal KV cache cleanup is handled by its own memory management
    }

    // Update kv_page_last_len based on new total
    if (new_total_tokens == 0) {
        kv_page_last_len_ = 0;
    } else {
        uint32_t last_page_len = new_total_tokens % kv_page_size_;
        kv_page_last_len_ = (last_page_len == 0) ? kv_page_size_ : last_page_len;
    }
}

size_t MetalGenerationContext::calculate_total_kv_capacity() const {
    return kv_page_ids_.size() * kv_page_size_;
}

size_t MetalGenerationContext::calculate_committed_tokens() const {
    if (kv_page_ids_.empty()) {
        return 0;
    } else {
        return (kv_page_ids_.size() - 1) * kv_page_size_ + kv_page_last_len_;
    }
}

// --- Core Generation Logic ---

void MetalGenerationContext::flush() {
    if (verbose_) {
        log_state_transition("flush()");
    }

    // CUDA behavior: need at least 2 tokens to flush (keep last as seed)
    if (token_ids_pending_.size() < 2) {
        return;
    }

    size_t process_count = token_ids_pending_.size() - 1;

    // Extract tokens to process (all but last)
    std::vector<uint32_t> tokens_to_process(
        token_ids_pending_.begin(),
        token_ids_pending_.begin() + process_count
    );

    // Remove processed tokens from pending, keeping the last one
    token_ids_pending_.erase(token_ids_pending_.begin(), token_ids_pending_.begin() + process_count);

    // Generate position IDs for new tokens
    std::vector<uint32_t> new_position_ids = generate_position_ids(tokens_to_process.size());

    // Grow KV cache to accommodate new tokens
    grow_kv_pages(tokens_to_process.size());

    // CRITICAL FIX: Match CUDA behavior - use EMPTY output_indices during flush
    // CUDA flush uses empty output_indices and still properly updates KV cache
    // Metal was incorrectly forcing output_indices during flush
    std::vector<uint32_t> flush_output_indices = {}; // Empty like CUDA

    std::cout << "    🔍 [FLUSH DEBUG] Tokens to process: [";
    for (size_t i = 0; i < std::min(size_t(10), tokens_to_process.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens_to_process[i];
    }
    std::cout << "], positions: [";
    for (size_t i = 0; i < std::min(size_t(10), new_position_ids.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << new_position_ids[i];
    }
    std::cout << "], output_indices: [";
    for (size_t i = 0; i < flush_output_indices.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << flush_output_indices[i];
    }
    std::cout << "] (EMPTY like CUDA)" << std::endl;
    setup_buffer_for_forward_pass(tokens_to_process, new_position_ids, flush_output_indices);

    try {
        // Execute forward pass (this populates KV cache with input context)
        auto results = model_->forward(*buffer_, *kv_cache_);

        // Commit and wait for Metal command buffer completion
        [buffer_->commandBuffer commit];
        [buffer_->commandBuffer waitUntilCompleted];

        // IMPORTANT: flush() with empty output_indices produces no output (like CUDA)
        // The KV cache is still updated properly. Real output comes from decode_step().
        if (!results.first.empty() && verbose_) {
            std::cout << "    ⚠️  [FLUSH] Unexpected output from flush with empty indices: "
                      << results.first.size() << " probabilities" << std::endl;
        }

        // Update committed state
        token_ids_.insert(token_ids_.end(), tokens_to_process.begin(), tokens_to_process.end());
        position_ids_.insert(position_ids_.end(), new_position_ids.begin(), new_position_ids.end());

        // Reset memory pool temporary allocations for next forward pass
        memory_pool_->reset_temporary();

    } catch (const std::exception& e) {
        throw std::runtime_error("Forward pass failed during flush: " + std::string(e.what()));
    }
}

MetalGenerationContext::Distribution MetalGenerationContext::decode_step() {
    if (verbose_) {
        log_state_transition("decode_step()");
    }

    if (token_ids_pending_.empty()) {
        throw std::runtime_error("Must have at least one seed token for decode_step");
    }

    // Take all pending tokens for processing
    std::vector<uint32_t> tokens_to_process = std::move(token_ids_pending_);
    token_ids_pending_.clear();

    // Generate position IDs for new tokens
    std::vector<uint32_t> new_position_ids = generate_position_ids(tokens_to_process.size());

    // Grow KV cache for new tokens
    grow_kv_pages(tokens_to_process.size());

    // Setup buffer for forward pass (request output for last token)
    std::vector<uint32_t> output_indices = {static_cast<uint32_t>(tokens_to_process.size() - 1)};
    std::cout << "    🔍 [DECODE DEBUG] Tokens to process: [";
    for (size_t i = 0; i < std::min(size_t(10), tokens_to_process.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens_to_process[i];
    }
    std::cout << "], positions: [";
    for (size_t i = 0; i < std::min(size_t(10), new_position_ids.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << new_position_ids[i];
    }
    std::cout << "], output_indices: [";
    for (size_t i = 0; i < output_indices.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << output_indices[i];
    }
    std::cout << "]" << std::endl;
    setup_buffer_for_forward_pass(tokens_to_process, new_position_ids, output_indices);

    try {
        // Execute forward pass
        auto results = model_->forward(*buffer_, *kv_cache_);

        // Commit and wait for Metal command buffer completion
        [buffer_->commandBuffer commit];
        [buffer_->commandBuffer waitUntilCompleted];

        if (results.first.empty() || results.second.empty()) {
            throw std::runtime_error("Forward pass produced no output distributions");
        }

        // Update committed state
        token_ids_.insert(token_ids_.end(), tokens_to_process.begin(), tokens_to_process.end());
        position_ids_.insert(position_ids_.end(), new_position_ids.begin(), new_position_ids.end());

        // Reset memory pool temporary allocations for next forward pass
        memory_pool_->reset_temporary();

        // Convert result to our Distribution format
        Distribution result;
        result.probabilities = results.first;
        result.token_ids.clear();
        for (int32_t token_id : results.second) {
            result.token_ids.push_back(static_cast<uint32_t>(token_id));
        }

        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error("Forward pass failed during decode_step: " + std::string(e.what()));
    }
}

std::string MetalGenerationContext::generate(Sampler& sampler, StopCondition& stop_condition) {
    if (verbose_) {
        log_state_transition("generate()");
    }

    std::vector<uint32_t> generated_token_ids;
    std::string result;

    while (!stop_condition.should_stop(generated_token_ids)) {
        // Perform decode step
        Distribution dist = decode_step();

        // Sample next token
        uint32_t next_token = sampler.sample(dist.token_ids, dist.probabilities);
        generated_token_ids.push_back(next_token);

        // Debug: print each generated token
        if (verbose_) {
            std::cout << "    Generated token " << generated_token_ids.size()
                     << ": " << next_token << " (top prob: " << dist.probabilities[0] << ")" << std::endl;
        }

        // Add token to pending buffer for next iteration
        fill_token(next_token);
    }

    // Decode all generated tokens
    try {
        auto tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
        result = tokenizer.decode(generated_token_ids);
    } catch (const std::exception& e) {
        result = "[decode error: " + std::string(e.what()) + "]";
    }

    return result;
}

std::string MetalGenerationContext::generate_until(const std::string& stop_str, size_t max_tokens) {
    if (verbose_) {
        log_state_transition("generate_until(stop=\"" + stop_str + "\", max=" + std::to_string(max_tokens) + ")");
    }

    // Create stop condition that combines length limit and string matching
    // For simplicity, we'll just use length limit for now
    // TODO: Implement proper string-based stop condition
    LengthStopCondition length_stop(max_tokens);
    GreedySampler greedy_sampler;

    return generate(greedy_sampler, length_stop);
}

// --- Helper Functions ---

std::vector<uint32_t> MetalGenerationContext::generate_position_ids(size_t num_new_tokens) {
    std::vector<uint32_t> new_position_ids;
    new_position_ids.reserve(num_new_tokens);

    uint32_t start_pos = position_ids_.empty() ? 0 : position_ids_.back() + 1;
    for (size_t i = 0; i < num_new_tokens; ++i) {
        new_position_ids.push_back(start_pos + i);
    }

    return new_position_ids;
}

void MetalGenerationContext::setup_buffer_for_forward_pass(const std::vector<uint32_t>& token_ids,
                                                 const std::vector<uint32_t>& position_ids,
                                                 const std::vector<uint32_t>& output_indices) {
    if (!buffer_) {
        throw std::runtime_error("Buffer not initialized");
    }

    if (!memory_pool_) {
        throw std::runtime_error("Memory pool not initialized");
    }

    // Get Metal context for command buffer
    auto& metal_ctx = MetalContext::getInstance();
    id<MTLCommandBuffer> commandBuffer = [metal_ctx.getCommandQueue() commandBuffer];

    // Setup basic buffer state
    buffer_->num_tokens = token_ids.size();
    buffer_->batch_size = 1;  // Single batch for now
    buffer_->commandBuffer = commandBuffer;

    // Prepare KV page management arrays
    std::vector<uint32_t> kv_page_indptr;
    std::vector<uint32_t> kv_last_page_lens;
    std::vector<uint32_t> qo_indptr;

    if (!kv_page_ids_.empty()) {
        // Setup for existing KV cache
        kv_page_indptr = {0, static_cast<uint32_t>(kv_page_ids_.size())};
        kv_last_page_lens = {kv_page_last_len_};
        qo_indptr = {0, static_cast<uint32_t>(token_ids.size())};
    } else {
        // Initialize empty arrays for first use
        kv_page_indptr = {0, 0};
        kv_last_page_lens = {0};
        qo_indptr = {0, static_cast<uint32_t>(token_ids.size())};
    }

    // Debug: Show buffer indices being passed
    std::cout << "🔍 [BUFFER DEBUG] KV buffer indices (" << kv_buffer_indices_.size() << "): [";
    for (size_t i = 0; i < std::min(size_t(5), kv_buffer_indices_.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << kv_buffer_indices_[i];
    }
    std::cout << "]" << std::endl;

    // Use planWithMapping for zero-copy buffer setup
    buffer_->planWithMapping(
        commandBuffer,
        *memory_pool_,

        // Token IDs (cast uint32_t* to int32_t*)
        reinterpret_cast<const int32_t*>(token_ids.data()), token_ids.size(),

        // Position IDs (cast uint32_t* to int32_t*)
        reinterpret_cast<const int32_t*>(position_ids.data()), position_ids.size(),

        // KV page management (cast uint32_t* to int32_t*) - using buffer indices instead of page IDs
        reinterpret_cast<const int32_t*>(kv_buffer_indices_.data()), kv_buffer_indices_.size(),
        reinterpret_cast<const int32_t*>(kv_page_indptr.data()), kv_page_indptr.size(),
        reinterpret_cast<const int32_t*>(kv_last_page_lens.data()), kv_last_page_lens.size(),

        // Query organization (cast uint32_t* to int32_t*)
        reinterpret_cast<const int32_t*>(qo_indptr.data()), qo_indptr.size(),

        // Masking (empty for basic generation)
        nullptr, 0,  // packed_custom_mask
        nullptr, 0,  // mask_indptr

        // Batch indices (simple case)
        nullptr, 0,  // kv_batch_indices
        nullptr, 0,  // kv_positions

        // Output selection (cast uint32_t* to int32_t*)
        reinterpret_cast<const int32_t*>(output_indices.data()), output_indices.size()
    );
}

// --- Debugging ---

void MetalGenerationContext::print_state() const {
    std::cout << "=== MetalContext State ===" << std::endl;
    std::cout << "  Committed tokens: " << token_ids_.size() << std::endl;
    std::cout << "  Pending tokens: " << token_ids_pending_.size() << std::endl;
    std::cout << "  Position IDs: " << position_ids_.size() << std::endl;
    std::cout << "  KV pages: " << kv_page_ids_.size() << " (page IDs: ";
    for (size_t i = 0; i < kv_page_ids_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << kv_page_ids_[i];
    }
    std::cout << ", buffer indices: ";
    for (size_t i = 0; i < kv_buffer_indices_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << kv_buffer_indices_[i];
    }
    std::cout << ")" << std::endl;
    std::cout << "  KV page last len: " << kv_page_last_len_ << std::endl;
    std::cout << "  KV page size: " << kv_page_size_ << std::endl;
    std::cout << "  Total KV capacity: " << calculate_total_kv_capacity() << std::endl;
    std::cout << "  Committed tokens (calc): " << calculate_committed_tokens() << std::endl;

    if (!token_ids_.empty()) {
        std::cout << "  Last 5 committed tokens: [";
        size_t start = token_ids_.size() >= 5 ? token_ids_.size() - 5 : 0;
        for (size_t i = start; i < token_ids_.size(); ++i) {
            std::cout << token_ids_[i];
            if (i < token_ids_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    if (!token_ids_pending_.empty()) {
        std::cout << "  Pending tokens: [";
        for (size_t i = 0; i < std::min(size_t(5), token_ids_pending_.size()); ++i) {
            std::cout << token_ids_pending_[i];
            if (i < std::min(size_t(5), token_ids_pending_.size()) - 1) std::cout << ", ";
        }
        if (token_ids_pending_.size() > 5) std::cout << "...";
        std::cout << "]" << std::endl;
    }
    std::cout << "==========================" << std::endl;
}

bool MetalGenerationContext::validate_state() const {
    // Check that position_ids matches token_ids
    if (position_ids_.size() != token_ids_.size()) {
        return false;
    }

    // Check that calculated committed tokens matches actual token_ids size
    if (calculate_committed_tokens() != token_ids_.size()) {
        return false;
    }

    // Check that kv_page_last_len is within bounds
    if (kv_page_last_len_ > kv_page_size_) {
        return false;
    }

    // Check that we have enough KV capacity for committed tokens
    if (calculate_total_kv_capacity() < token_ids_.size()) {
        return false;
    }

    return true;
}

void MetalGenerationContext::reset() {
    if (verbose_) {
        log_state_transition("reset()");
    }

    // Clear all token state
    token_ids_.clear();
    token_ids_pending_.clear();
    position_ids_.clear();

    // Clear KV cache page allocations AND zero the cache data
    kv_page_ids_.clear();
    kv_buffer_indices_.clear();
    kv_page_last_len_ = 0;

    // CRITICAL FIX: Zero the actual KV cache data to prevent stale data contamination
    if (kv_cache_) {
        kv_cache_->zero(); // Zero out all KV cache buffers

        // CRITICAL: Ensure zeroing completes before proceeding
        auto& metal_ctx = MetalContext::getInstance();
        id<MTLCommandBuffer> cmdBuffer = [metal_ctx.getCommandQueue() commandBuffer];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        if (verbose_) {
            std::cout << "    🔄 KV cache data cleared and synchronized" << std::endl;
        }
    }

    // Reset memory pool to clean state
    if (memory_pool_) {
        memory_pool_->reset_temporary();
    }

    if (verbose_) {
        std::cout << "    ✅ Context reset to fresh state" << std::endl;
    }
}

// Simplified kernel-aligned extraction for demonstration
std::vector<float> MetalGenerationContext::extract_kernel_state(const std::vector<uint32_t>& token_ids,
                                                               const std::string& extraction_point) {
    if (verbose_) {
        std::cout << "🧪 [EXTRACTION] " << extraction_point << " from " << token_ids.size() << " tokens" << std::endl;
    }

    try {
        if (extraction_point == "attention_input") {
            // For now, demonstrate the concept by running a minimal forward pass
            // and extracting embeddings using existing functionality

            reset();  // Clean state
            fill_tokens(token_ids);

            // This triggers embedding lookup in the Metal model
            // In a full implementation, we would access embeddings after this point
            auto result = decode_step();

            if (verbose_) {
                std::cout << "    ✅ Demonstrated " << extraction_point << " extraction concept" << std::endl;
                std::cout << "    📝 Note: Full implementation requires Metal backend modifications" << std::endl;
            }

            // For demonstration, return placeholder values with correct size
            // Real implementation would return actual embedding values
            size_t embedding_size = token_ids.size() * config_.hidden_size;
            return std::vector<float>(embedding_size, 0.5f);  // Placeholder

        } else {
            if (verbose_) {
                std::cout << "    ⚠️ " << extraction_point << " extraction not yet implemented" << std::endl;
                std::cout << "    📝 This demonstrates the kernel-aligned validation framework" << std::endl;
            }
            return std::vector<float>();  // Empty for other extraction points
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ [EXTRACTION ERROR] Failed to extract " << extraction_point
                  << ": " << e.what() << std::endl;
        throw;
    }
}

void MetalGenerationContext::log_state_transition(const std::string& operation) const {
    if (!verbose_) return;

    std::cout << "  🔄 " << operation << std::endl;
    std::cout << "    Before: committed=" << token_ids_.size()
              << ", pending=" << token_ids_pending_.size()
              << ", kv_pages=" << kv_page_ids_.size()
              << ", kv_last_len=" << kv_page_last_len_ << std::endl;
}