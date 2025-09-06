#include "cuda_context.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <set>

// --- Constructor / Destructor ---

CudaContext::CudaContext(std::unique_ptr<Model> model,
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

    // Initialize empty state
    token_ids_.clear();
    token_ids_pending_.clear();
    position_ids_.clear();
    kv_page_ids_.clear();

}

CudaContext::~CudaContext() {
    if (!kv_page_ids_.empty() && model_) {
        for (uint32_t page_id : kv_page_ids_) {
            Model::DeallocateCommand dealloc_cmd;
            dealloc_cmd.kind = Model::ObjectKind::KV_BLOCK;
            dealloc_cmd.object_id_offset = page_id;
            dealloc_cmd.count = 1;

            try {
                model_->handle_deallocate({dealloc_cmd});
            } catch (const std::exception&) {
                // Ignore deallocation errors during cleanup
            }
        }
    }
}

// --- Samplers ---

uint32_t CudaContext::GreedySampler::sample(const std::vector<uint32_t>& token_ids,
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

void CudaContext::fill(const std::string& text) {
    auto tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
    std::set<std::string> allowed_special;
    std::vector<uint32_t> new_token_ids = tokenizer.encode(text, allowed_special);
    fill_tokens(new_token_ids);
}

void CudaContext::fill_tokens(const std::vector<uint32_t>& token_ids) {
    if (verbose_) {
        log_state_transition("fill_tokens(count=" + std::to_string(token_ids.size()) + ")");
    }

    token_ids_pending_.insert(token_ids_pending_.end(), token_ids.begin(), token_ids.end());
}

void CudaContext::fill_token(uint32_t token_id) {
    if (verbose_) {
        log_state_transition("fill_token(" + std::to_string(token_id) + ")");
    }

    token_ids_pending_.push_back(token_id);
}

std::string CudaContext::get_text() const {
    try {
        auto tokenizer = bpe::llama3_tokenizer(tokenizer_path_);
        return tokenizer.decode(token_ids_);
    } catch (const std::exception& e) {
        return "[decode error: " + std::string(e.what()) + "]";
    }
}

// --- KV Cache Management (Critical fixes based on Rust implementation) ---

void CudaContext::grow_kv_pages(size_t num_tokens) {
    adjust_kv_pages(static_cast<int64_t>(num_tokens));
}

void CudaContext::shrink_kv_pages(size_t num_tokens) {
    adjust_kv_pages(-static_cast<int64_t>(num_tokens));
}

void CudaContext::adjust_kv_pages(int64_t num_tokens) {
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
        // Grow: Allocate new pages
        size_t new_pages_needed = required_pages - current_pages;

        // Allocate actual KV pages through the CUDA backend
        uint32_t next_page_id = kv_page_ids_.empty() ? 1000 : (*std::max_element(kv_page_ids_.begin(), kv_page_ids_.end())) + 1;

        // Create allocation command
        Model::AllocateCommand alloc_cmd;
        alloc_cmd.kind = Model::ObjectKind::KV_BLOCK;
        alloc_cmd.object_id_offset = next_page_id;
        alloc_cmd.count = static_cast<uint32_t>(new_pages_needed);

        // Actually allocate the pages in the backend
        model_->handle_allocate({alloc_cmd});

        // Add the allocated page IDs to our tracking
        for (size_t i = 0; i < new_pages_needed; ++i) {
            kv_page_ids_.push_back(next_page_id + i);
        }

    } else if (required_pages < current_pages) {
        // Shrink: Deallocate unused pages
        size_t pages_to_remove = current_pages - required_pages;

        // Get the page IDs to deallocate (from the end)
        std::vector<uint32_t> pages_to_deallocate;
        for (size_t i = current_pages - pages_to_remove; i < current_pages; ++i) {
            pages_to_deallocate.push_back(kv_page_ids_[i]);
        }

        // Create deallocation command
        if (!pages_to_deallocate.empty()) {
            Model::DeallocateCommand dealloc_cmd;
            dealloc_cmd.kind = Model::ObjectKind::KV_BLOCK;
            dealloc_cmd.object_id_offset = pages_to_deallocate[0];
            dealloc_cmd.count = static_cast<uint32_t>(pages_to_deallocate.size());

            // Actually deallocate in the backend
            model_->handle_deallocate({dealloc_cmd});
        }

        // Remove pages from our tracking
        for (size_t i = 0; i < pages_to_remove; ++i) {
            kv_page_ids_.pop_back();
        }

    }

    // Update kv_page_last_len based on new total
    if (new_total_tokens == 0) {
        kv_page_last_len_ = 0;
    } else {
        uint32_t last_page_len = new_total_tokens % kv_page_size_;
        kv_page_last_len_ = (last_page_len == 0) ? kv_page_size_ : last_page_len;
    }

}

size_t CudaContext::calculate_total_kv_capacity() const {
    return kv_page_ids_.size() * kv_page_size_;
}

size_t CudaContext::calculate_committed_tokens() const {
    if (kv_page_ids_.empty()) {
        return 0;
    } else {
        return (kv_page_ids_.size() - 1) * kv_page_size_ + kv_page_last_len_;
    }
}

// --- Core Generation Logic (Based on Rust Context) ---

void CudaContext::flush() {
    if (verbose_) {
        log_state_transition("flush()");
    }

    // Rust behavior: need at least 2 tokens to flush (keep last as seed)
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


    // Prepare forward pass command (no output needed for flush)
    Model::ForwardTextCommand forward_cmd;
    forward_cmd.kv_page_last_len = kv_page_last_len_;
    forward_cmd.kv_page_ids = kv_page_ids_;
    forward_cmd.token_ids = tokens_to_process;
    forward_cmd.position_ids = new_position_ids;
    forward_cmd.brle_masks = create_empty_masks(tokens_to_process.size());
    forward_cmd.output_indices = {}; // No output needed for flush

    try {
        // Execute forward pass (this populates KV cache)
        std::vector<std::vector<Model::Distribution>> results = model_->handle_forward_text({forward_cmd});

        // Update committed state
        token_ids_.insert(token_ids_.end(), tokens_to_process.begin(), tokens_to_process.end());
        position_ids_.insert(position_ids_.end(), new_position_ids.begin(), new_position_ids.end());


    } catch (const std::exception& e) {
        throw std::runtime_error("Forward pass failed during flush: " + std::string(e.what()));
    }
}

CudaContext::Distribution CudaContext::decode_step() {
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


    // Prepare forward pass command (request output for last token)
    Model::ForwardTextCommand forward_cmd;
    forward_cmd.kv_page_last_len = kv_page_last_len_;
    forward_cmd.kv_page_ids = kv_page_ids_;
    forward_cmd.token_ids = tokens_to_process;
    forward_cmd.position_ids = new_position_ids;
    forward_cmd.brle_masks = create_empty_masks(tokens_to_process.size());
    forward_cmd.output_indices = {static_cast<uint32_t>(tokens_to_process.size() - 1)}; // Last token

    try {
        // Execute forward pass
        std::vector<std::vector<Model::Distribution>> results = model_->handle_forward_text({forward_cmd});

        if (results.empty() || results[0].empty()) {
            throw std::runtime_error("Forward pass produced no output distributions");
        }

        // Update committed state
        token_ids_.insert(token_ids_.end(), tokens_to_process.begin(), tokens_to_process.end());
        position_ids_.insert(position_ids_.end(), new_position_ids.begin(), new_position_ids.end());

        // Convert result to our Distribution format
        const auto& model_dist = results[0][0];
        Distribution result;
        result.token_ids = model_dist.token_ids;
        result.probabilities = model_dist.probabilities;


        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error("Forward pass failed during decode_step: " + std::string(e.what()));
    }
}

std::string CudaContext::generate(Sampler& sampler, StopCondition& stop_condition) {
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

std::string CudaContext::generate_until(const std::string& stop_str, size_t max_tokens) {
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

std::vector<uint32_t> CudaContext::generate_position_ids(size_t num_new_tokens) {
    std::vector<uint32_t> new_position_ids;
    new_position_ids.reserve(num_new_tokens);

    uint32_t start_pos = position_ids_.empty() ? 0 : position_ids_.back() + 1;
    for (size_t i = 0; i < num_new_tokens; ++i) {
        new_position_ids.push_back(start_pos + i);
    }

    return new_position_ids;
}

std::vector<std::vector<uint32_t>> CudaContext::create_empty_masks(size_t num_tokens) {
    // Create empty BRLE masks for all tokens (no masking)
    return std::vector<std::vector<uint32_t>>(num_tokens);
}

// --- Debugging ---

void CudaContext::print_state() const {
    std::cout << "=== CudaContext State ===" << std::endl;
    std::cout << "  Committed tokens: " << token_ids_.size() << std::endl;
    std::cout << "  Pending tokens: " << token_ids_pending_.size() << std::endl;
    std::cout << "  Position IDs: " << position_ids_.size() << std::endl;
    std::cout << "  KV pages: " << kv_page_ids_.size() << std::endl;
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
    std::cout << "=========================" << std::endl;
}

bool CudaContext::validate_state() const {
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

void CudaContext::log_state_transition(const std::string& operation) const {
    if (!verbose_) return;

    std::cout << "  🔄 " << operation << std::endl;
    std::cout << "    Before: committed=" << token_ids_.size()
              << ", pending=" << token_ids_pending_.size()
              << ", kv_pages=" << kv_page_ids_.size()
              << ", kv_last_len=" << kv_page_last_len_ << std::endl;
}