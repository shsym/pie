#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include <Metal/Metal.h>
#include <vector>
#include <memory>
#include <iostream>

// Forward declarations
struct L4maConfig;

/**
 * @brief Metal equivalent of CUDA StackAllocator
 *
 * Provides stack-based memory allocation for temporary buffers during model execution.
 * This matches the interface of the CUDA StackAllocator from stack_allocator.cuh.
 */
class MetalStackAllocator {
public:
    explicit MetalStackAllocator(size_t total_size);
    ~MetalStackAllocator() = default;

    // Allocate aligned memory
    template<typename T>
    MetalTensor<T> allocate(size_t count);

    // Allocate remaining memory as buffer
    MetalBuffer<uint8_t> allocate_rest();

    // Deallocate (stack-based, so just moves pointer back)
    template<typename T>
    void deallocate(MetalTensor<T>& tensor);

    // Reset allocator to beginning
    void reset();

    // Query methods
    size_t total_size() const { return total_size_; }
    size_t allocated_size() const { return current_offset_; }
    size_t available_size() const { return total_size_ - current_offset_; }

private:
    size_t align_offset(size_t offset, size_t alignment) const;

    MetalBuffer<uint8_t> buffer_;
    size_t total_size_;
    size_t current_offset_;
};

/**
 * @brief Persistent memory pool for zero-copy cross-layer memory reuse
 *
 * Manages a large pre-allocated buffer that can be partitioned into persistent
 * regions (for data that persists across layers) and temporary regions (for
 * intermediate computations that can be reused).
 */
class PersistentMemoryPool {
public:
    struct MemoryRegion {
        size_t offset;           // Offset in bytes from buffer start
        size_t size;             // Size in bytes
        bool in_use;             // Whether region is currently allocated
        bool is_persistent;      // Whether region persists across layers
        std::string name;        // Optional name for debugging

        MemoryRegion(size_t off, size_t sz, bool persistent = false, const std::string& n = "")
            : offset(off), size(sz), in_use(false), is_persistent(persistent), name(n) {}
    };

    explicit PersistentMemoryPool(size_t total_size);
    ~PersistentMemoryPool() = default;

    // Initialize the pool with a pre-allocated buffer
    bool initialize();

    // Allocate a persistent region that survives layer boundaries
    MemoryRegion* allocate_persistent(size_t size, const std::string& name = "");

    // Allocate a temporary region that gets reset per layer
    MemoryRegion* allocate_temporary(size_t size, const std::string& name = "");

    // Free a specific region
    void free(MemoryRegion* region);

    // Reset all temporary regions (called between layers)
    void reset_temporary();

    // Reset all regions (called between inference runs)
    void reset_all();

    // Create tensor view into a memory region
    template<typename T>
    MetalTensorView<T> create_tensor_view(MemoryRegion* region, const std::vector<size_t>& shape);

    // Get the underlying Metal buffer
    id<MTLBuffer> getBuffer() const { return buffer_.getBuffer(); }

    // Query methods
    size_t total_size() const { return total_size_; }
    size_t allocated_size() const;
    size_t available_size() const { return total_size_ - allocated_size(); }
    size_t persistent_allocated() const;
    size_t temporary_allocated() const;

    // Debug utilities
    void print_layout() const;
    bool validate_layout() const;

private:
    size_t find_free_region(size_t size, bool persistent);
    void coalesce_free_regions();

    MetalBuffer<uint8_t> buffer_;
    size_t total_size_;
    std::vector<std::unique_ptr<MemoryRegion>> regions_;
    size_t persistent_watermark_;  // High water mark for persistent allocations
};

/**
 * @brief Metal equivalent of L4maBuffer from l4ma.cuh
 *
 * Consolidates workspace memory, intermediate computations, and index pointers
 * for a single forward pass using Metal compute resources.
 */
template <typename T>
class MetalL4maBuffer {
public:
    // Device-side tensors, valid after plan() is called
    MetalTensor<int32_t> input_ids;
    MetalTensor<int32_t> position_ids;
    MetalTensor<int32_t> kv_page_indices;
    MetalTensor<int32_t> kv_page_indptr;
    MetalTensor<int32_t> kv_last_page_lens;
    MetalTensor<int32_t> qo_indptr;
    MetalTensor<uint8_t> custom_mask;
    MetalTensor<int32_t> mask_indptr;
    MetalTensor<int32_t> kv_batch_indices;
    MetalTensor<int32_t> kv_positions;
    MetalTensor<int32_t> output_indices_src;

    // Configuration and context
    const L4maConfig& config;
    const int32_t page_size;
    const int32_t dist_size;
    size_t num_tokens;
    size_t batch_size;

    // Metal command buffer for operations
    id<MTLCommandBuffer> commandBuffer;

    // Constructor/Destructor
    MetalL4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size);
    ~MetalL4maBuffer() = default;

    // Deleted copy/move operations
    MetalL4maBuffer(const MetalL4maBuffer&) = delete;
    MetalL4maBuffer& operator=(const MetalL4maBuffer&) = delete;
    MetalL4maBuffer(MetalL4maBuffer&&) = delete;
    MetalL4maBuffer& operator=(MetalL4maBuffer&&) = delete;

    // Static method for workspace size calculation
    static size_t get_workspace_size(
        const L4maConfig& config,
        size_t max_num_tokens,
        size_t max_batch_size,
        size_t max_kv_seqlens,
        size_t dist_size
    );

    // Plan buffer allocation and data transfer (traditional approach)
    void plan(
        id<MTLCommandBuffer> command_buffer,
        std::vector<int32_t>& input_ids_host,
        std::vector<int32_t>& position_ids_host,
        std::vector<int32_t>& kv_page_indices_host,
        std::vector<int32_t>& kv_page_indptr_host,
        std::vector<int32_t>& kv_last_page_lens_host,
        std::vector<int32_t>& qo_indptr_host,
        std::vector<bool>& packed_custom_mask_host,
        std::vector<int32_t>& mask_indptr_host,
        std::vector<int32_t>& kv_batch_indices_host,
        std::vector<int32_t>& kv_positions_host,
        std::vector<int32_t>& output_indices_src_host
    );

    // Zero-copy plan with memory mapping (new approach)
    void planWithMapping(
        id<MTLCommandBuffer> command_buffer,
        PersistentMemoryPool& memory_pool,
        const int32_t* input_ids_ptr,
        size_t input_ids_count,
        const int32_t* position_ids_ptr,
        size_t position_ids_count,
        const int32_t* kv_page_indices_ptr,
        size_t kv_page_indices_count,
        const int32_t* kv_page_indptr_ptr,
        size_t kv_page_indptr_count,
        const int32_t* kv_last_page_lens_ptr,
        size_t kv_last_page_lens_count,
        const int32_t* qo_indptr_ptr,
        size_t qo_indptr_count,
        const uint8_t* custom_mask_ptr,
        size_t custom_mask_count,
        const int32_t* mask_indptr_ptr,
        size_t mask_indptr_count,
        const int32_t* kv_batch_indices_ptr,
        size_t kv_batch_indices_count,
        const int32_t* kv_positions_ptr,
        size_t kv_positions_count,
        const int32_t* output_indices_src_ptr,
        size_t output_indices_src_count
    );

    // Map existing host memory directly (zero-copy)
    void mapHostMemory(void* host_ptr, size_t size);

    // Create layer workspace views for persistent cross-layer memory
    struct LayerWorkspace {
        MetalTensorView<T> hidden_states;
        MetalTensorView<T> attention_output;
        MetalTensorView<T> mlp_output;
        MetalTensorView<T> residual_buffer;

        LayerWorkspace(PersistentMemoryPool::MemoryRegion* region,
                       id<MTLBuffer> buffer,
                       const L4maConfig& config,
                       size_t num_tokens);
    };

    // Initialize persistent workspaces for all layers
    bool initializePersistentWorkspaces(PersistentMemoryPool& memory_pool);

    // Get workspace for specific layer
    LayerWorkspace* getLayerWorkspace(size_t layer_idx);

    // Reset temporary regions between layers
    void resetTemporaryRegions();

    // Allocator wrappers (matching CUDA interface)
    template <typename U>
    MetalTensor<U> allocate(size_t count);

    MetalBuffer<uint8_t> allocate_rest();

    template <typename U>
    void deallocate(MetalTensor<U>& tensor);

private:
    size_t buffer_size_;
    std::unique_ptr<MetalStackAllocator> allocator_;

    // Memory optimization support
    PersistentMemoryPool* memory_pool_ = nullptr;
    std::vector<std::unique_ptr<LayerWorkspace>> layer_workspaces_;
    id<MTLBuffer> mapped_host_buffer_ = nullptr;
    bool using_zero_copy_ = false;
};

/**
 * @brief Metal batch prefill handler
 *
 * Equivalent to flashinfer::BatchPrefillHandler for Metal compute.
 * Manages batch attention computation with paged KV cache.
 */
class MetalBatchPrefillHandler {
public:
    MetalBatchPrefillHandler();
    ~MetalBatchPrefillHandler() = default;

    // Initialize handler with workspace
    bool initialize(size_t workspace_size);

    // Plan batch prefill operation
    template<typename DTypeQ, typename DTypeKV, typename DTypeO>
    bool plan(
        id<MTLCommandBuffer> commandBuffer,
        const std::vector<int32_t>& qo_indptr_host,
        const std::vector<int32_t>& kv_page_indices_host,
        const std::vector<int32_t>& kv_page_indptr_host,
        const std::vector<int32_t>& kv_last_page_lens_host,
        size_t num_qo_heads,
        size_t num_kv_heads,
        size_t head_dim,
        size_t page_size
    );

    // Execute batch prefill attention
    template<typename DTypeQ, typename DTypeKV, typename DTypeO>
    void forward(
        id<MTLCommandBuffer> commandBuffer,
        const DTypeQ* q_data,
        const DTypeKV* kv_data,
        DTypeO* o_data,
        const std::vector<bool>& custom_mask,
        const std::vector<int32_t>& mask_indptr
    );

private:
    bool initialized_;
    size_t workspace_size_;
    MetalBuffer<uint8_t> workspace_;
};

// Template implementations

template<typename T>
MetalTensor<T> MetalStackAllocator::allocate(size_t count) {
    size_t required_bytes = count * sizeof(T);
    size_t aligned_offset = align_offset(current_offset_, alignof(T));
    std::cout << "Current offset: " << current_offset_ << ", Aligned offset: " << aligned_offset << std::endl;

    if (aligned_offset + required_bytes > total_size_) {
        std::cout << "Current offset: " << current_offset_ << ", Aligned offset: " << aligned_offset << ", Required bytes: " << required_bytes << ", Total size: " << total_size_ << std::endl;
        throw std::runtime_error("MetalStackAllocator: Insufficient memory");
    }

    // Return a tensor view into the pre-allocated stack buffer (no copy, no new allocation)
    id<MTLBuffer> metal_buffer = buffer_.getBuffer();
    MetalTensor<T> tensor = MetalTensor<T>::createView(metal_buffer, {count}, aligned_offset);
    std::cout << "Created MetalTensor view at offset " << aligned_offset << " with size " << required_bytes << " bytes." << std::endl;

    current_offset_ = aligned_offset + required_bytes;
    return tensor;
}

template<typename T>
void MetalStackAllocator::deallocate(MetalTensor<T>& tensor) {
    // Stack allocator - deallocate by moving pointer back
    size_t tensor_bytes = tensor.byteSize();
    if (current_offset_ >= tensor_bytes) {
        current_offset_ -= tensor_bytes;
    }
}

template <typename T>
MetalL4maBuffer<T>::MetalL4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size)
    : config(cfg), page_size(page_size), dist_size(dist_size), buffer_size_(workspace_size) {

    // Initialize the stack allocator
    allocator_ = std::make_unique<MetalStackAllocator>(workspace_size);

    // Initialize Metal command buffer
    auto& context = MetalContext::getInstance();
    if (!context.isInitialized()) {
        throw std::runtime_error("Metal context not initialized");
    }
}

template <typename T>
size_t MetalL4maBuffer<T>::get_workspace_size(
    const L4maConfig& config,
    size_t max_num_tokens,
    size_t max_batch_size,
    size_t max_kv_seqlens,
    size_t dist_size
) {
    // Calculate workspace size requirements
    // This should match the CUDA version's calculations

    size_t tensor_sizes = 0;

    // Input/position IDs
    tensor_sizes += max_num_tokens * sizeof(int32_t) * 2;

    // KV page management
    tensor_sizes += max_kv_seqlens * sizeof(int32_t) * 3; // indices, indptr, lens

    // Batch management
    tensor_sizes += max_batch_size * sizeof(int32_t) * 2; // qo_indptr, batch_indices

    // Masks
    tensor_sizes += max_num_tokens * max_kv_seqlens / 8; // Custom mask (bits)
    tensor_sizes += max_batch_size * sizeof(int32_t); // mask_indptr

    // Intermediate computations (Q, K, V projections, etc.)
    size_t hidden_size = config.hidden_size;
    size_t intermediate_size = config.intermediate_size;
    size_t head_size = config.head_size;
    size_t num_heads = config.num_query_heads;
    size_t num_kv_heads = config.num_key_value_heads;

    // Q, K, V projections
    tensor_sizes += max_num_tokens * (num_heads + 2 * num_kv_heads) * head_size * sizeof(T);

    // MLP intermediate
    tensor_sizes += max_num_tokens * intermediate_size * sizeof(T) * 2; // gate + up

    // Attention output
    tensor_sizes += max_num_tokens * hidden_size * sizeof(T);

    // Metal attention workspace (this was missing!)
    // Each attention layer needs workspace for K/V cache conversion and intermediate buffers
    // Based on metal_batch_prefill_get_workspace calculation:
    // - Q buffer: max_num_tokens * head_dim * sizeof(uint16_t)
    // - K buffer: max_kv_pages * page_size * kv_head_dim * sizeof(uint16_t)
    // - V buffer: max_kv_pages * page_size * kv_head_dim * sizeof(uint16_t)
    // - Plus index and debug buffers
    size_t max_kv_pages = 128; // Conservative estimate for large models
    size_t page_size = 16;     // Standard page size
    size_t attention_workspace = 0;
    attention_workspace += max_num_tokens * num_heads * head_size * sizeof(uint16_t); // Q buffer
    attention_workspace += max_kv_pages * page_size * num_kv_heads * head_size * sizeof(uint16_t) * 2; // K+V buffers
    attention_workspace += max_num_tokens * num_heads * head_size * sizeof(uint16_t); // Output buffer
    attention_workspace += 1024; // Index and debug buffers
    tensor_sizes += attention_workspace;

    // Distribution storage
    tensor_sizes += max_num_tokens * dist_size * (sizeof(T) + sizeof(int32_t)); // values + indices

    // Add padding for alignment and safety margin (increased for multi-token sequences)
    tensor_sizes += 32768; // 32KB padding to handle multi-token conversational input

    return tensor_sizes;
}

template <typename T>
void MetalL4maBuffer<T>::plan(
    id<MTLCommandBuffer> command_buffer,
    std::vector<int32_t>& input_ids_host,
    std::vector<int32_t>& position_ids_host,
    std::vector<int32_t>& kv_page_indices_host,
    std::vector<int32_t>& kv_page_indptr_host,
    std::vector<int32_t>& kv_last_page_lens_host,
    std::vector<int32_t>& qo_indptr_host,
    std::vector<bool>& packed_custom_mask_host,
    std::vector<int32_t>& mask_indptr_host,
    std::vector<int32_t>& kv_batch_indices_host,
    std::vector<int32_t>& kv_positions_host,
    std::vector<int32_t>& output_indices_src_host
) {
    commandBuffer = command_buffer;

    // Reset allocator
    allocator_->reset();

    // Set sizes
    num_tokens = input_ids_host.size();
    batch_size = qo_indptr_host.size() - 1;

    // Allocate and copy tensors
    if (!input_ids_host.empty()) {
        input_ids = allocator_->allocate<int32_t>(input_ids_host.size());
        input_ids.copyFromHost(input_ids_host.data());
    }

    if (!position_ids_host.empty()) {
        position_ids = allocator_->allocate<int32_t>(position_ids_host.size());
        position_ids.copyFromHost(position_ids_host.data());
    }

    if (!kv_page_indices_host.empty()) {
        kv_page_indices = allocator_->allocate<int32_t>(kv_page_indices_host.size());
        kv_page_indices.copyFromHost(kv_page_indices_host.data());
    }

    if (!kv_page_indptr_host.empty()) {
        kv_page_indptr = allocator_->allocate<int32_t>(kv_page_indptr_host.size());
        kv_page_indptr.copyFromHost(kv_page_indptr_host.data());
    }

    if (!kv_last_page_lens_host.empty()) {
        kv_last_page_lens = allocator_->allocate<int32_t>(kv_last_page_lens_host.size());
        kv_last_page_lens.copyFromHost(kv_last_page_lens_host.data());
    }

    if (!qo_indptr_host.empty()) {
        qo_indptr = allocator_->allocate<int32_t>(qo_indptr_host.size());
        qo_indptr.copyFromHost(qo_indptr_host.data());
    }

    // Handle custom mask (convert bool vector to uint8_t)
    if (!packed_custom_mask_host.empty()) {
        size_t mask_bytes = (packed_custom_mask_host.size() + 7) / 8; // Round up to bytes
        custom_mask = allocator_->allocate<uint8_t>(mask_bytes);

        // Pack bool vector to bytes
        std::vector<uint8_t> packed_mask(mask_bytes, 0);
        for (size_t i = 0; i < packed_custom_mask_host.size(); ++i) {
            if (packed_custom_mask_host[i]) {
                packed_mask[i / 8] |= (1 << (i % 8));
            }
        }
        custom_mask.copyFromHost(packed_mask.data());
    }

    if (!mask_indptr_host.empty()) {
        mask_indptr = allocator_->allocate<int32_t>(mask_indptr_host.size());
        mask_indptr.copyFromHost(mask_indptr_host.data());
    }

    if (!kv_batch_indices_host.empty()) {
        kv_batch_indices = allocator_->allocate<int32_t>(kv_batch_indices_host.size());
        kv_batch_indices.copyFromHost(kv_batch_indices_host.data());
    }

    if (!kv_positions_host.empty()) {
        kv_positions = allocator_->allocate<int32_t>(kv_positions_host.size());
        kv_positions.copyFromHost(kv_positions_host.data());
    }

    if (!output_indices_src_host.empty()) {
        output_indices_src = allocator_->allocate<int32_t>(output_indices_src_host.size());
        output_indices_src.copyFromHost(output_indices_src_host.data());
    }
}

// Zero-copy plan implementation
template <typename T>
void MetalL4maBuffer<T>::planWithMapping(
    id<MTLCommandBuffer> command_buffer,
    PersistentMemoryPool& memory_pool,
    const int32_t* input_ids_ptr,
    size_t input_ids_count,
    const int32_t* position_ids_ptr,
    size_t position_ids_count,
    const int32_t* kv_page_indices_ptr,
    size_t kv_page_indices_count,
    const int32_t* kv_page_indptr_ptr,
    size_t kv_page_indptr_count,
    const int32_t* kv_last_page_lens_ptr,
    size_t kv_last_page_lens_count,
    const int32_t* qo_indptr_ptr,
    size_t qo_indptr_count,
    const uint8_t* custom_mask_ptr,
    size_t custom_mask_count,
    const int32_t* mask_indptr_ptr,
    size_t mask_indptr_count,
    const int32_t* kv_batch_indices_ptr,
    size_t kv_batch_indices_count,
    const int32_t* kv_positions_ptr,
    size_t kv_positions_count,
    const int32_t* output_indices_src_ptr,
    size_t output_indices_src_count) {

    commandBuffer = command_buffer;
    memory_pool_ = &memory_pool;
    using_zero_copy_ = true;

    // Set sizes
    num_tokens = input_ids_count;
    batch_size = qo_indptr_count - 1;

    // Create tensor views directly into host memory regions
    // Note: In Metal unified memory, we can directly reference host pointers
    id<MTLBuffer> pool_buffer = memory_pool.getBuffer();

    if (input_ids_ptr && input_ids_count > 0) {
        // Create a region in the memory pool for input_ids
        auto* region = memory_pool.allocate_persistent(input_ids_count * sizeof(int32_t), "input_ids");
        if (region) {
            // Copy data to the region
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, input_ids_ptr, input_ids_count * sizeof(int32_t));

            // Create tensor view
            input_ids = MetalTensor<int32_t>::createView(pool_buffer, {input_ids_count}, region->offset);
        }
    }

    if (position_ids_ptr && position_ids_count > 0) {
        auto* region = memory_pool.allocate_persistent(position_ids_count * sizeof(int32_t), "position_ids");
        if (region) {
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, position_ids_ptr, position_ids_count * sizeof(int32_t));
            position_ids = MetalTensor<int32_t>::createView(pool_buffer, {position_ids_count}, region->offset);
        }
    }

    if (kv_page_indices_ptr && kv_page_indices_count > 0) {
        auto* region = memory_pool.allocate_persistent(kv_page_indices_count * sizeof(int32_t), "kv_page_indices");
        if (region) {
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, kv_page_indices_ptr, kv_page_indices_count * sizeof(int32_t));
            kv_page_indices = MetalTensor<int32_t>::createView(pool_buffer, {kv_page_indices_count}, region->offset);
        }
    }

    if (kv_page_indptr_ptr && kv_page_indptr_count > 0) {
        auto* region = memory_pool.allocate_persistent(kv_page_indptr_count * sizeof(int32_t), "kv_page_indptr");
        if (region) {
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, kv_page_indptr_ptr, kv_page_indptr_count * sizeof(int32_t));
            kv_page_indptr = MetalTensor<int32_t>::createView(pool_buffer, {kv_page_indptr_count}, region->offset);
        }
    }

    if (kv_last_page_lens_ptr && kv_last_page_lens_count > 0) {
        auto* region = memory_pool.allocate_persistent(kv_last_page_lens_count * sizeof(int32_t), "kv_last_page_lens");
        if (region) {
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, kv_last_page_lens_ptr, kv_last_page_lens_count * sizeof(int32_t));
            kv_last_page_lens = MetalTensor<int32_t>::createView(pool_buffer, {kv_last_page_lens_count}, region->offset);
        }
    }

    if (qo_indptr_ptr && qo_indptr_count > 0) {
        auto* region = memory_pool.allocate_persistent(qo_indptr_count * sizeof(int32_t), "qo_indptr");
        if (region) {
            void* base_ptr = [pool_buffer contents];
            int32_t* region_ptr = reinterpret_cast<int32_t*>(static_cast<char*>(base_ptr) + region->offset);
            std::memcpy(region_ptr, qo_indptr_ptr, qo_indptr_count * sizeof(int32_t));
            qo_indptr = MetalTensor<int32_t>::createView(pool_buffer, {qo_indptr_count}, region->offset);
        }
    }

    // Note: page_size is set in constructor and cannot be changed here

    std::cout << "Planned buffer with zero-copy memory mapping, "
              << num_tokens << " tokens, " << batch_size << " batches" << std::endl;
}

template <typename T>
void MetalL4maBuffer<T>::mapHostMemory(void* host_ptr, size_t size) {
    // Create a Metal buffer that directly maps host memory
    auto& context = MetalContext::getInstance();
    mapped_host_buffer_ = [context.getDevice() newBufferWithBytesNoCopy:host_ptr
                                                                  length:size
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
    using_zero_copy_ = true;

    std::cout << "Mapped " << size << " bytes of host memory for zero-copy access" << std::endl;
}

template <typename T>
bool MetalL4maBuffer<T>::initializePersistentWorkspaces(PersistentMemoryPool& memory_pool) {
    memory_pool_ = &memory_pool;
    layer_workspaces_.clear();
    layer_workspaces_.reserve(config.num_layers);

    // Calculate workspace size per layer
    size_t hidden_size = config.hidden_size;
    size_t workspace_size_per_layer = num_tokens * hidden_size * sizeof(T) * 4; // hidden + attn + mlp + residual

    for (int layer_idx = 0; layer_idx < config.num_layers; ++layer_idx) {
        auto* region = memory_pool.allocate_persistent(workspace_size_per_layer,
                                                       "layer_" + std::to_string(layer_idx) + "_workspace");
        if (!region) {
            std::cerr << "Failed to allocate workspace for layer " << layer_idx << std::endl;
            return false;
        }

        layer_workspaces_.emplace_back(
            std::make_unique<LayerWorkspace>(region, memory_pool.getBuffer(), config, num_tokens)
        );
    }

    std::cout << "Initialized persistent workspaces for " << config.num_layers << " layers" << std::endl;
    return true;
}

template <typename T>
typename MetalL4maBuffer<T>::LayerWorkspace* MetalL4maBuffer<T>::getLayerWorkspace(size_t layer_idx) {
    if (layer_idx >= layer_workspaces_.size()) {
        return nullptr;
    }
    return layer_workspaces_[layer_idx].get();
}

template <typename T>
void MetalL4maBuffer<T>::resetTemporaryRegions() {
    if (memory_pool_) {
        memory_pool_->reset_temporary();
    }
}

// LayerWorkspace constructor implementation
template <typename T>
MetalL4maBuffer<T>::LayerWorkspace::LayerWorkspace(PersistentMemoryPool::MemoryRegion* region,
                                                   id<MTLBuffer> buffer,
                                                   const L4maConfig& config,
                                                   size_t num_tokens)
    : hidden_states(buffer, {num_tokens, static_cast<size_t>(config.hidden_size)}, region->offset),
      attention_output(buffer, {num_tokens, static_cast<size_t>(config.hidden_size)}, region->offset + num_tokens * config.hidden_size * sizeof(T)),
      mlp_output(buffer, {num_tokens, static_cast<size_t>(config.hidden_size)}, region->offset + 2 * num_tokens * config.hidden_size * sizeof(T)),
      residual_buffer(buffer, {num_tokens, static_cast<size_t>(config.hidden_size)}, region->offset + 3 * num_tokens * config.hidden_size * sizeof(T)) {
}

template <typename T>
template <typename U>
MetalTensor<U> MetalL4maBuffer<T>::allocate(size_t count) {
    return allocator_->allocate<U>(count);
}

template <typename T>
MetalBuffer<uint8_t> MetalL4maBuffer<T>::allocate_rest() {
    return allocator_->allocate_rest();
}

template <typename T>
template <typename U>
void MetalL4maBuffer<T>::deallocate(MetalTensor<U>& tensor) {
    allocator_->deallocate(tensor);
}