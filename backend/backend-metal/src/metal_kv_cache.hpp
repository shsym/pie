#pragma once

#include "metal_common.hpp"
#include "metal_tensor.hpp"
#include <Metal/Metal.h>
#include <utility>
#include <vector>

// Forward declaration
struct L4maConfig;

/**
 * @brief Metal equivalent of L4maKVCache from l4ma.cuh
 * 
 * Manages paged Key-Value cache memory for all transformer layers using Metal buffers.
 * Provides layer-specific cache pointers and handles memory layout efficiently.
 */
template <typename T>
class MetalL4maKVCache {
public:
    /**
     * @brief Calculate total device memory required for KV cache
     * @param config Model configuration containing layer and head information
     * @param num_kv_pages Total number of pages available for the cache
     * @param page_size Number of tokens each page can hold
     * @return Required memory size in bytes
     */
    static size_t get_workspace_size(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size);
    
    /**
     * @brief Construct KV cache with specified configuration
     * @param config Model configuration
     * @param num_kv_pages Total number of pages for the cache
     * @param page_size Size of each page in tokens
     */
    MetalL4maKVCache(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size);
    
    ~MetalL4maKVCache() = default;
    
    // Delete copy/move operations
    MetalL4maKVCache(const MetalL4maKVCache&) = delete;
    MetalL4maKVCache& operator=(const MetalL4maKVCache&) = delete;
    MetalL4maKVCache(MetalL4maKVCache&&) = delete;
    MetalL4maKVCache& operator=(MetalL4maKVCache&&) = delete;
    
    /**
     * @brief Get key and value cache pointers for a specific layer
     * @param layer_idx Index of the transformer decoder layer
     * @return Pair of (key_cache_ptr, value_cache_ptr) for the layer
     */
    std::pair<T*, T*> get_layer_pointers(size_t layer_idx);
    
    /**
     * @brief Get Metal buffers for a specific layer
     * @param layer_idx Index of the transformer decoder layer
     * @return Pair of (key_buffer, value_buffer) for the layer
     */
    std::pair<id<MTLBuffer>, id<MTLBuffer>> get_layer_buffers(size_t layer_idx);
    
    /**
     * @brief Get the underlying Metal buffer for the entire cache
     * @return Metal buffer containing all KV cache data
     */
    id<MTLBuffer> getKVCacheBuffer() const { return kv_cache_.getMetalBuffer(); }
    
    /**
     * @brief Copy KV cache data between pages
     * @param commandQueue Metal command queue for GPU operations
     * @param source_layer_idx Source layer index
     * @param dest_layer_idx Destination layer index
     * @param source_page Source page ID
     * @param dest_page Destination page ID
     * @param source_start Starting position within source page
     * @param dest_start Starting position within destination page
     * @param length Number of tokens to copy
     */
    void copy_kv_data(
        id<MTLCommandQueue> commandQueue,
        size_t source_layer_idx, size_t dest_layer_idx,
        uint32_t source_page, uint32_t dest_page,
        uint32_t source_start, uint32_t dest_start,
        uint32_t length
    );
    
    /**
     * @brief Zero out KV cache data for specific pages
     * @param commandQueue Metal command queue for GPU operations
     * @param layer_idx Layer index to clear
     * @param page_id Page ID to clear
     * @param start_pos Starting position within page
     * @param length Number of tokens to clear
     */
    void clear_kv_data(
        id<MTLCommandQueue> commandQueue,
        size_t layer_idx,
        uint32_t page_id,
        uint32_t start_pos,
        uint32_t length
    );
    
    // Query methods
    int32_t get_num_pages() const { return num_kv_pages_; }
    int32_t get_page_size() const { return page_size_; }
    size_t get_num_layers() const { return config_.num_layers; }
    size_t get_num_heads() const { return config_.num_key_value_heads; }
    size_t get_head_size() const { return config_.head_size; }
    
    // Memory layout information
    size_t get_layer_offset(size_t layer_idx) const;
    size_t get_page_offset(size_t layer_idx, uint32_t page_id) const;
    size_t get_token_offset(size_t layer_idx, uint32_t page_id, uint32_t token_pos) const;

private:
    void calculate_memory_layout();
    size_t calculate_layer_size() const;
    
    const L4maConfig& config_;
    int32_t num_kv_pages_;
    int32_t page_size_;
    
    // Single Metal tensor for both K and V caches
    MetalTensor<T> kv_cache_;
    
    // Memory layout information
    size_t elements_per_token_;  // num_kv_heads * head_size
    size_t elements_per_page_;   // elements_per_token * page_size
    size_t elements_per_layer_;  // elements_per_page * num_kv_pages * 2 (K + V)
    size_t k_cache_offset_;      // Offset for K cache within each layer
    size_t v_cache_offset_;      // Offset for V cache within each layer
};

/**
 * @brief KV cache utility functions
 */
namespace MetalKVCacheUtils {
    
    /**
     * @brief Calculate memory requirements for KV cache
     * @param num_layers Number of transformer layers
     * @param num_kv_heads Number of key-value heads
     * @param head_size Size of each attention head
     * @param num_pages Number of cache pages
     * @param page_size Tokens per page
     * @return Memory size in bytes
     */
    template<typename T>
    size_t calculate_kv_memory_size(
        size_t num_layers,
        size_t num_kv_heads, 
        size_t head_size,
        int32_t num_pages,
        int32_t page_size
    );
    
    /**
     * @brief Validate KV cache configuration
     * @param config Model configuration
     * @param num_pages Number of pages
     * @param page_size Page size
     * @return true if configuration is valid
     */
    bool validate_kv_config(const L4maConfig& config, int32_t num_pages, int32_t page_size);
}

// Template implementations

template <typename T>
size_t MetalL4maKVCache<T>::get_workspace_size(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size) {
    return MetalKVCacheUtils::calculate_kv_memory_size<T>(
        config.num_layers,
        config.num_key_value_heads,
        config.head_size,
        num_kv_pages,
        page_size
    );
}

template <typename T>
MetalL4maKVCache<T>::MetalL4maKVCache(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size)
    : config_(config), num_kv_pages_(num_kv_pages), page_size_(page_size) {
    
    // Validate configuration
    if (!MetalKVCacheUtils::validate_kv_config(config, num_kv_pages, page_size)) {
        throw std::runtime_error("Invalid KV cache configuration");
    }
    
    // Calculate memory layout
    calculate_memory_layout();
    
    // Allocate the KV cache tensor
    size_t total_elements = config_.num_layers * elements_per_layer_;
    kv_cache_ = MetalTensor<T>({total_elements});
    
    // Zero initialize the cache
    kv_cache_.zero();
}

template <typename T>
std::pair<T*, T*> MetalL4maKVCache<T>::get_layer_pointers(size_t layer_idx) {
    if (layer_idx >= config_.num_layers) {
        throw std::runtime_error("Layer index out of bounds");
    }
    
    T* base_ptr = kv_cache_.data();
    T* layer_base = base_ptr + layer_idx * elements_per_layer_;
    
    T* k_ptr = layer_base + k_cache_offset_;
    T* v_ptr = layer_base + v_cache_offset_;
    
    return {k_ptr, v_ptr};
}

template <typename T>
std::pair<id<MTLBuffer>, id<MTLBuffer>> MetalL4maKVCache<T>::get_layer_buffers(size_t layer_idx) {
    if (layer_idx >= config_.num_layers) {
        throw std::runtime_error("Layer index out of bounds");
    }
    
    // For now, return the same buffer for both K and V
    // In a more sophisticated implementation, we might create buffer views
    id<MTLBuffer> buffer = kv_cache_.getMetalBuffer();
    return {buffer, buffer};
}

template <typename T>
void MetalL4maKVCache<T>::copy_kv_data(
    id<MTLCommandQueue> commandQueue,
    size_t source_layer_idx, size_t dest_layer_idx,
    uint32_t source_page, uint32_t dest_page,
    uint32_t source_start, uint32_t dest_start,
    uint32_t length
) {
    if (source_layer_idx >= config_.num_layers || dest_layer_idx >= config_.num_layers) {
        throw std::runtime_error("Layer index out of bounds");
    }
    
    if (source_page >= num_kv_pages_ || dest_page >= num_kv_pages_) {
        throw std::runtime_error("Page ID out of bounds");
    }
    
    size_t copy_elements = length * elements_per_token_;
    size_t copy_bytes = copy_elements * sizeof(T);
    
    id<MTLBuffer> buffer = kv_cache_.getMetalBuffer();
    
    // Copy K cache
    size_t source_k_offset = get_token_offset(source_layer_idx, source_page, source_start);
    size_t dest_k_offset = get_token_offset(dest_layer_idx, dest_page, dest_start);
    
    MetalMemory::copyBuffer(commandQueue, 
                           buffer, source_k_offset * sizeof(T),
                           buffer, dest_k_offset * sizeof(T),
                           copy_bytes);
    
    // Copy V cache
    size_t source_v_offset = source_k_offset + (elements_per_layer_ / 2);
    size_t dest_v_offset = dest_k_offset + (elements_per_layer_ / 2);
    
    MetalMemory::copyBuffer(commandQueue,
                           buffer, source_v_offset * sizeof(T),
                           buffer, dest_v_offset * sizeof(T),
                           copy_bytes);
}

template <typename T>
void MetalL4maKVCache<T>::clear_kv_data(
    id<MTLCommandQueue> commandQueue,
    size_t layer_idx,
    uint32_t page_id,
    uint32_t start_pos,
    uint32_t length
) {
    if (layer_idx >= config_.num_layers) {
        throw std::runtime_error("Layer index out of bounds");
    }
    
    if (page_id >= num_kv_pages_) {
        throw std::runtime_error("Page ID out of bounds");
    }
    
    size_t clear_elements = length * elements_per_token_;
    size_t clear_bytes = clear_elements * sizeof(T);
    
    id<MTLBuffer> buffer = kv_cache_.getMetalBuffer();
    
    // Clear K cache
    size_t k_offset = get_token_offset(layer_idx, page_id, start_pos);
    MetalMemory::zeroBuffer(commandQueue, buffer, clear_bytes);
    
    // Clear V cache
    size_t v_offset = k_offset + (elements_per_layer_ / 2);
    MetalMemory::zeroBuffer(commandQueue, buffer, clear_bytes);
}

template <typename T>
size_t MetalL4maKVCache<T>::get_layer_offset(size_t layer_idx) const {
    return layer_idx * elements_per_layer_;
}

template <typename T>
size_t MetalL4maKVCache<T>::get_page_offset(size_t layer_idx, uint32_t page_id) const {
    return get_layer_offset(layer_idx) + page_id * elements_per_page_;
}

template <typename T>
size_t MetalL4maKVCache<T>::get_token_offset(size_t layer_idx, uint32_t page_id, uint32_t token_pos) const {
    return get_page_offset(layer_idx, page_id) + token_pos * elements_per_token_;
}

template <typename T>
void MetalL4maKVCache<T>::calculate_memory_layout() {
    elements_per_token_ = config_.num_key_value_heads * config_.head_size;
    elements_per_page_ = elements_per_token_ * page_size_;
    
    // Each layer has K and V caches
    size_t k_cache_size = num_kv_pages_ * elements_per_page_;
    size_t v_cache_size = num_kv_pages_ * elements_per_page_;
    elements_per_layer_ = k_cache_size + v_cache_size;
    
    // K cache comes first, then V cache
    k_cache_offset_ = 0;
    v_cache_offset_ = k_cache_size;
}

template <typename T>
size_t MetalL4maKVCache<T>::calculate_layer_size() const {
    return elements_per_layer_ * sizeof(T);
}

// Utility function implementations
namespace MetalKVCacheUtils {
    
    template<typename T>
    size_t calculate_kv_memory_size(
        size_t num_layers,
        size_t num_kv_heads,
        size_t head_size,
        int32_t num_pages,
        int32_t page_size
    ) {
        size_t elements_per_token = num_kv_heads * head_size;
        size_t elements_per_page = elements_per_token * page_size;
        size_t elements_per_layer = elements_per_page * num_pages * 2; // K + V
        size_t total_elements = elements_per_layer * num_layers;
        
        return total_elements * sizeof(T);
    }
    
    bool validate_kv_config(const L4maConfig& config, int32_t num_pages, int32_t page_size);
}