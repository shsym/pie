#import "metal_kv_cache.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <stdexcept>

// Explicit template instantiations for common types
template class MetalL4maKVCache<float>;
template class MetalL4maKVCache<bfloat16_t>;

// MetalKVCacheUtils explicit template instantiations
namespace MetalKVCacheUtils {
    
    // Template instantiations for calculate_kv_memory_size
    template size_t calculate_kv_memory_size<float>(
        size_t num_layers,
        size_t num_kv_heads,
        size_t head_size,
        int32_t num_pages,
        int32_t page_size
    );
    
    template size_t calculate_kv_memory_size<bfloat16_t>(
        size_t num_layers,
        size_t num_kv_heads,
        size_t head_size,
        int32_t num_pages,
        int32_t page_size
    );
    
    // Implementation of validate_kv_config (already defined in header but needs to be in .mm for linking)
    bool validate_kv_config(const L4maConfig& config, int32_t num_pages, int32_t page_size) {
        if (num_pages <= 0 || page_size <= 0) {
            std::cerr << "Invalid KV cache parameters: num_pages=" << num_pages 
                      << ", page_size=" << page_size << std::endl;
            return false;
        }
        
        if (config.num_layers == 0 || config.num_key_value_heads == 0 || config.head_size == 0) {
            std::cerr << "Invalid model config: num_layers=" << config.num_layers
                      << ", num_kv_heads=" << config.num_key_value_heads
                      << ", head_size=" << config.head_size << std::endl;
            return false;
        }
        
        // Check for reasonable limits to prevent memory overflow
        const size_t MAX_PAGES = 1000000;  // Arbitrary large limit
        const size_t MAX_PAGE_SIZE = 8192; // Arbitrary large limit
        
        if (static_cast<size_t>(num_pages) > MAX_PAGES || static_cast<size_t>(page_size) > MAX_PAGE_SIZE) {
            std::cerr << "KV cache parameters exceed limits: num_pages=" << num_pages 
                      << " (max: " << MAX_PAGES << "), page_size=" << page_size
                      << " (max: " << MAX_PAGE_SIZE << ")" << std::endl;
            return false;
        }
        
        // Calculate memory requirements and check if reasonable
        size_t memory_size_bytes = calculate_kv_memory_size<float>(
            config.num_layers,
            config.num_key_value_heads,
            config.head_size,
            num_pages,
            page_size
        );
        
        // Warn if memory usage is very large (>1GB)
        const size_t LARGE_MEMORY_THRESHOLD = 1ULL << 30; // 1GB
        if (memory_size_bytes > LARGE_MEMORY_THRESHOLD) {
            std::cout << "Warning: KV cache will use " << (memory_size_bytes / (1024.0 * 1024.0)) 
                      << " MB of GPU memory" << std::endl;
        }
        
        return true;
    }
}

// Additional helper functions that might be needed

/**
 * @brief Helper function to print KV cache statistics
 */
template<typename T>
void print_kv_cache_info(const MetalL4maKVCache<T>& cache) {
    std::cout << "=== MetalL4maKVCache Info ===" << std::endl;
    std::cout << "Num layers: " << cache.get_num_layers() << std::endl;
    std::cout << "Num KV heads: " << cache.get_num_heads() << std::endl;
    std::cout << "Head size: " << cache.get_head_size() << std::endl;
    std::cout << "Num pages: " << cache.get_num_pages() << std::endl;
    std::cout << "Page size: " << cache.get_page_size() << std::endl;
    
    size_t total_memory = MetalL4maKVCache<T>::get_workspace_size(
        L4maConfig{}, // This would need the actual config
        cache.get_num_pages(), 
        cache.get_page_size()
    );
    
    std::cout << "Total memory: " << (total_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "============================" << std::endl;
}

// Explicit instantiations for helper functions
template void print_kv_cache_info<float>(const MetalL4maKVCache<float>& cache);
template void print_kv_cache_info<bfloat16_t>(const MetalL4maKVCache<bfloat16_t>& cache);