#import "metal_buffer.hpp"
#import "metal_common.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>
#include <iostream>

// MetalStackAllocator Implementation
MetalStackAllocator::MetalStackAllocator(size_t total_size) 
    : total_size_(total_size), current_offset_(0) {
    
    // Allocate the main buffer
    buffer_ = MetalBuffer<uint8_t>(total_size);
    if (!buffer_.isValid()) {
        throw std::runtime_error("Failed to allocate Metal stack buffer");
    }
}

MetalBuffer<uint8_t> MetalStackAllocator::allocate_rest() {
    size_t remaining = available_size();
    if (remaining == 0) {
        throw std::runtime_error("No remaining memory in stack allocator");
    }
    
    // Create a view of the remaining buffer
    MetalBuffer<uint8_t> rest_buffer(remaining);
    
    // Copy the remaining data from the main buffer
    auto& context = MetalContext::getInstance();
    MetalMemory::copyBuffer(context.getCommandQueue(),
                           buffer_.getBuffer(), current_offset_,
                           rest_buffer.getBuffer(), 0,
                           remaining);
    
    current_offset_ = total_size_; // Mark all memory as used
    return rest_buffer;
}

void MetalStackAllocator::reset() {
    current_offset_ = 0;
}

size_t MetalStackAllocator::align_offset(size_t offset, size_t alignment) const {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// MetalBatchPrefillHandler Implementation
MetalBatchPrefillHandler::MetalBatchPrefillHandler() : initialized_(false), workspace_size_(0) {
}

bool MetalBatchPrefillHandler::initialize(size_t workspace_size) {
    try {
        workspace_ = MetalBuffer<uint8_t>(workspace_size);
        workspace_size_ = workspace_size;
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize MetalBatchPrefillHandler: " << e.what() << std::endl;
        initialized_ = false;
        return false;
    }
}

template<typename DTypeQ, typename DTypeKV, typename DTypeO>
bool MetalBatchPrefillHandler::plan(
    id<MTLCommandBuffer> commandBuffer,
    const std::vector<int32_t>& qo_indptr_host,
    const std::vector<int32_t>& kv_page_indices_host,
    const std::vector<int32_t>& kv_page_indptr_host,
    const std::vector<int32_t>& kv_last_page_lens_host,
    size_t num_qo_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t page_size
) {
    if (!initialized_) {
        std::cerr << "Handler not initialized" << std::endl;
        return false;
    }
    
    // Store configuration for forward pass
    // In a full implementation, this would set up workspace allocation
    // and prepare for the attention computation
    
    return true;
}

template<typename DTypeQ, typename DTypeKV, typename DTypeO>
void MetalBatchPrefillHandler::forward(
    id<MTLCommandBuffer> commandBuffer,
    const DTypeQ* q_data,
    const DTypeKV* kv_data,
    DTypeO* o_data,
    const std::vector<bool>& custom_mask,
    const std::vector<int32_t>& mask_indptr
) {
    if (!initialized_) {
        throw std::runtime_error("Handler not initialized");
    }
    
    // This would delegate to the metal_batch_prefill_attention kernel
    // For now, this is a placeholder that would need to be connected
    // to the actual Metal kernel implementation
    
    std::cerr << "MetalBatchPrefillHandler::forward not fully implemented" << std::endl;
}

// Explicit template instantiations for common types
template class MetalL4maBuffer<float>;
template class MetalL4maBuffer<bfloat16_t>;

// MetalBatchPrefillHandler template instantiations
template bool MetalBatchPrefillHandler::plan<float, float, float>(
    id<MTLCommandBuffer>, const std::vector<int32_t>&, const std::vector<int32_t>&, 
    const std::vector<int32_t>&, const std::vector<int32_t>&, size_t, size_t, size_t, size_t);

template bool MetalBatchPrefillHandler::plan<bfloat16_t, bfloat16_t, bfloat16_t>(
    id<MTLCommandBuffer>, const std::vector<int32_t>&, const std::vector<int32_t>&, 
    const std::vector<int32_t>&, const std::vector<int32_t>&, size_t, size_t, size_t, size_t);

template void MetalBatchPrefillHandler::forward<float, float, float>(
    id<MTLCommandBuffer>, const float*, const float*, float*, 
    const std::vector<bool>&, const std::vector<int32_t>&);

template void MetalBatchPrefillHandler::forward<bfloat16_t, bfloat16_t, bfloat16_t>(
    id<MTLCommandBuffer>, const bfloat16_t*, const bfloat16_t*, bfloat16_t*, 
    const std::vector<bool>&, const std::vector<int32_t>&);

// PersistentMemoryPool Implementation
PersistentMemoryPool::PersistentMemoryPool(size_t total_size) 
    : total_size_(total_size), persistent_watermark_(0) {
}

bool PersistentMemoryPool::initialize() {
    // Allocate the main buffer
    buffer_ = MetalBuffer<uint8_t>(total_size_);
    if (!buffer_.isValid()) {
        std::cerr << "Failed to allocate persistent memory pool buffer" << std::endl;
        return false;
    }
    
    // Initialize with one large free region covering the entire buffer
    regions_.emplace_back(std::make_unique<MemoryRegion>(0, total_size_, false, "free"));
    
    std::cout << "Initialized persistent memory pool with " << total_size_ << " bytes" << std::endl;
    return true;
}

PersistentMemoryPool::MemoryRegion* PersistentMemoryPool::allocate_persistent(size_t size, const std::string& name) {
    size_t aligned_size = (size + 15) & ~15; // Align to 16 bytes
    size_t region_idx = find_free_region(aligned_size, true);
    
    if (region_idx == SIZE_MAX) {
        std::cerr << "Failed to allocate persistent region of size " << size << std::endl;
        return nullptr;
    }
    
    MemoryRegion* free_region = regions_[region_idx].get();
    size_t offset = free_region->offset;
    
    // Split the free region if necessary
    if (free_region->size > aligned_size) {
        // Create new region for the remainder
        size_t remaining_offset = offset + aligned_size;
        size_t remaining_size = free_region->size - aligned_size;
        regions_.emplace_back(std::make_unique<MemoryRegion>(remaining_offset, remaining_size, false, "free"));
    }
    
    // Convert the free region to allocated persistent region
    free_region->size = aligned_size;
    free_region->in_use = true;
    free_region->is_persistent = true;
    free_region->name = name.empty() ? "persistent" : name;
    
    // Update persistent watermark
    if (offset + aligned_size > persistent_watermark_) {
        persistent_watermark_ = offset + aligned_size;
    }
    
    std::cout << "Allocated persistent region '" << free_region->name 
              << "' at offset " << offset << ", size " << aligned_size << std::endl;
    
    return free_region;
}

PersistentMemoryPool::MemoryRegion* PersistentMemoryPool::allocate_temporary(size_t size, const std::string& name) {
    size_t aligned_size = (size + 15) & ~15; // Align to 16 bytes
    size_t region_idx = find_free_region(aligned_size, false);
    
    if (region_idx == SIZE_MAX) {
        std::cerr << "Failed to allocate temporary region of size " << size << std::endl;
        return nullptr;
    }
    
    MemoryRegion* free_region = regions_[region_idx].get();
    size_t offset = free_region->offset;
    
    // Split the free region if necessary
    if (free_region->size > aligned_size) {
        // Create new region for the remainder
        size_t remaining_offset = offset + aligned_size;
        size_t remaining_size = free_region->size - aligned_size;
        regions_.emplace_back(std::make_unique<MemoryRegion>(remaining_offset, remaining_size, false, "free"));
    }
    
    // Convert the free region to allocated temporary region
    free_region->size = aligned_size;
    free_region->in_use = true;
    free_region->is_persistent = false;
    free_region->name = name.empty() ? "temporary" : name;
    
    std::cout << "Allocated temporary region '" << free_region->name 
              << "' at offset " << offset << ", size " << aligned_size << std::endl;
    
    return free_region;
}

void PersistentMemoryPool::free(MemoryRegion* region) {
    if (!region || !region->in_use) {
        return;
    }
    
    region->in_use = false;
    region->is_persistent = false;
    region->name = "free";
    
    // Coalesce adjacent free regions
    coalesce_free_regions();
}

void PersistentMemoryPool::reset_temporary() {
    // Free all temporary regions
    for (auto& region : regions_) {
        if (region->in_use && !region->is_persistent) {
            region->in_use = false;
            region->is_persistent = false;
            region->name = "free";
        }
    }
    
    // Coalesce free regions
    coalesce_free_regions();
    
    std::cout << "Reset temporary regions in memory pool" << std::endl;
}

void PersistentMemoryPool::reset_all() {
    // Free all regions
    regions_.clear();
    regions_.emplace_back(std::make_unique<MemoryRegion>(0, total_size_, false, "free"));
    persistent_watermark_ = 0;
    
    std::cout << "Reset all regions in memory pool" << std::endl;
}

template<typename T>
MetalTensorView<T> PersistentMemoryPool::create_tensor_view(MemoryRegion* region, const std::vector<size_t>& shape) {
    if (!region || !region->in_use) {
        throw std::runtime_error("Cannot create tensor view from invalid or free region");
    }
    
    // Calculate required size
    size_t total_elements = 1;
    for (size_t dim : shape) {
        total_elements *= dim;
    }
    size_t required_bytes = total_elements * sizeof(T);
    
    if (required_bytes > region->size) {
        throw std::runtime_error("Tensor shape too large for memory region");
    }
    
    return MetalTensorView<T>(buffer_.getBuffer(), shape, region->offset);
}

size_t PersistentMemoryPool::allocated_size() const {
    size_t total = 0;
    for (const auto& region : regions_) {
        if (region->in_use) {
            total += region->size;
        }
    }
    return total;
}

size_t PersistentMemoryPool::persistent_allocated() const {
    size_t total = 0;
    for (const auto& region : regions_) {
        if (region->in_use && region->is_persistent) {
            total += region->size;
        }
    }
    return total;
}

size_t PersistentMemoryPool::temporary_allocated() const {
    size_t total = 0;
    for (const auto& region : regions_) {
        if (region->in_use && !region->is_persistent) {
            total += region->size;
        }
    }
    return total;
}

void PersistentMemoryPool::print_layout() const {
    std::cout << "=== Memory Pool Layout ===" << std::endl;
    std::cout << "Total size: " << total_size_ << " bytes" << std::endl;
    std::cout << "Allocated: " << allocated_size() << " bytes" << std::endl;
    std::cout << "Available: " << available_size() << " bytes" << std::endl;
    std::cout << "Persistent: " << persistent_allocated() << " bytes" << std::endl;
    std::cout << "Temporary: " << temporary_allocated() << " bytes" << std::endl;
    
    for (const auto& region : regions_) {
        std::cout << "  [" << region->offset << "-" << (region->offset + region->size) 
                  << "] " << region->size << " bytes - " << region->name;
        if (region->in_use) {
            std::cout << " (in_use, " << (region->is_persistent ? "persistent" : "temporary") << ")";
        } else {
            std::cout << " (free)";
        }
        std::cout << std::endl;
    }
    std::cout << "=========================" << std::endl;
}

bool PersistentMemoryPool::validate_layout() const {
    // Check for overlapping regions
    for (size_t i = 0; i < regions_.size(); ++i) {
        for (size_t j = i + 1; j < regions_.size(); ++j) {
            const auto& r1 = regions_[i];
            const auto& r2 = regions_[j];
            
            if (!(r1->offset + r1->size <= r2->offset || r2->offset + r2->size <= r1->offset)) {
                std::cerr << "Overlapping regions detected: " << r1->name << " and " << r2->name << std::endl;
                return false;
            }
        }
    }
    
    // Check that all regions fit within total size
    for (const auto& region : regions_) {
        if (region->offset + region->size > total_size_) {
            std::cerr << "Region " << region->name << " exceeds total buffer size" << std::endl;
            return false;
        }
    }
    
    return true;
}

size_t PersistentMemoryPool::find_free_region(size_t size, bool persistent) {
    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto& region = regions_[i];
        if (!region->in_use && region->size >= size) {
            // For persistent allocations, prefer regions at the beginning
            // For temporary allocations, prefer regions after the persistent watermark
            if (persistent || region->offset >= persistent_watermark_) {
                return i;
            }
        }
    }
    return SIZE_MAX; // Not found
}

void PersistentMemoryPool::coalesce_free_regions() {
    // Sort regions by offset
    std::sort(regions_.begin(), regions_.end(), 
              [](const std::unique_ptr<MemoryRegion>& a, const std::unique_ptr<MemoryRegion>& b) {
                  return a->offset < b->offset;
              });
    
    // Coalesce adjacent free regions
    for (size_t i = 0; i < regions_.size(); ) {
        auto& current = regions_[i];
        if (current->in_use) {
            ++i;
            continue;
        }
        
        // Look for adjacent free regions
        size_t j = i + 1;
        while (j < regions_.size() && 
               !regions_[j]->in_use && 
               current->offset + current->size == regions_[j]->offset) {
            
            // Merge regions[j] into current
            current->size += regions_[j]->size;
            regions_.erase(regions_.begin() + j);
        }
        
        ++i;
    }
}

// Explicit template instantiations for PersistentMemoryPool::create_tensor_view
template MetalTensorView<float> PersistentMemoryPool::create_tensor_view<float>(MemoryRegion*, const std::vector<size_t>&);
template MetalTensorView<bfloat16_t> PersistentMemoryPool::create_tensor_view<bfloat16_t>(MemoryRegion*, const std::vector<size_t>&);