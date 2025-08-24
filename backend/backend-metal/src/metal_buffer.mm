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