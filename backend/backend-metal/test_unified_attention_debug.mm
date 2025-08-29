#include "metal_batch_prefill_attention_unified.hpp"
#include "metal_common.hpp"
#include <iostream>

int main() {
    std::cout << "Testing unified attention initialization..." << std::endl;
    
    // Initialize the unified attention system
    bool success = metal::unified_attention::initialize();
    
    if (success) {
        std::cout << "✅ Unified attention initialized successfully" << std::endl;
    } else {
        std::cout << "❌ Unified attention initialization failed" << std::endl;
        return 1;
    }
    
    // Test with simple parameters to see what gets passed
    std::vector<float> debug_out(20, -999.0f);
    
    const int num_qo = 2;
    const int num_sequences = 1; 
    const int head_dim = 8;
    const int head_size = 4;
    const int page_size = 4;
    
    metal::unified_attention::UnifiedParams params = {
        .num_qo = num_qo,
        .num_sequences = num_sequences,
        .head_dim = head_dim,
        .head_size = head_size,
        .num_heads = head_dim / head_size,
        .page_size = page_size,
        .max_seq_len = 4,
        .scale = 0.5f
    };
    
    std::cout << "Parameters being passed:" << std::endl;
    std::cout << "  num_qo: " << params.num_qo << std::endl;
    std::cout << "  num_sequences: " << params.num_sequences << std::endl;
    std::cout << "  head_dim: " << params.head_dim << std::endl;
    std::cout << "  head_size: " << params.head_size << std::endl;
    std::cout << "  num_heads: " << params.num_heads << std::endl;
    std::cout << "  page_size: " << params.page_size << std::endl;
    std::cout << "  max_seq_len: " << params.max_seq_len << std::endl;
    std::cout << "  scale: " << params.scale << std::endl;
    
    // Create minimal test data
    std::vector<uint16_t> q_data(num_qo * head_dim, 0x3c00); // 1.0 in bfloat16
    std::vector<uint16_t> k_data(1 * page_size * head_dim, 0x3c00);
    std::vector<uint16_t> v_data(1 * page_size * head_dim, 0x3c00);
    std::vector<uint16_t> output(num_qo * head_dim, 0);
    
    std::vector<int32_t> qo_indptr = {0, num_qo};
    std::vector<int32_t> kv_page_indptr = {0, 1};
    std::vector<int32_t> kv_page_indices = {0};
    std::vector<int32_t> kv_last_page_lens = {page_size};
    
    try {
        std::cout << "Calling unified attention..." << std::endl;
        
        metal::unified_attention::unified_batch_prefill_attention_bf16(
            (const bfloat16_t*)q_data.data(),
            (const bfloat16_t*)k_data.data(),  
            (const bfloat16_t*)v_data.data(),
            qo_indptr.data(),
            kv_page_indptr.data(),
            kv_page_indices.data(),
            kv_last_page_lens.data(),
            (bfloat16_t*)output.data(),
            params,
            debug_out.data()
        );
        
        std::cout << "✅ Unified attention call completed" << std::endl;
        
        // Check debug output
        std::cout << "Debug output (first 10 values):" << std::endl;
        for (int i = 0; i < 10 && i < debug_out.size(); i++) {
            std::cout << "  debug[" << i << "] = " << debug_out[i] << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}