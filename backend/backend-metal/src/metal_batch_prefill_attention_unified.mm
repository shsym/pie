#include "metal_batch_prefill_attention_unified.hpp"
#include "metal_common.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace metal::unified_attention {

// === Global State ===
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_command_queue = nil; 
static id<MTLLibrary> g_library = nil;
static id<MTLFunction> g_unified_function_bf16 = nil; // Store bf16 function for dynamic pipeline creation
static id<MTLFunction> g_unified_function_f32 = nil;  // Store f32 function for dynamic pipeline creation
static bool g_initialized = false;

// === Implementation ===

bool initialize() {
    std::cout << "[UnifiedAttention] Starting initialization..." << std::endl;
    std::cout.flush();
    
    if (g_initialized) {
        std::cout << "[UnifiedAttention] Already initialized" << std::endl;
        std::cout.flush();
        return true;
    }
    
    // Initialize Metal context
    std::cout << "[UnifiedAttention] Initializing Metal context..." << std::endl;
    std::cout.flush();
    MetalContext& context = MetalContext::getInstance();
    if (!context.initialize()) {
        std::cerr << "[UnifiedAttention] Failed to initialize Metal context\n";
        std::cerr.flush();
        return false;
    }
    
    g_device = context.getDevice();
    g_command_queue = context.getCommandQueue();
    
    std::cout << "[UnifiedAttention] Metal context ready - device: " 
              << (g_device ? [[g_device name] UTF8String] : "null") << std::endl;
    std::cout.flush();
    
    // Load Metal library from source
    NSError* error = nil;
    
    // Try FlashAttention kernel first, then fallback to working kernel
    NSArray* possiblePaths = @[
        @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_flashattention.metal",
        @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_working.metal",
        @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/build/shaders/metal_batch_prefill_attention_flashattention.metal",
        @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/build/shaders/metal_batch_prefill_attention_unified.metal",
        @"/Users/seung-seoblee/Dev/pie/backend/backend-metal/src/metal_batch_prefill_attention_unified.metal",
        @"../backend-metal/build/shaders/metal_batch_prefill_attention_unified.metal",
        @"../../backend/backend-metal/build/shaders/metal_batch_prefill_attention_unified.metal"
    ];
    
    NSString* validShaderPath = nil;
    std::cout << "[UnifiedAttention] Searching for shader file..." << std::endl;
    std::cout.flush();
    for (NSString* path in possiblePaths) {
        std::cout << "  Checking: " << path.UTF8String << std::endl;
        std::cout.flush();
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            validShaderPath = path;
            std::cout << "  Found shader at: " << path.UTF8String << std::endl;
            std::cout.flush();
            break;
        }
    }
    
    if (validShaderPath) {
        std::cout << "[UnifiedAttention] Loading shader source from: " << validShaderPath.UTF8String << std::endl;
        NSString* shaderSource = [NSString stringWithContentsOfFile:validShaderPath 
                                                            encoding:NSUTF8StringEncoding 
                                                               error:&error];
        if (error) {
            std::cerr << "[UnifiedAttention] Failed to read shader file: " << error.localizedDescription.UTF8String << std::endl;
        } else if (shaderSource) {
            std::cout << "[UnifiedAttention] Shader source loaded: " << shaderSource.length << " characters" << std::endl;
            
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            // Use newer API instead of deprecated fastMathEnabled
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeFast;
            } else {
                #pragma clang diagnostic push
                #pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = YES;
                #pragma clang diagnostic pop
            }
            
            std::cout << "[UnifiedAttention] Compiling Metal shader..." << std::endl;
            g_library = [g_device newLibraryWithSource:shaderSource options:options error:&error];
            if (g_library) {
                std::cout << "[UnifiedAttention] Successfully compiled Metal library from source" << std::endl;
            } else {
                std::cerr << "[UnifiedAttention] Failed to compile shader: " 
                          << (error ? error.localizedDescription.UTF8String : "Unknown compilation error") << std::endl;
            }
        }
    } else {
        std::cout << "[UnifiedAttention] No valid shader path found, trying default library" << std::endl;
    }
    
    // Fallback to default library
    if (!g_library) {
        std::cout << "[UnifiedAttention] Trying default Metal library..." << std::endl;
        g_library = [g_device newDefaultLibrary];
        if (g_library) {
            std::cout << "[UnifiedAttention] Using default Metal library" << std::endl;
        }
    }
    
    if (!g_library) {
        std::cerr << "[UnifiedAttention] Failed to load any Metal library: " 
                  << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
        return false;
    }
    
    // Load functions for dynamic pipeline creation with function constants
    // Try FlashAttention functions first, then fallback to working kernels
    
    // Load bf16 function
    g_unified_function_bf16 = [g_library newFunctionWithName:@"unified_batch_prefill_attention_bf16_flashattention"];
    if (!g_unified_function_bf16) {
        std::cout << "[UnifiedAttention] BF16 FlashAttention kernel not found, trying working kernel..." << std::endl;
        g_unified_function_bf16 = [g_library newFunctionWithName:@"unified_batch_prefill_attention_bf16_working"];
    }
    
    // Load f32 function
    g_unified_function_f32 = [g_library newFunctionWithName:@"unified_batch_prefill_attention_f32_flashattention"];
    if (!g_unified_function_f32) {
        std::cout << "[UnifiedAttention] F32 FlashAttention kernel not found, trying working kernel..." << std::endl;
        g_unified_function_f32 = [g_library newFunctionWithName:@"unified_batch_prefill_attention_f32_working"];
    }
    
    if (!g_unified_function_bf16 && !g_unified_function_f32) {
        std::cerr << "[UnifiedAttention] Failed to load any unified attention kernels\n";
        return false;
    }
    
    // Log which kernels are being used
    if (g_unified_function_bf16) {
        NSString* functionName = g_unified_function_bf16.name;
        std::cout << "[UnifiedAttention] Loaded BF16 kernel: " << functionName.UTF8String << std::endl;
    }
    if (g_unified_function_f32) {
        NSString* functionName = g_unified_function_f32.name;
        std::cout << "[UnifiedAttention] Loaded F32 kernel: " << functionName.UTF8String << std::endl;
    }
    
    g_initialized = true;
    std::cout << "[UnifiedAttention] Successfully initialized\n";
    return true;
}

void cleanup() {
    g_unified_function_bf16 = nil;
    g_unified_function_f32 = nil;
    g_library = nil;
    g_command_queue = nil;
    g_device = nil;
    g_initialized = false;
}

bool is_initialized() {
    return g_initialized;
}

TileConfig calculate_tile_config(
    const std::vector<int>& sequence_lengths,
    int head_dim,
    int page_size
) {
    TileConfig config = {};
    
    // Find maximum sequence length
    int max_seq_len = *std::max_element(sequence_lengths.begin(), sequence_lengths.end());
    
    // Dynamic tile size selection based on sequence characteristics
    if (max_seq_len <= 512) {
        config.q_tile_size = 32;
        config.kv_tile_size = 32;
    } else if (max_seq_len <= 2048) {
        config.q_tile_size = 64;
        config.kv_tile_size = 64;
    } else {
        config.q_tile_size = 128;
        config.kv_tile_size = 128;
    }
    
    // Calculate maximum tiles needed across all sequences
    config.max_q_tiles_global = 0;
    config.max_kv_tiles_global = 0;
    
    for (int seq_len : sequence_lengths) {
        int q_tiles = (seq_len + config.q_tile_size - 1) / config.q_tile_size;
        int kv_tiles = (seq_len + config.kv_tile_size - 1) / config.kv_tile_size;
        
        config.max_q_tiles_global = std::max(config.max_q_tiles_global, q_tiles);
        config.max_kv_tiles_global = std::max(config.max_kv_tiles_global, kv_tiles);
    }
    
    // For compatibility, set average values
    config.num_q_tiles = config.max_q_tiles_global;
    config.num_kv_tiles = config.max_kv_tiles_global;
    
    return config;
}

DispatchConfig generate_dispatch_config(
    const TileConfig& tile_config,
    int num_sequences,
    int num_heads
) {
    DispatchConfig dispatch = {};
    
    // 3D dispatch grid: [q_tiles, kv_tiles, batch_heads]
    // Note: Each threadgroup processes one (q_tile, kv_tile, head) combination
    dispatch.threadgroups_per_grid = MTLSizeMake(
        tile_config.max_q_tiles_global,     // x: query tiles
        tile_config.max_kv_tiles_global,    // y: KV tiles  
        num_sequences * num_heads           // z: batch_head combinations
    );
    
    // Threadgroup size - optimize for tile operations
    int threads_per_group = std::min(DEFAULT_THREADGROUP_SIZE, 
                                    tile_config.q_tile_size * tile_config.kv_tile_size);
    
    dispatch.threads_per_threadgroup = MTLSizeMake(threads_per_group, 1, 1);
    
    // Function constants for dynamic tile sizing
    dispatch.function_constants = {
        tile_config.q_tile_size,   // TILE_SIZE_32
        tile_config.kv_tile_size,  // TILE_SIZE_64
        std::max(tile_config.q_tile_size, tile_config.kv_tile_size) // TILE_SIZE_128
    };
    
    return dispatch;
}

void unified_batch_prefill_attention_bf16(
    const bfloat16_t* q_input,
    const bfloat16_t* paged_k_cache,
    const bfloat16_t* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    bfloat16_t* output,
    const UnifiedParams& params,
    float* debug_out
) {
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize unified attention");
    }
    
    if (!validate_parameters(params)) {
        throw std::runtime_error("Invalid parameters for unified attention");
    }
    
    // Calculate sequence lengths for tile configuration
    std::vector<int> sequence_lengths;
    for (int i = 0; i < params.num_sequences; i++) {
        int q_len = qo_indptr[i + 1] - qo_indptr[i];
        
        // Calculate KV length from paged structure
        int kv_start_page = kv_page_indptr[i];
        int kv_end_page = kv_page_indptr[i + 1];
        int num_pages = kv_end_page - kv_start_page;
        int kv_len = (num_pages > 0) ? (num_pages - 1) * params.page_size + kv_last_page_lens[i] : 0;
        
        sequence_lengths.push_back(std::max(q_len, kv_len));
    }
    
    // Configure tiling and dispatch
    TileConfig tile_config = calculate_tile_config(sequence_lengths, params.head_dim, params.page_size);
    DispatchConfig dispatch = generate_dispatch_config(tile_config, params.num_sequences, params.num_heads);
    
    // Create specialized function with function constants 
    std::cout << "[UnifiedAttention] Creating specialized function with function constants..." << std::endl;
    NSError* error = nil;
    
    // Set up function constants for tile configuration
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    int tile_size_q = tile_config.q_tile_size;
    int tile_size_kv = tile_config.kv_tile_size;  
    bool enable_causal = false; // Default to no causal masking for now
    
    [constants setConstantValue:&tile_size_q type:MTLDataTypeInt atIndex:0];
    [constants setConstantValue:&tile_size_kv type:MTLDataTypeInt atIndex:1];
    [constants setConstantValue:&enable_causal type:MTLDataTypeBool atIndex:2];
    
    std::cout << "[UnifiedAttention] Function constants: Q_tile=" << tile_size_q 
              << ", KV_tile=" << tile_size_kv << ", causal=" << enable_causal << std::endl;
    
    // Create specialized function with constants using the loaded bf16 function name
    if (!g_unified_function_bf16) {
        std::cerr << "[UnifiedAttention] BF16 function not loaded\n";
        return;
    }
    
    NSString* functionName = g_unified_function_bf16.name;
    id<MTLFunction> specialized_function = [g_library newFunctionWithName:functionName
                                                           constantValues:constants
                                                                    error:&error];
    if (!specialized_function || error) {
        std::cerr << "[UnifiedAttention] Failed to create specialized function: "
                  << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
        throw std::runtime_error("Failed to create specialized function");
    }
    
    std::cout << "[UnifiedAttention] Successfully created specialized function" << std::endl;
    
    // Create compute pipeline state with specialized function
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:specialized_function error:&error];
    if (!pipeline) {
        std::cerr << "[UnifiedAttention] Failed to create compute pipeline: "
                  << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
        
        // Print more detailed error information
        if (error && error.userInfo) {
            std::cerr << "[UnifiedAttention] Error details:" << std::endl;
            for (NSString* key in error.userInfo) {
                NSString* value = [error.userInfo[key] description];
                std::cerr << "  " << key.UTF8String << ": " << value.UTF8String << std::endl;
            }
        }
        throw std::runtime_error("Failed to create compute pipeline");
    }
    
    std::cout << "[UnifiedAttention] Successfully created compute pipeline" << std::endl;
    
    // Create command buffer
    id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
    command_buffer.label = @"UnifiedBatchPrefillAttention";
    
    // Create compute encoder
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    encoder.label = @"UnifiedAttentionCompute";
    
    [encoder setComputePipelineState:pipeline];
    
    // Calculate buffer sizes
    size_t q_input_size = params.num_qo * params.head_dim * sizeof(bfloat16_t);
    size_t output_size = params.num_qo * params.head_dim * sizeof(bfloat16_t);
    size_t qo_indptr_size = (params.num_sequences + 1) * sizeof(int32_t);
    size_t kv_page_indptr_size = (params.num_sequences + 1) * sizeof(int32_t);
    size_t kv_last_page_lens_size = params.num_sequences * sizeof(int32_t);
    
    // Calculate KV cache sizes
    int total_pages = kv_page_indptr[params.num_sequences]; // Last value in page indptr
    size_t kv_cache_size = total_pages * params.page_size * params.head_dim * sizeof(bfloat16_t);
    
    // Create Metal buffers
    id<MTLBuffer> q_buffer = [g_device newBufferWithBytes:q_input 
                                                    length:q_input_size 
                                                   options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> paged_k_buffer = [g_device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                               length:kv_cache_size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    
    
    id<MTLBuffer> paged_v_buffer = [g_device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                               length:kv_cache_size
                                                              options:MTLResourceStorageModeShared  
                                                          deallocator:nil];
    
    id<MTLBuffer> qo_indptr_buffer = [g_device newBufferWithBytes:qo_indptr
                                                            length:qo_indptr_size
                                                           options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> kv_page_indptr_buffer = [g_device newBufferWithBytes:kv_page_indptr
                                                                length:kv_page_indptr_size
                                                               options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> kv_page_indices_buffer = [g_device newBufferWithBytesNoCopy:(void*)kv_page_indices
                                                                       length:SIZE_MAX // Dynamic size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
    
    id<MTLBuffer> kv_last_page_lens_buffer = [g_device newBufferWithBytes:kv_last_page_lens
                                                                   length:kv_last_page_lens_size
                                                                  options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> output_buffer = [g_device newBufferWithBytesNoCopy:(void*)output
                                                              length:output_size
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
    
    id<MTLBuffer> params_buffer = [g_device newBufferWithBytes:&params
                                                        length:sizeof(UnifiedParams)
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> debug_buffer = nil;
    if (debug_out) {
        debug_buffer = [g_device newBufferWithBytesNoCopy:(void*)debug_out
                                                   length:1024 * sizeof(float) // Reasonable debug size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    }
    
    // Set buffers
    [encoder setBuffer:q_buffer offset:0 atIndex:0];
    [encoder setBuffer:paged_k_buffer offset:0 atIndex:1];
    [encoder setBuffer:paged_v_buffer offset:0 atIndex:2];
    [encoder setBuffer:qo_indptr_buffer offset:0 atIndex:3];
    [encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:4];
    [encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:5];
    [encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:6];
    [encoder setBuffer:output_buffer offset:0 atIndex:7];
    [encoder setBuffer:params_buffer offset:0 atIndex:8];
    if (debug_buffer) {
        [encoder setBuffer:debug_buffer offset:0 atIndex:9];
    }
    
    // Dispatch 3D grid
    [encoder dispatchThreadgroups:dispatch.threadgroups_per_grid
            threadsPerThreadgroup:dispatch.threads_per_threadgroup];
    
    [encoder endEncoding];
    
    // Execute
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    // Check for errors
    if (command_buffer.status == MTLCommandBufferStatusError) {
        NSString* errorDesc = command_buffer.error ? command_buffer.error.localizedDescription : @"Unknown error";
        std::cerr << "[UnifiedAttention] Command buffer error: " << errorDesc.UTF8String << std::endl;
        throw std::runtime_error("Metal command buffer execution failed");
    }
    
    // Debug output
    if (debug_out && debug_buffer) {
        float* debug_data = (float*)debug_buffer.contents;
        std::cout << "[UnifiedAttention Debug] ";
        std::cout << "scale=" << debug_data[0] << ", ";
        std::cout << "head_dim=" << debug_data[1] << ", ";
        std::cout << "num_heads=" << debug_data[2] << ", ";
        std::cout << "max_seq_len=" << debug_data[3] << ", ";
        std::cout << "tile_q=" << debug_data[4] << ", ";
        std::cout << "tile_kv=" << debug_data[5] << ", ";
        std::cout << "seq_len=" << debug_data[6] << ", ";
        std::cout << "running_sum=" << debug_data[7] << ", ";
        std::cout << "q_tile_size=" << debug_data[8] << ", ";
        std::cout << "kv_tiles=" << debug_data[9] << ", ";
        std::cout << "output0=" << debug_data[10] << std::endl;
    }
}

void unified_batch_prefill_attention_auto(
    const bfloat16_t* q_input,
    const bfloat16_t* paged_k_cache,
    const bfloat16_t* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    bfloat16_t* output,
    int num_qo,
    int num_sequences,
    int head_dim,
    int head_size,
    int page_size,
    float scale,
    float* debug_out
) {
    // Calculate sequence lengths
    std::vector<int> sequence_lengths;
    int max_seq_len = 0;
    
    for (int i = 0; i < num_sequences; i++) {
        int q_len = qo_indptr[i + 1] - qo_indptr[i];
        
        int kv_start_page = kv_page_indptr[i];
        int kv_end_page = kv_page_indptr[i + 1];
        int num_pages = kv_end_page - kv_start_page;
        int kv_len = (num_pages > 0) ? (num_pages - 1) * page_size + kv_last_page_lens[i] : 0;
        
        int seq_len = std::max(q_len, kv_len);
        sequence_lengths.push_back(seq_len);
        max_seq_len = std::max(max_seq_len, seq_len);
    }
    
    // Create unified parameters
    UnifiedParams params = {
        .num_qo = num_qo,
        .num_sequences = num_sequences,
        .head_dim = head_dim,
        .head_size = head_size,
        .num_heads = head_dim / head_size,
        .page_size = page_size,
        .max_seq_len = max_seq_len,
        .scale = scale
    };
    
    // Call main function
    unified_batch_prefill_attention_bf16(
        q_input, paged_k_cache, paged_v_cache,
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_lens,
        output, params, debug_out
    );
}

// === Utility Functions ===

size_t calculate_workspace_size(
    int max_seq_len,
    int num_sequences,
    int head_dim,
    int num_heads
) {
    // Calculate tile configuration
    TileConfig config;
    if (max_seq_len <= 512) {
        config.q_tile_size = config.kv_tile_size = 32;
    } else if (max_seq_len <= 2048) {
        config.q_tile_size = config.kv_tile_size = 64;
    } else {
        config.q_tile_size = config.kv_tile_size = 128;
    }
    
    // Workspace for threadgroup memory (per threadgroup):
    // - Query tile: tile_size * head_dim * sizeof(half)
    // - KV tiles: 2 * tile_size * head_dim * sizeof(half)  
    // - Score matrix: tile_size * tile_size * sizeof(float)
    // - Accumulator: tile_size * head_dim * sizeof(float)
    // - Softmax states: tile_size * sizeof(OnlineSoftmaxState2D)
    
    size_t per_threadgroup_size = 
        config.q_tile_size * head_dim * sizeof(uint16_t) +  // Q tile
        2 * config.kv_tile_size * head_dim * sizeof(uint16_t) + // K,V tiles
        config.q_tile_size * config.kv_tile_size * sizeof(float) + // Scores
        config.q_tile_size * head_dim * sizeof(float) + // Accumulator
        config.q_tile_size * 3 * sizeof(float); // Softmax states
    
    // Total concurrent threadgroups
    int concurrent_threadgroups = num_sequences * num_heads;
    
    return per_threadgroup_size * concurrent_threadgroups;
}

bool validate_parameters(const UnifiedParams& params) {
    if (params.num_qo <= 0 || params.num_sequences <= 0) {
        std::cerr << "[UnifiedAttention] Invalid num_qo or num_sequences\n";
        return false;
    }
    
    if (params.head_dim <= 0 || params.head_size <= 0) {
        std::cerr << "[UnifiedAttention] Invalid head dimensions\n";
        return false;
    }
    
    if (params.head_dim % params.head_size != 0) {
        std::cerr << "[UnifiedAttention] head_dim must be divisible by head_size\n";
        return false;
    }
    
    if (params.num_heads != params.head_dim / params.head_size) {
        std::cerr << "[UnifiedAttention] Inconsistent num_heads calculation\n";
        return false;
    }
    
    if (params.page_size <= 0) {
        std::cerr << "[UnifiedAttention] Invalid page_size\n";
        return false;
    }
    
    if (params.scale <= 0.0f) {
        std::cerr << "[UnifiedAttention] Invalid scale factor\n";
        return false;
    }
    
    return true;
}

void unified_batch_prefill_attention_f32(
    const float* q_input,
    const float* paged_k_cache,
    const float* paged_v_cache,
    const int32_t* qo_indptr,
    const int32_t* kv_page_indptr,
    const int32_t* kv_page_indices,
    const int32_t* kv_last_page_lens,
    float* output,
    const UnifiedParams& params,
    float* debug_out
) {
    if (!is_initialized() && !initialize()) {
        throw std::runtime_error("Failed to initialize unified attention");
    }
    
    if (!validate_parameters(params)) {
        throw std::runtime_error("Invalid parameters for unified attention");
    }
    
    if (!g_unified_function_f32) {
        throw std::runtime_error("F32 unified attention function not loaded");
    }
    
    // Calculate tile sizes similar to bf16 version
    int tile_size_q = 32; // Default tile size for Q
    int tile_size_kv = 32; // Default tile size for KV
    bool enable_causal = false; // Default causal masking
    
    // Adjust tile sizes based on sequence characteristics
    if (params.max_seq_len > 256) {
        tile_size_q = 64;
        tile_size_kv = 64;
    } else if (params.max_seq_len < 16) {
        tile_size_q = 16;
        tile_size_kv = 16;
    }
    
    std::cout << "[UnifiedAttention] Creating specialized f32 function with function constants..." << std::endl;
    
    // Create function constants for dynamic tile sizing
    MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
    [constants setConstantValue:&tile_size_q type:MTLDataTypeInt atIndex:0]; // TILE_SIZE_Q
    [constants setConstantValue:&tile_size_kv type:MTLDataTypeInt atIndex:1]; // TILE_SIZE_KV
    [constants setConstantValue:&enable_causal type:MTLDataTypeBool atIndex:2]; // ENABLE_CAUSAL_MASK
    
    NSError* error = nil;
    
    std::cout << "[UnifiedAttention] Function constants: Q_tile=" << tile_size_q 
              << ", KV_tile=" << tile_size_kv << ", causal=" << enable_causal << std::endl;
    
    // Create specialized function with constants using the loaded f32 function name
    NSString* functionName = g_unified_function_f32.name;
    id<MTLFunction> specialized_function = [g_library newFunctionWithName:functionName
                                                           constantValues:constants
                                                                    error:&error];
    if (!specialized_function || error) {
        std::cerr << "[UnifiedAttention] Failed to create specialized f32 function: "
                  << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
        throw std::runtime_error("Failed to create specialized f32 function");
    }
    
    std::cout << "[UnifiedAttention] Successfully created specialized f32 function" << std::endl;
    
    // Create compute pipeline state
    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:specialized_function error:&error];
    if (!pipeline || error) {
        std::cerr << "[UnifiedAttention] Failed to create f32 compute pipeline: " 
                  << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
        throw std::runtime_error("Failed to create f32 compute pipeline");
    }
    
    std::cout << "[UnifiedAttention] Successfully created f32 compute pipeline" << std::endl;
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> command_buffer = [g_command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    
    // Calculate buffer sizes (f32 uses 4 bytes per element instead of 2)
    size_t q_size = params.num_qo * params.head_dim * sizeof(float);
    size_t kv_cache_size = params.max_seq_len * params.head_dim * sizeof(float);
    size_t output_size = params.num_qo * params.head_dim * sizeof(float);
    
    // Create Metal buffers (f32 data)
    id<MTLBuffer> q_buffer = [g_device newBufferWithBytesNoCopy:(void*)q_input
                                                         length:q_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
    
    id<MTLBuffer> paged_k_buffer = [g_device newBufferWithBytesNoCopy:(void*)paged_k_cache
                                                               length:kv_cache_size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
                                                          
    id<MTLBuffer> paged_v_buffer = [g_device newBufferWithBytesNoCopy:(void*)paged_v_cache
                                                               length:kv_cache_size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
    
    id<MTLBuffer> qo_indptr_buffer = [g_device newBufferWithBytesNoCopy:(void*)qo_indptr
                                                                 length:(params.num_sequences + 1) * sizeof(int32_t)
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
    
    id<MTLBuffer> kv_page_indptr_buffer = [g_device newBufferWithBytesNoCopy:(void*)kv_page_indptr
                                                                      length:(params.num_sequences + 1) * sizeof(int32_t)
                                                                     options:MTLResourceStorageModeShared
                                                                 deallocator:nil];
    
    id<MTLBuffer> kv_page_indices_buffer = [g_device newBufferWithBytesNoCopy:(void*)kv_page_indices
                                                                       length:100 * sizeof(int32_t) // Estimate
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
    
    id<MTLBuffer> kv_last_page_lens_buffer = [g_device newBufferWithBytesNoCopy:(void*)kv_last_page_lens
                                                                         length:params.num_sequences * sizeof(int32_t)
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
    
    id<MTLBuffer> output_buffer = [g_device newBufferWithBytesNoCopy:output
                                                              length:output_size
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
    
    id<MTLBuffer> params_buffer = [g_device newBufferWithBytes:&params
                                                        length:sizeof(UnifiedParams)
                                                       options:MTLResourceStorageModeShared];
    
    id<MTLBuffer> debug_buffer = nil;
    if (debug_out) {
        debug_buffer = [g_device newBufferWithBytesNoCopy:(void*)debug_out
                                                   length:1024 * sizeof(float) // Reasonable debug size
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    }
    
    // Set buffers
    [encoder setBuffer:q_buffer offset:0 atIndex:0];
    [encoder setBuffer:paged_k_buffer offset:0 atIndex:1];
    [encoder setBuffer:paged_v_buffer offset:0 atIndex:2];
    [encoder setBuffer:qo_indptr_buffer offset:0 atIndex:3];
    [encoder setBuffer:kv_page_indptr_buffer offset:0 atIndex:4];
    [encoder setBuffer:kv_page_indices_buffer offset:0 atIndex:5];
    [encoder setBuffer:kv_last_page_lens_buffer offset:0 atIndex:6];
    [encoder setBuffer:output_buffer offset:0 atIndex:7];
    [encoder setBuffer:params_buffer offset:0 atIndex:8];
    if (debug_buffer) {
        [encoder setBuffer:debug_buffer offset:0 atIndex:9];
    }
    
    // Calculate dispatch configuration (same as bf16 version)
    TileConfig tile_config = {};
    tile_config.q_tile_size = tile_size_q;
    tile_config.kv_tile_size = tile_size_kv;
    
    DispatchConfig dispatch = generate_dispatch_config(tile_config, params.num_sequences, params.num_heads);
    
    // Dispatch 3D grid
    [encoder dispatchThreadgroups:dispatch.threadgroups_per_grid
            threadsPerThreadgroup:dispatch.threads_per_threadgroup];
    
    [encoder endEncoding];
    
    // Execute
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    
    // Debug output (same format as bf16 version)
    if (debug_out) {
        std::cout << "[UnifiedAttention Debug] scale=" << params.scale 
                  << ", head_dim=" << params.head_dim
                  << ", num_heads=" << params.num_heads << ", max_seq_len=" << params.max_seq_len
                  << ", tile_q=" << tile_size_q << ", tile_kv=" << tile_size_kv
                  << ", seq_len=" << debug_out[6] << ", running_sum=" << debug_out[7] 
                  << ", q_tile_size=" << debug_out[8] << ", kv_tiles=" << debug_out[9]
                  << ", output0=" << debug_out[10] << std::endl;
    }
}

PerformanceEstimate estimate_performance(
    const TileConfig& tile_config,
    int num_sequences,
    int num_heads,
    const std::vector<int>& sequence_lengths
) {
    PerformanceEstimate estimate = {};
    
    // Calculate total FLOPS
    double total_flops = 0.0;
    for (int seq_len : sequence_lengths) {
        // Attention computation: O(seq_len^2 * head_size) per head
        double seq_flops = static_cast<double>(seq_len) * seq_len * 64.0; // Assume head_size=64
        total_flops += seq_flops * num_heads;
    }
    
    estimate.estimated_flops = total_flops;
    
    // Estimate memory bandwidth (approximate)
    double total_memory_access = 0.0;
    for (int seq_len : sequence_lengths) {
        // Query: seq_len * head_dim * sizeof(half)
        // KV cache: seq_len * head_dim * 2 * sizeof(half) 
        // Output: seq_len * head_dim * sizeof(half)
        total_memory_access += seq_len * 64 * num_heads * 4 * 2; // 4 arrays, 2 bytes each
    }
    
    estimate.memory_bandwidth_gb = total_memory_access / (1024.0 * 1024.0 * 1024.0);
    
    // Rough time estimate (very approximate)
    // Assume 1 TFLOPS peak performance and 500 GB/s memory bandwidth
    double compute_time = estimate.estimated_flops / (1e12); // seconds for compute
    double memory_time = estimate.memory_bandwidth_gb / 500.0; // seconds for memory
    
    estimate.estimated_time_ms = std::max(compute_time, memory_time) * 1000.0;
    
    return estimate;
}

} // namespace metal::unified_attention