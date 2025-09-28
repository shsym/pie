#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace py = pybind11;

class MetalAttentionExecutor {
private:
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> commandQueue;

    enum class TensorDType {
        Float32,
        BFloat16,
    };

    static inline float bf16_to_float(uint16_t value) {
        uint32_t tmp = static_cast<uint32_t>(value) << 16;
        float result;
        std::memcpy(&result, &tmp, sizeof(float));
        return result;
    }

    static inline uint16_t float_to_fp16(float value) {
        __fp16 half = static_cast<__fp16>(value);
        uint16_t bits;
        std::memcpy(&bits, &half, sizeof(uint16_t));
        return bits;
    }

    static inline float fp16_to_float(uint16_t value) {
        __fp16 half;
        std::memcpy(&half, &value, sizeof(uint16_t));
        return static_cast<float>(half);
    }

    // Helper function to validate and convert numpy arrays to float32
    py::array_t<float> validate_and_convert_to_float32(py::array input, const std::string& name) {
        auto buf = input.request();

        // Check for common unsupported dtypes
        if (buf.format == py::format_descriptor<double>::format()) {
            // Convert double to float32
            auto converted = py::array_t<float>(buf.size);
            auto conv_buf = converted.request();
            const double* src = static_cast<const double*>(buf.ptr);
            float* dst = static_cast<float*>(conv_buf.ptr);
            for (size_t i = 0; i < buf.size; i++) {
                dst[i] = static_cast<float>(src[i]);
            }
            return converted;
        } else if (buf.format == py::format_descriptor<float>::format()) {
            // Already float32
            return py::cast<py::array_t<float>>(input);
        } else {
            throw std::runtime_error("Unsupported dtype for " + name + ". Expected float32 or float64, got: " + buf.format);
        }
    }

public:
    MetalAttentionExecutor(const std::string& metallib_path) {
        @autoreleasepool {
            // Get default Metal device
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                throw std::runtime_error("Metal device not available");
            }

            // Create command queue
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                throw std::runtime_error("Failed to create Metal command queue");
            }

            // Load Metal library
            NSString *libraryPath = [NSString stringWithUTF8String:metallib_path.c_str()];
            NSError *error = nil;
            NSURL *libraryURL = [NSURL fileURLWithPath:libraryPath];
            library = [device newLibraryWithURL:libraryURL error:&error];

            if (!library) {
                NSString *errorDesc = error ? [error localizedDescription] : @"Unknown error";
                std::string errorMsg = "Failed to load Metal library: " + std::string([errorDesc UTF8String]);
                throw std::runtime_error(errorMsg);
            }
        }
    }

    ~MetalAttentionExecutor() {
        @autoreleasepool {
            if (library) [library release];
            if (commandQueue) [commandQueue release];
            if (device) [device release];
        }
    }

    py::array_t<float> execute_attention_with_kv_cache(
        py::array query, py::array kv_cache,
        py::array_t<int> kv_page_indices, py::array_t<int> kv_page_indptr, py::array_t<int> kv_last_page_lens,
        py::array_t<int> qo_indptr,
        int num_query_heads = 32, int num_kv_heads = 32, int head_size = 128, int page_size = 16) {
        @autoreleasepool {
            auto q_buf = query.request();
            auto kv_buf = kv_cache.request();
            auto kv_indices_buf = kv_page_indices.request();
            auto kv_indptr_buf = kv_page_indptr.request();
            auto kv_lens_buf = kv_last_page_lens.request();
            auto qo_indptr_buf = qo_indptr.request();

            TensorDType tensor_dtype;
            if (q_buf.format == py::format_descriptor<float>::format()) {
                tensor_dtype = TensorDType::Float32;
            } else if (q_buf.format == py::format_descriptor<uint16_t>::format()) {
                tensor_dtype = TensorDType::BFloat16;
            } else {
                throw std::runtime_error("Unsupported query dtype for attention kernel: " + std::string(q_buf.format));
            }

            if ((tensor_dtype == TensorDType::Float32 && kv_buf.format != py::format_descriptor<float>::format()) ||
                (tensor_dtype == TensorDType::BFloat16 && kv_buf.format != py::format_descriptor<uint16_t>::format())) {
                throw std::runtime_error("KV cache dtype mismatch with query dtype");
            }

            NSString *kernelName = nil;
            if (tensor_dtype == TensorDType::BFloat16) {
                kernelName = @"batch_prefill_attention_unified_bf16_simdgroup_kernel";
            } else {
                kernelName = @"batch_prefill_attention_unified_f32_simdgroup_kernel";
            }

            id<MTLFunction> kernelFunction = [library newFunctionWithName:kernelName];

            if (!kernelFunction) {
                throw std::runtime_error("Production attention kernel not found");
            }

            // Validate query shape - expect [batch*seq, num_heads, head_size] or [batch*seq, num_heads * head_size]
            if (q_buf.ndim < 2 || q_buf.ndim > 3) {
                throw std::runtime_error("Query must be 2D or 3D tensor");
            }

            size_t batch_seq_len = q_buf.shape[0];
            size_t query_dim = (q_buf.ndim == 3) ? q_buf.shape[1] * q_buf.shape[2] : q_buf.shape[1];

            // Infer attention configuration from actual tensor shapes to avoid metadata mismatches
            py::ssize_t inferred_page_size = kv_buf.shape[2];
            py::ssize_t inferred_num_kv_heads = kv_buf.shape[3];
            py::ssize_t inferred_head_size = kv_buf.shape[4];

            if (inferred_head_size <= 0 || inferred_num_kv_heads <= 0 || inferred_page_size <= 0) {
                throw std::runtime_error("Invalid KV cache shape detected when inferring attention parameters");
            }

            if (query_dim % static_cast<size_t>(inferred_head_size) != 0) {
                throw std::runtime_error("Query dimensions are not divisible by inferred head size");
            }

            int inferred_num_query_heads = static_cast<int>(query_dim / static_cast<size_t>(inferred_head_size));

            num_query_heads = inferred_num_query_heads;
            num_kv_heads = static_cast<int>(inferred_num_kv_heads);
            head_size = static_cast<int>(inferred_head_size);
            page_size = static_cast<int>(inferred_page_size);

            // Create parameter structure matching the Metal kernel
            struct Params {
                int num_qo;
                int head_dim;
                int kv_head_dim;
                int head_size;
                int page_size;
                int num_query_heads;
                int num_kv_heads;
                float scale;
            };

            Params params = {
                .num_qo = static_cast<int>(batch_seq_len),
                .head_dim = static_cast<int>(query_dim),
                .kv_head_dim = num_kv_heads * head_size,
                .head_size = head_size,
                .page_size = page_size,
                .num_query_heads = num_query_heads,
                .num_kv_heads = num_kv_heads,
                .scale = 1.0f / sqrtf(static_cast<float>(head_size))
            };

            // Create compute pipeline state
            NSError *error = nil;
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
            if (!pipelineState) {
                throw std::runtime_error("Failed to create attention pipeline state");
            }

            std::vector<py::ssize_t> result_shape = {static_cast<py::ssize_t>(batch_seq_len), static_cast<py::ssize_t>(query_dim)};
            py::array_t<float> result(result_shape);
            auto result_buf = result.request();

            size_t total_elements = static_cast<size_t>(batch_seq_len) * static_cast<size_t>(query_dim);

            const void* query_src_ptr = q_buf.ptr;
            size_t query_element_size = sizeof(float);
            std::vector<uint16_t> query_fp16_storage;

            if (tensor_dtype == TensorDType::BFloat16) {
                const uint16_t* q_bf16_ptr = static_cast<const uint16_t*>(q_buf.ptr);
                query_fp16_storage.resize(total_elements);
                for (size_t i = 0; i < total_elements; ++i) {
                    float val = bf16_to_float(q_bf16_ptr[i]);
                    query_fp16_storage[i] = float_to_fp16(val);
                }
                query_src_ptr = query_fp16_storage.data();
                query_element_size = sizeof(uint16_t);
            }

            size_t query_size = total_elements * query_element_size;
            size_t output_size = total_elements * sizeof(float);  // Output is always Float32
            id<MTLBuffer> qBuffer = [device newBufferWithBytes:query_src_ptr length:query_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:output_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params length:sizeof(Params) options:MTLResourceStorageModeShared];

            // Split captured KV cache tensors into distinct K and V buffers
            if (kv_buf.ndim != 5 || kv_buf.shape[1] != 2) {
                throw std::runtime_error("KV cache tensor must have shape [num_pages, 2, page_size, num_kv_heads, head_size]");
            }

            py::ssize_t num_pages = kv_buf.shape[0];
            py::ssize_t elements_per_page = kv_buf.shape[2] * kv_buf.shape[3] * kv_buf.shape[4];
            size_t kv_elements_per_cache = static_cast<size_t>(num_pages) * static_cast<size_t>(elements_per_page);

            size_t kv_element_size = tensor_dtype == TensorDType::Float32 ? sizeof(float) : sizeof(uint16_t);
            size_t single_cache_size = kv_elements_per_cache * kv_element_size;

            std::vector<uint16_t> k_fp16_storage;
            std::vector<uint16_t> v_fp16_storage;

            id<MTLBuffer> kCacheBuffer = [device newBufferWithLength:single_cache_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> vCacheBuffer = [device newBufferWithLength:single_cache_size options:MTLResourceStorageModeShared];

            if (!qBuffer || !kCacheBuffer || !vCacheBuffer || !outputBuffer || !paramsBuffer) {
                if (qBuffer) [qBuffer release];
                if (kCacheBuffer) [kCacheBuffer release];
                if (vCacheBuffer) [vCacheBuffer release];
                if (outputBuffer) [outputBuffer release];
                if (paramsBuffer) [paramsBuffer release];
                [pipelineState release];
                [kernelFunction release];
                throw std::runtime_error("Failed to create Metal buffers for KV cache attention");
            }

            void* k_dest_raw = [kCacheBuffer contents];
            void* v_dest_raw = [vCacheBuffer contents];

            if (!k_dest_raw || !v_dest_raw) {
                [qBuffer release];
                [kCacheBuffer release];
                [vCacheBuffer release];
                [outputBuffer release];
                [paramsBuffer release];
                [pipelineState release];
                [kernelFunction release];
                throw std::runtime_error("Failed to map Metal buffers for KV cache attention");
            }

            size_t page_stride = static_cast<size_t>(elements_per_page);
            size_t interleaved_stride = page_stride * 2;  // account for combined K/V dimension
            if (tensor_dtype == TensorDType::Float32) {
                const float* kv_data = static_cast<const float*>(kv_buf.ptr);
                float* k_dest = static_cast<float*>(k_dest_raw);
                float* v_dest = static_cast<float*>(v_dest_raw);
                for (py::ssize_t page = 0; page < num_pages; ++page) {
                    const float* page_base = kv_data + static_cast<size_t>(page) * interleaved_stride;
                    std::memcpy(k_dest + static_cast<size_t>(page) * page_stride, page_base, page_stride * sizeof(float));
                    std::memcpy(v_dest + static_cast<size_t>(page) * page_stride, page_base + page_stride, page_stride * sizeof(float));
                }
            } else {
                const uint16_t* kv_data = static_cast<const uint16_t*>(kv_buf.ptr);
                k_fp16_storage.resize(kv_elements_per_cache);
                v_fp16_storage.resize(kv_elements_per_cache);
                for (py::ssize_t page = 0; page < num_pages; ++page) {
                    const uint16_t* page_base = kv_data + static_cast<size_t>(page) * interleaved_stride;
                    uint16_t* k_dest_vec = k_fp16_storage.data() + static_cast<size_t>(page) * page_stride;
                    uint16_t* v_dest_vec = v_fp16_storage.data() + static_cast<size_t>(page) * page_stride;
                    for (size_t idx = 0; idx < page_stride; ++idx) {
                        float k_val = bf16_to_float(page_base[idx]);
                        float v_val = bf16_to_float(page_base[idx + page_stride]);
                        k_dest_vec[idx] = float_to_fp16(k_val);
                        v_dest_vec[idx] = float_to_fp16(v_val);
                    }
                }
                std::memcpy(k_dest_raw, k_fp16_storage.data(), single_cache_size);
                std::memcpy(v_dest_raw, v_fp16_storage.data(), single_cache_size);
            }

            // Use actual L4MA page indices
            id<MTLBuffer> kvPageIndicesBuffer = [device newBufferWithBytes:kv_indices_buf.ptr
                                                                    length:kv_indices_buf.size * sizeof(int)
                                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> kvPageIndptrBuffer = [device newBufferWithBytes:kv_indptr_buf.ptr
                                                                   length:kv_indptr_buf.size * sizeof(int)
                                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> kvLastPageLensBuffer = [device newBufferWithBytes:kv_lens_buf.ptr
                                                                     length:kv_lens_buf.size * sizeof(int)
                                                                    options:MTLResourceStorageModeShared];

            // Use the real QO indptr from input
            id<MTLBuffer> qoIndptrBuffer = [device newBufferWithBytes:qo_indptr_buf.ptr
                                                               length:qo_indptr_buf.size * sizeof(int)
                                                              options:MTLResourceStorageModeShared];

            // Execute the kernel with actual L4MA layout
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            id<MTLBuffer> debugBuffer = [device newBufferWithLength:100 * sizeof(float) options:MTLResourceStorageModeShared];

            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:qBuffer offset:0 atIndex:0];           // q_input
            [encoder setBuffer:kCacheBuffer offset:0 atIndex:1];      // paged_k_cache
            [encoder setBuffer:vCacheBuffer offset:0 atIndex:2];      // paged_v_cache
            [encoder setBuffer:qoIndptrBuffer offset:0 atIndex:3];    // qo_indptr
            [encoder setBuffer:kvPageIndptrBuffer offset:0 atIndex:4]; // kv_page_indptr
            [encoder setBuffer:kvPageIndicesBuffer offset:0 atIndex:5]; // kv_page_indices
            [encoder setBuffer:kvLastPageLensBuffer offset:0 atIndex:6]; // kv_last_page_lens
            [encoder setBuffer:outputBuffer offset:0 atIndex:7];      // output
            [encoder setBuffer:paramsBuffer offset:0 atIndex:8];      // params
            [encoder setBuffer:debugBuffer offset:0 atIndex:9];       // debug_out

            // Dispatch the kernel
            MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);  // TGP_SIZE from metal_attention_common.metal
            MTLSize threadgroupsPerGrid = MTLSizeMake(batch_seq_len, 1, 1);
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Copy result back
            if (tensor_dtype == TensorDType::Float32) {
                std::memcpy(result_buf.ptr, [outputBuffer contents], total_elements * sizeof(float));
            } else {
                const uint16_t* src = static_cast<const uint16_t*>([outputBuffer contents]);
                float* dst = static_cast<float*>(result_buf.ptr);
                for (size_t i = 0; i < total_elements; ++i) {
                    dst[i] = fp16_to_float(src[i]);
                }
            }

            // Debug buffer cleanup (debug prints removed)
            if (debugBuffer) {
                // No debug output needed in production
            }

            // Cleanup
            [qBuffer release];
            [kCacheBuffer release];
            [vCacheBuffer release];
            [outputBuffer release];
            [paramsBuffer release];
            [qoIndptrBuffer release];
            [kvPageIndptrBuffer release];
            [kvPageIndicesBuffer release];
            [kvLastPageLensBuffer release];
            if (debugBuffer) {
                [debugBuffer release];
            }
            [pipelineState release];
            [kernelFunction release];

            return result;
        }
    }

    py::array_t<float> execute_attention(
        py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
        int num_query_heads = 32, int num_kv_heads = 32, int head_size = 128, int page_size = 16) {
        @autoreleasepool {
            // Use production paged attention kernel
            NSString *kernelName = @"batch_prefill_attention_unified_f32_simdgroup_kernel";
            id<MTLFunction> kernelFunction = [library newFunctionWithName:kernelName];

            if (!kernelFunction) {
                // Try half precision version if f32 not found
                kernelName = @"batch_prefill_attention_unified_bf16_simdgroup_kernel";
                kernelFunction = [library newFunctionWithName:kernelName];
            }

            if (!kernelFunction) {
                throw std::runtime_error("Production attention kernel not found. Build system may not have compiled kernels properly.");
            }

            // Convert inputs to proper format and validate
            py::array_t<float> q_f32 = validate_and_convert_to_float32(q, "query");
            py::array_t<float> k_f32 = validate_and_convert_to_float32(k, "key");
            py::array_t<float> v_f32 = validate_and_convert_to_float32(v, "value");

            auto q_buf = q_f32.request();
            auto k_buf = k_f32.request();
            auto v_buf = v_f32.request();

            // Validate shapes - expect [batch_size * seq_len, num_heads * head_size]
            if (q_buf.ndim != 2 || k_buf.ndim != 2 || v_buf.ndim != 2) {
                throw std::runtime_error("Attention inputs must be 2D tensors");
            }

            size_t batch_seq_len = q_buf.shape[0];
            size_t hidden_dim = q_buf.shape[1];

            // Use passed parameters from model metadata
            // Validate that head_size matches the tensor dimensions
            if (hidden_dim != num_query_heads * head_size) {
                // Try to infer head_size if mismatch (for debug framework compatibility)
                head_size = hidden_dim / num_query_heads;
            }

            // For debug framework, simulate paged attention with simple batched attention
            // In production, this would use proper page tables and KV cache

            // Create parameter structure matching metal_attention_common.metal Params struct
            struct Params {
                int num_qo;
                int head_dim;        // Query head dimension (num_query_heads * head_size)
                int kv_head_dim;     // KV head dimension (num_kv_heads * head_size)
                int head_size;
                int page_size;
                int num_query_heads; // Number of query heads
                int num_kv_heads;    // Number of KV heads (for MQA/GQA support)
                float scale;
            };

            Params params = {
                .num_qo = static_cast<int>(batch_seq_len),
                .head_dim = static_cast<int>(hidden_dim),
                .kv_head_dim = static_cast<int>(hidden_dim),
                .head_size = static_cast<int>(head_size),
                .page_size = static_cast<int>(page_size),
                .num_query_heads = static_cast<int>(num_query_heads),
                .num_kv_heads = static_cast<int>(num_kv_heads),
                .scale = 1.0f / sqrtf(static_cast<float>(head_size))
            };

            // Create compute pipeline state
            NSError *error = nil;
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
            if (!pipelineState) {
                NSString *errorDesc = error ? [error localizedDescription] : @"Unknown error";
                std::string errorMsg = "Failed to create attention pipeline state:" + std::string([errorDesc UTF8String]);
                throw std::runtime_error(errorMsg);
            }

            // Create output tensor
            auto result = py::array_t<float>({static_cast<py::ssize_t>(batch_seq_len), static_cast<py::ssize_t>(hidden_dim)});
            auto result_buf = result.request();

            // Create Metal buffers
            size_t data_size = batch_seq_len * hidden_dim * sizeof(float);
            id<MTLBuffer> qBuffer = [device newBufferWithBytes:q_buf.ptr length:data_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> kBuffer = [device newBufferWithBytes:k_buf.ptr length:data_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> vBuffer = [device newBufferWithBytes:v_buf.ptr length:data_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
            id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params length:sizeof(Params) options:MTLResourceStorageModeShared];

            // For debug framework, create dummy page tables (in production these would be real page indices)
            std::vector<int> qo_indptr = {0, static_cast<int>(batch_seq_len)};
            std::vector<int> kv_page_indptr = {0, 1};
            std::vector<int> kv_page_indices = {0};
            std::vector<int> kv_last_page_lens = {static_cast<int>(batch_seq_len)};

            id<MTLBuffer> qoIndptrBuffer = [device newBufferWithBytes:qo_indptr.data() length:qo_indptr.size() * sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> kvPageIndptrBuffer = [device newBufferWithBytes:kv_page_indptr.data() length:kv_page_indptr.size() * sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> kvPageIndicesBuffer = [device newBufferWithBytes:kv_page_indices.data() length:kv_page_indices.size() * sizeof(int) options:MTLResourceStorageModeShared];
            id<MTLBuffer> kvLastPageLensBuffer = [device newBufferWithBytes:kv_last_page_lens.data() length:kv_last_page_lens.size() * sizeof(int) options:MTLResourceStorageModeShared];

            if (!qBuffer || !kBuffer || !vBuffer || !outputBuffer || !paramsBuffer) {
                throw std::runtime_error("Failed to create Metal buffers for attention");
            }

            // Create command buffer and encoder
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            // Create debug buffer for kernel debugging output
            id<MTLBuffer> debugBuffer = [device newBufferWithLength:100 * sizeof(float) options:MTLResourceStorageModeShared];

            // Set pipeline state and buffers matching kernel signature
            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:qBuffer offset:0 atIndex:0];  // q_input
            [encoder setBuffer:kBuffer offset:0 atIndex:1];  // paged_k_cache
            [encoder setBuffer:vBuffer offset:0 atIndex:2];  // paged_v_cache
            [encoder setBuffer:qoIndptrBuffer offset:0 atIndex:3];  // qo_indptr
            [encoder setBuffer:kvPageIndptrBuffer offset:0 atIndex:4];  // kv_page_indptr
            [encoder setBuffer:kvPageIndicesBuffer offset:0 atIndex:5];  // kv_page_indices
            [encoder setBuffer:kvLastPageLensBuffer offset:0 atIndex:6];  // kv_last_page_lens
            [encoder setBuffer:outputBuffer offset:0 atIndex:7];  // output
            [encoder setBuffer:paramsBuffer offset:0 atIndex:8];  // params
            [encoder setBuffer:debugBuffer offset:0 atIndex:9];  // debug_out

            // Set threadgroup memory for the kernel
            [encoder setThreadgroupMemoryLength:32768 atIndex:0];

            // Dispatch one threadgroup per query token
            MTLSize threadsPerThreadgroup = MTLSizeMake(128, 1, 1);  // TGP_SIZE = 128
            MTLSize threadgroupsPerGrid = MTLSizeMake(batch_seq_len, 1, 1);

            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];

            // Submit and wait
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Check for execution errors
            if ([commandBuffer status] == MTLCommandBufferStatusError) {
                NSString *errorDesc = [[commandBuffer error] localizedDescription];
                std::string errorMsg = "Metal attention kernel execution failed: " + std::string([errorDesc UTF8String]);
                throw std::runtime_error(errorMsg);
            }

            // Copy result back
            memcpy(result_buf.ptr, [outputBuffer contents], data_size);

            // Cleanup
            [qBuffer release];
            [kBuffer release];
            [vBuffer release];
            [outputBuffer release];
            [paramsBuffer release];
            [qoIndptrBuffer release];
            [kvPageIndptrBuffer release];
            [kvPageIndicesBuffer release];
            [kvLastPageLensBuffer release];
            [debugBuffer release];
            [pipelineState release];
            [kernelFunction release];

            return result;
        }
    }

    std::vector<std::string> list_available_kernels() {
        @autoreleasepool {
            std::vector<std::string> kernels;
            NSArray *functionNames = [library functionNames];
            for (NSString *name in functionNames) {
                kernels.push_back(std::string([name UTF8String]));
            }
            return kernels;
        }
    }

    std::string get_device_info() {
        @autoreleasepool {
            NSString *deviceName = [device name];
            return std::string([deviceName UTF8String]);
        }
    }
};

PYBIND11_MODULE(metal_attention_bindings, m) {
    m.doc() = "Metal attention kernel executor for PIE debug framework";

    py::class_<MetalAttentionExecutor>(m, "MetalAttentionExecutor")
        .def(py::init<const std::string&>(), "Initialize with metallib path")
        .def("execute_attention", &MetalAttentionExecutor::execute_attention,
             "Execute attention kernel with Q, K, V inputs")
        .def("execute_attention_with_kv_cache", &MetalAttentionExecutor::execute_attention_with_kv_cache,
             "Execute attention kernel with Q and KV cache in L4MA format")
        .def("list_available_kernels", &MetalAttentionExecutor::list_available_kernels,
             "List all available kernel functions")
        .def("get_device_info", &MetalAttentionExecutor::get_device_info,
             "Get Metal device information");
}