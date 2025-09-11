#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <string>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

class MetalKernelExecutor {
private:
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> commandQueue;
    
public:
    MetalKernelExecutor(const std::string& metallib_path) {
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
    
    ~MetalKernelExecutor() {
        @autoreleasepool {
            if (library) [library release];
            if (commandQueue) [commandQueue release];
            if (device) [device release];
        }
    }
    
    py::array_t<float> execute_softmax(py::array_t<float> input) {
        @autoreleasepool {
            // Get kernel function
            NSString *kernelName = @"metal_softmax_kernel";
            id<MTLFunction> kernelFunction = [library newFunctionWithName:kernelName];
            if (!kernelFunction) {
                throw std::runtime_error("Failed to find softmax kernel function");
            }
            
            // Create compute pipeline state
            NSError *error = nil;
            id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
            if (!pipelineState) {
                NSString *errorDesc = error ? [error localizedDescription] : @"Unknown error";
                std::string errorMsg = "Failed to create pipeline state: " + std::string([errorDesc UTF8String]);
                throw std::runtime_error(errorMsg);
            }
            
            // Get input data
            auto buf = input.request();
            size_t num_elements = buf.size;
            size_t buffer_size = num_elements * sizeof(float);
            
            // Create Metal buffers
            id<MTLBuffer> inputBuffer = [device newBufferWithBytes:buf.ptr 
                                                            length:buffer_size 
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:buffer_size 
                                                             options:MTLResourceStorageModeShared];
            
            if (!inputBuffer || !outputBuffer) {
                throw std::runtime_error("Failed to create Metal buffers");
            }
            
            // Create command buffer
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            
            // Set pipeline state and buffers
            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:outputBuffer offset:0 atIndex:1];
            
            // Set kernel parameters
            uint32_t num_elements_uint = static_cast<uint32_t>(num_elements);
            [encoder setBytes:&num_elements_uint length:sizeof(uint32_t) atIndex:2];
            
            // Dispatch threads
            MTLSize threadsPerThreadgroup = MTLSizeMake(64, 1, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((num_elements + 63) / 64, 1, 1);
            
            [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
            
            // Submit and wait
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            
            // Create output array
            auto result = py::array_t<float>(num_elements);
            auto result_buf = result.request();
            memcpy(result_buf.ptr, [outputBuffer contents], buffer_size);
            
            // Cleanup
            [inputBuffer release];
            [outputBuffer release];
            [pipelineState release];
            [kernelFunction release];
            
            return result;
        }
    }
    
    py::array_t<float> execute_attention(py::array_t<float> q, py::array_t<float> k, py::array_t<float> v) {
        // Placeholder for attention kernel execution
        // For now, just return q as a basic implementation
        auto result = py::array_t<float>(q.size());
        auto result_buf = result.request();
        auto q_buf = q.request();
        memcpy(result_buf.ptr, q_buf.ptr, q.size() * sizeof(float));
        return result;
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

PYBIND11_MODULE(metal_bindings, m) {
    m.doc() = "Metal kernel executor for PIE debug framework";
    
    py::class_<MetalKernelExecutor>(m, "MetalKernelExecutor")
        .def(py::init<const std::string&>(), "Initialize with metallib path")
        .def("execute_softmax", &MetalKernelExecutor::execute_softmax, 
             "Execute softmax kernel on input array")
        .def("execute_attention", &MetalKernelExecutor::execute_attention,
             "Execute attention kernel with Q, K, V inputs")
        .def("list_available_kernels", &MetalKernelExecutor::list_available_kernels,
             "List all available kernel functions")
        .def("get_device_info", &MetalKernelExecutor::get_device_info,
             "Get Metal device information");
}