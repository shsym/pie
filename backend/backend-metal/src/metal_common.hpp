#pragma once

#include <Metal/Metal.h>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <stdexcept>

// Metal bfloat16 type mapping - matches existing convention
using bfloat16_t = uint16_t;

// Forward declarations for compatibility
struct MetalConfig;
class MetalProfiler;

/**
 * @brief Configuration for the L4ma model architecture (Metal equivalent)
 * Matches the CUDA L4maConfig structure exactly for compatibility
 */
struct L4maConfig {
    std::string type;
    int num_layers;
    int num_query_heads;
    int num_key_value_heads;
    int head_size;
    int hidden_size;
    int intermediate_size;
    int vocab_size;
    bool use_qkv_bias;
    float rms_norm_eps;
    float rope_factor;
    float rope_high_frequency_factor;
    float rope_low_frequency_factor;
    float rope_theta;
};

/**
 * @brief Error checking macro for Metal operations
 * Similar to CUDA_CHECK but for Metal API calls
 */
#define METAL_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Metal Error: " + std::string(message) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

/**
 * @brief Check if Metal device/queue is valid
 */
#define METAL_DEVICE_CHECK(device) \
    METAL_CHECK(device != nullptr, "Metal device is null")

#define METAL_QUEUE_CHECK(queue) \
    METAL_CHECK(queue != nullptr, "Metal command queue is null")

/**
 * @brief Metal device management singleton
 * Provides centralized access to Metal device and command queue
 */
class MetalContext {
public:
    static MetalContext& getInstance();

    // Initialize Metal context
    bool initialize();

    // Cleanup Metal context
    void cleanup();

    // Get Metal device
    id<MTLDevice> getDevice() const { return device_; }

    // Get command queue
    id<MTLCommandQueue> getCommandQueue() const { return commandQueue_; }

    // Check if initialized
    bool isInitialized() const { return initialized_; }

private:
    MetalContext() = default;
    ~MetalContext() { cleanup(); }

    id<MTLDevice> device_ = nullptr;
    id<MTLCommandQueue> commandQueue_ = nullptr;
    bool initialized_ = false;
};

/**
 * @brief Metal buffer wrapper for type-safe memory management
 * Provides RAII management of Metal buffers with type information
 */
template<typename T>
class MetalBuffer {
public:
    MetalBuffer() = default;

    // Create buffer with size
    explicit MetalBuffer(size_t count);

    // Create buffer from data
    MetalBuffer(const T* data, size_t count);

    // Move constructor
    MetalBuffer(MetalBuffer&& other) noexcept;

    // Move assignment
    MetalBuffer& operator=(MetalBuffer&& other) noexcept;

    // Delete copy constructor and assignment
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    ~MetalBuffer() = default;

    // Get raw Metal buffer
    id<MTLBuffer> getBuffer() const { return buffer_; }

    // Get typed pointer (for CPU access)
    T* data() const;

    // Get element count
    size_t size() const { return count_; }

    // Get byte size
    size_t byteSize() const { return count_ * sizeof(T); }

    // Check if valid
    bool isValid() const { return buffer_ != nullptr; }

    // Copy data from host
    void copyFromHost(const T* hostData, size_t count);

    // Copy data to host
    void copyToHost(T* hostData, size_t count) const;

private:
    id<MTLBuffer> buffer_ = nullptr;
    size_t count_ = 0;
};

/**
 * @brief Profile scope for Metal operations
 * Similar to CUDA ProfileScope but for Metal command buffers
 */
class MetalProfileScope {
public:
    MetalProfileScope(MetalProfiler* profiler, const std::string& name, id<MTLCommandBuffer> commandBuffer);
    ~MetalProfileScope();

    // Record a checkpoint
    void record(const std::string& checkpoint);

    // Create sub-scope
    MetalProfileScope scope(const std::string& name);

private:
    MetalProfiler* profiler_;
    std::string name_;
    id<MTLCommandBuffer> commandBuffer_;
};

/**
 * @brief Profiler for Metal operations
 * Tracks timing and performance metrics for Metal operations
 */
class MetalProfiler {
public:
    explicit MetalProfiler(bool enabled = true);

    // Create profile scope
    MetalProfileScope scope(const std::string& name, id<MTLCommandBuffer> commandBuffer);

    // Print profiling report
    void print_report();

    // Enable/disable profiling
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    friend class MetalProfileScope;
    bool enabled_;
    std::vector<std::pair<std::string, double>> timings_;
};

/**
 * @brief Type casting utilities for Metal
 * Handles conversions between different numeric types
 */
namespace MetalCast {

    /**
     * @brief Cast between different types using Metal compute
     */
    template<typename InType, typename OutType>
    void cast_type(id<MTLDevice> device,
                   id<MTLCommandQueue> commandQueue,
                   const InType* input,
                   OutType* output,
                   size_t count);

    // Specializations for common conversions
    void bfloat16_to_float(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const bfloat16_t* input, float* output, size_t count);

    void float_to_bfloat16(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                          const float* input, bfloat16_t* output, size_t count);
}

/**
 * @brief Metal command buffer error handling and retry logic
 */
namespace MetalErrorHandling {

    struct RetryConfig {
        int max_retries = 3;
        double initial_delay_ms = 10.0;
        double backoff_multiplier = 2.0;
        double max_delay_ms = 1000.0;
        bool exponential_backoff = true;
    };

    struct GPUMemoryInfo {
        size_t current_allocated;
        size_t recommended_max;
        size_t max_buffer_length;
        double usage_ratio;
        bool has_unified_memory;
    };

    /**
     * @brief Get current GPU memory information
     */
    GPUMemoryInfo getGPUMemoryInfo(id<MTLDevice> device);

    /**
     * @brief Check if GPU memory pressure is high
     */
    bool isMemoryPressureHigh(id<MTLDevice> device, double threshold = 0.8);

    /**
     * @brief Execute command buffer with enhanced retry logic
     */
    bool executeCommandBufferWithRetry(id<MTLCommandBuffer> commandBuffer,
                                      const RetryConfig& config = RetryConfig{});

    /**
     * @brief Create command buffer with memory pressure checks
     */
    id<MTLCommandBuffer> createSafeCommandBuffer(id<MTLCommandQueue> commandQueue,
                                                id<MTLDevice> device);
}

/**
 * @brief Memory utilities for Metal buffers
 */
namespace MetalMemory {

    /**
     * @brief Copy data between Metal buffers
     */
    void copyBuffer(id<MTLCommandQueue> commandQueue,
                   id<MTLBuffer> source, size_t sourceOffset,
                   id<MTLBuffer> destination, size_t destOffset,
                   size_t size);

    /**
     * @brief Fill buffer with value
     */
    template<typename T>
    void fillBuffer(id<MTLCommandQueue> commandQueue,
                   id<MTLBuffer> buffer, const T& value, size_t count);

    /**
     * @brief Zero out buffer
     */
    void zeroBuffer(id<MTLCommandQueue> commandQueue,
                   id<MTLBuffer> buffer, size_t size);
}

/**
 * @brief Compute pipeline management for Metal kernels
 * Manages compute pipeline states for different kernel types
 */
class MetalComputePipelineManager {
public:
    static MetalComputePipelineManager& getInstance();

    // Get or create compute pipeline for kernel
    id<MTLComputePipelineState> getComputePipeline(const std::string& kernelName);

    // Register kernel library
    bool registerKernelLibrary(const std::string& libraryPath);
    bool registerKernelLibraryFromSource(const std::string& source);

private:
    MetalComputePipelineManager() = default;

    id<MTLLibrary> defaultLibrary_ = nullptr;
    std::map<std::string, id<MTLComputePipelineState>> pipelines_;
};

/**
 * @brief Utility functions for Metal kernel dispatch
 */
namespace MetalDispatch {

    /**
     * @brief Calculate optimal thread group sizes for 1D dispatch
     */
    std::pair<MTLSize, MTLSize> calculateThreadGroups1D(size_t totalThreads, size_t maxThreadsPerGroup = 1024);

    /**
     * @brief Calculate optimal thread group sizes for 2D dispatch
     */
    std::pair<MTLSize, MTLSize> calculateThreadGroups2D(size_t width, size_t height,
                                                        size_t maxThreadsPerGroup = 1024);

    /**
     * @brief Calculate optimal thread group sizes for 3D dispatch
     */
    std::pair<MTLSize, MTLSize> calculateThreadGroups3D(size_t width, size_t height, size_t depth,
                                                        size_t maxThreadsPerGroup = 1024);
}

// Template implementations

template<typename T>
MetalBuffer<T>::MetalBuffer(size_t count) : count_(count) {
    auto& context = MetalContext::getInstance();
    METAL_DEVICE_CHECK(context.getDevice());

    buffer_ = [context.getDevice() newBufferWithLength:count * sizeof(T)
                                               options:MTLResourceStorageModeShared];
    METAL_CHECK(buffer_ != nullptr, "Failed to create Metal buffer");
}

template<typename T>
MetalBuffer<T>::MetalBuffer(const T* data, size_t count) : MetalBuffer(count) {
    copyFromHost(data, count);
}

template<typename T>
MetalBuffer<T>::MetalBuffer(MetalBuffer&& other) noexcept
    : buffer_(other.buffer_), count_(other.count_) {
    other.buffer_ = nullptr;
    other.count_ = 0;
}

template<typename T>
MetalBuffer<T>& MetalBuffer<T>::operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
        buffer_ = other.buffer_;
        count_ = other.count_;
        other.buffer_ = nullptr;
        other.count_ = 0;
    }
    return *this;
}

template<typename T>
T* MetalBuffer<T>::data() const {
    if (!buffer_) return nullptr;
    return static_cast<T*>([buffer_ contents]);
}

template<typename T>
void MetalBuffer<T>::copyFromHost(const T* hostData, size_t count) {
    METAL_CHECK(buffer_ != nullptr, "Buffer is null");
    METAL_CHECK(count <= count_, "Copy count exceeds buffer size");

    T* bufferData = static_cast<T*>([buffer_ contents]);
    std::memcpy(bufferData, hostData, count * sizeof(T));
}

template<typename T>
void MetalBuffer<T>::copyToHost(T* hostData, size_t count) const {
    METAL_CHECK(buffer_ != nullptr, "Buffer is null");
    METAL_CHECK(count <= count_, "Copy count exceeds buffer size");

    const T* bufferData = static_cast<const T*>([buffer_ contents]);
    std::memcpy(hostData, bufferData, count * sizeof(T));
}