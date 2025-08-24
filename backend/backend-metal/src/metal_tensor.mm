#import "metal_tensor.hpp"
#import "metal_common.hpp"
#import <Metal/Metal.h>
#include <algorithm>
#include <numeric>

// MetalTensorOps implementations
namespace MetalTensorOps {
    
    // Element-wise operations
    template<typename T>
    void add(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result) {
        if (a.shape() != b.shape() || a.shape() != result.shape()) {
            throw std::runtime_error("Shape mismatch in tensor addition");
        }
        
        size_t size = a.size();
        if (size == 0) return;
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result.data();
        
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
    }
    
    template<typename T>
    void mul(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result) {
        if (a.shape() != b.shape() || a.shape() != result.shape()) {
            throw std::runtime_error("Shape mismatch in tensor multiplication");
        }
        
        size_t size = a.size();
        if (size == 0) return;
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result.data();
        
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = a_data[i] * b_data[i];
        }
    }
    
    template<typename T>
    void addScalar(const MetalTensor<T>& input, T scalar, MetalTensor<T>& result) {
        if (input.shape() != result.shape()) {
            throw std::runtime_error("Shape mismatch in scalar addition");
        }
        
        size_t size = input.size();
        if (size == 0) return;
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* input_data = input.data();
        T* result_data = result.data();
        
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = input_data[i] + scalar;
        }
    }
    
    template<typename T>
    void mulScalar(const MetalTensor<T>& input, T scalar, MetalTensor<T>& result) {
        if (input.shape() != result.shape()) {
            throw std::runtime_error("Shape mismatch in scalar multiplication");
        }
        
        size_t size = input.size();
        if (size == 0) return;
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* input_data = input.data();
        T* result_data = result.data();
        
        for (size_t i = 0; i < size; ++i) {
            result_data[i] = input_data[i] * scalar;
        }
    }
    
    // Reduction operations
    template<typename T>
    T sum(const MetalTensor<T>& tensor) {
        size_t size = tensor.size();
        if (size == 0) return T(0);
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* data = tensor.data();
        T result = T(0);
        
        for (size_t i = 0; i < size; ++i) {
            result += data[i];
        }
        
        return result;
    }
    
    template<typename T>
    T mean(const MetalTensor<T>& tensor) {
        size_t size = tensor.size();
        if (size == 0) return T(0);
        
        T total = sum(tensor);
        return total / static_cast<T>(size);
    }
    
    template<typename T>
    T max(const MetalTensor<T>& tensor) {
        size_t size = tensor.size();
        if (size == 0) throw std::runtime_error("Cannot find max of empty tensor");
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* data = tensor.data();
        T result = data[0];
        
        for (size_t i = 1; i < size; ++i) {
            if (data[i] > result) {
                result = data[i];
            }
        }
        
        return result;
    }
    
    template<typename T>
    T min(const MetalTensor<T>& tensor) {
        size_t size = tensor.size();
        if (size == 0) throw std::runtime_error("Cannot find min of empty tensor");
        
        // For now, do CPU operation - should be replaced with Metal kernel
        const T* data = tensor.data();
        T result = data[0];
        
        for (size_t i = 1; i < size; ++i) {
            if (data[i] < result) {
                result = data[i];
            }
        }
        
        return result;
    }
    
    // Matrix operations
    template<typename T>
    void matmul(const MetalTensor<T>& a, const MetalTensor<T>& b, MetalTensor<T>& result) {
        const auto& a_shape = a.shape();
        const auto& b_shape = b.shape();
        const auto& result_shape = result.shape();
        
        // Simple 2D matrix multiplication check
        if (a_shape.size() != 2 || b_shape.size() != 2 || result_shape.size() != 2) {
            throw std::runtime_error("matmul currently only supports 2D tensors");
        }
        
        size_t M = a_shape[0];
        size_t K = a_shape[1];
        size_t N = b_shape[1];
        
        if (b_shape[0] != K) {
            throw std::runtime_error("Matrix dimension mismatch in matmul");
        }
        
        if (result_shape[0] != M || result_shape[1] != N) {
            throw std::runtime_error("Result shape mismatch in matmul");
        }
        
        // Simple CPU implementation - should be replaced with Metal kernel or use metal_gemm
        const T* a_data = a.data();
        const T* b_data = b.data();
        T* result_data = result.data();
        
        // Initialize result to zero
        std::fill(result_data, result_data + M * N, T(0));
        
        // Perform matrix multiplication
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                for (size_t k = 0; k < K; ++k) {
                    result_data[m * N + n] += a_data[m * K + k] * b_data[k * N + n];
                }
            }
        }
    }
    
    // Transpose operations
    template<typename T>
    void transpose2D(const MetalTensor<T>& input, MetalTensor<T>& result) {
        const auto& input_shape = input.shape();
        const auto& result_shape = result.shape();
        
        if (input_shape.size() != 2 || result_shape.size() != 2) {
            throw std::runtime_error("transpose2D requires 2D tensors");
        }
        
        size_t rows = input_shape[0];
        size_t cols = input_shape[1];
        
        if (result_shape[0] != cols || result_shape[1] != rows) {
            throw std::runtime_error("Result shape mismatch in transpose2D");
        }
        
        // CPU implementation - should be replaced with Metal kernel
        const T* input_data = input.data();
        T* result_data = result.data();
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result_data[j * rows + i] = input_data[i * cols + j];
            }
        }
    }
    
    template<typename T>
    void transpose(const MetalTensor<T>& input, MetalTensor<T>& result) {
        // For now, only support 2D transpose
        transpose2D(input, result);
    }
}

// Additional MetalTensor member function implementations that need to be in .mm file

template<typename T>
MetalTensor<T> MetalTensor<T>::view(const std::vector<size_t>& new_shape) const {
    size_t new_size = 1;
    for (size_t dim : new_shape) {
        new_size *= dim;
    }
    
    if (new_size != total_size_) {
        throw std::runtime_error("View size mismatch");
    }
    
    MetalTensor<T> view_tensor;
    view_tensor.shape_ = new_shape;
    view_tensor.total_size_ = new_size;
    view_tensor.buffer_ = std::move(const_cast<MetalBuffer<T>&>(buffer_)); // Share buffer
    view_tensor.calculateStrides();
    
    return view_tensor;
}

template<typename T>
MetalTensor<T> MetalTensor<T>::slice(size_t start, size_t end) const {
    if (shape_.size() != 1) {
        throw std::runtime_error("Simple slice only supports 1D tensors");
    }
    
    if (start >= end || end > shape_[0]) {
        throw std::runtime_error("Invalid slice range");
    }
    
    size_t slice_size = end - start;
    MetalTensor<T> result({slice_size});
    
    // Copy sliced data
    const T* src = data() + start;
    T* dst = result.data();
    std::copy(src, src + slice_size, dst);
    
    return result;
}

template<typename T>
MetalTensor<T> MetalTensor<T>::slice(const std::vector<std::pair<size_t, size_t>>& ranges) const {
    if (ranges.size() != shape_.size()) {
        throw std::runtime_error("Range count must match tensor dimensions");
    }
    
    // Calculate new shape
    std::vector<size_t> new_shape;
    for (size_t i = 0; i < ranges.size(); ++i) {
        size_t start = ranges[i].first;
        size_t end = ranges[i].second;
        
        if (start >= end || end > shape_[i]) {
            throw std::runtime_error("Invalid slice range for dimension " + std::to_string(i));
        }
        
        new_shape.push_back(end - start);
    }
    
    MetalTensor<T> result(new_shape);
    
    // For now, simple copy - should be optimized with proper slicing logic
    // This is a simplified implementation
    if (ranges.size() == 2 && new_shape.size() == 2) {
        // 2D slice
        size_t src_cols = shape_[1];
        size_t dst_cols = new_shape[1];
        
        const T* src_data = data();
        T* dst_data = result.data();
        
        for (size_t i = 0; i < new_shape[0]; ++i) {
            size_t src_row = ranges[0].first + i;
            size_t src_col_start = ranges[1].first;
            
            const T* src_row_ptr = src_data + src_row * src_cols + src_col_start;
            T* dst_row_ptr = dst_data + i * dst_cols;
            
            std::copy(src_row_ptr, src_row_ptr + dst_cols, dst_row_ptr);
        }
    } else {
        throw std::runtime_error("Multi-dimensional slicing not fully implemented");
    }
    
    return result;
}

// Factory function implementations that need Metal context
namespace MetalTensorFactory {
    
    template<typename T>
    MetalTensor<T> view(id<MTLBuffer> buffer, const std::vector<size_t>& shape, size_t offset) {
        MetalTensor<T> tensor;
        tensor.shape_ = shape;
        tensor.total_size_ = tensor.calculateTotalSize();
        tensor.calculateStrides();
        
        // Create a MetalBuffer wrapper around the existing MTLBuffer
        // This is a bit tricky since MetalBuffer expects to own the buffer
        // For now, we'll throw an error - this would need proper implementation
        throw std::runtime_error("MetalTensorFactory::view not fully implemented - needs buffer sharing mechanism");
    }
}

// Explicit template instantiations for common types
template class MetalTensor<float>;
template class MetalTensor<bfloat16_t>;
template class MetalTensor<int32_t>;
template class MetalTensor<uint32_t>;

// Template instantiations for MetalTensorOps
namespace MetalTensorOps {
    // Element-wise operations
    template void add<float>(const MetalTensor<float>&, const MetalTensor<float>&, MetalTensor<float>&);
    template void add<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, MetalTensor<bfloat16_t>&);
    
    template void mul<float>(const MetalTensor<float>&, const MetalTensor<float>&, MetalTensor<float>&);
    template void mul<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, MetalTensor<bfloat16_t>&);
    
    template void addScalar<float>(const MetalTensor<float>&, float, MetalTensor<float>&);
    template void addScalar<bfloat16_t>(const MetalTensor<bfloat16_t>&, bfloat16_t, MetalTensor<bfloat16_t>&);
    
    template void mulScalar<float>(const MetalTensor<float>&, float, MetalTensor<float>&);
    template void mulScalar<bfloat16_t>(const MetalTensor<bfloat16_t>&, bfloat16_t, MetalTensor<bfloat16_t>&);
    
    // Reduction operations
    template float sum<float>(const MetalTensor<float>&);
    template float mean<float>(const MetalTensor<float>&);
    template float max<float>(const MetalTensor<float>&);
    template float min<float>(const MetalTensor<float>&);
    
    // Matrix operations
    template void matmul<float>(const MetalTensor<float>&, const MetalTensor<float>&, MetalTensor<float>&);
    template void matmul<bfloat16_t>(const MetalTensor<bfloat16_t>&, const MetalTensor<bfloat16_t>&, MetalTensor<bfloat16_t>&);
    
    // Transpose operations
    template void transpose<float>(const MetalTensor<float>&, MetalTensor<float>&);
    template void transpose<bfloat16_t>(const MetalTensor<bfloat16_t>&, MetalTensor<bfloat16_t>&);
    template void transpose2D<float>(const MetalTensor<float>&, MetalTensor<float>&);
    template void transpose2D<bfloat16_t>(const MetalTensor<bfloat16_t>&, MetalTensor<bfloat16_t>&);
}

// Factory function instantiations
namespace MetalTensorFactory {
    template MetalTensor<float> zeros<float>(const std::vector<size_t>&);
    template MetalTensor<bfloat16_t> zeros<bfloat16_t>(const std::vector<size_t>&);
    
    template MetalTensor<float> ones<float>(const std::vector<size_t>&);
    template MetalTensor<bfloat16_t> ones<bfloat16_t>(const std::vector<size_t>&);
    
    template MetalTensor<float> full<float>(const std::vector<size_t>&, const float&);
    template MetalTensor<bfloat16_t> full<bfloat16_t>(const std::vector<size_t>&, const bfloat16_t&);
    
    template MetalTensor<float> fromData<float>(const float*, const std::vector<size_t>&);
    template MetalTensor<bfloat16_t> fromData<bfloat16_t>(const bfloat16_t*, const std::vector<size_t>&);
}