#include "tensor.hpp"
#include "common.cuh"

#include <iostream>
#include <iomanip>
#include <numeric>

template <typename T>
__global__ void sum_reduction_kernel(const T* input, float* output, int size) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            sdata[tid] = __bfloat162float(input[i]);
        } else {
            sdata[tid] = static_cast<float>(input[i]);
        }
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


// --- Tensor Method Implementations ---

template <typename T>
void Tensor<T>::CudaDeleter::operator()(void* ptr) const {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

template <typename T>
Tensor<T>::Tensor(size_t size) : size_(size), offset_(0) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size_ * sizeof(T)));
    data_ = std::shared_ptr<T>(ptr, CudaDeleter());
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& other, size_t offset)
    : data_(other.data_), size_(other.size_ - offset), offset_(other.offset_ + offset) {
    if (offset >= other.size_) {
        throw std::out_of_range("Offset is out of bounds of the parent tensor.");
    }
}

template <typename T>
Tensor<T>::Tensor(std::shared_ptr<T> data, size_t size, size_t offset)
    : data_(std::move(data)), size_(size), offset_(offset) {}



template <typename T>
Tensor<T>::Tensor(const ByteTensor& byte_buffer, size_t byte_offset, size_t count) {
    if (byte_offset + count * sizeof(T) > byte_buffer.size() * sizeof(uint8_t)) {
        throw std::out_of_range("Reinterpreting view is out of bounds of the source byte buffer.");
    }
    
    // This now works because of the corrected 'friend' declaration.
    T* typed_ptr = reinterpret_cast<T*>(byte_buffer.data() + byte_offset);
    this->data_ = std::shared_ptr<T>(byte_buffer.data_, typed_ptr);
    this->size_ = count;
    this->offset_ = 0;
}


template <typename T>
size_t Tensor<T>::size() const { return size_; }

template <typename T>
T* Tensor<T>::data() const { return data_.get() + offset_; }

template <typename T>
std::vector<T> Tensor<T>::to_vector() const {
    std::vector<T> host_vector(size_);
    CUDA_CHECK(cudaMemcpy(host_vector.data(), this->data(), size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return host_vector;
}

template <typename T>
void Tensor<T>::from_vector(const std::vector<T>& data) {
    if (data.size() != size_) {
        throw std::invalid_argument("Host vector size does not match tensor size.");
    }
    CUDA_CHECK(cudaMemcpy(this->data(), data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void Tensor<T>::from_pointer(const T* data, size_t count) {
    if (count != size_) {
        throw std::invalid_argument("Host data count does not match tensor size.");
    }
    CUDA_CHECK(cudaMemcpy(this->data(), data, size_ * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
float Tensor<T>::mean() const requires std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>
{
    if (size_ == 0) return 0.0f;

    const int threads_per_block = 256;
    const int num_blocks = (size_ + threads_per_block - 1) / threads_per_block;
    Tensor<float> partial_sums(num_blocks);

    size_t shared_mem_size = threads_per_block * sizeof(float);
    sum_reduction_kernel<T><<<num_blocks, threads_per_block, shared_mem_size>>>(this->data(), partial_sums.data(), size_);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_partial_sums = partial_sums.to_vector();
    float total_sum = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0f);
    return total_sum / size_;
}

template <typename T>
void Tensor<T>::print(size_t start_index, size_t count) const {
    if (start_index >= size_) {
        std::cout << "[]" << std::endl;
        return;
    }
    if (count == static_cast<size_t>(-1) || start_index + count > size_) {
        count = size_ - start_index;
    }
    if (count == 0) {
        std::cout << "[]" << std::endl;
        return;
    }

    Tensor<T> view(*this, start_index);
    std::vector<T> host_data(count);
    CUDA_CHECK(cudaMemcpy(host_data.data(), view.data(), count * sizeof(T), cudaMemcpyDeviceToHost));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "[";
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            std::cout << __bfloat162float(host_data[i]);
        } else {
            std::cout << host_data[i];
        }
        if (i < count - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

template class Tensor<float>;
template class Tensor<__nv_bfloat16>;
template class Tensor<int32_t>;
template class Tensor<uint32_t>;
template class Tensor<uint8_t>;