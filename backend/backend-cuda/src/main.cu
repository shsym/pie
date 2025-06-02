
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>     
#include <cmath>         

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "flashinfer_ops.cuh"
#include <zmq.hpp>

int main() {
    std::cout << "hello world!" << std::endl;

    constexpr size_t head_dim_ckv = 512;
    constexpr size_t head_dim_kpe = head_dim_ckv / 8;
    const size_t num_qo_heads = 32;

    size_t batch_size = 8;
    size_t seqlen = 128;
    size_t page_size = 32;

    auto pages_per_seq = (seqlen + page_size - 1) / page_size;
    auto num_pages = pages_per_seq * batch_size;
    std::vector<int32_t> kv_indptr_host{0};
    std::vector<int32_t> kv_indicies_host;
    std::vector<int32_t> kv_last_page_len_host;
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t p = 0; p < pages_per_seq; ++p) {
        kv_indicies_host.push_back(i * pages_per_seq + p);
        }
        kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
        kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
    }
    thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
    thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
    thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);

    thrust::device_vector<float> q_nope(batch_size * num_qo_heads * head_dim_ckv);
    thrust::device_vector<float> q_pe(batch_size * num_qo_heads * head_dim_kpe);
    thrust::device_vector<float> ckv_data(num_pages * page_size * head_dim_ckv);
    thrust::device_vector<float> kpe_data(num_pages * page_size * head_dim_kpe);
    thrust::device_vector<float> o(q_nope.size());


    return 0;
}