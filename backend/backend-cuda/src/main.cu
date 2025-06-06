#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include "flashinfer_ops.cuh"
#include "l4ma.cuh"
#include <zmq.hpp>
#include <random>
#include "ztensor.hpp"

// Helper to fill thrust::device_vector with random floats
void fill_random(thrust::device_vector<float> &vec, float min = -1.0f, float max = 1.0f)
{
    std::vector<float> host(vec.size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &v : host)
        v = dis(gen);
    vec = host;
}

int main()
{
    std::cout << "hello world!" << std::endl;

    // constexpr size_t head_dim_ckv = 512;
    // constexpr size_t head_dim_kpe = head_dim_ckv / 8;
    // const size_t num_qo_heads = 32;

    // size_t batch_size = 8;
    // size_t seqlen = 128;
    // size_t page_size = 32;

    // auto pages_per_seq = (seqlen + page_size - 1) / page_size;
    // auto num_pages = pages_per_seq * batch_size;
    // std::vector<int32_t> kv_indptr_host{0};
    // std::vector<int32_t> kv_indicies_host;
    // std::vector<int32_t> kv_last_page_len_host;
    // for (size_t i = 0; i < batch_size; ++i)
    // {
    //     for (size_t p = 0; p < pages_per_seq; ++p)
    //     {
    //         kv_indicies_host.push_back(i * pages_per_seq + p);
    //     }
    //     kv_indptr_host.push_back(kv_indptr_host.back() + pages_per_seq);
    //     kv_last_page_len_host.push_back((seqlen - 1) % page_size + 1);
    // }
    // thrust::device_vector<int32_t> kv_indptr(kv_indptr_host);
    // thrust::device_vector<int32_t> kv_indices(kv_indicies_host);
    // thrust::device_vector<int32_t> kv_last_page_len(kv_last_page_len_host);

    // thrust::device_vector<float> q_nope(batch_size * num_qo_heads * head_dim_ckv);
    // thrust::device_vector<float> q_pe(batch_size * num_qo_heads * head_dim_kpe);
    // thrust::device_vector<float> ckv_data(num_pages * page_size * head_dim_ckv);
    // thrust::device_vector<float> kpe_data(num_pages * page_size * head_dim_kpe);
    // thrust::device_vector<float> o(q_nope.size());

    // L4maMlp test parameters
    int num_tokens = 32;
    int hidden_size = 128;
    int intermediate_size = 256;

    L4maConfig config;
    config.hidden_size = hidden_size;
    config.intermediate_size = intermediate_size;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 2;
    config.use_qkv_bias = false;
    config.rms_norm_eps = 1e-5f;
    config.vocab_size = 1000;
    config.pad_token_id = 0;
    config.num_hidden_layers = 1;

    // Random weights and biases
    thrust::device_vector<float> gate_proj_weights(intermediate_size * hidden_size);
    thrust::device_vector<float> up_proj_weights(intermediate_size * hidden_size);
    thrust::device_vector<float> down_proj_weights(hidden_size * intermediate_size);
    fill_random(gate_proj_weights);
    fill_random(up_proj_weights);
    fill_random(down_proj_weights);

    // Optionally, random biases
    std::optional<thrust::device_vector<float>> gate_proj_bias, up_proj_bias, down_proj_bias;
    gate_proj_bias = thrust::device_vector<float>(intermediate_size);
    fill_random(*gate_proj_bias);
    up_proj_bias = thrust::device_vector<float>(intermediate_size);
    fill_random(*up_proj_bias);
    down_proj_bias = thrust::device_vector<float>(hidden_size);
    fill_random(*down_proj_bias);

    // Random input
    thrust::device_vector<float> x(num_tokens * hidden_size);
    fill_random(x);

    // Output and temp buffer
    thrust::device_vector<float> output(num_tokens * hidden_size);
    thrust::device_vector<float> temp_buffer_mlp(2 * num_tokens * intermediate_size + num_tokens * hidden_size);

    // Create MLP
    L4maMlp<float> mlp(config, gate_proj_weights, up_proj_weights, down_proj_weights, gate_proj_bias, up_proj_bias, down_proj_bias);

    // Run forward
    mlp.forward(output, x, num_tokens, temp_buffer_mlp, 0);

    // Print a few output values
    std::vector<float> output_host(output.size());
    thrust::copy(output.begin(), output.end(), output_host.begin());
    std::cout << "L4maMlp output (first 8 values): ";
    for (int i = 0; i < std::min(8, (int)output_host.size()); ++i)
    {
        std::cout << output_host[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}