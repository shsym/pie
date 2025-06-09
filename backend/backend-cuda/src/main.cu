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
#include <cuda_bf16.h>
#include <unordered_map>
#include <cstring>


int main()
{
    std::cout << "hello world!" << std::endl;


    // --- Print ztensor metadata for llama1b.zt ---
    std::string pie_home;
    const char* env_pie_home = std::getenv("PIE_HOME");
    if (env_pie_home && env_pie_home[0] != '\0') {
        pie_home = env_pie_home;
    } else {
        const char* home = std::getenv("HOME");
        if (!home) {
            std::cerr << "Could not determine $HOME for PIE_HOME fallback." << std::endl;
            return 1;
        }
        pie_home = std::string(home) + "/.cache/pie";
    }
    std::string zt_path = pie_home + "/llama1b.zt";
    std::cout << "Reading ztensor file: " << zt_path << std::endl;
    std::unordered_map<std::string, thrust::device_vector<__nv_bfloat16>> device_tensors;
    try {
        ztensor::zTensorReader reader(zt_path);
        auto tensor_names = reader.list_tensors();
        std::cout << "Tensors in file (" << tensor_names.size() << "):\n";
        for (const auto& name : tensor_names) {
            const auto& info = reader.get_tensor_info(name);
            std::cout << "- name: " << info.name << "\n"
                      << "  dtype: " << info.dtype << "\n"
                      << "  shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                std::cout << info.shape[i];
                if (i + 1 < info.shape.size()) std::cout << ", ";
            }
            std::cout << "]\n" << std::endl;

            // --- Efficiently load and upload as bfloat16 ---
            const void* raw_ptr = reader.get_raw_tensor_pointer(name);
            size_t numel = info.num_elements();
            thrust::device_vector<__nv_bfloat16> dev_bf16(numel);
            cudaMemcpy(thrust::raw_pointer_cast(dev_bf16.data()), raw_ptr, numel * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
            device_tensors[name] = std::move(dev_bf16);
        }
        // --- Print first 10 elements for each tensor ---
        std::cout << "\nTensor first 10 elements (on GPU):\n";
        for (const auto& kv : device_tensors) {
            const std::string& name = kv.first;
            if (name.find("model.layers.1.mlp") != 0) continue;
            const thrust::device_vector<__nv_bfloat16>& dev_bf16 = kv.second;
            size_t numel = dev_bf16.size();
            size_t print_count = std::min<size_t>(10, numel);
            std::vector<__nv_bfloat16> host_bf16(print_count);
            cudaMemcpy(host_bf16.data(), thrust::raw_pointer_cast(dev_bf16.data()), print_count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            std::cout << "- " << name << ": ";
            for (size_t i = 0; i < print_count; ++i) {
                float val = __bfloat162float(host_bf16[i]);
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }

        // --- L4MA model config and weights from files ---
        auto model = L4maModel<__nv_bfloat16>::from_files("./l4ma.yaml", zt_path);
        // --- Dummy input for forward pass ---
        int num_tokens = 4;
        thrust::device_vector<__nv_bfloat16> x(num_tokens * model.config_.hidden_size, __float2bfloat16(1.0f));
        thrust::device_vector<__nv_bfloat16> output(num_tokens * model.config_.hidden_size);
        // Forward pass (stub)
        model.forward(output, x, nullptr, /*kv_cache_k*/ x, /*kv_cache_v*/ x, nullptr, nullptr, nullptr, nullptr, nullptr, false, num_tokens, 1, 0, 0, 0);
        // Print a few output values
        std::vector<__nv_bfloat16> output_host(output.size());
        cudaMemcpy(output_host.data(), thrust::raw_pointer_cast(output.data()), output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        std::cout << "L4maModel output (first 8 values): ";
        for (int i = 0; i < std::min(8, (int)output_host.size()); ++i) {
            std::cout << __bfloat162float(output_host[i]) << " ";
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error reading ztensor file: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

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
// int num_tokens = 32;
// int hidden_size = 128;
// int intermediate_size = 256;

// L4maConfig config;
// config.hidden_size = hidden_size;
// config.intermediate_size = intermediate_size;
// config.num_attention_heads = 2;
// config.num_key_value_heads = 2;
// config.use_qkv_bias = false;
// config.rms_norm_eps = 1e-5f;
// config.vocab_size = 1000;
// config.pad_token_id = 0;
// config.num_hidden_layers = 1;

// // Random weights and biases
// thrust::device_vector<float> gate_proj_weights(intermediate_size * hidden_size);
// thrust::device_vector<float> up_proj_weights(intermediate_size * hidden_size);
// thrust::device_vector<float> down_proj_weights(hidden_size * intermediate_size);
// fill_random(gate_proj_weights);
// fill_random(up_proj_weights);
// fill_random(down_proj_weights);

// // Optionally, random biases
// std::optional<thrust::device_vector<float>> gate_proj_bias, up_proj_bias, down_proj_bias;
// gate_proj_bias = thrust::device_vector<float>(intermediate_size);
// fill_random(*gate_proj_bias);
// up_proj_bias = thrust::device_vector<float>(intermediate_size);
// fill_random(*up_proj_bias);
// down_proj_bias = thrust::device_vector<float>(hidden_size);
// fill_random(*down_proj_bias);

// // Random input
// thrust::device_vector<float> x(num_tokens * hidden_size);
// fill_random(x);

// // Output and temp buffer
// thrust::device_vector<float> output(num_tokens * hidden_size);
// thrust::device_vector<float> temp_buffer_mlp(2 * num_tokens * intermediate_size + num_tokens * hidden_size);

// // Create MLP
// L4maMlp<float> mlp(config, gate_proj_weights, up_proj_weights, down_proj_weights, gate_proj_bias, up_proj_bias, down_proj_bias);

// // Run forward
// mlp.forward(output, x, num_tokens, temp_buffer_mlp, 0);

// // Print a few output values
// std::vector<float> output_host(output.size());
// thrust::copy(output.begin(), output.end(), output_host.begin());
// std::cout << "L4maMlp output (first 8 values): ";
// for (int i = 0; i < std::min(8, (int)output_host.size()); ++i)
// {
//     std::cout << output_host[i] << " ";
// }
// std::cout << std::endl;
