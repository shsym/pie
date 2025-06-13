#include "l4ma.cuh"
#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include "bpe.hpp"
#include <string>
#include <format>

void print_tokens(const std::vector<bpe::Rank> &tokens)
{
    std::cout << "[";
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        std::cout << tokens[i] << (i == tokens.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Finds the index of the maximum element in a portion of a device vector.
 * @param logits The device vector of logits.
 * @param offset The starting offset for the search.
 * @param size The number of elements to search.
 * @return The index of the maximum logit relative to the offset.
 */
int get_next_token(const thrust::device_vector<float> &logits, size_t offset, size_t size)
{
    // Find the iterator to the maximum element in the specified range
    auto max_it = thrust::max_element(logits.begin() + offset, logits.begin() + offset + size);
    // Return the index of that element by calculating the distance from the beginning of the range
    return thrust::distance(logits.begin() + offset, max_it);
}

// Formats a prompt for the Llama 3 model.
std::string llama3_format(
    const std::string &prompt,
    const std::optional<std::string> &hint,
    const std::string &system = "You are a helpful, respectful and honest assistant.")
{
    std::string temp = "<|begin_of_text|>";
    temp += std::format("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", system);
    temp += std::format("<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>", prompt);
    temp += "<|start_header_id|>assistant<|end_header_id|>\n\n";

    if (hint)
    {
        temp += *hint;
    }

    return temp;
}

int main()
{
    std::cout << "hello world!" << std::endl;

    /// tokenizer test

    std::string model_path = "/home/ingim/Workspace/model-index/meta-llama--Llama-3.2-1B-Instruct/tokenizer.model";
    auto tokenizer = bpe::llama3_tokenizer(model_path);

    std::string text = llama3_format("What is the capital of France?", std::nullopt);

    std::cout << "Original text: " << text << std::endl;

    // Encode the text
    auto tokens = tokenizer.encode_with_special_tokens(text);
    std::cout << "Encoded tokens: ";
    print_tokens(tokens);

    // Decode the tokens
    std::string decoded_text = tokenizer.decode(tokens);
    std::cout << "Decoded text: " << decoded_text << std::endl;

    // --- Print ztensor metadata for llama1b.zt ---
    std::string pie_home;
    const char *env_pie_home = std::getenv("PIE_HOME");
    if (env_pie_home && env_pie_home[0] != '\0')
    {
        pie_home = env_pie_home;
    }
    else
    {
        const char *home = std::getenv("HOME");
        if (!home)
        {
            std::cerr << "Could not determine $HOME for PIE_HOME fallback." << std::endl;
            return 1;
        }
        pie_home = std::string(home) + "/.cache/pie";
    }
    std::string zt_path = pie_home + "/llama1b.zt";

    // set config_path to "./l4ma.yaml"
    std::string config_path = "../../l4ma.yaml";
    const int MAX_TOTAL_TOKENS = 2048;

    try
    {
        // --- 2. Load Model ---
        std::cout << "Loading model from files..." << std::endl;
        // Use the static factory method to load config, weights, and construct the model
        auto model = L4maModel<__nv_bfloat16>::from_files(config_path, zt_path);
        std::cout << "Model loaded successfully." << std::endl;

        // Extract config details needed for setup
        // IMPORTANT: The model class should expose its config. For this example, we re-load it.
        // In a better design, model.config() would be a public method.
        L4maConfig config = model.get_config();
        config.print();

        // construct input_ids from tokens
        thrust::device_vector<uint32_t> input_ids(tokens.begin(), tokens.end());

        // create a uninitalized vector with size equal to the number of len(input_ids) * config.hidden_size
        thrust::device_vector<__nv_bfloat16> embed_output(input_ids.size() * config.hidden_size);

        model.embed_input_ids(input_ids, embed_output);



        

               // // --- 3. Prepare Inputs (Simulate a Tokenized Prompt) ---
        // // In a real application, this would come from a tokenizer.
        // // Let's create a sample prompt with 5 tokens.
        // thrust::host_vector<int32_t> h_input_ids = {101, 2054, 2003, 2026, 102};
        // thrust::device_vector<int32_t> d_input_ids = h_input_ids;
        // int num_input_tokens = d_input_ids.size();

        // // Create position IDs: [0, 1, 2, 3, 4]
        // thrust::device_vector<int32_t> d_position_ids(num_input_tokens);
        // thrust::sequence(d_position_ids.begin(), d_position_ids.end());
        // std::cout << "Prepared input with " << num_input_tokens << " tokens." << std::endl;

        // // --- 4. Prepare Paged KV Cache ---
        // // This simulates what an inference server's memory manager would do.
        // const int batch_size = 1; // We are processing one prompt
        // const int num_kv_heads = config.num_key_value_heads;
        // const int head_dim = config.head_dim();
        // const int num_layers = config.num_hidden_layers;

        // // Allocate the main KV cache buffers
        // const int num_cache_pages = (MAX_TOTAL_TOKENS / PAGE_SIZE) * num_layers;
        // size_t cache_buffer_size = num_cache_pages * PAGE_SIZE * num_kv_heads * head_dim;
        // thrust::device_vector<__nv_bfloat16> kv_cache_k(cache_buffer_size);
        // thrust::device_vector<__nv_bfloat16> kv_cache_v(cache_buffer_size);

        // // Metadata to describe the cache layout for this request
        // // For a single prompt prefill, the layout is simple.
        // int pages_for_request = (num_input_tokens + PAGE_SIZE - 1) / PAGE_SIZE;

        // // kv_page_indices: The list of physical page numbers assigned to this request
        // thrust::device_vector<int32_t> d_kv_page_indices(pages_for_request);
        // thrust::sequence(d_kv_page_indices.begin(), d_kv_page_indices.end()); // Assign pages [0, 1, 2, ...]

        // // kv_page_indptr: Start and end pointers into the kv_page_indices list for each sequence in the batch
        // thrust::device_vector<int32_t> d_kv_page_indptr = {0, pages_for_request};

        // // kv_last_page_lens: The number of tokens in the last page of each sequence. For prefill, it's 0.
        // thrust::device_vector<int32_t> d_kv_last_page_lens = {0};

        // // qo_indptr: Start and end indices for tokens in the flat input_ids tensor.
        // thrust::device_vector<int32_t> d_qo_indptr = {0, num_input_tokens};

        // std::cout << "KV Cache allocated and configured for prefill." << std::endl;

        // // --- 5. Run Inference ---
        // thrust::device_vector<float> d_logits;
        // cudaStream_t stream = 0; // Use default stream

        // std::cout << "\nRunning forward pass..." << std::endl;
        // model.forward(d_logits, d_input_ids, d_position_ids,
        //               kv_cache_k, kv_cache_v,
        //               thrust::raw_pointer_cast(d_kv_page_indices.data()),
        //               thrust::raw_pointer_cast(d_kv_page_indptr.data()),
        //               thrust::raw_pointer_cast(d_kv_last_page_lens.data()),
        //               thrust::raw_pointer_cast(d_qo_indptr.data()),
        //               batch_size, stream);

        // // Wait for all CUDA kernels to finish
        // cudaDeviceSynchronize();
        // std::cout << "Forward pass complete." << std::endl;

        // // --- 6. Get Result ---
        // // We want the logits for the *last* token to predict the next one.
        // size_t last_token_offset = (num_input_tokens - 1) * config.vocab_size;
        // int next_token_id = get_next_token(d_logits, last_token_offset, config.vocab_size);

        // std::cout << "\n--- Inference Result ---" << std::endl;
        // std::cout << "Predicted Next Token ID: " << next_token_id << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
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
