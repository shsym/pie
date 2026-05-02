#include "model/qwen3_5.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/gated_delta_net.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor& must(const Engine& e, const std::string& name) {
    if (!e.has(name)) {
        throw std::runtime_error("qwen3_5: missing weight '" + name + "'");
    }
    return e.get(name);
}

const DeviceTensor* maybe(const Engine& e, const std::string& name) {
    return e.has(name) ? &e.get(name) : nullptr;
}

// Materialise an owned fp32 copy of `t`, accepting either fp32 or bf16
// on disk. Copies even when the source is already fp32 so the bound
// pointer is owned by `Qwen3_5Weights::owned_fp32_buffers` (uniform
// lifetime).
DeviceBuffer<float> to_fp32(const DeviceTensor& t) {
    std::size_t n = 1;
    for (auto d : t.shape()) n *= static_cast<std::size_t>(d);
    auto buf = DeviceBuffer<float>::alloc(n);
    if (t.dtype() == DType::FP32) {
        CUDA_CHECK(cudaMemcpy(buf.data(), t.data(),
                              n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    } else if (t.dtype() == DType::BF16) {
        kernels::launch_bf16_to_fp32(t.data(), buf.data(), n, /*stream=*/0);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        throw std::runtime_error("qwen3_5: unsupported dtype for fp32 conversion");
    }
    return buf;
}

// Qwen3.5 (multimodal config) nests the text tower under
// `model.language_model.`; the vision tower lives under `model.visual.`
// and is unused on the text-only path.
constexpr const char* kPrefix = "model.language_model.";

}  // namespace

Qwen3_5Weights bind_qwen3_5(Engine& engine) {
    const auto& cfg = engine.hf_config();
    const int L = cfg.num_hidden_layers;

    if (cfg.layer_types.empty() ||
        static_cast<int>(cfg.layer_types.size()) != L) {
        throw std::runtime_error(
            "qwen3_5: HfConfig.layer_types must have num_hidden_layers entries");
    }
    if (cfg.linear_num_value_heads <= 0 || cfg.linear_num_key_heads <= 0
            || cfg.linear_key_head_dim <= 0 || cfg.linear_value_head_dim <= 0
            || cfg.linear_conv_kernel_dim <= 0) {
        throw std::runtime_error(
            "qwen3_5: linear-attn dimensions are unset; check the loader's "
            "HfConfig parsing for linear_num_*_heads / linear_*_head_dim / "
            "linear_conv_kernel_dim.");
    }

    Qwen3_5Weights w;
    w.layers.resize(static_cast<std::size_t>(L));

    const std::string p = kPrefix;

    w.embed      = &must(engine, p + "embed_tokens.weight");
    w.final_norm = &must(engine, p + "norm.weight");
    // Tied lm_head: HF omits the tensor and aliases to embed_tokens.
    w.lm_head    = cfg.tie_word_embeddings
                       ? w.embed
                       : &must(engine, "lm_head.weight");

    // KV cache slot is assigned only to full-attention layers, in
    // ascending order. Linear layers don't occupy KV-cache slots —
    // their state lives in the recurrent/conv caches built by the
    // forward.
    int kv_slot = 0;
    for (int li = 0; li < L; ++li) {
        const std::string lp = p + "layers." + std::to_string(li) + ".";
        auto& Lw = w.layers[li];
        const auto& kind = cfg.layer_types[li];

        Lw.attn_norm_pre = &must(engine, lp + "input_layernorm.weight");
        Lw.mlp_norm_pre  = &must(engine, lp + "post_attention_layernorm.weight");

        // MLP weights are present on every layer (linear or full).
        Lw.gate_proj = &must(engine, lp + "mlp.gate_proj.weight");
        Lw.up_proj   = &must(engine, lp + "mlp.up_proj.weight");
        Lw.down_proj = &must(engine, lp + "mlp.down_proj.weight");

        if (kind == "linear_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::LinearAttn;
            const std::string la = lp + "linear_attn.";
            Lw.la_in_proj_qkv = &must(engine, la + "in_proj_qkv.weight");
            Lw.la_in_proj_z   = &must(engine, la + "in_proj_z.weight");
            Lw.la_in_proj_b   = &must(engine, la + "in_proj_b.weight");
            Lw.la_in_proj_a   = &must(engine, la + "in_proj_a.weight");
            // Depthwise causal conv. HF stores conv1d.weight as
            // [conv_dim, 1, K] (groups == conv_dim). bias may be omitted.
            Lw.la_conv1d_w = &must(engine, la + "conv1d.weight");
            Lw.la_conv1d_b = maybe(engine, la + "conv1d.bias");
            Lw.la_dt_bias  = &must(engine, la + "dt_bias");
            // Materialise fp32 copies of A_log + RMSNormGated.weight.
            // HF ships these as fp32 on Qwen3.5-4B and bf16 on
            // Qwen3.6-35B-A3B; the kernels read fp32 either way.
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "A_log")));
            Lw.la_A_log_fp32  = w.owned_fp32_buffers.back().data();
            w.owned_fp32_buffers.push_back(to_fp32(must(engine, la + "norm.weight")));
            Lw.la_norm_w_fp32 = w.owned_fp32_buffers.back().data();
            Lw.la_out_proj = &must(engine, la + "out_proj.weight");
            Lw.kv_layer = -1;
        } else if (kind == "full_attention") {
            Lw.kind = Qwen3_5LayerWeights::Kind::FullAttn;
            const std::string fa = lp + "self_attn.";
            // q_proj is 2× wide (query + gate fused).
            Lw.fa_q_proj = &must(engine, fa + "q_proj.weight");
            Lw.fa_k_proj = &must(engine, fa + "k_proj.weight");
            Lw.fa_v_proj = &must(engine, fa + "v_proj.weight");
            Lw.fa_o_proj = &must(engine, fa + "o_proj.weight");
            Lw.fa_q_norm = &must(engine, fa + "q_norm.weight");
            Lw.fa_k_norm = &must(engine, fa + "k_norm.weight");
            Lw.kv_layer = kv_slot++;
        } else {
            throw std::runtime_error(
                "qwen3_5: unknown layer_type '" + kind + "' at layer " +
                std::to_string(li));
        }
    }

    return w;
}

}  // namespace pie_cuda_driver::model
