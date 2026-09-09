#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

constexpr constant int kMaxBlocks = 32;

inline float blend_block_sum(
    float acc, threadgroup float* partials, uint simd_lane, uint simd_group,
    uint simds) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  if (simd_group == 0) {
    total = simd_sum(simd_lane < simds ? partials[simd_lane] : 0.0f);
    if (simd_lane == 0) partials[0] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return partials[0];
}

template <typename T>
[[kernel]] void res_blend(
    const device T* prefix          [[buffer(0)]],
    const device T* blocks          [[buffer(1)]],
    const device T* norm_weight     [[buffer(2)]],
    const device T* proj_weight     [[buffer(3)]],
    device T* out                   [[buffer(4)]],
    const constant uint& n_blocks   [[buffer(5)]],
    const constant uint& hidden     [[buffer(6)]],
    const constant uint& block_rows [[buffer(7)]],
    const constant float& eps       [[buffer(8)]],
    uint gid                        [[threadgroup_position_in_grid]],
    uint lid                        [[thread_position_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_group                 [[simdgroup_index_in_threadgroup]],
    uint tg_size                    [[threads_per_threadgroup]]) {
  threadgroup float partials[32];
  threadgroup float prob_s[kMaxBlocks + 1];

  const uint row = gid;
  const size_t token_off = size_t(row) * size_t(hidden);
  const uint candidates = n_blocks + 1u;
  const uint simds = (tg_size + 31u) / 32u;

  for (uint j = 0; j < candidates; ++j) {
    const device T* v = j < n_blocks
        ? blocks + (size_t(j) * size_t(block_rows) + size_t(row)) * size_t(hidden)
        : prefix + token_off;

    float ss = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      const float x = float(v[h]);
      ss = fma(x, x, ss);
    }
    ss = blend_block_sum(ss, partials, simd_lane, simd_group, simds);
    const float scale = precise::rsqrt(ss / float(hidden) + eps);

    float dot = 0.0f;
    for (uint h = lid; h < hidden; h += tg_size) {
      dot = fma(float(v[h]) * scale, float(norm_weight[h]) * float(proj_weight[h]), dot);
    }
    dot = blend_block_sum(dot, partials, simd_lane, simd_group, simds);
    if (lid == 0) prob_s[j] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lid == 0) {
    float m = prob_s[0];
    for (uint j = 1; j < candidates; ++j) m = max(m, prob_s[j]);
    float sum = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      prob_s[j] = precise::exp(prob_s[j] - m);
      sum += prob_s[j];
    }
    const float inv = 1.0f / sum;
    for (uint j = 0; j < candidates; ++j) prob_s[j] *= inv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint h = lid; h < hidden; h += tg_size) {
    float acc = 0.0f;
    for (uint j = 0; j < candidates; ++j) {
      const device T* v = j < n_blocks
          ? blocks + (size_t(j) * size_t(block_rows) + size_t(row)) * size_t(hidden)
          : prefix + token_off;
      acc = fma(prob_s[j], float(v[h]), acc);
    }
    out[token_off + h] = static_cast<T>(acc);
  }
}

#define instantiate_res_blend(name, itype)                                \
  template [[host_name("res_blend_" #name)]]                              \
  [[kernel]] void res_blend<itype>(                                       \
      const device itype*, const device itype*, const device itype*,      \
      const device itype*, device itype*, const constant uint&,           \
      const constant uint&, const constant uint&, const constant float&,  \
      uint, uint, uint, uint, uint);

instantiate_res_blend(bfloat16, bfloat)
instantiate_res_blend(float32, float)
