#include <metal_stdlib>
using namespace metal;

METAL_FUNC float layernorm_group_sum(
    float acc, threadgroup float* out, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) out[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out[0];
}

template <typename T>
[[kernel]] void layernorm(
    const device T* x              [[buffer(0)]],
    const device T* w              [[buffer(1)]],
    const device T* b              [[buffer(2)]],
    device T* out                  [[buffer(3)]],
    const constant float& eps      [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float reduced[1], partials[32];

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  device T* outr = out + row;

  float acc = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    acc += float(xr[i]);
  }
  const float mean =
      layernorm_group_sum(acc, reduced, partials, simd_lane, simd_group) /
      float(axis_size);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float spread = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = float(xr[i]) - mean;
    spread += c * c;
  }
  const float inv = precise::rsqrt(
      layernorm_group_sum(spread, reduced, partials, simd_lane, simd_group) /
          float(axis_size) +
      eps);

  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = (float(xr[i]) - mean) * inv;
    outr[i] = static_cast<T>(fma(c, float(w[i]), float(b[i])));
  }
}

#define instantiate_layernorm(name, itype)                            \
  template [[host_name("layernorm_" #name)]]                          \
  [[kernel]] void layernorm<itype>(                                   \
      const device itype*, const device itype*, const device itype*,  \
      device itype*, const constant float&, const constant uint&,     \
      uint, uint, uint, uint, uint);

instantiate_layernorm(bfloat16, bfloat)

template <typename T>
[[kernel]] void layernorm_no_scale(
    const device T* x              [[buffer(0)]],
    device T* out                  [[buffer(1)]],
    const constant float& eps      [[buffer(2)]],
    const constant uint& axis_size [[buffer(3)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float reduced[1], partials[32];

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  device T* outr = out + row;

  float acc = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    acc += float(xr[i]);
  }
  const float mean =
      layernorm_group_sum(acc, reduced, partials, simd_lane, simd_group) /
      float(axis_size);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float spread = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = float(xr[i]) - mean;
    spread += c * c;
  }
  const float inv = precise::rsqrt(
      layernorm_group_sum(spread, reduced, partials, simd_lane, simd_group) /
          float(axis_size) +
      eps);

  for (uint i = lid; i < axis_size; i += tg_size) {
    outr[i] = static_cast<T>((float(xr[i]) - mean) * inv);
  }
}

#define instantiate_layernorm_no_scale(name, itype)                   \
  template [[host_name("layernorm_no_scale_" #name)]]                 \
  [[kernel]] void layernorm_no_scale<itype>(                          \
      const device itype*, device itype*,                             \
      const constant float&, const constant uint&,                    \
      uint, uint, uint, uint, uint);

instantiate_layernorm_no_scale(bfloat16, bfloat)
instantiate_layernorm_no_scale(float32, float)
