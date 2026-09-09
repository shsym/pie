#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;

template <typename TA, typename TW, typename TY>
[[kernel]] void lane_gemm(
    const device TA* act        [[buffer(0)]],
    const device TW* w          [[buffer(1)]],
    device TY* y                [[buffer(2)]],
    const constant uint& rows   [[buffer(3)]],
    const constant uint& n      [[buffer(4)]],
    const constant uint& k      [[buffer(5)]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    uint  simd_gid              [[simdgroup_index_in_threadgroup]],
    uint  simd_lid              [[thread_index_in_simdgroup]],
    uint  simds                 [[simdgroups_per_threadgroup]]) {
  const uint c = tgid.x * simds + simd_gid;
  const uint r = tgid.y;
  if (c >= n || r >= rows) return;

  const device TA* a = act + size_t(r) * size_t(k);
  const device TW* ww = w + size_t(c) * size_t(k);
  float acc = 0.0f;
  for (uint i = simd_lid; i < k; i += 32u) {
    acc = fma(float(a[i]), float(ww[i]), acc);
  }
  acc = simd_sum(acc);
  if (simd_lid == 0) {
    y[size_t(r) * size_t(n) + size_t(c)] = static_cast<TY>(acc);
  }
}

#define instantiate_lane_gemm(name, atype, wtype, ytype)                 \
  template [[host_name("lane_gemm_" #name)]]                             \
  [[kernel]] void lane_gemm<atype, wtype, ytype>(                        \
      const device atype*, const device wtype*, device ytype*,           \
      const constant uint&, const constant uint&, const constant uint&,  \
      uint2, uint, uint, uint);

instantiate_lane_gemm(f32_bf16_f32, float, bfloat, float)
instantiate_lane_gemm(f32_bf16_bf16, float, bfloat, bfloat)
instantiate_lane_gemm(f32_f32_f32, float, float, float)
instantiate_lane_gemm(bf16_bf16_f32, bfloat, bfloat, float)
