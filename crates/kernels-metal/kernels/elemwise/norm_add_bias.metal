#include <metal_stdlib>
using namespace metal;

template <typename T, typename TB>
[[kernel]] void add_bias(
    device T* out            [[buffer(0)]],
    const device TB* bias    [[buffer(1)]],
    const constant int& width [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T(float(out[i]) + float(bias[tid.x]));
}

#define instantiate_add_bias(name, itype, btype)                   \
  template [[host_name("add_bias_" #name)]]                        \
  [[kernel]] void add_bias<itype, btype>(                          \
      device itype*, const device btype*, const constant int&, uint2);

instantiate_add_bias(bfloat16, bfloat, bfloat)
instantiate_add_bias(float32, float, float)
instantiate_add_bias(float32_bf16, float, bfloat)
