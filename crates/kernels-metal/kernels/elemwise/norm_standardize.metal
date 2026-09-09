#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void standardize(
    device T* out             [[buffer(0)]],
    const device T* bias      [[buffer(1)]],
    const device T* scale     [[buffer(2)]],
    const constant int& width [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T((float(out[i]) - float(bias[tid.x])) * float(scale[tid.x]));
}

#define instantiate_standardize(name, itype)                          \
  template [[host_name("standardize_" #name)]]                        \
  [[kernel]] void standardize<itype>(                                 \
      device itype*, const device itype*, const device itype*,        \
      const constant int&, uint2);

instantiate_standardize(bfloat16, bfloat)
