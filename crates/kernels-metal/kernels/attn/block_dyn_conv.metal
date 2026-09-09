

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void block_dyn_conv(
    const device T* x              [[buffer(0)]],
    const device int* indptr       [[buffer(1)]],
    const device T* coeff          [[buffer(2)]],
    const device T* base           [[buffer(3)]],
    device T* y                    [[buffer(4)]],
    const constant int& channels   [[buffer(5)]],
    const constant int& side       [[buffer(6)]],
    const constant int& taps       [[buffer(7)]],
    const constant int& group      [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  const int c = int(pos.x);
  const int r = int(pos.y);
  if (c >= channels) {
    return;
  }
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }
  const int span = end - begin;
  const int groups = channels / group;
  const int g = c / group;
  const size_t chans = size_t(channels);
  const size_t pitch = size_t(2 * taps) * size_t(groups);

  for (int t = 0; t < span; ++t) {
    const size_t row = size_t(begin + t);
    float acc = 0.0f;
    for (int k = 0; k < taps; ++k) {
      const int src = t - k;
      if (src < 0) {
        break;
      }
      const int at = side * taps + k;
      const float coef = float(base[size_t(at) * chans + size_t(c)])
                       + float(coeff[row * pitch + size_t(at) * size_t(groups) + size_t(g)]);
      acc += coef * float(x[size_t(begin + src) * chans + size_t(c)]);
    }
    y[row * chans + size_t(c)] = T(acc);
  }
}

#define instantiate_block_dyn_conv(name, itype)                         \
  template [[host_name("block_dyn_conv_" #name)]]                       \
  [[kernel]] void block_dyn_conv<itype>(                                \
      const device itype*, const device int*, const device itype*,      \
      const device itype*, device itype*, const constant int&,          \
      const constant int&, const constant int&, const constant int&,    \
      uint2);

instantiate_block_dyn_conv(bfloat16, bfloat)
