#include <metal_stdlib>
using namespace metal;

[[kernel]] void sinusoid(
    const device float* t                [[buffer(0)]],
    device float* o                      [[buffer(1)]],
    const constant uint& dim             [[buffer(2)]],
    const constant float& max_period     [[buffer(3)]],
    const constant uint& flip_sin_cos    [[buffer(4)]],
    const constant float& scale          [[buffer(5)]],
    uint2 tid                            [[thread_position_in_grid]]) {
  const uint half_dim = dim / 2;
  const uint i = tid.x;
  const uint row = tid.y;
  if (i >= half_dim) return;

  const float tv = t[row];
  const float log_period = precise::log(max_period);
  device float* orow = o + size_t(row) * size_t(dim);

  const float freq = precise::exp(-log_period * float(i) / float(half_dim));

  const float angle = scale * (tv * freq);
  const float sin_v = precise::sin(angle);
  const float cos_v = precise::cos(angle);
  orow[i] = flip_sin_cos != 0 ? cos_v : sin_v;
  orow[i + half_dim] = flip_sin_cos != 0 ? sin_v : cos_v;

  if ((dim & 1u) != 0u && i == 0u) orow[dim - 1] = 0.0f;
}
