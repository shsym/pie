#include <metal_stdlib>
using namespace metal;

constexpr constant int kRopeInterleaved = 0;
constexpr constant int kRopeNeox = 1;
constexpr constant int kRopeSplit = 2;
constexpr constant int kRopeSplitLadder = 3;

inline float theta_pow(float base, float e) {
  return precise::exp(e * precise::log(base));
}

template <typename T>
[[kernel]] void rope_axes(
    const device T* x                  [[buffer(0)]],
    const device float* positions      [[buffer(1)]],
    device T* o                        [[buffer(2)]],
    const constant uint& d0            [[buffer(3)]],
    const constant uint& d1            [[buffer(4)]],
    const constant uint& d2            [[buffer(5)]],
    const constant uint& d3            [[buffer(6)]],
    const constant float& t0           [[buffer(7)]],
    const constant float& t1           [[buffer(8)]],
    const constant float& t2           [[buffer(9)]],
    const constant float& t3           [[buffer(10)]],
    const constant uint& axes          [[buffer(11)]],
    const constant uint& rotary_dim    [[buffer(12)]],
    const constant uint& head_dim      [[buffer(13)]],
    const constant uint& heads         [[buffer(14)]],
    const constant int& form           [[buffer(15)]],
    const constant uint& copy_tail     [[buffer(16)]],
    uint2 tid                          [[thread_position_in_grid]]) {

  const uint dims[4] = {d0, d1, d2, d3};
  const float thetas[4] = {t0, t1, t2, t3};
  const uint angles = rotary_dim / 2;
  const uint width = heads * head_dim;
  const uint idx = tid.x;
  const uint row = tid.y;

  const device float* pos = positions + size_t(row) * size_t(axes);
  const device T* xr = x + size_t(row) * size_t(width);
  device T* orow = o + size_t(row) * size_t(width);

  if (idx >= heads * angles) {
    if (copy_tail == 0) return;
    const uint per_head = head_dim - rotary_dim;
    if (per_head == 0) return;
    const uint tail = idx - heads * angles;
    const uint head = tail / per_head;
    if (head >= heads) return;
    const uint i = rotary_dim + tail % per_head;
    orow[head * head_dim + i] = xr[head * head_dim + i];
    return;
  }

  const uint head = idx / angles;
  const uint angle = idx % angles;
  const device T* xh = xr + size_t(head) * size_t(head_dim);
  device T* oh = orow + size_t(head) * size_t(head_dim);

  uint lo;
  uint hi;
  float cos_v = 1.0f;
  float sin_v = 0.0f;

  if (form == kRopeSplitLadder) {
    uint span = 0;
    for (uint a = 0; a < axes; ++a) span += dims[a];
    const uint pad = (heads * rotary_dim - span) / 2;
    if (idx >= pad) {
      const uint slot = idx - pad;
      const uint axis = slot % axes;
      const uint f = slot / axes;
      const uint ladder = dims[axis] / 2;
      const float exponent = ladder > 1 ? float(f) / float(ladder - 1) : 0.0f;
      const float a = pos[axis] * theta_pow(thetas[axis], exponent);
      sin_v = precise::sin(a);
      cos_v = precise::cos(a);
    }
    lo = angle;
    hi = angle + angles;
  } else {

    uint axis = 0;
    uint first_angle = 0;
    uint first_channel = 0;
    while (axis < axes && angle >= first_angle + dims[axis] / 2) {
      first_angle += dims[axis] / 2;
      first_channel += dims[axis];
      ++axis;
    }

    if (axis >= axes) return;
    const uint within = angle - first_angle;
    const float freq =
        theta_pow(thetas[axis], -2.0f * float(within) / float(dims[axis]));
    const float a = pos[axis] * freq;
    sin_v = precise::sin(a);
    cos_v = precise::cos(a);

    if (form == kRopeInterleaved) {
      lo = first_channel + 2 * within;
      hi = lo + 1;
    } else if (form == kRopeNeox) {
      lo = angle;
      hi = angle + angles;
    } else {
      lo = first_channel + within;
      hi = first_channel + dims[axis] / 2 + within;
    }
  }

  const float a = float(xh[lo]);
  const float b = float(xh[hi]);
  oh[lo] = static_cast<T>(a * cos_v - b * sin_v);
  oh[hi] = static_cast<T>(b * cos_v + a * sin_v);
}

#define instantiate_rope_axes(name, itype)                              \
  template [[host_name("rope_axes_" #name)]]                            \
  [[kernel]] void rope_axes<itype>(                                     \
      const device itype*, const device float*, device itype*,          \
      const constant uint&, const constant uint&, const constant uint&, \
      const constant uint&, const constant float&, const constant float&,\
      const constant float&, const constant float&,                     \
      const constant uint&, const constant uint&, const constant uint&, \
      const constant uint&, const constant int&, const constant uint&,  \
      uint2);

instantiate_rope_axes(bfloat16, bfloat)
instantiate_rope_axes(float32, float)
