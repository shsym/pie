#include <metal_stdlib>
using namespace metal;

constexpr constant int kModScaleShift = 0;
constexpr constant int kModScale = 1;
constexpr constant int kModTanhGate = 2;

inline uint modulation_row(
    const device int* lane_of_row, uint has_lanes, uint row) {
  return has_lanes != 0 ? uint(lane_of_row[row]) : row;
}

template <typename T, typename TM>
[[kernel]] void modulate(
    const device T* x               [[buffer(0)]],
    const device TM* m              [[buffer(1)]],
    const device int* lane_of_row   [[buffer(2)]],
    device T* o                     [[buffer(3)]],
    const constant uint& width      [[buffer(4)]],
    const constant uint& m_width    [[buffer(5)]],
    const constant int& form        [[buffer(6)]],
    const constant uint& has_lanes  [[buffer(7)]],
    uint2 tid                       [[thread_position_in_grid]]) {
  const uint i = tid.x;
  const uint row = tid.y;
  if (i >= width) return;

  const size_t at = size_t(row) * size_t(width) + size_t(i);
  const size_t mat =
      size_t(modulation_row(lane_of_row, has_lanes, row)) * size_t(m_width);

  const float xv = float(x[at]);
  float v;
  if (form == kModScaleShift) {
    v = fma(xv, 1.0f + float(m[mat + i]), float(m[mat + i + width]));
  } else if (form == kModScale) {
    v = xv * (1.0f + float(m[mat + i]));
  } else {
    v = precise::tanh(float(m[mat + i])) * xv;
  }
  o[at] = static_cast<T>(v);
}

#define instantiate_modulate(name, itype, mtype)                       \
  template [[host_name("modulate_" #name)]]                            \
  [[kernel]] void modulate<itype, mtype>(                              \
      const device itype*, const device mtype*, const device int*,     \
      device itype*, const constant uint&, const constant uint&,       \
      const constant int&, const constant uint&, uint2);

instantiate_modulate(bfloat16, bfloat, bfloat)
instantiate_modulate(bfloat16_f32, bfloat, float)
instantiate_modulate(float32, float, float)

template <typename T, typename TM>
[[kernel]] void gated_residual_add(
    const device T* r               [[buffer(0)]],
    const device TM* g              [[buffer(1)]],
    const device T* y               [[buffer(2)]],
    const device int* lane_of_row   [[buffer(3)]],
    device T* r_out                 [[buffer(4)]],
    const constant uint& width      [[buffer(5)]],
    const constant uint& has_lanes  [[buffer(6)]],
    uint2 tid                       [[thread_position_in_grid]]) {
  const uint i = tid.x;
  const uint row = tid.y;
  if (i >= width) return;

  const size_t at = size_t(row) * size_t(width) + size_t(i);
  const size_t gat =
      size_t(modulation_row(lane_of_row, has_lanes, row)) * size_t(width) + size_t(i);
  r_out[at] = static_cast<T>(fma(float(g[gat]), float(y[at]), float(r[at])));
}

#define instantiate_gated_residual_add(name, itype, mtype)             \
  template [[host_name("gated_residual_add_" #name)]]                  \
  [[kernel]] void gated_residual_add<itype, mtype>(                    \
      const device itype*, const device mtype*, const device itype*,   \
      const device int*, device itype*, const constant uint&,          \
      const constant uint&, uint2);

instantiate_gated_residual_add(bfloat16, bfloat, bfloat)
instantiate_gated_residual_add(bfloat16_f32, bfloat, float)
instantiate_gated_residual_add(float32, float, float)
