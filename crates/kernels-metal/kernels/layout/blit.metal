#include <metal_stdlib>
using namespace metal;

kernel void blit_bfloat16(
    const device ushort* src [[buffer(0)]],
    device ushort* dst [[buffer(1)]],
    const constant uint& elements [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= elements) return;
  dst[tid] = src[tid];
}

kernel void rs_copy_words(
    const device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    const constant uint& words [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= words) return;
  dst[tid] = src[tid];
}
