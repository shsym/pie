

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;

struct Params {
    merged: i32,
    rows: i32,
}

struct Params {
    width: i32,
    block: i32,
    rows: i32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x * 2u;
    let o = gid.y;
    if (o >= u32(params.rows)) {
        return;
    }

    if (c >= u32(params.merged)) {
        return;
    }
    let i = (o * u32(params.merged) + c) >> 1u;
    y[i] = x[i];

    if (c >= u32(params.width)) {
        return;
    }
    let width = u32(params.width);
    let base = o * u32(params.block) * width;
    var lo = 0.0;
    var hi = 0.0;
    for (var r = 0u; r < u32(params.block); r = r + 1u) {
        let word = x[(base + r * width + c) >> 1u];
        lo = lo + pie_bf16_to_f32(word & 0xffffu);
        hi = hi + pie_bf16_to_f32(word >> 16u);
    }
    let inv = 1.0 / f32(params.block);
    y[(o * width + c) >> 1u] = pie_pack_bf16(lo * inv, hi * inv);

}

