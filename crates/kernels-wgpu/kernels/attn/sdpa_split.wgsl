

const PIE_SPLIT_NEG_INF: f32 = -3.0e38;

@group(0) @binding(0) var<storage, read> part_o: array<u32>;
@group(0) @binding(1) var<storage, read> part_lse: array<f32>;
@group(0) @binding(2) var<storage, read_write> o_out: array<u32>;

@group(0) @binding(3) var<storage, read_write> lse_out: array<f32>;


struct Params {
    head_dim: i32,
    heads: i32,
    rows: i32,
    splits: i32,
}

@group(0) @binding(4) var<uniform> params: Params;

@group(0) @binding(3) var<uniform> params: Params;


fn finite_(x: f32) -> bool {
    return abs(x) < 3.0e38;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x * 2u;
    if (d >= u32(params.head_dim)) {
        return;
    }
    let h = gid.y;
    let t = gid.z;
    let heads = u32(params.heads);
    let rows = u32(params.rows);
    let splits = u32(params.splits);
    let col = t * heads + h;

    var top = PIE_SPLIT_NEG_INF;
    for (var s = 0u; s < splits; s++) {
        let l = part_lse[(s * rows + t) * heads + h];
        if (finite_(l)) {
            top = max(top, l);
        }
    }
    if (top == PIE_SPLIT_NEG_INF) {
        o_out[(col * u32(params.head_dim) + d) >> 1u] = pie_pack_bf16(0.0, 0.0);

        if (d == 0u) {
            lse_out[col] = bitcast<f32>(0xff800000u);
        }

        return;
    }

    var total = 0.0;
    var lo = 0.0;
    var hi = 0.0;
    for (var s = 0u; s < splits; s++) {
        let l = part_lse[(s * rows + t) * heads + h];
        if (!finite_(l)) {
            continue;
        }
        let w = exp2(l - top);
        total += w;
        let i = (((s * rows + t) * heads + h) * u32(params.head_dim) + d) >> 1u;
        let word = part_o[i];
        lo += pie_bf16_to_f32(word & 0xffffu) * w;
        hi += pie_bf16_to_f32(word >> 16u) * w;
    }
    let inv = select(1.0 / total, 1.0, total == 0.0);
    o_out[(col * u32(params.head_dim) + d) >> 1u] = pie_pack_bf16(lo * inv, hi * inv);

    if (d == 0u) {
        lse_out[col] = top + log2(total);
    }

}

