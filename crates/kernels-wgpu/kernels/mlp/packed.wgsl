

@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params {
    intermediate: u32,
    beta: f32,
    up_cap: f32,
}

struct Params {
    intermediate: u32,
    limit: f32,
    alpha: f32,
}

struct Params {
    intermediate: u32,
    limit: f32,
}

struct Params {
    intermediate: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn one(g0: f32, u0: f32) -> f32 {
    var g = g0;
    var u = u0;

    let sg = params.beta * pie_tanh(g / params.beta) / (1.0 + exp(-g));
    if (params.up_cap > 0.0) {
        u = params.up_cap * pie_tanh(u / params.up_cap);
    }
    return sg * u;

    let k = 0.7978845608028654;
    let inner = k * (g + 0.044715 * g * g * g);
    let gelu = 0.5 * g * (1.0 + tanh(inner));
    return gelu * u;

    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    let sig = 1.0 / (1.0 + exp(-params.alpha * g));
    return (g * sig) * (u + 1.0);

    g = min(g, params.limit);
    u = clamp(u, -params.limit, params.limit);
    return (g / (1.0 + exp(-g))) * u;

    return (g / (1.0 + exp(-g))) * u;

}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words = params.intermediate / 2u;
    let i = gid.x;
    if (i >= words) {
        return;
    }
    let row = gid.y * words;
    let packed_row = row * 2u;
    let g = packed[packed_row + i];
    let u = packed[packed_row + words + i];
    out_[row + i] = pie_pack_bf16(
        one(pie_bf16_to_f32(g & 0xffffu), pie_bf16_to_f32(u & 0xffffu)),
        one(pie_bf16_to_f32(g >> 16u), pie_bf16_to_f32(u >> 16u)),
    );
}

