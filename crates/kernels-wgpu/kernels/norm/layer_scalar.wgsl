

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;

struct Params {
    scalar: f32,
    n: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@group(0) @binding(1) var<storage, read> scalar: array<u32>;
struct Params {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: Params;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let n = params.n;
    if (w >= (n + 1u) / 2u) {
        return;
    }
    let a = out_[w];

    let v0 = pie_bf16_to_f32(a & 0xffffu) * params.scalar;
    let v1 = pie_bf16_to_f32(a >> 16u) * params.scalar;
    let lo = v0 / (1.0 + exp(-v0));
    let hi = v1 / (1.0 + exp(-v1));

    let s = pie_bf16_to_f32(pie_f32_to_bf16(params.scalar));

    let s = pie_bf16_to_f32(scalar[0] & 0xffffu);

    let lo = pie_bf16_to_f32(a & 0xffffu) * s;
    let hi = pie_bf16_to_f32(a >> 16u) * s;

    if (2u * w + 1u < n) {
        out_[w] = pie_pack_bf16(lo, hi);
    } else {
        out_[w] = pie_bf16_into(out_[w], 0u, lo);
    }
}

