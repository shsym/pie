
@group(0) @binding(0) var<storage, read> in_: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_: array<f32>;

@group(0) @binding(0) var<storage, read> in_: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

@group(0) @binding(2) var<storage, read> rows: array<i32>;

@group(0) @binding(2) var<storage, read> rows: array<u32>;

struct Params {
    width: u32,
    count: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    let c = gid.x;
    let pitch = params.width;


    let c = gid.x;
    let pitch = params.width >> 1u;

    let i = gid.y;
    if (c >= pitch || i >= params.count) {
        return;
    }

    let at = rows[i];
    if (at < 0) {
        return;
    }
    out_[u32(at) * pitch + c] = in_[i * pitch + c];

    out_[rows[i] * pitch + c] = in_[i * pitch + c];

    out_[i * pitch + c] = in_[rows[i] * pitch + c];

}

