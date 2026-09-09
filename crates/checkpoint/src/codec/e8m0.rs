#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn encode_e8m0(absmax: f32) -> u8 {
    if !(absmax > 0.0) {
        return 0;
    }
    let b = ((absmax / 6.0).log2().ceil() + 127.0) as i32;
    b.clamp(0, 254) as u8
}

pub fn exp2_e8m0(sb: u8) -> f32 {
    if sb == 0 {
        f32::from_bits(1 << 22)
    } else {
        f32::from_bits(u32::from(sb) << 23)
    }
}

pub fn exp2i(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}
