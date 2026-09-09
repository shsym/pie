use super::e8m0::exp2i;

pub fn decode_fp8_e4m3_elements(bytes: &[u8]) -> Vec<f64> {
    bytes
        .iter()
        .map(|&byte| {
            let sign = if byte & 0x80 != 0 { -1.0f64 } else { 1.0 };
            let exponent = i32::from((byte >> 3) & 0x0F);
            let mantissa = f64::from(byte & 0x07);
            if exponent == 0x0F && mantissa == 7.0 {
                return f64::NAN;
            }
            let magnitude = if exponent == 0 {
                mantissa / 8.0 * (-6.0f64).exp2()
            } else {
                (1.0 + mantissa / 8.0) * f64::from(exponent - 7).exp2()
            };
            sign * magnitude
        })
        .collect()
}

pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (byte >> 3) & 0xF;
    let mant = (byte & 0x7) as f32;
    if exp == 0xF && byte & 0x7 == 0x7 {
        return f32::NAN;
    }
    let value = if exp == 0 {
        mant * exp2i(-9)
    } else {
        (1.0 + mant / 8.0) * exp2i(i32::from(exp) - 7)
    };
    sign * value
}

pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        return sign | 0x7F;
    }
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    if a < 0.015625 {
        let q = (a * 512.0).round_ties_even() as u32;
        return sign | q as u8;
    }
    let bits = a.to_bits();
    let mut e = ((bits >> 23) as i32) - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let mut q = (m * 8.0).round_ties_even() as u32;
    if q == 16 {
        e += 1;
        q = 8;
    }
    sign | (((e + 7) as u8) << 3) | (q as u8 - 8)
}
