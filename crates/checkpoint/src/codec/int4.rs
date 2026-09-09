pub fn decode_int4b8_elements(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() * 2);
    for byte in bytes {
        values.push(f64::from((byte & 0xF) as i8 - 8));
        values.push(f64::from((byte >> 4) as i8 - 8));
    }
    values
}
