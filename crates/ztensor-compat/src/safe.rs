use ztensor::{Error, Result};

fn bad(what: &str, detail: String) -> Error {
    Error::InvalidInput(format!("{what}: {detail}"))
}

#[cfg(any(feature = "npz", feature = "pickle", feature = "hdf5"))]
pub fn to_usize(what: &str, v: u64) -> Result<usize> {
    usize::try_from(v).map_err(|_| bad(what, format!("value {v} exceeds this platform's usize")))
}

pub fn add(what: &str, a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| bad(what, format!("{a} + {b} overflows")))
}

pub fn mul(what: &str, a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| bad(what, format!("{a} * {b} overflows")))
}

pub fn product(what: &str, dims: &[u64]) -> Result<u64> {
    dims.iter()
        .try_fold(1u64, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad(what, "shape product overflows".into()))
}

#[cfg(any(feature = "pickle", feature = "hdf5"))]
pub fn range(what: &str, offset: u64, length: u64, len: usize) -> Result<(usize, usize)> {
    let end = add(what, offset, length)?;
    if end > len as u64 {
        return Err(bad(
            what,
            format!("range {offset}..{end} extends past {len} bytes"),
        ));
    }
    Ok((to_usize(what, offset)?, to_usize(what, end)?))
}

#[cfg(feature = "hdf5")]
pub fn slice<'a>(what: &str, buf: &'a [u8], offset: u64, length: u64) -> Result<&'a [u8]> {
    let (start, end) = range(what, offset, length, buf.len())?;
    Ok(&buf[start..end])
}

pub fn capacity(count: u64, min_bytes_per_item: usize, available: usize) -> usize {
    let ceiling = available / min_bytes_per_item.max(1);
    count.min(ceiling as u64) as usize
}

#[cfg(any(feature = "npz", feature = "pickle", feature = "hdf5"))]
pub const MAX_ALLOC: u64 = 1 << 30;

#[cfg(any(feature = "npz", feature = "pickle", feature = "hdf5"))]
pub fn alloc_size(what: &str, n: u64) -> Result<usize> {
    if n > MAX_ALLOC {
        return Err(bad(
            what,
            format!("declared size {n} exceeds the {MAX_ALLOC}-byte allocation cap"),
        ));
    }
    to_usize(what, n)
}
