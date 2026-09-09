use std::collections::HashMap;

use crate::contract::TensorType;
use crate::contract::compile::{Leaf, Lowering};
use crate::error::{Error, OrOverflow, Result};
use crate::extent::Rect;
use crate::types::{TensorDecl, encoding_dense_element_bytes};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorValue {
    pub decl: TensorDecl,
    pub data: Vec<i64>,
}

impl TensorValue {
    pub fn new(decl: TensorDecl, data: Vec<i64>) -> Result<Self> {
        let expected = element_count(&decl.shape)?;
        if expected != data.len() {
            return Err(Error::Contract(format!(
                "reference tensor '{}' has {} values for shape {:?}",
                decl.name,
                data.len(),
                decl.shape
            )));
        }
        Ok(Self { decl, data })
    }
}

type Provenance = (usize, u64);

pub fn replay(
    lowering: &Lowering,
    ty: &TensorType,
    leaves: &HashMap<String, TensorValue>,
) -> Result<Vec<i64>> {
    let width = encoding_dense_element_bytes(&ty.encoding).ok_or_else(|| {
        Error::Unsupported(format!(
            "the reference evaluator does not model the packed encoding {:?}",
            ty.encoding
        ))
    })?;
    let elements =
        usize::try_from(lowering.elements()).or_overflow("reference output element count")?;
    let bytes = elements
        .checked_mul(width as usize)
        .or_overflow("reference output byte size")?;

    let mut from: Vec<Option<Provenance>> = vec![None; bytes];
    let rects = match lowering {
        Lowering::Copy(copies) => copies.byte_pieces(&ty.encoding)?,
        Lowering::Gather(gather) => gather.byte_rects(&ty.encoding)?,
    };
    for rect in rects {
        scatter(&rect, &mut from)?;
    }

    let zero_fill = matches!(lowering, Lowering::Copy(copies) if copies.needs_zero_fill());
    (0..elements)
        .map(|at| element(at, width, &from, lowering.leaves(), leaves, zero_fill))
        .collect()
}

fn scatter(rect: &Rect, from: &mut [Option<Provenance>]) -> Result<()> {
    let mut at = vec![0i64; rect.dims.len()];
    loop {
        let (mut src, mut dst) = (0i64, 0i64);
        for (dim, step) in rect.dims.iter().zip(&at) {
            src += dim.src_stride * step;
            dst += dim.dst_stride * step;
        }
        let src = offset(rect.src_offset, src, "source")?;
        let dst = offset(rect.dst_offset, dst, "destination")?;
        let size = from.len();
        let slot = from.get_mut(dst as usize).ok_or_else(|| {
            Error::Internal(format!(
                "the lowering writes byte {dst}, past the end of its {size}-byte output"
            ))
        })?;
        if slot.is_some() {
            return Err(Error::Internal(format!(
                "the lowering writes output byte {dst} twice"
            )));
        }
        *slot = Some((rect.leaf, src));
        if !advance(&mut at, &rect.dims) {
            return Ok(());
        }
    }
}

fn element(
    at: usize,
    width: u64,
    from: &[Option<Provenance>],
    named: &[Leaf],
    leaves: &HashMap<String, TensorValue>,
    zero_fill: bool,
) -> Result<i64> {
    let base = at * width as usize;
    let Some((leaf, src)) = from[base] else {
        if zero_fill {
            return Ok(0);
        }
        return Err(Error::Internal(format!(
            "the lowering leaves output element {at} uninitialized and does not fill"
        )));
    };
    for step in 1..width {
        let want = Some((leaf, src + step));
        if from[base + step as usize] != want {
            return Err(Error::Internal(format!(
                "output element {at} is assembled from more than one source element"
            )));
        }
    }
    if src % width != 0 {
        return Err(Error::Internal(format!(
            "output element {at} reads from byte {src}, which is not a {width}-byte boundary"
        )));
    }
    let name = named
        .get(leaf)
        .ok_or_else(|| Error::Internal(format!("the lowering names leaf {leaf}, which it has no")))?
        .name();
    let value = leaves
        .get(name)
        .ok_or_else(|| Error::Contract(format!("no reference value for '{name}'")))?;
    value
        .data
        .get((src / width) as usize)
        .copied()
        .ok_or_else(|| {
            Error::Internal(format!(
                "the lowering reads past the end of '{name}': element {} of {}",
                src / width,
                value.data.len()
            ))
        })
}

fn advance(at: &mut [i64], dims: &[crate::extent::Dim]) -> bool {
    for level in (0..at.len()).rev() {
        at[level] += 1;
        if at[level] < dims[level].count {
            return true;
        }
        at[level] = 0;
    }
    false
}

fn offset(base: u64, delta: i64, what: &str) -> Result<u64> {
    let base = i64::try_from(base).or_overflow(format!("{what} byte offset"))?;
    u64::try_from(base + delta)
        .map_err(|_| Error::Internal(format!("the {what} offset {} is negative", base + delta)))
}

fn element_count(shape: &[i64]) -> Result<usize> {
    let mut count = 1i64;
    for dim in shape {
        if *dim < 0 {
            return Err(Error::Contract(format!("negative dimension in {shape:?}")));
        }
        count = count
            .checked_mul(*dim)
            .or_overflow(format!("element count of {shape:?}"))?;
    }
    usize::try_from(count).or_overflow(format!("element count of {shape:?}"))
}
