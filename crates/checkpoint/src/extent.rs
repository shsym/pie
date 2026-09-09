use serde::{Deserialize, Serialize};

use crate::error::{Error, OrOverflow, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dim {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: Vec<Dim>,
}

impl Extent {
    pub fn byte_run(bytes: u64) -> Self {
        Self {
            base_offset: 0,
            element_bytes: 1,
            dims: vec![Dim {
                count: i64::try_from(bytes).unwrap_or(i64::MAX),
                src_stride: 1,
                dst_stride: 1,
            }],
        }
    }

    pub fn is_dense(&self) -> bool {
        self.walk_dense(|dim, stride| dim.src_stride == stride && dim.dst_stride == stride)
    }

    pub fn has_dense_destination(&self) -> bool {
        self.walk_dense(|dim, stride| dim.dst_stride == stride)
    }

    pub fn is_byte_run(&self) -> bool {
        self.base_offset == 0 && self.element_bytes == 1 && self.dims.len() == 1 && self.is_dense()
    }

    pub fn dense(shape: &[i64], element_bytes: u64) -> Self {
        let mut stride = i64::try_from(element_bytes).unwrap_or(i64::MAX);
        let mut dims = Vec::with_capacity(shape.len());
        for dim in shape.iter().rev() {
            dims.push(Dim {
                count: *dim,
                src_stride: stride,
                dst_stride: stride,
            });
            stride = stride.saturating_mul(*dim);
        }
        dims.reverse();
        Self {
            base_offset: 0,
            element_bytes: u32::try_from(element_bytes).unwrap_or(u32::MAX),
            dims,
        }
    }

    fn walk_dense(&self, packed: impl Fn(&Dim, i64) -> bool) -> bool {
        let mut stride = i64::from(self.element_bytes);
        for dim in self.dims.iter().rev() {
            if dim.count < 0 || !packed(dim, stride) {
                return false;
            }
            match stride.checked_mul(dim.count) {
                Some(next) => stride = next,
                None => return false,
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rect {
    pub leaf: usize,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub dims: Vec<Dim>,
}

impl Rect {
    pub fn span(leaf: usize, src_offset: u64, dst_offset: u64, bytes: u64) -> Self {
        Self {
            leaf,
            src_offset,
            dst_offset,
            dims: vec![Dim {
                count: i64::try_from(bytes).unwrap_or(i64::MAX),
                src_stride: 1,
                dst_stride: 1,
            }],
        }
    }

    pub fn bytes(&self) -> u64 {
        self.dims.iter().map(|dim| dim.count).product::<i64>() as u64
    }

    pub fn is_byte_run(&self) -> bool {
        self.dims.len() == 1 && self.dims[0].src_stride == 1 && self.dims[0].dst_stride == 1
    }

    pub fn split(&self) -> Result<(Extent, Extent)> {
        let bytes = self.bytes();
        if self.is_byte_run() {
            return Ok((Extent::byte_run(bytes), Extent::byte_run(bytes)));
        }
        let (inner, outer) = self
            .dims
            .split_last()
            .ok_or_else(|| Error::Contract("copy has no extent".to_string()))?;
        if inner.src_stride != 1 || inner.dst_stride != 1 {
            return Err(Error::Contract(
                "copy has no contiguous inner block".to_string(),
            ));
        }
        let element_bytes =
            u32::try_from(inner.count).or_overflow("copy inner block exceeds 4 GiB")?;

        let mut dense = i64::from(element_bytes);
        let mut source_dims = Vec::with_capacity(outer.len());
        let mut dest_dims = Vec::with_capacity(outer.len());
        for dim in outer.iter().rev() {
            if dim.dst_stride != dense {
                return Err(Error::Contract(
                    "copy writes a non-contiguous destination".to_string(),
                ));
            }
            source_dims.push(Dim {
                count: dim.count,
                src_stride: dim.src_stride,
                dst_stride: dense,
            });
            dest_dims.push(Dim {
                count: dim.count,
                src_stride: dense,
                dst_stride: dense,
            });
            dense = dense
                .checked_mul(dim.count)
                .or_overflow("copy extent overflow")?;
        }
        source_dims.reverse();
        dest_dims.reverse();
        Ok((
            Extent {
                base_offset: 0,
                element_bytes,
                dims: source_dims,
            },
            Extent {
                base_offset: 0,
                element_bytes,
                dims: dest_dims,
            },
        ))
    }
}
