use alloc::vec::Vec;

use eta_ir::types::{Dtype, Shape, ValueType};

use super::Value;

pub(super) fn lanes_f32(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(x) => x.clone(),
        Value::I32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::U32(x) => x.iter().map(|&a| a as f32).collect(),
        Value::Bool(x) => x.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect(),
    }
}

pub(super) fn lanes_i64(v: &Value) -> Vec<i64> {
    match v {
        Value::F32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::I32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::U32(x) => x.iter().map(|&a| a as i64).collect(),
        Value::Bool(x) => x.iter().map(|&b| b as i64).collect(),
    }
}

pub(super) fn from_i64(dtype: Dtype, x: Vec<i64>) -> Value {
    match dtype {
        Dtype::I32 => Value::I32(x.iter().map(|&a| a as i32).collect()),
        Dtype::U32 => Value::U32(x.iter().map(|&a| a as u32).collect()),
        Dtype::F32 => Value::F32(x.iter().map(|&a| a as f32).collect()),
        Dtype::Bool => Value::Bool(x.iter().map(|&a| a != 0).collect()),
        _ => super::no_interpreter_lane(dtype),
    }
}

pub(super) fn pick(len: usize, i: usize) -> usize {
    if len == 1 { 0 } else { i }
}

pub(super) fn bin_arith(
    a: &Value,
    b: &Value,
    dtype: Dtype,
    f_f: impl Fn(f32, f32) -> f32,
    f_i: impl Fn(i64, i64) -> i64,
) -> Value {
    if dtype == Dtype::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::F32(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        from_i64(
            dtype,
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

pub(super) fn cmp_op(
    a: &Value,
    b: &Value,
    in_dtype: Dtype,
    f_f: impl Fn(f32, f32) -> bool,
    f_i: impl Fn(i64, i64) -> bool,
) -> Value {
    if in_dtype == Dtype::F32 {
        let (av, bv) = (lanes_f32(a), lanes_f32(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_f(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    } else {
        let (av, bv) = (lanes_i64(a), lanes_i64(b));
        let n = av.len().max(bv.len());
        Value::Bool(
            (0..n)
                .map(|i| f_i(av[pick(av.len(), i)], bv[pick(bv.len(), i)]))
                .collect(),
        )
    }
}

pub(super) fn map_f32(v: &Value, f: impl Fn(f32) -> f32) -> Value {
    Value::F32(lanes_f32(v).into_iter().map(f).collect())
}

pub(super) fn canonical_reduce<T: Copy>(
    row: &[T],
    identity: T,
    combine: impl Fn(T, T) -> T + Copy,
) -> T {
    if row.is_empty() {
        return identity;
    }
    let mut level = row.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(32));
        for chunk in level.chunks(32) {
            let mut lanes = [identity; 32];
            lanes[..chunk.len()].copy_from_slice(chunk);
            for offset in [16usize, 8, 4, 2, 1] {
                for lane in 0..offset {
                    lanes[lane] = combine(lanes[lane], lanes[lane + offset]);
                }
            }
            next.push(lanes[0]);
        }
        level = next;
    }
    level[0]
}

#[derive(Clone, Copy)]
pub(super) struct ArgmaxCandidate {
    value: f32,
    index: u32,
    have: bool,
}

pub(super) fn combine_argmax(left: ArgmaxCandidate, right: ArgmaxCandidate) -> ArgmaxCandidate {
    match (left.have, right.have) {
        (false, false) => left,
        (true, false) => left,
        (false, true) => right,
        (true, true) => {
            if right.value > left.value || (right.value == left.value && right.index < left.index) {
                right
            } else {
                left
            }
        }
    }
}

pub(super) fn scan_rows<T: Copy>(
    lanes: &[T],
    rows: usize,
    identity: T,
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let len = lanes.len().checked_div(rows).unwrap_or(0);
    let mut out = Vec::with_capacity(lanes.len());
    for row in 0..rows {
        let mut acc = identity;
        for &lane in &lanes[row * len..(row + 1) * len] {
            acc = combine(acc, lane);
            out.push(acc);
        }
    }
    out
}

pub(super) fn argmax_row(row: &[f32]) -> i32 {
    let candidates: Vec<_> = row
        .iter()
        .enumerate()
        .map(|(index, &value)| ArgmaxCandidate {
            value,
            index: index as u32,
            have: !value.is_nan(),
        })
        .collect();
    canonical_reduce(
        &candidates,
        ArgmaxCandidate {
            value: f32::NEG_INFINITY,
            index: 0,
            have: false,
        },
        combine_argmax,
    )
    .index as i32
}

pub(super) fn argmax_ordered<T: Ord>(row: &[T]) -> i32 {
    let Some((mut best_index, mut best)) = row.first().map(|value| (0usize, value)) else {
        return 0;
    };
    for (index, value) in row.iter().enumerate().skip(1) {
        if value > best {
            best = value;
            best_index = index;
        }
    }
    best_index as i32
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Extremum {
    Max,
    Min,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NanPair {
    Identity,
    Left,
}

pub(super) fn extremum(left: f32, right: f32, end: Extremum, pair: NanPair) -> f32 {
    match (left.is_nan(), right.is_nan()) {
        (true, true) => match pair {
            NanPair::Identity => match end {
                Extremum::Max => f32::NEG_INFINITY,
                Extremum::Min => f32::INFINITY,
            },
            NanPair::Left => left,
        },
        (true, false) => right,
        (false, true) => left,
        (false, false) if left == 0.0 && right == 0.0 => {
            let negative = match end {
                Extremum::Max => left.is_sign_negative() && right.is_sign_negative(),
                Extremum::Min => left.is_sign_negative() || right.is_sign_negative(),
            };
            if negative { -0.0 } else { 0.0 }
        }
        (false, false) => match end {
            Extremum::Max => left.max(right),
            Extremum::Min => left.min(right),
        },
    }
}

pub(super) fn canonical_max(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Max, NanPair::Identity)
}

pub(super) fn canonical_min(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Min, NanPair::Identity)
}

pub(super) fn element_max(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Max, NanPair::Left)
}

pub(super) fn element_min(left: f32, right: f32) -> f32 {
    extremum(left, right, Extremum::Min, NanPair::Left)
}

pub(super) fn sort_desc_order(row: &[f32]) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (row[a as usize], row[b as usize]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => a.cmp(&b),
            (true, false) => core::cmp::Ordering::Greater,
            (false, true) => core::cmp::Ordering::Less,
            (false, false) => y.partial_cmp(&x).unwrap().then(a.cmp(&b)),
        }
    });
    idx
}

pub(super) fn rows_of(shape: Shape) -> usize {
    usize::try_from(shape.rows()).unwrap_or(usize::MAX)
}

#[derive(Clone, Copy)]
pub(super) enum ReduceKind {
    Sum,
    Max,
    Min,
}

pub(super) fn reduce_rows(kind: ReduceKind, ty: ValueType, data: &Value) -> Value {
    let rows = rows_of(ty.shape);
    let len = data.len().checked_div(rows).unwrap_or(0);
    if ty.dtype == Dtype::F32 {
        let x = lanes_f32(data);
        let f: fn(&[f32]) -> f32 = match kind {
            ReduceKind::Sum => |row| canonical_reduce(row, 0.0, |a, b| a + b),
            ReduceKind::Max => |row| canonical_reduce(row, f32::NEG_INFINITY, canonical_max),
            ReduceKind::Min => |row| canonical_reduce(row, f32::INFINITY, canonical_min),
        };
        Value::F32((0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect())
    } else {
        let x = lanes_i64(data);
        let f: fn(&[i64]) -> i64 = match kind {
            ReduceKind::Sum => |row| canonical_reduce(row, 0, i64::wrapping_add),
            ReduceKind::Max => |row| canonical_reduce(row, i64::MIN, i64::max),
            ReduceKind::Min => |row| canonical_reduce(row, i64::MAX, i64::min),
        };
        from_i64(
            ty.dtype,
            (0..rows).map(|r| f(&x[r * len..(r + 1) * len])).collect(),
        )
    }
}
