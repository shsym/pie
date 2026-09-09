use alloc::vec::Vec;

use super::op::Op;
use crate::types::{Literal, Predicate, RngKind, Shape, ValueId};

pub fn next_id(ops: &[Op]) -> ValueId {
    ops.iter().map(|o| o.result_count()).sum()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepShape {
    Row,
    Reduced,
    Scalar,
    RowMask,
    ReducedIndex,
}

pub trait Sink {
    fn push(&mut self, op: Op, shape: StepShape) -> ValueId;
}

impl Sink for Vec<Op> {
    fn push(&mut self, op: Op, _shape: StepShape) -> ValueId {
        let id = next_id(self);
        Vec::push(self, op);
        id
    }
}

fn push(sink: &mut impl Sink, op: Op, shape: StepShape) -> ValueId {
    sink.push(op, shape)
}

pub fn gumbel(sink: &mut impl Sink, state: ValueId, shape: Shape) -> ValueId {
    push(
        sink,
        Op::RngKeyed {
            state,
            shape,
            kind: RngKind::Gumbel,
        },
        StepShape::Row,
    )
}

pub fn mask_apply(sink: &mut impl Sink, logits: ValueId, mask: ValueId) -> ValueId {
    let ninf = push(
        sink,
        Op::Const(Literal::F32(f32::NEG_INFINITY)),
        StepShape::Scalar,
    );
    push(
        sink,
        Op::Select {
            cond: mask,
            a: logits,
            b: ninf,
        },
        StepShape::Row,
    )
}

pub fn softmax(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = push(sink, Op::Broadcast { value: m, shape }, StepShape::Row);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let sb = push(sink, Op::Broadcast { value: s, shape }, StepShape::Row);
    push(sink, Op::Div(e, sb), StepShape::Row)
}

pub fn log_softmax(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let m = push(sink, Op::ReduceMax(x), StepShape::Reduced);
    let mb = push(sink, Op::Broadcast { value: m, shape }, StepShape::Row);
    let c = push(sink, Op::Sub(x, mb), StepShape::Row);
    let e = push(sink, Op::Exp(c), StepShape::Row);
    let s = push(sink, Op::ReduceSum(e), StepShape::Reduced);
    let l = push(sink, Op::Log(s), StepShape::Reduced);
    let lb = push(sink, Op::Broadcast { value: l, shape }, StepShape::Row);
    push(sink, Op::Sub(c, lb), StepShape::Row)
}

pub fn l2norm(sink: &mut impl Sink, x: ValueId, shape: Shape) -> ValueId {
    let sq = push(sink, Op::Mul(x, x), StepShape::Row);
    let s = push(sink, Op::ReduceSum(sq), StepShape::Reduced);
    let lg = push(sink, Op::Log(s), StepShape::Reduced);
    let half = push(sink, Op::Const(Literal::F32(0.5)), StepShape::Scalar);
    let h = push(sink, Op::Mul(lg, half), StepShape::Reduced);
    let rt = push(sink, Op::Exp(h), StepShape::Reduced);
    let rb = push(sink, Op::Broadcast { value: rt, shape }, StepShape::Row);
    push(sink, Op::Div(x, rb), StepShape::Row)
}

pub fn nucleus_sample(
    sink: &mut impl Sink,
    logits: ValueId,
    top_p: ValueId,
    state: ValueId,
    shape: Shape,
) -> ValueId {
    let probabilities = softmax(sink, logits, shape);
    let keep = push(
        sink,
        Op::PivotThreshold {
            input: probabilities,
            predicate: Predicate::CummassLe(top_p),
        },
        StepShape::RowMask,
    );
    let masked = mask_apply(sink, logits, keep);
    let noise = gumbel(sink, state, shape);
    let perturbed = push(sink, Op::Add(masked, noise), StepShape::Row);
    push(sink, Op::ReduceArgmax(perturbed), StepShape::ReducedIndex)
}
