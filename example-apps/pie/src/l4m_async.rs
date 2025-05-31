use crate::l4m;
use crate::wstd::runtime::AsyncPollable;
use std::rc::Rc;

pub async fn sample_top_k(
    model: Rc<l4m::Model>,
    stream_id: u32,
    dists: Vec<u32>,
    k: u32,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let res = model.sample_top_k(stream_id, &dists, k);
    let a = res.pollable();
    AsyncPollable::new(a).wait_for().await;
    res.get().unwrap()
}
