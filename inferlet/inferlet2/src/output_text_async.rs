use crate::wstd::runtime::AsyncPollable;
use crate::{core, output_text};

pub async fn get_next_token_distribution(
    queue: &core::Queue,
    emb_ids: Vec<u32>,
) -> Vec<(Vec<u32>, Vec<f32>)> {
    let res = output_text::get_next_token_distribution(queue, &emb_ids);
    let a = res.pollable();
    AsyncPollable::new(a).wait_for().await;
    res.get().unwrap() // New result is Option<Option<...>>
}
