use crate::l4m;
use tokio::task;

// Async wrapper for l4m::tokenize

pub async fn tokenize(text: String) -> Vec<u32> {
    task::spawn_local(async move { l4m::tokenize(&text) })
        .await
        .unwrap()
}

pub async fn detokenize(token_ids: Vec<u32>) -> String {
    task::spawn_local(async move { l4m::detokenize(&token_ids) })
        .await
        .unwrap()
}

pub async fn get_vocabs() -> Vec<Vec<u8>> {
    task::spawn_local(async move { l4m::get_vocabs() })
        .await
        .unwrap()
}

// self.stream, slice::from_ref(&next_dist), 32);
pub async fn sample_top_k(stream_id: u32, dists: Vec<u32>, k: u32) -> Vec<(Vec<u32>, Vec<f32>)> {
    task::spawn_local(async move { l4m::sample_top_k(stream_id, &dists, k) })
        .await
        .unwrap()
}
