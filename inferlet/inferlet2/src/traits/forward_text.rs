use crate::Queue;
use crate::bindings::pie::inferlet::forward_text;
use crate::traits::output_text::Distribution;
use wstd::io::AsyncPollable;

pub trait ForwardText {
    async fn forward_text(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        output_indices: &[u32],
    ) -> Vec<Distribution>;
}

impl ForwardText for Queue {
    async fn forward_text(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        output_indices: &[u32],
    ) -> Vec<Distribution> {
        let result_future = forward_text::forward_text(
            &self.inner,
            last_kv_page_len,
            kv_page_ids,
            tokens,
            positions,
            output_indices,
        );

        // Get the pollable handle associated with the future.
        let pollable = result_future.pollable();

        // Asynchronously wait for the result to become available.
        AsyncPollable::new(pollable).wait_for().await;

        // Once ready, get the result from the future. The result of a WIT future
        // is typically wrapped in Option<T>.
        let distributions: Vec<(Vec<u32>, Vec<f32>)> = result_future
            .get()
            .expect("WASI pollable did not yield a result.");

        // Map the raw result into the strongly-typed `Distribution` struct.
        distributions
            .into_iter()
            .map(|(ids, probs)| Distribution { ids, probs })
            .collect()
    }
}
