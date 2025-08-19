use crate::Queue;
use crate::forward_text;
use crate::brle::Brle;
use crate::traits::output_text::Distribution;
use wstd::io::AsyncPollable;

pub fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Brle> {
    let mut mask = Vec::new();
    let offset = num_total_tokens - num_input_tokens;
    for i in 0..num_input_tokens {
        mask.push(Brle::new((offset + i + 1) as usize));
    }
    mask
}

pub trait ForwardText {
    async fn forward_text(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Vec<Distribution>;

    fn forward_text_no_output(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
    );
}

impl ForwardText for Queue {
    async fn forward_text(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Vec<Distribution> {
        let result_future = forward_text::forward_text(
            &self.inner,
            last_kv_page_len,
            kv_page_ids,
            tokens,
            positions,
            mask,
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

    fn forward_text_no_output(
        &self,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
    ) {
        forward_text::forward_text_no_output(
            &self.inner,
            last_kv_page_len,
            kv_page_ids,
            tokens,
            positions,
            mask,
        );
    }
}
