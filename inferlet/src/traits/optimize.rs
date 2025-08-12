use crate::Queue;
use crate::bindings::pie::inferlet::optimize;
use crate::traits::output_text::Distribution;
use wstd::io::AsyncPollable;

pub trait Optimize {
    fn create_adapter(&self, name: &str);

    fn destroy_adapter(&self, name: &str);

    fn update_adapter(&self, name: &str, scores: &[f32], seeds: &[i64]);

    async fn forward_with_mutation(
        &self,
        adapter: &str,
        seed: i64,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Option<Vec<Distribution>>;
}

impl Optimize for Queue {
    fn create_adapter(&self, name: &str) {
        optimize::create_adapter(name);
    }

    fn destroy_adapter(&self, name: &str) {
        optimize::destroy_adapter(name);
    }

    fn update_adapter(&self, name: &str, scores: &[f32], seeds: &[i64]) {
        optimize::update_adapter(name, scores, seeds);
    }

    async fn forward_with_mutation(
        &self,
        adapter: &str,
        seed: i64,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Option<Vec<Distribution>> {
        let result_future = optimize::forward_with_mutation(
            &self.inner,
            adapter,
            seed,
            last_kv_page_len,
            kv_page_ids,
            tokens,
            positions,
            mask,
            output_indices,
        );

        if let Some(result_future) = result_future {
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
            Some(
                distributions
                    .into_iter()
                    .map(|(ids, probs)| Distribution { ids, probs })
                    .collect(),
            )
        } else {
            None
        }
    }
}
