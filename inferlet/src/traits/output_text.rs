use crate::Queue;
use crate::{output_text, wstd::runtime::AsyncPollable};

/// Represents a probability distribution over a set of tokens.
#[derive(Clone, Debug)]
pub struct Distribution {
    /// A vector of token IDs.
    pub ids: Vec<u32>,
    /// A vector of probabilities corresponding to the token IDs.
    pub probs: Vec<f32>,
}

/// Provides a way to generate text by predicting the next token.
pub trait OutputText {
    /// Asynchronously computes the probability distribution for the next token
    /// based on a sequence of embeddings.
    ///
    /// # Arguments
    ///
    /// * `embed_ids` - A slice of embedding IDs to base the prediction on.
    ///
    /// # Returns
    ///
    /// A `Vec<Distribution>` representing the probability distribution for the next token.
    async fn get_next_token_distribution(&self, embed_ids: &[u32]) -> Vec<Distribution>;
}

impl OutputText for Queue {
    async fn get_next_token_distribution(&self, embed_ids: &[u32]) -> Vec<Distribution> {
        // Call the underlying synchronous WIT binding, which returns a future.
        let result_future = output_text::get_next_token_distribution(&self.inner, embed_ids);

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
