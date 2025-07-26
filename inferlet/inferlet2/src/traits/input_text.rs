use crate::Queue;
use crate::input_text;

/// Provides an interface for converting tokenized text into dense vector embeddings.
///
/// This operation is a fundamental step in natural language processing, transforming
/// discrete tokens into a continuous vector space representation that a model can process.
pub trait InputText {
    /// Converts a sequence of input tokens into their corresponding embedding vectors.
    ///
    /// This function takes token IDs and their positions, looks up the appropriate embeddings,
    /// and writes the resulting vectors into the specified pre-allocated output buffers.
    ///
    /// # Parameters
    /// * `embed_ids`: A slice of pre-allocated object IDs where the resulting embedding
    ///   vectors will be stored.
    /// * `tokens`: A slice of token IDs to be embedded.
    /// * `positions`: A slice of positional indices corresponding to each token. The length
    ///   of this slice must match the length of the `tokens` slice.
    fn embed_text(&self, embed_ids: &[u32], tokens: &[u32], positions: &[u32]);
}

impl InputText for Queue {
    fn embed_text(&self, embed_ids: &[u32], tokens: &[u32], positions: &[u32]) {
        input_text::embed_text(&self.inner, embed_ids, tokens, positions)
    }
}