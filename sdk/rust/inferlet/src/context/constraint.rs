//! Sampling constraints for grammar-guided generation.

use crate::model::Model;
use crate::Result;

use super::Matcher;

/// Trait for token sampling constraints (e.g., grammar-based token masking).
pub trait Constrain {
    /// Returns the logit mask as BRLE-encoded data.
    fn mask(&self) -> Vec<u32>;

    /// Called with the accepted tokens to update constraint state.
    fn accept(&mut self, tokens: &[u32]);

    /// Resets the constraint to its initial state.
    fn reset(&mut self);

    /// Rolls back the last `num_tokens` accepted tokens.
    fn rollback(&mut self, num_tokens: usize);
}

/// Constraint adapter that wraps a WIT [`Matcher`] to implement [`Constrain`].
///
/// Used internally by [`super::TokenStream::collect_json`] and available for
/// standalone use when you need grammar-constrained generation.
pub struct GrammarConstraint {
    matcher: Matcher,
}

impl GrammarConstraint {
    /// Create a grammar constraint from a [`Matcher`].
    pub fn new(matcher: Matcher) -> Self {
        Self { matcher }
    }

    /// Create a grammar constraint from a JSON schema string.
    pub fn from_json_schema(
        schema: &str,
        model: &Model,
    ) -> Result<Self> {
        let grammar = crate::inference::Grammar::from_json_schema(schema)?;
        let tokenizer = model.tokenizer();
        let matcher = Matcher::new(&grammar, &tokenizer);
        Ok(Self { matcher })
    }
}

impl Constrain for GrammarConstraint {
    fn mask(&self) -> Vec<u32> {
        self.matcher.next_token_logit_mask()
    }

    fn accept(&mut self, tokens: &[u32]) {
        let _ = self.matcher.accept_tokens(tokens);
    }

    fn reset(&mut self) {
        self.matcher.reset();
    }

    fn rollback(&mut self, _num_tokens: usize) {
        // Matcher doesn't support rollback — reset instead.
        self.matcher.reset();
    }
}
