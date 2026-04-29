//! Speculative decoding types.

use crate::inference::Output;

/// Trait for custom speculative decoding.
pub trait Speculate {
    /// Generates draft tokens and their positions based on current context.
    fn draft(&self) -> (Vec<u32>, Vec<u32>);

    /// Called with the accepted tokens from the model.
    fn accept(&mut self, tokens: &[u32]);

    /// Resets the speculator to its initial state.
    fn reset(&mut self);

    /// Rolls back the last `num_tokens` accepted tokens.
    fn rollback(&mut self, num_tokens: usize);
}

/// Speculation enum - either system-provided or custom.
pub enum Speculation {
    /// Default speculation that uses runtime-provided speculative tokens.
    Default {
        spec_tokens: Vec<u32>,
        spec_positions: Vec<u32>,
    },

    /// Custom speculation that implements the [Speculate] trait.
    Custom(Box<dyn Speculate>),
}

impl Default for Speculation {
    fn default() -> Self {
        Speculation::Default {
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
        }
    }
}

impl Speculation {
    /// Creates a new system speculation.
    pub fn system() -> Self {
        Self::default()
    }

    /// Creates a custom speculation from a Speculate implementation.
    pub fn custom<S: Speculate + 'static>(speculator: S) -> Self {
        Speculation::Custom(Box::new(speculator))
    }

    pub(crate) fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        match self {
            Speculation::Default {
                spec_tokens,
                spec_positions,
            } => {
                let tokens = std::mem::take(spec_tokens);
                let positions = std::mem::take(spec_positions);
                (tokens, positions)
            }
            Speculation::Custom(s) => s.draft(),
        }
    }

    pub(crate) fn accept(&mut self, output: Output) -> Vec<u32> {
        // The new Output shape always carries the next-iter spec channel as
        // separate fields; accepted tokens are whatever Token slots came back
        // (in spec mode the verifier produces a sequence of them).
        let accepted: Vec<u32> = output.tokens().collect();
        match self {
            Speculation::Default {
                spec_tokens,
                spec_positions,
            } => {
                *spec_tokens = output.spec_tokens;
                *spec_positions = output.spec_positions;
                accepted
            }
            Speculation::Custom(s) => {
                s.accept(&accepted);
                accepted
            }
        }
    }
}
