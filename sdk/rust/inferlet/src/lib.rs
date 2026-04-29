//! Inferlet SDK for Pie
//!
//! This crate provides the core types and traits for building inferlets
//! that run on the Pie inference engine.

/// Result type for inferlet operations (compatible with WIT bindings).
pub type Result<T> = std::result::Result<T, String>;

// Re-export wstd for the macro to use
pub use wstd;

// Re-export wit_bindgen so the macro-generated inline WIT can reference it
pub use wit_bindgen;

// Re-export serde and serde_json so the macro-generated JSON bridge can use them
pub use serde;
pub use serde_json;
pub use schemars;

// Re-export the main attribute macro
pub use inferlet_macros::main;

// Generate WIT bindings directly in lib.rs
wit_bindgen::generate!({
    path: "wit",
    world: "inferlet",
    pub_export_macro: true,
    with: {
         "wasi:io/poll@0.2.4": ::wasi::io::poll,
    },
    generate_all,
});

// Re-export types that don't need async wrappers directly
pub use pie::core::types;
pub use pie::mcp;
pub use pie::zo;

// =============================================================================
// Context (new wrapper module)
// =============================================================================

mod context;

pub use context::{
    Context, RawContext,
    TokenStream, EventStream,
    Speculate, Speculation, Constrain,
    GrammarConstraint, Schema,
};

// =============================================================================
// Adapter
// =============================================================================

pub mod adapter {
    pub use crate::pie::core::adapter::Adapter;
}

// =============================================================================
// Model
// =============================================================================

pub mod model {
    pub use crate::pie::core::model::{Model, Tokenizer};
}

// =============================================================================
// Other re-exports
// =============================================================================

pub mod runtime {
    pub use crate::pie::core::runtime::*;
}

pub mod scheduling {
    pub use crate::pie::core::scheduling::*;
}

pub mod messaging {
    pub use crate::pie::core::messaging::*;
}

pub mod inference {
    pub use crate::pie::core::inference::*;
}

impl crate::pie::core::inference::Sampler {
    /// Argmax sampling — deterministic, picks the highest-probability token.
    ///
    /// Recommended default for grammar-constrained generation: most masked
    /// positions have only a handful of valid tokens and stochastic sampling
    /// rarely improves quality.
    pub const ARGMAX: Self = Self::TopP((0.0, 1.0));

    /// Greedy / argmax sampling. Alias for [`ARGMAX`](Self::ARGMAX).
    pub const fn greedy() -> Self { Self::ARGMAX }

    /// Top-p (nucleus) sampling.
    ///
    /// `temperature = 0.0` collapses to argmax. `p = 1.0` allows the full
    /// distribution. Common defaults: `top_p(0.6, 0.95)`.
    pub const fn top_p(temperature: f32, p: f32) -> Self {
        Self::TopP((temperature, p))
    }

    /// Top-k sampling: sample from the top `k` tokens by probability.
    pub const fn top_k(temperature: f32, k: u32) -> Self {
        Self::TopK((temperature, k))
    }

    /// Min-p sampling: keep tokens with probability ≥ `p × max_prob`.
    pub const fn min_p(temperature: f32, p: f32) -> Self {
        Self::MinP((temperature, p))
    }

    /// Combined top-k + top-p: first restrict to top `k`, then apply nucleus `p`.
    pub const fn top_k_top_p(temperature: f32, k: u32, p: f32) -> Self {
        Self::TopKTopP((temperature, k, p))
    }

    /// Plain multinomial: sample from the full distribution after temperature
    /// scaling. The `u32` is a draws-per-sample multiplier (typically 1).
    pub const fn multinomial(temperature: f32, draws: u32) -> Self {
        Self::Multinomial((temperature, draws))
    }

    /// Distribution output: returns the top-`k` token IDs with their
    /// probabilities instead of a sampled token. Useful for tree search,
    /// best-of-n, or external samplers.
    pub const fn distribution(temperature: f32, k: u32) -> Self {
        Self::Dist((temperature, k))
    }

    /// Raw logits output: returns the model's pre-softmax, untemperatured
    /// logits as a packed little-endian f32 byte buffer (length =
    /// `vocab_size * 4`) per requested position. Decode via
    /// `bytemuck::cast_slice::<u8, f32>(&buf)` or equivalent. Useful for
    /// custom sampling or beam search where the full distribution matters.
    pub const fn raw_logits() -> Self {
        Self::RawLogits
    }

    /// Per-position log p(token | context). Returned as `Output::Logprobs`
    /// with a length-1 inner list per slot. Computed via log_softmax with
    /// no temperature scaling — the value reflects the model's native
    /// distribution. Use for perplexity / cross-entropy / scoring.
    pub const fn logprob(token_id: u32) -> Self {
        Self::Logprob(token_id)
    }

    /// Per-position log p(t | context) for each `t` in `token_ids`. Returned
    /// as `Output::Logprobs` with a length-K inner list per slot, in the
    /// same order. Use for multi-candidate scoring (yes/no, multiple choice,
    /// reranking).
    pub fn logprobs(token_ids: Vec<u32>) -> Self {
        Self::Logprobs(token_ids)
    }

    /// Shannon entropy `H(p) = -sum(p log p)` of the unscaled distribution
    /// at this position. Returned as `Output::Entropies`. Useful for
    /// uncertainty / confidence-based decisions.
    pub const fn entropy() -> Self {
        Self::Entropy
    }
}

impl Default for crate::pie::core::inference::Sampler {
    /// Argmax. The most predictable default — switch to `top_p` for creative
    /// tasks.
    fn default() -> Self { Self::ARGMAX }
}

// =============================================================================
// Output helpers
//
// `Output` is now a `record { slots, spec-tokens, spec-positions }` and each
// slot is a typed `SlotOutput` variant. Inferlets that attach exactly one
// sampler want a single-line accessor; these helpers cover the common cases
// without forcing a `match` on every call site.
// =============================================================================

impl crate::pie::core::inference::Output {
    /// First slot's sampled token, if it's a Token slot.
    pub fn first_token(&self) -> Option<u32> {
        self.slots.first().and_then(|s| match s {
            crate::pie::core::inference::SlotOutput::Token(t) => Some(*t),
            _ => None,
        })
    }

    /// Iterator over every Token slot, in slot order. In spec-decode mode,
    /// these are the verifier-accepted tokens.
    pub fn tokens(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().filter_map(|s| match s {
            crate::pie::core::inference::SlotOutput::Token(t) => Some(*t),
            _ => None,
        })
    }

    /// First slot's distribution `(token_ids, probs)`, if it's a Distribution slot.
    pub fn first_distribution(&self) -> Option<(&[u32], &[f32])> {
        self.slots.first().and_then(|s| match s {
            crate::pie::core::inference::SlotOutput::Distribution((ids, ps)) => {
                Some((ids.as_slice(), ps.as_slice()))
            }
            _ => None,
        })
    }

    /// First slot's raw logits buffer (native-endian f32 bytes), if any.
    pub fn first_logits(&self) -> Option<&[u8]> {
        self.slots.first().and_then(|s| match s {
            crate::pie::core::inference::SlotOutput::Logits(b) => Some(b.as_slice()),
            _ => None,
        })
    }

    /// First slot's logprob list. Length 1 for `Sampler::logprob`, length K
    /// for `Sampler::logprobs`.
    pub fn first_logprobs(&self) -> Option<&[f32]> {
        self.slots.first().and_then(|s| match s {
            crate::pie::core::inference::SlotOutput::Logprobs(v) => Some(v.as_slice()),
            _ => None,
        })
    }

    /// First slot's entropy.
    pub fn first_entropy(&self) -> Option<f32> {
        self.slots.first().and_then(|s| match s {
            crate::pie::core::inference::SlotOutput::Entropy(h) => Some(*h),
            _ => None,
        })
    }
}

pub mod instruct {
    pub mod chat {
        pub use crate::pie::instruct::chat::*;
    }
    pub mod tool_use {
        pub use crate::pie::instruct::tool_use::*;
    }
    pub mod reasoning {
        pub use crate::pie::instruct::reasoning::*;
    }
}

// =============================================================================
// Async Extension Traits
// =============================================================================

use wstd::io::AsyncPollable;

/// Extension trait for async forward pass operations.
pub trait ForwardPassExt {
    /// Executes the forward pass and waits for the result asynchronously.
    fn execute_async(&self) -> impl std::future::Future<Output = Result<inference::Output>>;
}

impl ForwardPassExt for inference::ForwardPass {
    async fn execute_async(&self) -> Result<inference::Output> {
        let future_output = self.execute()?;
        let pollable = future_output.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        future_output.get().ok_or_else(|| "No output available".to_string())
    }
}

/// Extension trait for async messaging subscription operations.
pub trait SubscriptionExt {
    /// Gets the next message from a subscription asynchronously.
    fn get_async(&self) -> impl std::future::Future<Output = Option<String>>;
}

impl SubscriptionExt for messaging::Subscription {
    async fn get_async(&self) -> Option<String> {
        let pollable = self.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        self.get()
    }
}

/// Extension trait for FutureString (used by receive and spawn).
pub trait FutureStringExt {
    /// Waits for the result asynchronously.
    fn wait_async(&self) -> impl std::future::Future<Output = Option<String>>;
}

impl FutureStringExt for types::FutureString {
    async fn wait_async(&self) -> Option<String> {
        let pollable = self.pollable();
        AsyncPollable::new(pollable).wait_for().await;
        self.get()
    }
}

// =============================================================================
// Decoder (Unified) — re-exported from context module
// =============================================================================

pub use context::{
    Decoder, Event,
    // Re-exported WIT decoder / event types
    ChatDecoder, ChatEvent,
    ToolDecoder, ToolEvent,
    ReasoningDecoder, ReasoningEvent,
    Matcher,
};

// =============================================================================
// Model Extension Trait
// =============================================================================

/// Extension trait that adds convenience methods to [`Model`].
pub trait ModelExt {
    /// Create a [`Decoder`] for this model.
    ///
    /// ```ignore
    /// let mut decoder = model.decoder()
    ///     .with_reasoning()
    ///     .with_tool_use();
    /// ```
    fn decoder(&self) -> Decoder;
}

impl ModelExt for model::Model {
    fn decoder(&self) -> Decoder {
        Decoder::new(self)
    }
}

// =============================================================================
// Argument Parsing (re-exported from pico_args)
// =============================================================================

/// Re-export of `pico_args::Arguments` for ergonomic CLI argument parsing.
pub use pico_args::Arguments;

/// Parses a `Vec<String>` (as received from the WIT entry point) into
/// a `pico_args::Arguments` for flag/option extraction.
pub fn parse_args(args: Vec<String>) -> Arguments {
    Arguments::from_vec(args.into_iter().map(std::ffi::OsString::from).collect())
}

/// Prelude module for convenient imports.
///
/// `use inferlet::prelude::*;` covers the common case so inferlets don't
/// have to maintain a hand-rolled import grocery list.
pub mod prelude {
    pub use crate::main;
    pub use crate::{Context, Event, Result, Schema};
    pub use crate::inference::{ForwardPass, Output, Sampler};
    pub use crate::model::Model;
    pub use crate::runtime;
    pub use crate::messaging;
    pub use crate::adapter::Adapter;

    // Extension traits
    pub use crate::ModelExt;
    pub use crate::ForwardPassExt;
    pub use crate::SubscriptionExt;
    pub use crate::FutureStringExt;
}