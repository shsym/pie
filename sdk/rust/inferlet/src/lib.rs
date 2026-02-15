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
    GrammarConstraint,
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
pub mod prelude {
    pub use crate::main;
    pub use crate::Context;
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