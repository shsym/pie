//! SDK Context — stateful wrapper over the WIT `Context` resource.
//!
//! Owns the native context handle, caches page metadata, buffers instruct
//! tokens, and exposes ergonomic fill / flush / generate methods.

mod decoder;
mod constraint;
mod speculation;
mod stream;

// Re-export submodule public types.
pub use decoder::*;
pub use constraint::*;
pub use speculation::*;
pub use stream::*;

use crate::inference::{ForwardPass, Sampler};
use crate::model::Model;
use crate::ForwardPassExt;
use crate::Result;

/// The raw WIT context resource, re-exported for power users.
pub use crate::pie::core::context::Context as RawContext;

// Instruct WIT bindings.
use crate::pie::instruct::chat;
use crate::pie::instruct::tool_use;

// =============================================================================
// Context
// =============================================================================

/// High-level inference context.
///
/// Wraps the native WIT [`RawContext`] resource and provides:
/// - **Buffered instruct fills** (`system`, `user`, `cue`, …) that accumulate
///   tokens locally.
/// - **`flush()`** to drain the buffer through a forward pass and commit pages.
/// - **`generate()`** to create a [`TokenStream`] for token-by-token generation.
/// - **Cached page metadata** (`seq_len`, `committed_pages`, `working_tokens`)
///   to avoid redundant WIT host calls.
pub struct Context {
    pub(crate) inner: RawContext,
    pub(crate) model: Model,
    pub(crate) page_size: u32,
    /// SDK-side token buffer filled by instruct operations.
    pub(crate) buffer: Vec<u32>,
    /// Total tokens in committed + working pages (tracked locally).
    pub(crate) seq_len: u32,
    /// Number of committed pages (tracked locally).
    pub(crate) committed_pages: u32,
    /// Number of currently reserved working pages (tracked locally).
    pub(crate) working_pages: u32,
    /// Number of tokens in working (uncommitted) pages (tracked locally).
    pub(crate) working_tokens: u32,
}

impl Context {
    // ── Constructors ────────────────────────────────────────────────

    /// Create a fresh empty context for the given model.
    pub fn new(model: &Model) -> Result<Self> {
        let inner = RawContext::create(model)?;
        Ok(Self::wrap(inner))
    }

    /// Open a saved snapshot (implicit fork — snapshot stays immutable).
    pub fn open(model: &Model, name: &str) -> Result<Self> {
        let inner = RawContext::open(model, name)?;
        Ok(Self::wrap(inner))
    }

    /// Take ownership of a saved snapshot (snapshot is deleted).
    pub fn take(model: &Model, name: &str) -> Result<Self> {
        let inner = RawContext::take(model, name)?;
        Ok(Self::wrap(inner))
    }

    /// Delete a saved snapshot by name (static — no context needed).
    pub fn delete(model: &Model, name: &str) -> Result<()> {
        RawContext::delete(model, name)
    }

    /// Wrap an existing raw context, syncing cached state from the host.
    fn wrap(inner: RawContext) -> Self {
        let page_size = inner.tokens_per_page();
        let committed_pages = inner.committed_page_count();
        let working_pages = inner.working_page_count();
        let working_tokens = inner.working_page_token_count();
        let seq_len = committed_pages * page_size + working_tokens;
        let model = inner.model();
        Self {
            inner,
            model,
            page_size,
            buffer: Vec::new(),
            seq_len,
            committed_pages,
            working_pages,
            working_tokens,
        }
    }

    // ── Lifecycle ───────────────────────────────────────────────────

    /// Fork into a new anonymous context (working pages are copied).
    ///
    /// The forked context inherits a copy of the parent's buffered tokens.
    pub fn fork(&self) -> Result<Self> {
        let forked = self.inner.fork()?;
        let model = forked.model();
        Ok(Self {
            inner: forked,
            model,
            page_size: self.page_size,
            buffer: self.buffer.clone(),
            seq_len: self.seq_len,
            committed_pages: self.committed_pages,
            working_pages: self.working_pages,
            working_tokens: self.working_tokens,
        })
    }

    /// Save the context under a user-chosen name.
    pub fn save(&self, name: &str) -> Result<()> {
        self.inner.save(name)
    }

    /// Anonymous save — returns a runtime-generated snapshot name.
    pub fn snapshot(&self) -> Result<String> {
        self.inner.snapshot()
    }

    /// Force-destroy the context immediately, consuming it.
    pub fn destroy(self) {
        self.inner.destroy()
    }

    // ── Accessors (no WIT calls) ────────────────────────────────────

    /// The model this context was created with.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Tokens per page.
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    /// Total sequence length (committed + working tokens, excluding buffer).
    pub fn seq_len(&self) -> u32 {
        self.seq_len
    }

    /// Pending (buffered but not yet flushed) tokens.
    pub fn buffer(&self) -> &[u32] {
        &self.buffer
    }

    /// Access the underlying WIT context resource (escape hatch).
    pub fn inner(&self) -> &RawContext {
        &self.inner
    }

    // ── Instruct Fillers ────────────────────────────────────────────
    //
    // Each filler delegates to the WIT free function (which only needs the
    // model for template lookup / tokenization) and appends the resulting
    // tokens to the local buffer.

    /// Fill a system message; tokens are buffered for the next `flush()`.
    pub fn system(&mut self, message: &str) -> &mut Self {
        let tokens = chat::system(&self.model, message);
        self.buffer.extend(tokens);
        self
    }

    /// Fill a user message.
    pub fn user(&mut self, message: &str) -> &mut Self {
        let tokens = chat::user(&self.model, message);
        self.buffer.extend(tokens);
        self
    }

    /// Fill an assistant message (for history replay).
    pub fn assistant(&mut self, message: &str) -> &mut Self {
        let tokens = chat::assistant(&self.model, message);
        self.buffer.extend(tokens);
        self
    }

    /// Cue the model to generate (fills the generation header).
    pub fn cue(&mut self) -> &mut Self {
        let tokens = chat::cue(&self.model);
        self.buffer.extend(tokens);
        self
    }

    /// Seal the current turn (insert stop token).
    pub fn seal(&mut self) -> &mut Self {
        let tokens = chat::seal(&self.model);
        self.buffer.extend(tokens);
        self
    }

    /// Register available tools (list of JSON schema strings).
    pub fn equip_tools(&mut self, tools: &[String]) -> Result<&mut Self> {
        let tools_vec: Vec<String> = tools.to_vec();
        let tokens = tool_use::equip(&self.model, &tools_vec)?;
        self.buffer.extend(tokens);
        Ok(self)
    }

    /// Provide a tool result after a tool call.
    pub fn answer_tool(&mut self, name: &str, value: &str) -> &mut Self {
        let tokens = tool_use::answer(&self.model, name, value);
        self.buffer.extend(tokens);
        self
    }

    /// Append raw tokens to the buffer directly.
    pub fn append(&mut self, tokens: &[u32]) -> &mut Self {
        self.buffer.extend_from_slice(tokens);
        self
    }

    // ── Flush ───────────────────────────────────────────────────────

    /// Drain the buffered tokens through a forward pass and commit pages.
    ///
    /// After flush, the buffer is empty and `seq_len` reflects all
    /// consumed tokens.
    pub async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let tokens = std::mem::take(&mut self.buffer);
        let num_tokens = tokens.len() as u32;

        // Reserve pages if we need more than currently allocated.
        let total_tokens_after = self.working_tokens + num_tokens;
        let pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        if pages_needed > self.working_pages {
            self.inner
                .reserve_working_pages(pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
            self.working_pages = pages_needed;
        }

        // Build and execute a forward pass.
        let pass = ForwardPass::new(&self.model);
        pass.context(&self.inner);

        let positions: Vec<u32> = (self.seq_len..self.seq_len + num_tokens).collect();
        pass.input_tokens(&tokens, &positions);

        pass.execute_async()
            .await
            .map_err(|e| format!("Forward pass failed: {}", e))?;

        // Commit full pages.
        let new_working = self.working_tokens + num_tokens;
        let pages_to_commit = new_working / self.page_size;
        if pages_to_commit > 0 {
            self.inner
                .commit_working_pages(pages_to_commit)
                .map_err(|e| format!("Failed to commit pages: {}", e))?;
        }

        // Update cached state.
        self.committed_pages += pages_to_commit;
        self.working_pages -= pages_to_commit;
        self.working_tokens = new_working % self.page_size;
        self.seq_len += num_tokens;

        Ok(())
    }

    // ── Generate ────────────────────────────────────────────────────

    /// Creates a [`TokenStream`] for generation.
    ///
    /// Any tokens already in the buffer will be consumed on the first
    /// `step()`. The user can continue to fill tokens mid-generation
    /// (e.g., tool outputs) via [`TokenStream::ctx()`].
    pub fn generate(&mut self, sampler: Sampler) -> TokenStream<'_> {
        TokenStream::new(self, sampler)
    }
}
