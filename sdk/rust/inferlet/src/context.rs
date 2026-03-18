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

    // ── Market operations ────────────────────────────────────────────

    /// Suspend this context (release pages, stop rent).
    /// Restoration is system-driven based on bid priority.
    pub fn suspend(&self) -> Result<()> {
        self.inner.suspend()
    }

    /// Set bid (willingness to pay per page per step).
    /// Higher bid = harder to evict, restored first.
    pub fn bid(&self, value: f64) {
        self.inner.bid(value);
    }

    /// Yield priority for the duration of an external wait.
    ///
    /// Under critical-value (Vickrey) payments, the process pays the
    /// clearing price regardless of its bid. Bidding above clearing price
    /// costs the same — only eviction risk changes. The rational strategy:
    ///
    /// - **Hold** (keep generation bid): costs `rent × pages` per step,
    ///   but ensures instant resumption.
    /// - **Fold** (bid zero): costs nothing, but risks restore-queue delay.
    ///
    /// The process can afford to hold for `W` idle steps iff the surplus
    /// covers the idle rent. From the bid formula:
    ///
    /// ```text
    /// W / remaining  <  (bid − rent) / rent
    /// ```
    ///
    /// Processes well above the clearing price absorb long waits (large
    /// surplus). Marginal processes fold immediately (no surplus). The
    /// threshold is endogenous — no tuning parameter required.
    ///
    /// Returns a [`BidGuard`] that restores the generation bid on drop.
    ///
    /// ```ignore
    /// let _guard = ctx.yield_bid(Duration::from_millis(200));
    /// let result = tool_call().await;
    /// // guard dropped → bid restored
    /// ```
    pub fn yield_bid(&self, expected_wait: core::time::Duration) -> BidGuard<'_> {
        let balance = crate::scheduling::balance(&self.model);
        let rent = crate::scheduling::rent(&self.inner);
        let dividend = crate::scheduling::dividend(&self.model);
        let latency = crate::scheduling::latency(&self.inner);
        let pages = (self.committed_pages + self.working_pages) as f64;

        // Truthful generation bid (§5.1).
        let remaining = 4096.0; // conservative default; overridden by horizon if known
        let generation_bid = if pages > 0.0 {
            (balance / remaining - dividend) / pages
        } else { 0.0 };

        let step_secs = latency.max(0.001);
        let wait_steps = expected_wait.as_secs_f64() / step_secs;

        // Hold if surplus covers the idle rent (§7):
        //   W / remaining  <  (bid − rent) / rent
        // Equivalently: rent-free or bid well above clearing price.
        let hold = rent == 0.0
            || (generation_bid > rent
                && wait_steps < remaining * (generation_bid - rent) / rent);

        if !hold {
            self.inner.bid(0.0);
        }

        BidGuard { ctx: self, saved_bid: generation_bid }
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

        // Reserve additional pages if we need more than currently allocated.
        let total_tokens_after = self.working_tokens + num_tokens;
        let pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
        let additional = pages_needed.saturating_sub(self.working_pages);
        if additional > 0 {
            self.inner
                .reserve_working_pages(additional)
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

// =============================================================================
// BidGuard — RAII bid restoration
// =============================================================================

/// Restores the generation bid when dropped.
///
/// Created by [`Context::yield_bid()`]. During the wait, the bid is
/// either held (surplus covers idle rent) or set to zero (fold).
/// On drop, the truthful generation bid is restored.
pub struct BidGuard<'a> {
    ctx: &'a Context,
    saved_bid: f64,
}

impl<'a> Drop for BidGuard<'a> {
    fn drop(&mut self) {
        self.ctx.bid(self.saved_bid);
    }
}

