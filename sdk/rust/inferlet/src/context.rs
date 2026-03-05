//! SDK Context — stateful wrapper over the WIT `Context` resource.
//!
//! Owns the native context handle, caches page metadata, buffers instruct
//! tokens, and exposes ergonomic fill / flush / generate methods.

use crate::inference::{ForwardPass, Output, Sampler};
use crate::model::Model;
use crate::ForwardPassExt;
use crate::Result;

/// The raw WIT context resource, re-exported for power users.
pub use crate::pie::core::context::Context as RawContext;

// Instruct WIT bindings.
use crate::pie::instruct::chat;
use crate::pie::instruct::tool_use;
use crate::pie::instruct::reasoning;

// Re-export the instruct sub-interface types for convenience.
pub use chat::{Decoder as ChatDecoder, Event as ChatEvent};
pub use tool_use::{Decoder as ToolDecoder, Event as ToolEvent};
pub use reasoning::{Decoder as ReasoningDecoder, Event as ReasoningEvent};

// Re-export the matcher type from core inference.
pub use crate::pie::core::inference::Matcher;

// =============================================================================
// Unified Decoder
// =============================================================================

/// Unified event emitted by [`Decoder`].
///
/// Merges chat, reasoning, and tool-use decoder outputs into a single enum
/// so callers only need one `match` arm per token.
pub enum Event {
    /// Raw token with no semantic significance (yet).
    Token,
    /// Generated text chunk (outside of reasoning blocks).
    Text(String),
    /// Reasoning/thinking text chunk.
    Thinking(String),
    /// Reasoning block complete (full accumulated thinking text).
    ThinkingDone(String),
    /// A complete tool call was detected: (name, arguments_json).
    ToolCall(String, String),
    /// Generation complete (full accumulated response text).
    Done(String),
}

/// Tracks whether we are inside a reasoning (thinking) block.
enum DecoderState {
    /// Normal text generation.
    Normal,
    /// Inside a reasoning block — chat deltas are suppressed,
    /// reasoning deltas are forwarded as `Event::Thinking`.
    Reasoning,
}

/// Unified decoder that internally muxes the chat, reasoning, and tool-use
/// WIT decoder resources.
///
/// Created via [`Decoder::new`], then configured with builder methods and
/// fed tokens one batch at a time.
///
/// Event priority:
/// 1. Reasoning transitions (start / complete) override chat deltas.
/// 2. Tool calls override chat deltas.
/// 3. Chat done is always forwarded.
/// 4. Otherwise, chat deltas are forwarded as `Event::Text`.
pub struct Decoder {
    chat: ChatDecoder,
    reasoning: Option<ReasoningDecoder>,
    tools: Option<ToolDecoder>,
    state: DecoderState,
}

impl Decoder {
    /// Create a decoder (chat-only by default).
    ///
    /// Reasoning and tool-use decoders are created lazily when enabled
    /// via [`with_reasoning`](Self::with_reasoning) or
    /// [`with_tool_use`](Self::with_tool_use).
    pub fn new(model: &Model) -> Self {
        Self {
            chat: chat::create_decoder(model),
            reasoning: None,
            tools: None,
            state: DecoderState::Normal,
        }
    }

    /// Enable reasoning (thinking block) decoding.
    pub fn with_reasoning(mut self, model: &Model) -> Self {
        self.reasoning = Some(reasoning::create_decoder(model));
        self
    }

    /// Enable tool-use decoding.
    pub fn with_tool_use(mut self, model: &Model) -> Self {
        self.tools = Some(tool_use::create_decoder(model));
        self
    }

    /// Feed a batch of token IDs and get back a single unified event.
    pub fn feed(&mut self, tokens: &[u32]) -> Result<Event> {
        // 1. Reasoning decoder (highest priority for state transitions)
        if let Some(ref mut reasoning) = self.reasoning {
            match reasoning.feed(tokens)? {
                ReasoningEvent::Start => {
                    self.state = DecoderState::Reasoning;
                    let _ = self.chat.feed(tokens);
                    if let Some(ref mut tools) = self.tools {
                        let _ = tools.feed(tokens);
                    }
                    return Ok(Event::Thinking(String::new()));
                }
                ReasoningEvent::Complete(s) => {
                    self.state = DecoderState::Normal;
                    let _ = self.chat.feed(tokens);
                    if let Some(ref mut tools) = self.tools {
                        let _ = tools.feed(tokens);
                    }
                    return Ok(Event::ThinkingDone(s));
                }
                ReasoningEvent::Delta(s) => {
                    if matches!(self.state, DecoderState::Reasoning) && !s.is_empty() {
                        let _ = self.chat.feed(tokens);
                        if let Some(ref mut tools) = self.tools {
                            let _ = tools.feed(tokens);
                        }
                        return Ok(Event::Thinking(s));
                    }
                }
            }
        }

        // 2. Tool decoder
        if let Some(ref mut tools) = self.tools {
            match tools.feed(tokens)? {
                ToolEvent::Call(name_args) => {
                    let _ = self.chat.feed(tokens);
                    return Ok(Event::ToolCall(name_args.0, name_args.1));
                }
                ToolEvent::Start => {}
            }
        }

        // 3. Chat decoder (lowest priority)
        match self.chat.feed(tokens)? {
            ChatEvent::Done(s) => Ok(Event::Done(s)),
            ChatEvent::Delta(s) => {
                if matches!(self.state, DecoderState::Reasoning) {
                    Ok(Event::Token)
                } else {
                    Ok(Event::Text(s))
                }
            }
            ChatEvent::Interrupt(_) => Ok(Event::Token),
        }
    }

    /// Reset all sub-decoders to their initial state.
    pub fn reset(&mut self) {
        self.chat.reset();
        if let Some(ref mut reasoning) = self.reasoning {
            reasoning.reset();
        }
        if let Some(ref mut tools) = self.tools {
            tools.reset();
        }
        self.state = DecoderState::Normal;
    }
}

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
    inner: RawContext,
    model: Model,
    page_size: u32,
    /// SDK-side token buffer filled by instruct operations.
    buffer: Vec<u32>,
    /// Total tokens in committed + working pages (tracked locally).
    seq_len: u32,
    /// Number of committed pages (tracked locally).
    committed_pages: u32,
    /// Number of currently reserved working pages (tracked locally).
    working_pages: u32,
    /// Number of tokens in working (uncommitted) pages (tracked locally).
    working_tokens: u32,
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
    /// (e.g., tool outputs) via the stream's fill methods.
    pub fn generate(&mut self, sampler: Sampler) -> TokenStream<'_> {
        TokenStream::new(self, sampler)
    }
}

// =============================================================================
// Speculation Types
// =============================================================================

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

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
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

    fn accept(&mut self, output: Output) -> Vec<u32> {
        match self {
            Speculation::Default {
                spec_tokens,
                spec_positions,
            } => match output {
                Output::TokensWithSpeculation((accepted, next_spec, next_pos)) => {
                    *spec_tokens = next_spec;
                    *spec_positions = next_pos;
                    accepted
                }
                Output::Tokens(tokens) => tokens,
                _ => vec![],
            },
            Speculation::Custom(s) => {
                let tokens = match output {
                    Output::Tokens(tokens) => tokens,
                    Output::TokensWithSpeculation((accepted, _, _)) => accepted,
                    _ => vec![],
                };
                s.accept(&tokens);
                tokens
            }
        }
    }
}

// =============================================================================
// TokenStream
// =============================================================================

/// Async stream of generated tokens.
///
/// On each `step()`, the stream drains [`Context::buffer`] as input tokens.
/// Use [`ctx()`](Self::ctx) to access the underlying context for mid-generation
/// fills (e.g., tool outputs) between `next()` calls.
pub struct TokenStream<'a> {
    ctx: &'a mut Context,
    sampler: Sampler,
    stop_tokens: Vec<u32>,
    speculation: Speculation,
    constraint: Option<Box<dyn Constrain>>,
    done: bool,
    max_tokens: Option<usize>,
    tokens_generated: usize,
    adapter: Option<&'a crate::adapter::Adapter>,
    zo_seed: Option<i64>,
}

impl<'a> TokenStream<'a> {
    /// Creates a new token stream.
    fn new(ctx: &'a mut Context, sampler: Sampler) -> Self {
        let stop_tokens = chat::stop_tokens(&ctx.model);
        Self {
            ctx,
            sampler,
            stop_tokens,
            speculation: Speculation::system(),
            constraint: None,
            done: false,
            max_tokens: None,
            tokens_generated: 0,
            adapter: None,
            zo_seed: None,
        }
    }

    /// Access the underlying context for mid-generation fills.
    ///
    /// Tokens added to the context's buffer will be consumed on the
    /// next `step()` call.
    ///
    /// ```ignore
    /// let mut stream = ctx.generate(sampler);
    /// while let Some(tokens) = stream.next().await? {
    ///     if tool_call_detected(&tokens) {
    ///         stream.ctx().answer_tool("name", "result").cue();
    ///     }
    /// }
    /// ```
    pub fn ctx(&mut self) -> &mut Context {
        self.ctx
    }

    /// Sets a custom speculation strategy for speculative decoding.
    pub fn with_speculation(mut self, speculation: Speculation) -> Self {
        self.speculation = speculation;
        self
    }

    /// Sets a sampling constraint for logit masking.
    pub fn with_constraint<C: Constrain + 'static>(mut self, constraint: C) -> Self {
        self.constraint = Some(Box::new(constraint));
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets an adapter to apply on every forward pass.
    pub fn with_adapter(mut self, adapter: &'a crate::adapter::Adapter) -> Self {
        self.adapter = Some(adapter);
        self
    }

    /// Sets a zo (Evolution Strategies) seed on every forward pass.
    pub fn with_zo_seed(mut self, seed: i64) -> Self {
        self.zo_seed = Some(seed);
        self
    }

    /// Gets the next batch of generated tokens.
    ///
    /// Returns `Some(tokens)` with one or more tokens (speculative decoding
    /// may accept multiple tokens per step), or `None` when generation is
    /// complete (stop token or max tokens reached).
    pub async fn next(&mut self) -> Result<Option<Vec<u32>>> {
        // Check max tokens limit
        if let Some(max) = self.max_tokens {
            if self.tokens_generated >= max {
                return Ok(None);
            }
        }

        if self.done {
            return Ok(None);
        }

        let mut tokens = self.step().await?;
        if tokens.is_empty() {
            return Ok(None);
        }

        // Truncate at the first stop token
        if let Some(pos) = tokens.iter().position(|t| self.stop_tokens.contains(t)) {
            tokens.truncate(pos);
            self.done = true;
            if tokens.is_empty() {
                return Ok(None);
            }
        }

        // Enforce max tokens limit
        if let Some(max) = self.max_tokens {
            let remaining = max - self.tokens_generated;
            if tokens.len() > remaining {
                tokens.truncate(remaining);
                self.done = true;
            }
        }

        self.tokens_generated += tokens.len();
        Ok(Some(tokens))
    }

    /// Collects all tokens from the stream (until stop token or max_tokens limit).
    pub async fn collect_tokens(mut self) -> Result<Vec<u32>> {
        let mut all = Vec::new();
        while let Some(tokens) = self.next().await? {
            all.extend(tokens);
        }
        Ok(all)
    }

    /// Collects all tokens and decodes them to text.
    pub async fn collect_text(self) -> Result<String> {
        let tokenizer = self.ctx.model.tokenizer();
        let tokens = self.collect_tokens().await?;
        tokenizer.decode(&tokens)
    }

    async fn step(&mut self) -> Result<Vec<u32>> {
        // Drain ctx.buffer as pending input tokens for this step.
        let pending = std::mem::take(&mut self.ctx.buffer);
        let n_pending = pending.len() as u32;

        // Get draft tokens for speculative decoding (before reserve, so we
        // can account for their page slots).
        let (draft_tokens, draft_positions) = self.speculation.draft();
        let n_drafted = draft_tokens.len() as u32;

        // Short-circuit: nothing to do if no input and no drafts.
        if n_pending == 0 && n_drafted == 0 {
            return Ok(vec![]);
        }

        // Reserve pages for pending input + speculative draft tokens.
        let n_total_input = n_pending + n_drafted;
        let total_tokens_after = self.ctx.working_tokens + n_total_input;
        let pages_needed =
            (total_tokens_after + self.ctx.page_size - 1) / self.ctx.page_size;
        if pages_needed > self.ctx.working_pages {
            self.ctx
                .inner
                .reserve_working_pages(pages_needed)
                .map_err(|e| format!("Failed to reserve pages: {}", e))?;
            self.ctx.working_pages = pages_needed;
        }

        let pass = ForwardPass::new(&self.ctx.model);
        pass.context(&self.ctx.inner);

        if let Some(adapter) = self.adapter {
            pass.adapter(adapter);
        }
        if let Some(seed) = self.zo_seed {
            crate::pie::zo::zo::adapter_seed(&pass, seed);
        }

        if n_pending > 0 {
            let positions: Vec<u32> =
                (self.ctx.seq_len..self.ctx.seq_len + n_pending).collect();
            pass.input_tokens(&pending, &positions);
        }

        if !draft_tokens.is_empty() {
            pass.input_speculative_tokens(&draft_tokens, &draft_positions);
        }

        if matches!(self.speculation, Speculation::Default { .. }) {
            pass.output_speculative_tokens(true);
        }

        // Sample at the last input token position (or last cached position if no input).
        let sample_idx = if n_pending > 0 { n_pending - 1 } else { 0 };
        pass.sampler(&[sample_idx], self.sampler.clone());

        // Apply logit mask if constraint is available.
        if let Some(ref constraint) = self.constraint {
            pass.logit_mask(&constraint.mask());
        }

        let output = pass.execute_async().await?;
        let new_tokens = self.speculation.accept(output);

        // Update constraint with accepted tokens.
        if let Some(ref mut constraint) = self.constraint {
            constraint.accept(&new_tokens);
        }

        if new_tokens.is_empty() {
            return Ok(vec![]);
        }

        // Roll back rejected speculative tokens from the KV cache.
        // The accepted list includes verified drafts + 1 newly sampled token.
        // The newly sampled token is NOT in KV (it's returned, not cached),
        // so n_verified = new_tokens.len() - 1.
        if n_drafted > 0 {
            let n_verified = (new_tokens.len() as u32).saturating_sub(1);
            let n_rejected = n_drafted.saturating_sub(n_verified);
            if n_rejected > 0 {
                self.ctx.inner.pop_working_page_tokens(n_rejected);
            }
        }

        // Commit full pages and update cached state.
        // Both pending tokens and verified speculative tokens occupy KV slots.
        let n_verified_drafts = if n_drafted > 0 {
            (new_tokens.len() as u32).saturating_sub(1)
        } else {
            0
        };
        let n_kv_tokens = n_pending + n_verified_drafts;

        if n_kv_tokens > 0 {
            let new_working = self.ctx.working_tokens + n_kv_tokens;
            let pages_to_commit = new_working / self.ctx.page_size;

            if pages_to_commit > 0 {
                self.ctx
                    .inner
                    .commit_working_pages(pages_to_commit)
                    .map_err(|e| format!("Failed to commit pages: {}", e))?;
            }

            self.ctx.committed_pages += pages_to_commit;
            self.ctx.working_pages -= pages_to_commit;
            self.ctx.working_tokens = new_working % self.ctx.page_size;
            self.ctx.seq_len += n_kv_tokens;
        }

        // Seed the last generated token back into ctx.buffer for the next step.
        if let Some(&last_token) = new_tokens.last() {
            self.ctx.buffer.push(last_token);
        }

        Ok(new_tokens)
    }

    /// Transition into an [`EventStream`] by attaching a [`Decoder`].
    ///
    /// The resulting stream returns [`Event`]s directly — no raw token
    /// IDs are exposed.
    ///
    /// ```ignore
    /// let mut events = ctx.generate(sampler)
    ///     .with_max_tokens(256)
    ///     .decode()
    ///     .with_reasoning()
    ///     .with_tool_use();
    ///
    /// while let Some(event) = events.next().await? {
    ///     match event {
    ///         Event::Text(s) => print!("{}", s),
    ///         Event::Done(_) => break,
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn decode(self) -> EventStream<'a> {
        EventStream::new(self)
    }
}

// =============================================================================
// EventStream
// =============================================================================

/// Fused token stream + decoder that yields [`Event`]s directly.
///
/// Created by calling [`TokenStream::decode()`]. Optionally chain
/// [`.with_reasoning()`](EventStream::with_reasoning) and/or
/// [`.with_tool_use()`](EventStream::with_tool_use) to enable
/// additional decoding modes.
pub struct EventStream<'a> {
    stream: TokenStream<'a>,
    decoder: Decoder,
}

impl<'a> EventStream<'a> {
    fn new(stream: TokenStream<'a>) -> Self {
        let decoder = Decoder::new(&stream.ctx.model);
        Self { stream, decoder }
    }

    /// Enable reasoning (thinking block) decoding.
    pub fn with_reasoning(mut self) -> Self {
        let model = &self.stream.ctx.model;
        self.decoder = self.decoder.with_reasoning(model);
        self
    }

    /// Enable tool-use decoding.
    pub fn with_tool_use(mut self) -> Self {
        let model = &self.stream.ctx.model;
        self.decoder = self.decoder.with_tool_use(model);
        self
    }

    /// Get the next event from the stream.
    ///
    /// Returns `None` when generation is complete.
    pub async fn next(&mut self) -> Result<Option<Event>> {
        match self.stream.next().await? {
            Some(tokens) => self.decoder.feed(&tokens).map(Some),
            None => Ok(None),
        }
    }
}
