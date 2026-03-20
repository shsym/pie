//! Token and event streams for generation.

use crate::inference::{ForwardPass, Sampler};
use crate::ForwardPassExt;
use crate::Result;

use crate::pie::instruct::chat;

use super::{Context, Decoder, Event, Speculation, Constrain, GrammarConstraint};

/// Budget-exhausting bid: maximum sustainable rent per page per step.
///
/// Formula (§5 of SCHED.md):
///   bid = (B/μ + d − 1/s) / (p + μ(1 + cv²) / (2s))
///
/// where B = balance, μ = expected remaining steps, d = dividend per step,
/// s = page_size, p = current pages, cv² = squared coefficient of variation
/// of the remaining-steps distribution.
fn compute_bid(balance: f64, pages: f64, mu: f64, cv2: f64, page_size: f64, dividend: f64) -> f64 {
    let mu = mu.max(1.0);
    let g = 1.0 / page_size;  // make cost rate (pages per step)
    let numerator = balance / mu + dividend - g;
    let denominator = pages + mu * (1.0 + cv2) / (2.0 * page_size);
    if denominator > 0.0 { numerator / denominator } else { numerator }
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
    pub(super) ctx: &'a mut Context,
    sampler: Sampler,
    stop_tokens: Vec<u32>,
    speculation: Speculation,
    constraint: Option<Box<dyn Constrain>>,
    done: bool,
    max_tokens: Option<usize>,
    tokens_generated: usize,
    adapter: Option<&'a crate::adapter::Adapter>,
    zo_seed: Option<i64>,
    /// Expected output length for bid planning. Falls back to
    /// `max_tokens`, then 4096 if unset.
    horizon: Option<usize>,

}

impl<'a> TokenStream<'a> {
    /// Creates a new token stream.
    pub(super) fn new(ctx: &'a mut Context, sampler: Sampler) -> Self {
        let stop_tokens = chat::stop_tokens(&ctx.model);

        // Set initial bid: budget-exhausting rate with geometric prior (cv²=1).
        let balance = crate::scheduling::balance(&ctx.model);
        let dividend = crate::scheduling::dividend(&ctx.model);
        let pages = (ctx.committed_pages + ctx.working_pages).max(1) as f64;
        let page_size = ctx.page_size as f64;
        ctx.bid(compute_bid(balance, pages, 4096.0, 1.0, page_size, dividend));

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
            horizon: None,
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
        // Use max_tokens as horizon if no explicit horizon was set,
        // so the bid formula reflects the actual generation length.
        if self.horizon.is_none() {
            self.with_horizon(max_tokens)
        } else {
            self
        }
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

    /// Sets the expected output length for budget planning.
    ///
    /// The bid is spread across `horizon` steps. Programs that know
    /// their expected output length should set this for tighter bidding.
    /// Falls back to `max_tokens`, then 4096.
    pub fn with_horizon(mut self, horizon: usize) -> Self {
        self.horizon = Some(horizon);
        // Re-compute bid with tight horizon (deterministic, cv²=0).
        let balance = crate::scheduling::balance(&self.ctx.model);
        let dividend = crate::scheduling::dividend(&self.ctx.model);
        let pages = (self.ctx.committed_pages + self.ctx.working_pages).max(1) as f64;
        let page_size = self.ctx.page_size as f64;
        let mu = (horizon - self.tokens_generated).max(1) as f64;
        self.ctx.bid(compute_bid(balance, pages, mu, 0.0, page_size, dividend));
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

    /// Generate JSON-constrained output and deserialize into `T`.
    ///
    /// This method:
    /// 1. Extracts a JSON schema from `T` using [`schemars::JsonSchema`].
    /// 2. Applies a grammar constraint so generation always produces valid JSON.
    /// 3. Collects all tokens, decodes to text, and deserializes into `T`.
    ///
    /// ```ignore
    /// #[derive(Deserialize, JsonSchema)]
    /// struct City { name: String, population: u64 }
    ///
    /// ctx.system("Extract city info.").user("Paris has 2M people.").cue();
    /// let city: City = ctx.generate(sampler).collect_json().await?;
    /// ```
    pub async fn collect_json<T>(mut self) -> Result<T>
    where
        T: serde::de::DeserializeOwned + schemars::JsonSchema,
    {
        // 1. Extract JSON schema from T.
        let schema = schemars::schema_for!(T);
        let schema_str = serde_json::to_string(&schema)
            .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;

        // 2. Create grammar constraint and apply.
        let constraint = GrammarConstraint::from_json_schema(
            &schema_str,
            &self.ctx.model,
        )?;
        self.constraint = Some(Box::new(constraint));

        // 3. Generate, decode, and deserialize.
        let text = self.collect_text().await?;
        serde_json::from_str(&text)
            .map_err(|e| format!("Failed to deserialize JSON: {e}"))
    }

    async fn step(&mut self) -> Result<Vec<u32>> {
        // ── Market: budget-exhausting bid (§5 of SCHED.md) ───────────────
        // bid = (B/μ + d − 1/s) / (p + μ(1+cv²)/(2s))
        // When B/μ + d < 1/s, bid goes negative → prefer exclusion.
        {
            let balance = crate::scheduling::balance(&self.ctx.model);
            let dividend = crate::scheduling::dividend(&self.ctx.model);
            let pages = (self.ctx.committed_pages + self.ctx.working_pages) as f64;
            let page_size = self.ctx.page_size as f64;

            // Horizon cascade: explicit → max_tokens → Lindy.
            let (mu, cv2) = if let Some(h) = self.horizon {
                // Deterministic: program declared its output length.
                ((h - self.tokens_generated).max(1) as f64, 0.0)
            } else if let Some(m) = self.max_tokens {
                // Hard cap known, stopping point unknown → geometric prior.
                ((m - self.tokens_generated).max(1) as f64, 1.0)
            } else {
                // Fully online: Lindy heuristic (μ = elapsed, floor 64).
                (self.tokens_generated.max(64) as f64, 1.0)
            };

            self.ctx.bid(compute_bid(balance, pages, mu, cv2, page_size, dividend));
        }

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

        // Reserve additional pages for pending input + speculative draft tokens.
        let n_total_input = n_pending + n_drafted;
        let total_tokens_after = self.ctx.working_tokens + n_total_input;
        let pages_needed =
            (total_tokens_after + self.ctx.page_size - 1) / self.ctx.page_size;
        let additional = pages_needed.saturating_sub(self.ctx.working_pages);
        if additional > 0 {
            self.ctx
                .inner
                .reserve_working_pages(additional)
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
                self.ctx.inner.truncate_working_page_tokens(n_rejected);
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
