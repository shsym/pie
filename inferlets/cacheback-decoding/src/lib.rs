//! Demonstrates CacheBack speculative decoding — using a cached draft model.
//!
//! This inferlet implements the `Speculate` trait from the SDK to provide
//! custom speculative token drafting. The drafter generates candidate tokens
//! which the verifier model checks in a single forward pass.

use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, InstructExt, Result,
    inference::{ForwardPass, Output, Sampler},
    Speculate, Speculation,
};
use std::time::Instant;

const HELP: &str = "\
Usage: cacheback-decoding [OPTIONS]

A program to demonstrate CacheBack speculative decoding with a custom drafter.

Options:
  -p, --prompt <PROMPT>      The prompt text [default: \"Explain quantum computing.\"]
  -n, --max-tokens <TOKENS>  Maximum number of tokens to generate [default: 256]
  -d, --draft-length <N>     Number of draft tokens per speculation round [default: 4]
  -h, --help                 Prints this help message";

/// A simple greedy drafter that uses a separate context for drafting.
///
/// This implements the `Speculate` trait from the SDK, providing draft tokens
/// for speculative decoding.
struct GreedyDrafter {
    model: Model,
    draft_ctx: Context,
    draft_length: usize,
    page_size: u32,
    pending_tokens: Vec<u32>,
}

impl GreedyDrafter {
    fn new(_model: &Model, source_ctx: &Context, draft_length: usize) -> Result<Self> {
        let draft_ctx = source_ctx.fork()?;
        let page_size = draft_ctx.tokens_per_page();
        Ok(Self {
            model: Model::load(&runtime::models()[0])?,
            draft_ctx,
            draft_length,
            page_size,
            pending_tokens: Vec::new(),
        })
    }
}

impl Speculate for GreedyDrafter {
    fn draft(&self) -> (Vec<u32>, Vec<u32>) {
        // Synchronously generate draft tokens using ForwardPass
        let mut tokens = Vec::new();
        let mut positions = Vec::new();

        if self.pending_tokens.is_empty() {
            return (tokens, positions);
        }

        let mut current_tokens = self.pending_tokens.clone();

        for _i in 0..self.draft_length {
            let wpt = self.draft_ctx.working_page_token_count();
            let seq_len = self.draft_ctx.committed_page_count() * self.page_size + wpt;

            let current_working_pages = self.draft_ctx.working_page_count();
            let total_tokens_after = wpt + current_tokens.len() as u32;
            let total_pages_needed = (total_tokens_after + self.page_size - 1) / self.page_size;
            let additional_pages = total_pages_needed.saturating_sub(current_working_pages);
            if additional_pages > 0 {
                if self.draft_ctx.reserve_working_pages(additional_pages).is_err() {
                    break;
                }
            }

            let pass = ForwardPass::new(&self.model);
            pass.context(&self.draft_ctx);
            let pos: Vec<u32> = (seq_len..seq_len + current_tokens.len() as u32).collect();
            pass.input_tokens(&current_tokens, &pos);
            let last_idx = (current_tokens.len() - 1) as u32;
            pass.sampler(&[last_idx], Sampler::TopP((0.0, 1.0)));

            let output = pass.execute();
            let Ok(future_output) = output else {
                break;
            };

            // Poll to completion
            let output = future_output.get();
            let Some(result) = output else {
                break;
            };

            let drafted = match result {
                Output::Tokens(t) => t,
                _ => break,
            };

            if let Some(&token) = drafted.first() {
                tokens.push(token);
                positions.push(seq_len + current_tokens.len() as u32);

                // Commit pages
                let new_wpt = wpt + current_tokens.len() as u32;
                let pages_to_commit = new_wpt / self.page_size;
                if pages_to_commit > 0 {
                    let _ = self.draft_ctx.commit_working_pages(pages_to_commit);
                }

                current_tokens = vec![token];
            } else {
                break;
            }
        }

        (tokens, positions)
    }

    fn accept(&mut self, accepted_tokens: &[u32]) {
        // Update the draft context to match the accepted state
        if let Some(&last) = accepted_tokens.last() {
            self.pending_tokens = vec![last];
        }
    }

    fn reset(&mut self) {
        // Reset the draft context
        self.pending_tokens.clear();
    }

    fn rollback(&mut self, _num_tokens: usize) {
        // Rollback is a no-op for a simple greedy drafter
    }
}

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or_else(|_| "Explain quantum computing.".to_string());
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);
    let draft_length: usize = args.value_from_str(["-d", "--draft-length"]).unwrap_or(4);

    let start = Instant::now();
    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;
    let _tokenizer = model.tokenizer();

    let ctx = Context::create(&model)?;

    let mut pending_tokens: Vec<u32> = Vec::new();
    pending_tokens.extend(ctx.system("You are a helpful assistant."));
    pending_tokens.extend(ctx.user(&prompt));
    pending_tokens.extend(ctx.cue());
    ctx.flush(&pending_tokens).await?;

    // Create the drafter
    let drafter = GreedyDrafter::new(&model, &ctx, draft_length)?;
    let speculation = Speculation::custom(drafter);

    let text = ctx
        .generate(Sampler::TopP((0.0, 1.0)))
        .with_speculation(speculation)
        .with_max_tokens(max_tokens)
        .collect_text()
        .await?;

    println!(
        "Generated in {:?}",
        start.elapsed()
    );
    println!("Output:\n{}", text);

    Ok(String::new())
}
