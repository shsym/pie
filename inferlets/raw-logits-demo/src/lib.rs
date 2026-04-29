//! Raw-logits correctness + overhead test.
//!
//! For a fixed prompt, runs N decode steps on a fresh forked context per
//! iteration, alternating between two sampler modes:
//!
//!   * `Sampler::top_k(0.0, 1)` — greedy, returns one token id.
//!   * `Sampler::raw_logits()`  — returns the full pre-softmax logit vector
//!                                 packed as native-endian f32 bytes.
//!
//! Asserts that argmax(raw_logits) == greedy token (per iteration), then
//! prints structured `KEY=VALUE` lines that the host-side test parses:
//!
//!   VOCAB_SIZE=<n>
//!   ITERS=<n>
//!   GREEDY_AVG_MS=<f>            mean across iters (sensitive to first-call JIT)
//!   GREEDY_MIN_MS=<f>            steady-state best — most informative
//!   RAW_LOGITS_AVG_MS=<f>
//!   RAW_LOGITS_MIN_MS=<f>
//!   OVERHEAD_AVG_MS=<f>          (raw_logits_avg - greedy_avg)
//!   OVERHEAD_MIN_MS=<f>          (raw_logits_min - greedy_min) — steady-state cost
//!   PAYLOAD_BYTES=<n>            vocab_size * 4
//!   ARGMAX_MATCHES_GREEDY=<n>/<n>

use inferlet::{
    Context, ForwardPassExt, Result,
    inference::{ForwardPass, Output, Sampler},
    model::Model,
    runtime,
};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize, Default)]
struct Input {
    /// Number of paired (greedy + raw-logits) decode steps to time.
    #[serde(default = "default_iters")]
    iters: usize,
}
fn default_iters() -> usize { 50 }

fn decode_logits(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u32;
        }
    }
    best_i
}

/// Run a single decode-step forward pass on a forked context with the given
/// sampler, returning (elapsed_ms, output).
async fn timed_step(
    model: &Model,
    base: &Context,
    sampler: Sampler,
) -> Result<(f64, Output)> {
    let ctx = base.fork()?;

    // The forked context already has the prompt's KV committed; we just
    // append a single placeholder token at the next position to drive one
    // decode step. We pick an arbitrary token (0) since we only care about
    // what the model would predict from the prompt.
    let page_size = ctx.page_size();
    let wpt = ctx.inner().working_page_token_count();
    let committed = ctx.inner().committed_page_count();
    let seq_len = committed * page_size + wpt;

    // Reserve an extra page if the new token would overflow the working tail.
    let working_pages = ctx.inner().working_page_count();
    let total_after = wpt + 1;
    let pages_needed = (total_after + page_size - 1) / page_size;
    let extra = pages_needed.saturating_sub(working_pages);
    if extra > 0 {
        ctx.inner()
            .reserve_working_pages(extra)
            .map_err(|e| format!("reserve_working_pages: {e}"))?;
    }

    let pass = ForwardPass::new(model);
    pass.context(ctx.inner());
    pass.input_tokens(&[0u32], &[seq_len]);
    pass.sampler(&[0u32], &sampler);

    let t0 = Instant::now();
    let output = pass
        .execute_async()
        .await
        .map_err(|e| format!("forward pass: {e}"))?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    Ok((elapsed_ms, output))
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let iters = input.iters.max(1);

    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let base = Context::new(&model)?;

    // Build a small fixed prompt and prefill it once.
    let mut prompt: Vec<u32> = Vec::new();
    prompt.extend(inferlet::instruct::chat::system(
        &model,
        "You complete the user's sentence with a single word.",
    ));
    prompt.extend(inferlet::instruct::chat::user(
        &model,
        "The capital of France is",
    ));
    prompt.extend(inferlet::instruct::chat::cue(&model));

    let page_size = base.page_size();
    let pages_needed = ((prompt.len() as u32) + page_size - 1) / page_size;
    if pages_needed > 0 {
        base.inner()
            .reserve_working_pages(pages_needed)
            .map_err(|e| format!("reserve_working_pages(prompt): {e}"))?;
    }

    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let prefill = ForwardPass::new(&model);
    prefill.context(base.inner());
    prefill.input_tokens(&prompt, &positions);
    // No sampler attached — we just want the KV populated.
    let _ = prefill
        .execute_async()
        .await
        .map_err(|e| format!("prefill: {e}"))?;

    // Commit completed pages so forks see the same prefix.
    let new_wpt = base.inner().working_page_token_count();
    let pages_to_commit = new_wpt / page_size;
    if pages_to_commit > 0 {
        base.inner()
            .commit_working_pages(pages_to_commit)
            .map_err(|e| format!("commit_working_pages: {e}"))?;
    }

    // -------- Warmup (don't time; covers JIT compile of attention, sampling,
    //          softmax kernels, plus any first-call allocator paths). 5 paired
    //          rounds is enough on flashinfer/triton in our experience. -----
    for _ in 0..5 {
        let _ = timed_step(&model, &base, Sampler::top_k(0.0, 1)).await?;
        let _ = timed_step(&model, &base, Sampler::raw_logits()).await?;
    }

    // -------- Timed loop ---------------------------------------------------
    let mut greedy_total = 0.0_f64;
    let mut raw_total = 0.0_f64;
    let mut greedy_min = f64::INFINITY;
    let mut raw_min = f64::INFINITY;
    let mut matches = 0usize;
    let mut vocab_size = 0usize;

    for i in 0..iters {
        // Greedy first.
        let (g_ms, g_out) = timed_step(&model, &base, Sampler::top_k(0.0, 1)).await?;
        let greedy_token = g_out.first_token().ok_or("greedy: no token slot")?;
        greedy_total += g_ms;
        if g_ms < greedy_min { greedy_min = g_ms; }

        // Raw logits.
        let (r_ms, r_out) = timed_step(&model, &base, Sampler::raw_logits()).await?;
        let logits_bytes = r_out.first_logits().ok_or("raw: no logits slot")?.to_vec();
        let logits = decode_logits(&logits_bytes);
        if vocab_size == 0 {
            vocab_size = logits.len();
        }
        let argmax_token = argmax(&logits);
        if argmax_token == greedy_token {
            matches += 1;
        } else {
            println!(
                "iter {i}: argmax({}) != greedy({}) — model may be applying a sampling mask",
                argmax_token, greedy_token
            );
        }
        raw_total += r_ms;
        if r_ms < raw_min { raw_min = r_ms; }
    }

    let greedy_avg = greedy_total / iters as f64;
    let raw_avg = raw_total / iters as f64;
    let overhead_avg = raw_avg - greedy_avg;
    let overhead_min = raw_min - greedy_min;
    let payload_bytes = vocab_size * 4;

    println!("VOCAB_SIZE={}", vocab_size);
    println!("ITERS={}", iters);
    println!("GREEDY_AVG_MS={:.3}", greedy_avg);
    println!("GREEDY_MIN_MS={:.3}", greedy_min);
    println!("RAW_LOGITS_AVG_MS={:.3}", raw_avg);
    println!("RAW_LOGITS_MIN_MS={:.3}", raw_min);
    println!("OVERHEAD_AVG_MS={:.3}", overhead_avg);
    println!("OVERHEAD_MIN_MS={:.3}", overhead_min);
    println!("PAYLOAD_BYTES={}", payload_bytes);
    println!("ARGMAX_MATCHES_GREEDY={}/{}", matches, iters);

    Ok(String::new())
}
