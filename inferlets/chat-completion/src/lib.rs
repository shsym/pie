//! OpenAI-compatible chat completion with automatic prefix caching (APC).
//!
//! Input shape:
//! ```json
//! {
//!   "messages": [
//!     {"role": "system", "content": "You are a helpful assistant."},
//!     {"role": "user",   "content": "What is 7*8?"}
//!   ],
//!   "max_tokens":  1024,    // optional; capped at 65536
//!   "temperature": 0.7,     // optional; clamped to [0, 2]
//!   "top_p":       0.95,    // optional; clamped to (0, 1]
//!   "stream":      true,    // optional; emit `{"content": "..."}` chunks to stdout
//!   "apc":         true     // optional; cache the prefix across calls
//! }
//! ```
//!
//! APC strategy: hash the prefix (messages except the trailing user turn) into
//! a stable key; try `Context::open(key)` first; on miss build fresh, flush,
//! and `Context::save(key)` so the next request with the same prefix hits.
//! The mechanism is fully inferlet-side — no engine support required beyond
//! the standard `Context::open` / `save` primitives.
//!
//! Defensive contract (mirroring `pie-pr-review`'s patterns):
//!  * Reject any message whose content contains a chat-template control
//!    sequence (`<|...|>` shape + Qwen3 `<tool_call>` / `</tool_call>`).
//!  * Reject unknown message roles loudly, surfacing the offending value.
//!  * Reject `max_tokens = 0`; clamp `max_tokens > 65536` to 65536.
//!  * Cap message count at 64 and per-message content at 256 KiB.

use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};
use serde::Deserialize;

// ── Defensive limits ──────────────────────────────────────────────────
const MAX_MESSAGES: usize = 64;
const MAX_CONTENT_BYTES: usize = 256 * 1024;
const MAX_OUTPUT_TOKENS: u32 = 1 << 16; // 65_536
const APC_KEY_PREFIX: &str = "pie-apc:";

// ── Input shape ───────────────────────────────────────────────────────
#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Input {
    messages: Vec<ChatMessage>,

    #[serde(default = "default_max_tokens")]
    max_tokens: u32,

    #[serde(default = "default_temperature")]
    temperature: f32,

    #[serde(default = "default_top_p")]
    top_p: f32,

    #[serde(default = "default_stream")]
    stream: bool,

    #[serde(default = "default_apc")]
    apc: bool,
}

fn default_max_tokens() -> u32 { 1024 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.95 }
fn default_stream() -> bool { true }
fn default_apc() -> bool { true }

// ── Entry ─────────────────────────────────────────────────────────────
#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let max_tokens = clamp_max_tokens(input.max_tokens)? as usize;
    let temperature = input.temperature.clamp(0.0, 2.0);
    let top_p = clamp_top_p(input.top_p);
    let messages = validate_messages(input.messages)?;

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or_else(|| "no models loaded".to_string())?;
    let model = Model::load(&model_name)?;

    let (prefix, tail) = split_prefix_tail(&messages);
    let cache_key = if input.apc && tail.is_some() {
        Some(apc_key(&model_name, prefix))
    } else {
        None
    };

    let mut ctx = open_or_build(&model, prefix, cache_key.as_deref()).await?;

    if let Some(tail_msg) = tail {
        append_message(&mut ctx, tail_msg);
    }
    ctx.cue();

    generate(&mut ctx, &model, max_tokens, temperature, top_p, input.stream).await
}

// ── Validation ────────────────────────────────────────────────────────
fn clamp_max_tokens(n: u32) -> Result<u32> {
    if n == 0 {
        return Err("max_tokens must be > 0".into());
    }
    Ok(n.min(MAX_OUTPUT_TOKENS))
}

fn clamp_top_p(p: f32) -> f32 {
    // top_p of 0 would collapse the distribution; treat as the default.
    if p > 0.0 && p <= 1.0 { p } else { default_top_p() }
}

fn validate_messages(msgs: Vec<ChatMessage>) -> Result<Vec<ChatMessage>> {
    if msgs.is_empty() {
        return Err("messages: must contain at least one message".into());
    }
    if msgs.len() > MAX_MESSAGES {
        return Err(format!(
            "messages length {} exceeds inferlet cap {MAX_MESSAGES}",
            msgs.len()
        ));
    }
    for (idx, m) in msgs.iter().enumerate() {
        if m.content.is_empty() {
            return Err(format!("messages[{idx}].content is empty"));
        }
        if m.content.len() > MAX_CONTENT_BYTES {
            return Err(format!(
                "messages[{idx}].content size {} exceeds inferlet cap {MAX_CONTENT_BYTES}",
                m.content.len()
            ));
        }
        if !matches!(m.role.as_str(), "system" | "user" | "assistant") {
            return Err(format!(
                "messages[{idx}].role `{}` is not one of system/user/assistant",
                m.role
            ));
        }
        if contains_control_token(&m.content) {
            return Err(format!(
                "messages[{idx}].content contains a chat-template control sequence (`<|…|>` or <tool_call>); refusing to forward"
            ));
        }
    }
    Ok(msgs)
}

/// Reject content that looks like a chat-template control sequence.
///
/// The host tokenizer's `encode()` runs an Aho-Corasick over every entry in
/// the model's `added_tokens` list, so a user who puts any of those literal
/// strings in their content can forge a fake role boundary before generation
/// starts. The proper fix is upstream (an `encode_ordinary` mode on the host
/// tokenizer, or an inferlet-SDK accessor like `Tokenizer::added_tokens()`)
/// so the deny list is exact. Until either lands, this is a defense-in-depth
/// screen at the inferlet boundary, covering the `<|...|>` shape that pie's
/// shipped chat templates use:
///
///   * `<|im_start|>` / `<|im_end|>`   — ChatML / Qwen
///   * `<|eot_id|>` / `<|start_header_id|>` etc. — Llama-3
///   * `<|endoftext|>`                  — OLMo-2 / GPT-style
///   * `<|system|>`                     — Phi-3
///   * `<tool_call>` / `</tool_call>`   — Qwen3 tool-calling
///
/// Intentionally NOT covered (false-positive on real code):
///   * Mistral `[INST]`, `[/INST]`, `[TOOL_CALLS]` — bracket family overlaps
///     with legitimate JSON/code content; can't reliably scan without
///     rejecting real prompts.
///   * Llama-2 / Gemma BOS/EOS: `<s>`, `</s>`, `<bos>`, `<eos>`,
///     `<start_of_turn>`, `<end_of_turn>` — same reason.
///
/// Deployments using those families should wait for the upstream tokenizer
/// fix (see pie-pr-review's pr-review inferlet for the same scan).
fn contains_control_token(s: &str) -> bool {
    // Literals that don't fit the `<|...|>` shape.
    const LITERALS: &[&str] = &["<tool_call>", "</tool_call>"];
    for lit in LITERALS {
        if s.contains(lit) {
            return true;
        }
    }

    // Shape scan: `<|` followed (on the same line) by `|>`. Same-line is the
    // key constraint — multi-line `<|`-prefixed text in a code diff (Verilog,
    // OCaml's metaprogramming, etc.) is a false positive; chat-template
    // markers never span a newline.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'<' && bytes[i + 1] == b'|' {
            let mut j = i + 2;
            while j + 1 < bytes.len() {
                if bytes[j] == b'\n' {
                    break;
                }
                if bytes[j] == b'|' && bytes[j + 1] == b'>' {
                    return true;
                }
                j += 1;
            }
        }
        i += 1;
    }
    false
}

// ── Prefix / tail split + APC key ─────────────────────────────────────
fn split_prefix_tail(msgs: &[ChatMessage]) -> (&[ChatMessage], Option<&ChatMessage>) {
    // Only treat the final message as "tail" when it is a user message.
    // For other shapes (assistant continuation, history-only) the whole
    // list is prefix and there is nothing to cache for next time.
    match msgs.last() {
        Some(last) if last.role == "user" => (&msgs[..msgs.len() - 1], Some(last)),
        _ => (msgs, None),
    }
}

fn apc_key(model_name: &str, prefix: &[ChatMessage]) -> String {
    // Stable, version-independent hash so saved snapshots survive engine
    // upgrades. FNV-1a 64-bit is more than enough collision resistance for
    // an in-process cache namespace.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(&mut h, model_name.as_bytes());
    mix(&mut h, b"\x00");
    for m in prefix {
        mix(&mut h, m.role.as_bytes());
        mix(&mut h, b"\x01");
        mix(&mut h, m.content.as_bytes());
        mix(&mut h, b"\x02");
    }
    format!("{APC_KEY_PREFIX}{:016x}", h)
}

// ── Context construction (open-or-build) ──────────────────────────────
async fn open_or_build(
    model: &Model,
    prefix: &[ChatMessage],
    cache_key: Option<&str>,
) -> Result<Context> {
    if let Some(key) = cache_key {
        if let Ok(ctx) = Context::open(model, key) {
            return Ok(ctx);
        }
    }

    let mut ctx = Context::new(model)?;
    for m in prefix {
        append_message(&mut ctx, m);
    }

    // Persist the prefix for future requests that share it. Flush first so
    // working pages are committed into the snapshot; save errors are
    // non-fatal — they just mean no caching for this key on this run.
    if let Some(key) = cache_key {
        if let Err(e) = ctx.flush().await {
            return Err(format!("flush before save failed: {e}"));
        }
        let _ = ctx.save(key);
    }

    Ok(ctx)
}

fn append_message(ctx: &mut Context, m: &ChatMessage) {
    match m.role.as_str() {
        "system" => { ctx.system(&m.content); }
        "user" => { ctx.user(&m.content); }
        "assistant" => { ctx.assistant(&m.content); }
        // validate_messages already rejected anything else.
        _ => unreachable!(),
    }
}

// ── Generation + streaming output ─────────────────────────────────────
async fn generate(
    ctx: &mut Context,
    model: &Model,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    stream: bool,
) -> Result<String> {
    let mut decoder = chat::Decoder::new(model);
    let mut full = String::new();

    let mut g = ctx
        .generate(Sampler::TopP { temperature, p: top_p })
        .max_tokens(max_tokens)
        .stop(&chat::stop_tokens(model));

    while let Some(step) = g.next()? {
        let out = step.execute().await?;
        if out.tokens.is_empty() {
            continue;
        }
        match decoder.feed(&out.tokens)? {
            chat::Event::Delta(s) => {
                if stream {
                    emit_delta(&s);
                }
                full.push_str(&s);
            }
            chat::Event::Done(s) => {
                full = s;
                break;
            }
            chat::Event::Interrupt(_) | chat::Event::Idle => {}
        }
    }

    Ok(full)
}

fn emit_delta(s: &str) {
    // OpenAI-shaped delta line. An HTTP bridge in front of pie can parse each
    // line as JSON and forward `content` as the SSE chunk's `delta.content`.
    let chunk = serde_json::json!({ "content": s });
    println!("{}", chunk);
}
