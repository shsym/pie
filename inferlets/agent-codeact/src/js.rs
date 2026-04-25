//! JavaScript execution helpers for the CodeACT agent.
//!
//! Wraps the Boa engine and extracts ```javascript blocks from
//! model responses.

use boa_engine::{Context, Source};

/// If `text` contains a ```javascript code block, evaluate it with Boa
/// and return the result. Returns `None` when no block is present.
pub fn try_eval_block(text: &str) -> Option<String> {
    let code = extract_block(text)?;
    Some(eval(&code))
}

fn extract_block(text: &str) -> Option<String> {
    // Use the last block — thinking models may emit several during reasoning.
    let start = text.rfind("```javascript")? + "```javascript".len();
    let end = text[start..].find("```")?;
    Some(text[start..start + end].trim().to_string())
}

fn eval(code: &str) -> String {
    let mut ctx = Context::default();
    match ctx.eval(Source::from_bytes(code)) {
        Ok(value) => value
            .to_string(&mut ctx)
            .ok()
            .and_then(|s| s.to_std_string().ok())
            .unwrap_or_else(|| "undefined".into()),
        Err(e) => format!("Execution Error: {e}"),
    }
}
