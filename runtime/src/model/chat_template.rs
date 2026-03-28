//! In-process chat template rendering via minijinja.
//!
//! When the Python backend reports the model's chat template is
//! minijinja-compatible, this module renders chat messages and tokenizes
//! the result using the in-process BytePairEncoder -- bypassing the
//! Python RPC entirely.
//!
//! Templates using `namespace()` work natively with minijinja's `builtins` feature.
//! Templates using `.startswith()` / `.endswith()` are pre-processed to use
//! minijinja's `is startingwith` / `is endingwith` test syntax.
//!
//! For truly incompatible templates (raise_exception(), etc.)
//! the caller falls back to the existing Python RPC path.

use super::tokenizer::BytePairEncoder;
use serde_json::Value;

/// Render a chat template and tokenize the result in-process.
///
/// # Arguments
/// * `chat_template` - Raw Jinja2 template string from HF tokenizer
/// * `messages_json` - JSON string: either a list of messages, or
///   `{"messages": [...], "chat_template_kwargs": {...}}`
/// * `tools_json` - Optional JSON string with tool definitions
/// * `add_generation_prompt` - Whether to append generation prompt marker
/// * `tokenizer` - BytePairEncoder for tokenization
///
/// # Returns
/// `Ok(token_ids)` on success, `Err(reason)` on failure (caller should
/// fall back to Python RPC).
pub fn render_and_tokenize(
    chat_template: &str,
    messages_json: &str,
    tools_json: Option<&str>,
    add_generation_prompt: bool,
    tokenizer: &BytePairEncoder,
) -> Result<Vec<u32>, String> {
    // Parse the messages payload (may be wrapped with chat_template_kwargs).
    let raw: Value = serde_json::from_str(messages_json)
        .map_err(|e| format!("failed to parse messages_json: {}", e))?;

    let (messages, kwargs): (Vec<Value>, Option<Value>) = if let Some(obj) = raw.as_object() {
        // Wrapped payload: {"messages": [...], "chat_template_kwargs": {...}}
        let msgs = obj
            .get("messages")
            .cloned()
            .and_then(|v| if v.is_array() { Some(v) } else { None })
            .ok_or_else(|| "wrapped payload missing 'messages' array".to_string())?;
        let msgs_vec: Vec<Value> = serde_json::from_value(msgs)
            .map_err(|e| format!("failed to parse messages array: {}", e))?;
        let kw = obj.get("chat_template_kwargs").cloned();
        (msgs_vec, kw)
    } else if let Some(arr) = raw.as_array() {
        (arr.clone(), None)
    } else {
        return Err("messages_json is neither an object nor an array".to_string());
    };

    // Parse tools if present.
    let tools: Option<Vec<Value>> = match tools_json {
        Some(s) if !s.is_empty() => {
            let parsed: Vec<Value> = serde_json::from_str(s)
                .map_err(|e| format!("failed to parse tools_json: {}", e))?;
            Some(parsed)
        }
        _ => None,
    };

    // Build minijinja render context.
    let rendered = render_template(chat_template, &messages, add_generation_prompt, &tools, &kwargs)?;

    // Tokenize the rendered string using BytePairEncoder.
    let token_ids = tokenizer.encode_with_special_tokens(&rendered);

    Ok(token_ids)
}

/// Pre-process a Jinja2 template to replace Python-specific syntax with
/// minijinja-compatible equivalents:
///   `.startswith('x')` → `is startingwith('x')`
///   `.endswith('x')` → `is endingwith('x')`
///
/// Uses simple pattern matching since these calls always appear as
/// `<expr>.startswith(<arg>)` in HF templates.
fn preprocess_template(template: &str) -> String {
    let mut result = String::with_capacity(template.len());
    let mut remaining = template;

    while !remaining.is_empty() {
        // Find the next .startswith( or .endswith(
        let sw_pos = remaining.find(".startswith(");
        let ew_pos = remaining.find(".endswith(");

        let (method, replacement, pos) = match (sw_pos, ew_pos) {
            (Some(s), Some(e)) if s < e => (".startswith(", " is startingwith(", s),
            (Some(s), Some(e)) if e < s => (".endswith(", " is endingwith(", e),
            (Some(s), _) => (".startswith(", " is startingwith(", s),
            (_, Some(e)) => (".endswith(", " is endingwith(", e),
            _ => {
                result.push_str(remaining);
                break;
            }
        };

        // Copy everything before the method call
        result.push_str(&remaining[..pos]);
        // Replace .method( with ` is test(`
        result.push_str(replacement);
        remaining = &remaining[pos + method.len()..];
    }

    result
}

/// Render a chat template string with minijinja.
fn render_template(
    template_str: &str,
    messages: &[Value],
    add_generation_prompt: bool,
    tools: &Option<Vec<Value>>,
    kwargs: &Option<Value>,
) -> Result<String, String> {
    let processed = preprocess_template(template_str);
    let mut env = minijinja::Environment::new();
    env.add_template("chat", &processed)
        .map_err(|e| format!("minijinja template parse error: {}", e))?;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("minijinja get_template error: {}", e))?;

    // Build the context. The HF chat template expects:
    //   messages: list of message dicts
    //   add_generation_prompt: bool
    //   tools: optional list of tool defs
    //   bos_token, eos_token: usually not needed in template
    //   + any chat_template_kwargs (e.g., enable_thinking)
    let mut ctx = serde_json::Map::new();
    ctx.insert(
        "messages".to_string(),
        serde_json::to_value(messages).unwrap(),
    );
    ctx.insert(
        "add_generation_prompt".to_string(),
        Value::Bool(add_generation_prompt),
    );

    if let Some(tools_val) = tools {
        ctx.insert(
            "tools".to_string(),
            serde_json::to_value(tools_val).unwrap(),
        );
    }

    // Merge chat_template_kwargs into the top-level context.
    if let Some(Value::Object(kw_map)) = kwargs {
        for (k, v) in kw_map {
            ctx.insert(k.clone(), v.clone());
        }
    }

    let ctx_value = Value::Object(ctx);
    let minijinja_ctx = minijinja::value::Value::from_serialize(&ctx_value);

    tmpl.render(minijinja_ctx)
        .map_err(|e| format!("minijinja render error: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_simple_chatml() {
        let template = concat!(
            "{%- for message in messages %}",
            "{%- if message.role == 'system' %}",
            "<|im_start|>system\n{{ message.content }}<|im_end|>\n",
            "{%- elif message.role == 'user' %}",
            "<|im_start|>user\n{{ message.content }}<|im_end|>\n",
            "{%- elif message.role == 'assistant' %}",
            "<|im_start|>assistant\n{{ message.content }}<|im_end|>\n",
            "{%- endif %}",
            "{%- endfor %}",
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}",
        );

        let messages = vec![
            serde_json::json!({"role": "system", "content": "You are helpful."}),
            serde_json::json!({"role": "user", "content": "Hello!"}),
        ];

        let result = render_template(template, &messages, true, &None, &None);
        assert!(result.is_ok(), "render failed: {:?}", result.err());
        let rendered = result.unwrap();
        assert!(rendered.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(rendered.contains("<|im_start|>user\nHello!<|im_end|>"));
        // The {%- endif %} strips trailing whitespace, so no \n at the end.
        assert!(
            rendered.ends_with("<|im_start|>assistant\n") || rendered.ends_with("<|im_start|>assistant"),
            "unexpected ending: {:?}",
            &rendered[rendered.len().saturating_sub(40)..],
        );
    }

    #[test]
    fn test_render_with_kwargs() {
        // Template that uses enable_thinking
        let template = concat!(
            "{%- if enable_thinking is defined and enable_thinking is false %}",
            "<|thinking_off|>",
            "{%- endif %}",
            "{%- for message in messages %}",
            "<|{{ message.role }}|>{{ message.content }}",
            "{%- endfor %}",
            "{%- if add_generation_prompt %}<|assistant|>{%- endif %}",
        );

        let messages = vec![
            serde_json::json!({"role": "user", "content": "Hi"}),
        ];
        let kwargs = Some(serde_json::json!({"enable_thinking": false}));

        let result = render_template(template, &messages, true, &None, &kwargs);
        assert!(result.is_ok(), "render failed: {:?}", result.err());
        let rendered = result.unwrap();
        assert!(rendered.contains("<|thinking_off|>"));
    }
}
