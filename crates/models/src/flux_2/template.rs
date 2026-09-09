use std::sync::Arc;

use chat_template::chatml::{ChatML, ChatMLInstruct};
use tokenizer::Tokenizer;

use crate::template::Instruct;

pub const NO_THINKING: &str = "<think>\n\n</think>\n\n";

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    Arc::new(ChatMLInstruct::new(
        tokenizer,
        ChatML {
            thinking: true,
            preserve_thinking: false,
            tools: false,
            generation_suffix: NO_THINKING,
            stop_tokens: crate::qwen_3::tokenizer::STOP_TOKENS,
        },
    ))
}
