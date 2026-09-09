use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::Instruct;

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    crate::qwen_3::template::chatml(tokenizer)
}
