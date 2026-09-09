use std::sync::Arc;

use chat_template::decode::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use chat_template::{ChatDecoder, ReasoningDecoder, ToolDecoder};
use tokenizer::Tokenizer;

use crate::template::Instruct;

struct RawText {
    tokenizer: Arc<Tokenizer>,
    stop: Vec<u32>,
}

impl RawText {
    fn text(&self, msg: &str) -> Vec<u32> {
        self.tokenizer.encode(msg)
    }
}

impl Instruct for RawText {
    fn system(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        self.text(msg)
    }

    fn cue(&self) -> Vec<u32> {
        Vec::new()
    }

    fn seal(&self) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            Arc::clone(&self.tokenizer),
            self.stop.clone(),
        ))
    }

    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

#[must_use]
pub fn instruct(tokenizer: Arc<Tokenizer>) -> Arc<dyn Instruct> {
    let stop = tokenizer.token_to_id("</s>").into_iter().collect();
    Arc::new(RawText { tokenizer, stop })
}
