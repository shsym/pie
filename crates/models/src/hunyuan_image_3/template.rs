use std::sync::Arc;

use tokenizer::Tokenizer;

use crate::template::{
    ChatDecoder, GenericChatDecoder, Instruct, NoopReasoningDecoder, NoopToolDecoder,
    ReasoningDecoder, ToolDecoder, special, specials,
};

use super::tokenizer::{END_OF_TEXT, START_OF_TEXT, STOP_TOKENS};

pub struct HunyuanImage3 {
    tokenizer: Arc<Tokenizer>,
    bos: u32,
    eos: u32,
    sep: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl HunyuanImage3 {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        let sep = tokenizer.encode("\n\n");
        Self {
            bos: special(&tokenizer, START_OF_TEXT),
            eos: special(&tokenizer, END_OF_TEXT),
            sep,
            stop_ids: specials(&tokenizer, STOP_TOKENS),
            tokenizer,
        }
    }

    fn turn(&self, role: &str, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("{role}: {}", msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }
}

impl Instruct for HunyuanImage3 {
    fn prefix(&self) -> Vec<u32> {
        vec![self.bos]
    }

    fn system(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.tokenizer.encode(msg.trim()));
        tokens.extend(&self.sep);
        tokens
    }

    fn first_user(&self, msg: &str) -> Vec<u32> {
        let mut tokens = vec![self.bos];
        tokens.extend(self.turn("User", msg));
        tokens
    }

    fn user(&self, msg: &str) -> Vec<u32> {
        self.turn("User", msg)
    }

    fn assistant(&self, msg: &str) -> Vec<u32> {
        let mut tokens = self.tokenizer.encode(&format!("Assistant: {}", msg.trim()));
        tokens.push(self.eos);
        tokens
    }

    fn cue(&self) -> Vec<u32> {
        self.tokenizer.encode("Assistant:")
    }

    fn seal(&self) -> Vec<u32> {
        vec![self.eos]
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(
            self.tokenizer.clone(),
            self.stop_ids.clone(),
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
    Arc::new(HunyuanImage3::new(tokenizer))
}
