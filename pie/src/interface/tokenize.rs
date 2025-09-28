use crate::instance::InstanceState;
use crate::interface::core::Model;
use crate::interface::pie::inferlet;
use crate::model::tokenizer::BytePairEncoder;
use anyhow::{bail, Result};
use std::sync::Arc;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub inner: Arc<BytePairEncoder>,
}

impl inferlet::tokenize::Host for InstanceState {
    async fn get_tokenizer(&mut self, model: Resource<Model>) -> Result<Resource<Tokenizer>> {
        let inner = self.ctx().table.get(&model)?.info.tokenizer.clone();

        Ok(self.ctx().table.push(Tokenizer { inner })?)
    }
}

impl inferlet::tokenize::HostTokenizer for InstanceState {
    async fn tokenize(&mut self, this: Resource<Tokenizer>, text: String) -> Result<Vec<u32>> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.encode_with_special_tokens(&text))
    }

    async fn detokenize(&mut self, this: Resource<Tokenizer>, tokens: Vec<u32>) -> Result<String> {
        let tokenizer = self.ctx().table.get(&this)?;
        let out = tokenizer.inner.decode(&tokens);

        if let Ok(out) = out {
            Ok(out)
        } else {
            println!("Failed to decode tokens: {:?}", out);
            bail!("Failed to decode tokens: {:?}", out);
        }
    }

    async fn get_vocabs(&mut self, this: Resource<Tokenizer>) -> Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.ctx().table.get(&this)?;
        Ok(tokenizer.inner.get_vocabs())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
