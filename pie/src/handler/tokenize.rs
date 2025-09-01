use crate::bindings;
use crate::handler::core::Queue;
use crate::instance::InstanceState;
use crate::model_old::Command;
use crate::tokenizer::BytePairEncoder;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub inner: Arc<BytePairEncoder>,
}

impl bindings::pie::inferlet::tokenize::Host for InstanceState {
    async fn get_tokenizer(
        &mut self,
        queue: Resource<Queue>,
    ) -> anyhow::Result<Resource<Tokenizer>> {
        let q = self.table().get(&queue)?;
        let (tx, rx) = oneshot::channel();
        Command::GetTokenizer { handle: tx }.dispatch(q.service_id)?;
        let inner = rx.await?;
        Ok(self.table().push(Tokenizer { inner })?)
    }
}

impl bindings::pie::inferlet::tokenize::HostTokenizer for InstanceState {
    async fn tokenize(
        &mut self,
        this: Resource<Tokenizer>,
        text: String,
    ) -> anyhow::Result<Vec<u32>> {
        let tokenizer = self.table().get(&this)?;
        Ok(tokenizer.inner.encode_with_special_tokens(&text))
    }

    async fn detokenize(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> anyhow::Result<String> {
        let tokenizer = self.table().get(&this)?;
        tokenizer.inner.decode(&tokens).map_err(Into::into)
    }

    async fn get_vocabs(
        &mut self,
        this: Resource<Tokenizer>,
    ) -> anyhow::Result<(Vec<u32>, Vec<Vec<u8>>)> {
        let tokenizer = self.table().get(&this)?;
        Ok(tokenizer.inner.get_vocabs())
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
