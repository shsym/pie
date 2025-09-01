use crate::bindings;
use crate::handler::core::Queue;
use crate::instance::InstanceState;
use crate::model::ResourceId;

use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug)]
pub struct ForwardPass {}

#[derive(Debug)]
pub struct DistributionResult {
    pub receiver: oneshot::Receiver<Vec<(Vec<u32>, Vec<f32>)>>,
    pub result: Vec<(Vec<u32>, Vec<f32>)>,
    pub done: bool,
}

#[derive(Debug)]
pub struct TokenResult {
    pub receiver: oneshot::Receiver<Vec<u32>>,
    pub result: Vec<u32>,
    pub done: bool,
}

#[async_trait]
impl Pollable for DistributionResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let res = (&mut self.receiver).await.unwrap();
        self.result = res;
        self.done = true;
    }
}

#[async_trait]
impl Pollable for TokenResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let res = (&mut self.receiver).await.unwrap();
        self.result = res;
        self.done = true;
    }
}

impl bindings::pie::inferlet::forward::Host for InstanceState {
    async fn execute(
        &mut self,
        queue: Resource<Queue>,
        pass: Resource<ForwardPass>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;

        Ok(())
    }

    async fn set_input_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        emb_ptrs: Vec<ResourceId>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    async fn set_input_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        input_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    async fn request_output_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        emb_ptrs: Vec<ResourceId>,
        indices: Vec<u32>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    async fn request_output_distributions(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
    ) -> anyhow::Result<Resource<DistributionResult>> {
        todo!()
    }

    async fn request_output_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        sampler: String,
    ) -> anyhow::Result<Resource<TokenResult>> {
        todo!()
    }

    async fn apply_mask(
        &mut self,
        pass: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    async fn use_kv_cache(
        &mut self,
        pass: Resource<ForwardPass>,
        kv_page_ptrs: Vec<ResourceId>,
        last_kv_page_len: u32,
    ) -> anyhow::Result<()> {
        todo!()
    }
}

impl bindings::pie::inferlet::forward::HostForwardPass for InstanceState {
    async fn new(&mut self) -> anyhow::Result<Resource<ForwardPass>> {
        todo!()
    }
    async fn drop(&mut self, rep: Resource<ForwardPass>) -> anyhow::Result<()> {
        todo!()
    }
}

impl bindings::pie::inferlet::forward::HostDistributionResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<DistributionResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<DistributionResult>,
    ) -> anyhow::Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let result = self.table().get_mut(&this)?;

        if result.done {
            Ok(Some(result.result.clone()))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<DistributionResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::forward::HostTokenResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<TokenResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.table(), this)
    }

    async fn get(&mut self, this: Resource<TokenResult>) -> anyhow::Result<Option<Vec<u32>>> {
        let result = self.table().get_mut(&this)?;

        if result.done {
            Ok(Some(result.result.clone()))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<TokenResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
