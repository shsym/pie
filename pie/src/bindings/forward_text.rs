use crate::bindings;
use crate::bindings::core;
use crate::instance::InstanceState;
use crate::model::Command;
use crate::object::IdRepr;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug)]
pub struct DistributionResult {
    receiver: oneshot::Receiver<Vec<(Vec<u32>, Vec<f32>)>>,
    result: Vec<(Vec<u32>, Vec<f32>)>,
    done: bool,
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

impl bindings::pie::inferlet::forward_text::Host for InstanceState {
    async fn forward_text(
        &mut self,
        queue: Resource<core::Queue>,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Vec<u32>>,
        output_indices: Vec<u32>,
    ) -> anyhow::Result<Resource<DistributionResult>> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;

        let (tx, rx) = oneshot::channel();

        Command::ForwardText {
            inst_id,
            stream_id: q.stream_id,
            kv_page_last_len: last_kv_page_len,
            kv_pages: kv_page_ids,
            text: tokens,
            positions,
            mask,
            output_indices,
            handle: Some(tx),
        }
        .dispatch(q.service_id)?;

        let res = DistributionResult {
            receiver: rx,
            result: vec![],
            done: false,
        };

        Ok(self.table().push(res)?)
    }

    async fn forward_text_no_output(
        &mut self,
        queue: Resource<core::Queue>,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Vec<u32>>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;

        Command::ForwardText {
            inst_id,
            stream_id: q.stream_id,
            kv_page_last_len: last_kv_page_len,
            kv_pages: kv_page_ids,
            text: tokens,
            positions,
            mask,
            output_indices: vec![],
            handle: None,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }
}

impl bindings::pie::inferlet::forward_text::HostDistributionResult for InstanceState {
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
