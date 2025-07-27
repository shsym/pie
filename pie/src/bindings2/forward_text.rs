use crate::bindings2;
use crate::bindings2::core;
use crate::instance::InstanceState;
use crate::l4m::Command;
use crate::object::IdRepr;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug)]
pub struct DistributionResult {
    receiver: Option<oneshot::Receiver<Vec<(Vec<u32>, Vec<f32>)>>>,
    result: Option<Vec<(Vec<u32>, Vec<f32>)>>,
    done: bool,
}

#[async_trait]
impl Pollable for DistributionResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        if let Some(rx) = self.receiver.take() {
            if let Ok(result) = rx.await {
                self.result = Some(result);
            }
        }
        self.done = true;
    }
}

impl bindings2::pie::inferlet::forward_text::Host for InstanceState {
    async fn forward_text(
        &mut self,
        queue: Resource<core::Queue>,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        output_indices: Vec<u32>,
    ) -> anyhow::Result<Resource<DistributionResult>> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;

        let (tx, rx) = oneshot::channel();

        Command::ForwardText {
            inst_id,
            stream_id: q.stream_id,
            last_block_len: last_kv_page_len,
            context: kv_page_ids,
            text: tokens,
            positions,
            output_indices,
            handle: tx,
        }
        .dispatch(q.service_id)?;

        let res = DistributionResult {
            receiver: Some(rx),
            result: None,
            done: false,
        };

        Ok(self.table().push(res)?)
    }
}

impl bindings2::pie::inferlet::forward_text::HostDistributionResult for InstanceState {
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
        Ok(self.table().get_mut(&this)?.result.clone())
    }

    async fn drop(&mut self, this: Resource<DistributionResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
