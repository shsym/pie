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
    receivers: Vec<oneshot::Receiver<(Vec<u32>, Vec<f32>)>>,
    results: Vec<(Vec<u32>, Vec<f32>)>,
    done: bool,
}

#[async_trait]
impl Pollable for DistributionResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        for rx in self.receivers.drain(..) {
            if let Ok(result) = rx.await {
                self.results.push(result);
            }
        }
        self.done = true;
    }
}

impl bindings2::pie::inferlet::output_text::Host for InstanceState {
    async fn get_next_token_distribution(
        &mut self,
        queue: Resource<core::Queue>,
        emb_ids: Vec<IdRepr>,
    ) -> anyhow::Result<Resource<DistributionResult>> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        let mut receivers = Vec::with_capacity(emb_ids.len());
        for emb_id in emb_ids {
            let (tx, rx) = oneshot::channel();
            receivers.push(rx);
            // Assuming K=1 is desired for next-token prediction. The new WIT doesn't specify K.
            // This might need adjustment. For now, using a default K or adapting the command.
            Command::SampleTopK {
                inst_id,
                stream_id: q.stream_id,
                emb_id,
                k: 1, // Defaulting K to 1 for next token.
                handle: tx,
            }
            .dispatch(q.service_id)?;
        }

        let dist_result = DistributionResult {
            receivers,
            results: Vec::new(),
            done: false,
        };
        Ok(self.table().push(dist_result)?)
    }
}

impl bindings2::pie::inferlet::output_text::HostDistributionResult for InstanceState {
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
            Ok(Some(result.results.clone()))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<DistributionResult>) -> anyhow::Result<()> {
        self.table().delete(this)?;
        Ok(())
    }
}
