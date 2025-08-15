use crate::bindings;
use crate::bindings::core::Queue;
use crate::bindings::forward_text::DistributionResult;
use crate::bindings::pie::inferlet::allocate::ObjectId;
use crate::instance::InstanceState;
use crate::model::Command;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings::pie::inferlet::optimize::Host for InstanceState {
    async fn create_adapter(
        &mut self,
        queue: Resource<Queue>,
        name: String,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::CreateAdapter {
            inst_id,
            stream_id: q.stream_id,
            name,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn destroy_adapter(
        &mut self,
        queue: Resource<Queue>,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::DestroyAdapter {
            inst_id,
            stream_id: q.stream_id,
            name,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn update_adapter(
        &mut self,
        queue: Resource<Queue>,
        name: String,
        scores: Vec<f32>,
        seeds: Vec<i64>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::UpdateAdapter {
            inst_id,
            stream_id: q.stream_id,
            name,
            scores,
            seeds,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn forward_with_mutation(
        &mut self,
        queue: Resource<Queue>,
        adapter: String,
        seed: i64,
        last_kv_page_len: u32,
        kv_page_ids: Vec<ObjectId>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
        mask: Vec<Vec<u32>>,
        output_indices: Vec<u32>,
    ) -> anyhow::Result<Option<Resource<DistributionResult>>> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;

        if output_indices.is_empty() {
            Command::ForwardWithMutation {
                inst_id,
                stream_id: q.stream_id,
                adapter,
                seed,
                kv_page_last_len: last_kv_page_len,
                kv_pages: kv_page_ids,
                text: tokens,
                positions,
                mask,
                output_indices,
                handle: None,
            }
            .dispatch(q.service_id)?;

            Ok(None)
        } else {
            let (tx, rx) = oneshot::channel();

            Command::ForwardWithMutation {
                inst_id,
                stream_id: q.stream_id,
                adapter,
                seed,
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

            Ok(Some(self.table().push(res)?))
        }
    }
}
