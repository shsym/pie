use crate::bindings;
use crate::handler::core::Queue;
use crate::handler::forward::ForwardPass;
use crate::instance::InstanceState;
use crate::model::ResourceId;
use crate::model_old::{Command, ManagedTypes};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings::pie::inferlet::evolve::Host for InstanceState {
    async fn set_adapter_seed(
        &mut self,
        pass: Resource<ForwardPass>,
        seed: i64,
    ) -> anyhow::Result<()> {
        todo!()
    }
    async fn initialize_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter_ptr: ResourceId,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::InitializeAdapter {
            inst_id,
            stream_id: q.stream_id,
            adapter: adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn update_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter_ptr: ResourceId,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::UpdateAdapter {
            inst_id,
            stream_id: q.stream_id,
            adapter: adapter_ptr,
            scores,
            seeds,
            max_sigma,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }
}
