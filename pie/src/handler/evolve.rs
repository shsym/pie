use crate::handler::Handler;
use crate::handler::core::Queue;
use crate::handler::forward::ForwardPass;
use crate::instance::InstanceState;
use crate::model::{InitializeAdapterRequest, UpdateAdapterRequest};
use crate::resource::ResourceId;
use crate::{bindings, model, resource};
use bytes::Bytes;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl bindings::pie::inferlet::evolve::Host for InstanceState {
    async fn set_adapter_seed(
        &mut self,
        pass: Resource<ForwardPass>,
        seed: i64,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;

        if pass.adapter.is_some() {
            pass.adapter_seed = Some(seed);
        } else {
            anyhow::bail!("Adapter not set");
        }

        Ok(())
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
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        let phys_adapter_ptr = self
            .translate_resource_ptr(svc_id, resource::ADAPTER_TYPE_ID, adapter_ptr)
            .ok_or_else(|| {
                anyhow::format_err!("Failed to translate adapter with ptr: {:?}", adapter_ptr)
            })?;

        let inst_id = self.id();
        let q = self.ctx().table.get(&queue)?;

        let req = InitializeAdapterRequest {
            adapter_ptr: phys_adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        };
        let data = Bytes::from(rmp_serde::to_vec_named(&req)?);

        model::Command::Submit {
            inst_id,
            cmd_queue_id: q.stream_id,
            handler: Handler::InitializeAdapter,
            data,
            response: None,
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
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        let phys_adapter_ptr = self
            .translate_resource_ptr(svc_id, resource::ADAPTER_TYPE_ID, adapter_ptr)
            .ok_or_else(|| {
                anyhow::format_err!("Failed to translate adapter with ptr: {:?}", adapter_ptr)
            })?;

        let inst_id = self.id();
        let q = self.ctx().table.get(&queue)?;

        let req = UpdateAdapterRequest {
            adapter_ptr: phys_adapter_ptr,
            scores,
            seeds,
            max_sigma,
        };
        let data = Bytes::from(rmp_serde::to_vec_named(&req)?);

        model::Command::Submit {
            inst_id,
            cmd_queue_id: q.stream_id,
            handler: Handler::UpdateAdapter,
            data,
            response: None,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }
}
