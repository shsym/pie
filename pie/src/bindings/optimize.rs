use crate::bindings;
use crate::bindings::core::Queue;
use crate::bindings::forward_text::DistributionResult;
use crate::bindings::pie::inferlet::allocate::ObjectId;
use crate::instance::InstanceState;
use crate::model::{Command, ManagedTypes};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings::pie::inferlet::optimize::Host for InstanceState {
    async fn allocate_adapters(
        &mut self,
        queue: Resource<Queue>,
        adapters: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Allocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::Adapter,
            ids: adapters,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn deallocate_adapters(
        &mut self,
        queue: Resource<Queue>,
        adapters: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Deallocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::Adapter,
            ids: adapters,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn export_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter: ObjectId,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Export {
            inst_id,
            ty: ManagedTypes::Adapter,
            ids: vec![adapter],
            resource_name: name,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn unexport_adapter(
        &mut self,
        queue: Resource<Queue>,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Unexport {
            inst_id,
            ty: ManagedTypes::Adapter,
            resource_name: name,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn import_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter: ObjectId,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Import {
            inst_id,
            ty: ManagedTypes::Adapter,
            ids: vec![adapter],
            resource_name: name,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn initialize_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter: ObjectId,
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
            adapter,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn mutate_adapters(
        &mut self,
        queue: Resource<Queue>,
        adapters: Vec<ObjectId>,
        parent: ObjectId,
        seeds: Vec<i64>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::MutateAdapters {
            inst_id,
            stream_id: q.stream_id,
            adapters,
            parent,
            seeds,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn update_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter: ObjectId,
        scores: Vec<f32>,
        seeds: Vec<i64>,
        max_sigma: f32,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::UpdateAdapter {
            inst_id,
            stream_id: q.stream_id,
            adapter,
            scores,
            seeds,
            max_sigma,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn forward_with_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter: ObjectId,
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
            Command::ForwardWithAdapter {
                inst_id,
                stream_id: q.stream_id,
                adapter,
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

            Command::ForwardWithAdapter {
                inst_id,
                stream_id: q.stream_id,
                adapter,
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
