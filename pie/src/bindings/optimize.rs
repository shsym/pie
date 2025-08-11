use crate::bindings;
use crate::bindings::core;
use crate::bindings::pie::inferlet::optimize::{DistributionResult, ObjectId, Queue};
use crate::instance::InstanceState;
use crate::model::Command;
use crate::object::IdRepr;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;

impl bindings::pie::inferlet::optimize::Host for InstanceState {
    async fn create_adapter(&mut self, name: String) -> anyhow::Result<()> {
        todo!()
    }

    async fn destroy_adapter(&mut self, name: String) -> anyhow::Result<()> {
        todo!()
    }

    async fn update_adapter(
        &mut self,
        name: String,
        score_list: Vec<f32>,
        seed_list: Vec<i64>,
    ) -> anyhow::Result<()> {
        todo!()
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
        todo!()
    }
}
