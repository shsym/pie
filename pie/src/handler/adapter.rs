use crate::bindings;
use crate::handler::forward::ForwardPass;
use crate::instance::InstanceState;
use crate::model::ResourceId;
use crate::model_old::{Command, ManagedTypes};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings::pie::inferlet::adapter::Host for InstanceState {
    async fn set_adapter(
        &mut self,
        pass: Resource<ForwardPass>,
        adapter_ptr: ResourceId,
    ) -> anyhow::Result<()> {
        todo!()
    }
}
