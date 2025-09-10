use crate::handler::forward::ForwardPass;
use crate::instance::InstanceState;
use crate::resource::ResourceId;
use crate::{bindings, resource};
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl bindings::pie::inferlet::adapter::Host for InstanceState {
    async fn set_adapter(
        &mut self,
        pass: Resource<ForwardPass>,
        adapter_ptr: ResourceId,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.service_id;
        let phys_adapter_ptr = self
            .translate_resource_ptr(svc_id, resource::ADAPTER_TYPE_ID, adapter_ptr)
            .ok_or(anyhow::format_err!(
                "Failed to translate adapter with ptr: {:?}",
                adapter_ptr
            ))?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.adapter = Some(phys_adapter_ptr);

        Ok(())
    }
}
