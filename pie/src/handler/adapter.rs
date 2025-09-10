use crate::handler::Handler;
use crate::handler::core::{Blob, BlobResult, Queue};
use crate::handler::forward::ForwardPass;
use crate::instance::InstanceState;
use crate::model::{DownloadAdapterRequest, UploadAdapterRequest};
use crate::resource::ResourceId;
use crate::{bindings, model, resource};
use bytes::Bytes;
use tokio::sync::oneshot;
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

    async fn download_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter_ptr: ResourceId,
        name: String,
    ) -> anyhow::Result<Resource<BlobResult>> {
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        let stream_id = self.ctx().table.get(&queue)?.stream_id;
        let inst_id = self.id();
        let phys_adapter_ptr = self
            .translate_resource_ptr(svc_id, resource::ADAPTER_TYPE_ID, adapter_ptr)
            .ok_or(anyhow::format_err!(
                "Failed to translate adapter with ptr: {:?}",
                adapter_ptr
            ))?;

        let req = DownloadAdapterRequest {
            adapter_ptr: phys_adapter_ptr,
            name,
        };
        let data = Bytes::from(rmp_serde::to_vec_named(&req)?);

        let (tx, rx) = oneshot::channel();

        model::Command::Submit {
            inst_id,
            cmd_queue_id: stream_id,
            handler: Handler::DownloadAdapter,
            data,
            response: Some(tx),
        }
        .dispatch(svc_id)?;

        let res = BlobResult {
            receiver: rx,
            result: None,
            done: false,
        };

        Ok(self.ctx().table.push(res)?)
    }

    async fn upload_adapter(
        &mut self,
        queue: Resource<Queue>,
        adapter_ptr: ResourceId,
        name: String,
        blob: Resource<Blob>,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        let stream_id = self.ctx().table.get(&queue)?.stream_id;
        let inst_id = self.id();
        let phys_adapter_ptr = self
            .translate_resource_ptr(svc_id, resource::ADAPTER_TYPE_ID, adapter_ptr)
            .ok_or(anyhow::format_err!(
                "Failed to translate adapter with ptr: {:?}",
                adapter_ptr
            ))?;
        let blob = self.ctx().table.get(&blob)?;
        let req = UploadAdapterRequest {
            adapter_ptr: phys_adapter_ptr,
            name,
            adapter_data: blob.data.to_vec(),
        };
        let data = Bytes::from(rmp_serde::to_vec_named(&req)?);

        model::Command::Submit {
            inst_id,
            cmd_queue_id: stream_id,
            handler: Handler::UploadAdapter,
            data,
            response: None,
        }
        .dispatch(svc_id)?;

        Ok(())
    }
}
