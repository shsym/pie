use crate::handler::Handler;
use crate::handler::core::Queue;
use crate::instance::InstanceState;
use crate::model::EmbedImageRequest;
use crate::resource::ResourceId;
use crate::{bindings, model, resource};
use bytes::Bytes;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

impl bindings::pie::inferlet::image::Host for InstanceState {
    async fn embed_image(
        &mut self,
        queue: Resource<Queue>,
        mut emb_ptrs: Vec<ResourceId>,
        image_blob: Vec<u8>,
        position_offset: u32,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&queue)?.service_id;
        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self
                .translate_resource_ptr(svc_id, resource::EMBED_TYPE_ID, *emb_ptr)
                .ok_or_else(|| {
                    anyhow::format_err!(
                        "Failed to translate image embedding with ptr: {:?}",
                        emb_ptr
                    )
                })?;
            Ok::<_, anyhow::Error>(())
        })?;

        let inst_id = self.id();
        let q = self.ctx().table.get(&queue)?;

        let req = EmbedImageRequest {
            embed_ptrs: emb_ptrs,
            image_blob,
            position_offset,
        };

        let data = Bytes::from(rmp_serde::to_vec_named(&req)?);

        model::Command::Submit {
            inst_id,
            cmd_queue_id: q.stream_id,
            handler: Handler::EmbedImage,
            data,
            response: None,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn calculate_embed_size(
        &mut self,
        _queue: Resource<Queue>,
        _image_width: u32,
        _image_height: u32,
    ) -> anyhow::Result<u32> {
        // Placeholder implementation
        // This would typically involve a call to the model service
        // to determine the number of tokens/embeddings for a given image size.
        Ok(1024) // e.g., return a fixed size for now
    }
}
