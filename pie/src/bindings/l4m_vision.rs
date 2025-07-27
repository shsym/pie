use crate::bindings;
use crate::bindings::l4m::Model;
use crate::instance::InstanceState;
use crate::model::Command;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

pub struct VisionModel {
    name: String,
}

impl bindings::wit::pie::nbi::l4m_vision::Host for InstanceState {
    async fn embed_image(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        emb_ids: Vec<u32>,
        image_blob: Vec<u8>,
    ) -> anyhow::Result<()> {
        let service_id = self.table().get(&model)?.service_id;

        Command::EmbedImage {
            inst_id: self.id(),
            stream_id,
            embs: emb_ids,
            image_blob,
        }
        .dispatch(service_id)?;
        Ok(())
    }
}
