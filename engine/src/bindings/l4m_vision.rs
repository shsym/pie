use crate::bindings;
use crate::instance::InstanceState;
use wasmtime::component::Resource;
use wasmtime_wasi::IoView;

pub struct VisionModel {
    name: String,
}

impl bindings::wit::symphony::app::l4m_vision::Host for InstanceState {
    async fn extend_model(
        &mut self,
        model: Resource<bindings::l4m::Model>,
    ) -> anyhow::Result<Option<Resource<VisionModel>>, wasmtime::Error> {
        let vision_model = VisionModel {
            name: "".to_string(),
        };

        let res = self.table().push(vision_model)?;

        Ok(Some(res))
    }
    //
    // async fn embed_image(
    //     &mut self,
    //     model: Resource<Model>,
    //     stream: u32,
    //     embs: Vec<object::IdRepr>,
    //     image_blob: Vec<u8>,
    // ) -> Result<(), wasmtime::Error> {
    //     let cmd = Command::EmbedImage {
    //         stream,
    //         embs: object::Id::map_from_repr(embs),
    //         image: "url".to_string(),
    //     };
    //
    //     self.cmd_buffer.send((self.id, cmd));
    //
    //     Ok(())
    // }
}

impl bindings::wit::symphony::app::l4m_vision::HostModel for InstanceState {
    async fn get_base_model(
        &mut self,
        model: Resource<VisionModel>,
    ) -> anyhow::Result<Resource<bindings::l4m::Model>> {
        todo!()
    }

    async fn embed_image(
        &mut self,
        model: Resource<VisionModel>,
        stream_id: u32,
        emb_ids: Vec<u32>,
        image_blob: Vec<u8>,
    ) -> anyhow::Result<()> {
        todo!()
    }

    async fn drop(&mut self, model: Resource<VisionModel>) -> anyhow::Result<()> {
        let _ = self.table().delete(model)?;
        Ok(())
    }
}
