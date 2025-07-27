use crate::bindings2;
use crate::bindings2::core;
use crate::instance::InstanceState;
use crate::model::Command;
use crate::object::IdRepr;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings2::pie::inferlet::input_text::Host for InstanceState {
    async fn embed_text(
        &mut self,
        queue: Resource<core::Queue>,
        emb_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::EmbedText {
            inst_id,
            stream_id: q.stream_id,
            embs: emb_ids,
            text: tokens,
            positions,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }
}
