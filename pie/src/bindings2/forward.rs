use crate::bindings2;
use crate::bindings2::model;
use crate::instance::InstanceState;
use crate::l4m::Command;
use crate::object::IdRepr;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings2::pie::inferlet::forward::Host for InstanceState {
    async fn get_all_adapters(
        &mut self,
        _queue: Resource<model::Queue>,
    ) -> anyhow::Result<Vec<String>> {
        // Placeholder
        Ok(vec![])
    }

    async fn forward(
        &mut self,
        queue: Resource<model::Queue>,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::FillBlock {
            inst_id,
            stream_id: q.stream_id,
            last_block_len: last_kv_page_len,
            context: kv_page_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn forward_with_adapter(
        &mut self,
        queue: Resource<model::Queue>,
        _adapter: String,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::FillBlock {
            inst_id,
            stream_id: q.stream_id,
            last_block_len: last_kv_page_len,
            context: kv_page_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn mask_kv_page(
        &mut self,
        queue: Resource<model::Queue>,
        kv_page_id: IdRepr,
        mask: Vec<bool>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::MaskBlock {
            inst_id,
            stream_id: q.stream_id,
            block: kv_page_id,
            mask,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }
}
