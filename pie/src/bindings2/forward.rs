use crate::bindings2;
use crate::bindings2::core;
use crate::instance::InstanceState;
use crate::model::Command;
use crate::object::IdRepr;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

impl bindings2::pie::inferlet::forward::Host for InstanceState {
    async fn get_all_adapters(
        &mut self,
        _queue: Resource<core::Queue>,
    ) -> anyhow::Result<Vec<String>> {
        // Placeholder
        Ok(vec![])
    }

    async fn forward(
        &mut self,
        queue: Resource<core::Queue>,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Forward {
            inst_id,
            stream_id: q.stream_id,
            kv_page_last_len: last_kv_page_len,
            kv_pages: kv_page_ids,
            input_embeds: input_emb_ids,
            output_embeds: output_emb_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn forward_with_adapter(
        &mut self,
        queue: Resource<core::Queue>,
        _adapter: String,
        last_kv_page_len: u32,
        kv_page_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Forward {
            inst_id,
            stream_id: q.stream_id,
            kv_page_last_len: last_kv_page_len,
            kv_pages: kv_page_ids,
            input_embeds: input_emb_ids,
            output_embeds: output_emb_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn mask_kv_page(
        &mut self,
        queue: Resource<core::Queue>,
        kv_page_id: IdRepr,
        mask: Vec<bool>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::MaskKvPage {
            inst_id,
            stream_id: q.stream_id,
            kv_page: kv_page_id,
            mask,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }
}
