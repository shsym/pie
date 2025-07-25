use crate::bindings2;
use crate::bindings2::model;
use crate::instance::InstanceState;
use crate::l4m::{Command, ManagedTypes};
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::IoView;

type ObjectId = bindings2::pie::inferlet::allocate::ObjectId;

impl bindings2::pie::inferlet::allocate::Host for InstanceState {
    async fn get_kv_page_size(&mut self, queue: Resource<model::Queue>) -> anyhow::Result<u32> {
        let q = self.table().get(&queue)?;
        let (tx, rx) = oneshot::channel();
        Command::GetBlockSize { handle: tx }.dispatch(q.service_id)?;
        let block_size = rx.await?;
        Ok(block_size)
    }

    async fn get_all_exported_kv_pages(
        &mut self,
        queue: Resource<model::Queue>,
    ) -> anyhow::Result<Vec<(String, u32)>> {
        let q = self.table().get(&queue)?;
        let (tx, rx) = oneshot::channel();
        Command::GetAllExportedBlocks { handle: tx }.dispatch(q.service_id)?;
        rx.await.map_err(Into::into)
    }

    async fn allocate_kv_pages(
        &mut self,
        queue: Resource<model::Queue>,
        kv_page_ids: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Allocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::KvBlock,
            ids: kv_page_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn deallocate_kv_pages(
        &mut self,
        queue: Resource<model::Queue>,
        kv_page_ids: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Deallocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::KvBlock,
            ids: kv_page_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn allocate_embeds(
        &mut self,
        queue: Resource<model::Queue>,
        embed_ids: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Allocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::TokenEmb,
            ids: embed_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn deallocate_embeds(
        &mut self,
        queue: Resource<model::Queue>,
        embed_ids: Vec<ObjectId>,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::Deallocate {
            inst_id,
            stream_id: q.stream_id,
            ty: ManagedTypes::TokenEmb,
            ids: embed_ids,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn copy_kv_page(
        &mut self,
        queue: Resource<model::Queue>,
        src_kv_page_id: ObjectId,
        dst_kv_page_id: ObjectId,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::CopyBlock {
            inst_id,
            stream_id: q.stream_id,
            src_block: src_kv_page_id,
            dst_block: dst_kv_page_id,
            src_token_offset: src_offset,
            dst_token_offset: dst_offset,
            size,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }

    async fn export_kv_pages(
        &mut self,
        queue: Resource<model::Queue>,
        src_kv_page_ids: Vec<ObjectId>,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::ExportBlocks {
            inst_id,
            blocks: src_kv_page_ids,
            resource_name: name,
        }
        .dispatch(q.service_id)?;

        Ok(())
    }

    async fn import_kv_pages(
        &mut self,
        queue: Resource<model::Queue>,
        dst_kv_page_ids: Vec<ObjectId>,
        name: String,
    ) -> anyhow::Result<()> {
        let inst_id = self.id();
        let q = self.table().get(&queue)?;
        Command::ImportBlocks {
            inst_id,
            blocks: dst_kv_page_ids,
            resource_name: name,
        }
        .dispatch(q.service_id)?;
        Ok(())
    }
}
