use crate::service::l4m::{LocalStreamId, StreamPriority};
use crate::instance::InstanceState;
use crate::object::IdRepr;
use crate::tokenizer::BytePairEncoder;
use crate::{bindings, service, object};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::{DynPollable, IoView, Pollable, subscribe};

#[derive(Debug, Clone)]
pub struct Model {
    name: String,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    inner: Arc<BytePairEncoder>,
}

#[derive(Debug)]
pub struct SampleTopKResult {
    receivers: Vec<oneshot::Receiver<(Vec<u32>, Vec<f32>)>>,
    results: Vec<(Vec<u32>, Vec<f32>)>,
    done: bool,
}
#[derive(Debug)]
pub struct SynchronizationResult {
    receiver: oneshot::Receiver<()>,
    done: bool,
}

#[async_trait]
impl Pollable for SampleTopKResult {
    async fn ready(&mut self) {
        // if results are already computed, return
        if self.done {
            return;
        }

        //println!("SampleTopKResult Polling");
        for rx in &mut self.receivers {
            let result = rx.await.unwrap();
            self.results.push(result);
        }
        self.done = true;
    }
}

#[async_trait]
impl Pollable for SynchronizationResult {
    async fn ready(&mut self) {
        // if results are already computed, return
        if self.done {
            return;
        }

        let _ = (&self.receiver).await.unwrap();
        self.done = true;
    }
}

#[derive(Debug)]
pub struct Cache {
    block_size: Option<u32>,
    tokenizer: Option<Tokenizer>,
}

fn map_object_types(
    ty: bindings::wit::symphony::app::l4m::ObjectType,
) -> service::l4m::ManagedTypes {
    match ty {
        bindings::wit::symphony::app::l4m::ObjectType::Block => service::l4m::ManagedTypes::KvBlock,
        bindings::wit::symphony::app::l4m::ObjectType::Dist => service::l4m::ManagedTypes::TokenDist,
        bindings::wit::symphony::app::l4m::ObjectType::Embed => service::l4m::ManagedTypes::TokenEmb,
    }
}

impl bindings::wit::symphony::app::l4m::Host for InstanceState {
    async fn get_model(&mut self, value: String) -> anyhow::Result<Option<Resource<Model>>> {
        let model = Model { name: value };
        let res = self.table().push(model)?;
        Ok(Some(res))
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(vec!["default".to_string()])
    }
}

impl bindings::wit::symphony::app::l4m::HostModel for InstanceState {
    async fn get_block_size(&mut self, model: Resource<Model>) -> Result<u32, wasmtime::Error> {
        
        let model = self.table().get(&model)?;
        
        service::NamedCommand::new(model.name, service::l4m::Command::GetBlockSize);
        
        
        let (tx, rx) = oneshot::channel();
        self.send_cmd(service::l4m::Command::GetBlockSize { handle: tx })?;
        let block_size = rx.await?;
        Ok(block_size)
    }

    async fn get_tokenizer(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Resource<Tokenizer>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();
        self.send_cmd(service::l4m::Command::GetTokenizer { handle: tx })?;
        let inner = rx.await?;
        let res = self.table().push(Tokenizer { inner })?;
        Ok(res)
    }

    async fn get_all_adapters(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<String>, wasmtime::Error> {
        Ok(vec![])
    }

    async fn get_all_exported_blocks(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<(String, u32)>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();

        self.send_cmd(service::l4m::Command::GetAllExportedBlocks { handle: tx })?;

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("GetAllExportedBlocks failed")))?;

        Ok(result)
    }

    async fn allocate(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        ty: bindings::wit::symphony::app::l4m::ObjectType,
        object_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::Allocate {
            stream_id,
            ty: map_object_types(ty),
            ids: object_ids,
        })
    }

    async fn deallocate(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        ty: bindings::wit::symphony::app::l4m::ObjectType,
        object_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::Deallocate {
            stream_id,
            ty: map_object_types(ty),
            ids: object_ids,
        })
    }

    async fn fill_block(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        block_id: IdRepr,
        context_block_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::FillBlock {
            stream_id,
            block: block_id,
            context: context_block_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        })
    }

    async fn fill_block_with_adapter(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        adapter: String,
        block_id: IdRepr,
        context_block_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::FillBlock {
            stream_id,
            block: block_id,
            context: context_block_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        })
    }

    async fn copy_block(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        src_block_id: IdRepr,
        dst_block_id: IdRepr,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::CopyBlock {
            stream_id,
            src_block: src_block_id,
            dst_block: dst_block_id,
            src_token_offset: src_offset,
            dst_token_offset: dst_offset,
            size,
        })
    }

    async fn mask_block(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        block_id: IdRepr,
        mask: Vec<bool>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::MaskBlock {
            stream_id,

            block: block_id,
            mask,
        })
    }

    async fn export_blocks(
        &mut self,
        model: Resource<Model>,
        src_block_ids: Vec<IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::ExportBlocks {
            blocks: src_block_ids,
            resource_name: name,
        })
    }

    async fn import_blocks(
        &mut self,
        model: Resource<Model>,
        dst_block_ids: Vec<IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::ImportBlocks {
            blocks: dst_block_ids,
            resource_name: name,
        })
    }

    async fn embed_text(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        emb_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::EmbedText {
            stream_id,
            embs: emb_ids,
            text: tokens,
            positions,
        })
    }

    async fn decode_token_dist(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        emb_ids: Vec<IdRepr>,
        dist_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::DecodeTokenDist {
            stream_id,
            embs: emb_ids,
            dists: dist_ids,
        })
    }

    async fn sample_top_k(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        dist_ids: Vec<IdRepr>,
        k: u32,
    ) -> Result<Resource<SampleTopKResult>, wasmtime::Error> {
        let mut receivers = Vec::with_capacity(dist_ids.len());
        for i in 0..dist_ids.len() {
            let (tx, rx) = oneshot::channel();
            receivers.push(rx);
            self.send_cmd(service::l4m::Command::SampleTopK {
                stream_id,
                dist: dist_ids[i],
                k,
                handle: tx,
            })?;
        }

        let top_k_result = SampleTopKResult {
            receivers,
            results: Vec::new(),
            done: false,
        };

        let res = self.table().push(top_k_result)?;
        Ok(res)
    }

    async fn synchronize(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
    ) -> Result<Resource<SynchronizationResult>, wasmtime::Error> {
        let (tx, rx) = oneshot::channel();
        self.send_cmd(service::l4m::Command::Synchronize {
            stream_id,
            handle: tx,
        });

        let result = SynchronizationResult {
            receiver: rx,
            done: false,
        };

        let res = self.table().push(result)?;
        Ok(res)
    }

    async fn set_stream_priority(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        priority: bindings::wit::symphony::app::l4m::StreamPriority,
    ) -> Result<(), wasmtime::Error> {
        self.send_cmd(service::l4m::Command::SetStreamPriority {
            stream_id,
            priority: match priority {
                bindings::wit::symphony::app::l4m::StreamPriority::High => StreamPriority::High,
                bindings::wit::symphony::app::l4m::StreamPriority::Normal => StreamPriority::Normal,
                bindings::wit::symphony::app::l4m::StreamPriority::Low => StreamPriority::Low,
            },
        })
    }

    async fn drop(&mut self, model: Resource<Model>) -> anyhow::Result<(), wasmtime::Error> {
        let _ = self.table().delete(model)?;
        Ok(())
    }
}

impl bindings::wit::symphony::app::l4m::HostTokenizer for InstanceState {
    async fn tokenize(
        &mut self,
        this: Resource<Tokenizer>,
        text: String,
    ) -> Result<Vec<u32>, wasmtime::Error> {
        let token_ids = self
            .table()
            .get(&this)?
            .inner
            .encode_with_special_tokens(&text);
        Ok(token_ids)
    }

    async fn detokenize(
        &mut self,
        this: Resource<Tokenizer>,
        tokens: Vec<u32>,
    ) -> Result<String, wasmtime::Error> {
        let text = self.table().get(&this)?.inner.decode(&tokens)?;
        Ok(text)
    }

    async fn get_vocabs(
        &mut self,
        this: Resource<Tokenizer>,
    ) -> Result<Vec<Vec<u8>>, wasmtime::Error> {
        let vocabs = self.table().get(&this)?.inner.get_vocabs();
        Ok(vocabs)
    }

    async fn drop(&mut self, this: Resource<Tokenizer>) -> Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::wit::symphony::app::l4m::HostSampleTopKResult for InstanceState {
    async fn subscribe(
        &mut self,
        this: Resource<SampleTopKResult>,
    ) -> Result<Resource<DynPollable>, wasmtime::Error> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<SampleTopKResult>,
    ) -> anyhow::Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let result = self.table().get_mut(&this)?;
        if result.done {
            Ok(Some(result.results.clone()))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<SampleTopKResult>) -> Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}

impl bindings::wit::symphony::app::l4m::HostSynchronizationResult for InstanceState {
    async fn subscribe(
        &mut self,
        this: Resource<SynchronizationResult>,
    ) -> Result<Resource<DynPollable>, wasmtime::Error> {
        subscribe(self.table(), this)
    }

    async fn get(
        &mut self,
        this: Resource<SynchronizationResult>,
    ) -> Result<Option<bool>, wasmtime::Error> {
        let result = self.table().get_mut(&this)?;
        if result.done {
            Ok(Some(true))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<SynchronizationResult>) -> Result<(), wasmtime::Error> {
        let _ = self.table().delete(this)?;
        Ok(())
    }
}
