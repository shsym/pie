use crate::instance::InstanceState;
use crate::l4m::{Command, ManagedTypes, StreamPriority, available_models};
use crate::object::IdRepr;
use crate::tokenizer::BytePairEncoder;
use crate::{bindings, service};
use std::sync::Arc;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, IoView, Pollable, subscribe};
#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub service_id: usize,
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
    receiver: Option<oneshot::Receiver<()>>,
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

        if let Some(receiver) = self.receiver.take() {
            let _ = receiver.await.unwrap();
        }
        self.done = true;
    }
}

#[derive(Debug)]
pub struct Cache {
    block_size: Option<u32>,
    tokenizer: Option<Tokenizer>,
}

fn map_object_types(ty: bindings::wit::symphony::nbi::l4m::ObjectType) -> ManagedTypes {
    match ty {
        bindings::wit::symphony::nbi::l4m::ObjectType::Block => ManagedTypes::KvBlock,
        bindings::wit::symphony::nbi::l4m::ObjectType::Embed => ManagedTypes::TokenEmb,
    }
}

impl bindings::wit::symphony::nbi::l4m::Host for InstanceState {
    async fn get_model(&mut self, value: String) -> anyhow::Result<Option<Resource<Model>>> {
        if let Some(service_id) = service::get_service_id(&value) {
            let model = Model {
                name: value,
                service_id,
            };
            let res = self.table().push(model)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        Ok(available_models().to_vec())
    }
}

impl bindings::wit::symphony::nbi::l4m::HostModel for InstanceState {
    async fn get_block_size(&mut self, model: Resource<Model>) -> Result<u32, wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        let (tx, rx) = oneshot::channel();
        Command::GetBlockSize { handle: tx }.dispatch(service_id)?;
        let block_size = rx.await?;
        Ok(block_size)
    }

    async fn get_tokenizer(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Resource<Tokenizer>, wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        let (tx, rx) = oneshot::channel();
        Command::GetTokenizer { handle: tx }.dispatch(service_id)?;
        let inner = rx.await?;
        let res = self.table().push(Tokenizer { inner })?;
        Ok(res)
    }

    async fn get_all_adapters(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<String>, wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Ok(vec![])
    }

    async fn get_all_exported_blocks(
        &mut self,
        model: Resource<Model>,
    ) -> Result<Vec<(String, u32)>, wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        let (tx, rx) = oneshot::channel();

        Command::GetAllExportedBlocks { handle: tx }.dispatch(service_id)?;

        let result = rx
            .await
            .or(Err(wasmtime::Error::msg("GetAllExportedBlocks failed")))?;

        Ok(result)
    }

    async fn allocate(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        ty: bindings::wit::symphony::nbi::l4m::ObjectType,
        object_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::Allocate {
            inst_id: self.id(),
            stream_id,
            ty: map_object_types(ty),
            ids: object_ids,
        }
        .dispatch(service_id)?;

        Ok(())
    }

    async fn deallocate(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        ty: bindings::wit::symphony::nbi::l4m::ObjectType,
        object_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::Deallocate {
            inst_id: self.id(),
            stream_id,
            ty: map_object_types(ty),
            ids: object_ids,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn fill_block(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        last_block_len: u32,
        context_block_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::FillBlock {
            inst_id: self.id(),
            stream_id,
            last_block_len: last_block_len,
            context: context_block_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn fill_block_with_adapter(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        adapter: String,
        last_block_len: u32,
        context_block_ids: Vec<IdRepr>,
        input_emb_ids: Vec<IdRepr>,
        output_emb_ids: Vec<IdRepr>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::FillBlock {
            inst_id: self.id(),
            stream_id,
            last_block_len: last_block_len,
            context: context_block_ids,
            inputs: input_emb_ids,
            outputs: output_emb_ids,
        }
        .dispatch(service_id)?;
        Ok(())
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
        let service_id = self.table().get(&model)?.service_id;

        Command::CopyBlock {
            inst_id: self.id(),
            stream_id,
            src_block: src_block_id,
            dst_block: dst_block_id,
            src_token_offset: src_offset,
            dst_token_offset: dst_offset,
            size,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn mask_block(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        block_id: IdRepr,
        mask: Vec<bool>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::MaskBlock {
            inst_id: self.id(),
            stream_id,
            block: block_id,
            mask,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn export_blocks(
        &mut self,
        model: Resource<Model>,
        src_block_ids: Vec<IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::ExportBlocks {
            inst_id: self.id(),
            blocks: src_block_ids,
            resource_name: name,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn import_blocks(
        &mut self,
        model: Resource<Model>,
        dst_block_ids: Vec<IdRepr>,
        name: String,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::ImportBlocks {
            inst_id: self.id(),
            blocks: dst_block_ids,
            resource_name: name,
        }
        .dispatch(service_id)?;
        Ok(())
    }

    async fn embed_text(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        emb_ids: Vec<IdRepr>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::EmbedText {
            inst_id: self.id(),
            stream_id,
            embs: emb_ids,
            text: tokens,
            positions,
        }
        .dispatch(service_id)?;
        Ok(())
    }
    //
    // async fn decode_token_dist(
    //     &mut self,
    //     model: Resource<Model>,
    //     stream_id: u32,
    //     emb_ids: Vec<IdRepr>,
    //     dist_ids: Vec<IdRepr>,
    // ) -> Result<(), wasmtime::Error> {
    //     let service_id = self.table().get(&model)?.service_id;
    //
    //     Command::DecodeTokenDist {
    //         inst_id: self.id(),
    //         stream_id,
    //         embs: emb_ids,
    //         dists: dist_ids,
    //     }
    //     .dispatch(service_id)?;
    //
    //     Ok(())
    // }

    async fn sample_top_k(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        emb_ids: Vec<IdRepr>,
        k: u32,
    ) -> Result<Resource<SampleTopKResult>, wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        let mut receivers = Vec::with_capacity(emb_ids.len());
        for i in 0..emb_ids.len() {
            let (tx, rx) = oneshot::channel();
            receivers.push(rx);
            Command::SampleTopK {
                inst_id: self.id(),
                stream_id,
                emb_id: emb_ids[i],
                k,
                handle: tx,
            }
            .dispatch(service_id)?;
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
        let service_id = self.table().get(&model)?.service_id;

        let (tx, rx) = oneshot::channel();
        Command::Synchronize {
            inst_id: self.id(),
            stream_id,
            handle: tx,
        }
        .dispatch(service_id)?;

        let result = SynchronizationResult {
            receiver: Some(rx),
            done: false,
        };

        let res = self.table().push(result)?;
        Ok(res)
    }

    async fn set_stream_priority(
        &mut self,
        model: Resource<Model>,
        stream_id: u32,
        priority: bindings::wit::symphony::nbi::l4m::StreamPriority,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::SetStreamPriority {
            inst_id: self.id(),
            stream_id,
            priority: match priority {
                bindings::wit::symphony::nbi::l4m::StreamPriority::High => StreamPriority::High,
                bindings::wit::symphony::nbi::l4m::StreamPriority::Normal => StreamPriority::Normal,
                bindings::wit::symphony::nbi::l4m::StreamPriority::Low => StreamPriority::Low,
            },
        }
        .dispatch(service_id)?;

        Ok(())
    }

    async fn drop(&mut self, model: Resource<Model>) -> anyhow::Result<(), wasmtime::Error> {
        let _ = self.table().delete(model)?;
        Ok(())
    }
}

impl bindings::wit::symphony::nbi::l4m::HostTokenizer for InstanceState {
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

impl bindings::wit::symphony::nbi::l4m::HostSampleTopKResult for InstanceState {
    async fn pollable(
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

impl bindings::wit::symphony::nbi::l4m::HostSynchronizationResult for InstanceState {
    async fn pollable(
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
