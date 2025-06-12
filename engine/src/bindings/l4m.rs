use crate::instance::InstanceState;
use crate::l4m::{Command, ManagedTypes, StreamPriority, available_models, L4m};
use crate::object::IdRepr;
use crate::tokenizer::BytePairEncoder;
use crate::{bindings, service, backend};
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

fn map_object_types(ty: bindings::wit::pie::nbi::l4m::ObjectType) -> ManagedTypes {
    match ty {
        bindings::wit::pie::nbi::l4m::ObjectType::Block => ManagedTypes::KvBlock,
        bindings::wit::pie::nbi::l4m::ObjectType::Embed => ManagedTypes::TokenEmb,
    }
}

impl bindings::wit::pie::nbi::l4m::Host for InstanceState {
    async fn get_model(&mut self, value: String) -> anyhow::Result<Option<Resource<Model>>> {
        // First check if service already exists for this model
        if let Some(service_id) = service::get_service_id(&value) {
            let model = Model {
                name: value,
                service_id,
            };
            let res = self.table().push(model)?;
            return Ok(Some(res));
        }

        // If no service exists, try to discover and connect to a backend for this model
        tracing::info!("No service found for model '{}', attempting backend discovery", value);

        // TODO: Get the engine-manager endpoint from configuration
        // For now, use the default endpoint
        let engine_manager_endpoint = std::env::var("SYMPHONY_ENGINE_MANAGER_ENDPOINT")
            .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

        match crate::backend_discovery::discover_backend_for_model(&engine_manager_endpoint, &value).await {
            Ok(endpoint) => {
                tracing::info!("Found backend for model '{}' at endpoint: {}", value, endpoint);

                // Create a new L4M service connected to this backend
                match self.create_l4m_service_for_backend(&value, &endpoint).await {
                    Ok(service_id) => {
                        let model = Model {
                            name: value,
                            service_id,
                        };
                        let res = self.table().push(model)?;
                        Ok(Some(res))
                    }
                    Err(e) => {
                        tracing::error!("Failed to create L4M service for backend: {}", e);
                        Ok(None)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Backend discovery failed for model '{}': {}", value, e);
                Ok(None)
            }
        }
    }

    async fn get_all_models(&mut self) -> anyhow::Result<Vec<String>> {
        tracing::info!("[L4M] get_all_models() called - checking available models");
        let current_models = available_models();
        tracing::info!("[L4M] Current cached models: {:?} (count: {})", current_models, current_models.len());

        // If no models are cached, try to discover them from backends
        if current_models.is_empty() {
            tracing::info!("[L4M] Models cache is empty, starting backend discovery");

            tracing::info!("[L4M] About to call discover_available_models()");
            match crate::l4m::discover_available_models().await {
                Ok(discovered_models) => {
                    tracing::info!("Backend discovery returned {} models: {:?}", discovered_models.len(), discovered_models);
                    if !discovered_models.is_empty() {
                        tracing::info!("Setting available models to discovered models");
                        crate::l4m::set_available_models(&discovered_models);
                        let updated_models = available_models();
                        tracing::info!("After update, cached models: {:?}", updated_models);
                        return Ok(discovered_models);
                    } else {
                        tracing::warn!("Backend discovery returned empty model list");
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to discover models from backends: {}", e);
                }
            }
        } else {
            tracing::info!("Returning cached models: {:?}", current_models);
        }

        // Return the current models (which may have been updated by discovery)
        let final_models = available_models();
        tracing::info!("Returning final models: {:?}", final_models);
        Ok(final_models)
    }
}

impl bindings::wit::pie::nbi::l4m::HostModel for InstanceState {
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
        ty: bindings::wit::pie::nbi::l4m::ObjectType,
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
        ty: bindings::wit::pie::nbi::l4m::ObjectType,
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
        priority: bindings::wit::pie::nbi::l4m::StreamPriority,
    ) -> Result<(), wasmtime::Error> {
        let service_id = self.table().get(&model)?.service_id;

        Command::SetStreamPriority {
            inst_id: self.id(),
            stream_id,
            priority: match priority {
                bindings::wit::pie::nbi::l4m::StreamPriority::High => StreamPriority::High,
                bindings::wit::pie::nbi::l4m::StreamPriority::Normal => StreamPriority::Normal,
                bindings::wit::pie::nbi::l4m::StreamPriority::Low => StreamPriority::Low,
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

impl bindings::wit::pie::nbi::l4m::HostTokenizer for InstanceState {
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

impl bindings::wit::pie::nbi::l4m::HostSampleTopKResult for InstanceState {
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

impl bindings::wit::pie::nbi::l4m::HostSynchronizationResult for InstanceState {
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

impl InstanceState {
    /// Create a new L4M service connected to the specified backend endpoint
    async fn create_l4m_service_for_backend(&mut self, model_name: &str, endpoint: &str) -> anyhow::Result<usize> {
        tracing::info!("Creating L4M service for model '{}' with backend at '{}'", model_name, endpoint);

        // Create a ZMQ backend connection to the endpoint
        let backend = backend::ZmqBackend::bind(endpoint).await
            .map_err(|e| anyhow::anyhow!("Failed to connect to backend at {}: {}", endpoint, e))?;

        // Create a new L4M service with this backend
        let l4m_service = L4m::new(backend).await;

        // Add the service dynamically to the controller
        match service::add_service_runtime(model_name, l4m_service) {
            Ok(_) => {
                tracing::info!("Successfully added L4M service for model '{}'", model_name);

                // Get the service ID for the newly added service
                service::get_service_id(model_name)
                    .ok_or_else(|| anyhow::anyhow!("Failed to get service ID for newly added model '{}'", model_name))
            }
            Err(e) => {
                Err(anyhow::anyhow!("Failed to add L4M service to controller: {:?}", e))
            }
        }
    }
}
