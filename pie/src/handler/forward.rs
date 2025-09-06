use crate::handler::core::Queue;
use crate::handler::{ForwardPassRequest, ForwardPassResponse, Handler};
use crate::instance::InstanceState;
use crate::resource::ResourceId;
use crate::{bindings, model, resource};
use bytes::Bytes;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;
use wasmtime_wasi::async_trait;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};

#[derive(Debug)]
pub struct ForwardPass {
    pub service_id: usize,
    pub stream_id: u32,
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    input_embed_ptrs: Vec<u32>,
    input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    mask: Vec<Vec<u32>>,
    kv_page_ptrs: Vec<u32>,
    kv_page_last_len: u32,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<u32>,
    output_dist_indices: Vec<u32>,
    output_embed_ptrs: Vec<u32>,
    output_embed_indices: Vec<u32>,
    sampler_temperature: f32,
    sampler_top_k: u32,
    sampler_top_p: f32,
    sampler_min_p: f32,
}

#[derive(Debug)]
pub struct ForwardPassResult {
    pub receiver: oneshot::Receiver<Bytes>,
    pub distributions: Vec<(Vec<u32>, Vec<f32>)>,
    pub tokens: Vec<u32>,
    pub done: bool,
}

#[async_trait]
impl Pollable for ForwardPassResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }

        let res = (&mut self.receiver).await.unwrap();

        if let Ok(res) = rmp_serde::from_slice::<ForwardPassResponse>(&res) {
            self.distributions = res.dists;
            self.tokens = res.tokens;
        }

        self.done = true;
    }
}

impl bindings::pie::inferlet::forward::Host for InstanceState {
    async fn create_forward_pass(
        &mut self,
        queue: Resource<Queue>,
    ) -> anyhow::Result<Resource<ForwardPass>> {
        let queue = self.ctx().table.get(&queue)?.clone();
        let pass = ForwardPass {
            service_id: queue.service_id,
            stream_id: queue.stream_id,
            input_tokens: vec![],
            input_token_positions: vec![],
            input_embed_ptrs: vec![],
            input_embed_positions: vec![],
            adapter: None,
            adapter_seed: None,
            mask: vec![],
            kv_page_ptrs: vec![],
            kv_page_last_len: 0,
            output_token_indices: vec![],
            output_token_samplers: vec![],
            output_dist_indices: vec![],
            output_embed_ptrs: vec![],
            output_embed_indices: vec![],
            sampler_temperature: 0.60,
            sampler_top_k: 50,
            sampler_top_p: 0.9,
            sampler_min_p: 0.2,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    async fn input_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        mut emb_ptrs: Vec<ResourceId>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.service_id;

        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self
                .translate_resource_ptr(svc_id, resource::EMBED_TYPE_ID, *emb_ptr)
                .ok_or_else(|| {
                    anyhow::format_err!(
                        "Failed to translate input embedding with ptr: {:?}",
                        emb_ptr
                    )
                })?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.input_embed_ptrs = emb_ptrs;
        pass.input_embed_positions = positions;
        Ok(())
    }

    async fn input_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        input_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.input_tokens = input_tokens;
        pass.input_token_positions = positions;
        Ok(())
    }

    async fn output_embeddings(
        &mut self,
        pass: Resource<ForwardPass>,
        mut emb_ptrs: Vec<ResourceId>,
        indices: Vec<u32>,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.service_id;
        emb_ptrs.iter_mut().try_for_each(|emb_ptr| {
            *emb_ptr = self
                .translate_resource_ptr(svc_id, resource::EMBED_TYPE_ID, *emb_ptr)
                .ok_or_else(|| {
                    anyhow::format_err!(
                        "Failed to translate output embedding with ptr: {:?}",
                        emb_ptr
                    )
                })?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_embed_ptrs = emb_ptrs;
        pass.output_embed_indices = indices;
        Ok(())
    }

    async fn output_distributions(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_dist_indices = indices;
        Ok(())
    }

    async fn output_tokens(
        &mut self,
        pass: Resource<ForwardPass>,
        indices: Vec<u32>,
        samplers: Vec<u32>,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.output_token_indices = indices;
        pass.output_token_samplers = samplers;
        Ok(())
    }

    async fn attention_mask(
        &mut self,
        pass: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.mask = mask;
        Ok(())
    }

    async fn kv_cache(
        &mut self,
        pass: Resource<ForwardPass>,
        mut kv_page_ptrs: Vec<ResourceId>,
        kv_page_last_len: u32,
    ) -> anyhow::Result<()> {
        let svc_id = self.ctx().table.get(&pass)?.service_id;

        kv_page_ptrs.iter_mut().try_for_each(|kv_page_ptr| {
            *kv_page_ptr = self
                .translate_resource_ptr(svc_id, resource::KV_PAGE_TYPE_ID, *kv_page_ptr)
                .ok_or_else(|| {
                    anyhow::format_err!(
                        "Failed to translate KV cache page with ptr: {:?}",
                        kv_page_ptr
                    )
                })?;
            Ok::<_, anyhow::Error>(())
        })?;

        let pass = self.ctx().table.get_mut(&pass)?;
        pass.kv_page_ptrs = kv_page_ptrs;
        pass.kv_page_last_len = kv_page_last_len;
        Ok(())
    }

    async fn sampler_temperature(
        &mut self,
        pass: Resource<ForwardPass>,
        temperature: f32,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.sampler_temperature = temperature;

        Ok(())
    }

    async fn sampler_top_k(
        &mut self,
        pass: Resource<ForwardPass>,
        top_k: u32,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.sampler_top_k = top_k;
        Ok(())
    }

    async fn sampler_top_p(
        &mut self,
        pass: Resource<ForwardPass>,
        top_p: f32,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.sampler_top_p = top_p;
        Ok(())
    }

    async fn sampler_min_p(
        &mut self,
        pass: Resource<ForwardPass>,
        min_p: f32,
    ) -> anyhow::Result<()> {
        let pass = self.ctx().table.get_mut(&pass)?;
        pass.sampler_min_p = min_p;
        Ok(())
    }
}

impl bindings::pie::inferlet::forward::HostForwardPass for InstanceState {
    async fn execute(
        &mut self,
        this: Resource<ForwardPass>,
    ) -> anyhow::Result<Option<Resource<ForwardPassResult>>> {
        // 1) Check whether we need output (immutable borrow)
        let returns_output = {
            let pass = self.ctx().table.get(&this)?;
            !pass.output_dist_indices.is_empty() || !pass.output_token_indices.is_empty()
        };

        // 2) Build the request by MOVING data out of the pass (mutable borrow)
        let (request, service_id, stream_id) = {
            use std::mem::take;

            let pass = self.ctx().table.get_mut(&this)?;
            let service_id = pass.service_id;
            let stream_id = pass.stream_id;

            let request = ForwardPassRequest {
                input_tokens: take(&mut pass.input_tokens),
                input_token_positions: take(&mut pass.input_token_positions),
                input_embed_ptrs: take(&mut pass.input_embed_ptrs),
                input_embed_positions: take(&mut pass.input_embed_positions),
                adapter: pass.adapter,
                adapter_seed: pass.adapter_seed,
                mask: take(&mut pass.mask),
                kv_page_ptrs: take(&mut pass.kv_page_ptrs),
                kv_page_last_len: pass.kv_page_last_len,
                output_token_indices: take(&mut pass.output_token_indices),
                output_token_samplers: take(&mut pass.output_token_samplers),
                output_dist_indices: take(&mut pass.output_dist_indices),
                output_embed_ptrs: take(&mut pass.output_embed_ptrs),
                output_embed_indices: take(&mut pass.output_embed_indices),
                sampler_temperature: pass.sampler_temperature,
                sampler_top_k: pass.sampler_top_k,
                sampler_top_p: pass.sampler_top_p,
                sampler_min_p: pass.sampler_min_p,
            };

            (request, service_id, stream_id)
        };

        let data = Bytes::from(rmp_serde::to_vec_named(&request)?);
        let inst_id = self.id();

        if returns_output {
            let (tx, rx) = oneshot::channel();

            model::Command::Submit {
                inst_id,
                cmd_queue_id: stream_id,
                handler: Handler::ForwardPass,
                data,
                response: Some(tx),
            }
            .dispatch(service_id)?;

            let res = ForwardPassResult {
                receiver: rx,
                distributions: vec![],
                tokens: vec![],
                done: false,
            };

            Ok(Some(self.ctx().table.push(res)?))
        } else {
            model::Command::Submit {
                inst_id,
                cmd_queue_id: stream_id,
                handler: Handler::ForwardPass,
                data,
                response: None,
            }
            .dispatch(service_id)?;

            Ok(None)
        }
    }
    async fn drop(&mut self, this: Resource<ForwardPass>) -> anyhow::Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl bindings::pie::inferlet::forward::HostForwardPassResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get_distributions(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Option<Vec<(Vec<u32>, Vec<f32>)>>> {
        let result = self.ctx().table.get_mut(&this)?;

        if result.done {
            Ok(Some(std::mem::take(&mut result.distributions)))
        } else {
            Ok(None)
        }
    }

    async fn get_tokens(
        &mut self,
        this: Resource<ForwardPassResult>,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        let result = self.ctx().table.get_mut(&this)?;

        if result.done {
            Ok(Some(std::mem::take(&mut result.tokens)))
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<ForwardPassResult>) -> anyhow::Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
