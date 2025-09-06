use crate::batching::BatchingConfig;
use crate::model::HandlerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

pub mod core;
pub mod forward;
pub mod image;

pub mod adapter;
pub mod evolve;
pub mod tokenize;

// big message table

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Handler {
    Handshake,
    Synchronize,
    Query,
    ForwardPass,
    EmbedImage,
    InitializeAdapter,
    UpdateAdapter,
}

impl Handler {
    pub fn get_handler_id(&self) -> HandlerId {
        match self {
            Self::Handshake => 0,
            Self::Synchronize => 0,
            Self::Query => 1,
            Self::ForwardPass => 2,
            Self::EmbedImage => 3,
            Self::InitializeAdapter => 4,
            Self::UpdateAdapter => 5,
        }
    }
}

pub fn get_batching_config() -> HashMap<Handler, BatchingConfig> {
    let mut config = HashMap::new();
    config.insert(
        Handler::Synchronize,
        BatchingConfig::Bounded {
            max_wait_time: Duration::ZERO,
            min_size: 1,
            max_size: None,
        },
    );
    config.insert(
        Handler::ForwardPass,
        BatchingConfig::Triggered {
            trigger: None,
            min_wait_time: Duration::ZERO,
        },
    );
    config
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    query: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    value: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    input_tokens: Vec<u32>,
    input_token_positions: Vec<u32>,
    input_embed_ptrs: Vec<u32>,
    input_embed_positions: Vec<u32>,
    adapter: Option<u32>,
    adapter_seed: Option<i64>,
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
#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    tokens: Vec<u32>,
    dists: Vec<(Vec<u32>, Vec<f32>)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    embed_ptrs: Vec<u32>,
    image_blob: Vec<u8>,
    position_offset: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitializeAdapterRequest {
    adapter_ptr: u32,
    rank: u32,
    alpha: f32,
    population_size: u32,
    mu_fraction: f32,
    initial_sigma: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateAdapterRequest {
    adapter_ptr: u32,
    scores: Vec<f32>,
    seeds: Vec<i64>,
    max_sigma: f32,
}
