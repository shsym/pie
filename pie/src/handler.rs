use crate::batching::{BatchSchedulingPolicy, BatchingConfig};
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
        Handler::ForwardPass,
        BatchingConfig::Triggered {
            trigger: None,
            min_wait_time: Duration::ZERO,
        },
    );
    config
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    // backend metadata
    pub version: String,

    // model metadata
    pub model_name: String,
    pub model_description: String,
    pub model_template: String,
    pub model_template_type: String,

    // resources
    pub kv_page_size: u32,
    pub resources: Vec<(u32, u32)>, // (id, capacity)

    // tokenizer
    pub tokenizer_merge_table: Vec<(u32, Vec<u8>)>,
    pub tokenizer_special_tokens: Vec<(String, u32)>,
    pub tokenizer_split_regex: String,
    pub tokenizer_escape_non_printable: bool,
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
    adapter: u32,
    mask: Vec<Vec<u32>>,
    kv_cache_page_ptrs: Vec<u32>,
    kv_cache_last_page_len: u32,
    output_token_indices: Vec<u32>,
    output_token_samplers: Vec<u32>,
    output_dist_indices: Vec<u32>,
    output_embed_ptrs: Vec<u32>,
    output_embed_indices: Vec<u32>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    tokens: Vec<u32>,
    dists: Vec<(u32, Vec<f32>)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    image_ptr: u32,
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
