use anyhow::{Result, bail};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::oneshot;

pub static HANDSHAKE_ID: u32 = 0;

pub static QUERY_ID: u32 = 2;
pub static FORWARD_PASS_ID: u32 = 3;
pub static EMBED_IMAGE_ID: u32 = 4;
pub static INITIALIZE_ADAPTER_ID: u32 = 5;
pub static UPDATE_ADAPTER_ID: u32 = 6;
pub static UPLOAD_ADAPTER_ID: u32 = 7;
pub static DOWNLOAD_ADAPTER_ID: u32 = 8;

#[derive(Debug)]
pub enum Request {
    Handshake(HandshakeRequest, oneshot::Sender<HandshakeResponse>),
    Query(QueryRequest, oneshot::Sender<QueryResponse>),
    Synchronize(oneshot::Sender<()>),

    ForwardPass(
        ForwardPassRequest,
        Option<oneshot::Sender<ForwardPassResponse>>,
    ),
    EmbedImage(EmbedImageRequest),
    InitializeAdapter(InitializeAdapterRequest),
    UpdateAdapter(UpdateAdapterRequest),
    UploadAdapter(UploadAdapterRequest),
    DownloadAdapter(DownloadAdapterRequest, oneshot::Sender<Bytes>),
}

impl Request {
    pub fn is_eager(&self) -> bool {
        match self {
            Request::ForwardPass(_, _) => false,
            _ => true,
        }
    }

    pub fn is_sync_req(&self) -> bool {
        match self {
            Request::Synchronize(_) => true,
            _ => false,
        }
    }

    pub fn has_response(&self) -> bool {
        match self {
            Request::Handshake(_, _) => true,
            Request::Query(_, _) => true,
            Request::ForwardPass(_, r) => r.is_some(),
            Request::DownloadAdapter(_, _) => true,
            _ => false,
        }
    }

    pub fn handler_id(&self) -> u32 {
        match self {
            Request::Handshake(_, _) => HANDSHAKE_ID,
            Request::Query(_, _) => QUERY_ID,
            Request::Synchronize(_) => unreachable!("Synchronize request has no handler ID"),

            Request::ForwardPass(_, _) => FORWARD_PASS_ID,
            Request::EmbedImage(_) => EMBED_IMAGE_ID,
            Request::InitializeAdapter(_) => INITIALIZE_ADAPTER_ID,
            Request::UpdateAdapter(_) => UPDATE_ADAPTER_ID,
            Request::UploadAdapter(_) => UPLOAD_ADAPTER_ID,
            Request::DownloadAdapter(_, _) => DOWNLOAD_ADAPTER_ID,
        }
    }

    pub fn serialize_req(&self) -> Result<Bytes> {
        let b = match self {
            Request::Handshake(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Query(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::Synchronize(_) => bail!("cannot serialize synchronize request"),
            Request::ForwardPass(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::EmbedImage(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::InitializeAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UpdateAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::UploadAdapter(req) => Bytes::from(rmp_serde::to_vec_named(&req)?),
            Request::DownloadAdapter(req, _) => Bytes::from(rmp_serde::to_vec_named(&req)?),
        };
        Ok(b)
    }

    pub fn deserialize_resp(self, b: Bytes) -> Result<()> {
        match self {
            Request::Handshake(_, resp) => {
                let r: HandshakeResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::Query(_, resp) => {
                let r: QueryResponse = rmp_serde::from_slice(&b)?;
                resp.send(r).ok();
            }
            Request::ForwardPass(_, resp) => {
                let r: ForwardPassResponse = rmp_serde::from_slice(&b)?;
                if let Some(tx) = resp {
                    tx.send(r).ok();
                }
            }
            Request::DownloadAdapter(_, resp) => {
                resp.send(b).ok();
            }
            _ => {
                bail!("cannot deserialize response for request {:?}", self);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeRequest {
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HandshakeResponse {
    pub version: String,
    pub model_name: String,
    pub model_traits: Vec<String>,
    pub model_description: String,
    pub prompt_template: String,
    pub prompt_template_type: String,
    pub prompt_stop_tokens: Vec<String>,
    pub kv_page_size: u32,
    pub max_batch_tokens: usize,
    pub max_batch_size: usize,
    pub resources: HashMap<u32, u32>,
    pub tokenizer_num_vocab: usize,
    pub tokenizer_merge_table: HashMap<u32, Vec<u8>>,
    pub tokenizer_special_tokens: HashMap<String, u32>,
    pub tokenizer_split_regex: String,
    pub tokenizer_escape_non_printable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub query: String,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResponse {
    pub value: String,
}



#[derive(Debug, Serialize, Deserialize)]
pub struct ForwardPassRequest {
    pub input_tokens: Vec<u32>,
    pub input_token_positions: Vec<u32>,
    pub input_embed_ptrs: Vec<u32>,
    pub input_embed_positions: Vec<u32>,
    pub adapter: Option<u32>,
    pub adapter_seed: Option<i64>,
    pub mask: Vec<Vec<u32>>,
    pub kv_page_ptrs: Vec<u32>,
    pub kv_page_last_len: u32,
    pub output_token_indices: Vec<u32>,
    pub output_token_samplers: Vec<HashMap<String, rmpv::Value>>,
    pub output_embed_ptrs: Vec<u32>,
    pub output_embed_indices: Vec<u32>,
    /// Arrival time for scheduler estimation (not serialized).
    #[serde(skip)]
    pub arrival_time: Option<Instant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedImageRequest {
    pub embed_ptrs: Vec<u32>,
    pub image_blob: Vec<u8>,
    pub position_offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeAdapterRequest {
    pub adapter_ptr: u32,
    pub rank: u32,
    pub alpha: f32,
    pub population_size: u32,
    pub mu_fraction: f32,
    pub initial_sigma: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateAdapterRequest {
    pub adapter_ptr: u32,
    pub scores: Vec<f32>,
    pub seeds: Vec<i64>,
    pub max_sigma: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
    pub adapter_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadAdapterRequest {
    pub adapter_ptr: u32,
    pub name: String,
}

// ============================================================================
// Batched Request/Response types for pycrust RPC
// ============================================================================

/// Batched forward pass request sent to Python via pycrust.
/// Rust performs partial batch formation (concatenating arrays),
/// while Python handles attention mask decoding and tensor creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassRequest {
    // Concatenated arrays from all requests in the batch
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,

    // KV cache layout (concatenated)
    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>, // [0, n1, n1+n2, ...] indices into kv_page_indices
    pub kv_last_page_lens: Vec<u32>, // One per request

    // Query/Output indirection
    pub qo_indptr: Vec<u32>, // [0, tokens1, tokens1+tokens2, ...]

    // Attention masks (BRLE encoded, flattened)
    pub flattened_masks: Vec<u32>, // Concatenation of all BRLE buffers
    pub mask_indptr: Vec<u32>,     // Pointers into flattened_masks for each token

    // Adapter info (one per request)
    pub adapter_indices: Vec<Option<u32>>,
    pub adapter_seeds: Vec<Option<i64>>,

    // Output specifications (per request)
    pub output_token_indices: Vec<Vec<u32>>,
    pub output_token_samplers: Vec<Vec<HashMap<String, rmpv::Value>>>,
    pub output_embed_ptrs: Vec<Vec<u32>>,
    pub output_embed_indices: Vec<Vec<u32>>,

    // Inference mode hint
    pub single_token_mode: bool,
}

impl BatchedForwardPassRequest {
    /// Create a new empty batched request.
    pub fn new() -> Self {
        Self {
            token_ids: Vec::new(),
            position_ids: Vec::new(),
            kv_page_indices: Vec::new(),
            kv_page_indptr: vec![0],
            kv_last_page_lens: Vec::new(),
            qo_indptr: vec![0],
            flattened_masks: Vec::new(),
            mask_indptr: vec![0],
            adapter_indices: Vec::new(),
            adapter_seeds: Vec::new(),
            output_token_indices: Vec::new(),
            output_token_samplers: Vec::new(),
            output_embed_ptrs: Vec::new(),
            output_embed_indices: Vec::new(),
            single_token_mode: true,
        }
    }

    /// Add a single ForwardPassRequest to the batch.
    pub fn add_request(&mut self, req: &ForwardPassRequest) {
        // Concatenate tokens and positions
        self.token_ids.extend(&req.input_tokens);
        self.position_ids.extend(&req.input_token_positions);

        // KV cache layout
        self.kv_page_indices.extend(&req.kv_page_ptrs);
        self.kv_page_indptr.push(self.kv_page_indices.len() as u32);
        self.kv_last_page_lens.push(req.kv_page_last_len);

        // Query/output indirection
        let total_tokens = self.token_ids.len() as u32;
        self.qo_indptr.push(total_tokens);

        // Masks (flatten nested structure)
        for token_mask in &req.mask {
            self.flattened_masks.extend(token_mask);
            self.mask_indptr.push(self.flattened_masks.len() as u32);
        }

        // Adapter info
        self.adapter_indices.push(req.adapter);
        self.adapter_seeds.push(req.adapter_seed);

        // Output specifications
        self.output_token_indices.push(req.output_token_indices.clone());
        self.output_token_samplers.push(req.output_token_samplers.clone());
        self.output_embed_ptrs.push(req.output_embed_ptrs.clone());
        self.output_embed_indices.push(req.output_embed_indices.clone());

        // Update inference mode hint
        if req.input_tokens.len() > 1 {
            self.single_token_mode = false;
        }
    }

    /// Get the number of requests in this batch.
    pub fn num_requests(&self) -> usize {
        self.adapter_indices.len()
    }

    /// Get the total number of tokens in this batch.
    pub fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }
}

impl Default for BatchedForwardPassRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Batched forward pass response from Python via pycrust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassResponse {
    /// Results indexed by request order in the batch.
    pub results: Vec<ForwardPassResponse>,
}
