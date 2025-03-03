use std::fmt::Debug;

use crate::object;

use crate::driver::DriverError;
use crate::object::VspaceId;
use crate::utils::Stream;
use tokio::sync::oneshot;
// object::Id definition ------------------------------------------------

// ------------------------------------------------------------

#[derive(Debug)]
pub struct KvBlock;
#[derive(Debug)]
pub struct TokenEmb;

// distribution
#[derive(Debug)]
pub struct TokenDist;

// ------------------------------------------------------------

// Backend trait for filling key-value blocks. (GPT-like models)
pub trait CausalTransformer: object::IdMapper<KvBlock> + object::IdMapper<TokenEmb> {
    fn fill(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        addr: object::Id<KvBlock>,
        ctx_addrs: Vec<object::Id<KvBlock>>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), DriverError>;

    fn copy_tokens(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        src_ptr: object::Id<KvBlock>,
        dst_ptr: object::Id<KvBlock>,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), DriverError>;

    fn mask_tokens(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        ptr: object::Id<KvBlock>,
        mask: &[bool],
    ) -> Result<(), DriverError>;
}

// probably unused in the first version. For BERT-like models.
pub trait FullTransformer: object::IdMapper<TokenEmb> {
    fn fill(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        mask: Vec<bool>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), DriverError>;
}

// could be used for other LLM architectures like SSMs
pub trait Rnn: object::IdMapper<TokenEmb> {
    fn fill(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        state: object::Id<KvBlock>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), DriverError>;
}

// ------------------------------------------------------------

// ------------------------------------------------------------

pub trait CausalLanguageModel: object::IdMapper<TokenEmb> + object::IdMapper<TokenDist> {
    fn embed_text(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        addrs: Vec<object::Id<TokenEmb>>,
        text_tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<(), DriverError>;

    fn next_token_dist(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        emb_ptr: Vec<object::Id<TokenEmb>>,
        dist_ptr: Vec<object::Id<TokenDist>>,
    ) -> Result<(), DriverError>;

    fn sample_top_k(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        dist_ptr: &object::Id<TokenDist>,
        k: u32,
        handle: oneshot::Sender<(Vec<u32>, Vec<f32>)>,
    ) -> Result<(), DriverError>;

    // todo: design a better struct to represent distributions
}

pub trait MaskedLanguageModel: object::IdMapper<TokenEmb> + object::IdMapper<TokenDist> {
    fn token_dist(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        emb_ptr: &object::Id<TokenEmb>,
        dist_ptr: &object::Id<TokenDist>,
    ) -> Result<(), DriverError>;
}

// ------------------------------------------------------------

// Trait for backends that can embed images.
pub trait ImageEmbedder: object::IdMapper<TokenEmb> {
    fn embed_img(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        addrs: Vec<object::Id<TokenEmb>>,
        url: String,
    ) -> Result<(), DriverError>;
}

// Trait for backends that can embed videos.
pub trait VideoEmbedder: object::IdMapper<TokenEmb> {
    fn embed_vid(
        &mut self,
        stream: Stream,
        space: &object::VspaceId,
        addrs: Vec<object::Id<TokenEmb>>,
        url: String,
    ) -> Result<(), DriverError>;
}

/////
