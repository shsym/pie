use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::{object, utils};

use crate::object::ObjectError;
use crate::utils::{Counter, Stream};
use tokio::sync::oneshot;

//pub type StreamId = (u128, u32);

// object::Id definition ------------------------------------------------


// ------------------------------------------------------------

#[derive(Debug)]
pub struct KvBlock {
    counter: Counter,
}

impl KvBlock {
    pub fn new() -> Self {
        KvBlock {
            counter: Counter::new(0),
        }
    }
}

impl object::Share for KvBlock {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}
#[derive(Debug)]
pub struct TokenEmb {
    counter: Counter,
}

impl TokenEmb {
    pub fn new() -> Self {
        TokenEmb {
            counter: Counter::new(0),
        }
    }
}

impl object::Share for TokenEmb {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}

// distribution
#[derive(Debug)]
pub struct TokenDist {
    counter: Counter,
}

impl TokenDist {
    pub fn new() -> Self {
        TokenDist {
            counter: Counter::new(0),
        }
    }
}

impl object::Share for TokenDist {
    fn add_ref(&self) {
        self.counter.inc();
    }

    fn release(&self) -> bool {
        self.counter.dec() <= 0
    }

    fn ref_count(&self) -> usize {
        self.counter.get() as usize
    }
}

// ------------------------------------------------------------

// Backend trait for filling key-value blocks. (GPT-like models)
pub trait CausalTransformer: object::Allocator<KvBlock> + object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        addr: object::Id<KvBlock>,
        ctx_addrs: Vec<object::Id<KvBlock>>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Option<Vec<object::Id<TokenEmb>>>,
    ) -> Result<(), ObjectError>;

    fn copy_tokens(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        src_ptr: object::Id<KvBlock>,
        dst_ptr: object::Id<KvBlock>,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), ObjectError>;

    fn mask_tokens(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        ptr: object::Id<KvBlock>,
        mask: &[bool],
    ) -> Result<(), ObjectError>;
}

// probably unused in the first version. For BERT-like models.
pub trait FullTransformer: object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        mask: Vec<bool>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), ObjectError>;
}

// could be used for other LLM architectures like SSMs
pub trait Rnn: object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        state: object::Id<KvBlock>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), ObjectError>;
}

// ------------------------------------------------------------

// ------------------------------------------------------------

pub trait CausalLanguageModel:
    object::MappedAllocator<TokenEmb> + object::MappedAllocator<TokenDist>
{
    fn next_token_dist(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        emb_ptr: object::Id<TokenEmb>,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<(), ObjectError>;

    fn sample_top_k(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        dist_ptr: object::Id<TokenDist>,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, ObjectError>;

    // todo: design a better struct to represent distributions
}

pub trait MaskedLanguageModel: object::Allocator<TokenEmb> + object::Allocator<TokenDist> {
    fn token_dist(
        &self,
        stream: Stream,
        vspace_id: &object::VspaceId,
        emb_ptr: object::Id<TokenEmb>,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<(), ObjectError>;
}

// ------------------------------------------------------------

// Trait for backends that can embed images.
pub trait ImageEmbedder: object::Allocator<TokenEmb> {
    fn embed_img(
        &self,
        stream_id: Stream,
        vspace_id: &object::VspaceId,
        addrs: Vec<object::Id<TokenEmb>>,
        url: String,
    ) -> Result<(), ObjectError>;
}

// Trait for backends that can embed videos.
pub trait VideoEmbedder: object::Allocator<TokenEmb> {
    fn embed_vid(
        &self,
        stream_id: Stream,
        vspace_id: &object::VspaceId,
        addrs: Vec<object::Id<TokenEmb>>,
        url: String,
    ) -> Result<(), ObjectError>;
}

/////
