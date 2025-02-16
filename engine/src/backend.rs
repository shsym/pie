use std::collections::HashMap;
use std::hash::Hash;

use crate::object;

use crate::object::{KvBlock, ObjectError, TokenDist, TokenEmb};
use tokio::sync::oneshot;
use uuid::Uuid;
use crate::utils::Stream;

pub type InstanceId = Uuid;
//pub type StreamId = (u128, u32);


#[derive(Debug)]
pub struct Resource {
    owner_id: InstanceId,
    addrs: Vec<object::Id<KvBlock>>,
}

impl Resource {
    pub fn new(owner_id: InstanceId, addrs: Vec<object::Id<KvBlock>>) -> Self {
        Self { owner_id, addrs }
    }
}

#[derive(Debug)]
pub struct Instance {
    owned_resources: Vec<String>,
    //usage_stats: HashMap<String, usize>,
}

impl Instance {
    pub fn new() -> Self {
        Self {
            owned_resources: vec![],
            //usage_stats: HashMap::new(),
        }
    }
}



// Backend trait for filling key-value blocks. (GPT-like models)
pub trait CausalTransformer<K>: object::Allocator<KvBlock> + object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream: Stream,
        vspace_id: K,
        addr: object::Id<KvBlock>,
        ctx_addrs: Vec<object::Id<KvBlock>>,
        input_embs: Vec<object::Id<TokenEmb>>,
        output_embs: Option<Vec<object::Id<TokenEmb>>>,
    ) -> Result<(), ObjectError>;

    fn copy_tokens(
        &self,
        stream: Stream,
        vspace_id: K,
        src_ptr: object::Id<KvBlock>,
        dst_ptr: object::Id<KvBlock>,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) -> Result<(), ObjectError>;

    fn mask_tokens(
        &self,
        stream: Stream,
        vspace_id: K,
        ptr: object::Id<KvBlock>,
        mask: &[bool],
    ) -> Result<(), ObjectError>;
}

// probably unused in the first version. For BERT-like models.
pub trait FullTransformer: object::Allocator<TokenEmb> {
    fn fill(
        &self,
        stream: Stream,
        vspace_id: K,
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
        state: object::Id<KvBlock>,
        output_embs: Vec<object::Id<TokenEmb>>,
    ) -> Result<(), ObjectError>;
}

// ------------------------------------------------------------

// ------------------------------------------------------------

pub trait CausalLanguageModel<K>:
object::MappedAllocator<TokenEmb, K> + object::MappedAllocator<TokenDist, K>
where
    K: Hash + Copy,
{
    fn next_token_dist(
        &self,
        stream: Stream,
        vspace_id: &K,
        emb_ptr: object::Id<TokenEmb>,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<(), ObjectError>;

    fn sample_top_k(
        &self,
        stream: Stream,
        dist_ptr: object::Id<TokenDist>,
        k: u32,
    ) -> Result<oneshot::Receiver<Vec<u32>>, ObjectError>;

    // todo: design a better struct to represent distributions
    fn get_raw_dist(
        &self,
        stream: Stream,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<oneshot::Receiver<Vec<f32>>, ObjectError>;
}

pub trait MaskedLanguageModel: object::Allocator<TokenEmb> + object::Allocator<TokenDist> {
    fn token_dist(
        &self,
        stream: Stream,
        emb_ptr: object::Id<TokenEmb>,
        dist_ptr: object::Id<TokenDist>,
    ) -> Result<(), ObjectError>;
}

// ------------------------------------------------------------

// Trait for backends that can embed images.
pub trait ImageEmbedder: object::Allocator<TokenEmb> {
    fn embed_img(
        &self,
        stream_id: StreamId,
        addrs: Vec<object::Id>,
        url: String,
    ) -> Result<(), BlockError>;
}

// Trait for backends that can embed videos.
pub trait VideoEmbedder: object::Allocator<TokenEmb> {
    fn embed_vid(
        &self,
        stream_id: StreamId,
        addrs: Vec<object::Id>,
        url: String,
    ) -> Result<(), BlockError>;
}




//
// pub struct ControllerManager<B> {
//     backend: B,
//     // resource name -> Resource handle
//     resources: HashMap<String, Resource>,
//     // instance_id -> Instance
//     instances: HashMap<InstanceId, Instance>,
//
//     // managers
//     kv_blocks: KvBlockManager<B>,
//     token_embs: TokenEmbManager<B>,
// }
// //
// //
// //
// // impl object::Allocator<KvBlock> for Controller<B> {
// //     fn alloc(&mut self, _local_stream_id: Option<u32>) -> Result<Addr, BlockError> {
// //         unimplemented!()
// //     }
// //
// //     fn dealloc(&mut self, _local_stream_id: Option<u32>, _addr: Addr) -> Result<(), BlockError> {
// //         unimplemented!()
// //     }
// // }
//
// impl<B> ControllerManager<B>
// where
//     B: CausalTransformer + Clone,
// {
//     pub fn new(backend: B) -> Self {
//         Self {
//             backend: backend.clone(),
//             resources: HashMap::new(),
//             instances: HashMap::new(),
//             kv_blocks: KvBlockManager::new(backend.clone()),
//             token_embs: TokenEmbManager::new(backend),
//             //fill_cmd_batcher: FillBlockCmdBatcher::new(),
//         }
//     }
//
//     pub fn init_instance(&mut self, inst_id: InstanceId) -> Result<(), BlockError> {
//         self.instances.insert(inst_id, Instance::new());
//         self.kv_blocks.init_instance(inst_id)?;
//         self.token_embs.init_instance(inst_id)?;
//         Ok(())
//     }
//
//     pub fn destroy_instance(&mut self, inst_id: &InstanceId) -> Result<(), BlockError> {
//         if let Some(inst) = self.instances.get(inst_id) {
//             for r in &inst.owned_resources {
//                 self.resources.remove(r);
//             }
//         }
//         self.instances.remove(inst_id);
//         self.kv_blocks.destroy_instance(inst_id)?;
//         self.token_embs.destroy_instance(inst_id)?;
//
//         Ok(())
//     }
//
//     pub fn allocate_kv_block(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//     ) -> Result<Addr, BlockError> {
//         self.kv_blocks
//             .alloc(inst_id, local_stream_id, KvBlock::new())
//     }
//
//     pub fn deallocate_kv_block(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         addr: Addr,
//     ) -> Result<(), BlockError> {
//         self.kv_blocks.dealloc(inst_id, local_stream_id, addr)
//     }
//
//     pub fn export_kv_blocks(
//         &mut self,
//         inst_id: &InstanceId,
//         resource_name: String,
//         addrs: Vec<Addr>,
//     ) -> Result<(), BlockError> {
//         self.resources
//             .insert(resource_name.clone(), Resource::new(*inst_id, addrs));
//         self.instances
//             .get_mut(inst_id)
//             .ok_or(BlockError::InstanceNotFound)?
//             .owned_resources
//             .push(resource_name);
//         Ok(())
//     }
//
//     pub fn import_kv_blocks(
//         &mut self,
//         inst_id: &InstanceId,
//         resource_name: String,
//     ) -> Result<Vec<Addr>, BlockError> {
//         let res = self
//             .resources
//             .get(&resource_name)
//             .ok_or(BlockError::ResourceNotFound)?;
//
//         let mut addrs = Vec::with_capacity(res.addrs.len());
//         for src_addr in &res.addrs {
//             addrs.push(
//                 self.kv_blocks
//                     .create_ref(inst_id, &res.owner_id, *src_addr)?,
//             );
//         }
//
//         Ok(addrs)
//     }
//
//     pub fn fill_kv_block(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         addr: Addr,
//         ctx_addrs: Vec<Addr>,
//         input_embs: Vec<Addr>,
//         output_embs: Option<Vec<Addr>>,
//     ) -> Result<(), BlockError> {
//         // create "resolved" cmd.
//
//         let block_ptr = self.kv_blocks.resolve(inst_id, addr)?;
//         let ctx_block_ptrs = self.kv_blocks.resolve_many(inst_id, &ctx_addrs)?;
//
//         let input_emb_ptrs = self.token_embs.resolve_many(inst_id, &input_embs)?;
//
//         // let output_emb_ptrs = if let Some(output_embs) = output_embs {
//         //     Some(self.token_embs.resolve_many(inst_id, &output_embs)?)
//         // } else {
//         //     None
//         // };
//
//         let output_emb_ptrs = output_embs
//             .map(|emb| self.token_embs.resolve_many(inst_id, &emb))
//             .transpose()?;
//
//         self.kv_blocks.get_mut(inst_id, addr)?.filled = false;
//
//         self.backend.fill(
//             get_stream_id(inst_id, local_stream_id),
//             block_ptr,
//             ctx_block_ptrs,
//             input_emb_ptrs,
//             output_emb_ptrs,
//         )?;
//
//         Ok(())
//     }
//
//     pub fn copy_kv_block(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         src_addr: Addr,
//         dst_addr: Addr,
//         src_token_offset: u32,
//         dst_token_offset: u32,
//         token_count: u32,
//     ) -> Result<(), BlockError> {
//         self.kv_blocks.copy_tokens(
//             inst_id,
//             local_stream_id,
//             src_addr,
//             dst_addr,
//             src_token_offset,
//             dst_token_offset,
//             token_count,
//         )
//     }
//
//     pub fn mask_kv_block(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         addr: Addr,
//         mask: Vec<bool>,
//     ) -> Result<(), BlockError> {
//         self.kv_blocks
//             .mask_tokens(inst_id, local_stream_id, addr, &mask)
//     }
//
//     pub fn allocate_token_emb(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//     ) -> Result<Addr, BlockError> {
//         self.token_embs
//             .alloc(inst_id, local_stream_id, TokenEmb::new())
//     }
//
//     pub fn deallocate_token_emb(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         addr: Addr,
//     ) -> Result<(), BlockError> {
//         self.token_embs.dealloc(inst_id, local_stream_id, addr)
//     }
// }
//
// // For causal LMs
//
// impl<B> ControllerManager<B>
// where
//     B: CausalLanguageModel,
// {
//     pub fn next_token_dist(
//         &self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         emb_ptr: Addr,
//         dist_ptr: Addr,
//     ) -> Result<(), BlockError> {
//         let emb_ptr = self.token_embs.resolve(inst_id, emb_ptr)?;
//         let dist_ptr = self.token_embs.resolve(inst_id, dist_ptr)?;
//
//         self.backend
//             .next_token_dist(get_stream_id(inst_id, local_stream_id), emb_ptr, dist_ptr)?;
//
//         Ok(())
//     }
//
//     pub fn sample_top_k(
//         &self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//         dist_ptr: Addr,
//         k: u32,
//     ) -> Result<oneshot::Receiver<Vec<u32>>, BlockError> {
//         let dist_ptr = self.token_embs.resolve(inst_id, dist_ptr)?;
//         self.backend
//             .sample_top_k(get_stream_id(inst_id, local_stream_id), dist_ptr, k)
//     }
// }

// For multimodal LLMs
// impl<B> ControllerManager<B>
// where
//     B: ImageEmbedder + VideoEmbedder + ObjectAllocator<TokenEmb>,
// {
//     pub fn embed_image(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//
//         token_addrs: Vec<Addr>,
//         image_url: String,
//     ) -> Result<(), BlockError> {
//         let ptrs = self.token_embs.resolve_many(inst_id, &token_addrs)?;
//
//         self.backend
//             .embed_img(get_stream_id(inst_id, local_stream_id), ptrs, image_url)?;
//
//         Ok(())
//     }
//
//     pub fn embed_video(
//         &mut self,
//         inst_id: &InstanceId,
//         local_stream_id: Option<u32>,
//
//         token_addrs: Vec<Addr>,
//         video_url: String,
//     ) -> Result<(), BlockError> {
//         let ptrs = self.token_embs.resolve_many(inst_id, &token_addrs)?;
//
//         self.backend
//             .embed_vid(get_stream_id(inst_id, local_stream_id), ptrs, video_url)?;
//
//         Ok(())
//     }
// }
