use crate::utils;
use crate::utils::{Counter, Stream};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use thiserror::Error;
use tokio::sync::oneshot;

//ub type Id = u32;

pub trait Object: Debug + Sized + Send + Sync {
    fn get_namespace() -> Namespace;
}

pub type IdRepr = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id<T>(IdRepr, PhantomData<T>);

impl<T> Id<T>
where
    T: Object,
{
    pub fn new(id: IdRepr) -> Self {
        Self(id, PhantomData)
    }

    pub fn namespace(&self) -> Namespace {
        T::get_namespace()
    }
}

impl<T> From<IdRepr> for Id<T> {
    fn from(id: IdRepr) -> Self {
        Self::new(id)
    }
}

impl<T> Into<IdRepr> for Id<T> {
    fn into(self) -> IdRepr {
        self.0
    }
}

/// Errors that can occur in block/object allocation and mapping.
#[derive(Debug, Error)]
pub enum ObjectError {
    /// Returned when trying to create an instance (or vspace) that already exists.
    #[error("instance already exists")]
    VSpaceAlreadyExists,

    /// Returned when the requested instance (or vspace) is not found.
    #[error("instance not found")]
    VSpaceNotFound,

    /// Returned when a block (or object) could not be found.
    #[error("block not found")]
    ObjectNotFound,

    /// Returned when there are no free blocks available in the ID pool.
    #[error("no free blocks available")]
    NoFreeBlocks,

    /// Returned when attempting to register a virtual address that already exists.
    #[error("virtual address already exists for vid: {0}")]
    VirtualAddressAlreadyExists(IdRepr),

    /// Returned when a virtual address cannot be found in the mapping.
    #[error("virtual address translation failed for vid: {0}")]
    VirtualAddressTranslationFailed(IdRepr),

    /// Returned when an error occurs in the underlying address/ID pool.
    #[error("address pool error: {0}")]
    AddressPoolError(String),

    /// Returned when an error occurs at the backend.
    #[error("backend error: {0}")]
    BackendError(String),
}

// ------------------------------------------------------------

// this helps the backend to make optimization decisions.

pub trait ReferenceCounted {
    /// Increments the reference count.
    fn add_ref(&self);

    /// Decrements the reference count.
    ///
    /// Returns `true` if the count has reached zero, indicating that the object
    /// can be safely cleaned up.
    fn release(&self) -> bool;

    /// Returns the current reference count.
    fn ref_count(&self) -> usize;
}

pub trait Allocator<T>
where
    T: Object,
{
    type RawRepr;

    fn objects(&self) -> &HashMap<Id<T>, T>;

    fn objects_mut(&mut self) -> &mut HashMap<Id<T>, T>;

    fn alloc(&mut self, stream: Stream, object: T) -> Result<Id<T>, ObjectError>;

    fn dealloc(&mut self, stream: Stream, id: Id<T>) -> Result<(), ObjectError>;

    fn get(&self, id: Id<T>) -> Result<&T, ObjectError> {
        self.objects().get(&id).ok_or(ObjectError::ObjectNotFound)
    }

    fn get_mut(&mut self, id: Id<T>) -> Result<&mut T, ObjectError> {
        self.objects_mut()
            .get_mut(&id)
            .ok_or(ObjectError::ObjectNotFound)
    }

    fn get_many(&self, ids: &[Id<T>]) -> Result<Vec<&T>, ObjectError> {
        ids.iter().map(|&id| self.get(id)).collect()
    }

    // retrieve the underlying data from the backend.
    // this is unlikely to be used in practice, except for debugging, implementing some new sampling algos, and so on.
    fn raw_repr(
        &self,
        stream: Stream,
        id: Id<T>,
        sender: oneshot::Sender<Self::RawRepr>,
    ) -> Result<(), ObjectError>;

    fn available(&self) -> usize;
}

// reference-counted object manager
pub trait MappedAllocator<T, K>: Allocator<T>
where
    T: ReferenceCounted + Object,
    K: Copy + Hash,
{
    fn vspaces(&self) -> &HashMap<K, HashMap<Id<T>, Id<T>>>;

    fn vspaces_mut(&mut self) -> &mut HashMap<K, HashMap<Id<T>, Id<T>>>;

    fn init_vspace(&mut self, vspace_id: K) -> Result<(), ObjectError> {
        if self.vspaces().contains_key(&vspace_id) {
            return Err(ObjectError::VSpaceAlreadyExists);
        }

        self.vspaces_mut().insert(vspace_id, HashMap::new());
        Ok(())
    }

    fn destroy_vspace(&mut self, vspace_id: &K) -> Result<(), ObjectError> {
        let addr_iter = self
            .vspaces()
            .get(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .keys();

        for addr in addr_iter {
            self.dealloc(Stream::default(), vspace_id, addr)?;
        }

        self.vspaces_mut().remove(vspace_id);

        Ok(())
    }

    fn alloc(
        &mut self,
        stream: Stream,
        obj: T,
        vspace_id: &K,
        vid: Id<T>,
    ) -> Result<(), ObjectError> {
        obj.add_ref();

        // Allocate using the Allocator, converting any error into a backend error.
        let id = Allocator::alloc(self, stream, obj).map_err(|e| {
            ObjectError::BackendError(format!("Failed to allocate object in Allocator: {}", e))
        })?;

        // Retrieve a mutable reference to the vspace, converting a missing vspace into an InstanceNotFound error.
        let vspace = self
            .vspaces_mut()
            .get_mut(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        match vspace.entry(vid.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(id);
                Ok(())
            }
            Entry::Occupied(_) => Err(ObjectError::VirtualAddressAlreadyExists(vid.into())),
        }
    }

    fn dealloc(&mut self, stream: Stream, vspace_id: &K, vid: Id<T>) -> Result<(), ObjectError> {
        // remove and get the global address
        let id = self
            .vspaces_mut()
            .get_mut(&vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .remove(&vid)
            .ok_or(ObjectError::VirtualAddressTranslationFailed(vid.into()))?;

        let remove_entirely = self
            .objects()
            .get(&id)
            .ok_or(ObjectError::ObjectNotFound)?
            .release();

        // remove the block if the ref count is 0
        if remove_entirely {
            Allocator::dealloc(self, stream, id).map_err(|e| {
                ObjectError::BackendError(format!(
                    "Failed to deallocate object in Allocator: {}",
                    e
                ))
            })?;
        }

        Ok(())
    }

    fn create_ref(
        &mut self,
        src_vspace_id: &K,
        src_vid: &Id<T>,
        dst_vspace_id: &K,
        dst_vid: &Id<T>,
    ) -> Result<(), ObjectError> {
        let src_id = self
            .vspaces()
            .get(&src_vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .get(src_vid)
            .ok_or(ObjectError::VirtualAddressTranslationFailed(src_vid.into()))?;

        let dst_vspace = self
            .vspaces_mut()
            .get_mut(dst_vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        // increase ref count
        self.objects()
            .get(&src_id)
            .ok_or(ObjectError::ObjectNotFound)?
            .add_ref();

        match dst_vspace.entry(*dst_vid) {
            Entry::Vacant(entry) => {
                entry.insert(*src_id);
                Ok(())
            }
            Entry::Occupied(_) => Err(ObjectError::VirtualAddressAlreadyExists(dst_vid.into())),
        }
    }

    fn resolve(&self, vspace_id: &K, vid: &Id<T>) -> Result<Id<T>, ObjectError> {
        self.vspaces()
            .get(&vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .get(vid)
            .copied()
            .ok_or(ObjectError::VirtualAddressTranslationFailed(vid.into()))
    }

    fn resolve_many(&self, vspace_id: &K, vids: &[Id<T>]) -> Result<Vec<Id<T>>, ObjectError> {
        let vspace = self
            .vspaces()
            .get(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        vids.iter()
            .map(|&vid| {
                vspace
                    .get(&vid)
                    .copied()
                    .ok_or(ObjectError::VirtualAddressTranslationFailed(vid.into()))
            })
            .collect()
    }

    fn get(&self, vspace_id: &K, vid: &Id<T>) -> Result<&T, ObjectError> {
        let id = self.resolve(vspace_id, vid)?;

        Allocator::get(self, id)
    }

    fn get_mut(&mut self, vspace_id: &K, vid: &Id<T>) -> Result<&mut T, ObjectError> {
        let id = self.resolve(vspace_id, vid)?;

        Allocator::get_mut(self, id)
    }

    fn get_many(&self, vspace_id: &K, vids: &[Id<T>]) -> Result<Vec<&T>, ObjectError> {
        let ids = self.resolve_many(vspace_id, vids)?;

        Allocator::get_many(self, &ids)
    }
}

// ------------------------------------------------------------

#[derive(Debug)]
pub struct KvBlock {
    counter: Counter,

}

impl Object for KvBlock {
    fn get_namespace() -> Namespace {
        Namespace::KvBlock
    }
}

impl KvBlock {
    pub fn new() -> Self {
        KvBlock {
            counter: Counter::new(0),
        }
    }
}

impl ReferenceCounted for KvBlock {
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

impl ReferenceCounted for TokenEmb {
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

impl ReferenceCounted for TokenDist {
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

#[derive(Debug)]
pub struct IdPool {
    kv_block_id_pool: utils::IdPool<Id<KvBlock>>,
    emb_id_pool: utils::IdPool<Id<TokenEmb>>,
    dist_id_pool: utils::IdPool<Id<TokenDist>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Namespace {
    KvBlock = 0,
    Emb = 1,
    Dist = 2,
    Cmd = 3,
}

impl IdPool {
    pub fn new(max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            kv_block_id_pool: utils::IdPool::new(max_kv_blocks),
            emb_id_pool: utils::IdPool::new(max_embs),
            dist_id_pool: utils::IdPool::new(max_embs),
        }
    }

    pub fn acquire(&mut self, ns: Namespace) -> Result<Id, ObjectError> {
        match ns {
            Namespace::KvBlock => self.kv_block_id_pool.acquire(),
            Namespace::Emb => self.emb_id_pool.acquire(),
            Namespace::Dist => self.dist_id_pool.acquire(),
            Namespace::Cmd => self.cmd_id_pool.acquire(),
        }
        .ok_or(ObjectError::NoFreeBlocks)
    }

    pub fn release(&mut self, ns: Namespace, id: Id) -> Result<(), ObjectError> {
        match ns {
            Namespace::KvBlock => self.kv_block_id_pool.release(id),
            Namespace::Emb => self.emb_id_pool.release(id),
            Namespace::Dist => self.dist_id_pool.release(id),
            Namespace::Cmd => self.cmd_id_pool.release(id),
        }
        .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn available(&self, ns: Namespace) -> usize {
        match ns {
            Namespace::KvBlock => self.kv_block_id_pool.available(),
            Namespace::Emb => self.emb_id_pool.available(),
            Namespace::Dist => self.dist_id_pool.available(),
            Namespace::Cmd => self.cmd_id_pool.available(),
        }
    }
}
