use crate::utils;
use crate::utils::{Counter, Stream};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::slice;
use thiserror::Error;
use tokio::sync::oneshot;

/// Errors that can occur in block/object allocation and mapping.
#[derive(Debug, Error)]
pub enum ObjectError {
    /// Returned when the requested instance (or vspace) is not found.
    #[error("instance not found")]
    VSpaceNotFound,

    /// Returned when attempting to register a virtual address that already exists.
    #[error("virtual address already exists for vid: {0}")]
    VSpaceAlreadyExists(IdRepr),

    /// Returned when a virtual address cannot be found in the mapping.
    #[error("virtual address translation failed for vid: {0}")]
    VSpaceTranslationFailed(IdRepr),

    /// Returned when a block (or object) could not be found.
    #[error("block not found")]
    ObjectNotFound,

    /// Returned when there are no free blocks available in the ID pool.
    #[error("no free blocks available")]
    NoAvailableSpace,

    /// Returned when an error occurs in the underlying address/ID pool.
    #[error("address pool error: {0}")]
    AddressPoolError(String),

    /// Returned when an error occurs at the backend.
    #[error("backend error: {0}")]
    BackendError(String),
}

// ------------------------------------------------------------

// Id definition ------------------------------------------------
pub trait Object: Debug + Sized + Send + Sync {
    const NAMESPACE: Namespace;
    //fn get_namespace() -> Namespace;
}

pub type IdRepr = u32;

#[derive(Debug)]
pub struct Id<T: Object>(IdRepr, PhantomData<T>);

impl<T: Object> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Object> Copy for Id<T> {}

impl<T: Object> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Object> Eq for Id<T> {}

impl<T: Object> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T: Object> Id<T> {
    pub fn new(id: IdRepr) -> Self {
        Self(id, PhantomData)
    }

    pub fn namespace(&self) -> Namespace {
        T::NAMESPACE
    }
}

impl<T: Object> From<IdRepr> for Id<T> {
    fn from(id: IdRepr) -> Self {
        Self::new(id)
    }
}

impl<T: Object> Into<IdRepr> for Id<T> {
    fn into(self) -> IdRepr {
        self.0
    }
}

// Id definition ------------------------------------------------

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

    fn alloc(&mut self, stream: Stream, object: T) -> Result<Id<T>, ObjectError> {
        self.alloc_many(stream, vec![object])
            .map(|mut ids| ids.pop().unwrap())
    }

    fn alloc_many(&mut self, stream: Stream, objects: Vec<T>) -> Result<Vec<Id<T>>, ObjectError>;

    fn dealloc(&mut self, stream: Stream, id: Id<T>) -> Result<(), ObjectError> {
        self.dealloc_many(stream, &[id])
    }

    fn dealloc_many(&mut self, stream: Stream, ids: &[Id<T>]) -> Result<(), ObjectError>;

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

type VspaceId = u32;

// reference-counted object manager
pub trait MappedAllocator<T>: Allocator<T>
where
    T: ReferenceCounted + Object,
{
    fn vspaces(&self) -> &HashMap<VspaceId, HashMap<Id<T>, Id<T>>>;

    fn vspaces_mut(&mut self) -> &mut HashMap<VspaceId, HashMap<Id<T>, Id<T>>>;

    fn init_vspace(&mut self, vspace_id: VspaceId) -> Result<(), ObjectError> {
        if self.vspaces().contains_key(&vspace_id) {
            return Err(ObjectError::VSpaceAlreadyExists(vspace_id));
        }

        self.vspaces_mut().insert(vspace_id, HashMap::new());
        Ok(())
    }

    fn destroy_vspace(&mut self, vspace_id: &VspaceId) -> Result<(), ObjectError> {
        let removed = self
            .vspaces_mut()
            .remove(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        for addr in removed.keys() {
            MappedAllocator::<T>::dealloc(self, Stream::default(), vspace_id, &addr)?;
        }

        Ok(())
    }

    fn alloc(
        &mut self,
        stream: Stream,
        obj: T,
        vspace_id: &VspaceId,
        vid: Id<T>,
    ) -> Result<(), ObjectError> {
        MappedAllocator::alloc_many(self, stream, vec![obj], vspace_id, vec![vid])
    }

    fn alloc_many(
        &mut self,
        stream: Stream,
        objs: Vec<T>,
        vspace_id: &VspaceId,
        vids: Vec<Id<T>>,
    ) -> Result<(), ObjectError> {
        objs.iter().for_each(|obj| obj.add_ref());

        let ids = Allocator::alloc_many(self, stream, objs)?;

        // Retrieve a mutable reference to the vspace, converting a missing vspace into an InstanceNotFound error.
        let vspace = self
            .vspaces_mut()
            .get_mut(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        for (vid, id) in vids.into_iter().zip(ids.into_iter()) {
            match vspace.entry(vid) {
                Entry::Vacant(entry) => {
                    entry.insert(id);
                }
                Entry::Occupied(_) => {
                    return Err(ObjectError::VSpaceAlreadyExists(vid.into()));
                }
            }
        }
        Ok(())
    }

    fn dealloc(
        &mut self,
        stream: Stream,
        vspace_id: &VspaceId,
        vid: &Id<T>,
    ) -> Result<(), ObjectError> {
        MappedAllocator::dealloc_many(self, stream, vspace_id, slice::from_ref(vid))
    }

    fn dealloc_many(
        &mut self,
        stream: Stream,
        vspace_id: &VspaceId,
        vids: &[Id<T>],
    ) -> Result<(), ObjectError> {
        // Borrow the vspace mutably and remove all vids,
        // collecting their corresponding global addresses.
        let vspace = self
            .vspaces_mut()
            .get_mut(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        let mut ids = Vec::with_capacity(vids.len());
        for vid in vids {
            let id = vspace
                .remove(vid)
                .ok_or(ObjectError::VSpaceTranslationFailed(vid.0))?;
            ids.push(id);
        }
        // The mutable borrow on vspace is dropped here.

        // Now iterate over the collected ids to release and possibly deallocate.
        for id in ids {
            let remove_entirely = self
                .objects()
                .get(&id)
                .ok_or(ObjectError::ObjectNotFound)?
                .release();

            if remove_entirely {
                Allocator::dealloc(self, stream, id).map_err(|e| {
                    ObjectError::BackendError(format!(
                        "Failed to deallocate object in Allocator: {}",
                        e
                    ))
                })?;
            }
        }

        Ok(())
    }
    fn create_ref(
        &mut self,
        src_vspace_id: &VspaceId,
        src_vid: &Id<T>,
        dst_vspace_id: &VspaceId,
        dst_vid: &Id<T>,
    ) -> Result<(), ObjectError> {
        let src_id = self
            .vspaces()
            .get(&src_vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .get(src_vid)
            .ok_or(ObjectError::VSpaceTranslationFailed(src_vid.0))?
            .clone();

        // increase ref count
        self.objects()
            .get(&src_id)
            .ok_or(ObjectError::ObjectNotFound)?
            .add_ref();

        let dst_vspace = self
            .vspaces_mut()
            .get_mut(dst_vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        match dst_vspace.entry(*dst_vid) {
            Entry::Vacant(entry) => {
                entry.insert(src_id);
                Ok(())
            }
            Entry::Occupied(_) => Err(ObjectError::VSpaceAlreadyExists(dst_vid.0)),
        }
    }

    fn resolve(&self, vspace_id: &VspaceId, vid: &Id<T>) -> Result<Id<T>, ObjectError> {
        self.vspaces()
            .get(&vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?
            .get(vid)
            .copied()
            .ok_or(ObjectError::VSpaceTranslationFailed(vid.0))
    }

    fn resolve_many(
        &self,
        vspace_id: &VspaceId,
        vids: &[Id<T>],
    ) -> Result<Vec<Id<T>>, ObjectError> {
        let vspace = self
            .vspaces()
            .get(vspace_id)
            .ok_or(ObjectError::VSpaceNotFound)?;

        vids.iter()
            .map(|&vid| {
                vspace
                    .get(&vid)
                    .copied()
                    .ok_or(ObjectError::VSpaceTranslationFailed(vid.into()))
            })
            .collect()
    }

    fn get(&self, vspace_id: &VspaceId, vid: &Id<T>) -> Result<&T, ObjectError> {
        let id = self.resolve(vspace_id, vid)?;

        Allocator::get(self, id)
    }

    fn get_mut(&mut self, vspace_id: &VspaceId, vid: &Id<T>) -> Result<&mut T, ObjectError> {
        let id = self.resolve(vspace_id, vid)?;

        Allocator::get_mut(self, id)
    }

    fn get_many(&self, vspace_id: &VspaceId, vids: &[Id<T>]) -> Result<Vec<&T>, ObjectError> {
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
    const NAMESPACE: Namespace = Namespace::KvBlock;
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

impl Object for TokenEmb {
    const NAMESPACE: Namespace = Namespace::Emb;
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

impl Object for TokenDist {
    const NAMESPACE: Namespace = Namespace::Dist;
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

#[derive(Debug, Clone, Copy)]
pub enum Namespace {
    KvBlock = 0,
    Emb = 1,
    Dist = 2,
}

#[derive(Debug)]
pub struct IdPool {
    kv_block_id_pool: utils::IdPool<IdRepr>,
    emb_id_pool: utils::IdPool<IdRepr>,
    dist_id_pool: utils::IdPool<IdRepr>,
}

impl IdPool {
    pub fn new(max_kv_blocks: u32, max_embs: u32) -> Self {
        Self {
            kv_block_id_pool: utils::IdPool::new(max_kv_blocks),
            emb_id_pool: utils::IdPool::new(max_embs),
            dist_id_pool: utils::IdPool::new(max_embs),
        }
    }

    pub fn acquire<T: Object>(&mut self) -> Result<Id<T>, ObjectError> {
        if let Ok(id) = match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_pool.acquire(),
            Namespace::Emb => self.emb_id_pool.acquire(),
            Namespace::Dist => self.dist_id_pool.acquire(),
        } {
            Ok(Id::new(id))
        } else {
            Err(ObjectError::NoAvailableSpace)
        }
    }

    pub fn acquire_many<T: Object>(&mut self, count: usize) -> Result<Vec<Id<T>>, ObjectError> {
        if let Ok(ids) = match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_pool.acquire_many(count),
            Namespace::Emb => self.emb_id_pool.acquire_many(count),
            Namespace::Dist => self.dist_id_pool.acquire_many(count),
        } {
            Ok(ids.into_iter().map(Id::new).collect())
        } else {
            Err(ObjectError::NoAvailableSpace)
        }
    }

    pub fn release<T: Object>(&mut self, id: Id<T>) -> Result<(), ObjectError> {
        let id = id.into();
        match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_pool.release(id),
            Namespace::Emb => self.emb_id_pool.release(id),
            Namespace::Dist => self.dist_id_pool.release(id),
        }
        .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn release_many<T: Object>(&mut self, ids: &[Id<T>]) -> Result<(), ObjectError> {
        let ids = ids.iter().map(|id| id.0).collect::<Vec<_>>();
        match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_pool.release_many(&ids),
            Namespace::Emb => self.emb_id_pool.release_many(&ids),
            Namespace::Dist => self.dist_id_pool.release_many(&ids),
        }
        .map_err(|e| ObjectError::AddressPoolError(e.to_string()))
    }

    pub fn available<T: Object>(&self) -> usize {
        match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_pool.available(),
            Namespace::Emb => self.emb_id_pool.available(),
            Namespace::Dist => self.dist_id_pool.available(),
        }
    }
}

#[derive(Debug)]
pub struct IdMap {
    kv_block_id_map: HashMap<IdRepr, IdRepr>,
    emb_id_map: HashMap<IdRepr, IdRepr>,
    dist_id_map: HashMap<IdRepr, IdRepr>,
}

impl IdMap {
    pub fn new() -> Self {
        Self {
            kv_block_id_map: HashMap::new(),
            emb_id_map: HashMap::new(),
            dist_id_map: HashMap::new(),
        }
    }

    pub fn insert<T: Object>(&mut self, vid: Id<T>, id: Id<T>) {
        let (src, dst) = (vid.into(), id.into());
        match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_map.insert(src, dst),
            Namespace::Emb => self.emb_id_map.insert(src, dst),
            Namespace::Dist => self.dist_id_map.insert(src, dst),
        };
    }

    pub fn remove<T: Object>(&mut self, vid: Id<T>) -> Result<Id<T>, ObjectError> {
        let vid = vid.into();
        let id = match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_map.remove(&vid),
            Namespace::Emb => self.emb_id_map.remove(&vid),
            Namespace::Dist => self.dist_id_map.remove(&vid),
        }
        .ok_or(ObjectError::ObjectNotFound)?;
        Ok(Id::<T>::new(id))
    }

    pub fn get<T: Object>(&self, vid: Id<T>) -> Result<Id<T>, ObjectError> {
        let vid = vid.into();
        let id = match T::NAMESPACE {
            Namespace::KvBlock => self.kv_block_id_map.get(&vid),
            Namespace::Emb => self.emb_id_map.get(&vid),
            Namespace::Dist => self.dist_id_map.get(&vid),
        }
        .ok_or(ObjectError::ObjectNotFound)?;

        Ok(Id::<T>::new(*id))
    }
}
