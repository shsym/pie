use crate::utils::Stream;
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

pub type IdRepr = u32;

#[repr(transparent)]
#[derive(Debug)]
pub struct Id<T>(IdRepr, PhantomData<T>);

impl<T> Id<T> {
    pub fn new(id: IdRepr) -> Self {
        Self(id, PhantomData)
    }
    pub fn into_repr(self) -> IdRepr {
        self.0
    }

    pub fn map_from_repr(reprs: Vec<IdRepr>) -> Vec<Id<T>> {
        // The safe way to do this is to use Vec::map...
        // reprs
        //     .into_iter()
        //     .map(|repr| Id(repr, PhantomData))
        //     .collect()
        // But we can do it in a more efficient way since the Id is repr(transparent).
        let capacity = reprs.capacity();
        let len = reprs.len();
        let ptr = reprs.as_ptr() as *mut Id<T>;
        std::mem::forget(reprs); // Prevent the original Vec<u32> from dropping
        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }

    pub fn map_to_repr(ids: Vec<Id<T>>) -> Vec<IdRepr> {
        // Safe way:
        // ids.into_iter().map(Id::into_repr).collect()

        let capacity = ids.capacity();
        let len = ids.len();
        let ptr = ids.as_ptr() as *mut u32;
        std::mem::forget(ids);
        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }

    pub fn ref_as_repr(ids: &[Id<T>]) -> &[u32] {
        unsafe { slice::from_raw_parts(ids.as_ptr() as *const u32, ids.len()) }
    }

    pub fn group_consecutive_ids(ids: &[Id<T>]) -> Vec<(Id<T>, IdRepr)> {
        let mut ranges = Vec::new();
        let ids = Id::ref_as_repr(ids);
        if ids.is_empty() {
            return ranges;
        }

        // Initialize the first range with the first id.
        let mut offset = ids[0];
        let mut size = 1;

        // Use windows to look at each consecutive pair.
        for pair in ids.windows(2) {
            let (current, next) = (pair[0], pair[1]);
            if next == current + 1 {
                // They are consecutive, so increase the current range.
                size += 1;
            } else {
                // A gap is found, so push the current range.
                ranges.push((Id::new(offset), size));
                // Start a new range with the next id.
                offset = next;
                size = 1;
            }
        }

        // Don't forget to push the last range.
        ranges.push((Id::new(offset), size));
        ranges
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
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

impl<T> Into<IdRepr> for &Id<T> {
    fn into(self) -> IdRepr {
        self.0
    }
}

// Id definition ------------------------------------------------

pub trait Allocator<T> {
    fn alloc(&mut self, stream: Stream) -> Result<Id<T>, ObjectError> {
        self.alloc_many(stream, 1)
            .map(|mut ids| ids.pop().unwrap())
    }

    fn alloc_many(&mut self, stream: Stream, count:usize) -> Result<Vec<Id<T>>, ObjectError>;

    fn dealloc(&mut self, stream: Stream, id: Id<T>) -> Result<(), ObjectError> {
        self.dealloc_many(stream, &[id])
    }

    fn dealloc_many(&mut self, stream: Stream, ids: &[Id<T>]) -> Result<(), ObjectError>;

    fn available(&self) -> usize;
}

pub trait Fetcher<T>: Allocator<T> {
    type RawRepr;

    fn fetch(&self, id: &Id<T>, sender: oneshot::Sender<Self::RawRepr>) -> Result<(), ObjectError>;
}

pub type VspaceId = u32;

// reference-counted object manager
pub trait MappedAllocator<T>: Allocator<T> {
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

    /// Increments the reference count.
    fn ref_inc(&mut self, id: &Id<T>) -> Result<(), ObjectError>;

    /// Decrements the reference count.
    ///
    /// Returns `true` if the count has reached zero, indicating that the object
    /// can be safely cleaned up.
    fn ref_dec(&mut self, id: &Id<T>) -> Result<bool, ObjectError>;

    /// Returns the current reference count.
    fn ref_count(&self, id: &Id<T>) -> Result<usize, ObjectError>;

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
            let remove_entirely = self.ref_dec(&id)?;

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
        let src_id = self.resolve(src_vspace_id, src_vid)?;

        // increase ref count
        self.ref_inc(&src_id)?;

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
}
