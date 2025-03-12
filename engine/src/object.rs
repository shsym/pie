use crate::utils::Stream;
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
        self.alloc_all(stream, 1).map(|mut ids| ids.pop().unwrap())
    }

    fn alloc_all(&mut self, stream: Stream, count: usize) -> Result<Vec<Id<T>>, ObjectError>;

    fn dealloc(&mut self, stream: Stream, id: &Id<T>) -> Result<(), ObjectError> {
        self.dealloc_all(stream, slice::from_ref(id))
    }

    fn dealloc_all(&mut self, stream: Stream, ids: &[Id<T>]) -> Result<(), ObjectError>;

    fn available(&self) -> usize;

    // /// Increments the reference count.
    fn increment_ref_count(&mut self, id: &Id<T>) -> Result<(), ObjectError>;

    // Decrements the reference count.
    //
    // Returns `true` if the count has reached zero, indicating that the object
    // can be safely cleaned up.
    fn decrement_ref_count(&mut self, id: &Id<T>) -> Result<bool, ObjectError>;

    //Returns the current reference count.
    fn ref_count(&self, id: &Id<T>) -> Result<usize, ObjectError>;
}

pub type VspaceId = u32;

// pub trait Vspace<T> {
//     fn contains(&self, vid: &Id<T>) -> bool;
//
//     fn get(&self, vid: &Id<T>) -> Result<Id<T>, ObjectError>;
//
//     fn get_many(&self, vids: &[Id<T>]) -> Result<Vec<Id<T>>, ObjectError>;
//
//     fn insert(&mut self, vid: Id<T>, id: Id<T>);
//
//     fn remove(&mut self, vid: &Id<T>) -> Result<object::Id<T>, ObjectError>;
//
//     fn all_vids(&self) -> Vec<Id<T>>;
// }

// reference-counted object manager
pub trait IdMapper<T>: Allocator<T> {
    fn exists(&self, space: &VspaceId, src: &Id<T>) -> bool;
    fn list(&self, space: &VspaceId) -> Result<Vec<Id<T>>, ObjectError>;

    fn lookup(&self, space: &VspaceId, src: &Id<T>) -> Result<Id<T>, ObjectError> {
        self.lookup_all(space, slice::from_ref(src))
            .map(|mut ids| ids.pop().unwrap())
    }

    fn lookup_all(&self, space: &VspaceId, srcs: &[Id<T>]) -> Result<Vec<Id<T>>, ObjectError>;

    fn assign(&mut self, space: &VspaceId, src: &Id<T>, tgt: &Id<T>) -> Result<(), ObjectError> {
        self.assign_all(space, slice::from_ref(src), slice::from_ref(tgt))
    }

    fn assign_all(
        &mut self,
        space: &VspaceId,
        srcs: &[Id<T>],
        tgts: &[Id<T>],
    ) -> Result<(), ObjectError>;

    fn alloc_and_assign(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        src: &Id<T>,
    ) -> Result<Id<T>, ObjectError> {
        self.alloc_and_assign_all(stream, space, slice::from_ref(src))
            .map(|mut ids| ids.pop().unwrap())
    }

    fn alloc_and_assign_all(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        srcs: &[Id<T>],
    ) -> Result<Vec<Id<T>>, ObjectError> {
        let ids = self.alloc_all(stream, srcs.len())?;
        self.assign_all(space, srcs, &ids)?;
        Ok(ids)
    }

    fn unassign(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        src: &Id<T>,
    ) -> Result<(), ObjectError> {
        self.unassign_all(stream, space, slice::from_ref(src))
    }

    fn unassign_all(
        &mut self,
        stream: Stream,
        space: &VspaceId,
        srcs: &[Id<T>],
    ) -> Result<(), ObjectError>;
}

pub fn group_consecutive_ids(ids: &[IdRepr]) -> Vec<(IdRepr, IdRepr)> {
    let mut ranges = Vec::new();
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
            ranges.push((offset, size));
            // Start a new range with the next id.
            offset = next;
            size = 1;
        }
    }

    // Don't forget to push the last range.
    ranges.push((offset, size));
    ranges
}
