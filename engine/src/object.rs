use crate::utils::{Counter, IdPool, Stream};
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

    #[error("remap not allowed: {0}")]
    RemapNotAllowed(String),

    #[error("unknown object type: {0}")]
    UnknownObjectType(String),
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

pub trait ObjectType: Eq + Hash + Debug + Copy {
    fn is_sharable(&self) -> bool;
    fn allow_remapping(&self) -> bool;

    fn max_capacity(&self) -> IdRepr {
        IdRepr::MAX
    }
}

#[derive(Debug)]
pub struct ObjectManager<NS, TY>
where
    NS: Eq + Hash + Debug + Clone,
    TY: ObjectType,
{
    id_pool: HashMap<TY, IdPool<IdRepr>>,
    ref_counter: HashMap<TY, HashMap<IdRepr, Counter>>,

    namespaces: HashMap<(TY, NS), HashMap<IdRepr, IdRepr>>,
}

impl<NS, TY> ObjectManager<NS, TY>
where
    NS: Eq + Hash + Debug + Clone,
    TY: ObjectType,
{
    pub fn new() -> Self {
        let id_pool = HashMap::new();
        let ref_counter = HashMap::new();

        Self {
            id_pool,
            ref_counter,
            namespaces: HashMap::new(),
        }
    }

    pub fn create(&mut self, ty: TY, ns: NS, name: IdRepr) -> Result<IdRepr, ObjectError> {
        // acquire an ID from the pool
        let id = self
            .id_pool
            .entry(ty)
            .or_insert_with(|| IdPool::new(ty.max_capacity()))
            .acquire()
            .map_err(|_| ObjectError::NoAvailableSpace)?;

        if ty.is_sharable() {
            self.ref_counter
                .entry(ty)
                .or_insert_with(HashMap::new)
                .insert(id, Counter::new(0));
        }

        // insert the name into the namespace
        // in case of failure, release the ID back to the pool
        if let Err(e) = self.create_ref(ty, ns, name, &id) {
            if ty.is_sharable() {
                self.ref_counter.get_mut(&ty).unwrap().remove(&id);
                self.id_pool.get_mut(&ty).unwrap().release(id);
            }
            return Err(e);
        }

        Ok(id)
    }

    pub fn create_many(
        &mut self,
        ty: TY,

        ns: NS,
        names: Vec<IdRepr>,
    ) -> Result<Vec<IdRepr>, ObjectError> {
        // Get or initialize the ID pool for this type.
        // Acquire the required number of IDs. If any acquisition fails,
        // the function returns an error.
        let ids = self
            .id_pool
            .entry(ty.clone())
            .or_insert_with(|| IdPool::new(ty.max_capacity()))
            .acquire_many(names.len())
            .map_err(|_| ObjectError::NoAvailableSpace)?;

        // If the type is sharable, initialize the reference counter for each ID.
        if ty.is_sharable() {
            let counter_map = self.ref_counter.entry(ty).or_insert_with(HashMap::new);
            for &id in &ids {
                counter_map.insert(id, Counter::new(1));
            }
        }

        // Insert the (name, id) pairs into the namespace.
        if let Err(e) = self.create_ref_many(ty, ns, names, &ids) {
            // If the insertion fails, release the acquired IDs and return an error.
            if ty.is_sharable() {
                for id in &ids {
                    self.ref_counter.get_mut(&ty).unwrap().remove(id);
                }
            }
            self.id_pool.get_mut(&ty).unwrap().release_many(&ids);
            return Err(e);
        }

        Ok(ids)
    }

    pub fn create_ref(
        &mut self,
        ty: TY,
        ns: NS,
        name: IdRepr,
        id: &IdRepr,
    ) -> Result<(), ObjectError> {
        // check if the name is already in use
        let ty_ns = (ty, ns);

        if !ty.allow_remapping() {
            if let Some(table) = self.namespaces.get(&ty_ns) {
                if table.contains_key(&name) {
                    return Err(ObjectError::RemapNotAllowed(format!("{:?}", name)));
                }
            }
        }

        // insert the name into the namespace
        self.namespaces
            .entry(ty_ns)
            .or_insert_with(HashMap::new)
            .insert(name, *id);

        // increment the reference count
        if ty.is_sharable() {
            self.ref_counter
                .entry(ty)
                .or_insert_with(HashMap::new)
                .get_mut(&id)
                .ok_or(ObjectError::ObjectNotFound)?
                .inc();
        }

        Ok(())
    }

    pub fn create_ref_many(
        &mut self,
        ty: TY,

        ns: NS,
        names: Vec<IdRepr>,
        ids: &[IdRepr],
    ) -> Result<(), ObjectError> {
        let ty_ns = (ty.clone(), ns.clone());

        // When remapping is not allowed, first ensure no name conflict exists,
        // including checking for duplicates in the input vector.
        if !ty.allow_remapping() {
            if let Some(table) = self.namespaces.get(&ty_ns) {
                for name in &names {
                    if table.contains_key(name) {
                        return Err(ObjectError::RemapNotAllowed(format!("{:?}", name)));
                    }
                }
            }
            let mut seen = std::collections::HashSet::new();
            for name in &names {
                if !seen.insert(name.clone()) {
                    return Err(ObjectError::RemapNotAllowed(format!("{:?}", name)));
                }
            }
        }

        // Insert the (name, id) pairs into the namespace.
        let table = self.namespaces.entry(ty_ns).or_insert_with(HashMap::new);
        for (name, id) in names.into_iter().zip(ids.iter().copied()) {
            table.insert(name, id);
        }

        Ok(())
    }

    pub fn destroy(
        &mut self,
        ty: TY,
        ns: NS,
        name: &IdRepr,
    ) -> Result<Option<IdRepr>, ObjectError> {
        let ty_ns = (ty, ns);

        let id = self
            .namespaces
            .get_mut(&ty_ns)
            .ok_or(ObjectError::VSpaceNotFound)?
            .remove(name)
            .ok_or(ObjectError::ObjectNotFound)?;

        let should_free = if ty.is_sharable() {
            self.ref_counter
                .get_mut(&ty)
                .unwrap()
                .get_mut(&id)
                .unwrap()
                .dec()
                <= 0
        } else {
            true
        };

        if should_free {
            self.id_pool.get_mut(&ty).unwrap().release(id);

            if self.namespaces.get(&ty_ns).unwrap().is_empty() {
                self.namespaces.remove(&ty_ns);
            }

            Ok(Some(id))
        } else {
            Ok(None)
        }
    }

    pub fn destroy_many(
        &mut self,
        ty: TY,
        ns: NS,
        names: &[IdRepr],
    ) -> Result<Vec<IdRepr>, ObjectError> {
        // Get the namespace table for the given type and namespace.
        let ty_ns = (ty.clone(), ns);
        let table = self
            .namespaces
            .get_mut(&ty_ns)
            .ok_or(ObjectError::VSpaceNotFound)?;

        let mut freed_ids = Vec::new();

        for name in names {
            // Remove the (name, id) pair from the namespace.
            let id = table.remove(name).ok_or(ObjectError::ObjectNotFound)?;

            // Determine if the object should be freed.
            let should_free = if ty.is_sharable() {
                self.ref_counter
                    .get_mut(&ty)
                    .unwrap()
                    .get_mut(&id)
                    .unwrap()
                    .dec()
                    <= 0
            }
            // Non-sharable objects are always freed.
            else {
                true
            };

            if should_free {
                self.id_pool.get_mut(&ty).unwrap().release(id);
                freed_ids.push(id);
            }
        }

        // cleanup the namespace table if it is empty
        if table.is_empty() {
            self.namespaces.remove(&ty_ns);
        }

        Ok(freed_ids)
    }

    pub fn all_names(&mut self, ty: TY, ns: NS) -> Result<Vec<IdRepr>, ObjectError> {
        let ty_ns = (ty.clone(), ns.clone());

        let names: Vec<IdRepr> = self
            .namespaces
            .get_mut(&ty_ns)
            .ok_or(ObjectError::VSpaceNotFound)?
            .drain()
            .map(|(n, _)| n)
            .collect();

        Ok(names)
    }

    pub fn translate(&self, ty: TY, ns: NS, name: &mut IdRepr) -> Result<(), ObjectError> {
        let ty_ns = (ty.clone(), ns);
        let table = self
            .namespaces
            .get(&ty_ns)
            .ok_or(ObjectError::VSpaceNotFound)?;

        *name = table
            .get(name)
            .copied()
            .ok_or(ObjectError::ObjectNotFound)?;

        Ok(())
    }

    pub fn translate_many(
        &self,
        ty: TY,
        ns: NS,
        names: &mut [IdRepr],
    ) -> Result<(), ObjectError> {
        let ty_ns = (ty.clone(), ns);
        let table = self
            .namespaces
            .get(&ty_ns)
            .ok_or(ObjectError::VSpaceNotFound)?;

        for name in names {
            *name = *table.get(name).ok_or(ObjectError::ObjectNotFound)?;
        }

        Ok(())
    }
}
