use crate::utils::{Counter, IdPool};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use thiserror::Error;

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

    // //fn max_capacity(&self) -> IdRepr {
    //     IdRepr::MAX
    // }
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

    pub fn set_capacity(&mut self, ty: TY, capacity: IdRepr) -> Result<(), ObjectError> {
        self.id_pool
            .entry(ty)
            .or_insert_with(|| IdPool::new(capacity))
            .set_capacity(capacity)
            .map_err(|_e| ObjectError::NoAvailableSpace)
    }

    pub fn capacity(&self, ty: TY) -> Result<IdRepr, ObjectError> {
        let id_pool = self.id_pool.get(&ty).ok_or(ObjectError::ObjectNotFound)?;
        Ok(id_pool.capacity())
    }

    pub fn available(&self, ty: TY) -> Result<usize, ObjectError> {
        let id_pool = self.id_pool.get(&ty).ok_or(ObjectError::ObjectNotFound)?;
        Ok(id_pool.available())
    }

    pub fn create(&mut self, ty: TY, ns: NS, name: IdRepr) -> Result<IdRepr, ObjectError> {
        // acquire an ID from the pool
        let id = self
            .id_pool
            .entry(ty)
            .or_insert_with(|| IdPool::new(1))
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
                let _ = self.id_pool.get_mut(&ty).unwrap().release(id);
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
            .or_insert_with(|| IdPool::new(1))
            .acquire_many(names.len())
            .map_err(|_| ObjectError::NoAvailableSpace)?;

        // If the type is sharable, initialize the reference counter for each ID.
        if ty.is_sharable() {
            let counter_map = self.ref_counter.entry(ty).or_insert_with(HashMap::new);
            for &id in &ids {
                counter_map.insert(id, Counter::new(0));
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
            let _ = self.id_pool.get_mut(&ty).unwrap().release_many(&ids);
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

        if ty.is_sharable() {
            if let Some(counter_map) = self.ref_counter.get_mut(&ty) {
                for id in ids {
                    counter_map.get_mut(id).unwrap().inc();
                }
            }
        }

        Ok(())
    }

    pub fn inc_ref_count_many(&mut self, ty: TY, ids: &[IdRepr]) {
        if ty.is_sharable() {
            if let Some(counter_map) = self.ref_counter.get_mut(&ty) {
                for id in ids {
                    counter_map.get_mut(id).unwrap().inc();
                }
            }
        }
    }

    pub fn dec_ref_count_many(&mut self, ty: TY, ids: &[IdRepr]) {
        if ty.is_sharable() {
            if let Some(counter_map) = self.ref_counter.get_mut(&ty) {
                for id in ids {
                    counter_map.get_mut(id).unwrap().dec();
                }
            }
        }
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
            let _ = self.id_pool.get_mut(&ty).unwrap().release(id);

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

        // print hashmap
        //println!("hashmap: {:?}", table);

        for name in names {
            // Remove the (name, id) pair from the namespace.
            if let Some(id) = table.remove(name) {
                //println!("attempting to free, type: {:?}, id: {:?}", ty, id);
                // Determine if the object should be freed.
                let should_free = if ty.is_sharable() {
                    // let rc =self.ref_counter
                    // .get_mut(&ty)
                    // .unwrap()
                    // .get_mut(&id).unwrap().get();

                    //println!("ref count: {:?}", rc);

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
                    //println!("freed, type: {:?}, id: {:?}", ty, id);
                    let _ = self.id_pool.get_mut(&ty).unwrap().release(id);
                    freed_ids.push(id);
                }
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
            .iter()
            .map(|(n, _)| *n)
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

    pub fn translate_many(&self, ty: TY, ns: NS, names: &mut [IdRepr]) -> Result<(), ObjectError> {
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
