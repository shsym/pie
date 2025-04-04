use num_traits::PrimInt;
use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use crate::l4m;
use crate::l4m::ObjectType;

/// Errors that can occur while using the ID pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdPoolError {
    /// The new capacity is smaller than the number of already allocated IDs.
    CapacityTooSmall,
    /// No more IDs can be allocated.
    PoolExhausted,
    /// The given ID was never allocated.
    IdNotAllocated,
}

impl fmt::Display for IdPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdPoolError::CapacityTooSmall => write!(f, "New capacity is too small"),
            IdPoolError::PoolExhausted => write!(f, "ID pool exhausted"),
            IdPoolError::IdNotAllocated => write!(f, "ID was never allocated"),
        }
    }
}

impl std::error::Error for IdPoolError {}

pub type Result<T> = std::result::Result<T, IdPoolError>;

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct ResourcePool {
    model: Rc<l4m::Model>,
    ty: ObjectType,
    /// The next (never‑allocated) ID.
    next: u32,
    /// The set of freed IDs.
    free: BTreeSet<u32>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: u32,

    tight: bool,
}

impl ResourcePool {
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(model: Rc<l4m::Model>, ty: ObjectType, max_capacity: u32, tight: bool) -> Self {
        Self {
            model,
            ty,
            next: 0,
            free: BTreeSet::new(),
            max_capacity,
            tight,
        }
    }

    /// Set a new capacity for the pool.
    ///
    /// Returns an error if the new capacity is less than the next available ID.
    pub fn set_capacity(&mut self, capacity: u32) -> Result<()> {
        if capacity < self.next {
            return Err(IdPoolError::CapacityTooSmall);
        }
        self.max_capacity = capacity;
        Ok(())
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Ok(id)` if an ID is available, or an error if the pool is exhausted.
    pub fn acquire(&mut self) -> Result<u32> {
        if let Some(&id) = self.free.iter().next() {
            // A freed ID is available. Remove and return it.
            self.free.remove(&id);
            if self.tight {
                self.model.allocate(0, self.ty, &[id]);
            }
            Ok(id)
        } else if self.next < self.max_capacity {
            // Allocate a fresh ID.
            let addr = self.next;
            self.next = self.next + 1;
            self.model.allocate(0, self.ty, &[addr]);
            Ok(addr)
        } else {
            Err(IdPoolError::PoolExhausted)
        }
    }

    /// Acquire many IDs at once.
    ///
    /// Returns a vector of IDs or an error if not enough IDs are available.
    pub fn acquire_many(&mut self, count: usize) -> Result<Vec<u32>> {
        if self.available() < count {
            return Err(IdPoolError::PoolExhausted);
        }

        let mut result = Vec::with_capacity(count);
        let mut need_alloc = Vec::new();
        for _ in 0..count {
            if let Some(&id) = self.free.iter().next() {
                self.free.remove(&id);
                if self.tight {
                    need_alloc.push(id);
                }
                result.push(id);
            } else if self.next < self.max_capacity {
                // Allocate a fresh ID.
                let addr = self.next;
                self.next = self.next + 1;
                need_alloc.push(addr);
                result.push(addr);
            } else {
                return Err(IdPoolError::PoolExhausted);
            }
        }
        if need_alloc.len() > 0 {
            self.model.allocate(0, self.ty, &need_alloc);
        }
        Ok(result)
    }

    /// Release an ID back into the pool so it can be re-used.
    ///
    /// Returns an error if the given ID was never allocated.
    pub fn release(&mut self, addr: u32) -> Result<()> {
        if addr >= self.next {
            return Err(IdPoolError::IdNotAllocated);
        }
        self.free.insert(addr);

        if self.tight {
            self.model.deallocate(0, self.ty, &[addr]);
        }
        if self.free.len() > 1000 {
            self.tail_optimization();
        }
        Ok(())
    }

    /// Release multiple IDs back into the pool.
    ///
    /// Returns an error if any of the given IDs were never allocated.
    pub fn release_many(&mut self, addrs: &[u32]) -> Result<()> {
        let mut to_deallocate = Vec::new();

        for &addr in addrs {
            if addr >= self.next {
                return Err(IdPoolError::IdNotAllocated);
            }
            self.free.insert(addr);
            if self.tight {
                to_deallocate.push(addr);
            }
        }

        if self.tight {
            self.model.deallocate(0, self.ty, &to_deallocate);
        }

        if self.free.len() > 1000 {
            self.tail_optimization();
        }
        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next) as usize
    }

    /// Tail optimization: if the largest freed ID is exactly next-1,
    /// collapse the free block by decrementing `next`.
    fn tail_optimization(&mut self) {
        let cur_next = self.next;
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - 1 {
                self.free.remove(&last);
                self.next = self.next - 1;
            } else {
                break;
            }
        }
        let diff = cur_next - self.next;

        // if tight, these IDs are already deallocated
        if diff > 0 && !self.tight {
            let ids = (self.next..cur_next).collect::<Vec<_>>();
            // Update the model with the new next ID.
            self.model.deallocate(0, self.ty, &ids);
        }
    }
}

/// A reference‑counted ID pool that re‑uses IDs only when their reference count drops to zero.
/// Uses a vector (instead of a `HashMap`) for storing reference counts for better performance in dense cases.
#[derive(Debug)]
pub struct ResourceRcPool {
    pool: ResourcePool,
    // Stores the reference count for each allocated ID.
    // The index is derived from `id.to_usize()`.
    rc: Vec<u8>,
}

impl ResourceRcPool {
    /// Create a new RcIdPool with the given maximum capacity.
    pub fn new(model: Rc<l4m::Model>, ty: ObjectType, max_capacity: u32, tight: bool) -> Self {
        Self {
            pool: ResourcePool::new(model, ty, max_capacity, tight),
            rc: Vec::new(),
        }
    }

    /// Set a new capacity for the underlying ID pool.
    pub fn set_capacity(&mut self, capacity: u32) -> Result<()> {
        self.pool.set_capacity(capacity)
    }

    /// Ensure the reference count vector is large enough to index the given ID.
    fn ensure_rc_capacity(&mut self, id: u32) {
        let index = id as usize;
        if index >= self.rc.len() {
            self.rc.resize(index + 1, 0);
        }
    }

    /// Increment the reference count for the given ID.
    pub fn increment_rc(&mut self, id: u32) {
        self.ensure_rc_capacity(id);
        self.rc[id as usize] += 1;
    }

    pub fn increment_rc_many(&mut self, ids: &[u32]) {
        for &id in ids {
            self.increment_rc(id);
        }
    }

    /// Acquire an ID and increment its reference count.
    ///
    /// Returns the allocated ID or an error if the pool is exhausted.
    pub fn acquire(&mut self) -> Result<u32> {
        let id = self.pool.acquire()?;
        self.increment_rc(id);
        Ok(id)
    }

    /// Acquire many IDs at once.
    ///
    /// Returns a vector of allocated IDs or an error if not enough IDs are available.
    pub fn acquire_many(&mut self, count: usize) -> Result<Vec<u32>> {
        let ids = self.pool.acquire_many(count)?;
        for &id in &ids {
            self.increment_rc(id);
        }
        Ok(ids)
    }

    /// Release an ID. The underlying ID is only returned to the pool when its reference count drops to zero.
    ///
    /// Returns the released ID or an error if the ID was never allocated.
    pub fn release(&mut self, id: u32) -> Result<u32> {
        let idx = id as usize;
        if idx >= self.rc.len() || self.rc[idx] == 0 {
            return Err(IdPoolError::IdNotAllocated);
        }
        self.rc[idx] -= 1;
        if self.rc[idx] == 0 {
            self.pool.release(id)?;
        }
        Ok(id)
    }

    /// Release many IDs. Only those whose reference count drops to zero are returned to the pool.
    ///
    /// Returns a vector of IDs that were actually deallocated or an error if any ID was never allocated.
    pub fn release_many(&mut self, ids: &[u32]) -> Result<Vec<u32>> {
        let mut deallocated = Vec::new();
        for &id in ids {
            let idx = (id as usize);
            if idx >= self.rc.len() || self.rc[idx] == 0 {
                return Err(IdPoolError::IdNotAllocated);
            }
            self.rc[idx] -= 1;
            if self.rc[idx] == 0 {
                deallocated.push(id);
            }
        }
        if !deallocated.is_empty() {
            self.pool.release_many(&deallocated)?;
        }
        Ok(deallocated)
    }
}
