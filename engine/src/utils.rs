use crate::object;
use crate::object::ObjectError;
use anyhow::{Error, Result};
use num_traits::PrimInt;
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicIsize, Ordering};
use uuid::Uuid;
use wasmtime::Ref;

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
pub struct Stream {
    pub inst_id: u128,
    pub local_stream_id: u32,
}

impl Stream {
    pub fn new(inst_id: &Uuid, local_stream_id: Option<u32>) -> Self {
        Self {
            inst_id: inst_id.as_u128(),
            local_stream_id: local_stream_id.unwrap_or(0),
        }
    }
}

/// A fast, thread-safe counter.
#[derive(Debug)]
pub struct Counter {
    count: AtomicIsize,
}

impl Counter {
    /// Creates a new counter starting at the given initial value.
    pub fn new(initial: isize) -> Self {
        Self {
            count: AtomicIsize::new(initial),
        }
    }

    /// Increments the counter by 1.
    pub fn inc(&self) -> isize {
        // Using relaxed ordering because we only care about the counter's value.
        self.count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements the counter by 1.
    pub fn dec(&self) -> isize {
        self.count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Returns the current count.
    pub fn get(&self) -> isize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct IdPool<T> {
    /// The next (never‑allocated) ID.
    next: T,
    /// The set of freed IDs.
    free: BTreeSet<T>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: T) -> Self {
        Self {
            next: T::zero(),
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Result<T> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Ok(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next = self.next + T::one();
            Ok(addr)
        } else {
            // Pool is exhausted.
            Err(Error::msg("ID pool exhausted"))
        }
    }

    pub fn acquire_many(&mut self, count: usize) -> Result<Vec<T>> {
        // check if we have enough available ids
        if self.available() < count {
            return Err(Error::msg("ID pool exhausted"));
        }

        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.acquire()?);
        }
        Ok(result)
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: T) -> Result<()> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(Error::msg("ID was never allocated"));
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }

        Ok(())
    }

    pub fn release_many(&mut self, addrs: &[T]) -> Result<()> {
        for &addr in addrs {
            if addr >= self.next {
                return Err(Error::msg("ID was never allocated"));
            }
            self.free.insert(addr);
        }
        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }
        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next).to_usize().unwrap()
    }

    fn tail_optimization(&mut self) {
        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - T::one() {
                self.free.remove(&last);
                self.next = self.next - T::one();
            } else {
                break;
            }
        }
    }
}

#[derive(Debug)]
pub struct TranslationTable<T> {
    table: HashMap<T, T>,
}

impl<T> TranslationTable<T>
where
    T: Eq + Hash + PrimInt,
{
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    pub fn exists(&self, src: &T) -> bool {
        self.table.contains_key(src)
    }

    pub fn lookup(&self, src: &T) -> Result<T, ObjectError> {
        self.table
            .get(src)
            .cloned()
            .ok_or(ObjectError::ObjectNotFound)
    }

    pub fn lookup_all(&self, srcs: &[T]) -> Result<Vec<T>, ObjectError> {
        srcs.iter().map(|k| self.lookup(k)).collect()
    }

    pub fn translate(&self, src: &mut T) -> Result<(), ObjectError> {
        *src = self.lookup(src)?;
        Ok(())
    }

    pub fn translate_all(&self, srcs: &mut [T]) -> Result<(), ObjectError> {
        for k in srcs.iter_mut() {
            self.translate(k)?;
        }
        Ok(())
    }

    pub fn assign(&mut self, src: T, dst: T) {
        self.table.insert(src, dst);
    }

    pub fn assign_all(&mut self, srcs: &[T], dsts: &[T]) {
        for (key, value) in srcs.iter().zip(dsts.iter()) {
            self.assign(*key, *value);
        }
    }

    pub fn unassign(&mut self, src: &T) -> Result<(), ObjectError> {
        self.table.remove(src).ok_or(ObjectError::ObjectNotFound)?;
        Ok(())
    }

    pub fn unassign_all(&mut self, srcs: &[T]) -> Result<(), ObjectError> {
        srcs.iter().map(|k| self.unassign(k));
        Ok(())
    }

    pub fn to_list(&self) -> Vec<T> {
        self.table.keys().cloned().collect()
    }
}

pub struct RefCounter<T>
where
    T: Eq + Hash + PrimInt,
{
    counters: HashMap<T, Counter>,
}

impl<T> RefCounter<T>
where
    T: Eq + Hash + PrimInt,
{
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
        }
    }

    pub fn init(&mut self, id: T) {
        self.counters.insert(id, Counter::new(1));
    }

    pub fn destroy(&mut self, id: T) {
        self.counters.remove(&id);
    }

    pub fn inc(&self, id: T) {
        self.counters.get(&id).unwrap().inc();
    }

    pub fn dec(&self, id: T) -> bool {
        self.counters.get(&id).unwrap().dec() <= 0
    }

    pub fn get(&self, id: T) -> usize {
        self.counters.get(&id).unwrap().get() as usize
    }

    pub fn init_all(&mut self, ids: &[T]) {
        for id in ids {
            self.init(id.clone());
        }
    }

    pub fn destroy_all(&mut self, ids: &[T]) {
        for id in ids {
            self.destroy(id.clone());
        }
    }

    pub fn inc_all(&self, ids: &[T]) {
        for id in ids {
            self.inc(id.clone());
        }
    }

    pub fn get_all(&self, ids: &[T]) -> Vec<usize> {
        ids.iter().map(|id| self.get(id.clone())).collect()
    }
}
