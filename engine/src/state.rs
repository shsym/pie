use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::atomic::{AtomicIsize, Ordering};

use std::time::{Duration, Instant};
use uuid::Uuid;

pub type InstanceId = Uuid;
pub type Address = usize;
pub type BlockPtr = usize;

// ------------------------------------------------------------

pub struct KvBlock {
    ptr: (u8, BlockPtr),
    references: Counter,
    recency: RecencyTracker,

    // pos ids and vacancy maps
    position_ids: Vec<u32>,
    occupied: Vec<bool>,
    filled: bool,
}

impl KvBlock {
    pub fn new(storage_id: u8, ptr: BlockPtr, block_size: usize) -> Self {
        Self {
            ptr: (storage_id, ptr),
            references: Counter::new(0),
            recency: RecencyTracker::new(),
            position_ids: vec![0; block_size],
            occupied: vec![false; block_size],
            filled: false,
        }
    }
}

enum BlockStorageAction {
    Allocate(Address),
    Deallocate(Address),
    Copy(Address, Address, usize, usize, usize),
}

// The "mirror" of the block in the storage in the backend
pub struct BlockStorage {
    block_size: usize,
    ptr_pool: AddressPool,
    staged_actions: Vec<BlockStorageAction>,
}

impl BlockStorage {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            ptr_pool: AddressPool::new(max_capacity),
            staged_actions: Vec::new(),
        }
    }

    pub fn allocate(&mut self) -> Result<Address, BlockError> {
        let ptr = self.ptr_pool.acquire().ok_or(BlockError::NoFreeBlocks)?;

        self.staged_actions
            .push(BlockStorageAction::Allocate(ptr));
        Ok(ptr)
    }

    pub fn deallocate(&mut self, ptr: Address) -> Result<(), BlockError> {
        self.ptr_pool.release(ptr)?;
        self.staged_actions
            .push(BlockStorageAction::Deallocate(ptr));
        Ok(())
    }

    pub fn copy(
        &mut self,
        src_ptr: Address,
        dst_ptr: Address,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        self.staged_actions.push(BlockStorageAction::Copy(
            src_ptr, dst_ptr, src_offset, dst_offset, size,
        ));
        Ok(())
    }
}

pub struct KvBlockManager {
    global_addr_map: HashMap<Address, KvBlock>,
    virtual_addr_maps: HashMap<InstanceId, AddressMap>,
    global_addr_pool: AddressPool,

    // primary storage
    storage: BlockStorage,
    secondary_storages: Vec<BlockStorage>,
}

impl KvBlockManager {
    pub fn new(storage: BlockStorage) -> Self {
        Self {
            global_addr_map: HashMap::new(),
            virtual_addr_maps: HashMap::new(),
            global_addr_pool: AddressPool::new(Address::MAX),
            storage,
            secondary_storages: Vec::new(),
        }
    }

    fn resolve_addr(
        &self,
        inst_id: InstanceId,
        virtual_addr: Address,
    ) -> Result<Address, BlockError> {
        let addr_map = self
            .virtual_addr_maps
            .get(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?;
        addr_map.resolve(virtual_addr)
    }

    pub fn create_addr_map(&mut self, inst_id: InstanceId) {
        self.virtual_addr_maps.insert(inst_id, AddressMap::new());
    }

    pub fn destroy_addr_map(&mut self, inst_id: InstanceId) {
        self.virtual_addr_maps.remove(&inst_id);
    }

    pub fn get_block(
        &self,
        inst_id: InstanceId,
        virtual_addr: Address,
    ) -> Result<&KvBlock, BlockError> {
        let global_addr = self.resolve_addr(inst_id, virtual_addr)?;
        self.global_addr_map
            .get(&global_addr)
            .ok_or(BlockError::BlockNotFound)
    }

    pub fn get_block_mut(
        &mut self,
        instance_id: InstanceId,
        virtual_addr: Address,
    ) -> Result<&mut KvBlock, BlockError> {
        let global_addr = self.resolve_addr(instance_id, virtual_addr)?;
        self.global_addr_map
            .get_mut(&global_addr)
            .ok_or(BlockError::BlockNotFound)
    }

    pub fn allocate_block(&mut self, inst_id: InstanceId) -> Result<Address, BlockError> {
        // create a new id
        let global_addr = self
            .global_addr_pool
            .acquire()
            .ok_or(BlockError::NoFreeBlocks)?;

        // create a new virtual address
        let virtual_addr = self
            .virtual_addr_maps
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .register(global_addr);

        let block_ptr = self.storage.allocate()?;
        let block = KvBlock::new(0, block_ptr, self.storage.block_size);

        // increase ref count
        block.references.inc();

        self.global_addr_map.insert(global_addr, block);
        Ok(virtual_addr)
    }

    pub fn deallocate_block(
        &mut self,
        inst_id: InstanceId,
        virtual_addr: Address,
    ) -> Result<(), BlockError> {
        // remove and get the global address
        let global_addr = self
            .virtual_addr_maps
            .get_mut(&inst_id)
            .ok_or(BlockError::InstanceNotFound)?
            .unregister(virtual_addr)?;

        let ref_count = self
            .global_addr_map
            .get(&global_addr)
            .ok_or(BlockError::BlockNotFound)?
            .references
            .dec();

        // remove the block if the ref count is 0
        if ref_count <= 0 {
            let block = self
                .global_addr_map
                .remove(&global_addr)
                .ok_or(BlockError::BlockNotFound)?;

            self.storage.deallocate(block.ptr.1)?;
            self.global_addr_map.remove(&global_addr);
            self.global_addr_pool.release(global_addr)?;
        }

        Ok(())
    }

    pub fn copy_tokens(
        &mut self,
        inst_id: InstanceId,
        src_virtual_addr: Address,
        dst_virtual_addr: Address,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> Result<(), BlockError> {
        let (_, src_block_ptr) = self.get_block(inst_id, src_virtual_addr)?.ptr;
        let (_, dst_block_ptr) = self.get_block(inst_id, dst_virtual_addr)?.ptr;

        self.storage
            .copy(src_block_ptr, dst_block_ptr, src_offset, dst_offset, size)?;

        // First, get a temporary copy of the data from the source block.
        // Here we use an immutable borrow so that we can later borrow the destination mutably.
        let (src_position_ids, src_occupied) = {
            let src_block = self.get_block(inst_id, src_virtual_addr)?;
            (
                src_block.position_ids[src_offset..src_offset + size].to_vec(),
                src_block.occupied[src_offset..src_offset + size].to_vec(),
            )
        };

        // Now get a mutable borrow of the destination block and update its data.
        let dst_block = self.get_block_mut(inst_id, dst_virtual_addr)?;
        for i in 0..size {
            dst_block.position_ids[dst_offset + i] = src_position_ids[i];
            dst_block.occupied[dst_offset + i] = src_occupied[i];
        }
        Ok(())
    }

    pub fn mask_tokens(
        &mut self,
        inst_id: InstanceId,
        virtual_addr: Address,
        mask: Vec<bool>,
    ) -> Result<(), BlockError> {
        let block = self.get_block_mut(inst_id, virtual_addr)?;
        for i in 0..mask.len() {
            block.occupied[i] = mask[i];
        }
        Ok(())
    }

    pub fn available_blocks(&self) -> usize {
        self.storage.ptr_pool.available()
    }
}
// ------------------------------------------------------------

pub struct TokenEmbed {
    ptr: BlockPtr,
}

// ------------------------------------------------------------

#[derive(Debug)]
pub enum BlockError {
    NoFreeBlocks,
    NotEnoughFreeBlocks { requested: usize, available: usize },
    VirtualAddressTranslationFailed,
    BlockNotFound,
    InstanceNotFound,
}

// ------------------------------------------------------------
// AddressMap
// ------------------------------------------------------------
#[derive(Debug)]
pub struct AddressMap {
    mapping: HashMap<Address, Address>,
    addr_pool: AddressPool,
}

impl AddressMap {
    pub fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            addr_pool: AddressPool::new(Address::MAX),
        }
    }

    pub fn register(&mut self, addr: Address) -> Address {
        let virtual_addr = self.addr_pool.acquire().unwrap();
        self.mapping.insert(virtual_addr, addr);
        virtual_addr
    }

    pub fn unregister(&mut self, virtual_addr: Address) -> Result<Address, BlockError> {
        let global_addr = self
            .mapping
            .remove(&virtual_addr)
            .ok_or(BlockError::BlockNotFound)?;
        self.addr_pool.release(virtual_addr)?;

        Ok(global_addr)
    }

    pub fn resolve(&self, virtual_addr: Address) -> Result<Address, BlockError> {
        self.mapping
            .get(&virtual_addr)
            .copied()
            .ok_or(BlockError::VirtualAddressTranslationFailed)
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
pub struct AddressPool {
    /// The next (never‑allocated) ID.
    next: Address,
    /// The set of freed IDs.
    free: BTreeSet<Address>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: usize,
}

impl AddressPool {
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            next: 0,
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Option<Address> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Some(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next += 1;
            Some(addr)
        } else {
            // Pool is exhausted.
            None
        }
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: Address) -> Result<(), BlockError> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(BlockError::VirtualAddressTranslationFailed);
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - 1 {
                self.free.remove(&last);
                self.next -= 1;
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next)
    }
}

/// A tracker that records the recency of usage.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecencyTracker {
    last_used: Option<Instant>,
}

impl RecencyTracker {
    /// Creates a new `RecencyTracker` with no recorded usage.
    pub fn new() -> Self {
        Self { last_used: None }
    }

    /// Returns the timestamp of the last use, or `None` if never refreshed.
    pub fn last_used(&self) -> Option<Instant> {
        self.last_used
    }

    /// Updates the tracker to the current time.
    pub fn refresh_last_used(&mut self) {
        self.last_used = Some(Instant::now());
    }
}
