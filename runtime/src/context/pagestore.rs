//! PageStore — Per-Device CAS Cache + Physical Page Pools.
//!
//! Content-addressed storage for KV cache pages. Pages are identified by
//! chained content hashes (`PageHash`). Sharing across contexts is structural:
//! two contexts with the same token prefix automatically share physical pages.
//!
//! Each model device gets its own `PageStore` — no cross-device coordination.
//! Owned exclusively by the `ContextManager` actor (no interior mutability).

use std::hash::{Hash, Hasher};

use anyhow::Result;
use rustc_hash::{FxHashMap, FxHasher};

use crate::inference::brle::Brle;

// =============================================================================
// Types
// =============================================================================

/// Content hash of a KV cache page, chained to its predecessor.
pub type PageHash = u64;

/// Physical page index in GPU or CPU memory.
pub type PhysicalPageId = u32;


// =============================================================================
// PagePool
// =============================================================================

/// Manages a pool of physical page slots (GPU or CPU) for a single device.
#[derive(Debug)]
struct PagePool {
    total: usize,
    free: Vec<PhysicalPageId>,
}

impl PagePool {
    fn new(total: usize) -> Self {
        PagePool {
            total,
            free: (0..total as PhysicalPageId).collect(),
        }
    }

    fn alloc(&mut self) -> Option<PhysicalPageId> {
        self.free.pop()
    }

    fn free(&mut self, id: PhysicalPageId) {
        self.free.push(id);
    }

    fn used(&self) -> usize {
        self.total - self.free.len()
    }

    fn available(&self) -> usize {
        self.free.len()
    }
}

// =============================================================================
// PageStore
// =============================================================================

/// Per-device page cache. Manages the content-addressed KV cache for one GPU.
///
/// Data structures:
/// - `pages`:       hash → (GPU physical page, refcount)
/// - `chain`:       hash → prev_hash (backward links for traversal, 0 = root)
/// - `index_cache`: tip_hash → flattened `Vec<PhysicalPageId>` from root to tip
///                  Sparse: only cached for active `committed_tip`s.
///
/// Committed pages that are evicted from GPU are simply discarded. They can be
/// replayed from lineage when the context needs to be restored. CPU memory is
/// used exclusively for working page swap (uncommitted pages during suspension).
#[derive(Debug)]
pub struct PageStore {
    page_size: usize,

    /// hash → (GPU physical page ID, refcount).
    /// Present for every GPU-resident committed page.
    pages: FxHashMap<PageHash, (PhysicalPageId, usize)>,

    /// hash → prev_hash for backward chain traversal. 0 = root (no predecessor).
    chain: FxHashMap<PageHash, PageHash>,

    /// Traversal cache: tip_hash → ordered physical page IDs from root to tip.
    /// Sparse: only maintained for hashes that are an active context's `committed_tip`.
    index_cache: FxHashMap<PageHash, Vec<PhysicalPageId>>,

    /// GPU physical page pool.
    gpu: PagePool,

    /// CPU physical page pool (exclusively for working page swap).
    cpu: PagePool,
}

impl PageStore {
    pub fn new(page_size: usize, num_gpu_pages: usize, num_cpu_pages: usize) -> Self {
        PageStore {
            page_size,
            pages: FxHashMap::default(),
            chain: FxHashMap::default(),
            index_cache: FxHashMap::default(),
            gpu: PagePool::new(num_gpu_pages),
            cpu: PagePool::new(num_cpu_pages),
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Returns (used_gpu_pages, total_gpu_pages).
    pub fn stats(&self) -> (usize, usize) {
        (self.gpu.used(), self.gpu.total)
    }



    /// Number of free GPU pages in the pool.
    pub fn available_gpu_pages(&self) -> usize {
        self.gpu.available()
    }

    // =========================================================================
    // Working pages (uncommitted, mutable, exclusive to one context)
    // =========================================================================

    /// Allocate `n` mutable GPU pages from the free pool only.
    /// Does NOT evict — eviction is the caller's responsibility.
    /// Returns None if not enough free pages.
    pub fn alloc_gpu_pages(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        if self.gpu.available() < n {
            return None;
        }
        let mut pages = Vec::with_capacity(n);
        for _ in 0..n {
            pages.push(self.gpu.alloc().unwrap());
        }
        Some(pages)
    }

    /// Free working pages back to the GPU pool.
    pub fn free_gpu_pages(&mut self, pages: &[PhysicalPageId]) {
        for &p in pages {
            self.gpu.free(p);
        }
    }

    // =========================================================================
    // Commit
    // =========================================================================

    /// Promote a working page to a committed page.
    ///
    /// Reuses the working page's physical ID (the forward pass wrote KV data there).
    /// If the hash is a dedup hit, the working page is freed back to the pool
    /// since the existing physical page already has the correct data.
    ///
    /// Returns `(physical_page_id, working_page_freed)`.
    pub fn promote_page(&mut self, hash: PageHash, prev_hash: PageHash, working_phys: PhysicalPageId) -> (PhysicalPageId, bool) {
        // Dedup: page already exists on GPU
        if let Some((phys, rc)) = self.pages.get_mut(&hash) {
            *rc += 1;
            let existing = *phys;
            self.gpu.free(working_phys);
            return (existing, true);
        }

        // Promote: register the working page's physical ID as the committed page
        self.pages.insert(hash, (working_phys, 1));
        self.chain.insert(hash, prev_hash);

        (working_phys, false)
    }

    // =========================================================================
    // Chain traversal & refcounting
    // =========================================================================

    /// Increment refcount for every page reachable from `tip`.
    pub fn acquire_chain(&mut self, tip: PageHash) {
        let mut h = tip;
        loop {
            if let Some((_, rc)) = self.pages.get_mut(&h) {
                *rc += 1;
            }
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
    }

    /// Decrement refcount for every page reachable from `tip`.
    /// Returns hashes that hit refcount=0 (candidates for eviction).
    pub fn release_chain(&mut self, tip: PageHash) -> Vec<PageHash> {
        let mut evictable = Vec::new();
        let mut h = tip;
        loop {
            if let Some((_, rc)) = self.pages.get_mut(&h) {
                *rc = rc.saturating_sub(1);
                if *rc == 0 {
                    evictable.push(h);
                }
            }
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
        evictable
    }

    /// Estimate how many GPU-resident pages would become evictable (rc→0)
    /// if `release_chain(tip)` were called followed by `evict_unreferenced()`.
    /// Read-only: does not modify refcounts.
    pub fn estimate_chain_release(&self, tip: PageHash) -> usize {
        let mut count = 0;
        let mut h = tip;
        loop {
            if let Some((_, rc)) = self.pages.get(&h) {
                if *rc <= 1 { count += 1; }
            }
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
        count
    }

    /// Walk the hash chain backward from `tip` to root, returning hashes in
    /// root-to-tip order.
    pub fn walk_chain(&self, tip: PageHash) -> Vec<PageHash> {
        let mut hashes = Vec::new();
        let mut h = tip;
        loop {
            hashes.push(h);
            match self.chain.get(&h) {
                Some(&prev) if prev != 0 => h = prev,
                _ => break,
            }
        }
        hashes.reverse();
        hashes
    }

    /// Get the refcount for a specific hash.
    pub fn refcount(&self, hash: PageHash) -> usize {
        self.pages.get(&hash).map(|(_, rc)| *rc).unwrap_or(0)
    }

    /// Insert a chain link (hash → prev_hash) without creating a page or refcount entry.
    /// Used by commit_pages_logical for metadata-only commits where no GPU page exists.
    pub fn insert_chain_link(&mut self, hash: PageHash, prev_hash: PageHash) {
        self.chain.insert(hash, prev_hash);
    }

    /// Check if a specific hash has a GPU-resident physical page.
    pub fn is_gpu_resident(&self, hash: PageHash) -> bool {
        self.pages.contains_key(&hash)
    }

    // =========================================================================
    // Eviction
    // =========================================================================

    /// Evict all unreferenced committed pages (rc=0) from GPU.
    /// Returns the number of GPU pages freed.
    pub fn evict_unreferenced(&mut self) -> usize {
        let victims: Vec<PageHash> = self.pages.iter()
            .filter(|(_, (_, rc))| *rc == 0)
            .map(|(&h, _)| h)
            .collect();

        let freed = victims.len();
        for hash in victims {
            if let Some((gpu_phys, _)) = self.pages.remove(&hash) {
                self.gpu.free(gpu_phys);
            }
        }
        freed
    }


    // =========================================================================
    // Prefix lookup (for restore from lineage)
    // =========================================================================

    /// Count the longest prefix of `hashes` present on GPU (read-only).
    /// Does NOT modify refcounts — use `acquire_chain` after successful restore.
    pub fn longest_prefix_length(&self, hashes: &[PageHash]) -> usize {
        hashes.iter().take_while(|h| self.pages.contains_key(h)).count()
    }

    // =========================================================================
    // Physical page resolution
    // =========================================================================

    /// Resolve the ordered list of GPU physical page IDs for a chain tip.
    /// O(1) if the tip is in index_cache; otherwise rebuilds from chain + pages.
    pub fn resolve_physical(&mut self, tip: PageHash) -> Vec<PhysicalPageId> {
        if let Some(cached) = self.index_cache.get(&tip) {
            let chain_len = self.walk_chain(tip).len();
            if cached.len() == chain_len {
                return cached.clone();
            }
            self.index_cache.remove(&tip);
        }
        let chain = self.walk_chain(tip);
        let chain_len = chain.len();
        let table: Vec<PhysicalPageId> = chain.iter()
            .filter_map(|h| self.pages.get(h).map(|(phys, _)| *phys))
            .collect();
        if table.len() == chain_len {
            self.index_cache.insert(tip, table.clone());
        }
        table
    }

    /// Update the index_cache for a context's new committed_tip.
    ///
    /// Move semantics: if `old_tip` is provided and has an index_cache entry,
    /// take it, append the new physical pages, and store under `new_tip`.
    /// Otherwise rebuild from chain + pages.
    pub fn update_index_cache(
        &mut self,
        new_tip: PageHash,
        old_tip: Option<PageHash>,
        new_phys_pages: &[PhysicalPageId],
    ) {
        // All committed pages hit dedup → tip unchanged, cache is correct.
        if old_tip == Some(new_tip) {
            return;
        }

        let mut page_table = match old_tip {
            Some(old) => self.index_cache.remove(&old).unwrap_or_default(),
            None => Vec::new(),
        };

        if page_table.is_empty() && old_tip.is_some() {
            page_table = self.resolve_physical(old_tip.unwrap());
        }

        page_table.extend_from_slice(new_phys_pages);

        let chain_len = self.walk_chain(new_tip).len();
        if page_table.len() == chain_len {
            self.index_cache.insert(new_tip, page_table);
        }
    }

    /// Remove an index_cache entry (e.g., when context is destroyed or suspended).
    pub fn remove_index_cache(&mut self, tip: PageHash) {
        self.index_cache.remove(&tip);
    }

    /// Allocate `n` CPU pages from the swap pool.
    /// Returns None if not enough free pages. Does NOT touch GPU pages.
    pub fn alloc_cpu_pages(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        if self.cpu.available() < n {
            return None;
        }
        let mut pages = Vec::with_capacity(n);
        for _ in 0..n {
            pages.push(self.cpu.alloc().unwrap());
        }
        Some(pages)
    }

    /// Free CPU pages back to the CPU pool.
    /// Used when destroying a suspended context that holds working pages on CPU.
    pub fn free_cpu_pages(&mut self, pages: &[PhysicalPageId]) {
        for &p in pages {
            self.cpu.free(p);
        }
    }

    /// Count committed pages cached on this device (GPU-resident) in a chain.
    pub fn chain_resident_count(&self, tip: PageHash) -> usize {
        let chain = self.walk_chain(tip);
        chain.iter().filter(|h| self.pages.contains_key(h)).count()
    }

    /// Debug: walk chain from tip and return per-hash info (refcount + residency).
    pub fn debug_chain_info(&self, tip: PageHash) -> String {
        let chain = self.walk_chain(tip);
        chain.iter()
            .map(|h| {
                let rc = self.pages.get(h).map(|(_, rc)| *rc).unwrap_or(0);
                let resident = self.pages.contains_key(h);
                format!("rc={rc},res={resident}")
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

// =============================================================================
// Utility
// =============================================================================

/// Compute FlashInfer's `last_page_len` from total KV tokens and page geometry.
///
/// FlashInfer equation: `seq_len = (num_pages - 1) * page_size + last_page_len`
pub fn compute_last_page_len(total_kv: u32, num_pages: u32, page_size: u32) -> u32 {
    if num_pages == 0 {
        0
    } else {
        let r = total_kv % page_size;
        if r == 0 { page_size } else { r }
    }
}

/// Compute chained hashes for a sequence of page-aligned token chunks.
/// Each hash depends on (tokens, positions, masks, prev_hash).
pub fn compute_page_hashes(
    page_size: usize,
    tokens: &[u32],
    positions: &[u32],
    masks: &[Brle],
    prev_hash: PageHash,
) -> Vec<PageHash> {
    let mut hashes = Vec::new();
    let mut running = prev_hash;

    for (chunk_idx, chunk) in tokens.chunks(page_size).enumerate() {
        let start = chunk_idx * page_size;
        let end = start + chunk.len();
        let chunk_pos = &positions[start..end];
        let chunk_masks = &masks[start..end];

        let mut hasher = FxHasher::default();
        chunk.hash(&mut hasher);
        for pos in chunk_pos { pos.hash(&mut hasher); }
        for mask in chunk_masks { mask.hash(&mut hasher); }
        let content_hash = hasher.finish();

        let mut chain_hasher = FxHasher::default();
        content_hash.hash(&mut chain_hasher);
        running.hash(&mut chain_hasher);
        let page_hash = chain_hasher.finish();

        hashes.push(page_hash);
        running = page_hash;
    }

    hashes
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_brle(n: usize) -> Vec<Brle> {
        (0..n).map(|_| Brle::new(0)).collect()
    }

    /// Test helper: allocate a working page, then promote it via promote_page.
    fn alloc_and_commit(store: &mut PageStore, hash: PageHash, prev: PageHash) -> PhysicalPageId {
        let working = store.alloc_gpu_pages(1).unwrap()[0];
        let (phys, _freed) = store.promote_page(hash, prev, working);
        phys
    }

    #[test]
    fn test_commit_and_resolve() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 2);

        let phys1 = alloc_and_commit(&mut store, hashes[0], 0);
        assert!(store.is_gpu_resident(hashes[0]));

        let phys2 = alloc_and_commit(&mut store, hashes[1], hashes[0]);

        store.update_index_cache(hashes[1], None, &[phys1, phys2]);

        let resolved = store.resolve_physical(hashes[1]);
        assert_eq!(resolved, vec![phys1, phys2]);
    }

    #[test]
    fn test_dedup_on_commit() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);

        let phys1 = alloc_and_commit(&mut store, hashes[0], 0);
        assert_eq!(store.refcount(hashes[0]), 1);

        let working2 = store.alloc_gpu_pages(1).unwrap()[0];
        let (phys2, freed) = store.promote_page(hashes[0], 0, working2);
        assert!(freed);
        assert_eq!(phys1, phys2);
        assert_eq!(store.refcount(hashes[0]), 2);
    }

    #[test]
    fn test_acquire_release_chain() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0], 0);
        alloc_and_commit(&mut store, hashes[1], hashes[0]);

        assert_eq!(store.refcount(hashes[0]), 1);
        assert_eq!(store.refcount(hashes[1]), 1);

        store.acquire_chain(hashes[1]);
        assert_eq!(store.refcount(hashes[0]), 2);
        assert_eq!(store.refcount(hashes[1]), 2);

        let evictable = store.release_chain(hashes[1]);
        assert!(evictable.is_empty());
        assert_eq!(store.refcount(hashes[0]), 1);
        assert_eq!(store.refcount(hashes[1]), 1);

        let evictable = store.release_chain(hashes[1]);
        assert_eq!(evictable.len(), 2);
        assert_eq!(store.refcount(hashes[0]), 0);
        assert_eq!(store.refcount(hashes[1]), 0);
    }

    #[test]
    fn test_estimate_chain_release() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..12).collect();
        let positions: Vec<u32> = (0..12).collect();
        let masks = make_brle(12);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 3);

        // Commit 3 pages with rc=1 each
        alloc_and_commit(&mut store, hashes[0], 0);
        alloc_and_commit(&mut store, hashes[1], hashes[0]);
        alloc_and_commit(&mut store, hashes[2], hashes[1]);

        // All rc=1 → all would become evictable
        assert_eq!(store.estimate_chain_release(hashes[2]), 3);

        // Acquire chain (simulating a second context sharing the prefix)
        store.acquire_chain(hashes[1]); // hashes[0]:rc=2, hashes[1]:rc=2

        // Only hashes[2] has rc=1 now
        assert_eq!(store.estimate_chain_release(hashes[2]), 1);

        // Estimate from shared tip: hashes[0]:rc=2, hashes[1]:rc=2 → 0 evictable
        assert_eq!(store.estimate_chain_release(hashes[1]), 0);

        // Verify estimate matches actual release
        let actual = store.release_chain(hashes[2]);
        assert_eq!(actual.len(), 1); // Only hashes[2] hit rc=0
    }

    #[test]
    fn test_shared_pages_survive_eviction() {
        let mut store = PageStore::new(4, 10, 5);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0], 0);
        alloc_and_commit(&mut store, hashes[1], hashes[0]);

        store.acquire_chain(hashes[1]);
        store.release_chain(hashes[1]);

        let freed = store.evict_unreferenced();
        assert_eq!(freed, 0);
        assert!(store.is_gpu_resident(hashes[0]));
        assert!(store.is_gpu_resident(hashes[1]));
    }

    #[test]
    fn test_evict_unreferenced() {
        let mut store = PageStore::new(4, 10, 5);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0], 0);

        store.release_chain(hashes[0]);

        let freed = store.evict_unreferenced();
        assert_eq!(freed, 1);
        assert!(!store.is_gpu_resident(hashes[0]));
    }

    #[test]
    fn test_working_pages_alloc_free() {
        let mut store = PageStore::new(4, 10, 5);

        let pages = store.alloc_gpu_pages(3).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(store.gpu.used(), 3);

        store.free_gpu_pages(&pages);
        assert_eq!(store.gpu.used(), 0);
    }

    #[test]
    fn test_alloc_gpu_pages_returns_none_when_full() {
        let mut store = PageStore::new(4, 3, 5);

        let pages = store.alloc_gpu_pages(3).unwrap();
        assert_eq!(pages.len(), 3);

        // No more pages available
        assert!(store.alloc_gpu_pages(1).is_none());

        store.free_gpu_pages(&pages);
        assert!(store.alloc_gpu_pages(1).is_some());
    }

    #[test]
    fn test_cpu_pages_alloc_free() {
        let mut store = PageStore::new(4, 10, 5);

        // Allocate GPU working pages
        let gpu_pages = store.alloc_gpu_pages(2).unwrap();
        assert_eq!(store.gpu.used(), 2);

        // Allocate CPU pages (simulating D2H copy target)
        let cpu_pages = store.alloc_cpu_pages(2).unwrap();
        assert_eq!(cpu_pages.len(), 2);
        assert_eq!(store.cpu.used(), 2);

        // Free GPU pages (simulating post-D2H swap-out)
        store.free_gpu_pages(&gpu_pages);
        assert_eq!(store.gpu.used(), 0);

        // Re-allocate GPU pages (simulating H2D swap-in target)
        let new_gpu = store.alloc_gpu_pages(2).unwrap();
        assert_eq!(store.gpu.used(), 2);

        // Free CPU pages (simulating post-H2D swap-in)
        store.free_cpu_pages(&cpu_pages);
        assert_eq!(store.cpu.used(), 0);

        store.free_gpu_pages(&new_gpu);
    }

    #[test]
    fn test_longest_prefix_length() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..12).collect();
        let positions: Vec<u32> = (0..12).collect();
        let masks = make_brle(12);
        let hashes = compute_page_hashes(store.page_size,&tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 3);

        alloc_and_commit(&mut store, hashes[0], 0);
        alloc_and_commit(&mut store, hashes[1], hashes[0]);

        let matched = store.longest_prefix_length(&hashes);
        assert_eq!(matched, 2);
    }

    #[test]
    fn test_index_cache_move_on_commit() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        let phys1 = alloc_and_commit(&mut store, hashes[0], 0);
        store.update_index_cache(hashes[0], None, &[phys1]);

        let phys2 = alloc_and_commit(&mut store, hashes[1], hashes[0]);
        store.update_index_cache(hashes[1], Some(hashes[0]), &[phys2]);

        assert!(!store.index_cache.contains_key(&hashes[0]));
        let resolved = store.resolve_physical(hashes[1]);
        assert_eq!(resolved, vec![phys1, phys2]);
    }
}
