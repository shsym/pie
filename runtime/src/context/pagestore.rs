//! PageStore — Per-Device CAS Cache + Physical Page Pools.
//!
//! Content-addressed storage for KV cache pages. Pages are identified by
//! chained content hashes (`PageHash`). Sharing across contexts is structural:
//! two contexts with the same token prefix automatically share physical pages.
//!
//! Each model device gets its own `PageStore` — no cross-device coordination.
//! Owned exclusively by the `ContextManager` actor (no interior mutability).

use std::hash::{Hash, Hasher};

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
/// - `pages`: hash → (GPU physical page, refcount)
///
/// Chain topology (hash ordering) is owned by `Context.committed_hashes`.
/// Committed pages that are evicted from GPU are simply discarded. They can be
/// replayed from lineage when the context needs to be restored. CPU memory is
/// used exclusively for working page swap (uncommitted pages during suspension).
#[derive(Debug)]
pub struct PageStore {
    page_size: usize,

    /// hash → (GPU physical page ID, refcount).
    /// Present for every GPU-resident committed page.
    committed_gpu_pages: FxHashMap<PageHash, (PhysicalPageId, usize)>,

    /// GPU physical page pool.
    gpu: PagePool,

    /// CPU physical page pool (exclusively for working page swap).
    cpu: PagePool,
}

impl PageStore {
    pub fn new(page_size: usize, num_gpu_pages: usize, num_cpu_pages: usize) -> Self {
        PageStore {
            page_size,
            committed_gpu_pages: FxHashMap::default(),
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

    /// Commit a page to the GPU cache (dedup-aware).
    /// If hash already exists, bumps rc and frees the duplicate `phys` page
    /// back to the GPU pool. If new, inserts with rc=1.
    pub fn commit(&mut self, hash: PageHash, phys: PhysicalPageId) {
        if let Some((_, rc)) = self.committed_gpu_pages.get_mut(&hash) {
            *rc += 1;
            self.gpu.free(phys);
        } else {
            self.committed_gpu_pages.insert(hash, (phys, 1));
        }
    }

    // =========================================================================
    // Refcounting (slice-based)
    // =========================================================================

    /// Increment refcount for every hash in the slice.
    pub fn retain(&mut self, hashes: &[PageHash]) {
        for &h in hashes {
            if let Some((_, rc)) = self.committed_gpu_pages.get_mut(&h) {
                *rc += 1;
            }
        }
    }

    /// Decrement refcount for every hash in the slice.
    /// Pages that hit rc=0 are eagerly evicted: removed from `self.committed_gpu_pages`
    /// and their GPU physical page freed back to the pool.
    /// Returns the number of GPU pages freed.
    pub fn release(&mut self, hashes: &[PageHash]) -> usize {
        let mut freed = 0;
        for &h in hashes {
            if let Some((_, rc)) = self.committed_gpu_pages.get_mut(&h) {
                *rc = rc.saturating_sub(1);
                if *rc == 0 {
                    if let Some((gpu_phys, _)) = self.committed_gpu_pages.remove(&h) {
                        self.gpu.free(gpu_phys);
                        freed += 1;
                    }
                }
            }
        }
        freed
    }

    /// Estimate how many GPU-resident pages would be freed (rc→0)
    /// if `release(hashes)` were called.
    /// Read-only: does not modify refcounts.
    pub fn count_reclaimable(&self, hashes: &[PageHash]) -> usize {
        hashes.iter().filter(|h| {
            self.committed_gpu_pages.get(h).map(|(_, rc)| *rc <= 1).unwrap_or(false)
        }).count()
    }

    /// Get the refcount for a specific hash.
    pub fn refcount(&self, hash: PageHash) -> usize {
        self.committed_gpu_pages.get(&hash).map(|(_, rc)| *rc).unwrap_or(0)
    }

    /// Check if a specific hash has a GPU-resident physical page.
    pub fn contains(&self, hash: PageHash) -> bool {
        self.committed_gpu_pages.contains_key(&hash)
    }

    // =========================================================================
    // Prefix lookup (for restore from lineage)
    // =========================================================================

    /// Count the longest prefix of `hashes` present on GPU (read-only).
    /// Does NOT modify refcounts — use `retain` after successful restore.
    pub fn prefix_len(&self, hashes: &[PageHash]) -> usize {
        hashes.iter().take_while(|h| self.committed_gpu_pages.contains_key(h)).count()
    }

    /// Resolve physical page IDs for a list of hashes.
    /// Returns a Vec of the GPU physical page IDs for each hash that is resident.
    pub fn physical_ids(&self, hashes: &[PageHash]) -> Vec<PhysicalPageId> {
        hashes.iter()
            .filter_map(|h| self.committed_gpu_pages.get(h).map(|(phys, _)| *phys))
            .collect()
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

    /// Count committed pages cached on this device (GPU-resident).
    pub fn cached_count(&self, hashes: &[PageHash]) -> usize {
        hashes.iter().filter(|h| self.committed_gpu_pages.contains_key(h)).count()
    }

    /// Debug: return per-hash info (refcount + residency).
    pub fn debug_info(&self, hashes: &[PageHash]) -> String {
        hashes.iter()
            .map(|h| {
                let rc = self.committed_gpu_pages.get(h).map(|(_, rc)| *rc).unwrap_or(0);
                let resident = self.committed_gpu_pages.contains_key(h);
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

    /// Test helper: allocate a working page, then commit it (dedup-aware).
    fn alloc_and_commit(store: &mut PageStore, hash: PageHash) -> PhysicalPageId {
        let working = store.alloc_gpu_pages(1).unwrap()[0];
        store.commit(hash, working);
        store.physical_ids(&[hash])[0]
    }

    #[test]
    fn test_commit_and_resolve() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 2);

        let phys1 = alloc_and_commit(&mut store, hashes[0]);
        assert!(store.contains(hashes[0]));

        let phys2 = alloc_and_commit(&mut store, hashes[1]);

        let resolved = store.physical_ids(&hashes);
        assert_eq!(resolved, vec![phys1, phys2]);
    }

    #[test]
    fn test_dedup_on_commit() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        let phys1 = alloc_and_commit(&mut store, hashes[0]);
        assert_eq!(store.refcount(hashes[0]), 1);

        let working2 = store.alloc_gpu_pages(1).unwrap()[0];
        // Dedup: same hash already exists — commit auto-frees working2
        store.commit(hashes[0], working2);
        let phys2 = store.physical_ids(&[hashes[0]])[0];
        assert_eq!(phys1, phys2);
        assert_eq!(store.refcount(hashes[0]), 2);
    }

    #[test]
    fn test_retain_release() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0]);
        alloc_and_commit(&mut store, hashes[1]);

        assert_eq!(store.refcount(hashes[0]), 1);
        assert_eq!(store.refcount(hashes[1]), 1);

        store.retain(&hashes);
        assert_eq!(store.refcount(hashes[0]), 2);
        assert_eq!(store.refcount(hashes[1]), 2);

        let freed = store.release(&hashes);
        assert_eq!(freed, 0);
        assert_eq!(store.refcount(hashes[0]), 1);
        assert_eq!(store.refcount(hashes[1]), 1);

        let freed = store.release(&hashes);
        assert_eq!(freed, 2);
        assert!(!store.contains(hashes[0]));
        assert!(!store.contains(hashes[1]));
    }

    #[test]
    fn test_count_reclaimable() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..12).collect();
        let positions: Vec<u32> = (0..12).collect();
        let masks = make_brle(12);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 3);

        // Commit 3 pages with rc=1 each
        alloc_and_commit(&mut store, hashes[0]);
        alloc_and_commit(&mut store, hashes[1]);
        alloc_and_commit(&mut store, hashes[2]);

        // All rc=1 → all would become evictable
        assert_eq!(store.count_reclaimable(&hashes), 3);

        // Acquire prefix (simulating a second context sharing the prefix)
        store.retain(&hashes[..2]); // hashes[0]:rc=2, hashes[1]:rc=2

        // Only hashes[2] has rc=1 now
        assert_eq!(store.count_reclaimable(&hashes), 1);

        // Estimate from shared prefix only: hashes[0]:rc=2, hashes[1]:rc=2 → 0 evictable
        assert_eq!(store.count_reclaimable(&hashes[..2]), 0);

        // Verify release matches estimate: only hashes[2] hit rc=0
        let freed = store.release(&hashes);
        assert_eq!(freed, 1);
        assert!(!store.contains(hashes[2]));
    }

    #[test]
    fn test_shared_pages_survive_release() {
        let mut store = PageStore::new(4, 10, 5);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0]);
        alloc_and_commit(&mut store, hashes[1]);

        store.retain(&hashes);
        store.release(&hashes);

        // Shared pages (rc=1) survive
        assert!(store.contains(hashes[0]));
        assert!(store.contains(hashes[1]));
    }

    #[test]
    fn test_release_frees_unreferenced() {
        let mut store = PageStore::new(4, 10, 5);

        let tokens: Vec<u32> = (0..4).collect();
        let positions: Vec<u32> = (0..4).collect();
        let masks = make_brle(4);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        alloc_and_commit(&mut store, hashes[0]);

        let freed = store.release(&hashes[..1]);
        assert_eq!(freed, 1);
        assert!(!store.contains(hashes[0]));
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
    fn test_prefix_len() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..12).collect();
        let positions: Vec<u32> = (0..12).collect();
        let masks = make_brle(12);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);
        assert_eq!(hashes.len(), 3);

        alloc_and_commit(&mut store, hashes[0]);
        alloc_and_commit(&mut store, hashes[1]);

        let matched = store.prefix_len(&hashes);
        assert_eq!(matched, 2);
    }

    #[test]
    fn test_physical_ids() {
        let mut store = PageStore::new(4, 100, 10);

        let tokens: Vec<u32> = (0..8).collect();
        let positions: Vec<u32> = (0..8).collect();
        let masks = make_brle(8);
        let hashes = compute_page_hashes(store.page_size, &tokens, &positions, &masks, 0);

        let phys1 = alloc_and_commit(&mut store, hashes[0]);
        let phys2 = alloc_and_commit(&mut store, hashes[1]);

        let resolved = store.physical_ids(&hashes);
        assert_eq!(resolved, vec![phys1, phys2]);
    }
}
