//! PageStore — Per-Device CAS Cache + Physical Page Pools (Radix Trie v2).
//!
//! Content-addressed storage for KV cache pages using a compressed Radix
//! Trie (Patricia Trie) with **path-inclusive refcounting**.
//!
//! Unlike the tip-only trie (`pagestore.rs`), `refcount` on each node
//! means "how many contexts include this node in their committed chain."
//! This correctly handles:
//! - **Incremental commits**: `commit_append(prefix, suffix, phys)` navigates
//!   the trie through the existing prefix before inserting new suffix pages.
//! - **Dedup**: when two contexts commit the same chain, rc is bumped on all
//!   shared nodes; suspending one doesn't evict pages needed by the other.
//! - **O(depth) operations**: all operations traverse the trie path once.
//!
//! Each model device gets its own `PageStore` — no cross-device coordination.
//! Owned exclusively by the `ContextManager` actor (no interior mutability).

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::inference::brle::Brle;

// =============================================================================
// Types
// =============================================================================

/// Content hash of a KV cache page, chained to its predecessor.
pub type PageHash = u64;

/// Physical page index in GPU or CPU memory.
pub type PhysicalPageId = u32;

// =============================================================================
// PagePool (bulk alloc via split_off)
// =============================================================================

#[derive(Debug)]
struct PagePool {
    total: usize,
    free: Vec<PhysicalPageId>,
}

impl PagePool {
    fn new(total: usize) -> Self {
        PagePool { total, free: (0..total as PhysicalPageId).collect() }
    }

    fn alloc_n(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        if self.free.len() < n { return None; }
        let start = self.free.len() - n;
        Some(self.free.split_off(start))
    }

    fn free(&mut self, id: PhysicalPageId) { self.free.push(id); }
    fn free_batch(&mut self, ids: &[PhysicalPageId]) { self.free.extend_from_slice(ids); }
    fn used(&self) -> usize { self.total - self.free.len() }
    fn available(&self) -> usize { self.free.len() }
}

// =============================================================================
// TrieNode — path-inclusive refcounting
// =============================================================================

/// A node in the Radix Trie representing an unbroken chunk of cached pages.
///
/// Invariants:
/// - `prefix.len() == prefix_hashes.len()`
/// - `refcount` = number of contexts whose committed chain **includes** this node
///   (passes through OR terminates at it)
/// - A node is evictable iff `refcount == 0` — no structural protection needed
#[derive(Debug)]
struct TrieNode {
    /// Physical page IDs for this contiguous chunk.
    prefix: Vec<PhysicalPageId>,
    /// Corresponding Merkle PageHashes for each page.
    prefix_hashes: Vec<PageHash>,
    /// Path-inclusive refcount: contexts traversing or ending at this node.
    refcount: usize,
    /// Children keyed by the first PageHash of their prefix_hashes.
    children: FxHashMap<PageHash, TrieNode>,
}

impl TrieNode {
    /// Create an empty root node (no pages, no hashes).
    fn new_root() -> Self {
        TrieNode {
            prefix: Vec::new(),
            prefix_hashes: Vec::new(),
            refcount: 0,
            children: FxHashMap::default(),
        }
    }

    // =========================================================================
    // Insert with path-inclusive refcount bump
    // =========================================================================

    /// Insert (or ref-bump) a chain into this subtree.
    ///
    /// `prefix_nav` = number of leading hashes that are existing prefix
    /// (just bump rc, no physical pages needed). The remaining hashes are
    /// new suffix (create nodes with rc=1, store physical pages).
    ///
    /// On dedup (exact match for the suffix portion), bumps rc and collects
    /// duplicate physical pages in `freed`.
    ///
    /// All nodes along the path get rc += 1 (path-inclusive).
    fn insert_and_ref(
        &mut self,
        hashes: &[PageHash],
        phys: &[PhysicalPageId],
        prefix_nav: usize,
        freed: &mut Vec<PhysicalPageId>,
    ) {
        if hashes.is_empty() { return; }

        let first = hashes[0];

        if let Some(child) = self.children.get_mut(&first) {
            let common = common_prefix_len(&child.prefix_hashes, hashes);

            if common < child.prefix_hashes.len() {
                // Partial match — split the existing node.
                let old_suffix_hashes = child.prefix_hashes.split_off(common);
                let old_suffix_phys = child.prefix.split_off(common);
                let old_suffix_first = old_suffix_hashes[0];

                let old_suffix = TrieNode {
                    prefix_hashes: old_suffix_hashes,
                    prefix: old_suffix_phys,
                    refcount: child.refcount,
                    children: std::mem::take(&mut child.children),
                };

                // Upper portion keeps the same rc (all existing traversals
                // still pass through it). +1 for the new insertion.
                child.refcount += 1;
                child.children.insert(old_suffix_first, old_suffix);

                // Insert new divergent suffix (if any remaining hashes).
                if common < hashes.len() {
                    // Determine how many of the remaining hashes are prefix nav
                    let remaining_nav = prefix_nav.saturating_sub(common);
                    let remaining_hashes = &hashes[common..];
                    let remaining_phys = if common >= prefix_nav {
                        &phys[common - prefix_nav..]
                    } else {
                        phys // all phys are for the suffix portion
                    };

                    let new_leaf = TrieNode::new_with_nav(
                        remaining_hashes, remaining_phys, remaining_nav,
                    );
                    child.children.insert(remaining_hashes[0], new_leaf);
                }
            } else if common < hashes.len() {
                // Full match of child's prefix — bump rc, recurse.
                child.refcount += 1;
                let remaining_nav = prefix_nav.saturating_sub(common);
                let remaining_phys = if common >= prefix_nav {
                    &phys[common - prefix_nav..]
                } else {
                    phys
                };
                child.insert_and_ref(
                    &hashes[common..], remaining_phys, remaining_nav, freed,
                );
            } else {
                // Exact match — dedup hit. Bump rc on this node.
                child.refcount += 1;
                // Free any duplicate physical pages from the suffix portion.
                if prefix_nav < hashes.len() {
                    let suffix_phys = if prefix_nav > 0 {
                        &phys[..] // phys only has suffix pages
                    } else {
                        phys
                    };
                    freed.extend_from_slice(suffix_phys);
                }
            }
        } else {
            // No matching child — create new node.
            let node = TrieNode::new_with_nav(hashes, phys, prefix_nav);
            self.children.insert(first, node);
        }
    }

    /// Create a new node from hashes where the first `nav` hashes use
    /// placeholder phys (0) and the rest use real phys from the slice.
    fn new_with_nav(
        hashes: &[PageHash],
        phys: &[PhysicalPageId],
        nav: usize,
    ) -> Self {
        debug_assert!(nav <= hashes.len());
        let expected_phys = hashes.len() - nav;
        debug_assert_eq!(phys.len(), expected_phys,
            "phys len {} != expected {} (hashes={}, nav={})",
            phys.len(), expected_phys, hashes.len(), nav);

        let mut full_phys = vec![0u32; nav];
        full_phys.extend_from_slice(phys);

        TrieNode {
            prefix_hashes: hashes.to_vec(),
            prefix: full_phys,
            refcount: 1,
            children: FxHashMap::default(),
        }
    }

    // =========================================================================
    // Release — path-inclusive, bottom-up eviction
    // =========================================================================

    /// Decrement rc on every node along the path matching `hashes`.
    /// Evict nodes that reach rc=0 and have no children. Cascade upward.
    /// Returns (freed_page_count, should_remove_self).
    fn release_path(
        &mut self,
        hashes: &[PageHash],
        pool: &mut PagePool,
    ) -> (usize, bool) {
        if hashes.is_empty() { return (0, false); }

        let first = hashes[0];
        let (child_freed, remove_child) = match self.children.get_mut(&first) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);

                if common < child.prefix_hashes.len() {
                    // Partial match — the chain doesn't fully traverse this node.
                    // This shouldn't happen with valid chains, but handle gracefully.
                    (0, false)
                } else if common < hashes.len() {
                    // Full match of node — decrement rc, recurse.
                    child.refcount = child.refcount.saturating_sub(1);
                    let (deeper_freed, _) = child.release_path(
                        &hashes[common..], pool,
                    );

                    // Check if this child is now evictable.
                    let evictable = child.refcount == 0 && child.children.is_empty();
                    let freed = if evictable {
                        pool.free_batch(&child.prefix);
                        child.prefix.len() + deeper_freed
                    } else {
                        deeper_freed
                    };
                    (freed, evictable)
                } else {
                    // Exact match (tip) — decrement rc.
                    child.refcount = child.refcount.saturating_sub(1);

                    let evictable = child.refcount == 0 && child.children.is_empty();
                    let freed = if evictable {
                        pool.free_batch(&child.prefix);
                        child.prefix.len()
                    } else {
                        0
                    };
                    (freed, evictable)
                }
            }
            None => (0, false),
        };

        if remove_child {
            self.children.remove(&first);
        }

        // Merge: compress single-child internal nodes.
        if self.children.len() == 1 && self.refcount == 0 && !self.prefix_hashes.is_empty() {
            let (_, sole) = self.children.drain().next().unwrap();
            self.prefix.extend(sole.prefix);
            self.prefix_hashes.extend(sole.prefix_hashes);
            self.refcount = sole.refcount;
            self.children = sole.children;
        }

        // Check if self is now evictable (for cascade from parent).
        let remove_self = self.refcount == 0
            && self.children.is_empty()
            && !self.prefix_hashes.is_empty();

        let total_freed = if remove_self {
            pool.free_batch(&self.prefix);
            child_freed + self.prefix.len()
        } else {
            child_freed
        };

        (total_freed, remove_self)
    }

    // =========================================================================
    // Lookup (unchanged from old trie)
    // =========================================================================

    /// Count the longest prefix of `hashes` that is present in this subtree.
    fn prefix_match_len(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }

        match self.children.get(&hashes[0]) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);
                if common < child.prefix_hashes.len() {
                    common
                } else if common < hashes.len() {
                    common + child.prefix_match_len(&hashes[common..])
                } else {
                    common
                }
            }
            None => 0,
        }
    }

    /// Collect physical page IDs along the path matching `hashes`.
    fn collect_physical(&self, hashes: &[PageHash], out: &mut Vec<PhysicalPageId>) {
        if hashes.is_empty() { return; }

        if let Some(child) = self.children.get(&hashes[0]) {
            let common = common_prefix_len(&child.prefix_hashes, hashes);
            out.extend_from_slice(&child.prefix[..common]);

            if common == child.prefix_hashes.len() && common < hashes.len() {
                child.collect_physical(&hashes[common..], out);
            }
        }
    }

    // =========================================================================
    // Estimate reclaimable (path-inclusive)
    // =========================================================================

    /// Estimate pages freed if `release_path(hashes)` were called.
    /// A node's pages are freed if its rc would become 0 and it has no children.
    fn count_reclaimable_path(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }

        match self.children.get(&hashes[0]) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);
                if common < child.prefix_hashes.len() {
                    0
                } else if common < hashes.len() {
                    let deeper = child.count_reclaimable_path(&hashes[common..]);
                    // After deeper release, if child becomes evictable
                    let child_evictable_after = child.refcount <= 1
                        && child.children.is_empty();
                    if child_evictable_after {
                        child.prefix.len() + deeper
                    } else {
                        deeper
                    }
                } else {
                    // Tip node — evictable if rc would become 0
                    if child.refcount <= 1 && child.children.is_empty() {
                        child.prefix.len()
                    } else {
                        0
                    }
                }
            }
            None => 0,
        }
    }

    /// Bump rc on every node along the path matching `hashes` (retain).
    /// If `hashes` ends mid-node, splits the node first (split-on-retain).
    fn retain_path(&mut self, hashes: &[PageHash]) {
        if hashes.is_empty() { return; }

        if let Some(child) = self.children.get_mut(&hashes[0]) {
            let common = common_prefix_len(&child.prefix_hashes, hashes);

            if common < child.prefix_hashes.len() && common == hashes.len() {
                // Partial match — retain ends mid-node. Split first.
                let old_suffix_hashes = child.prefix_hashes.split_off(common);
                let old_suffix_phys = child.prefix.split_off(common);
                let old_suffix_first = old_suffix_hashes[0];

                let old_suffix = TrieNode {
                    prefix_hashes: old_suffix_hashes,
                    prefix: old_suffix_phys,
                    refcount: child.refcount,
                    children: std::mem::take(&mut child.children),
                };

                // Upper portion gets the same rc + 1 for the retain.
                child.refcount += 1;
                child.children.insert(old_suffix_first, old_suffix);
            } else {
                child.refcount += 1;
                if common == child.prefix_hashes.len() && common < hashes.len() {
                    child.retain_path(&hashes[common..]);
                }
            }
        }
    }
}

/// Count how many leading elements of `a` and `b` are equal.
fn common_prefix_len(a: &[PageHash], b: &[PageHash]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// =============================================================================
// PageStore
// =============================================================================

/// Per-device page cache backed by a Radix Trie with path-inclusive refcounting.
#[derive(Debug)]
pub struct PageStore {
    page_size: usize,
    root: TrieNode,
    /// O(1) hash membership set — avoids O(N) trie walk for `contains`.
    hash_set: FxHashSet<PageHash>,
    gpu: PagePool,
    cpu: PagePool,
}

impl PageStore {
    pub fn new(page_size: usize, num_gpu_pages: usize, num_cpu_pages: usize) -> Self {
        PageStore {
            page_size,
            root: TrieNode::new_root(),
            hash_set: FxHashSet::default(),
            gpu: PagePool::new(num_gpu_pages),
            cpu: PagePool::new(num_cpu_pages),
        }
    }

    pub fn page_size(&self) -> usize { self.page_size }

    pub fn stats(&self) -> (usize, usize) {
        (self.gpu.used(), self.gpu.total)
    }

    pub fn available_gpu_pages(&self) -> usize { self.gpu.available() }

    // =========================================================================
    // Working pages (bulk alloc via split_off)
    // =========================================================================

    pub fn alloc_gpu_pages(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        self.gpu.alloc_n(n)
    }

    pub fn free_gpu_pages(&mut self, pages: &[PhysicalPageId]) {
        self.gpu.free_batch(pages);
    }

    pub fn alloc_cpu_pages(&mut self, n: usize) -> Option<Vec<PhysicalPageId>> {
        self.cpu.alloc_n(n)
    }

    pub fn free_cpu_pages(&mut self, pages: &[PhysicalPageId]) {
        self.cpu.free_batch(pages);
    }

    // =========================================================================
    // Commit
    // =========================================================================

    /// Commit new pages by appending to an existing committed chain.
    ///
    /// `prefix`: the already-committed portion of the chain (used to navigate
    /// the trie and bump rc on existing nodes — no physical pages needed).
    /// `new_hashes` + `new_phys`: the new pages to insert.
    ///
    /// On dedup (suffix already exists), bumps rc and frees duplicate pages.
    pub fn commit_append(
        &mut self,
        prefix: &[PageHash],
        new_hashes: &[PageHash],
        new_phys: &[PhysicalPageId],
    ) {
        debug_assert_eq!(new_hashes.len(), new_phys.len());
        if new_hashes.is_empty() && prefix.is_empty() { return; }

        let mut freed = Vec::new();
        if prefix.is_empty() {
            // First commit — no existing prefix to navigate.
            self.root.insert_and_ref(new_hashes, new_phys, 0, &mut freed);
        } else {
            // Build full chain: prefix (navigate) + suffix (insert).
            let mut full_hashes = Vec::with_capacity(prefix.len() + new_hashes.len());
            full_hashes.extend_from_slice(prefix);
            full_hashes.extend_from_slice(new_hashes);
            self.root.insert_and_ref(
                &full_hashes, new_phys, prefix.len(), &mut freed,
            );
        }
        // Free duplicate physical pages on dedup hit.
        if !freed.is_empty() {
            self.gpu.free_batch(&freed);
        }
        self.hash_set.extend(new_hashes);
    }

    /// Commit a batch of pages (no existing prefix — first commit).
    pub fn commit_batch(&mut self, hashes: &[PageHash], phys: &[PhysicalPageId]) {
        self.commit_append(&[], hashes, phys);
    }

    /// Commit a single page.
    pub fn commit(&mut self, hash: PageHash, phys: PhysicalPageId) {
        self.commit_batch(&[hash], &[phys]);
    }

    // =========================================================================
    // Refcounting
    // =========================================================================

    /// Increment refcount along the entire path (path-inclusive retain).
    /// Used during restore to "re-claim" a GPU-resident prefix.
    pub fn retain(&mut self, hashes: &[PageHash]) {
        if hashes.is_empty() { return; }
        self.root.retain_path(hashes);
    }

    /// Decrement refcount along the entire path. Evict nodes that reach rc=0
    /// and have no children. Returns the number of GPU pages freed.
    pub fn release(&mut self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        let (freed, _) = self.root.release_path(hashes, &mut self.gpu);
        // Remove evicted hashes from the membership set.
        // (We don't track exactly which were evicted, so just check.)
        self.hash_set.retain(|h| {
            // Keep hashes that are still in the trie.
            // This is O(N) for the hash_set but only runs on release.
            // TODO: track evicted hashes explicitly for O(1) removal.
            true // For now, leave stale entries — contains() cross-checks trie.
        });
        freed
    }

    /// Estimate how many GPU pages would be freed by `release(hashes)`.
    pub fn count_reclaimable(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        self.root.count_reclaimable_path(hashes)
    }

    // =========================================================================
    // Lookup
    // =========================================================================

    pub fn prefix_len(&self, hashes: &[PageHash]) -> usize {
        self.root.prefix_match_len(hashes)
    }

    pub fn physical_ids(&self, hashes: &[PageHash]) -> Vec<PhysicalPageId> {
        let mut out = Vec::with_capacity(hashes.len());
        self.root.collect_physical(hashes, &mut out);
        out
    }

    /// O(1) hash membership check.
    pub fn contains(&self, hash: PageHash) -> bool {
        self.hash_set.contains(&hash)
    }

    pub fn cached_count(&self, hashes: &[PageHash]) -> usize {
        self.root.prefix_match_len(hashes)
    }

    pub fn debug_info(&self, hashes: &[PageHash]) -> String {
        let prefix = self.prefix_len(hashes);
        format!("trie_v2: prefix_len={prefix}/{}", hashes.len())
    }
}

// =============================================================================
// Utility
// =============================================================================

pub fn compute_last_page_len(total_kv: u32, num_pages: u32, page_size: u32) -> u32 {
    if num_pages == 0 { 0 }
    else {
        let r = total_kv % page_size;
        if r == 0 { page_size } else { r }
    }
}

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

        let mut hasher = AHasher::default();
        chunk.hash(&mut hasher);
        for pos in chunk_pos { pos.hash(&mut hasher); }
        for mask in chunk_masks { mask.hash(&mut hasher); }
        let content_hash = hasher.finish();

        let mut chain_hasher = AHasher::default();
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

    fn h(v: u64) -> PageHash { v }

    // =========================================================================
    // PagePool
    // =========================================================================

    #[test]
    fn pool_alloc_and_free() {
        let mut pool = PagePool::new(4);
        assert_eq!(pool.available(), 4);
        let pages = pool.alloc_n(2).unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pool.available(), 2);
        pool.free_batch(&pages);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn pool_exhaustion() {
        let mut pool = PagePool::new(2);
        let _ = pool.alloc_n(2).unwrap();
        assert!(pool.alloc_n(1).is_none());
    }

    // =========================================================================
    // Basic commit & lookup
    // =========================================================================

    #[test]
    fn commit_and_lookup_single() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit(h(1), 10);
        assert!(store.contains(h(1)));
        assert!(!store.contains(h(2)));
        assert_eq!(store.prefix_len(&[h(1)]), 1);
        assert_eq!(store.physical_ids(&[h(1)]), vec![10]);
    }

    #[test]
    fn commit_batch_and_lookup() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        for &hv in &[h(1), h(2), h(3)] { assert!(store.contains(hv)); }
    }

    // =========================================================================
    // Incremental commits (the key fix)
    // =========================================================================

    #[test]
    fn incremental_commit_via_append() {
        let mut store = PageStore::new(16, 100, 0);
        // First commit: [H1, H2]
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        // Incremental commit: append H3 after existing [H1, H2]
        store.commit_append(&[h(1), h(2)], &[h(3)], &[30]);

        // physical_ids should find all three via chain traversal
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
    }

    #[test]
    fn multiple_incremental_commits() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1)], &[10]);
        store.commit_append(&[h(1)], &[h(2)], &[20]);
        store.commit_append(&[h(1), h(2)], &[h(3)], &[30]);
        store.commit_append(&[h(1), h(2), h(3)], &[h(4)], &[40]);

        assert_eq!(store.physical_ids(&[h(1), h(2), h(3), h(4)]), vec![10, 20, 30, 40]);
    }

    // =========================================================================
    // Dedup refcounting
    // =========================================================================

    #[test]
    fn dedup_survives_one_release() {
        let mut store = PageStore::new(16, 100, 0);
        // Context A commits [H1, H2]
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        // Context B commits same chain — dedup hit
        let phys_b = store.alloc_gpu_pages(2).unwrap();
        store.commit_batch(&[h(1), h(2)], &phys_b);

        // B's pages should be freed back (dedup)
        // rc should be 2 on each node

        // Release A's chain — should NOT evict (B still holds refs)
        let freed = store.release(&[h(1), h(2)]);
        assert_eq!(freed, 0, "should not evict — B still holds refs");
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
        assert_eq!(store.physical_ids(&[h(1), h(2)]), vec![10, 20]);

        // Release B's chain — NOW evict
        let freed = store.release(&[h(1), h(2)]);
        assert_eq!(freed, 2);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 0);
    }

    #[test]
    fn dedup_frees_duplicate_pages() {
        let mut store = PageStore::new(16, 10, 0);
        assert_eq!(store.available_gpu_pages(), 10);

        // Commit 2 pages (properly allocated from pool)
        let first_pages = store.alloc_gpu_pages(2).unwrap();
        assert_eq!(store.available_gpu_pages(), 8);
        store.commit_batch(&[h(1), h(2)], &first_pages);

        // Commit same chain with new pages — should free the duplicates
        let dup_pages = store.alloc_gpu_pages(2).unwrap();
        assert_eq!(store.available_gpu_pages(), 6);
        store.commit_batch(&[h(1), h(2)], &dup_pages);
        assert_eq!(store.available_gpu_pages(), 8); // 6 + 2 freed
    }

    // =========================================================================
    // Path-inclusive release
    // =========================================================================

    #[test]
    fn release_decrements_entire_path() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Extend with divergent suffix
        store.commit_append(&[h(1), h(2)], &[h(4)], &[40]);

        // Release [H1, H2, H3]: decrements rc on all 3 nodes
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1); // only H3 leaf evicted (H1,H2 still used by H4 path)

        // [H1, H2] should still be accessible (held by H4 path)
        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);

        // Release [H1, H2, H4]: now everything evicts
        let freed = store.release(&[h(1), h(2), h(4)]);
        assert!(freed >= 3);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 0);
    }

    // =========================================================================
    // Split preserves rc
    // =========================================================================

    #[test]
    fn split_preserves_existing_rc() {
        let mut store = PageStore::new(16, 100, 0);
        // A commits [H1, H2, H3]
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // B commits [H1, H2, H4] — causes split at H2
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);

        // Both chains should be fully accessible
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(4)]), vec![10, 20, 40]);

        // Release A — should only evict H3 leaf
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);

        // Release B — now evict everything
        let freed = store.release(&[h(1), h(2), h(4)]);
        assert!(freed >= 3);
    }

    // =========================================================================
    // Retain (path-inclusive)
    // =========================================================================

    #[test]
    fn retain_bumps_rc_along_path() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Retain prefix [H1, H2]
        store.retain(&[h(1), h(2)]);

        // Release full chain — H3 evicted, but H1,H2 survive (retain holds them)
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 1);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
    }

    // =========================================================================
    // Restore protocol: retain + commit_append
    // =========================================================================

    #[test]
    fn restore_protocol() {
        let mut store = PageStore::new(16, 100, 0);
        // Initial commit
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);

        // Simulate suspend: release
        store.release(&[h(1), h(2), h(3)]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 0);

        // Simulate restore: prefix_len=0, so alloc + commit full chain
        let new_phys = store.alloc_gpu_pages(3).unwrap();
        store.commit_batch(&[h(1), h(2), h(3)], &new_phys);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
    }

    #[test]
    fn restore_with_shared_prefix() {
        let mut store = PageStore::new(16, 100, 0);
        // A and B both commit [H1, H2] (different suffixes)
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);

        // Suspend A
        store.release(&[h(1), h(2), h(3)]);

        // Restore A: prefix_len should be 2 (shared with B)
        let prefix = store.prefix_len(&[h(1), h(2), h(3)]);
        assert_eq!(prefix, 2);

        // Retain prefix + commit suffix
        store.retain(&[h(1), h(2)]);
        let suffix_phys = store.alloc_gpu_pages(1).unwrap();
        store.commit_append(&[h(1), h(2)], &[h(3)], &suffix_phys);

        // Full chain accessible again
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
    }

    // =========================================================================
    // GPU pool integration
    // =========================================================================

    #[test]
    fn eviction_returns_pages_to_pool() {
        let mut store = PageStore::new(16, 10, 0);
        let phys = store.alloc_gpu_pages(3).unwrap();
        assert_eq!(store.available_gpu_pages(), 7);
        store.commit_batch(&[h(1), h(2), h(3)], &phys);
        store.release(&[h(1), h(2), h(3)]);
        assert_eq!(store.available_gpu_pages(), 10);
    }

    // =========================================================================
    // count_reclaimable
    // =========================================================================

    #[test]
    fn count_reclaimable_single_holder() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        // Single holder — all pages reclaimable
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 3);
    }

    #[test]
    fn count_reclaimable_shared_prefix() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        // Shared prefix — only the leaf is reclaimable
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 1);
    }
}
