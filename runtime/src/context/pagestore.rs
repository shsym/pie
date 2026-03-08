//! PageStore — Per-Device CAS Cache + Physical Page Pools.
//!
//! Content-addressed storage for KV cache pages using a compressed Radix
//! Trie (Patricia Trie) with chained Merkle hashes. Pages are identified
//! by `PageHash = ahash(content, prev_hash)`. Sharing across contexts is
//! structural: two contexts with the same token prefix share physical pages
//! via tip-only refcounting.
//!
//! Each model device gets its own `PageStore` — no cross-device coordination.
//! Owned exclusively by the `ContextManager` actor (no interior mutability).
//!
//! Key properties:
//! - `retain` / `release` are tip-only: O(depth) traversal + O(1) refcount op
//! - `commit_batch` inserts entire hash chains, minimizing trie splits
//! - Structural protection: ancestors can't be evicted while they have children
//! - Merge-on-eviction: single-child internal nodes compress automatically
//! - O(1) `contains` via auxiliary `FxHashSet`

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::inference::brle::Brle;

// =============================================================================
// Types (same as pagestore.rs)
// =============================================================================

/// Content hash of a KV cache page, chained to its predecessor.
pub type PageHash = u64;

/// Physical page index in GPU or CPU memory.
pub type PhysicalPageId = u32;

// =============================================================================
// PagePool (optimized with bulk alloc)
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
// TrieNode
// =============================================================================

/// A node in the Radix Trie representing an unbroken chunk of cached pages.
///
/// Invariants:
/// - `prefix.len() == prefix_hashes.len()`
/// - `refcount` counts contexts whose tip is *exactly* this node's last hash
/// - Internal nodes (with children) typically have `refcount == 0`
/// - A node is evictable iff `refcount == 0 && children.is_empty()`
#[derive(Debug)]
struct TrieNode {
    /// Physical page IDs for this contiguous chunk.
    prefix: Vec<PhysicalPageId>,
    /// Corresponding Merkle PageHashes for each page.
    prefix_hashes: Vec<PageHash>,
    /// Leaf-only refcount: number of contexts whose tip is this node's last hash.
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

    /// Create a leaf node with the given pages.
    fn new_leaf(hashes: Vec<PageHash>, phys: Vec<PhysicalPageId>) -> Self {
        debug_assert_eq!(hashes.len(), phys.len());
        TrieNode {
            prefix: phys,
            prefix_hashes: hashes,
            refcount: 0,
            children: FxHashMap::default(),
        }
    }

    // =========================================================================
    // Insert (optimized: single lookup + split_off)
    // =========================================================================

    /// Insert a sequence of pages into this subtree.
    fn insert(&mut self, hashes: &[PageHash], phys: &[PhysicalPageId]) {
        debug_assert_eq!(hashes.len(), phys.len());
        if hashes.is_empty() { return; }

        let first = hashes[0];

        // Single lookup instead of contains_key + get_mut.
        if let Some(child) = self.children.get_mut(&first) {
            let common = common_prefix_len(&child.prefix_hashes, hashes);

            if common < child.prefix_hashes.len() {
                // Partial match — SPLIT using split_off (avoids .to_vec() allocation).
                let old_suffix_hashes = child.prefix_hashes.split_off(common);
                let old_suffix_phys = child.prefix.split_off(common);
                let old_suffix_first = old_suffix_hashes[0];

                let old_suffix = TrieNode {
                    prefix_hashes: old_suffix_hashes,
                    prefix: old_suffix_phys,
                    refcount: child.refcount,
                    children: std::mem::take(&mut child.children),
                };

                // child is already truncated by split_off.
                child.refcount = 0; // internal node
                child.children.insert(old_suffix_first, old_suffix);

                // Insert new divergent suffix (if any remaining hashes).
                if common < hashes.len() {
                    let new_leaf = TrieNode::new_leaf(
                        hashes[common..].to_vec(),
                        phys[common..].to_vec(),
                    );
                    child.children.insert(hashes[common], new_leaf);
                }
            } else if common < hashes.len() {
                // Full match of child's prefix — recurse with remaining.
                child.insert(&hashes[common..], &phys[common..]);
            }
            // else: exact match — sequence already present, nothing to do.
        } else {
            // No matching child — create new leaf.
            self.children.insert(first, TrieNode::new_leaf(
                hashes.to_vec(), phys.to_vec(),
            ));
        }
    }

    // =========================================================================
    // Lookup
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
    // Refcounting
    // =========================================================================

    /// Traverse to the node matching `hashes` and increment its refcount.
    /// If `hashes` ends mid-node, splits the node first (split-on-retain).
    fn retain_at_tip(&mut self, hashes: &[PageHash]) -> bool {
        if hashes.is_empty() { return false; }

        match self.children.get_mut(&hashes[0]) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);
                if common == hashes.len() && common < child.prefix_hashes.len() {
                    // Split-on-retain: hashes consumed mid-node.
                    let suffix_hashes = child.prefix_hashes.split_off(common);
                    let suffix_phys = child.prefix.split_off(common);
                    let suffix_first = suffix_hashes[0];
                    let suffix_node = TrieNode {
                        prefix_hashes: suffix_hashes,
                        prefix: suffix_phys,
                        refcount: child.refcount,
                        children: std::mem::take(&mut child.children),
                    };
                    child.refcount = 1; // the retain
                    child.children.insert(suffix_first, suffix_node);
                    true
                } else if common < child.prefix_hashes.len() {
                    false // partial match, hashes diverge (impossible for valid chains)
                } else if common < hashes.len() {
                    child.retain_at_tip(&hashes[common..])
                } else {
                    child.refcount += 1;
                    true
                }
            }
            None => false,
        }
    }

    /// Release at the tip, evict + cascade + merge.
    /// Returns `(freed_page_count, should_remove_self, evicted_hashes)`.
    fn release_at_tip(
        &mut self,
        hashes: &[PageHash],
        pool: &mut PagePool,
        evicted: &mut Vec<PageHash>,
    ) -> (usize, bool) {
        if hashes.is_empty() { return (0, false); }

        let first = hashes[0];
        let (freed, remove_child) = match self.children.get_mut(&first) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);
                if common == hashes.len() && common < child.prefix_hashes.len() {
                    // Split-on-release: hashes consumed mid-node.
                    let suffix_hashes = child.prefix_hashes.split_off(common);
                    let suffix_phys = child.prefix.split_off(common);
                    let suffix_first = suffix_hashes[0];
                    let suffix_node = TrieNode {
                        prefix_hashes: suffix_hashes,
                        prefix: suffix_phys,
                        refcount: child.refcount,
                        children: std::mem::take(&mut child.children),
                    };
                    child.refcount = child.refcount.saturating_sub(1);
                    child.children.insert(suffix_first, suffix_node);
                    if child.refcount == 0 && child.children.is_empty() {
                        let count = child.prefix.len();
                        pool.free_batch(&child.prefix);
                        evicted.extend_from_slice(&child.prefix_hashes);
                        (count, true)
                    } else {
                        (0, false)
                    }
                } else if common < child.prefix_hashes.len() {
                    (0, false)
                } else if common < hashes.len() {
                    child.release_at_tip(&hashes[common..], pool, evicted)
                } else {
                    // Tip node found — decrement.
                    child.refcount = child.refcount.saturating_sub(1);
                    if child.refcount == 0 && child.children.is_empty() {
                        let count = child.prefix.len();
                        pool.free_batch(&child.prefix);
                        evicted.extend_from_slice(&child.prefix_hashes);
                        (count, true)
                    } else {
                        (0, false)
                    }
                }
            }
            None => (0, false),
        };

        if remove_child {
            self.children.remove(&first);
        }

        // Orphan sweep: after releasing, if the matched child still exists
        // and has rc=0, sweep its evictable (rc=0, childless) children.
        let mut total_freed = freed;
        if !remove_child {
            if let Some(child) = self.children.get_mut(&first) {
                if child.refcount == 0 {
                    total_freed += sweep_orphans(child, pool, evicted);
                }
            }
            // Re-check: if the child is now evictable after sweep, remove it.
            let should_remove = self.children.get(&first)
                .is_some_and(|c| c.refcount == 0 && c.children.is_empty() && !c.prefix_hashes.is_empty());
            if should_remove {
                if let Some(child) = self.children.remove(&first) {
                    pool.free_batch(&child.prefix);
                    evicted.extend_from_slice(&child.prefix_hashes);
                    total_freed += child.prefix.len();
                }
            }
        }

        // Merge: compress single-child internal nodes.
        if self.children.len() == 1 && self.refcount == 0 && !self.prefix_hashes.is_empty() {
            let (_, sole) = self.children.drain().next().unwrap();
            self.prefix.extend(sole.prefix);
            self.prefix_hashes.extend(sole.prefix_hashes);
            self.refcount = sole.refcount;
            self.children = sole.children;
        }

        // Cascade: check if self is now evictable.
        let remove_self = self.refcount == 0
            && self.children.is_empty()
            && !self.prefix_hashes.is_empty();

        if remove_self {
            pool.free_batch(&self.prefix);
            evicted.extend_from_slice(&self.prefix_hashes);
            total_freed += self.prefix.len();
        }

        (total_freed, remove_self)
    }

    /// Estimate pages freed on release. Read-only.
    fn count_reclaimable_at_tip(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }

        match self.children.get(&hashes[0]) {
            Some(child) => {
                let common = common_prefix_len(&child.prefix_hashes, hashes);
                if common < child.prefix_hashes.len() {
                    0
                } else if common < hashes.len() {
                    child.count_reclaimable_at_tip(&hashes[common..])
                } else {
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
}

/// Count how many leading elements of `a` and `b` are equal.
fn common_prefix_len(a: &[PageHash], b: &[PageHash]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Recursively sweep and evict childless children with rc=0 from a node.
/// Returns total pages freed. Bottom-up: recurse first so grandchildren
/// made evictable by deeper sweeps are cleaned up in the same pass.
fn sweep_orphans(node: &mut TrieNode, pool: &mut PagePool, evicted: &mut Vec<PageHash>) -> usize {
    let mut freed = 0;

    // 1. Recurse into rc=0 children first (bottom-up).
    for child in node.children.values_mut() {
        if child.refcount == 0 {
            freed += sweep_orphans(child, pool, evicted);
        }
    }

    // 2. Now remove any children that are (or became) evictable.
    node.children.retain(|_, child| {
        if child.refcount == 0 && child.children.is_empty() {
            freed += child.prefix.len();
            pool.free_batch(&child.prefix);
            evicted.extend_from_slice(&child.prefix_hashes);
            false
        } else {
            true
        }
    });

    freed
}

// =============================================================================
// PageStore
// =============================================================================

/// Per-device page cache backed by a Radix Trie.
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

    pub fn commit_batch(&mut self, hashes: &[PageHash], phys: &[PhysicalPageId]) {
        debug_assert_eq!(hashes.len(), phys.len());
        if hashes.is_empty() { return; }
        self.root.insert(hashes, phys);
        self.hash_set.extend(hashes);
    }

    pub fn commit(&mut self, hash: PageHash, phys: PhysicalPageId) {
        self.commit_batch(&[hash], &[phys]);
    }

    // =========================================================================
    // Refcounting (tip-only)
    // =========================================================================

    pub fn retain(&mut self, hashes: &[PageHash]) {
        if hashes.is_empty() { return; }
        self.root.retain_at_tip(hashes);
    }

    pub fn release(&mut self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        let mut evicted = Vec::new();
        let (freed, _) = self.root.release_at_tip(hashes, &mut self.gpu, &mut evicted);
        // Remove evicted hashes from the membership set.
        for h in &evicted { self.hash_set.remove(h); }
        freed
    }

    pub fn count_reclaimable(&self, hashes: &[PageHash]) -> usize {
        if hashes.is_empty() { return 0; }
        self.root.count_reclaimable_at_tip(hashes)
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

    /// O(1) hash membership check via auxiliary hash set.
    pub fn contains(&self, hash: PageHash) -> bool {
        self.hash_set.contains(&hash)
    }

    pub fn cached_count(&self, hashes: &[PageHash]) -> usize {
        self.root.prefix_match_len(hashes)
    }

    pub fn debug_info(&self, hashes: &[PageHash]) -> String {
        let prefix = self.prefix_len(hashes);
        format!("trie: prefix_len={prefix}/{}", hashes.len())
    }
}

// =============================================================================
// Utility (identical to pagestore.rs)
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
    // PageStore: basic commit & lookup
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
        let hashes = [h(1), h(2), h(3)];
        let phys = [10, 20, 30];

        store.commit_batch(&hashes, &phys);

        assert_eq!(store.prefix_len(&hashes), 3);
        assert_eq!(store.physical_ids(&hashes), vec![10, 20, 30]);
        for &hv in &hashes { assert!(store.contains(hv)); }
        assert!(!store.contains(h(99)));
    }

    #[test]
    fn prefix_len_partial_match() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);

        assert_eq!(store.prefix_len(&[h(1), h(2), h(3), h(4)]), 3);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(99)]), 2);
        assert_eq!(store.prefix_len(&[h(99)]), 0);
    }

    // =========================================================================
    // Trie splitting
    // =========================================================================

    #[test]
    fn commit_causes_split() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);

        assert_eq!(store.prefix_len(&[h(1), h(2), h(3)]), 3);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(4)]), vec![10, 20, 40]);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
        for &hv in &[h(1), h(2), h(3), h(4)] { assert!(store.contains(hv)); }
    }

    #[test]
    fn commit_prefix_of_existing_causes_split() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3), h(4)], &[10, 20, 30, 40]);
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3), h(4)]), 4);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
    }

    #[test]
    fn commit_extension_of_existing() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        store.commit_batch(&[h(1), h(2), h(3), h(4)], &[10, 20, 30, 40]);
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3), h(4)]), 4);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3), h(4)]), vec![10, 20, 30, 40]);
    }

    // =========================================================================
    // Retain & Release
    // =========================================================================

    #[test]
    fn retain_and_release_basic() {
        let mut store = PageStore::new(16, 100, 0);
        let chain = [h(1), h(2), h(3)];
        store.commit_batch(&chain, &[10, 20, 30]);
        store.retain(&chain);

        let freed = store.release(&chain);
        assert_eq!(freed, 3);
        assert_eq!(store.prefix_len(&chain), 0);
        assert!(!store.contains(h(1)));
    }

    #[test]
    fn multiple_retains_require_matching_releases() {
        let mut store = PageStore::new(16, 100, 0);
        let chain = [h(1), h(2)];
        store.commit_batch(&chain, &[10, 20]);
        store.retain(&chain);
        store.retain(&chain);

        assert_eq!(store.release(&chain), 0);
        assert_eq!(store.prefix_len(&chain), 2);
        assert_eq!(store.release(&chain), 2);
        assert_eq!(store.prefix_len(&chain), 0);
    }

    // =========================================================================
    // Structural protection & cascade
    // =========================================================================

    #[test]
    fn structural_protection_prevents_ancestor_eviction() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        store.retain(&[h(1), h(2), h(3)]);
        store.retain(&[h(1), h(2), h(4)]);

        assert_eq!(store.release(&[h(1), h(2), h(3)]), 1);
        assert!(store.contains(h(1)));
        assert!(!store.contains(h(3)));

        assert_eq!(store.release(&[h(1), h(2), h(4)]), 3);
        assert!(!store.contains(h(1)));
    }

    #[test]
    fn cascade_eviction_with_deep_chain() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3), h(4), h(5), h(6)], &[10, 20, 30, 40, 50, 60]);
        store.commit_batch(&[h(1), h(2), h(3), h(4), h(99)], &[10, 20, 30, 40, 99]);
        store.retain(&[h(1), h(2), h(3), h(4), h(5), h(6)]);
        store.retain(&[h(1), h(2), h(3), h(4), h(99)]);

        assert_eq!(store.release(&[h(1), h(2), h(3), h(4), h(5), h(6)]), 2);
        assert!(store.contains(h(4)));
        let freed = store.release(&[h(1), h(2), h(3), h(4), h(99)]);
        assert!(freed >= 5);
        assert!(!store.contains(h(1)));
    }

    #[test]
    fn merge_on_eviction_compresses_trie() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        store.retain(&[h(1), h(2), h(3)]);
        store.retain(&[h(1), h(2), h(4)]);
        store.release(&[h(1), h(2), h(3)]);

        assert_eq!(store.prefix_len(&[h(1), h(2), h(4)]), 3);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(4)]), vec![10, 20, 40]);
    }

    // =========================================================================
    // contains uses hash_set (O(1))
    // =========================================================================

    #[test]
    fn contains_uses_hash_set() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);

        // All hashes in the set.
        assert!(store.contains(h(1)));
        assert!(store.contains(h(2)));
        assert!(store.contains(h(3)));
        assert!(!store.contains(h(99)));

        // After eviction, hashes are removed from the set.
        store.retain(&[h(1), h(2), h(3)]);
        store.release(&[h(1), h(2), h(3)]);
        assert!(!store.contains(h(1)));
        assert!(!store.contains(h(2)));
        assert!(!store.contains(h(3)));
    }

    // =========================================================================
    // count_reclaimable
    // =========================================================================

    #[test]
    fn count_reclaimable_leaf_only() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.retain(&[h(1), h(2), h(3)]);
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 3);
        store.retain(&[h(1), h(2), h(3)]);
        assert_eq!(store.count_reclaimable(&[h(1), h(2), h(3)]), 0);
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
        store.retain(&[h(1), h(2), h(3)]);
        store.release(&[h(1), h(2), h(3)]);
        assert_eq!(store.available_gpu_pages(), 10);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn empty_operations() {
        let mut store = PageStore::new(16, 10, 0);
        store.commit_batch(&[], &[]);
        store.retain(&[]);
        assert_eq!(store.release(&[]), 0);
        assert_eq!(store.prefix_len(&[]), 0);
        assert_eq!(store.physical_ids(&[]), Vec::<PhysicalPageId>::new());
        assert_eq!(store.count_reclaimable(&[]), 0);
    }

    #[test]
    fn release_nonexistent_is_noop() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        store.retain(&[h(1), h(2)]);
        assert_eq!(store.release(&[h(99)]), 0);
        assert_eq!(store.prefix_len(&[h(1), h(2)]), 2);
    }

    // =========================================================================
    // Multi-context
    // =========================================================================

    #[test]
    fn multi_context_shared_system_prompt() {
        let mut store = PageStore::new(16, 100, 0);
        let system = [h(1), h(2), h(3), h(4)];
        let system_phys = [10, 20, 30, 40];

        let ctx_a: Vec<PageHash> = system.iter().copied().chain([h(10), h(11)]).collect();
        let ctx_a_phys: Vec<PhysicalPageId> = system_phys.iter().copied().chain([100, 110]).collect();
        let ctx_b: Vec<PageHash> = system.iter().copied().chain([h(20), h(21)]).collect();
        let ctx_b_phys: Vec<PhysicalPageId> = system_phys.iter().copied().chain([200, 210]).collect();

        store.commit_batch(&ctx_a, &ctx_a_phys);
        store.commit_batch(&ctx_b, &ctx_b_phys);
        store.retain(&ctx_a);
        store.retain(&ctx_b);

        assert_eq!(store.release(&ctx_a), 2);
        assert!(store.contains(h(1)));
        assert_eq!(store.release(&ctx_b), 6); // 2 suffix + 4 merged system
        assert!(!store.contains(h(1)));
    }

    #[test]
    fn three_way_fork() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2), h(3)], &[10, 20, 30]);
        store.commit_batch(&[h(1), h(2), h(4)], &[10, 20, 40]);
        store.commit_batch(&[h(1), h(2), h(5)], &[10, 20, 50]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(3)]), vec![10, 20, 30]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(4)]), vec![10, 20, 40]);
        assert_eq!(store.physical_ids(&[h(1), h(2), h(5)]), vec![10, 20, 50]);
    }

    #[test]
    fn disjoint_sequences() {
        let mut store = PageStore::new(16, 100, 0);
        store.commit_batch(&[h(1), h(2)], &[10, 20]);
        store.commit_batch(&[h(100), h(200)], &[30, 40]);
        store.retain(&[h(1), h(2)]);
        store.retain(&[h(100), h(200)]);
        store.release(&[h(1), h(2)]);
        assert!(store.contains(h(100)));
    }

    // =========================================================================
    // Split-on-retain / Split-on-release (partial prefix)
    // =========================================================================

    #[test]
    fn retain_partial_prefix_causes_split() {
        let mut store = PageStore::new(16, 100, 0);

        // Commit a 6-page chain as a single node.
        store.commit_batch(
            &[h(1), h(2), h(3), h(4), h(5), h(6)],
            &[10, 20, 30, 40, 50, 60],
        );

        // Retain at a partial prefix [h1, h2, h3] — mid-node.
        // This should split the node: [h1,h2,h3] (rc=1) → child [h4,h5,h6] (rc=0).
        store.retain(&[h(1), h(2), h(3)]);

        // The full chain is still queryable.
        assert_eq!(store.prefix_len(&[h(1), h(2), h(3), h(4), h(5), h(6)]), 6);

        // Release the partial prefix — frees 3 pages, suffix cascades.
        let freed = store.release(&[h(1), h(2), h(3)]);
        // Prefix [h1,h2,h3] evicted (3 pages), suffix [h4,h5,h6] cascades (3 pages) = 6 total.
        assert_eq!(freed, 6);

        // Everything gone.
        assert!(!store.contains(h(1)));
        assert!(!store.contains(h(6)));
    }

    #[test]
    fn retain_partial_with_existing_tip() {
        let mut store = PageStore::new(16, 100, 0);

        // Commit a chain and retain at the full tip.
        store.commit_batch(
            &[h(1), h(2), h(3), h(4), h(5), h(6)],
            &[10, 20, 30, 40, 50, 60],
        );
        store.retain(&[h(1), h(2), h(3), h(4), h(5), h(6)]); // rc=1 on full chain

        // Now also retain at partial prefix [h1,h2,h3].
        // Splits the node and gives the prefix rc=1.
        store.retain(&[h(1), h(2), h(3)]);

        // Release the full chain tip — suffix [h4..h6] freed (3 pages),
        // prefix protected by its own retain.
        let freed = store.release(&[h(1), h(2), h(3), h(4), h(5), h(6)]);
        assert_eq!(freed, 3); // only suffix freed
        assert!(store.contains(h(1))); // prefix still alive

        // Release the partial prefix — now prefix freed too.
        let freed = store.release(&[h(1), h(2), h(3)]);
        assert_eq!(freed, 3);
        assert!(!store.contains(h(1)));
    }

    #[test]
    fn release_partial_prefix_without_retain() {
        let mut store = PageStore::new(16, 100, 0);

        // Commit and retain at full tip.
        store.commit_batch(&[h(1), h(2), h(3), h(4)], &[10, 20, 30, 40]);
        store.retain(&[h(1), h(2), h(3), h(4)]);

        // Release at partial prefix [h1, h2] — the prefix has rc=0 after split,
        // but can't evict because suffix child still exists.
        let freed = store.release(&[h(1), h(2)]);
        assert_eq!(freed, 0); // prefix has children, can't evict
        assert!(store.contains(h(1))); // still there

        // Release at full tip — now both evict.
        let freed = store.release(&[h(1), h(2), h(3), h(4)]);
        assert!(freed >= 4); // everything freed
    }
}
