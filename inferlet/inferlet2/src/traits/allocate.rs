use crate::Queue;
use crate::pool::Allocator;
use crate::{allocate, core, pool};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Provides an interface for managing memory resources, such as KV cache pages and
/// embedding tables, through a command queue.
///
/// This trait abstracts the underlying allocation, deallocation, and data transfer
/// operations, often leveraging a resource pooling mechanism to efficiently manage
/// hardware resources.
pub trait Allocate {
    /// Gets the size of a single KV page in bytes.
    fn get_kv_page_size(&self) -> u32;

    /// Retrieves a list of all currently exported KV page sets.
    ///
    /// # Returns
    /// A `Vec` of tuples, where each tuple contains the `name` of an exported set
    /// and the `number of pages` in that set.
    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)>;

    /// Allocates a specified number of KV pages.
    ///
    /// # Parameters
    /// * `num_pages`: The number of KV pages to allocate.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the unique IDs of the newly allocated pages.
    fn allocate_kv_pages(&self, num_pages: usize) -> Vec<u32>;

    /// Deallocates a set of KV pages, returning them to the resource pool.
    ///
    /// # Parameters
    /// * `page_ids`: A slice of page IDs to deallocate.
    fn deallocate_kv_pages(&self, page_ids: &[u32]);

    fn increase_ref_count(&self, page_ids: &[u32]);

    /// Allocates a specified number of embedding slots.
    ///
    /// # Parameters
    /// * `num_embeds`: The number of embedding slots to allocate.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the unique IDs of the newly allocated embeddings.
    fn allocate_embeds(&self, num_embeds: usize) -> Vec<u32>;

    /// Deallocates a set of embedding slots, returning them to the resource pool.
    ///
    /// # Parameters
    /// * `embed_ids`: A slice of embedding IDs to deallocate.
    fn deallocate_embeds(&self, embed_ids: &[u32]);

    /// Copies a block of data from one KV page to another.
    ///
    /// # Parameters
    /// * `src_page_id`: The ID of the source page.
    /// * `dst_page_id`: The ID of the destination page.
    /// * `src_offset`: The starting offset in the source page, in bytes.
    /// * `dst_offset`: The starting offset in the destination page, in bytes.
    /// * `size`: The number of bytes to copy.
    fn copy_kv_page(
        &self,
        src_page_id: u32,
        dst_page_id: u32,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    );

    /// Exports a set of KV pages under a given name for later retrieval.
    ///
    /// This makes the state of the given pages available for subsequent import operations.
    ///
    /// # Parameters
    /// * `src_page_ids`: The IDs of the pages to export.
    /// * `name`: A `String` name to associate with the exported pages.
    fn export_kv_pages(&self, src_page_ids: &[u32], name: String);

    /// Imports a previously exported set of KV pages by name.
    ///
    /// This operation first allocates a new set of KV pages and then copies the
    /// content from the named exported set into the new pages.
    ///
    /// # Parameters
    /// * `name`: The name of the exported page set to import.
    ///
    /// # Returns
    /// A `Vec<u32>` containing the IDs of the newly allocated pages that now hold
    /// the imported data. Returns an empty `Vec` if the name is not found.
    fn import_kv_pages(&self, name: String) -> Vec<u32>;
}

#[derive(Debug)]
struct KvPageAllocator {}

impl Allocator for KvPageAllocator {
    fn allocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        allocate::allocate_kv_pages(queue, ids);
        Ok(())
    }
    fn deallocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        allocate::deallocate_kv_pages(queue, ids);
        Ok(())
    }
}

#[derive(Debug)]
struct EmbedAllocator {}

impl Allocator for EmbedAllocator {
    fn allocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        allocate::allocate_embeds(queue, ids);
        Ok(())
    }
    fn deallocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        allocate::deallocate_embeds(queue, ids);
        Ok(())
    }
}

// NOTE: The ResourcePool and RcResourcePool implementations are assumed to exist
// as in the reference code, managing the acquisition and release of resource IDs.
// For brevity, their definitions are omitted here, but the functions below use them.

thread_local! {
    static KV_PAGE_POOL: RefCell<HashMap<u32, Rc<RefCell<pool::RcResourcePool<KvPageAllocator>>>>> = RefCell::new(HashMap::new());
    static EMBED_POOL: RefCell<HashMap<u32, Rc<RefCell<pool::ResourcePool<EmbedAllocator>>>>> = RefCell::new(HashMap::new());
}

fn get_kv_page_pool(queue: &core::Queue) -> Rc<RefCell<pool::RcResourcePool<KvPageAllocator>>> {
    KV_PAGE_POOL.with(|pools| {
        let mut pools_map = pools.borrow_mut();
        pools_map
            .entry(queue.get_service_id())
            .or_insert_with(|| {
                Rc::new(RefCell::new(pool::RcResourcePool::new(
                    KvPageAllocator {},
                    u32::MAX,
                    true,
                    20,
                )))
            })
            .clone()
    })
}

fn get_embed_pool(queue: &core::Queue) -> Rc<RefCell<pool::ResourcePool<EmbedAllocator>>> {
    EMBED_POOL.with(|pools| {
        let mut pools_map = pools.borrow_mut();
        pools_map
            .entry(queue.get_service_id())
            .or_insert_with(|| {
                Rc::new(RefCell::new(pool::ResourcePool::new(
                    EmbedAllocator {},
                    u32::MAX,
                    true,
                    20,
                )))
            })
            .clone()
    })
}

// --- Completed Queue Implementations ---

impl Allocate for Queue {
    fn get_kv_page_size(&self) -> u32 {
        allocate::get_kv_page_size(&self.inner)
    }

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        allocate::get_all_exported_kv_pages(&self.inner)
    }

    fn allocate_kv_pages(&self, num_pages: usize) -> Vec<u32> {
        let pool = get_kv_page_pool(&self.inner);
        pool.borrow_mut()
            .acquire_many(&self.inner, num_pages)
            .expect("Failed to allocate KV pages from pool")
    }

    fn deallocate_kv_pages(&self, page_ids: &[u32]) {
        let pool = get_kv_page_pool(&self.inner);
        pool.borrow_mut()
            .release_many(&self.inner, page_ids)
            .expect("Failed to deallocate KV pages from pool");
    }

    fn increase_ref_count(&self, page_ids: &[u32]) {
        let pool = get_kv_page_pool(&self.inner);
        pool.borrow_mut().increment_rc_many(page_ids);
    }

    fn allocate_embeds(&self, num_embeds: usize) -> Vec<u32> {
        let pool = get_embed_pool(&self.inner);
        pool.borrow_mut()
            .acquire_many(&self.inner, num_embeds)
            .expect("Failed to allocate embeds from pool")
    }

    fn deallocate_embeds(&self, embed_ids: &[u32]) {
        let pool = get_embed_pool(&self.inner);
        pool.borrow_mut()
            .release_many(&self.inner, embed_ids)
            .expect("Failed to deallocate embeds from pool");
    }

    fn copy_kv_page(
        &self,
        src_page_id: u32,
        dst_page_id: u32,
        src_offset: u32,
        dst_offset: u32,
        size: u32,
    ) {
        allocate::copy_kv_page(
            &self.inner,
            src_page_id,
            dst_page_id,
            src_offset,
            dst_offset,
            size,
        )
    }

    fn export_kv_pages(&self, src_page_ids: &[u32], name: String) {
        allocate::export_kv_pages(&self.inner, src_page_ids, &name)
    }

    fn import_kv_pages(&self, name: String) -> Vec<u32> {
        // Fetch all exported KV pages
        let available_kv_pages: Vec<(String, u32)> = self.get_all_exported_kv_pages();

        // Find the page size for the matching name
        let page_size = available_kv_pages
            .iter()
            .find(|(n, _)| n == &name)
            .map(|(_, size)| *size);

        if page_size.is_none() {
            return vec![];
        }

        let allocated_ids = self.allocate_kv_pages(page_size.unwrap() as usize);

        // Use the page size if necessary (it's unclear from your snippet if it's used)
        // But if it's only `allocate::import_kv_pages`, call it and return the result:
        allocate::import_kv_pages(&self.inner, &allocated_ids, &name);

        allocated_ids
    }
}
