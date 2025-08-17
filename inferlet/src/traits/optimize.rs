use crate::Queue;
use crate::pool::Allocator;
use crate::traits::output_text::Distribution;
use crate::{core, optimize, pool};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use wstd::io::AsyncPollable;

pub trait Optimize {
    fn allocate_adapters(&self, num_adapters: usize) -> Vec<u32>;

    fn deallocate_adapters(&self, adapter_ids: &[u32]);

    fn export_adapter(&self, adapter_id: u32, name: &str);

    fn unexport_adapter(&self, name: &str);

    fn import_adapter(&self, name: &str) -> u32;

    fn initialize_adapter(
        &self,
        adapter_id: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    );

    fn mutate_adapters(&self, adapter_ids: &[u32], parent_id: u32, seeds: &[i64]);

    fn update_adapter(&self, adapter_id: u32, scores: &[f32], seeds: &[i64], max_sigma: f32);

    async fn forward_with_adapter(
        &self,
        adapter_id: u32,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Option<Vec<Distribution>>;
}

#[derive(Debug)]
struct AdapterAllocator {}

impl Allocator for AdapterAllocator {
    fn allocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        optimize::allocate_adapters(queue, ids);
        Ok(())
    }
    fn deallocate(&self, queue: &core::Queue, ids: &[u32]) -> Result<(), pool::ResourcePoolError> {
        optimize::deallocate_adapters(queue, ids);
        Ok(())
    }

    fn import(
        &self,
        queue: &core::Queue,
        ids: &[u32],
        name: &str,
    ) -> Result<(), pool::ResourcePoolError> {
        optimize::import_adapter(queue, ids[0], name);
        Ok(())
    }
}

thread_local! {
    static ADAPTER_POOL: RefCell<HashMap<u32, Rc<RefCell<pool::ResourcePool<AdapterAllocator>>>>> = RefCell::new(HashMap::new());
}

fn get_adapter_pool(queue: &core::Queue) -> Rc<RefCell<pool::ResourcePool<AdapterAllocator>>> {
    ADAPTER_POOL.with(|pools| {
        let mut pools_map = pools.borrow_mut();
        pools_map
            .entry(queue.get_service_id())
            .or_insert_with(|| {
                Rc::new(RefCell::new(pool::ResourcePool::new(
                    AdapterAllocator {},
                    u32::MAX,
                    true,
                    20,
                )))
            })
            .clone()
    })
}

impl Optimize for Queue {
    fn allocate_adapters(&self, num_adapters: usize) -> Vec<u32> {
        let pool = get_adapter_pool(&self.inner);
        pool.borrow_mut()
            .acquire_many(&self.inner, num_adapters)
            .expect("Failed to allocate adapters from pool")
    }

    fn deallocate_adapters(&self, adapter_ids: &[u32]) {
        let pool = get_adapter_pool(&self.inner);
        pool.borrow_mut()
            .release_many(&self.inner, adapter_ids)
            .expect("Failed to deallocate adapters from pool");
    }
    fn export_adapter(&self, adapter_id: u32, name: &str) {
        optimize::export_adapter(&self.inner, adapter_id, name);
    }

    fn unexport_adapter(&self, name: &str) {
        optimize::unexport_adapter(&self.inner, name);
    }

    fn import_adapter(&self, name: &str) -> u32 {
        let pool = get_adapter_pool(&self.inner);
        pool.borrow_mut()
            .import(&self.inner, 1, &name)
            .expect("Failed to allocate adapter from pool")
            .into_iter()
            .next()
            .unwrap()
    }

    fn initialize_adapter(
        &self,
        adapter_id: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        optimize::initialize_adapter(
            &self.inner,
            adapter_id,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        );
    }

    fn mutate_adapters(&self, adapter_ids: &[u32], parent_id: u32, seeds: &[i64]) {
        optimize::mutate_adapters(&self.inner, adapter_ids, parent_id, seeds);
    }

    fn update_adapter(&self, adapter_id: u32, scores: &[f32], seeds: &[i64], max_sigma: f32) {
        optimize::update_adapter(&self.inner, adapter_id, scores, seeds, max_sigma);
    }

    async fn forward_with_adapter(
        &self,
        adapter_id: u32,
        last_kv_page_len: u32,
        kv_page_ids: &[u32],
        tokens: &[u32],
        positions: &[u32],
        mask: &[Vec<u32>],
        output_indices: &[u32],
    ) -> Option<Vec<Distribution>> {
        let result_future = optimize::forward_with_adapter(
            &self.inner,
            adapter_id,
            last_kv_page_len,
            kv_page_ids,
            tokens,
            positions,
            mask,
            output_indices,
        );

        if let Some(result_future) = result_future {
            // Get the pollable handle associated with the future.
            let pollable = result_future.pollable();

            // Asynchronously wait for the result to become available.
            AsyncPollable::new(pollable).wait_for().await;

            // Once ready, get the result from the future. The result of a WIT future
            // is typically wrapped in Option<T>.
            let distributions: Vec<(Vec<u32>, Vec<f32>)> = result_future
                .get()
                .expect("WASI pollable did not yield a result.");

            // Map the raw result into the strongly-typed `Distribution` struct.
            Some(
                distributions
                    .into_iter()
                    .map(|(ids, probs)| Distribution { ids, probs })
                    .collect(),
            )
        } else {
            None
        }
    }
}
