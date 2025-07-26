use crate::drafter::Drafter;
use crate::pool::{Allocator, RcResourcePool, ResourcePool};
use crate::sampler::Sampler;
use crate::stop_condition::StopCondition;
use crate::{
    allocate, core, forward, input_text, output_text_async, sampler, stop_condition, tokenize,
};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::{fmt, mem};

#[derive(Debug)]
struct KvPageAllocator {}

#[derive(Debug)]
struct EmbedAllocator {}

impl Allocator for KvPageAllocator {
    fn allocate(&self, queue: &core::Queue, ids: &[u32]) -> crate::pool::Result<()> {
        allocate::allocate_kv_pages(queue, ids);
        Ok(())
    }

    fn deallocate(&self, queue: &core::Queue, ids: &[u32]) -> crate::pool::Result<()> {
        allocate::deallocate_kv_pages(queue, ids);
        Ok(())
    }
}

impl Allocator for EmbedAllocator {
    fn allocate(&self, queue: &core::Queue, ids: &[u32]) -> crate::pool::Result<()> {
        allocate::allocate_embeds(queue, ids);
        Ok(())
    }
    fn deallocate(&self, queue: &core::Queue, ids: &[u32]) -> crate::pool::Result<()> {
        allocate::deallocate_embeds(queue, ids);
        Ok(())
    }
}

thread_local! {
    static KV_PAGE_POOL: RefCell<HashMap<u32, Rc<RefCell<RcResourcePool<KvPageAllocator>>>>> = RefCell::new(HashMap::new());
    static EMBED_POOL: RefCell<HashMap<u32, Rc<RefCell<ResourcePool<EmbedAllocator>>>>> = RefCell::new(HashMap::new());
}

/// Helper function to get the KV page pool for a given queue.
/// It accesses the global `OnceCell` and initializes the pool map on first use.
fn get_kv_page_pool(queue: &core::Queue) -> Rc<RefCell<RcResourcePool<KvPageAllocator>>> {
    let service_id = queue.get_service_id();

    // Use .with() to access the thread-local static.
    KV_PAGE_POOL.with(|pools| {
        // Borrow the map mutably to insert a new pool if needed for this service_id.
        let mut pools_map = pools.borrow_mut();
        pools_map
            .entry(service_id)
            .or_insert_with(|| {
                let new_pool = RcResourcePool::new(KvPageAllocator {}, u32::MAX, true, 20);
                Rc::new(RefCell::new(new_pool))
            })
            .clone()
    })
}

fn get_embed_pool(queue: &core::Queue) -> Rc<RefCell<ResourcePool<EmbedAllocator>>> {
    let service_id = queue.get_service_id();

    EMBED_POOL.with(|pools| {
        let mut pools_map = pools.borrow_mut();
        pools_map
            .entry(service_id)
            .or_insert_with(|| {
                let new_pool = ResourcePool::new(EmbedAllocator {}, u32::MAX, true, 20);
                Rc::new(RefCell::new(new_pool))
            })
            .clone()
    })
}

pub fn allocate_kv_pages(queue: &core::Queue, count: usize) -> Result<Vec<u32>, ContextError> {
    let pool = get_kv_page_pool(queue);
    pool.borrow_mut()
        .acquire_many(queue, count)
        .map_err(|_| ContextError::OutOfMemory("Failed to allocate KV pages".to_string()))
}

pub fn deallocate_kv_pages(queue: &core::Queue, ids: &[u32]) -> Result<(), ContextError> {
    let pool = get_kv_page_pool(queue);
    pool.borrow_mut()
        .release_many(queue, ids)
        .map_err(|_| ContextError::OutOfMemory("Failed to deallocate KV pages".to_string()))
}

pub fn increment_rc_kv_pages(queue: &core::Queue, ids: &[u32]) {
    let pool = get_kv_page_pool(queue);
    pool.borrow_mut().increment_rc_many(ids);
}

pub fn allocate_embeds(queue: &core::Queue, count: usize) -> Result<Vec<u32>, ContextError> {
    let pool = get_embed_pool(queue);
    pool.borrow_mut()
        .acquire_many(queue, count)
        .map_err(|_| ContextError::OutOfMemory("Failed to allocate embeds".to_string()))
}

pub fn deallocate_embeds(queue: &core::Queue, ids: &[u32]) -> Result<(), ContextError> {
    let pool = get_embed_pool(queue);
    pool.borrow_mut()
        .release_many(queue, ids)
        .map_err(|_| ContextError::OutOfMemory("Failed to deallocate embeds".to_string()))
}

#[derive(Debug)]
pub struct Context {
    queue: core::Queue,
    model: Rc<core::Model>,

    kv_page_ids: Vec<u32>,
    last_kv_page_len: usize,

    token_ids: Vec<u32>,
    pending_token_ids: Vec<u32>,

    kv_page_size: usize,

    tokenizer: Rc<tokenize::Tokenizer>,
}

#[derive(Debug)]
pub enum ContextError {
    IncompatibleModel(String),
    OutOfMemory(String),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContextError::IncompatibleModel(msg) => write!(f, "Incompatible model: {}", msg),
            ContextError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let taken = mem::take(&mut self.kv_page_ids);
        deallocate_kv_pages(&self.queue, &taken).unwrap();
    }
}

impl Context {
    pub fn new(model: core::Model) -> Result<Self, ContextError> {
        Self::check_model_traits(&model, &["input_text", "tokenize", "output_text"])?;

        let queue = model.create_queue();
        let kv_page_size = allocate::get_kv_page_size(&queue) as usize;
        let tokenizer = tokenize::get_tokenizer(&queue);

        Ok(Context {
            queue,
            model: Rc::new(model),
            kv_page_ids: Vec::new(),
            last_kv_page_len: 0,
            token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
            kv_page_size,
            tokenizer: Rc::new(tokenizer),
        })
    }

    fn check_model_traits(
        model: &core::Model,
        required_traits: &[&str],
    ) -> Result<(), ContextError> {
        // Store the returned Vec<String> in a variable to extend its lifetime.
        let available_traits_vec = model.get_traits();

        let available_traits: HashSet<&str> =
            available_traits_vec.iter().map(String::as_str).collect();

        // Find any required traits that are not in the available set.
        let missing: Vec<_> = required_traits
            .iter()
            .filter(|t| !available_traits.contains(*t))
            .cloned()
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(ContextError::IncompatibleModel(format!(
                "Model '{}' is missing required traits: {:?}",
                model.get_name(),
                missing
            )))
        }
    }

    pub fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
    }

    pub fn fork_unsafe(&self) -> Self {
        increment_rc_kv_pages(&self.queue, &self.kv_page_ids);

        Context {
            queue: self.model.create_queue(),
            model: self.model.clone(),
            kv_page_size: self.kv_page_size,
            tokenizer: self.tokenizer.clone(),
            kv_page_ids: self.kv_page_ids.clone(),
            last_kv_page_len: self.kv_page_size,
            token_ids: self.token_ids.clone(),
            pending_token_ids: self.pending_token_ids.clone(),
        }
    }

    pub fn fork(&mut self) -> Self {
        // flush the pending tokens
        if !self.pending_token_ids.is_empty() {
            self.flush();
        }

        let forked = if self.last_kv_page_len == self.kv_page_size {
            // easy case: the last page is full
            Context {
                queue: self.model.create_queue(),
                model: self.model.clone(),
                kv_page_size: self.kv_page_size,
                tokenizer: self.tokenizer.clone(),
                kv_page_ids: self.kv_page_ids.clone(),
                last_kv_page_len: self.kv_page_size,
                token_ids: self.token_ids.clone(),
                pending_token_ids: self.pending_token_ids.clone(),
            }
        } else {
            let kept_kv_page_len = self.kv_page_ids.len() - 1;
            let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

            let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
            let forked_pending_token_ids = [
                &self.token_ids[kept_tokens_len..],
                &self.pending_token_ids[..],
            ]
            .concat();
            let forked_kv_page_ids = self.kv_page_ids[..kept_kv_page_len].to_vec();
            let forked_last_kv_page_len = if !forked_kv_page_ids.is_empty() {
                self.kv_page_size
            } else {
                0
            };

            Context {
                queue: self.model.create_queue(),
                model: self.model.clone(),
                kv_page_size: self.kv_page_size,
                tokenizer: self.tokenizer.clone(),
                kv_page_ids: forked_kv_page_ids,
                last_kv_page_len: forked_last_kv_page_len,
                token_ids: forked_token_ids,
                pending_token_ids: forked_pending_token_ids,
            }
        };

        // increase the refcount
        increment_rc_kv_pages(&forked.queue, &forked.kv_page_ids);

        forked
    }

    pub async fn generate_until(&mut self, stop_str: &str, max_tokens: usize) -> String {
        let mut sampler = sampler::GreedySampler::new();
        let stop_str_token_ids = self.tokenizer.tokenize(stop_str);
        let mut stop_condition = stop_condition::any(
            stop_condition::Until::new(stop_str_token_ids),
            stop_condition::Length::new(max_tokens),
        );
        self.generate(&mut sampler, &mut stop_condition).await
    }

    pub fn fill(&mut self, text: &str) {
        let new_token_ids = self.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids);
    }

    pub fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        self.pending_token_ids.extend(new_token_ids);
    }

    pub fn fill_token(&mut self, new_token_id: u32) {
        self.pending_token_ids.push(new_token_id);
    }

    pub fn flush(&mut self) {
        if self.pending_token_ids.len() < 2 {
            return;
        }

        let pending_token_ids = self
            .pending_token_ids
            .drain(..self.pending_token_ids.len() - 1)
            .collect::<Vec<u32>>();

        let position_ids = (self.token_ids.len() as u32
            ..(self.token_ids.len() + pending_token_ids.len()) as u32)
            .collect::<Vec<u32>>();

        let embed_ids = allocate_embeds(&self.queue, pending_token_ids.len()).unwrap();
        input_text::embed_text(&self.queue, &embed_ids, &pending_token_ids, &position_ids);

        let available_space = if !self.kv_page_ids.is_empty() {
            self.kv_page_size - self.last_kv_page_len
        } else {
            0
        };

        if pending_token_ids.len() > available_space {
            let needed_page_count =
                (pending_token_ids.len() - available_space).div_ceil(self.kv_page_size);
            let new_kv_page_ids = allocate_kv_pages(&self.queue, needed_page_count).unwrap();
            self.kv_page_ids.extend(new_kv_page_ids);

            let remaining_tokens = (pending_token_ids.len() - available_space) % self.kv_page_size;
            self.last_kv_page_len = if remaining_tokens == 0 {
                self.kv_page_size
            } else {
                remaining_tokens
            };
        } else {
            self.last_kv_page_len += pending_token_ids.len();
        }

        forward::forward(
            &self.queue,
            self.last_kv_page_len as u32,
            &self.kv_page_ids,
            &embed_ids,
            &[],
        );

        self.token_ids.extend(pending_token_ids);

        deallocate_embeds(&self.queue, &embed_ids).unwrap();
    }

    pub async fn apply_sink(&mut self, sink_size: usize, window_size: usize) {
        if self.pending_token_ids.len() > 1 {
            self.flush();
        }

        let num_pages_to_retain_begin = sink_size.div_ceil(self.kv_page_size);
        let num_pages_to_retain_end = window_size.div_ceil(self.kv_page_size);

        if self.kv_page_ids.len() > num_pages_to_retain_begin + num_pages_to_retain_end {
            let sink_start = num_pages_to_retain_begin;
            let sink_end = self.kv_page_ids.len() - num_pages_to_retain_end;

            self.kv_page_ids = self.kv_page_ids[..=sink_start]
                .iter()
                .chain(self.kv_page_ids[sink_end..].iter())
                .cloned()
                .collect();
        }
    }

    pub async fn apply_window(&mut self, window_size: usize) {
        if self.pending_token_ids.len() > 1 {
            self.flush();
        }

        let num_pages_to_retain = window_size.div_ceil(self.kv_page_size);

        if self.kv_page_ids.len() > num_pages_to_retain {
            let sink_start = self.kv_page_ids.len() - num_pages_to_retain;
            self.kv_page_ids = self.kv_page_ids[sink_start..].to_vec();
        }
    }

    pub async fn generate<S: Sampler, C: StopCondition>(
        &mut self,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        if self.pending_token_ids.len() > 1 {
            self.flush();
        }

        assert_eq!(self.pending_token_ids.len(), 1, "Must have one seed token");
        assert_ne!(
            self.last_kv_page_len, 0,
            "Context must be filled before generation"
        );

        let mut generated_token_ids = Vec::new();
        let input_embed_id = allocate_embeds(&self.queue, 1).unwrap();
        let output_embed_id = allocate_embeds(&self.queue, 1).unwrap();

        let mut next_token_id = self.pending_token_ids[0];
        let mut next_pos_id = self.token_ids.len() as u32;

        loop {
            input_text::embed_text(
                &self.queue,
                &input_embed_id,
                &[next_token_id],
                &[next_pos_id],
            );

            if self.last_kv_page_len == self.kv_page_size || self.kv_page_ids.is_empty() {
                let new_kv_page_ids = allocate_kv_pages(&self.queue, 1).unwrap();
                self.kv_page_ids.extend(new_kv_page_ids);
                self.last_kv_page_len = 1;
            } else {
                self.last_kv_page_len += 1;
            }

            forward::forward(
                &self.queue,
                self.last_kv_page_len as u32,
                &self.kv_page_ids,
                &input_embed_id,
                &output_embed_id,
            );

            let sampled = output_text_async::get_next_token_distribution(
                &self.queue,
                output_embed_id.clone(),
            )
            .await;

            let (next_token_ids, next_token_logits) = &sampled[0];
            next_token_id = sampler.sample(next_token_ids, next_token_logits);
            next_pos_id += 1;

            generated_token_ids.push(next_token_id);

            if stop_condition.should_stop(&generated_token_ids) {
                break;
            } else {
                self.token_ids.push(next_token_id);
            }
        }

        self.pending_token_ids.clear();
        self.pending_token_ids.push(next_token_id);
        deallocate_embeds(&self.queue, &input_embed_id).unwrap();
        deallocate_embeds(&self.queue, &output_embed_id).unwrap();

        self.tokenizer.detokenize(&generated_token_ids)
    }

    pub async fn next(&mut self) -> (Vec<u32>, Vec<f32>) {
        if self.pending_token_ids.len() > 1 {
            self.flush();
        }

        assert_eq!(self.pending_token_ids.len(), 1, "Must have one seed token");
        assert_ne!(
            self.last_kv_page_len, 0,
            "Context must be filled before generation"
        );

        let input_embed_id = allocate_embeds(&self.queue, 1).unwrap();
        let output_embed_id = allocate_embeds(&self.queue, 1).unwrap();

        let next_token_id = self.pending_token_ids.pop().unwrap();
        let next_pos_id = self.token_ids.len() as u32;

        input_text::embed_text(
            &self.queue,
            &input_embed_id,
            &[next_token_id],
            &[next_pos_id],
        );

        if self.last_kv_page_len == self.kv_page_size || self.kv_page_ids.is_empty() {
            let new_kv_page_ids = allocate_kv_pages(&self.queue, 1).unwrap();
            self.kv_page_ids.extend(new_kv_page_ids);
            self.last_kv_page_len = 1;
        } else {
            self.last_kv_page_len += 1;
        }

        forward::forward(
            &self.queue,
            self.last_kv_page_len as u32,
            &self.kv_page_ids,
            &input_embed_id,
            &output_embed_id,
        );

        let sampled =
            output_text_async::get_next_token_distribution(&self.queue, output_embed_id.clone())
                .await;

        self.token_ids.push(next_token_id);

        deallocate_embeds(&self.queue, &input_embed_id).unwrap();
        deallocate_embeds(&self.queue, &output_embed_id).unwrap();

        sampled.into_iter().next().unwrap()
    }

    pub async fn generate_with_beam<C: StopCondition>(
        &mut self,
        stop_condition: &mut C,
        beam_size: usize,
    ) -> String {
        let mut beams = Vec::new();
        beams.push((self.fork(), vec![], 0.0f32));

        loop {
            if let Some((_, generated_tokens, _)) = beams
                .iter()
                .find(|(_, genx, _)| stop_condition.should_stop(genx))
            {
                return self.tokenizer.detokenize(generated_tokens);
            }

            let mut next_beams = Vec::new();
            for (mut beam, generated, score) in beams.into_iter() {
                let (next_token_ids, next_scores) = beam.next().await;
                for i in 0..beam_size.min(next_token_ids.len()) {
                    let mut next_beam = beam.fork();
                    next_beam.fill_token(next_token_ids[i]);
                    let mut next_generated = generated.clone();
                    next_generated.push(next_token_ids[i]);
                    let next_score = score + next_scores[i];
                    next_beams.push((next_beam, next_generated, next_score));
                }
            }

            next_beams.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            next_beams.truncate(beam_size);
            beams = next_beams;

            if beams.is_empty() {
                return String::new(); // Or handle as an error
            }
        }
    }

    pub async fn generate_with_drafter<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        if self.pending_token_ids.len() > 1 {
            self.flush();
        }

        assert_eq!(self.pending_token_ids.len(), 1, "Must have one seed token");

        let mut generated_token_ids = Vec::new();
        let mut next_token_ids = vec![self.pending_token_ids[0]];

        drafter.update(&self.token_ids);

        loop {
            drafter.update(&next_token_ids);
            let (spec_token_ids, spec_pos_ids) = drafter.draft();
            let combined_token_ids =
                [next_token_ids.as_slice(), spec_token_ids.as_slice()].concat();

            let combined_pos_ids = {
                let offset = self.token_ids.len() as u32;
                let mut res = (offset..offset + next_token_ids.len() as u32).collect::<Vec<u32>>();
                res.extend(spec_pos_ids.iter().map(|&x| x + offset));
                res
            };

            let input_embed_id = allocate_embeds(&self.queue, combined_token_ids.len()).unwrap();
            let output_embed_id = allocate_embeds(&self.queue, combined_token_ids.len()).unwrap();

            input_text::embed_text(
                &self.queue,
                &input_embed_id,
                &combined_token_ids,
                &combined_pos_ids,
            );

            let available_space = self.kv_page_size - self.last_kv_page_len;
            if combined_token_ids.len() > available_space {
                let needed_page_count =
                    (combined_token_ids.len() - available_space).div_ceil(self.kv_page_size);

                let new_kv_page_ids = allocate_kv_pages(&self.queue, needed_page_count).unwrap();
                self.kv_page_ids.extend(new_kv_page_ids);
                self.last_kv_page_len = (self.token_ids.len() + combined_token_ids.len())
                    - (self.kv_page_ids.len() - 1) * self.kv_page_size;
            } else {
                self.last_kv_page_len += combined_token_ids.len();
            }

            forward::forward(
                &self.queue,
                self.last_kv_page_len as u32,
                &self.kv_page_ids,
                &input_embed_id,
                &output_embed_id,
            );

            let sampled = output_text_async::get_next_token_distribution(
                &self.queue,
                output_embed_id[next_token_ids.len() - 1..].to_vec(),
            )
            .await;

            let mut verified_tokens = Vec::new();
            let mut correct_spec_tokens = 0;
            for i in 0..sampled.len() {
                let (predicted_tokens, predicted_logits) = &sampled[i];
                let sampled_token = sampler.sample(predicted_tokens, predicted_logits);

                if i < spec_token_ids.len() && sampled_token == spec_token_ids[i] {
                    verified_tokens.push(sampled_token);
                    correct_spec_tokens += 1;
                } else {
                    verified_tokens.push(sampled_token);
                    break;
                }
            }

            let redundant_token_count = spec_token_ids.len() - correct_spec_tokens;
            if self.last_kv_page_len > redundant_token_count {
                self.last_kv_page_len -= redundant_token_count;
            } else {
                let mut remaining_redundancy = redundant_token_count - self.last_kv_page_len;
                let redundant = self.kv_page_ids.pop().unwrap();
                deallocate_kv_pages(&self.queue, &[redundant]).unwrap();

                while remaining_redundancy > self.kv_page_size {
                    let redundant = self.kv_page_ids.pop().unwrap();

                    deallocate_kv_pages(&self.queue, &[redundant]).unwrap();
                    remaining_redundancy -= self.kv_page_size;
                }
                self.last_kv_page_len = self.kv_page_size - remaining_redundancy;
            }

            next_token_ids = verified_tokens;
            generated_token_ids.extend(&next_token_ids);

            if stop_condition.should_stop(&generated_token_ids) {
                break;
            } else {
                self.token_ids.extend(&next_token_ids);
            }

            deallocate_embeds(&self.queue, &input_embed_id).unwrap();
            deallocate_embeds(&self.queue, &output_embed_id).unwrap();
        }

        self.pending_token_ids.clear();
        self.pending_token_ids.extend(&next_token_ids);

        self.tokenizer.detokenize(&generated_token_ids)
    }
}
