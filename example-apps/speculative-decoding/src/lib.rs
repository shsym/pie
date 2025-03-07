use std::time::Instant;
use symphony::{RunSync, l4m};

use std::collections::HashMap;
use std::{mem, slice};
use symphony::sampler::Sampler;
use symphony::stop_condition::StopCondition;

/// A simple fixed-size LRU cache implemented using a fixed-size array.
/// It stores items of type `T` (which must be Copy and comparable) and
/// maintains at most `CAP` items in LRU order (most-recent at index 0).
#[derive(Debug, Clone, Copy)]
struct LruCache<T: Copy + PartialEq, const CAP: usize> {
    items: [Option<T>; CAP],
    len: usize,
}

impl<T: Copy + PartialEq, const CAP: usize> LruCache<T, CAP> {
    /// Creates a new, empty LRU cache.
    pub fn new() -> Self {
        Self {
            items: [None; CAP],
            len: 0,
        }
    }

    /// Checks if the given item is present in the cache.
    pub fn is_hit(&self, item: &T) -> bool {
        self.items[..self.len].iter().any(|&x| x == Some(*item))
    }

    /// Updates the cache with the given item:
    /// - If it exists, moves it to the front.
    /// - If not, inserts it at the front.
    /// - If the capacity is exceeded, the least-recently used item is dropped.
    pub fn update(&mut self, item: T) {
        if let Some(pos) = self.items[..self.len].iter().position(|&x| x == Some(item)) {
            // Found: shift items between index 0 and pos right by one.
            for i in (1..=pos).rev() {
                self.items[i] = self.items[i - 1];
            }
            self.items[0] = Some(item);
        } else {
            // Not found: need to insert at front.
            if self.len < CAP {
                // Shift elements right, then insert.
                for i in (1..=self.len).rev() {
                    self.items[i] = self.items[i - 1];
                }
                self.items[0] = Some(item);
                self.len += 1;
            } else {
                // Capacity full: shift right and overwrite the last element.
                for i in (1..CAP).rev() {
                    self.items[i] = self.items[i - 1];
                }
                self.items[0] = Some(item);
            }
        }
    }
}

/// A cable table (cache) that maps a fixed-size array of previous tokens to an LRU cache
/// of next token sequences. The sizes of the token arrays and the LRU capacity are known at compile time.
pub struct CacheTable<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> {
    table: HashMap<[u32; N_PREV], LruCache<[u32; N_NEXT], CACHE_SIZE>>,
}

impl<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize>
    CacheTable<N_PREV, N_NEXT, CACHE_SIZE>
{
    /// Creates a new cache table.
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    /// Checks if for the given `prev_tokens` key the specified `next_tokens` is cached.
    pub fn is_hit(&self, prev_tokens: &[u32; N_PREV], next_tokens: &[u32; N_NEXT]) -> bool {
        self.table
            .get(prev_tokens)
            .map_or(false, |lru| lru.is_hit(next_tokens))
    }

    /// Updates the cache for the given `prev_tokens` key with the `next_tokens` sequence.
    /// This moves the entry to the front (most-recent) and enforces the cache size limit.
    pub fn update(&mut self, prev_tokens: [u32; N_PREV], next_tokens: [u32; N_NEXT]) {
        self.table
            .entry(prev_tokens)
            .or_insert_with(LruCache::new)
            .update(next_tokens);
    }

    /// Clears the entire cache.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

struct SpeculativeContext<'a> {
    cache_table: CacheTable<2, 1, 2>,
    context: symphony::Context<'a>,
}

impl<'a> SpeculativeContext<'a> {
    pub fn new() -> Self {
        Self {
            cache_table: CacheTable::new(),
            context: symphony::Context::new(),
        }
    }

    pub fn fill(&mut self, text: &str) {
        self.context.fill(text);
    }

    // This overrides the generate method in the symphony::Context struct
    pub fn generate<S: Sampler, C: StopCondition>(
        &mut self,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        //let until_token_ids = l4m::tokenize(until);

        let block_size = l4m::get_block_size() as usize;
        // the seed must not be empty
        assert!(!self.context.leftover_token_ids.is_empty());

        // initialize the working block
        // ensure we have enough blocks
        if self.context.free_block_ids.is_empty() {
            self.context.grow(block_size);
        }
        let pos_offset = self.context.occupied_block_ids.len() * block_size;
        let mut working_block_id = self.context.free_block_ids.pop().unwrap();
        self.context.occupied_block_ids.push(working_block_id);

        let mut working_token_ids = mem::take(&mut self.context.leftover_token_ids);
        let mut working_position_ids: Vec<u32> =
            (pos_offset as u32..(pos_offset + working_token_ids.len()) as u32).collect();

        let input_block_embeds = l4m::allocate_embeds(self.context.stream, block_size as u32);
        let output_block_embeds = l4m::allocate_embeds(self.context.stream, block_size as u32);
        let next_dist = l4m::allocate_dists(self.context.stream, 1)[0];

        // put the remaining tokens into the last block
        l4m::embed_text(
            self.context.stream,
            &input_block_embeds[..working_token_ids.len()],
            &working_token_ids,
            &working_position_ids,
        );

        let mut generated_token_ids = Vec::new();
        let parent_occupied_block_ids = self.context.get_parent_occupied_block_ids();

        loop {
            let ctx_block_ids = [
                parent_occupied_block_ids.as_slice(),
                self.context.occupied_block_ids.as_slice(),
            ]
            .concat();

            l4m::fill_block(
                self.context.stream,
                working_block_id,
                &ctx_block_ids,
                &input_block_embeds[..working_token_ids.len()],
                &output_block_embeds[..working_token_ids.len()],
            );

            // let's sample the next token
            l4m::decode_token_dist(
                self.context.stream,
                slice::from_ref(&output_block_embeds[working_token_ids.len() - 1]),
                slice::from_ref(&next_dist),
            );

            let sampled = l4m::sample_top_k(self.context.stream, slice::from_ref(&next_dist), 32);

            let (next_token_ids, next_token_logits) = &sampled[0];

            let next_token_id = sampler.sample(next_token_ids, &next_token_logits);

            //let next_token_id = next_token_ids[0];
            let next_position_id = working_position_ids.last().unwrap() + 1;

            generated_token_ids.push(next_token_id);

            // if this was the last block,
            if working_token_ids.len() == block_size {
                // get the new working block
                if self.context.free_block_ids.is_empty() {
                    self.context.grow(block_size);
                }

                working_block_id = self.context.free_block_ids.pop().unwrap();
                self.context.occupied_block_ids.push(working_block_id);

                working_position_ids.clear();
                working_token_ids.clear();
            }

            working_token_ids.push(next_token_id);
            working_position_ids.push(next_position_id);

            // check if
            if stop_condition.should_stop(&generated_token_ids) {
                break;
            }

            // embed the next token
            l4m::embed_text(
                self.context.stream,
                slice::from_ref(&input_block_embeds[working_token_ids.len() - 1]),
                &[next_token_id],
                slice::from_ref(&working_position_ids[working_token_ids.len() - 1]),
            );
        }

        // free the resources
        l4m::deallocate_embeds(self.context.stream, &input_block_embeds);
        l4m::deallocate_embeds(self.context.stream, &output_block_embeds);
        l4m::deallocate_dists(self.context.stream, &[next_dist]);

        // pop the last block
        self.context
            .free_block_ids
            .push(self.context.occupied_block_ids.pop().unwrap());

        self.context.leftover_token_ids.clear();
        self.context
            .leftover_token_ids
            .append(&mut working_token_ids);

        // decode the generated tokens
        let result = l4m::detokenize(&generated_token_ids);

        result
    }
}

struct SpeculativeDecoding;

// create a default stream constant

impl RunSync for SpeculativeDecoding {
    fn run() -> Result<(), String> {
        let start = Instant::now();

        // TODO: Prepopulate the cache table with some entries

        let mut ctx = SpeculativeContext::new();
        ctx.fill("<|begin_of_text|>");
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>");
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let output_text = ctx.generate_until("<|eot_id|>", max_num_outputs);

        println!("Output: {:?} (elapsed: {:?})", output_text, start.elapsed());

        Ok(())
    }
}

symphony::main_sync!(SpeculativeDecoding);
