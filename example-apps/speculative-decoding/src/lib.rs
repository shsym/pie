use std::time::Instant;
use symphony::{RunSync, l4m, sampler, stop_condition};

use std::collections::HashMap;
use std::{mem};
use symphony::sampler::Sampler;
use symphony::stop_condition::StopCondition;

pub struct FixedSizeQueue<T, const N: usize> {
    buf: [T; N],
    head: usize, // index of the oldest element
    len: usize,  // number of valid elements in the queue
}

impl<T: Default, const N: usize> FixedSizeQueue<T, N> {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            // Initialize the buffer with T::default()
            buf: std::array::from_fn(|_| T::default()),
            head: 0,
            len: 0,
        }
    }

    /// Pushes an item onto the queue.
    /// If the queue is full, it overwrites the oldest item.
    pub fn push(&mut self, item: T) {
        if self.len == N {
            // Overwrite the oldest element at head and move head forward.
            self.buf[self.head] = item;
            self.head = (self.head + 1) % N;
        } else {
            // Place the new item at the tail.
            let tail = (self.head + self.len) % N;
            self.buf[tail] = item;
            self.len += 1;
        }
    }

    /// Removes and returns the oldest item in the queue.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // Replace the element at head with the default value.
            let item = std::mem::take(&mut self.buf[self.head]);
            self.head = (self.head + 1) % N;
            self.len -= 1;
            Some(item)
        }
    }

    /// Extends the queue with items from an iterator.
    ///
    /// - If the iterator yields at least N items, only the last N items are kept.
    /// - Otherwise, enough items are removed from the front so that after pushing the new items,
    ///   the total number of elements does not exceed N.
    pub fn extend(&mut self, items: impl ExactSizeIterator<Item = T>) {
        let count = items.len();
        if count >= N {
            // There are at least N new items.
            // Skip the first count - N items so that only the last N are used.
            let mut iter = items.skip(count - N);
            for i in 0..N {
                // It is safe to unwrap since we expect exactly N items.
                self.buf[i] = iter.next().unwrap();
            }
            self.head = 0;
            self.len = N;
        } else {
            // Remove as many items as needed so that len + count does not exceed N.
            let to_remove = (self.len + count).saturating_sub(N);
            for _ in 0..to_remove {
                self.pop_front();
            }
            // Now push the new items.
            for item in items {
                self.push(item);
            }
        }
    }
}

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

    pub fn update(&mut self, tokens: &[u32]) {
        // Take a window of (N_PREV + N_NEXT) tokens from the start of the sequence, shifting by 1 token each time.
        // Then update the cache with the previous N_PREV tokens and the next N_NEXT tokens.
        let window_size = N_PREV + N_NEXT;
        if tokens.len() < window_size {
            return;
        }
        // Slide over the tokens one at a time
        for window in tokens.windows(window_size) {
            // Convert slices into fixed-size arrays
            let prev_tokens: [u32; N_PREV] = window[..N_PREV].try_into().unwrap();
            let next_tokens: [u32; N_NEXT] = window[N_PREV..].try_into().unwrap();
            self.update_entry(prev_tokens, next_tokens);
        }
    }

    /// Updates the cache for the given `prev_tokens` key with the `next_tokens` sequence.
    /// This moves the entry to the front (most-recent) and enforces the cache size limit.
    pub fn update_entry(&mut self, prev_tokens: [u32; N_PREV], next_tokens: [u32; N_NEXT]) {
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

struct SpeculativeContext<'a, const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> {
    cache_table: CacheTable<N_PREV, N_NEXT, CACHE_SIZE>,
    context: symphony::Context<'a>,
}

impl<'a, const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize>
    SpeculativeContext<'a, N_PREV, N_NEXT, CACHE_SIZE>
{
    pub fn new() -> Self {
        Self {
            cache_table: CacheTable::new(),
            context: symphony::Context::new(),
        }
    }

    pub fn fill(&mut self, text: &str) {
        let token_ids = l4m::tokenize(text);
        self.cache_table.update(&token_ids);
        self.context.fill_tokens(token_ids);
    }

    pub fn speculate(&mut self, ctx: &[u32; N_PREV], max_trie_size: usize) -> (Vec<u32>, Vec<u32>) {
        // build a speculation Trie. (https://en.wikipedia.org/wiki/Trie)
        // Rn, its just a single level Trie.
        let mut spec_token_ids = Vec::new();
        let mut spec_pos_ids = Vec::new();

        if let Some(cache) = self.cache_table.table.get(ctx) {
            for item in cache.items {
                if let Some(item) = item {
                    spec_token_ids.extend(item);
                    spec_pos_ids.extend(1..=item.len() as u32);
                }
            }
        }

        if spec_token_ids.len() >= max_trie_size {
            spec_token_ids.truncate(max_trie_size);
            spec_pos_ids.truncate(max_trie_size);
        }

        (spec_token_ids, spec_pos_ids)
    }

    pub fn generate_until(&mut self, stop_str: &str, max_tokens: usize) -> String {
        let mut sampler = sampler::GreedySampler::new();

        let mut stop_condition = stop_condition::any(
            stop_condition::Until::new(stop_str),
            stop_condition::Length::new(max_tokens),
        );

        self.generate(&mut sampler, &mut stop_condition)
    }

    // This overrides the generate method in the symphony::Context struct
    pub fn generate<S: Sampler, C: StopCondition>(
        &mut self,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        let block_size = l4m::get_block_size() as usize;
        // the seed must not be empty
        assert!(!self.context.pending_token_ids.is_empty());

        // initialize the working block
        // ensure we have enough blocks
        if self.context.free_block_ids.is_empty() {
            self.context.grow(block_size);
        }

        let pos_offset = self.context.occupied_block_ids.len() * block_size;
        let mut working_block_id = self.context.free_block_ids.pop().unwrap();
        self.context.occupied_block_ids.push(working_block_id);

        let input_block_embeds = l4m::allocate_embeds(self.context.stream, block_size as u32);
        let output_block_embeds = l4m::allocate_embeds(self.context.stream, block_size as u32);
        let next_dists = l4m::allocate_dists(self.context.stream, block_size as u32);

        let mut generated_token_ids = Vec::new();
        let parent_occupied_block_ids = self.context.get_parent_occupied_block_ids();

        let mut processed_token_ids = Vec::new();
        let mut processed_position_ids = Vec::new();

        let mut processing_token_ids = mem::take(&mut self.context.pending_token_ids);
        let mut processing_position_ids: Vec<u32> =
            (pos_offset as u32..(pos_offset + processed_token_ids.len()) as u32).collect();

        let mut spec_ctx = FixedSizeQueue::<u32, N_PREV>::new();
        spec_ctx.extend(self.context.processed_token_ids.iter().copied());

        loop {
            // come up with the speculations
            let max_trie_size =
                block_size - (processed_token_ids.len() + processing_token_ids.len());

            spec_ctx.extend(processing_token_ids.iter().copied());

            let (spec_token_ids, spec_pos_ids) = self.speculate(&spec_ctx.buf, max_trie_size);

            let combined_token_ids =
                [processing_token_ids.as_slice(), spec_token_ids.as_slice()].concat();
            let combined_position_ids = {
                let mut res = processing_position_ids.clone();
                let offset = processing_position_ids.last().unwrap();

                res.extend(spec_pos_ids.iter().map(|&x| x + offset));
                res
            };

            let offset_prev = processed_token_ids.len() - 1;
            let offset_last_token = processing_token_ids.len() - 1;
            let valid_len = combined_token_ids.len();

            // embed queued_token_ids + spec_token_ids
            l4m::embed_text(
                self.context.stream,
                &input_block_embeds[offset_prev..=offset_prev + valid_len],
                &combined_token_ids,
                &combined_position_ids,
            );

            let ctx_block_ids = [
                parent_occupied_block_ids.as_slice(),
                self.context.occupied_block_ids.as_slice(),
            ]
            .concat();

            // "Full" fill_block
            l4m::fill_block(
                self.context.stream,
                working_block_id,
                &ctx_block_ids,
                &input_block_embeds[..=offset_prev + valid_len],
                &output_block_embeds[..=offset_prev + valid_len],
            );

            // let's sample the next token
            l4m::decode_token_dist(
                self.context.stream,
                &output_block_embeds[offset_prev + offset_last_token..=offset_prev + valid_len],
                &next_dists[offset_prev + offset_last_token..=offset_prev + valid_len],
            );

            let sampled = l4m::sample_top_k(
                self.context.stream,
                &next_dists[offset_prev + offset_last_token..=offset_prev + valid_len],
                32,
            );

            assert_eq!(sampled.len(), max_trie_size + 1);

            // the verification
            let mut sampled_next_token_ids = Vec::new();

            // The speculation "Trie" is a tree of possibilities. The first token is always correct.
            // R[n] (P[n+1] P[n+2] P[n+3] P[n+2] P[n+3]) (P[n+1] P[n+2]) (P[n+1] P[n+2]) ...

            let mut i = 0;
            while i < sampled.len() {
                let (next_token_ids, next_token_logits) = &sampled[i];
                let next_token_id = sampler.sample(next_token_ids, next_token_logits);

                // The first token (the root of Trie) is always correct.
                if i == 0 {
                    sampled_next_token_ids.push(next_token_id);
                    i += 1;
                    continue;
                }

                // we just got a freebie
                if next_token_id == *sampled_next_token_ids.last().unwrap() {
                    sampled_next_token_ids.push(next_token_id);

                    // check if it has children
                    if i < spec_pos_ids.len() && spec_pos_ids[i - 1] + 1 == spec_pos_ids[i] {
                        i += 1;
                    } else {
                        break;
                    }
                }
                // we may have more chance, in one of our siblings
                else if i < spec_pos_ids.len() {
                    // check if the failed node has siblings
                    let cur_lev = spec_pos_ids[i - 1];
                    let mut next_sibling_offset = None;
                    for (j, lev) in spec_pos_ids[i..].iter().copied().enumerate() {
                        if lev < cur_lev {
                            // we do not have any other options... T_T
                            // our journey ends here
                            break;
                        } else if lev == cur_lev {
                            // another "same" parent option.
                            next_sibling_offset = Some(j + 1);
                            break;
                        }
                    }

                    if let Some(next_sibling_offset) = next_sibling_offset {
                        i += next_sibling_offset;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            let next_position_id = if processed_token_ids.is_empty() {
                processing_position_ids.last().unwrap()
            } else {
                processed_position_ids.last().unwrap()
            } + 1;

            generated_token_ids.extend(&sampled_next_token_ids);

            // update the cache
            self.cache_table.update(
                &generated_token_ids[generated_token_ids
                    .len()
                    .saturating_sub(N_PREV + N_NEXT + 1)..],
            );

            // if this was the last block,
            if processed_token_ids.len() + processing_token_ids.len() == block_size {
                // get the new working block
                if self.context.free_block_ids.is_empty() {
                    self.context.grow(block_size);
                }

                working_block_id = self.context.free_block_ids.pop().unwrap();
                self.context.occupied_block_ids.push(working_block_id);
                self.context
                    .processed_token_ids
                    .extend(&processed_token_ids);
                processed_position_ids.clear();
                processed_token_ids.clear();
            }

            processed_token_ids.append(&mut processing_token_ids);
            processed_position_ids.append(&mut processing_position_ids);

            processing_token_ids.append(&mut sampled_next_token_ids);
            processing_position_ids
                .extend(next_position_id..(next_position_id + sampled_next_token_ids.len() as u32));

            // check if
            if stop_condition.should_stop(&generated_token_ids) {
                break;
            }
        }

        // free the resources
        l4m::deallocate_embeds(self.context.stream, &input_block_embeds);
        l4m::deallocate_embeds(self.context.stream, &output_block_embeds);
        l4m::deallocate_dists(self.context.stream, &next_dists);

        // pop the last block
        self.context
            .free_block_ids
            .push(self.context.occupied_block_ids.pop().unwrap());

        self.context.pending_token_ids.clear();
        self.context
            .pending_token_ids
            .append(&mut processed_token_ids);

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
        let max_num_outputs = 128;

        let mut ctx = SpeculativeContext::<1, 1, 16>::new();
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
