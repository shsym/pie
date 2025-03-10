use std::time::Instant;
use symphony::{Context, Run, sampler, stop_condition};

use std::collections::HashMap;
use symphony::drafter::Drafter;

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
pub struct CacheDrafter<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> {
    table: HashMap<[u32; N_PREV], LruCache<[u32; N_NEXT], CACHE_SIZE>>,
    curr: [u32; N_PREV],
}

impl<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize>
    CacheDrafter<N_PREV, N_NEXT, CACHE_SIZE>
{
    /// Creates a new cache table.
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
            curr: [0; N_PREV],
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
    pub fn update_cache(&mut self, prev_tokens: [u32; N_PREV], next_tokens: [u32; N_NEXT]) {
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

impl<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> Drafter
    for CacheDrafter<N_PREV, N_NEXT, CACHE_SIZE>
{
    fn update(&mut self, context: &[u32]) {
        // Update the pointer to the last N_PREV tokens of the context.

        if context.len() >= N_PREV {
            let start = context.len().saturating_sub(N_PREV);
            for i in 0..N_PREV {
                self.curr[i] = context[start + i];
            }
        }

        // Take a window of (N_PREV + N_NEXT) tokens from the start of the sequence, shifting by 1 token each time.
        // Then update the cache with the previous N_PREV tokens and the next N_NEXT tokens.
        let window_size = N_PREV + N_NEXT;
        if context.len() >= window_size {
            // Slide over the tokens one at a time
            for window in context.windows(window_size) {
                // Convert slices into fixed-size arrays
                let prev_tokens: [u32; N_PREV] = window[..N_PREV].try_into().unwrap();
                let next_tokens: [u32; N_NEXT] = window[N_PREV..].try_into().unwrap();
                self.update_cache(prev_tokens, next_tokens);
            }
        }
    }

    fn draft(&mut self, max_tokens: usize) -> (Vec<u32>, Vec<u32>) {
        // build a speculation Trie. (https://en.wikipedia.org/wiki/Trie)
        // Rn, its just a single level Trie.
        let mut spec_token_ids = Vec::new();
        let mut spec_pos_ids = Vec::new();

        if let Some(cache) = self.table.get(&self.curr) {
            for item in cache.items {
                if let Some(item) = item {
                    spec_token_ids.extend(item);
                    spec_pos_ids.extend(1..=item.len() as u32);
                }
            }
        }

        if spec_token_ids.len() >= max_tokens {
            spec_token_ids.truncate(max_tokens);
            spec_pos_ids.truncate(max_tokens);
        }

        (spec_token_ids, spec_pos_ids)
    }
}

struct SpeculativeDecoding;

// create a default stream constant

impl Run for SpeculativeDecoding {
    async fn run() -> Result<(), String> {
        let start = Instant::now();

        // TODO: Prepopulate the cache table with some entries
        let max_num_outputs = 128;

        let model = symphony::Model::new(&symphony::available_models()[0]).unwrap();

        let mut ctx = model.create_context();
        ctx.fill("<|begin_of_text|>").await;
        ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain the LLM decoding process ELI5.<|eot_id|>").await;
        ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n")
            .await;

        let mut drafter = CacheDrafter::<1, 1, 16>::new();
        let mut sampler = sampler::GreedySampler::new();

        let mut stop_condition = stop_condition::any(
            stop_condition::Until::new(model.tokenize("<|eot_id|>")),
            stop_condition::Length::new(max_num_outputs),
        );

        let output = ctx
            .generate(&mut drafter, &mut sampler, &mut stop_condition)
            .await;

        println!("Out {:?}, elapsed: {:?}", output, start.elapsed());

        Ok(())
    }
}

symphony::main!(SpeculativeDecoding);
