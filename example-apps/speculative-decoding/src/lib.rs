use inferlet::{sampler, stop_condition, traits::Tokenize};
use pico_args::Arguments;
use std::collections::HashMap;
use std::ffi::OsString;
use std::time::Instant;

use inferlet::drafter::Drafter;

/// A simple fixed-size queue.
pub struct FixedSizeQueue<T, const N: usize> {
    buf: [T; N],
    head: usize,
    len: usize,
}

impl<T: Default, const N: usize> FixedSizeQueue<T, N> {
    pub fn new() -> Self {
        Self {
            buf: std::array::from_fn(|_| T::default()),
            head: 0,
            len: 0,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.len == N {
            self.buf[self.head] = item;
            self.head = (self.head + 1) % N;
        } else {
            let tail = (self.head + self.len) % N;
            self.buf[tail] = item;
            self.len += 1;
        }
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let item = std::mem::take(&mut self.buf[self.head]);
            self.head = (self.head + 1) % N;
            self.len -= 1;
            Some(item)
        }
    }

    pub fn extend(&mut self, items: impl ExactSizeIterator<Item = T>) {
        let count = items.len();
        if count >= N {
            let mut iter = items.skip(count - N);
            for i in 0..N {
                self.buf[i] = iter.next().unwrap();
            }
            self.head = 0;
            self.len = N;
        } else {
            let to_remove = (self.len + count).saturating_sub(N);
            for _ in 0..to_remove {
                self.pop_front();
            }
            for item in items {
                self.push(item);
            }
        }
    }
}

/// A simple fixed-size LRU cache.
#[derive(Debug, Clone, Copy)]
struct LruCache<T: Copy + PartialEq, const CAP: usize> {
    items: [Option<T>; CAP],
    len: usize,
}

impl<T: Copy + PartialEq, const CAP: usize> LruCache<T, CAP> {
    pub fn new() -> Self {
        Self {
            items: [None; CAP],
            len: 0,
        }
    }

    pub fn is_hit(&self, item: &T) -> bool {
        self.items[..self.len].iter().any(|&x| x == Some(*item))
    }

    pub fn update(&mut self, item: T) {
        if let Some(pos) = self.items[..self.len].iter().position(|&x| x == Some(item)) {
            for i in (1..=pos).rev() {
                self.items[i] = self.items[i - 1];
            }
            self.items[0] = Some(item);
        } else {
            if self.len < CAP {
                for i in (1..=self.len).rev() {
                    self.items[i] = self.items[i - 1];
                }
                self.items[0] = Some(item);
                self.len += 1;
            } else {
                for i in (1..CAP).rev() {
                    self.items[i] = self.items[i - 1];
                }
                self.items[0] = Some(item);
            }
        }
    }
}

/// A cache table that maps previous tokens to next token sequences.
pub struct CacheDrafter<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> {
    table: HashMap<[u32; N_PREV], LruCache<[u32; N_NEXT], CACHE_SIZE>>,
    curr: [u32; N_PREV],
}

impl<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize>
    CacheDrafter<N_PREV, N_NEXT, CACHE_SIZE>
{
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
            curr: [0; N_PREV],
        }
    }

    pub fn is_hit(&self, prev_tokens: &[u32; N_PREV], next_tokens: &[u32; N_NEXT]) -> bool {
        self.table
            .get(prev_tokens)
            .map_or(false, |lru| lru.is_hit(next_tokens))
    }

    pub fn update_cache(&mut self, prev_tokens: [u32; N_PREV], next_tokens: [u32; N_NEXT]) {
        self.table
            .entry(prev_tokens)
            .or_insert_with(LruCache::new)
            .update(next_tokens);
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl<const N_PREV: usize, const N_NEXT: usize, const CACHE_SIZE: usize> Drafter
    for CacheDrafter<N_PREV, N_NEXT, CACHE_SIZE>
{
    fn update(&mut self, context: &[u32]) {
        if context.len() > N_PREV {
            let start = context.len().saturating_sub(N_PREV);
            //println!("start: {:?}", &context[start..]);
            self.curr.copy_from_slice(&context[start..]);
        }

        let window_size = N_PREV + N_NEXT;
        if context.len() >= window_size {
            for window in context.windows(window_size) {
                let prev_tokens: [u32; N_PREV] = window[..N_PREV].try_into().unwrap();
                let next_tokens: [u32; N_NEXT] = window[N_PREV..].try_into().unwrap();
                self.update_cache(prev_tokens, next_tokens);
            }
        }
    }

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
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

        (spec_token_ids, spec_pos_ids)
    }
}

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: program [OPTIONS]

A simple inferlet to run a chat model with a configurable CacheDrafter.

Options:
  -p, --prompt <STRING>    The prompt to send to the model
                           (default: "Keep print 'helloworld' 100 times")
  -n, --max-tokens <INT>   The maximum number of new tokens to generate
                           (default: 256)
  -h, --help               Print help information
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // --- Argument Parsing ---
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Keep print 'helloworld' 100 times".to_string());

    let max_num_outputs: usize = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(256);

    let n_next: usize = args
        .opt_value_from_str("--n-next")
        .map_err(|e| e.to_string())?
        .unwrap_or(1);

    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    // --- Main Logic ---
    let start = Instant::now();

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = model.create_context();
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill(&format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
        prompt
    ));
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let mut sampler = sampler::GreedySampler::new();
    let mut stop_condition = stop_condition::any(
        stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        stop_condition::Length::new(max_num_outputs),
    );

    // --- Generation with Drafter Selection ---
    let mut drafter = CacheDrafter::<1, 2, 4>::new();

    let output = ctx
        .generate_with_drafter(&mut drafter, &mut sampler, &mut stop_condition)
        .await;

    let output_token_ids = tokenizer.tokenize(&output);

    println!(
        "Output: {:?} (total elapsed: {:?})",
        output,
        start.elapsed()
    );

    // Compute per-token latency, avoiding division by zero.
    if !output_token_ids.is_empty() {
        println!(
            "Per token latency: {:?}",
            start.elapsed() / (output_token_ids.len() as u32)
        );
    }

    Ok(())
}
