use inferlet2::sampler::{self, Sampler};
use inferlet2::stop_condition::{self, StopCondition};
use inferlet2::traits::forward::Forward;
use inferlet2::traits::tokenize::{Tokenize, Tokenizer};
use inferlet2::traits::{Allocate, ForwardText};
use inferlet2::{Model, Queue};
use std::cmp::Ordering;
use std::time::Instant;

#[derive(Debug)]
pub struct Context {
    queue: Queue,
    model: Model,
    tokenizer: Tokenizer,
    token_ids: Vec<u32>,
    token_ids_pending: Vec<u32>,
    kv_page_ids: Vec<u32>,
    kv_page_last_len: usize,
    kv_page_size: usize,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.queue.deallocate_kv_pages(&self.kv_page_ids);
    }
}

impl Context {
    pub fn new(model: &Model) -> Self {
        if !model.has_traits(&["input_text", "tokenize", "output_text"]) {
            panic!("Model must have input_text, tokenize, and output_text traits");
        }

        let queue = model.create_queue();
        let kv_page_size = queue.get_kv_page_size() as usize;
        let tokenizer = queue.get_tokenizer();

        Context {
            queue,
            model: model.clone(),
            tokenizer,
            token_ids: Vec::new(),
            token_ids_pending: Vec::new(),
            kv_page_ids: Vec::new(),
            kv_page_last_len: 0,
            kv_page_size,
        }
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn fill(&mut self, text: &str) {
        let new_token_ids = self.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids);
    }

    pub fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        self.token_ids_pending.extend(new_token_ids);
    }

    pub fn fill_token(&mut self, new_token_id: u32) {
        self.token_ids_pending.push(new_token_id);
    }

    pub fn fork(&self) -> Self {
        let (new_tokens, new_pending, new_kv_pages, new_last_len) =
            if self.kv_page_last_len == self.kv_page_size {
                (
                    self.token_ids.clone(),
                    self.token_ids_pending.clone(),
                    self.kv_page_ids.clone(),
                    self.kv_page_last_len,
                )
            } else {
                let kept_kv_page_len = self.kv_page_ids.len().saturating_sub(1);
                let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

                let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
                let forked_kv_page_ids = self.kv_page_ids[..kept_kv_page_len].to_vec();

                let forked_pending_token_ids = [
                    &self.token_ids[kept_tokens_len..],
                    &self.token_ids_pending[..],
                ]
                .concat();

                let forked_last_kv_page_len = if !forked_kv_page_ids.is_empty() {
                    self.kv_page_size
                } else {
                    0
                };
                (
                    forked_token_ids,
                    forked_pending_token_ids,
                    forked_kv_page_ids,
                    forked_last_kv_page_len,
                )
            };

        self.queue.increase_ref_count(&new_kv_pages);

        Context {
            queue: self.model.create_queue(),
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            token_ids: new_tokens,
            token_ids_pending: new_pending,
            kv_page_ids: new_kv_pages,
            kv_page_last_len: new_last_len,
            kv_page_size: self.kv_page_size,
        }
    }

    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = self.token_ids.len();
        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_page_ids.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                let new_pages_needed = required_pages - current_pages;
                let new_kv_page_ids = self.queue.allocate_kv_pages(new_pages_needed);
                self.kv_page_ids.extend(new_kv_page_ids);
            }
            Ordering::Less => {
                let redundant_page_ids = self.kv_page_ids.split_off(required_pages);
                self.queue.deallocate_kv_pages(&redundant_page_ids);
            }
            Ordering::Equal => {}
        }

        let last_page_len = new_total_tokens % self.kv_page_size;
        self.kv_page_last_len = if last_page_len == 0 && new_total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };
    }

    fn grow_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(num_tokens as isize);
    }

    fn shrink_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(-(num_tokens as isize));
    }

    pub fn flush(&mut self) {
        if self.token_ids_pending.len() < 2 {
            return;
        }

        let process_count = self.token_ids_pending.len() - 1;
        let pending_token_ids = self
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let position_ids = (self.token_ids.len() as u32
            ..(self.token_ids.len() + pending_token_ids.len()) as u32)
            .collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        self.queue.forward_text_no_output(
            self.kv_page_last_len as u32,
            &self.kv_page_ids,
            &pending_token_ids,
            &position_ids,
            &[],
        );

        self.token_ids.extend(pending_token_ids);
    }
}

/// Generates text using Parallel Jacobi Decoding as a standalone function.
///
/// This function implements a form of speculative decoding where the main model
/// acts as its own drafter. It accelerates generation by predicting a sequence
/// of future tokens in parallel and then verifying them in a single, efficient
/// autoregressive pass.
///
/// # Arguments
///
/// * `ctx`: A mutable reference to the `Context` which holds the model state.
/// * `sampler`: A mutable reference to an object implementing the `Sampler` trait.
/// * `stop_condition`: A mutable reference to an object implementing `StopCondition`.
/// * `speculation_length`: The number of future tokens to predict in parallel.
///
/// # Returns
///
/// A `String` containing the full sequence of generated text.
pub async fn generate_with_parallel_jacobi_decoding<S: Sampler, C: StopCondition>(
    ctx: &mut Context,
    sampler: &mut S,
    stop_condition: &mut C,
    speculation_length: usize,
) -> String {
    ctx.flush();

    assert_eq!(
        ctx.token_ids_pending.len(),
        1,
        "Must have exactly one seed token for Jacobi decoding."
    );

    let mut all_generated_tokens = Vec::new();
    // NOTE: Only works for llama-3. Different models may have different padding tokens.
    let pad_token_id = 0;

    loop {
        // --- 1. Draft Phase (Parallel Prediction) ✍️ ---
        let draft_tokens = {
            // Fork the context to run a temporary forward pass without affecting the main state.
            let mut draft_context = ctx.fork();
            let seed_token = *ctx.token_ids_pending.last().unwrap();

            // Prepare the input: [seed, pad, pad, ...]
            let mut draft_input_tokens = vec![seed_token];
            draft_input_tokens.resize(1 + speculation_length, pad_token_id);

            let draft_input_positions = {
                let base_pos = draft_context.token_ids.len() as u32;
                (base_pos..base_pos + draft_input_tokens.len() as u32).collect::<Vec<u32>>()
            };

            // The output at index `i` is for the token *after* `input[i]`.
            // We want `speculation_length` draft tokens.
            let output_indices = (0..speculation_length)
                .map(|i| i as u32)
                .collect::<Vec<u32>>();

            // Run the parallel forward pass to get draft distributions.
            let draft_distributions = draft_context
                .queue()
                .forward_text(
                    draft_context.kv_page_last_len as u32,
                    &draft_context.kv_page_ids,
                    &draft_input_tokens,
                    &draft_input_positions,
                    &[],
                    &output_indices,
                )
                .await;

            // Sample from each distribution to create the draft.
            draft_distributions
                .iter()
                .map(|dist| sampler.sample(&dist.ids, &dist.probs))
                .collect::<Vec<u32>>()
            // draft_context is dropped here, cleaning up its resources.
        };

        // --- 2. Verification Phase ✅ ---
        // Pop the seed token from the main context to begin the verification pass.
        let seed_token = ctx.token_ids_pending.pop().unwrap();

        // Combine the seed and the draft into a single verification batch.
        let verification_batch_tokens = [&[seed_token], draft_tokens.as_slice()].concat();
        let verification_batch_positions = {
            let base_position = ctx.token_ids.len() as u32;
            (base_position..base_position + verification_batch_tokens.len() as u32)
                .collect::<Vec<u32>>()
        };

        // Grow the main context's KV cache for the entire verification batch.
        ctx.grow_kv_pages(verification_batch_tokens.len());

        // Run the single autoregressive verification pass.
        let output_distributions = ctx
            .queue()
            .forward_text(
                ctx.kv_page_last_len as u32,
                &ctx.kv_page_ids,
                &verification_batch_tokens,
                &verification_batch_positions,
                &[],
                &(0..verification_batch_tokens.len() as u32).collect::<Vec<u32>>(),
            )
            .await;

        // --- 3. Acceptance & Correction 🤝 ---
        let mut accepted_tokens = Vec::new();
        let mut num_accepted_drafts = 0;

        for (i, token_distribution) in output_distributions.iter().enumerate() {
            let verifier_token = sampler.sample(&token_distribution.ids, &token_distribution.probs);

            if i < draft_tokens.len() && verifier_token == draft_tokens[i] {
                accepted_tokens.push(verifier_token);
                num_accepted_drafts += 1;
            } else {
                accepted_tokens.push(verifier_token);
                break;
            }
        }

        // --- 4. State Update 🔄 ---
        let redundant_draft_count = draft_tokens.len() - num_accepted_drafts;
        ctx.shrink_kv_pages(redundant_draft_count);

        ctx.token_ids.push(seed_token);
        ctx.token_ids
            .extend_from_slice(&accepted_tokens[..num_accepted_drafts]);

        all_generated_tokens.extend_from_slice(&accepted_tokens);

        if stop_condition.should_stop(&all_generated_tokens) {
            break;
        }

        let next_seed_token = *accepted_tokens.last().unwrap();
        ctx.fill_token(next_seed_token);
    }

    ctx.tokenizer().detokenize(&all_generated_tokens)
}

#[inferlet2::main]
async fn main() -> Result<(), String> {
    let start = Instant::now();

    let max_num_outputs = 256;
    let speculation_length = 1; // Number of tokens to speculate in parallel

    let model = inferlet2::get_auto_model();
    let tokenizer = model.get_tokenizer();

    let mut ctx = Context::new(&model);
    ctx.fill("<|begin_of_text|>");
    ctx.fill("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant.<|eot_id|>");
    ctx.fill("<|start_header_id|>user<|end_header_id|>\n\nExplain LLM decoding process in ELI5.<|eot_id|>");
    ctx.fill("<|start_header_id|>assistant<|end_header_id|>\n\n");

    let mut sampler = sampler::GreedySampler::new();

    let mut stop_condition = stop_condition::any(
        stop_condition::Until::new(tokenizer.tokenize("<|eot_id|>")),
        stop_condition::Length::new(max_num_outputs),
    );

    //let mut stop_condition =         stop_condition::Length::new(max_num_outputs);

    println!(
        "Starting generation with Parallel Jacobi Decoding (speculation_length = {})...",
        speculation_length
    );

    // Call the standalone function
    let output = generate_with_parallel_jacobi_decoding(
        &mut ctx,
        &mut sampler,
        &mut stop_condition,
        speculation_length,
    )
    .await;

    let elapsed = start.elapsed();
    let output_token_ids = tokenizer.tokenize(&output);

    println!("\n--- Output ---\n{}\n--------------", output);

    println!(
        "Total elapsed: {:?}, Tokens generated: {}",
        elapsed,
        output_token_ids.len()
    );

    // compute per token latency
    if !output_token_ids.is_empty() {
        println!(
            "Per-token latency: {:?}",
            elapsed / output_token_ids.len() as u32
        );
    }

    Ok(())
}
