use crate::drafter::Drafter;
use crate::sampler::Sampler;
use crate::stop_condition::StopCondition;
use crate::traits::allocate::Allocate;
use crate::traits::tokenize::{Tokenize, Tokenizer};
use crate::{Model, Queue, sampler, stop_condition};

use crate::traits::forward::Forward;
use crate::traits::input_text::InputText;
use crate::traits::output_text::{Distribution, OutputText};
use std::cmp::Ordering;
use std::{fmt, mem};
use wit_bindgen::rt::async_support::futures::future::join_all;

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

    pub fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
    }

    /// Creates a safe, copy-on-write fork of the context.
    ///
    /// This method creates a new context that shares the immutable history of the current
    /// one. If the last KV-cache page is not full, its tokens are moved to the
    /// `token_ids_pending` buffer of the new context to be recomputed, ensuring state isolation.
    ///
    /// This function will flush any pending tokens in the current context before forking.
    pub fn fork(&mut self) -> Self {
        // flush the pending tokens
        if !self.token_ids_pending.is_empty() {
            self.flush();
        }

        let (new_tokens, new_pending, new_kv_pages, new_last_len) =
            if self.kv_page_last_len == self.kv_page_size {
                // Easy case: the last page is full, we can share everything.
                (
                    self.token_ids.clone(),
                    self.token_ids_pending.clone(),
                    self.kv_page_ids.clone(),
                    self.kv_page_last_len,
                )
            } else {
                // Hard case: the last page is partially full and must be recomputed in the new context.
                let kept_kv_page_len = self.kv_page_ids.len().saturating_sub(1);
                let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

                let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
                let forked_kv_page_ids = self.kv_page_ids[..kept_kv_page_len].to_vec();

                // All tokens from the partial page and any pending tokens must be reprocessed.
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
        self.token_ids_pending.extend(new_token_ids);
    }

    pub fn fill_token(&mut self, new_token_id: u32) {
        self.token_ids_pending.push(new_token_id);
    }

    /// Adjusts the number of KV pages to match the required number of tokens.
    ///
    /// This function handles both allocating new pages (growing) and deallocating
    /// unused pages (shrinking).
    ///
    /// # Arguments
    ///
    /// * `num_tokens`: The number of tokens to add or remove. A positive value
    ///   grows the KV cache, while a negative value shrinks it.
    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = self.token_ids.len();
        // Safely calculate the new total number of tokens after the adjustment.
        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_page_ids.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                // Grow: Allocate new pages if more are needed.
                let new_pages_needed = required_pages - current_pages;
                let new_kv_page_ids = self.queue.allocate_kv_pages(new_pages_needed);
                self.kv_page_ids.extend(new_kv_page_ids);
            }
            Ordering::Less => {
                // Shrink: Deallocate pages that are no longer needed.
                let redundant_page_ids = self.kv_page_ids.split_off(required_pages);
                self.queue.deallocate_kv_pages(&redundant_page_ids);
            }
            Ordering::Equal => {
                // No change in the number of pages is required.
            }
        }

        // Finally, update the length of the last page based on the new total.
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
        // Convert the number of tokens to a negative adjustment for shrinking.
        self.adjust_kv_pages(-(num_tokens as isize));
    }

    /// Processes a batch of pending tokens to update the model's internal state.
    ///
    /// This function is a key step between providing input and generating output. It takes
    /// all tokens from the pending buffer (`token_ids_pending`) **except for the last one**
    /// and runs a forward pass on the model. This operation populates the model's
    /// Key-Value (KV) cache with the state corresponding to the flushed tokens.
    ///
    /// The final token is intentionally left in the pending buffer to serve as the input
    /// "seed" for the subsequent generation step (e.g., a call to `generate()` or `next()`).
    ///
    /// This method will do nothing if there are fewer than two tokens in the pending buffer,
    /// as there must be at least one token to flush and one to keep.
    pub fn flush(&mut self) {
        // We need at least two pending tokens: one to be the "seed" for the next forward pass,
        // and one (or more) to be flushed into the KV cache now.
        if self.token_ids_pending.len() < 2 {
            return;
        }

        // Process all but the last pending token, leaving it for the next generation step.
        let process_count = self.token_ids_pending.len() - 1;
        let pending_token_ids = self
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let position_ids = (self.token_ids.len() as u32
            ..(self.token_ids.len() + pending_token_ids.len()) as u32)
            .collect::<Vec<u32>>();

        let embed_ids = self.queue.allocate_embeds(pending_token_ids.len());
        self.queue
            .embed_text(&embed_ids, &pending_token_ids, &position_ids);

        // First, ensure we have enough KV pages for all tokens.
        self.grow_kv_pages(pending_token_ids.len());

        self.queue.forward(
            self.kv_page_last_len as u32,
            &self.kv_page_ids,
            &embed_ids,
            &[],
        );

        self.token_ids.extend(pending_token_ids);
        self.queue.deallocate_embeds(&embed_ids);
    }

    /// Performs a single, atomic autoregressive decoding step.
    ///
    /// This function is the core of the generation process. It takes the last token
    /// from the pending buffer (`token_ids_pending`), runs a forward pass through the model,
    /// and returns the resulting probability distribution for the next token.
    ///
    /// This operation is stateful and modifies the context:
    /// 1.  The pending token is consumed and moved to the main `token_ids` history.
    /// 2.  The model's internal state (KV cache) is updated to reflect the new token.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Distribution` over the next possible tokens,
    /// or an error if the generation step could not be performed.
    async fn decode_step(&mut self) -> Distribution {
        self.flush();

        assert_eq!(self.token_ids_pending.len(), 1, "Must have one seed token");
        assert_ne!(
            self.kv_page_last_len, 0,
            "Context must be filled before generation"
        );

        let input_embed_id = self.queue.allocate_embeds(1);
        let output_embed_id = self.queue.allocate_embeds(1);

        let next_token_id = self.token_ids_pending.pop().unwrap();
        let next_pos_id = self.token_ids.len() as u32;

        self.queue
            .embed_text(&input_embed_id, &[next_token_id], &[next_pos_id]);

        self.grow_kv_pages(1);

        self.queue.forward(
            self.kv_page_last_len as u32,
            &self.kv_page_ids,
            &input_embed_id,
            &output_embed_id,
        );

        let sampled = self
            .queue
            .get_next_token_distribution(&output_embed_id)
            .await;

        self.token_ids.push(next_token_id);

        self.queue.deallocate_embeds(&input_embed_id);
        self.queue.deallocate_embeds(&output_embed_id);

        // Only one token is generated. So it is safe to unwrap here.
        sampled.into_iter().next().unwrap()
    }

    /// Generates text autoregressively until a stop condition is met.
    ///
    /// This function drives the text generation loop. In each iteration, it calls
    /// `decode_step()` to get a probability distribution, uses the provided `sampler`
    /// to choose the next token, and adds it to the context. The loop continues
    /// until the `stop_condition` signals that generation should end.
    ///
    /// # Arguments
    ///
    /// * `sampler`: A mutable reference to a struct implementing the `Sampler` trait,
    ///   which will be used to sample a token from the model's output distribution at each step.
    /// * `stop_condition`: A mutable reference to a struct implementing the `StopCondition`
    ///   trait, which determines when to halt the generation process.
    ///
    /// # Returns
    ///
    /// A `Result` containing the generated `String` upon successful completion.
    pub async fn generate<S: Sampler, C: StopCondition>(
        &mut self,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        let mut generated_token_ids = Vec::new();

        // The autoregressive generation loop
        loop {
            let dist = self.decode_step().await;
            let next_token_id = sampler.sample(&dist.ids, &dist.probs);
            self.fill_token(next_token_id);

            generated_token_ids.push(next_token_id);

            if stop_condition.should_stop(&generated_token_ids) {
                break;
            }
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }

    /// Generates text using beam search decoding until a stop condition is met.
    ///
    /// Beam search is an autoregressive decoding algorithm that explores multiple
    /// potential sequences (hypotheses) simultaneously. At each step, it maintains
    /// a set of `beam_size` candidate sequences, called "beams."
    ///
    /// # Algorithm
    /// 1.  **Expansion**: For each existing beam, the model predicts the probability
    ///     distribution for the next token. The top `beam_size` most likely next
    ///     tokens are used to create new, longer candidate beams.
    /// 2.  **Scoring**: Each new beam's score is calculated by adding the
    ///     log probability of the new token to the score of its parent beam.
    /// 3.  **Pruning**: All new candidate beams from all parents are gathered,
    ///     sorted by their score in descending order, and only the top `beam_size`
    ///     are kept for the next generation step.
    /// 4.  **Termination**: The generation loop continues until the highest-scoring
    ///     beam in the current set satisfies the `stop_condition`. The text from
    ///     this winning beam is then returned. Note that this heuristic stops at the
    ///     first valid complete beam and may not be the global highest-scoring sequence.
    ///
    /// # Side Effects
    /// Upon completion, this method **modifies the `Context` (`self`)** to adopt the
    /// full state (token history and KV cache) of the winning beam.
    ///
    /// # Arguments
    ///
    /// * `stop_condition`: A mutable reference to a struct implementing the
    ///   `StopCondition` trait. The generation process will halt once the condition
    ///   is met by the highest-scoring beam.
    /// * `beam_size`: The number of candidate sequences to maintain at each step.
    ///   A larger `beam_size` increases the search space and potential quality at the
    ///   cost of computational resources.
    ///
    /// # Returns
    ///
    /// A `String` containing the generated text from the winning beam.
    pub async fn generate_with_beam<C: StopCondition>(
        &mut self,
        stop_condition: &mut C,
        beam_size: usize,
    ) -> String {
        let mut beams = Vec::new();
        // The score is the cumulative probability, starting at 1.0.
        beams.push((self.fork(), vec![], 1.0f32));

        loop {
            // Beams are sorted by score, so the first match is the best valid one found so far.
            if let Some((beam, generated_tokens, _)) =
                beams.iter().find(|(_, g, _)| stop_condition.should_stop(g))
            {
                // Deallocate the pages previously held by `self`.
                let old_pages = mem::take(&mut self.kv_page_ids);
                self.queue.deallocate_kv_pages(&old_pages);

                // Adopt the state from the winning beam.
                self.kv_page_last_len = beam.kv_page_last_len;
                self.token_ids = beam.token_ids.clone();
                self.token_ids_pending = beam.token_ids_pending.clone();
                self.kv_page_ids = beam.kv_page_ids.clone();

                // Increment the ref count for the newly adopted pages, as `self` is a new owner.
                self.queue.increase_ref_count(&self.kv_page_ids);

                return self.tokenizer.detokenize(generated_tokens);
            }

            // Progress the beams in parallel.
            let mut next_dist_futures = Vec::with_capacity(beams.len());
            for (beam, _, _) in beams.iter_mut() {
                let next_dist = beam.decode_step();
                next_dist_futures.push(next_dist);
            }

            // Wait for all forward passes to complete.
            let next_dists = join_all(next_dist_futures).await;

            let mut next_beams = Vec::new();
            for ((mut beam, generated, score), next_dist) in beams.into_iter().zip(next_dists) {
                // Expand this beam with the top candidates.
                for i in 0..beam_size.min(next_dist.ids.len()) {
                    let mut next_beam = beam.fork();
                    // We assume the distribution is sorted by probability in descending order.
                    next_beam.fill_token(next_dist.ids[i]);

                    let mut next_generated = generated.clone();
                    next_generated.push(next_dist.ids[i]);

                    // Update score by multiplying probabilities.
                    let next_score = score * next_dist.probs[i];

                    next_beams.push((next_beam, next_generated, next_score));
                }
            }

            // Prune: Sort all new candidates by score and keep only the top `beam_size`.
            next_beams.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            next_beams.truncate(beam_size);
            beams = next_beams;
        }
    }

    /// Generates text using speculative decoding for faster inference.
    ///
    /// This function accelerates text generation by using a small, fast "drafter"
    /// model to propose a sequence of candidate tokens. These draft tokens are then
    /// verified in a single, parallel forward pass by the main model.
    ///
    /// ### Algorithm
    /// 1.  **✍️ Draft**: A single pending token in the context is used to prompt the
    ///     `drafter`, which generates a short sequence of speculative future tokens.
    /// 2.  **✅ Verify**: The seed token and all draft tokens are combined into a
    ///     single batch and processed by the main model in one forward pass. This is
    ///     the core optimization, as it computes multiple steps' worth of information
    ///     simultaneously.
    /// 3.  **🤝 Accept & Correct**: The main model's output distributions are compared
    ///     against the draft tokens. The function accepts draft tokens as long as they
    ///     match the tokens sampled from the main model's output. When a mismatch
    ///     occurs, the draft is rejected, and the main model's sampled token is used
    ///     as the correction.
    /// 4.  **🔄 Update**: The context's state (token history and KV cache) is efficiently
    ///     updated to reflect all accepted tokens. The loop then continues, using the
    ///     last accepted token as the seed for the next draft.
    ///
    /// This process allows the model to generate multiple tokens for the cost of a
    /// single forward pass when the drafter's predictions are correct, significantly
    /// improving throughput.
    ///
    /// # Arguments
    ///
    /// * `drafter`: A mutable reference to an object implementing the `Drafter` trait,
    ///   used to generate speculative token sequences.
    /// * `sampler`: A mutable reference to an object implementing the `Sampler` trait,
    ///   used to sample from the main model's distributions during verification.
    /// * `stop_condition`: A mutable reference to an object implementing the `StopCondition`
    ///   trait, which determines when to halt generation.
    ///
    /// # Returns
    ///
    /// A `String` containing the full sequence of generated text.
    ///
    /// # Panics
    ///
    /// This function will panic if the context's pending token buffer does not contain
    /// exactly one token before the call, as this token is required to seed the
    /// speculative decoding process.
    pub async fn generate_with_drafter<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        // Ensure any prior inputs are processed and the KV cache is up to date.
        self.flush();

        // Speculative decoding requires a single seed token to start the process.
        assert_eq!(
            self.token_ids_pending.len(),
            1,
            "Must have exactly one seed token to start speculative generation."
        );

        // Synchronize the drafter with the main model's history before starting.
        drafter.update(&self.token_ids);

        let mut all_generated_tokens = Vec::new();

        // --- Generation Loop ---
        // Each iteration performs one step of speculative decoding, which may accept
        // multiple tokens at once.
        loop {
            // --- 1. Draft Phase ✍️ ---
            // Pop the pending token, which serves as the seed for this speculative step.
            let seed_token = self.token_ids_pending.pop().unwrap();

            // Update the drafter with the seed and generate a sequence of draft tokens.
            drafter.update(&[seed_token]);
            let (draft_tokens, draft_pos_ids) = drafter.draft();

            // --- 2. Verification Phase ✅ ---
            // Combine the seed and draft tokens into a single batch for the main model.
            let verification_batch_tokens = [&[seed_token], draft_tokens.as_slice()].concat();
            let verification_batch_positions = {
                let base_position = self.token_ids.len() as u32;
                let mut positions = Vec::with_capacity(verification_batch_tokens.len());
                positions.push(base_position); // Position for the seed token
                // Add the drafter's relative positions to the current context length.
                positions.extend(draft_pos_ids.iter().map(|&pos| pos + base_position));
                positions
            };

            // Allocate resources and expand the KV cache to accommodate the entire batch.
            self.grow_kv_pages(verification_batch_tokens.len());
            let input_embeds = self.queue.allocate_embeds(verification_batch_tokens.len());
            let output_embeds = self.queue.allocate_embeds(verification_batch_tokens.len());

            // Run a single, efficient forward pass on the main model with the combined tokens.
            self.queue.embed_text(
                &input_embeds,
                &verification_batch_tokens,
                &verification_batch_positions,
            );
            self.queue.forward(
                self.kv_page_last_len as u32,
                &self.kv_page_ids,
                &input_embeds,
                &output_embeds,
            );

            // Get the resulting probability distributions for each token in the batch.
            let output_distributions = self.queue.get_next_token_distribution(&output_embeds).await;

            // --- 3. Acceptance & Correction 🤝 ---
            // Compare the main model's predictions with the drafter's guesses to see how many can be accepted.
            let mut accepted_tokens = Vec::new();
            let mut num_accepted_drafts = 0;

            for (i, token_distribution) in output_distributions.iter().enumerate() {
                // Sample a token from the main (verifier) model's output distribution.
                let verifier_token =
                    sampler.sample(&token_distribution.ids, &token_distribution.probs);

                // Check if the verifier's token matches the drafter's guess.
                if i < draft_tokens.len() && verifier_token == draft_tokens[i] {
                    // Match: The draft token was correct. Accept it and continue.
                    accepted_tokens.push(verifier_token);
                    num_accepted_drafts += 1;
                } else {
                    // Mismatch: The verifier's token is the correct one.
                    // Accept this token and terminate the verification for this step.
                    accepted_tokens.push(verifier_token);
                    break;
                }
            }

            // --- 4. State Update 🔄 ---
            // Rewind the KV cache to discard the states of any rejected draft tokens.
            let redundant_draft_count = draft_tokens.len() - num_accepted_drafts;
            self.shrink_kv_pages(redundant_draft_count);

            // Update the main context's history to match the new KV cache state.
            // The cache now holds the state for the `seed_token` and all accepted draft tokens.
            self.token_ids.push(seed_token);
            self.token_ids
                .extend_from_slice(&accepted_tokens[..num_accepted_drafts]);

            // Update the drafter with all newly accepted tokens to improve future drafts.
            drafter.update(&accepted_tokens[..num_accepted_drafts]);

            // Add all tokens from this step (seed + accepted) to the final output list.
            all_generated_tokens.extend_from_slice(&accepted_tokens);

            // The last accepted token (the correction) becomes the seed for the next iteration.
            let next_seed_token = *accepted_tokens.last().unwrap();
            self.fill_token(next_seed_token);

            // Clean up allocated resources for this step.
            self.queue.deallocate_embeds(&input_embeds);
            self.queue.deallocate_embeds(&output_embeds);

            // Check if the stop condition has been met after the step is complete.
            if stop_condition.should_stop(&all_generated_tokens) {
                break;
            }
        }

        self.tokenizer.detokenize(&all_generated_tokens)
    }
}
