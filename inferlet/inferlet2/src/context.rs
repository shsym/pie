use crate::drafter::Drafter;
use crate::sampler::Sampler;
use crate::stop_condition::StopCondition;
use crate::traits::allocate::Allocate;
use crate::traits::tokenize::{Tokenize, Tokenizer};
use crate::{Model, Queue, sampler, stop_condition};

use crate::traits::forward::Forward;
use crate::traits::input_text::InputText;
use crate::traits::output_text::{Distribution, OutputText};
use std::{fmt, mem};
use wit_bindgen::rt::async_support::futures;
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
        self.queue.deallocate_kv_pages(&taken);
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
        let total_tokens = self.token_ids.len() + pending_token_ids.len();
        let required_pages = total_tokens.div_ceil(self.kv_page_size);
        if required_pages > self.kv_page_ids.len() {
            let new_pages_needed = required_pages - self.kv_page_ids.len();
            let new_kv_page_ids = self.queue.allocate_kv_pages(new_pages_needed);
            self.kv_page_ids.extend(new_kv_page_ids);
        }

        // Then, calculate the length of the new last page.
        let last_page_len = total_tokens % self.kv_page_size;
        self.kv_page_last_len = if last_page_len == 0 && total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };

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

        if self.kv_page_last_len == self.kv_page_size || self.kv_page_ids.is_empty() {
            let new_kv_page_ids = self.queue.allocate_kv_pages(1);
            self.kv_page_ids.extend(new_kv_page_ids);
            self.kv_page_last_len = 1;
        } else {
            self.kv_page_last_len += 1;
        }

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

    pub async fn generate_with_drafter<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        if self.token_ids_pending.len() > 1 {
            self.flush();
        }

        assert_eq!(self.token_ids_pending.len(), 1, "Must have one seed token");

        let mut generated_token_ids = Vec::new();
        let mut next_token_ids = vec![self.token_ids_pending[0]];

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

            let input_embed_id = self.queue.allocate_embeds(combined_token_ids.len());
            let output_embed_id = self.queue.allocate_embeds(combined_token_ids.len());

            self.queue
                .embed_text(&input_embed_id, &combined_token_ids, &combined_pos_ids);

            let available_space = self.kv_page_size - self.kv_page_last_len;
            if combined_token_ids.len() > available_space {
                let needed_page_count =
                    (combined_token_ids.len() - available_space).div_ceil(self.kv_page_size);

                let new_kv_page_ids = self.queue.allocate_kv_pages(needed_page_count);
                self.kv_page_ids.extend(new_kv_page_ids);
                self.kv_page_last_len = (self.token_ids.len() + combined_token_ids.len())
                    - (self.kv_page_ids.len() - 1) * self.kv_page_size;
            } else {
                self.kv_page_last_len += combined_token_ids.len();
            }

            self.queue.forward(
                self.kv_page_last_len as u32,
                &self.kv_page_ids,
                &input_embed_id,
                &output_embed_id,
            );

            let sampled = self
                .queue
                .get_next_token_distribution(&output_embed_id[next_token_ids.len() - 1..])
                .await;

            let mut verified_tokens = Vec::new();
            let mut correct_spec_tokens = 0;
            for i in 0..sampled.len() {
                let predicted = &sampled[i];
                let sampled_token = sampler.sample(&predicted.ids, &predicted.probs);

                if i < spec_token_ids.len() && sampled_token == spec_token_ids[i] {
                    verified_tokens.push(sampled_token);
                    correct_spec_tokens += 1;
                } else {
                    verified_tokens.push(sampled_token);
                    break;
                }
            }

            let redundant_token_count = spec_token_ids.len() - correct_spec_tokens;
            if self.kv_page_last_len > redundant_token_count {
                self.kv_page_last_len -= redundant_token_count;
            } else {
                let mut remaining_redundancy = redundant_token_count - self.kv_page_last_len;
                let redundant = self.kv_page_ids.pop().unwrap();
                self.queue.deallocate_kv_pages(&[redundant]);

                while remaining_redundancy > self.kv_page_size {
                    let redundant = self.kv_page_ids.pop().unwrap();

                    self.queue.deallocate_kv_pages(&[redundant]);
                    remaining_redundancy -= self.kv_page_size;
                }
                self.kv_page_last_len = self.kv_page_size - remaining_redundancy;
            }

            next_token_ids = verified_tokens;
            generated_token_ids.extend(&next_token_ids);

            if stop_condition.should_stop(&generated_token_ids) {
                break;
            } else {
                self.token_ids.extend(&next_token_ids);
            }

            self.queue.deallocate_embeds(&input_embed_id);
            self.queue.deallocate_embeds(&output_embed_id);
        }

        self.token_ids_pending.clear();
        self.token_ids_pending.extend(&next_token_ids);

        self.tokenizer.detokenize(&generated_token_ids)
    }
}
