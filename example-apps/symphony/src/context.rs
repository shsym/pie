use crate::drafter::Drafter;
use crate::l4m::ObjectType;
use crate::sampler::Sampler;
use crate::stop_condition::StopCondition;
use crate::utils::{IdPool, RcIdPool};
use crate::{drafter, l4m, l4m_async, sampler, stop_condition};
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

static STREAM: AtomicU32 = AtomicU32::new(0);

fn get_unique_stream() -> u32 {
    STREAM.fetch_add(1, Ordering::SeqCst)
}

#[derive(Debug)]
struct ResourcePool {
    block_ids: RcIdPool<u32>,
    embed_ids: IdPool<u32>,
    dist_ids: IdPool<u32>,
}

#[derive(Clone, Debug)]
pub struct Model {
    model: Rc<l4m::Model>,
    block_size: usize,
    tokenizer: Rc<l4m::Tokenizer>,
    resources: Rc<RefCell<ResourcePool>>,
}

#[derive(Debug)]
pub struct Context {
    inner: Model,
    stream: u32,
    block_ids: Vec<u32>,
    last_block_len: usize,

    token_ids: Vec<u32>,
    pending_token_ids: Vec<u32>,
    // occupied_block_ids: Vec<u32>,
    // free_block_ids: Vec<u32>,
    // pending_token_ids: Vec<u32>,
    // processed_token_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    tokenizer: Rc<l4m::Tokenizer>,
}

impl Model {
    pub fn available_models() -> Vec<String> {
        l4m::get_all_models()
    }

    pub fn new(model_name: &str) -> Option<Self> {
        if let Some(model) = l4m::get_model(model_name) {
            // Prefetch the tokenizer
            let tokenizer = model.get_tokenizer();
            let resources = ResourcePool {
                block_ids: RcIdPool::new(u32::MAX),
                embed_ids: IdPool::new(u32::MAX),
                dist_ids: IdPool::new(u32::MAX),
            };
            let block_size = model.get_block_size() as usize;
            Some(Self {
                model: Rc::new(model),
                block_size,
                tokenizer: Rc::new(tokenizer),
                resources: Rc::new(RefCell::new(resources)),
            })
        } else {
            None
        }
    }

    pub fn create_context(&self) -> Context {
        Context {
            inner: self.clone(),
            stream: get_unique_stream(),
            block_ids: Vec::new(),
            last_block_len: 0,
            token_ids: Vec::new(),
            pending_token_ids: Vec::new(),
        }
    }

    pub fn get_tokenizer(&self) -> Tokenizer {
        Tokenizer {
            tokenizer: self.tokenizer.clone(),
        }
    }

    pub fn get_block_size(&self) -> usize {
        self.model.get_block_size() as usize
    }
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.tokenize(text)
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        self.tokenizer.detokenize(token_ids)
    }

    pub fn get_vocabs(&self) -> Vec<Vec<u8>> {
        self.tokenizer.get_vocabs()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let taken = mem::take(&mut self.block_ids);
        self.release(l4m::ObjectType::Block, &taken);
    }
}

impl Context {
    pub fn stream(&self) -> u32 {
        self.stream
    }

    fn alloc(&mut self, ty: ObjectType, count: usize) -> Vec<u32> {
        let ids = match ty {
            ObjectType::Block => RefCell::borrow_mut(&self.inner.resources)
                .block_ids
                .acquire_many(count)
                .unwrap(),
            ObjectType::Embed => RefCell::borrow_mut(&self.inner.resources)
                .embed_ids
                .acquire_many(count)
                .unwrap(),
        };

        self.inner.model.allocate(self.stream, ty, &ids);
        ids
    }

    fn release(&mut self, ty: ObjectType, ids: &[u32]) {
        match ty {
            ObjectType::Block => {
                let should_be_freed = RefCell::borrow_mut(&self.inner.resources)
                    .block_ids
                    .release_many(ids)
                    .unwrap();
                self.inner
                    .model
                    .deallocate(self.stream, ty, &should_be_freed);
            }
            ObjectType::Embed => RefCell::borrow_mut(&self.inner.resources)
                .embed_ids
                .release_many(ids)
                .unwrap(),
        }
    }

    fn block_size(&self) -> usize {
        self.inner.block_size
    }

    pub async fn fork(&mut self) -> Self {
        // flush the pending tokens
        if self.pending_token_ids.len() > 0 {
            self.flush().await;
        }

        let forked = if self.last_block_len == self.block_size() {
            // easy case: the last block is full
            Context {
                inner: self.inner.clone(),
                stream: get_unique_stream(),
                block_ids: self.block_ids.clone(),
                last_block_len: self.block_size(),
                token_ids: self.token_ids.clone(),
                pending_token_ids: self.pending_token_ids.clone(),
            }
        } else {
            let kept_block_len = self.block_ids.len() - 1;
            let kept_tokens_len = kept_block_len * self.block_size();

            let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
            let forked_pending_token_ids = [
                &self.token_ids[kept_tokens_len..],
                &self.pending_token_ids[..],
            ]
            .concat();
            let forked_block_ids = self.block_ids[..kept_block_len].to_vec();
            let forked_last_block_len = if forked_block_ids.len() > 0 {
                self.block_size()
            } else {
                0
            };

            Context {
                inner: self.inner.clone(),
                stream: get_unique_stream(),
                block_ids: forked_block_ids,
                last_block_len: forked_last_block_len,
                token_ids: forked_token_ids,
                pending_token_ids: forked_pending_token_ids,
            }
        };

        // increase the refcount
        self.inner
            .resources
            .borrow_mut()
            .block_ids
            .increment_rc_many(&forked.block_ids);

        forked
    }

    pub async fn generate_until(&mut self, stop_str: &str, max_tokens: usize) -> String {
        let mut drafter = drafter::Empty {};
        let mut sampler = sampler::GreedySampler::new();

        let stop_str_token_ids = self.inner.tokenizer.tokenize(stop_str);

        let mut stop_condition = stop_condition::any(
            stop_condition::Until::new(stop_str_token_ids),
            stop_condition::Length::new(max_tokens),
        );

        self.generate_with_drafter(&mut drafter, &mut sampler, &mut stop_condition)
            .await
    }

    pub async fn fill(&mut self, text: &str) {
        let new_token_ids = self.inner.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids).await;
    }

    pub async fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        self.pending_token_ids.extend(new_token_ids);
    }

    pub async fn flush(&mut self) {
        if self.pending_token_ids.len() < 2 {
            return;
        }

        // take all pending tokens except the last one.
        let pending_token_ids = self
            .pending_token_ids
            .drain(..self.pending_token_ids.len() - 1)
            .collect::<Vec<u32>>();

        let position_ids = (self.token_ids.len() as u32
            ..(self.token_ids.len() + pending_token_ids.len()) as u32)
            .collect::<Vec<u32>>();

        let embed_ids = self.alloc(ObjectType::Embed, pending_token_ids.len());

        self.inner
            .model
            .embed_text(self.stream, &embed_ids, &pending_token_ids, &position_ids);

        // ensure we have enough blocks

        let available_space = self.block_size() - self.last_block_len;

        if pending_token_ids.len() > available_space {
            let needed_block_count =
                (pending_token_ids.len() - available_space).div_ceil(self.block_size());
            let new_block_ids = self.alloc(ObjectType::Block, needed_block_count);
            self.block_ids.extend(new_block_ids);
            self.last_block_len = (self.token_ids.len() + pending_token_ids.len())
                - (self.block_ids.len() - 1) * self.block_size();
        } else {
            self.last_block_len += pending_token_ids.len();
        }

        self.inner.model.fill_block(
            self.stream,
            self.last_block_len as u32,
            &self.block_ids,
            &embed_ids,
            &[],
        );

        self.token_ids.extend(pending_token_ids);
        // Free embeds
        self.release(ObjectType::Embed, &embed_ids);
    }

    pub async fn fill_image(&mut self, image_blob: &[u8]) {
        //l4m_vision::embed_image(&self.model, self.stream, &[], image_blob);
    }

    // Simple autoregressive generation
    pub async fn generate<S: Sampler, C: StopCondition>(
        &mut self,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        if self.pending_token_ids.len() > 1 {
            self.flush().await;
        }

        // the seed must not be empty
        assert!(self.pending_token_ids.len() == 1);

        let mut generated_token_ids = Vec::new();

        let input_embed_id = self.alloc(ObjectType::Embed, 1);
        let output_embed_id = self.alloc(ObjectType::Embed, 1);

        let mut next_token_id = self.pending_token_ids[0];
        let mut next_pos_id = self.token_ids.len() as u32;
        loop {
            // embed the next token
            self.inner.model.embed_text(
                self.stream,
                &input_embed_id,
                &[next_token_id],
                &[next_pos_id],
            );

            // extend the block as needed
            if self.last_block_len == self.block_size() || self.block_ids.len() == 0 {
                let new_block_ids = self.alloc(ObjectType::Block, 1);
                self.block_ids.extend(new_block_ids);
                self.last_block_len = 1;
            } else {
                self.last_block_len += 1;
            }

            // fill the block
            self.inner.model.fill_block(
                self.stream,
                self.last_block_len as u32,
                &self.block_ids,
                &input_embed_id,
                &output_embed_id,
            );

            // sample the next token
            let sampled = l4m_async::sample_top_k(
                self.inner.model.clone(),
                self.stream,
                output_embed_id.clone(),
                32,
            )
            .await;

            let (next_token_ids, next_token_logits) = &sampled[0];
            // get the next token id
            next_token_id = sampler.sample(next_token_ids, next_token_logits);
            // update the position id
            next_pos_id += 1;

            generated_token_ids.push(next_token_id);

            // stop condition
            if stop_condition.should_stop(&generated_token_ids) {
                break;
            } else {
                self.token_ids.push(next_token_id);
            }
        }

        self.pending_token_ids.clear();
        self.pending_token_ids.push(next_token_id);

        // free the resources
        self.release(ObjectType::Embed, &input_embed_id);
        self.release(ObjectType::Embed, &output_embed_id);

        // decode the generated tokens
        let result = self.inner.tokenizer.detokenize(&generated_token_ids);

        result
    }

    pub async fn generate_with_drafter<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
        //stream: Option<UnboundedSender<u32>>,
    ) -> String {
        if self.pending_token_ids.len() > 1 {
            self.flush().await;
        }

        // the seed must not be empty
        assert!(self.pending_token_ids.len() == 1);

        let mut generated_token_ids = Vec::new();

        let mut next_token_ids = Vec::new();

        next_token_ids.push(self.pending_token_ids[0]);

        drafter.update(&self.token_ids);

        loop {
            drafter.update(&next_token_ids);

            let (spec_token_ids, spec_pos_ids) = drafter.draft();

            let combined_token_ids =
                [next_token_ids.as_slice(), spec_token_ids.as_slice()].concat();
            //let combined_pos_ids = [next_pos_ids.as_slice(), spec_pos_ids.as_slice()].concat();

            let combined_pos_ids = {
                let offset = self.token_ids.len();
                let mut res: Vec<u32> =
                    (offset as u32..(offset + next_token_ids.len()) as u32).collect::<Vec<u32>>();

                res.extend(spec_pos_ids.iter().map(|&x| x + offset as u32));
                res
            };

            let input_embed_id = self.alloc(ObjectType::Embed, combined_token_ids.len());
            let output_embed_id = self.alloc(ObjectType::Embed, combined_token_ids.len());

            // embed the next token
            self.inner.model.embed_text(
                self.stream,
                &input_embed_id,
                &combined_token_ids,
                &combined_pos_ids,
            );

            // extend the block as needed
            let available_space = self.block_size() - self.last_block_len;

            if combined_token_ids.len() > available_space {
                let needed_block_count =
                    (combined_token_ids.len() - available_space).div_ceil(self.block_size());
                let new_block_ids = self.alloc(ObjectType::Block, needed_block_count);
                self.block_ids.extend(new_block_ids);
                self.last_block_len = (self.token_ids.len() + combined_token_ids.len())
                    - (self.block_ids.len() - 1) * self.block_size();
            } else {
                self.last_block_len += combined_token_ids.len();
            }

            // fill the block
            self.inner.model.fill_block(
                self.stream,
                self.last_block_len as u32,
                &self.block_ids,
                &input_embed_id,
                &output_embed_id,
            );

            let sampled = l4m_async::sample_top_k(
                self.inner.model.clone(),
                self.stream,
                output_embed_id[next_token_ids.len() - 1..].to_vec(),
                32,
            )
            .await;

            //assert_eq!(sampled.len(), max_trie_size + 1);

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

            // exclude the tokens that are already in right place
            let mut num_correct_specs_in_place = 0;
            for (i, token_id) in sampled_next_token_ids.iter().enumerate() {
                if spec_token_ids[i] == *token_id {
                    num_correct_specs_in_place += 1;
                } else {
                    break;
                }
            }

            // shrink the blocks
            let redundant_token_count = spec_token_ids.len() - num_correct_specs_in_place;

            // easy case. just adjust the last block length
            if self.last_block_len > redundant_token_count {
                self.last_block_len -= redundant_token_count;
            } else if self.last_block_len == redundant_token_count {
                let block_id = self.block_ids.pop().unwrap();
                self.last_block_len = self.block_size();
                self.release(ObjectType::Block, &[block_id]);
            }
            // hard case. we need to shrink the blocks
            else {
                let redundant_block_count =
                    (redundant_token_count - self.last_block_len).div_ceil(self.block_size());
                // pop the blocks
                for _ in 0..redundant_block_count {
                    let block_id = self.block_ids.pop().unwrap();
                    self.release(ObjectType::Block, &[block_id]);
                }
                self.last_block_len =
                    (self.token_ids.len() + next_token_ids.len() + num_correct_specs_in_place)
                        - (self.block_ids.len() - 1) * self.block_size();
            }

            next_token_ids = sampled_next_token_ids;

            generated_token_ids.extend(&next_token_ids);

            // stop condition
            if stop_condition.should_stop(&generated_token_ids) {
                break;
            } else {
                self.token_ids.extend(&next_token_ids);
            }

            // free the resources
            self.release(ObjectType::Embed, &input_embed_id);
            self.release(ObjectType::Embed, &output_embed_id);
        }

        self.pending_token_ids.clear();
        self.pending_token_ids.extend(&next_token_ids);

        // decode the generated tokens
        let result = self.inner.tokenizer.detokenize(&generated_token_ids);

        result
    }
}
