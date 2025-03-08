use crate::drafter::Drafter;
use crate::sampler::Sampler;
use crate::stop_condition::StopCondition;
use crate::{drafter, l4m, l4m_async, sampler, stop_condition};
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tokio_stream::Stream;
use tokio_stream::wrappers::UnboundedReceiverStream;

static STREAM: AtomicU32 = AtomicU32::new(0);

fn get_unique_stream() -> u32 {
    STREAM.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone, Debug)]
pub struct Context {
    //pub parent: Option<&'a Context<'a>>,
    inner: Rc<RefCell<Inner>>, // WASM is single-threaded, so we can use Rc<RefCell<Inner>> instead of Arc<Mutex<Inner>>
}

impl Context {
    pub fn stream(&self) -> u32 {
        self.inner.borrow().stream
    }

    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner::with_capacity(0))),
        }
    }

    pub fn with_capacity(num_tokens: u32) -> Self {
        Self {
            inner: Rc::new(RefCell::new(Inner::with_capacity(num_tokens))),
        }
    }

    pub async fn fork(&self) -> Self {
        let inner = self.inner.borrow();

        let (parents, inherited_block_ids) = {
            let mut parents = inner.parents.clone();
            parents.push(self.clone());

            let inherited_block_ids = [
                inner.inherited_block_ids.as_slice(),
                inner.occupied_block_ids.as_slice(),
            ]
            .concat();
            (parents, inherited_block_ids)
        };

        let child_inner = Inner {
            stream: get_unique_stream(),
            parents,
            inherited_block_ids,
            occupied_block_ids: Vec::new(),
            free_block_ids: Vec::new(),
            pending_token_ids: inner.pending_token_ids.clone(),
            processed_token_ids: inner.processed_token_ids.clone(),
        };

        Self {
            inner: Rc::new(RefCell::new(child_inner)),
        }
    }

    pub async fn fill(&mut self, text: &str) {
        self.inner.borrow_mut().fill(text).await;
    }

    pub async fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        self.inner.borrow_mut().fill_tokens(new_token_ids).await;
    }

    pub async fn generate_until(&mut self, stop_str: &str, max_tokens: usize) -> String {
        let mut drafter = drafter::Empty {};
        let mut sampler = sampler::GreedySampler::new();

        let mut stop_condition = stop_condition::any(
            stop_condition::Until::new(&stop_str),
            stop_condition::Length::new(max_tokens),
        );

        self.generate(&mut drafter, &mut sampler, &mut stop_condition, None)
            .await
    }

    pub async fn generate_stream_until(
        &mut self,
        stop_str: &str,
        max_tokens: usize,
    ) -> impl Stream<Item = u32> {
        let drafter = drafter::Empty {};
        let sampler = sampler::GreedySampler::new();

        let stop_condition = stop_condition::any(
            stop_condition::Until::new(&stop_str),
            stop_condition::Length::new(max_tokens),
        );

        self.generate_stream(drafter, sampler, stop_condition).await
    }

    async fn generate_stream<D, S, C>(
        &mut self,
        mut drafter: D,
        mut sampler: S,
        mut stop_condition: C,
    ) -> impl Stream<Item = u32>
    where
        D: Drafter + 'static,
        S: Sampler + 'static,
        C: StopCondition + 'static,
    {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut self_cloned = self.clone();
        tokio::task::spawn_local(async move {
            self_cloned
                .generate(&mut drafter, &mut sampler, &mut stop_condition, Some(tx))
                .await;
        });
        UnboundedReceiverStream::new(rx)
    }

    pub async fn generate<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
        stream: Option<UnboundedSender<u32>>,
    ) -> String {
        self.inner
            .borrow_mut()
            .generate(drafter, sampler, stop_condition, stream)
            .await
    }
}

#[derive(Debug)]
struct Inner {
    pub stream: u32,
    pub parents: Vec<Context>, // just for memory management for inherited blocks. So that they are not deallocated until all children are done.
    pub inherited_block_ids: Vec<u32>,
    pub occupied_block_ids: Vec<u32>,
    pub free_block_ids: Vec<u32>,
    pub pending_token_ids: Vec<u32>,
    pub processed_token_ids: Vec<u32>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.clear();
    }
}

impl Inner {
    pub fn with_capacity(num_tokens: u32) -> Self {
        // allocate block ids
        let stream = get_unique_stream();

        let num_needed_blocks = num_tokens.div_ceil(l4m::get_block_size());
        let free_block_ids = l4m::allocate_blocks(stream, num_needed_blocks);

        Self {
            stream,
            parents: Vec::new(),
            inherited_block_ids: Vec::new(),
            occupied_block_ids: Vec::new(),
            free_block_ids,
            pending_token_ids: Vec::new(),
            processed_token_ids: Vec::new(),
        }
    }

    pub fn grow(&mut self, num_tokens: usize) {
        // allocate block ids
        let num_needed_blocks = (num_tokens as u32).div_ceil(l4m::get_block_size());
        let new_block_ids = l4m::allocate_blocks(self.stream, num_needed_blocks);

        // append new block ids
        self.free_block_ids.extend(new_block_ids);
    }

    pub fn clear(&mut self) {
        // deallocate all blocks
        l4m::deallocate_blocks(self.stream, &self.occupied_block_ids);
        l4m::deallocate_blocks(self.stream, &self.free_block_ids);

        self.occupied_block_ids.clear();
        self.free_block_ids.clear();
        self.pending_token_ids.clear();
    }

    pub async fn fill(&mut self, text: &str) {
        // to satisfy the borrow checker
        let text_str = text.to_string();
        let new_token_ids = l4m_async::tokenize(text_str).await;

        self.fill_tokens(new_token_ids).await;
    }

    pub async fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        let block_size = l4m::get_block_size() as usize;

        // tokenize the text

        let token_ids = {
            self.pending_token_ids.extend(new_token_ids);

            // there should be at least one leftover token for generation.
            if self.pending_token_ids.len() < block_size + 1 {
                return;
            }

            let drain_amount = (self.pending_token_ids.len() / block_size) * block_size;
            self.pending_token_ids
                .drain(..drain_amount)
                .collect::<Vec<u32>>()
        };

        assert_eq!(token_ids.len() % block_size, 0);
        ////////

        let pos_offset =
            (self.inherited_block_ids.len() + self.occupied_block_ids.len()) * block_size;
        let position_ids =
            (pos_offset as u32..(pos_offset + token_ids.len()) as u32).collect::<Vec<u32>>();

        let embed_ids = l4m::allocate_embeds(self.stream, token_ids.len() as u32);
        l4m::embed_text(self.stream, &embed_ids, &token_ids, &position_ids);

        // ensure we have enough blocks
        let required_blocks = token_ids.len() / block_size;
        if required_blocks > self.free_block_ids.len() {
            let num_needed_blocks = required_blocks - self.free_block_ids.len();
            self.grow(num_needed_blocks * block_size);
        }

        // fill the blocks
        for i in 0..required_blocks {
            let offset = i * block_size;
            self.occupied_block_ids
                .push(self.free_block_ids.pop().unwrap());

            let ctx_block_ids = [
                self.inherited_block_ids.as_slice(),
                self.occupied_block_ids.as_slice(),
            ]
            .concat();

            l4m::fill_block(
                self.stream,
                *self.occupied_block_ids.last().unwrap(),
                &ctx_block_ids,
                &embed_ids[offset..offset + block_size],
                &[],
            );
        }

        // Free embeds
        l4m::deallocate_embeds(self.stream, &embed_ids);

        self.processed_token_ids.extend(token_ids);
    }

    pub async fn generate<D: Drafter, S: Sampler, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
        stream: Option<UnboundedSender<u32>>,
    ) -> String {
        //let until_token_ids = l4m::tokenize(until);

        let block_size = l4m::get_block_size() as usize;
        // the seed must not be empty
        assert!(!self.pending_token_ids.is_empty());

        // initialize the working block
        // ensure we have enough blocks
        if self.free_block_ids.is_empty() {
            self.grow(block_size);
        }
        let pos_offset =
            (self.inherited_block_ids.len() + self.occupied_block_ids.len()) * block_size;
        let mut working_block_id = self.free_block_ids.pop().unwrap();
        self.occupied_block_ids.push(working_block_id);

        //println!("block_size: {}", block_size);

        // L4M objects
        let input_block_embeds = l4m::allocate_embeds(self.stream, block_size as u32);
        let output_block_embeds = l4m::allocate_embeds(self.stream, block_size as u32);
        let next_dists = l4m::allocate_dists(self.stream, block_size as u32);

        // Tokens that have been generated in the working block
        let mut processed_token_ids = Vec::new();
        let mut processed_position_ids = Vec::new();

        // Tokens to be processed in the working block
        let mut processing_token_ids = mem::take(&mut self.pending_token_ids);
        let mut processing_position_ids: Vec<u32> =
            (pos_offset as u32..(pos_offset + processing_token_ids.len()) as u32).collect();

        drafter.update(&self.processed_token_ids);

        // put the remaining tokens into the last block
        // l4m::embed_text(
        //     self.stream,
        //     &input_block_embeds[..processing_token_ids.len()],
        //     &processing_token_ids,
        //     &processing_position_ids,
        // );

        let mut generated_token_ids = Vec::new();

        loop {
            drafter.update(&processing_token_ids);

            let max_trie_size =
                block_size - (processed_token_ids.len() + processing_token_ids.len());

            let (spec_token_ids, spec_pos_ids) = drafter.draft(max_trie_size);

            let combined_token_ids =
                [processing_token_ids.as_slice(), spec_token_ids.as_slice()].concat();

            let combined_position_ids = {
                let mut res = processing_position_ids.clone();
                let offset = processing_position_ids.last().unwrap();

                res.extend(spec_pos_ids.iter().map(|&x| x + offset));
                res
            };

            let offset_prev = processed_token_ids.len();
            let offset_last_token = processing_token_ids.len() - 1;
            let valid_len = combined_token_ids.len();

            l4m::embed_text(
                self.stream,
                &input_block_embeds[offset_prev..offset_prev + valid_len],
                &combined_token_ids,
                &combined_position_ids,
            );

            let ctx_block_ids = [
                self.inherited_block_ids.as_slice(),
                self.occupied_block_ids.as_slice(),
            ]
            .concat();

            // "Full" fill_block
            l4m::fill_block(
                self.stream,
                working_block_id,
                &ctx_block_ids,
                &input_block_embeds[..offset_prev + valid_len],
                &output_block_embeds[..offset_prev + valid_len],
            );
            // println!(
            //     "input_block_embeds: {:?}",
            //     &input_block_embeds[..offset_prev + valid_len]
            // );
            // println!("processed_token_ids: {:?}", &processed_token_ids);
            // println!("processing_token_ids: {:?}", &processing_token_ids);

            // let's sample the next token
            l4m::decode_token_dist(
                self.stream,
                &output_block_embeds[offset_prev + offset_last_token..offset_prev + valid_len],
                &next_dists[offset_prev + offset_last_token..offset_prev + valid_len],
            );

            let sampled = l4m_async::sample_top_k(
                self.stream,
                next_dists[offset_prev + offset_last_token..offset_prev + valid_len].to_vec(),
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

            let next_position_id = processing_position_ids.last().unwrap() + 1;

            generated_token_ids.extend(&sampled_next_token_ids);

            // if the stream is not None, send the token id to the stream
            if let Some(stream) = stream.as_ref() {
                for next_token_id in &sampled_next_token_ids {
                    stream.send(*next_token_id).unwrap();
                }
            }
            //println!("all len: {}, {}", processed_token_ids.len(), processing_token_ids.len());
            // if this was the last block,
            if processed_token_ids.len() + processing_token_ids.len() == block_size {
                // get the new working block
                if self.free_block_ids.is_empty() {
                    self.grow(block_size);
                }

                working_block_id = self.free_block_ids.pop().unwrap();
                self.occupied_block_ids.push(working_block_id);
                self.processed_token_ids.extend(&processed_token_ids);
                processed_position_ids.clear();
                processed_token_ids.clear();
            } else {
                processed_token_ids.append(&mut processing_token_ids);
                processed_position_ids.append(&mut processing_position_ids);
            }

            processing_token_ids.clear();
            processing_position_ids.clear();

            processing_token_ids.append(&mut sampled_next_token_ids);
            processing_position_ids.extend(
                next_position_id..=(next_position_id + sampled_next_token_ids.len() as u32),
            );

            // check if
            if stop_condition.should_stop(&generated_token_ids) {
                break;
            }

            //
            // let ctx_block_ids = [
            //     self.inherited_block_ids.as_slice(),
            //     self.occupied_block_ids.as_slice(),
            // ]
            // .concat();
            //
            // l4m::fill_block(
            //     self.stream,
            //     working_block_id,
            //     &ctx_block_ids,
            //     &input_block_embeds[..processing_token_ids.len()],
            //     &output_block_embeds[..processing_token_ids.len()],
            // );
            //
            // // let's sample the next token
            // l4m::decode_token_dist(
            //     self.stream,
            //     slice::from_ref(&output_block_embeds[processing_token_ids.len() - 1]),
            //     slice::from_ref(&next_dist),
            // );
            //
            // let sampled = l4m_async::sample_top_k(self.stream, vec![next_dist], 32).await;
            //
            // let (next_token_ids, next_token_logits) = &sampled[0];
            //
            // let next_token_id = sampler.sample(next_token_ids, &next_token_logits);
            //
            // //let next_token_id = next_token_ids[0];
            // let next_position_id = processing_position_ids.last().unwrap() + 1;
            //
            // generated_token_ids.push(next_token_id);
            //
            // // send the token id to the parent stream
            // if let Some(stream) = stream.as_ref() {
            //     stream.send(next_token_id).await.unwrap();
            // }
            //
            // // if this was the last block,
            // if processing_token_ids.len() == block_size {
            //     // get the new working block
            //     if self.free_block_ids.is_empty() {
            //         self.grow(block_size);
            //     }
            //
            //     working_block_id = self.free_block_ids.pop().unwrap();
            //     self.occupied_block_ids.push(working_block_id);
            //     self.processed_token_ids.extend(&processing_token_ids);
            //
            //     processing_position_ids.clear();
            //     processing_token_ids.clear();
            // }
            //
            // processing_token_ids.push(next_token_id);
            // processing_position_ids.push(next_position_id);
            //
            // // check if
            // if stop_condition.should_stop(&generated_token_ids) {
            //     break;
            // }
            //
            // // embed the next token
            // l4m::embed_text(
            //     self.stream,
            //     slice::from_ref(&input_block_embeds[processing_token_ids.len() - 1]),
            //     &[next_token_id],
            //     slice::from_ref(&processing_position_ids[processing_token_ids.len() - 1]),
            // );
        }

        // free the resources
        l4m::deallocate_embeds(self.stream, &input_block_embeds);
        l4m::deallocate_embeds(self.stream, &output_block_embeds);
        l4m::deallocate_dists(self.stream, &next_dists);

        // pop the last block
        self.free_block_ids
            .push(self.occupied_block_ids.pop().unwrap());

        self.pending_token_ids.clear();
        self.pending_token_ids.append(&mut processing_token_ids);

        // decode the generated tokens
        let result = l4m_async::detokenize(generated_token_ids).await;

        result
    }
}
