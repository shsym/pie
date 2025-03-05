use crate::l4m;
use std::sync::atomic::{AtomicU32, Ordering};
use std::{mem, slice};

static GLOBAL_COUNTER: AtomicU32 = AtomicU32::new(0);

fn increment_counter() -> u32 {
    GLOBAL_COUNTER.fetch_add(1, Ordering::SeqCst)
}
pub struct Context<'a> {
    parent: Option<&'a Context<'a>>,
    stream: u32,
    occupied_block_ids: Vec<u32>,
    free_block_ids: Vec<u32>,
    leftover_token_ids: Vec<u32>,
}

impl<'a> Context<'a> {
    pub fn new() -> Self {
        let stream = increment_counter();

        Self {
            parent: None,
            stream,
            occupied_block_ids: Vec::new(),
            free_block_ids: Vec::new(),
            leftover_token_ids: Vec::new(),
        }
    }

    pub fn with_capacity(num_tokens: u32) -> Self {
        // allocate block ids
        let stream = increment_counter();

        let num_needed_blocks = num_tokens.div_ceil(l4m::get_block_size());
        let free_block_ids = l4m::allocate_blocks(stream, num_needed_blocks);

        Self {
            parent: None,
            stream,
            occupied_block_ids: Vec::new(),
            free_block_ids,
            leftover_token_ids: Vec::new(),
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
        self.leftover_token_ids.clear();
    }

    pub fn fill(&mut self, text: &str) {
        let block_size = l4m::get_block_size() as usize;

        // tokenize the text

        let token_ids = {
            let new_token_ids = l4m::tokenize(&text);
            self.leftover_token_ids.extend(new_token_ids);

            // there should be at least one leftover token for generation.
            if self.leftover_token_ids.len() < block_size + 1 {
                return;
            }

            let drain_amount = (self.leftover_token_ids.len() / block_size) * block_size;
            self.leftover_token_ids
                .drain(..drain_amount)
                .collect::<Vec<u32>>()
        };

        assert_eq!(token_ids.len() % block_size, 0);
        ////////

        let pos_offset = self.occupied_block_ids.len() * block_size;
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

        let parent_occupied_block_ids = self.get_parent_occupied_block_ids();

        // fill the blocks
        for i in 0..required_blocks {
            let offset = i * block_size;
            self.occupied_block_ids
                .push(self.free_block_ids.pop().unwrap());

            let ctx_block_ids = [
                parent_occupied_block_ids.as_slice(),
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
    }

    pub fn generate_until(&mut self, until: &str, max_output_tokens: usize) -> String {
        let until_token_ids = l4m::tokenize(until);

        let block_size = l4m::get_block_size() as usize;
        // the seed must not be empty
        assert!(!self.leftover_token_ids.is_empty());

        // initialize the working block
        // ensure we have enough blocks
        if self.free_block_ids.is_empty() {
            self.grow(block_size);
        }
        let pos_offset = self.occupied_block_ids.len() * block_size;
        let mut working_block_id = self.free_block_ids.pop().unwrap();
        self.occupied_block_ids.push(working_block_id);

        let mut working_token_ids = mem::take(&mut self.leftover_token_ids);
        let mut working_position_ids: Vec<u32> =
            (pos_offset as u32..(pos_offset + working_token_ids.len()) as u32).collect();

        let input_block_embeds = l4m::allocate_embeds(self.stream, block_size as u32);
        let output_block_embeds = l4m::allocate_embeds(self.stream, block_size as u32);
        let next_dist = l4m::allocate_dists(self.stream, 1)[0];

        // put the remaining tokens into the last block
        l4m::embed_text(
            self.stream,
            &input_block_embeds[..working_token_ids.len()],
            &working_token_ids,
            &working_position_ids,
        );

        let mut generated_token_ids = Vec::new();
        let parent_occupied_block_ids = self.get_parent_occupied_block_ids();

        for _ in 0..max_output_tokens {
            let ctx_block_ids = [
                parent_occupied_block_ids.as_slice(),
                self.occupied_block_ids.as_slice(),
            ]
            .concat();

            l4m::fill_block(
                self.stream,
                working_block_id,
                &ctx_block_ids,
                &input_block_embeds[..working_token_ids.len()],
                &output_block_embeds[..working_token_ids.len()],
            );

            // let's sample the next token
            l4m::decode_token_dist(
                self.stream,
                slice::from_ref(&output_block_embeds[working_token_ids.len() - 1]),
                slice::from_ref(&next_dist),
            );

            let sampled = l4m::sample_top_k(self.stream, slice::from_ref(&next_dist), 1);

            let (top_next_token_ids, _) = &sampled[0];
            let next_token_id = top_next_token_ids[0];
            let next_position_id = working_position_ids.last().unwrap() + 1;

            generated_token_ids.push(next_token_id);

            // if this was the last block,
            if working_token_ids.len() == block_size {
                // get the new working block
                if self.free_block_ids.is_empty() {
                    self.grow(block_size);
                }

                working_block_id = self.free_block_ids.pop().unwrap();
                self.occupied_block_ids.push(working_block_id);

                working_position_ids.clear();
                working_token_ids.clear();
            }

            working_token_ids.push(next_token_id);
            working_position_ids.push(next_position_id);

            // check if

            if generated_token_ids.len() >= until_token_ids.len() {
                if generated_token_ids[generated_token_ids.len() - until_token_ids.len()..]
                    == until_token_ids
                {
                    break;
                }
            }

            // embed the next token
            l4m::embed_text(
                self.stream,
                slice::from_ref(&input_block_embeds[working_token_ids.len() - 1]),
                &[next_token_id],
                slice::from_ref(&working_position_ids[working_token_ids.len() - 1]),
            );
        }

        // free the resources
        l4m::deallocate_embeds(self.stream, &input_block_embeds);
        l4m::deallocate_embeds(self.stream, &output_block_embeds);
        l4m::deallocate_dists(self.stream, &[next_dist]);

        // pop the last block
        self.free_block_ids
            .push(self.occupied_block_ids.pop().unwrap());

        self.leftover_token_ids.clear();
        self.leftover_token_ids.append(&mut working_token_ids);

        // decode the generated tokens
        let result = l4m::detokenize(&generated_token_ids);

        result
    }

    pub fn fork(&'a self) -> Self {
        Self {
            parent: Some(&self),
            stream: increment_counter(),
            occupied_block_ids: Vec::new(),
            free_block_ids: Vec::new(),
            leftover_token_ids: self.leftover_token_ids.clone(),
        }
    }

    fn get_parent_occupied_block_ids(&self) -> Vec<u32> {
        let mut occupied_block_ids = Vec::new();
        let mut current = self;
        while let Some(parent) = current.parent {
            occupied_block_ids.extend_from_slice(&current.occupied_block_ids);
            current = parent;
        }
        occupied_block_ids
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        self.clear();
    }
}
