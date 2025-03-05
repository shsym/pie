use std::{mem, slice};

pub mod bindings {

    wit_bindgen::generate!({
        path: "../../api/wit",
        world: "app",
        pub_export_macro: true,
        export_macro_name: "export",
        generate_all,
    });
}

pub use crate::bindings::{
    export, exports::spi::app::run::Guest as RunSync, spi::app::l4m, spi::app::l4m_vision,
    spi::app::ping, spi::app::system,
};

pub struct Context {
    stream: u32,
    occupied_block_ids: Vec<u32>,
    free_block_ids: Vec<u32>,
    leftover_token_ids: Vec<u32>,
}

impl Context {
    pub fn new(stream: u32) -> Self {
        Self {
            stream,
            occupied_block_ids: Vec::new(),
            free_block_ids: Vec::new(),
            leftover_token_ids: Vec::new(),
        }
    }

    pub fn with_capacity(stream: u32, num_tokens: u32) -> Self {
        // allocate block ids
        let num_needed_blocks = num_tokens.div_ceil(l4m::get_block_size());
        let free_block_ids = l4m::allocate_blocks(stream, num_needed_blocks);

        Self {
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

        // fill the blocks
        for i in 0..required_blocks {
            let offset = i * block_size;
            self.occupied_block_ids
                .push(self.free_block_ids.pop().unwrap());

            l4m::fill_block(
                self.stream,
                *self.occupied_block_ids.last().unwrap(),
                &self.occupied_block_ids,
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

        for _ in 0..max_output_tokens {
            l4m::fill_block(
                self.stream,
                working_block_id,
                &self.occupied_block_ids, // the context should be inclusive of the current block
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
}

impl Drop for Context {
    fn drop(&mut self) {
        self.clear();
    }
}

#[trait_variant::make(LocalRun: Send)]
pub trait Run {
    async fn run() -> Result<(), String>;
}

pub struct App<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> RunSync for App<T>
where
    T: Run,
{
    fn run() -> Result<(), String> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        // Run the async main function inside the runtime
        let result = runtime.block_on(T::run());

        if let Err(e) = result {
            return Err(format!("{:?}", e));
        }

        Ok(())
    }
}

#[macro_export]
macro_rules! main_sync {
    ($app:ident) => {
        symphony::export!($app with_types_in symphony::bindings);
    };
}

#[macro_export]
macro_rules! main {
    ($app:ident) => {
        type _App = symphony::App<$app>;
        symphony::export!(_App with_types_in symphony::bindings);
    };
}
