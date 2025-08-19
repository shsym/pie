use crate::Queue;
use crate::forward;

/// Defines the interface for executing forward passes in a neural network model,
/// including support for PEFT adapters and KV cache manipulation.
pub trait Forward {
    /// Returns a list of all available PEFT (Parameter-Efficient Fine-Tuning)
    /// adapter names that can be used during a forward pass.
    fn get_all_adapters(&self) -> Vec<String>;

    /// Executes a standard forward pass using the base model.
    ///
    /// # Arguments
    ///
    /// * `last_kv_page_len` - The number of valid entries in the last KV cache page.
    /// * `page_ids` - A list of memory page IDs representing the context's KV cache.
    /// * `input_embed_ids` - A list of object IDs for the input embeddings.
    /// * `output_embed_ids` - A list of object IDs where the output embeddings will be written.
    fn forward(
        &self,
        last_kv_page_len: u32,
        page_ids: &[u32],
        input_embed_ids: &[u32],
        output_embed_ids: &[u32],
    );

    /// Executes a forward pass using a specified PEFT adapter.
    ///
    /// # Arguments
    ///
    /// * `adapter_name` - The name of the PEFT adapter to apply during this forward pass.
    /// * `last_kv_page_len` - The number of valid entries in the last KV cache page.
    /// * `page_ids` - A list of memory page IDs representing the context's KV cache.
    /// * `input_embed_ids` - A list of object IDs for the input embeddings.
    /// * `output_embed_ids` - A list of object IDs where the output embeddings will be written.
    // fn forward_with_adapter(
    //     &self,
    //     adapter_name: String,
    //     last_kv_page_len: u32,
    //     page_ids: &[u32],
    //     input_embed_ids: &[u32],
    //     output_embed_ids: &[u32],
    // );

    /// Applies a boolean mask to a specific KV cache page.
    ///
    /// This is used to selectively ignore certain entries in the cache.
    ///
    /// # Arguments
    ///
    /// * `page_id` - The ID of the target KV cache page to modify.
    /// * `mask` - A boolean mask where `true` keeps an entry and `false` ignores it.
    fn mask_kv_page(&self, page_id: u32, mask: &[bool]);
}

impl Forward for Queue {
    fn get_all_adapters(&self) -> Vec<String> {
        forward::get_all_adapters(&self.inner)
    }

    fn forward(
        &self,
        last_kv_page_len: u32,
        page_ids: &[u32],
        input_embed_ids: &[u32],
        output_embed_ids: &[u32],
    ) {
        forward::forward(
            &self.inner,
            last_kv_page_len,
            page_ids,
            input_embed_ids,
            output_embed_ids,
        )
    }

    // fn forward_with_adapter(
    //     &self,
    //     adapter_name: String,
    //     last_kv_page_len: u32,
    //     page_ids: &[u32],
    //     input_embed_ids: &[u32],
    //     output_embed_ids: &[u32],
    // ) {
    //     forward::forward_with_adapter(
    //         &self.inner,
    //         &adapter_name,
    //         last_kv_page_len,
    //         page_ids,
    //         input_embed_ids,
    //         output_embed_ids,
    //     )
    // }

    fn mask_kv_page(&self, page_id: u32, mask: &[bool]) {
        forward::mask_kv_page(&self.inner, page_id, mask)
    }
}
