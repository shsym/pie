"""TODO: Add module docstring."""
from __future__ import annotations

import time

import numpy as np
import torch

import message
from adapter import CmaesAdapter, AdapterSubpass
import flashinfer as ops

from config.common import ModelInfo


class Handler:
    """TODO: Add class docstring."""

    max_num_kv_pages: int
    max_num_embeds: int
    max_num_adapters: int
    max_adapter_rank: int

    def __init__(
            self,
            model,
            model_info: ModelInfo,
            kv_page_size: int,
            dist_size: int,
            max_num_kv_pages: int,
            max_num_embeds: int,
            max_num_adapters: int,
            max_adapter_rank: int,
            dtype: torch.dtype,
            device: str,
    ):
        """TODO: Add method docstring."""
        self.adapters = {}

        self.lm = model
        self.model_info = model_info
        self.kv_page_size = kv_page_size
        self.dist_size = dist_size
        self.max_num_kv_pages = max_num_kv_pages
        self.max_num_embeds = max_num_embeds
        self.max_num_adapters = max_num_adapters
        self.max_adapter_rank = max_adapter_rank
        self.dtype = dtype
        self.device = device

        self.kv_cache_at_layer = [
            torch.zeros(
                (
                    max_num_kv_pages,
                    2,
                    kv_page_size,
                    self.lm.config.num_key_value_heads,
                    self.lm.config.head_size,
                ),
                dtype=dtype,
                device=device,
            )
            for _ in range(self.lm.config.num_layers)
        ]

        self.adapter_at_layer = [
            (
                torch.zeros(
                    (
                        max_num_adapters,
                        max_adapter_rank * 3,
                        self.lm.config.hidden_size,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros(
                    (
                        max_num_adapters,
                        self.lm.config.head_size * (
                                self.lm.config.num_query_heads + self.lm.config.num_key_value_heads * 2),
                        max_adapter_rank,
                    ),
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in range(self.lm.config.num_layers)
        ]

        self.embeds = torch.empty(
            (max_num_embeds, self.lm.config.hidden_size), device=device, dtype=dtype
        )

        self.inter_fill_time = time.time()

    def handshake(self, reqs: list[message.HandshakeRequest]) -> list[message.HandshakeResponse]:
        resps = []
        for req in reqs:
            # req.version

            resp = message.HandshakeResponse(
                version=self.model_info.version,
                model_name=self.model_info.name,
                model_traits=["todo"],
                model_description=self.model_info.description,
                prompt_template=self.model_info.template_content,
                prompt_template_type=self.model_info.template_type,
                kv_page_size=self.kv_page_size,
                resources={
                    0: self.max_num_kv_pages,
                    1: self.max_num_embeds,
                    2: self.max_num_adapters,
                },
                tokenizer_merge_table=self.model_info.tokenizer.merge_table,
                tokenizer_special_tokens=self.model_info.tokenizer.special_tokens,
                tokenizer_split_regex=self.model_info.tokenizer.split_regex,
                tokenizer_escape_non_printable=self.model_info.tokenizer.escape_non_printable,
            )
            resps.append(resp)
        return resps

    def query(self, reqs: list[message.QueryRequest]) -> list[message.QueryResponse]:
        resps = []
        for req in reqs:
            value = "unknown query"
            match req.query:
                case "ping":
                    value = "pong"
            resp = message.QueryResponse(value=value)
            resps.append(resp)
        return resp

    def embed_image(self, reqs: list[message.EmbedImageRequest]):
        """
        Embeds images into the specified embed pointers.
        """
        for req in reqs:
            if len(req.embed_ptrs) > self.max_num_embeds:
                raise ValueError(f"Number of embed pointers {len(req.embed_ptrs)} exceeds maximum {self.max_num_embeds}.")

            image_tensor = ops.image.decode_image(req.image_blob, dtype=self.dtype, device=self.device)
            image_embed = self.lm.model.embed_image(image_tensor)

            for i, ptr in enumerate(req.embed_ptrs):
                if ptr < 0 or ptr >= self.max_num_embeds:
                    raise ValueError(f"Embed pointer {ptr} out of range [0, {self.max_num_embeds}).")
                self.embeds[ptr].copy_(image_embed[i], non_blocking=True)

    @torch.inference_mode()
    def initialize_adapter(self, reqs: list[message.InitializeAdapterRequest]):

        cfg = self.lm.config

        for req in reqs:
            self.adapters[req.adapter_ptr] = CmaesAdapter(
                rank=req.rank,
                alpha=req.alpha,
                in_features=cfg.hidden_size,
                out_features=[cfg.head_size * cfg.num_query_heads,
                              cfg.head_size * cfg.num_key_value_heads,
                              cfg.head_size * cfg.num_key_value_heads],
                num_layers=cfg.num_layers,
                population_size=req.population_size,
                mu_fraction=req.mu_fraction,
                initial_sigma=req.initial_sigma,
                min_sigma=1e-7,
                min_var=1e-8,
                max_var=1e4,
                device=self.device,
                dtype=self.dtype,
            )

    @torch.inference_mode()
    def update_adapter(self, reqs: list[message.UpdateAdapterRequest]):

        for req in reqs:
            if req.adapter_ptr in self.adapters:
                adapter = self.adapters[req.adapter_ptr]
                if isinstance(adapter, CmaesAdapter):
                    adapter.update(req.scores, req.seeds, req.max_sigma)

    @torch.inference_mode()
    def forward_pass(self, reqs: list[message.ForwardPassRequest]):
        """
        Processes a batch of forward pass requests through the language model.
        """
        # Sort requests by adapter to optimize the adapter subpass.
        reqs = sorted(reqs, key=lambda o: (o.adapter is None, o.adapter))

        # 1. Consolidate and process all requests into a single batch.
        batch = ForwardPassBatch(self)
        for req in reqs:
            batch.add_request(req)

        # 2. Finalize the batch to get model inputs as tensors.
        model_inputs = batch.finalize()

        # 3. Run the forward pass through the model.
        with torch.cuda.device(self.device):
            output_embeds = self.lm.model.forward(**model_inputs)

            # 4. Package the model outputs into response messages.
            responses = batch.package_responses(output_embeds)

        return responses


def _decode_brle(brle_buffer: list[int]) -> np.ndarray:
    """
    Decodes a Binary Run-Length Encoded buffer into a boolean numpy array.
    The format assumes alternating runs of False and True, starting with False.
    """
    if not brle_buffer:
        return np.array([], dtype=bool)

    total_size = sum(brle_buffer)
    if total_size == 0:
        return np.array([], dtype=bool)

    decoded_array = np.empty(total_size, dtype=bool)
    current_pos = 0
    value = True  # In attention masking, True means attend.
    for run_len in brle_buffer:
        if run_len > 0:
            decoded_array[current_pos: current_pos + run_len] = value
        current_pos += run_len
        value = not value  # Flip value for the next run
    return decoded_array


class ForwardPassBatch:
    """Consolidates and processes a batch of forward pass requests."""

    def __init__(self, handler: Handler):
        """Initializes the batch processor."""
        self._handler = handler
        self._original_reqs: list[message.ForwardPassRequest] = []

        # Inputs for the model
        self.adapter_indices: list[int] = []
        self.seeds: list[int] = []
        self.kv_page_indices: list[int] = []
        self.kv_page_indptr: list[int] = [0]
        self.kv_last_page_lengths: list[int] = []
        self.qo_indptr: list[int] = [0]
        self.attention_masks: list[np.ndarray] = []
        self.batch_token_ids: list[int] = []
        self.batch_position_ids: list[int] = []

        # Tracking state
        self.total_tokens_in_batch: int = 0
        self.single_token_inference_mode: bool = True
        self.adapter_subpass_needed: bool = False

        # Output mapping
        self.indices_for_logits: list[int] = []
        self.subindices_for_dists: list[int] = []
        self.subindices_for_sampling: list[int] = []
        self.indices_for_embed_storage: list[int] = []
        self.embed_storage_pointers: list[int] = []

        # Attributes for multiple samplers
        self.sampler_type: list[int] = []
        self.sampler_top_k: list[int] = []
        self.sampler_top_p: list[float] = []
        self.sampler_min_p: list[float] = []
        self.sampler_temperature: list[float] = []

    def add_request(self, req: message.ForwardPassRequest):
        """Processes and adds a single request to the batch."""
        self._original_reqs.append(req)

        # Handle adapter information
        if req.adapter is not None and req.adapter in self._handler.adapters:
            seed = req.adapter_seed if req.adapter_seed is not None else 0
            self.seeds.extend([seed] * len(req.input_tokens))
            self.adapter_indices.append(req.adapter)
            self.adapter_subpass_needed = True

        # Handle KV cache pages
        self.kv_page_indices.extend(req.kv_page_ptrs)
        self.kv_page_indptr.append(len(self.kv_page_indices))
        self.kv_last_page_lengths.append(req.kv_page_last_len)

        # Handle output mappings for embeddings that need to be stored
        for token_idx, storage_ptr in zip(req.output_embed_indices, req.output_embed_ptrs):
            self.indices_for_embed_storage.append(token_idx + self.total_tokens_in_batch)
            self.embed_storage_pointers.append(storage_ptr)

        # Handle output mappings for tokens requiring logits (for distributions or sampling)
        output_indices_set = set(req.output_dist_indices) | set(req.output_token_indices)
        sorted_output_indices = sorted(list(output_indices_set))

        logit_tensor_offset = len(self.indices_for_logits)
        for token_idx in req.output_dist_indices:
            self.subindices_for_dists.append(sorted_output_indices.index(token_idx) + logit_tensor_offset)

        # Iterate through token and sampler indices, storing sampler type and parameters.
        for token_idx, sampler_idx in zip(req.output_token_indices, req.output_token_samplers):
            self.subindices_for_sampling.append(sorted_output_indices.index(token_idx) + logit_tensor_offset)
            self.sampler_type.append(sampler_idx)
            self.sampler_top_k.append(req.sampler_top_k)
            self.sampler_top_p.append(req.sampler_top_p)
            self.sampler_min_p.append(req.sampler_min_p)
            self.sampler_temperature.append(req.sampler_temperature)

        for token_idx in sorted_output_indices:
            self.indices_for_logits.append(token_idx + self.total_tokens_in_batch)

        # Handle input tokens and positions
        self.batch_token_ids.extend(req.input_tokens)
        self.batch_position_ids.extend(req.input_token_positions)
        self.total_tokens_in_batch += len(req.input_tokens)
        self.qo_indptr.append(self.total_tokens_in_batch)

        if len(req.input_tokens) > 1:
            self.single_token_inference_mode = False

        attention_mask = self._generate_mask_for_request(req)
        self.attention_masks.append(attention_mask)

    def _generate_mask_for_request(self, req: message.ForwardPassRequest) -> np.ndarray:
        """Generates the custom attention mask for a single request."""
        if len(req.mask) != len(req.input_tokens):
            raise ValueError(
                f"Mismatch between number of masks ({len(req.mask)}) and "
                f"input tokens ({len(req.input_tokens)})."
            )

        sequence_length = self._handler.kv_page_size * (len(req.kv_page_ptrs) - 1) + req.kv_page_last_len
        context_length = sequence_length - len(req.input_tokens)

        request_attention_mask = np.zeros((len(req.input_tokens), sequence_length), dtype=np.bool_)
        for i, brle_buffer in enumerate(req.mask):
            decoded_mask = _decode_brle(brle_buffer)
            expected_len = context_length + i + 1
            if len(decoded_mask) != expected_len:
                raise ValueError(
                    f"Decoded mask for token {i} has length {len(decoded_mask)}, but expected {expected_len}"
                )
            request_attention_mask[i, :expected_len] = decoded_mask

        return request_attention_mask.flatten()

    def finalize(self) -> dict:
        """Finalizes batch preparation, creating tensors and the adapter subpass."""
        device = self._handler.device

        adapter_subpass = None
        if self.adapter_subpass_needed:
            seeds_tensor = torch.as_tensor(self.seeds, device=device, dtype=torch.long)
            adapter_subpass = AdapterSubpass(
                adapter_at_layer=self._handler.adapter_at_layer,
                adapter_indices=self.adapter_indices,
                adapter_extras=self._handler.adapters,
                rand_seeds=seeds_tensor,
                qo_indptr=self.qo_indptr,
            )

        batched_attention_mask = np.concatenate(self.attention_masks) if self.attention_masks else np.array([], dtype=np.bool_)
        token_ids_tensor = torch.as_tensor(self.batch_token_ids, device=device, dtype=torch.int32)
        input_embeds = self._handler.lm.model.embed_tokens(token_ids_tensor)

        return {
            "input_embeds": input_embeds,
            "position_ids": torch.as_tensor(self.batch_position_ids, device=device, dtype=torch.int32),
            "qo_indptr": torch.as_tensor(self.qo_indptr, device=device, dtype=torch.int32),
            "kv_cache_at_layer": self._handler.kv_cache_at_layer,
            "kv_page_indices": torch.as_tensor(self.kv_page_indices, device=device, dtype=torch.int32),
            "kv_page_indptr": torch.as_tensor(self.kv_page_indptr, device=device, dtype=torch.int32),
            "kv_last_page_lens": torch.as_tensor(self.kv_last_page_lengths, device=device, dtype=torch.int32),
            "custom_mask": torch.as_tensor(batched_attention_mask, device=device, dtype=torch.bool),
            "single_token_inference_mode": self.single_token_inference_mode,
            "adapter_subpass": adapter_subpass,
        }

    def package_responses(self, output_embeds: torch.Tensor) -> list[message.ForwardPassResponse]:
        """Packages the model outputs into responses for each original request."""
        # Handle storing specified embeddings
        if self.indices_for_embed_storage:
            embeddings_to_store = output_embeds[self.indices_for_embed_storage]
            for i, ptr in enumerate(self.embed_storage_pointers):
                self._handler.embeds[ptr].copy_(embeddings_to_store[i], non_blocking=True)

        if not self.indices_for_logits:
            return [message.ForwardPassResponse(dists=[], tokens=[]) for _ in self._original_reqs]

        # Calculate logits only for the required tokens
        logits = self._handler.lm.lm_head(output_embeds[self.indices_for_logits])

        # Get top-k distributions from unscaled logits
        topk_token_ids, topk_probabilities = [], []
        if self.subindices_for_dists:
            probs_for_dists = torch.softmax(logits[self.subindices_for_dists], dim=-1)
            topk_results = torch.topk(probs_for_dists, k=self._handler.dist_size, sorted=True)
            topk_token_ids = topk_results.indices.tolist()
            topk_probabilities = topk_results.values.tolist()

        # Sample tokens from temperature-scaled logits
        sampled_token_ids = []
        if self.subindices_for_sampling:
            logits_for_sampling = logits[self.subindices_for_sampling]

            # Apply temperature scaling
            temperatures = torch.tensor(
                self.sampler_temperature,
                device=self._handler.device,
                dtype=self._handler.dtype
            ).unsqueeze(1)
            # Clamp temperature to avoid division by zero
            scaled_logits = logits_for_sampling / torch.clamp(temperatures, min=1e-6)

            probs_for_sampling = torch.softmax(scaled_logits, dim=-1)

            num_samples = len(self.subindices_for_sampling)
            sampled_tokens_tensor = torch.empty(num_samples, dtype=torch.long, device=self._handler.device)

            # Group requests by sampler type for efficient batch processing
            sampler_groups = {}
            for i, sampler_idx in enumerate(self.sampler_type):
                if sampler_idx not in sampler_groups:
                    sampler_groups[sampler_idx] = []
                sampler_groups[sampler_idx].append(i)

            # Process each group of samplers
            for sampler_idx, indices in sampler_groups.items():
                if not indices:
                    continue

                indices_tensor = torch.tensor(indices, device=self._handler.device, dtype=torch.long)
                group_probs = probs_for_sampling.index_select(0, indices_tensor)
                sampled = None

                if sampler_idx == 0:  # sampling_from_probs
                    sampled = ops.sampling.sampling_from_probs(group_probs)
                elif sampler_idx == 1:  # top_p_sampling_from_probs
                    top_p_vals = torch.tensor([self.sampler_top_p[i] for i in indices], device=self._handler.device, dtype=self._handler.dtype)
                    sampled = ops.sampling.top_p_sampling_from_probs(group_probs, top_p=top_p_vals)
                elif sampler_idx == 2:  # top_k_sampling_from_probs
                    top_k_vals = torch.tensor([self.sampler_top_k[i] for i in indices], device=self._handler.device, dtype=torch.long)
                    sampled = ops.sampling.top_k_sampling_from_probs(group_probs, top_k=top_k_vals)
                elif sampler_idx == 3:  # min_p_sampling_from_probs
                    min_p_vals = torch.tensor([self.sampler_min_p[i] for i in indices], device=self._handler.device, dtype=self._handler.dtype)
                    sampled = ops.sampling.min_p_sampling_from_probs(group_probs, min_p=min_p_vals)
                elif sampler_idx == 4:  # top_k_top_p_sampling_from_probs
                    top_k_vals = torch.tensor([self.sampler_top_k[i] for i in indices], device=self._handler.device, dtype=torch.long)
                    top_p_vals = torch.tensor([self.sampler_top_p[i] for i in indices], device=self._handler.device, dtype=self._handler.dtype)
                    sampled = ops.sampling.top_k_top_p_sampling_from_probs(group_probs, top_k=top_k_vals, top_p=top_p_vals)
                else:
                    raise ValueError(f"Unknown sampler index: {sampler_idx}")

                # Place the sampled tokens back into the results tensor in the correct order
                sampled_tokens_tensor.scatter_(0, indices_tensor, sampled)

            sampled_token_ids = sampled_tokens_tensor.tolist()

        # Distribute batched results back to individual responses
        responses = []
        dist_cursor, token_cursor = 0, 0
        for req in self._original_reqs:
            num_dists = len(req.output_dist_indices)
            ids_slice = topk_token_ids[dist_cursor: dist_cursor + num_dists]
            probs_slice = topk_probabilities[dist_cursor: dist_cursor + num_dists]
            request_dists = list(zip(ids_slice, probs_slice))
            dist_cursor += num_dists

            num_tokens = len(req.output_token_indices)
            request_tokens = sampled_token_ids[token_cursor: token_cursor + num_tokens]
            token_cursor += num_tokens

            responses.append(message.ForwardPassResponse(dists=request_dists, tokens=request_tokens))

        return responses
