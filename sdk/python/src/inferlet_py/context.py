"""
Context class for managing conversation state and text generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from .model import Model, Queue
from .tokenizer import Tokenizer
from .chat import ChatFormatter
from .kv_page import KvPageManager
from .sampler import Sampler
from .brle import Brle

if TYPE_CHECKING:
    from .forward import ForwardPass
    from .drafter import Drafter


@dataclass
class GenerateResult:
    """Result of text generation."""

    text: str
    tokens: list[int]
    finish_reason: str  # "stop", "max_tokens", "eos"


class Context:
    """
    High-level context for managing conversation and text generation.

    Provides a fluent API for building prompts and generating text.

    Example:
        model = get_auto_model()
        with Context(model) as ctx:
            ctx.system("You are a helpful assistant.")
            ctx.user("What is Python?")
            result = ctx.generate(max_tokens=100)
            print(result.text)
    """

    def __init__(self, model: Model) -> None:
        self._model = model
        self._queue = model.create_queue()
        self._tokenizer = model.tokenizer
        self._formatter = ChatFormatter(model)
        self._kv_manager = KvPageManager(self._queue, model.kv_page_size)

        self._messages: list[tuple[str, str]] = []
        self._token_ids: list[int] = []
        self._pending_tokens: list[int] = []

        # Attention mask tracking (mirrors JS/Rust implementation)
        self._token_mask_current = Brle.new(0)
        self._token_mask_pending: list[Brle] = []

        self._position_ids: list[int] = []
        self._begin_of_sequence = True

        # Adapter state for LoRA inference
        self._adapter_ptr: int | None = None
        self._adapter_random_seed: int | None = None

    @classmethod
    def from_imported_state(
        cls,
        model: Model,
        kv_page_ptrs: list[int],
        prefix_tokens: list[int],
        kv_page_last_len: int,
    ) -> "Context":
        """
        Creates a new Context from previously exported and now imported KV pages.
        This is used to restore a context's state from a cache.

        Args:
            model: The model to use
            kv_page_ptrs: Imported KV page pointers
            prefix_tokens: Token IDs that are already in the KV cache
            kv_page_last_len: Length of the last KV page

        Returns:
            A new Context with the imported state
        """
        kv_page_size = model.kv_page_size

        # Validate kv_page_last_len is within valid range
        if kv_page_last_len < 0 or kv_page_last_len > kv_page_size:
            raise ValueError(
                f"kv_page_last_len out of range: expected 0..{kv_page_size}, "
                f"got {kv_page_last_len}"
            )

        # Handle empty state and validate kv_page_ptrs/kv_page_last_len consistency
        if len(kv_page_ptrs) == 0:
            if kv_page_last_len != 0:
                raise ValueError(
                    f"Invalid state: kv_page_ptrs is empty but kv_page_last_len is "
                    f"{kv_page_last_len} (must be 0 when kv_page_ptrs is empty)"
                )
            expected_tokens = 0
        else:
            expected_tokens = (len(kv_page_ptrs) - 1) * kv_page_size + kv_page_last_len

        # Verify the token count matches the KV page state
        if len(prefix_tokens) != expected_tokens:
            raise ValueError(
                f"Token count mismatch: expected {expected_tokens}, "
                f"got {len(prefix_tokens)} (kv_page_ptrs.length={len(kv_page_ptrs)}, "
                f"kv_page_last_len={kv_page_last_len}, kv_page_size={kv_page_size})"
            )

        ctx = cls(model)
        ctx._token_ids = list(prefix_tokens)
        ctx._position_ids = list(range(len(prefix_tokens)))

        # Import pages into manager
        ctx._kv_manager.import_pages_from_state(kv_page_ptrs, kv_page_last_len)

        ctx._token_mask_current = Brle.new(len(prefix_tokens))
        ctx._begin_of_sequence = False

        return ctx

    @property
    def model(self) -> Model:
        """The model being used."""
        return self._model

    @property
    def queue(self) -> Queue:
        """The command queue."""
        return self._queue

    @property
    def tokenizer(self) -> Tokenizer:
        """The tokenizer."""
        return self._tokenizer

    @property
    def token_ids(self) -> list[int]:
        """All committed token IDs."""
        return self._token_ids.copy()

    @property
    def kv_page_ptrs(self) -> list[int]:
        """The KV page pointers currently in use."""
        return self._kv_manager.page_ptrs

    @property
    def kv_page_last_len(self) -> int:
        """The length of the last KV page."""
        return self._kv_manager.last_page_len

    @property
    def text(self) -> str:
        """The text representation of all tokens."""
        return self._tokenizer.decode(self._token_ids)

    # ==============================
    # Adapter methods (LoRA support)
    # ==============================

    def set_adapter(self, adapter_ptr: int) -> None:
        """
        Set the adapter pointer for LoRA inference.

        Args:
            adapter_ptr: Pointer to an allocated adapter resource
        """
        self._adapter_ptr = adapter_ptr

    def remove_adapter(self) -> None:
        """Remove the current adapter."""
        self._adapter_ptr = None

    def set_adapter_random_seed(self, seed: int) -> None:
        """
        Set the random seed for adapter perturbation.

        Args:
            seed: Random seed for deterministic perturbation
        """
        self._adapter_random_seed = seed

    def __enter__(self) -> "Context":
        return self

    def __exit__(self, *args: object) -> None:
        self.clear()

    # ==============================
    # Message building (fluent API)
    # ==============================

    def system(self, content: str) -> "Context":
        """
        Add a system message.

        Args:
            content: The system message content

        Returns:
            self for method chaining
        """
        self._messages.append(("system", content))
        return self

    def user(self, content: str) -> "Context":
        """
        Add a user message.

        Args:
            content: The user message content

        Returns:
            self for method chaining
        """
        self._messages.append(("user", content))
        return self

    def assistant(self, content: str) -> "Context":
        """
        Add an assistant message.

        Args:
            content: The assistant message content

        Returns:
            self for method chaining
        """
        self._messages.append(("assistant", content))
        return self

    def fill(self, text: str) -> "Context":
        """
        Fill with raw text (tokenizes and adds to pending).

        Args:
            text: Text to add

        Returns:
            self for method chaining
        """
        tokens = self._tokenizer.encode(text)
        self.fill_tokens(tokens)
        return self

    def fill_tokens(self, tokens: list[int]) -> "Context":
        """
        Fill with raw token IDs.

        Args:
            tokens: Token IDs to add

        Returns:
            self for method chaining
        """
        n = len(tokens)
        self._pending_tokens.extend(tokens)

        # Build attention masks incrementally (false = can attend)
        for _ in range(n):
            self._token_mask_current.append(False)
            self._token_mask_pending.append(self._token_mask_current.clone())

        self._begin_of_sequence = False
        return self

    def fill_token(self, token_id: int) -> "Context":
        """
        Fill with a single token ID.

        Args:
            token_id: Token ID to add

        Returns:
            self for method chaining
        """
        self._pending_tokens.append(token_id)
        self._token_mask_current.append(False)
        self._token_mask_pending.append(self._token_mask_current.clone())
        self._begin_of_sequence = False
        return self

    def fill_user_only(self, text: str) -> "Context":
        """
        Add a user message without generation prompt.

        Unlike fill_user(), this method does not add the assistant turn prefix.
        Useful when you want to append user content without triggering generation.

        Args:
            text: User message content

        Returns:
            self for method chaining
        """
        self._formatter.user(text)
        formatted = self._formatter.render(
            self._model.prompt_template,
            add_generation_prompt=False,
            begin_of_sequence=self._begin_of_sequence,
        )
        self._formatter.clear()
        self.fill(formatted)
        return self

    # ==============================
    # Token Masking
    # ==============================

    def mask_tokens(self, indices: list[int], mask: bool) -> None:
        """
        Mask specific token indices.

        Args:
            indices: Token indices to mask
            mask: True to mask (cannot attend), False to unmask
        """
        self._token_mask_current.mask(indices, mask)

    def mask_token_range(self, start: int, end: int, mask: bool) -> None:
        """
        Mask a range of tokens.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            mask: True to mask (cannot attend), False to unmask
        """
        self._token_mask_current.mask_range(start, end, mask)

    def mask_token(self, index: int, mask: bool) -> None:
        """
        Mask a single token.

        Args:
            index: Token index
            mask: True to mask (cannot attend), False to unmask
        """
        self._token_mask_current.mask([index], mask)

    # ==============================
    # KV Cache Management
    # ==============================

    def shrink_kv_pages(self, num_tokens: int) -> None:
        """
        Shrink the KV cache by the specified number of tokens.

        Args:
            num_tokens: Number of tokens to remove from the cache
        """
        self._kv_manager.shrink(num_tokens)

    def flush(self) -> None:
        """
        Process all pending tokens to update the model's internal KV cache state.

        This commits pending tokens without sampling new ones. Useful for
        prefilling context before starting generation, or for processing
        input tokens in batches.
        """
        if not self._pending_tokens:
            return

        # Take all pending tokens
        pending_token_ids = self._pending_tokens.copy()
        self._pending_tokens.clear()

        # Take all pending masks
        pending_masks = self._token_mask_pending.copy()
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        # Calculate positions
        last_pos = self._position_ids[-1] + 1 if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        # Grow KV cache
        self._kv_manager.grow(len(pending_token_ids))

        # Create and execute forward pass
        fwd = self._queue.create_forward_pass()
        fwd.input_tokens(pending_token_ids, position_ids)
        fwd.kv_cache(self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        fwd.attention_mask(mask_buffers)

        # Execute forward pass to populate KV cache only (no output request needed)
        # Note: Rust flush() does NOT set adapter - only decode_step() does
        # This matches JS/Rust implementations which don't request token output in flush()
        fwd.execute()

        # Commit tokens and positions
        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

    # ==============================
    # Generation
    # ==============================

    def _prepare_prompt(self) -> None:
        """Flush messages to tokens if needed."""
        if self._messages:
            formatted = self._formatter.format(
                [{"role": role, "content": content} for role, content in self._messages],
                add_generation_prompt=True,
            )
            self._messages.clear()
            tokens = self._tokenizer.encode(formatted)
            self.fill_tokens(tokens)

    def _check_stop(
        self,
        generated: list[int],
        max_tokens: int | None,
        stop_sequences: list[list[int]] | None,
    ) -> str | None:
        """Check if generation should stop. Returns finish reason or None."""
        if max_tokens is not None and len(generated) >= max_tokens:
            return "max_tokens"

        if stop_sequences:
            for seq in stop_sequences:
                if len(seq) > 0 and len(generated) >= len(seq):
                    if generated[-len(seq) :] == seq:
                        return "stop"

        return None

    def generate(
        self,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 0,
        stop: list[str] | None = None,
        stream: bool = False,
    ) -> GenerateResult | Iterator[str]:
        """
        Generate text completion.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (0 = disabled)
            stop: Stop sequences (strings)
            stream: If True, returns an iterator yielding tokens

        Returns:
            GenerateResult if stream=False, Iterator[str] if stream=True
        """
        self._prepare_prompt()

        # Build stop sequences from model EOS + custom stop strings
        stop_sequences: list[list[int]] = []
        for stop_token in self._model.stop_tokens:
            stop_sequences.append(self._tokenizer.encode(stop_token))
        if stop:
            for s in stop:
                stop_sequences.append(self._tokenizer.encode(s))

        sampler = Sampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if stream:
            return self._generate_stream(max_tokens, sampler, stop_sequences)
        else:
            return self._generate_sync(max_tokens, sampler, stop_sequences)

    def _generate_sync(
        self,
        max_tokens: int,
        sampler: Sampler,
        stop_sequences: list[list[int]],
    ) -> GenerateResult:
        """Synchronous generation."""
        generated_tokens: list[int] = []
        finish_reason = "max_tokens"

        for _ in range(max_tokens):
            token = self._decode_step(sampler)
            generated_tokens.append(token)

            reason = self._check_stop(generated_tokens, max_tokens, stop_sequences)
            if reason:
                finish_reason = reason
                break

        text = self._tokenizer.decode(generated_tokens)
        return GenerateResult(text=text, tokens=generated_tokens, finish_reason=finish_reason)

    def _generate_stream(
        self,
        max_tokens: int,
        sampler: Sampler,
        stop_sequences: list[list[int]],
    ) -> Iterator[str]:
        """Streaming generation - yields decoded tokens."""
        generated_tokens: list[int] = []

        for _ in range(max_tokens):
            token = self._decode_step(sampler)
            generated_tokens.append(token)

            # Yield decoded token
            yield self._tokenizer.decode([token])

            if self._check_stop(generated_tokens, max_tokens, stop_sequences):
                break

    def _decode_step(self, sampler: Sampler) -> int:
        """
        Perform a single autoregressive decoding step.
        Returns the sampled token ID.
        """
        # Ensure we have tokens to process
        if not self._pending_tokens:
            raise RuntimeError("No pending tokens for decoding")

        # Take all pending tokens
        pending_token_ids = self._pending_tokens.copy()
        self._pending_tokens.clear()

        # Take all pending masks
        pending_masks = self._token_mask_pending.copy()
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        # Calculate positions
        last_pos = self._position_ids[-1] + 1 if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        # Grow KV cache by number of tokens
        self._kv_manager.grow(len(pending_token_ids))

        # Create and execute forward pass
        fwd = self._queue.create_forward_pass()
        fwd.input_tokens(pending_token_ids, position_ids)
        fwd.kv_cache(self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        fwd.attention_mask(mask_buffers)

        # Set adapter if configured
        if self._adapter_ptr is not None:
            fwd.set_adapter(self._adapter_ptr)
        if self._adapter_random_seed is not None:
            fwd.set_adapter_seed(self._adapter_random_seed)

        # Configure sampling
        output_idx = [len(pending_token_ids) - 1]
        if sampler.temperature == 0:
            fwd.output_tokens(output_idx, 0.0)
        elif sampler.top_k > 0 and sampler.top_p < 1.0:
            fwd.output_tokens_top_k_top_p(output_idx, sampler.temperature, sampler.top_k, sampler.top_p)
        elif sampler.top_k > 0:
            fwd.output_tokens_top_k(output_idx, sampler.temperature, sampler.top_k)
        elif sampler.top_p < 1.0:
            fwd.output_tokens_top_p(output_idx, sampler.temperature, sampler.top_p)
        else:
            fwd.output_tokens(output_idx, sampler.temperature)

        # Execute
        result = fwd.execute()
        if not result.tokens:
            raise RuntimeError("No token generated")

        sampled_token = result.tokens[0]

        # Commit tokens and positions
        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

        # Add the sampled token to pending for next step
        self.fill_token(sampled_token)

        return sampled_token

    def clear(self) -> None:
        """Clear conversation history and release resources."""
        self._messages.clear()
        self._token_ids.clear()
        self._pending_tokens.clear()
        self._token_mask_pending.clear()
        self._token_mask_current = Brle.new(0)
        self._position_ids.clear()
        self._kv_manager.release_all()
        self._begin_of_sequence = True

    def release(self) -> None:
        """
        Release all KV pages held by this context.

        Call this when the context is no longer needed to free resources.
        Safe to call multiple times. Used in beam search for cleanup.
        """
        self._kv_manager.release_all()

    def fork(self) -> "Context":
        """
        Creates a safe, copy-on-write fork of the context.

        Shares only FULL KV pages. Partial pages are dropped and their
        tokens moved to pending buffer for recomputation. This ensures
        state isolation - shared pages are read-only, writable pages
        are unique per context.

        Returns:
            A new Context that shares immutable history with this one.
        """
        forked = Context(self._model)

        # Fork the KV page manager (only keeps full pages)
        forked_kv_manager, dropped_token_count = self._kv_manager.fork()
        forked._kv_manager = forked_kv_manager

        kept_tokens_len = forked._kv_manager.total_tokens

        # Copy committed tokens and positions (up to kept length)
        forked._token_ids = self._token_ids[:kept_tokens_len]
        forked._position_ids = self._position_ids[:kept_tokens_len]

        # Combine dropped tokens with pending tokens
        forked._pending_tokens = (
            self._token_ids[kept_tokens_len:] + self._pending_tokens.copy()
        )

        # Rebuild the mask for pending tokens
        # Start with a mask covering committed tokens
        mask_builder = self._token_mask_current.clone()
        parent_total_mask_len = len(self._token_ids) + len(self._pending_tokens)
        mask_builder.remove_range(kept_tokens_len, parent_total_mask_len)

        # Build masks for each pending token
        pending_count = len(forked._pending_tokens)
        forked._token_mask_pending = []
        for _ in range(pending_count):
            mask_builder.append(False)
            forked._token_mask_pending.append(mask_builder.clone())

        forked._token_mask_current = mask_builder

        forked._begin_of_sequence = self._begin_of_sequence
        forked._messages = []  # Don't copy pending messages

        # Copy adapter state
        forked._adapter_ptr = self._adapter_ptr
        forked._adapter_random_seed = self._adapter_random_seed

        return forked

    def decode_step_dist(self) -> tuple[list[int], list[float]]:
        """
        Perform a single decoding step and return the probability distribution.

        Unlike _decode_step which samples a token, this returns the full
        distribution for use in beam search.

        Returns:
            (token_ids, probabilities) - sorted by probability descending
        """
        if not self._pending_tokens:
            raise RuntimeError("No pending tokens for decoding")

        # Take all pending tokens
        pending_token_ids = self._pending_tokens.copy()
        self._pending_tokens.clear()

        # Take all pending masks
        pending_masks = self._token_mask_pending.copy()
        self._token_mask_pending.clear()
        mask_buffers = [m.buffer for m in pending_masks]

        # Calculate positions
        last_pos = self._position_ids[-1] + 1 if self._position_ids else 0
        position_ids = list(range(last_pos, last_pos + len(pending_token_ids)))

        # Grow KV cache
        self._kv_manager.grow(len(pending_token_ids))

        # Create forward pass
        fwd = self._queue.create_forward_pass()
        fwd.input_tokens(pending_token_ids, position_ids)
        fwd.kv_cache(self._kv_manager.page_ptrs, self._kv_manager.last_page_len)
        fwd.attention_mask(mask_buffers)

        # Set adapter if configured
        if self._adapter_ptr is not None:
            fwd.set_adapter(self._adapter_ptr)
        if self._adapter_random_seed is not None:
            fwd.set_adapter_seed(self._adapter_random_seed)

        # Request distribution output at the last position
        output_idx = [len(pending_token_ids) - 1]
        fwd.output_distributions(output_idx, 1.0, None)

        # Execute
        result = fwd.execute()
        if not result.distributions:
            raise RuntimeError("No distribution returned")

        dist = result.distributions[0]

        # Commit tokens and positions
        self._token_ids.extend(pending_token_ids)
        self._position_ids.extend(position_ids)

        return dist

    def generate_with_beam(
        self,
        *,
        beam_size: int,
        max_tokens: int = 256,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate text using beam search decoding.

        Beam search maintains multiple candidate sequences (beams) at each step,
        selecting the most likely overall sequences rather than greedily choosing
        the best token at each position.

        Args:
            beam_size: Number of candidate sequences to maintain at each step.
            max_tokens: Maximum tokens to generate.
            stop: Optional stop sequences (strings).

        Returns:
            The generated text from the winning beam.
        """
        import math

        self._prepare_prompt()

        # Build stop sequences from model EOS + custom stop strings
        stop_sequences: list[list[int]] = []
        for stop_token in self._model.stop_tokens:
            stop_sequences.append(self._tokenizer.encode(stop_token))
        if stop:
            for s in stop:
                stop_sequences.append(self._tokenizer.encode(s))

        def check_stop(generated: list[int]) -> bool:
            """Check if generation should stop."""
            if len(generated) >= max_tokens:
                return True
            for seq in stop_sequences:
                if len(seq) > 0 and len(generated) >= len(seq):
                    if generated[-len(seq):] == seq:
                        return True
            return False

        # Type alias for beam state: (context, generated_tokens, score)
        BeamState = tuple["Context", list[int], float]

        # Initialize with a single beam (forked from self)
        beams: list[BeamState] = [(self.fork(), [], 0.0)]

        while True:
            # Check if any beam satisfies the stop condition
            # Beams are sorted by score, so first match is best
            for beam_ctx, generated_tokens, score in beams:
                if check_stop(generated_tokens):
                    # Adopt state from winning beam
                    self._kv_manager.adopt(beam_ctx._kv_manager)
                    self._token_ids = beam_ctx._token_ids.copy()
                    self._position_ids = beam_ctx._position_ids.copy()
                    self._pending_tokens = beam_ctx._pending_tokens.copy()

                    # Release all beams
                    for ctx, _, _ in beams:
                        ctx.release()

                    return self._tokenizer.decode(generated_tokens)

            # Get distributions from all beams
            next_dists = []
            for beam_ctx, _, _ in beams:
                dist = beam_ctx.decode_step_dist()
                next_dists.append(dist)

            # Expand beams
            next_beams: list[BeamState] = []
            for i, (beam_ctx, generated, score) in enumerate(beams):
                token_ids, probs = next_dists[i]

                # Expand with top candidates
                expand_count = min(beam_size, len(token_ids))
                for j in range(expand_count):
                    prob = probs[j]
                    # Skip zero-probability tokens
                    if prob <= 0:
                        continue

                    next_beam = beam_ctx.fork()
                    token_id = token_ids[j]
                    next_beam.fill_token(token_id)

                    next_generated = generated + [token_id]
                    next_score = score + math.log(prob)

                    next_beams.append((next_beam, next_generated, next_score))

                # Release the old beam after forking
                beam_ctx.release()

            # Prune: Sort by score (descending) and keep top beam_size
            next_beams.sort(key=lambda x: x[2], reverse=True)

            # Release pruned beams
            for ctx, _, _ in next_beams[beam_size:]:
                ctx.release()

            beams = next_beams[:beam_size]

    def generate_with_drafter(
        self,
        drafter: "Drafter",
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> str:
        """
        Generate text using speculative decoding with a drafter model.

        Speculative decoding accelerates text generation by using a small, fast
        drafter model to propose candidate tokens. These are then verified by
        the main model, allowing multiple tokens to be generated per forward pass
        when the drafter's predictions are correct.

        Args:
            drafter: The drafter that proposes candidate tokens.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling threshold.
            stop: Stop sequences (strings).

        Returns:
            Generated text.
        """
        self._prepare_prompt()

        # Initialize drafter with current context
        drafter.update(self._token_ids + self._pending_tokens)

        # Build stop sequences from model EOS + custom stop strings
        stop_sequences: list[list[int]] = []
        for stop_token in self._model.stop_tokens:
            stop_sequences.append(self._tokenizer.encode(stop_token))
        if stop:
            for s in stop:
                stop_sequences.append(self._tokenizer.encode(s))

        sampler = Sampler(temperature=temperature, top_p=top_p)
        generated_tokens: list[int] = []

        while True:
            # Get draft tokens
            draft_tokens, draft_pos_ids = drafter.draft()

            if not draft_tokens:
                # No drafts - fall back to regular decode
                token = self._decode_step(sampler)
                generated_tokens.append(token)
                drafter.update([token])
            else:
                # Verify drafts with main model
                accepted = self._verify_drafts(draft_tokens, draft_pos_ids, sampler)
                generated_tokens.extend(accepted)
                drafter.update(accepted)

            # Check stop conditions
            if len(generated_tokens) >= max_tokens:
                break
            if self._check_stop(generated_tokens, max_tokens, stop_sequences):
                break

        return self._tokenizer.decode(generated_tokens)

    def _verify_drafts(
        self,
        draft_tokens: list[int],
        draft_pos_ids: list[int],
        sampler: Sampler,
    ) -> list[int]:
        """
        Verify draft tokens with the main model.

        Returns the tokens that were accepted.

        Args:
            draft_tokens: Proposed token IDs in DFS order.
            draft_pos_ids: Depth of each token in the trie.
            sampler: Sampler for fallback token selection.

        Returns:
            List of accepted token IDs.
        """
        accepted: list[int] = []

        for i, draft_token in enumerate(draft_tokens):
            token_ids, probs = self.decode_step_dist()
            predicted_token = token_ids[0]  # Most likely token

            if predicted_token == draft_token:
                accepted.append(predicted_token)
                self.fill_token(predicted_token)
            else:
                # Draft rejected - use model's prediction
                accepted.append(predicted_token)
                self.fill_token(predicted_token)
                break

        return accepted
