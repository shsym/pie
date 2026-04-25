"""Context, TokenStream, EventStream — the core of the inferlet SDK.

Usage::

    from inferlet import Model, Context, Sampler, Event, runtime

    model = Model.load(runtime.models()[0])
    ctx = Context(model)

    ctx.system("You are helpful.")
    ctx.user("Hello!")

    # One-shot generation
    text = await ctx.generate_text(Sampler.top_p())

    # Streaming with events
    async for event in await ctx.generate(Sampler.top_p(), decode=True):
        match event:
            case Event.Text(text=t):
                print(t, end="")
            case Event.Done():
                break
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable

from wit_world.imports import context as _ctx
from wit_world.imports import inference as _inf
from wit_world.imports import chat as _chat
from wit_world.imports import tool_use as _tool
from wit_world.imports import reasoning as _reasoning

from ._async import await_future
from .model import Model, Tokenizer
from .sampler import Sampler
from .grammar import Constraint, Schema, _StaticMaskConstraint

if TYPE_CHECKING:
    from pydantic import BaseModel

# Optional pydantic v2 integration. Detected once at module load; the typed
# generation method is exposed on Context only if pydantic v2 is available.
try:
    import pydantic as _pydantic
    _PYDANTIC_AVAILABLE = (
        getattr(_pydantic, "VERSION", "0").split(".", 1)[0] == "2"
    )
except ImportError:
    _pydantic = None  # type: ignore[assignment]
    _PYDANTIC_AVAILABLE = False



# =============================================================================
# Event types
# =============================================================================


class Event:
    """Discriminated union of generation events.

    Match with ``match``/``case``::

        match event:
            case Event.Text(text=t):
                print(t, end="")
            case Event.Done(text=full):
                break
    """

    __slots__ = ()

    @dataclass(slots=True)
    class Text:
        """A chunk of generated text."""
        text: str

    @dataclass(slots=True)
    class Thinking:
        """A chunk of reasoning / thinking text."""
        text: str

    @dataclass(slots=True)
    class ThinkingDone:
        """Reasoning complete — ``text`` is the full accumulated reasoning."""
        text: str

    @dataclass(slots=True)
    class ToolCallStart:
        """A tool call has been detected (more tokens expected)."""
        pass

    @dataclass(slots=True)
    class ToolCall:
        """A complete tool call."""
        name: str
        arguments: str

    @dataclass(slots=True)
    class Done:
        """Generation complete — ``text`` is the full accumulated text."""
        text: str


# Type alias for event instances
AnyEvent = (
    Event.Text
    | Event.Thinking
    | Event.ThinkingDone
    | Event.ToolCallStart
    | Event.ToolCall
    | Event.Done
)


# =============================================================================
# Decoder (unified mux of chat + reasoning + tool-use decoders)
# =============================================================================


class Decoder:
    """Unified decoder muxing chat, reasoning, and tool-use WIT decoders.

    Event priority:
    1. Reasoning transitions (start/complete) override chat deltas.
    2. Tool calls override chat deltas.
    3. Chat done is always forwarded.
    4. Otherwise, chat deltas are forwarded as ``Event.Text``.
    """

    __slots__ = ("_chat_dec", "_reasoning_dec", "_tool_dec", "_in_reasoning")

    def __init__(
        self,
        model: Model,
        *,
        reasoning: bool = False,
        tool_use: bool = False,
    ) -> None:
        self._chat_dec = _chat.create_decoder(model._handle)
        self._reasoning_dec = _reasoning.create_decoder(model._handle) if reasoning else None
        self._tool_dec = _tool.create_decoder(model._handle) if tool_use else None
        self._in_reasoning = False

    def feed(self, tokens: list[int]) -> AnyEvent:
        """Feed a batch of token IDs and return a unified event."""
        # Always feed the chat decoder
        chat_event = self._chat_dec.feed(tokens)

        # Check reasoning first (highest priority)
        if self._reasoning_dec is not None:
            reasoning_event = self._reasoning_dec.feed(tokens)
            if isinstance(reasoning_event, _reasoning.Event_Start):
                self._in_reasoning = True
                return Event.Thinking("")
            elif isinstance(reasoning_event, _reasoning.Event_Delta):
                if self._in_reasoning and reasoning_event.value:
                    return Event.Thinking(reasoning_event.value)
            elif isinstance(reasoning_event, _reasoning.Event_Complete):
                self._in_reasoning = False
                return Event.ThinkingDone(reasoning_event.value)

        # Check tool use (second priority)
        if self._tool_dec is not None:
            tool_event = self._tool_dec.feed(tokens)
            if isinstance(tool_event, _tool.Event_Start):
                return Event.ToolCallStart()
            elif isinstance(tool_event, _tool.Event_Call):
                name, args = tool_event.value
                return Event.ToolCall(name=name, arguments=args)

        # Fall through to chat events
        if isinstance(chat_event, _chat.Event_Done):
            return Event.Done(chat_event.value)
        elif isinstance(chat_event, _chat.Event_Delta):
            return Event.Text(chat_event.value)
        elif isinstance(chat_event, _chat.Event_Interrupt):
            return Event.Done("")

        # Should not reach here
        return Event.Text("")

    def reset(self) -> None:
        """Reset all sub-decoders."""
        self._chat_dec.reset()
        self._in_reasoning = False
        if self._reasoning_dec is not None:
            self._reasoning_dec.reset()
        if self._tool_dec is not None:
            self._tool_dec.reset()


# =============================================================================
# Page management helpers
# =============================================================================


async def _reserve_and_run(
    ctx_handle,
    model_handle,
    tokens: list[int],
    page_size: int,
    sampler_variant=None,
    logit_mask: list[int] | None = None,
):
    """Reserve pages, run a forward pass, commit pages.

    Uses ``working_page_token_count`` and ``committed_page_count`` to derive
    sequence positions.  Returns the Output (or None if no sampler was given).
    """
    num_tokens = len(tokens)
    wpt = ctx_handle.working_page_token_count()
    seq_start = ctx_handle.committed_page_count() * page_size + wpt

    # Reserve additional pages
    current_working = ctx_handle.working_page_count()
    total_tokens = wpt + num_tokens
    pages_needed = (total_tokens + page_size - 1) // page_size
    additional = max(0, pages_needed - current_working)
    if additional > 0:
        ctx_handle.reserve_working_pages(additional)

    # Build forward pass
    fwd = _inf.ForwardPass(model_handle)
    fwd.context(ctx_handle)
    fwd.input_tokens(tokens, list(range(seq_start, seq_start + num_tokens)))

    if sampler_variant is not None:
        fwd.sampler([num_tokens - 1], sampler_variant)

    if logit_mask is not None:
        fwd.logit_mask(logit_mask)

    # Execute
    future = fwd.execute()
    output = await await_future(future, "Forward pass failed")

    # Commit pages
    new_wpt = wpt + num_tokens
    pages_to_commit = new_wpt // page_size
    if pages_to_commit > 0:
        ctx_handle.commit_working_pages(pages_to_commit)

    return output


# =============================================================================
# TokenStream
# =============================================================================


class TokenStream:
    """Async iterator of generated token batches.

    Created by ``Context.generate(sampler)``. Handles the full generation
    loop internally: page reservation, forward pass, page commit, cursor
    update.

    Usage::

        async for tokens in await ctx.generate(Sampler.top_p(), max_tokens=256):
            print(tokens)
    """

    __slots__ = (
        "_ctx",
        "_model_handle",
        "_page_size",
        "_stop_tokens",
        "_sampler",
        "_constraint",
        "_constraint_pending",
        "_done",
        "_max_tokens",
        "_tokens_generated",
        "_pending_tokens",
    )

    def __init__(
        self,
        ctx: Context,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        constraint: Constraint | None = None,
        stop_tokens: Iterable[int] | None = None,
        pending_tokens: list[int] | None = None,
    ) -> None:
        self._ctx = ctx
        self._model_handle = ctx._handle.model()
        self._page_size: int = ctx._handle.tokens_per_page()
        if stop_tokens is None:
            self._stop_tokens: frozenset[int] = frozenset(
                _chat.stop_tokens(self._model_handle)
            )
        else:
            self._stop_tokens = frozenset(stop_tokens)
        self._sampler = sampler
        self._constraint = constraint
        self._constraint_pending: list[int] = []
        self._done = False
        self._max_tokens = max_tokens
        self._tokens_generated = 0
        self._pending_tokens: list[int] = pending_tokens or []

    # --- Generation ---

    async def _step(self) -> list[int]:
        """One generation step: forward pass → sample → commit.

        If a constraint is attached, advances it on the previous step's
        accepted tokens and applies the resulting mask to the forward pass.
        """
        # Advance the constraint and compute this step's mask.
        mask: list[int] | None = None
        if self._constraint is not None:
            pending = self._constraint_pending
            self._constraint_pending = []
            m = self._constraint.step(pending)
            mask = m if m else None

        if self._pending_tokens:
            output = await _reserve_and_run(
                self._ctx._handle,
                self._model_handle,
                self._pending_tokens,
                self._page_size,
                sampler_variant=self._sampler._variant,
                logit_mask=mask,
            )
        else:
            # Bootstrap: sample from the last cached KV position (after flush)
            fwd = _inf.ForwardPass(self._model_handle)
            fwd.context(self._ctx._handle)
            fwd.sampler([0], self._sampler._variant)
            if mask is not None:
                fwd.logit_mask(mask)
            future = fwd.execute()
            output = await await_future(future, "Bootstrap forward pass failed")

        # Extract tokens from output
        new_tokens = self._extract_tokens(output)
        if not new_tokens:
            return []

        # Stash accepted tokens for the next step's constraint advance.
        if self._constraint is not None:
            self._constraint_pending.extend(new_tokens)

        # Seed pending_tokens with the last generated token for the next step
        self._pending_tokens = [new_tokens[-1]]

        return new_tokens

    @staticmethod
    def _extract_tokens(output) -> list[int]:
        """Pull token IDs from the ``Output`` variant."""
        if isinstance(output, _inf.Output_Tokens):
            return list(output.value)
        elif isinstance(output, _inf.Output_TokensWithSpeculation):
            accepted, _, _ = output.value
            return list(accepted)
        return []

    # --- Convenience ---

    async def text(self) -> str:
        """Consume the stream, decode, and return full text."""
        tokenizer = Tokenizer(self._model_handle.tokenizer())
        return tokenizer.decode(await self.tokens())

    async def tokens(self) -> list[int]:
        """Consume the stream and return all tokens as a flat list."""
        all_tokens: list[int] = []
        async for batch in self:
            all_tokens.extend(batch)
        return all_tokens

    # --- Iterator protocol ---

    def __aiter__(self) -> AsyncIterator[list[int]]:
        return self

    async def __anext__(self) -> list[int]:
        if self._done:
            raise StopAsyncIteration

        if self._max_tokens is not None and self._tokens_generated >= self._max_tokens:
            raise StopAsyncIteration

        tokens = await self._step()
        if not tokens:
            raise StopAsyncIteration

        # Truncate at stop token
        for i, t in enumerate(tokens):
            if t in self._stop_tokens:
                tokens = tokens[:i]
                self._done = True
                if not tokens:
                    raise StopAsyncIteration
                break

        # Enforce max tokens
        if self._max_tokens is not None:
            remaining = self._max_tokens - self._tokens_generated
            if len(tokens) > remaining:
                tokens = tokens[:remaining]
                self._done = True

        self._tokens_generated += len(tokens)
        return tokens


# =============================================================================
# EventStream
# =============================================================================


class EventStream:
    """Async decoded event stream wrapping a ``TokenStream`` + ``Decoder``.

    Created by ``Context.generate(sampler, decode=True)``.

    Usage::

        async for event in await ctx.generate(Sampler.top_p(), decode=True, reasoning=True):
            match event:
                case Event.Text(text=t):
                    print(t, end="")
    """

    __slots__ = ("_stream", "_decoder")

    def __init__(
        self,
        stream: TokenStream,
        *,
        reasoning: bool = False,
        tool_use: bool = False,
    ) -> None:
        self._stream = stream
        model = Model(stream._model_handle)
        self._decoder = Decoder(model, reasoning=reasoning, tool_use=tool_use)

    def __aiter__(self) -> AsyncIterator[AnyEvent]:
        return self

    async def __anext__(self) -> AnyEvent:
        tokens = await self._stream.__anext__()
        return self._decoder.feed(tokens)


# =============================================================================
# Context
# =============================================================================


class Context:
    """Host-managed conversation context.

    Wraps ``pie:core/context`` and ``pie:instruct/*``.

    Usage::

        ctx = Context(model)
        ctx.system("You are helpful.")
        ctx.user("Tell me a joke.")

        async for event in await ctx.generate(Sampler.top_p(), decode=True):
            ...
    """

    __slots__ = ("_handle", "_model", "_pending_tokens")

    def __init__(
        self,
        model: Model,
    ) -> None:
        """Create a new anonymous context.

        Args:
            model: The model to create a context for.
        """
        self._handle = _ctx.Context.create(model._handle)
        self._model = model
        self._pending_tokens: list[int] = []

    @classmethod
    def open(cls, model: Model, name: str) -> Context | None:
        """Open a saved (named) context."""
        raw = _ctx.Context.open(model._handle, name)
        if raw is None:
            return None
        obj = object.__new__(cls)
        obj._handle = raw
        obj._model = model
        obj._pending_tokens = []
        return obj

    @classmethod
    def lookup(cls, model: Model, name: str) -> Context | None:
        """Look up an existing context by name. Alias for open()."""
        return cls.open(model, name)

    # --- Fill (ContextExt) ---

    def fill(self, text: str) -> Context:
        """Fill the context buffer with text (encodes to tokens)."""
        tokenizer = self._model.tokenizer()
        tokens = tokenizer.encode(text)
        self._pending_tokens.extend(tokens)
        return self

    def fill_tokens(self, tokens: list[int]) -> Context:
        """Fill the context buffer with raw token IDs."""
        self._pending_tokens.extend(tokens)
        return self

    async def flush(self) -> None:
        """Flush pending tokens: run forward pass and commit pages.

        Processes all pending tokens through the forward pass and commits
        full pages.  After flush, ``_pending_tokens`` is empty.
        """
        if not self._pending_tokens:
            return

        await _reserve_and_run(
            self._handle,
            self._model._handle,
            self._pending_tokens,
            self._handle.tokens_per_page(),
        )

        self._pending_tokens = []

    # --- Instruct (pie:instruct/chat) ---
    #
    # Filler methods return ``self`` so they can be chained:
    #
    #     ctx.system("...").user("...").cue()

    def system(self, message: str) -> Context:
        """Fill a system message."""
        self._pending_tokens.extend(_chat.system(self._model._handle, message))
        return self

    def user(self, message: str) -> Context:
        """Fill a user message."""
        self._pending_tokens.extend(_chat.user(self._model._handle, message))
        return self

    def assistant(self, message: str) -> Context:
        """Fill a (previous) assistant message."""
        self._pending_tokens.extend(_chat.assistant(self._model._handle, message))
        return self

    def cue(self) -> Context:
        """Cue the model to generate (fills the generation header)."""
        self._pending_tokens.extend(_chat.cue(self._model._handle))
        return self

    def seal(self) -> Context:
        """Seal the current turn (inserts stop token)."""
        self._pending_tokens.extend(_chat.seal(self._model._handle))
        return self

    def stop_tokens(self) -> list[int]:
        """Get the stop token IDs for this model."""
        return list(_chat.stop_tokens(self._model._handle))

    # --- Tool Use (pie:instruct/tool-use) ---

    def equip_tools(self, tools: list[str | dict]) -> Context:
        """Register available tools.

        Each tool may be either a JSON-schema string or a dict (auto-stringified).
        """
        strs = [t if isinstance(t, str) else _json.dumps(t) for t in tools]
        self._pending_tokens.extend(_tool.equip(self._model._handle, strs))
        return self

    def answer_tool(self, name: str, value: str | dict | list) -> Context:
        """Provide a tool call result.

        ``value`` may be a string or a JSON-serializable dict / list
        (auto-stringified).
        """
        s = value if isinstance(value, str) else _json.dumps(value)
        self._pending_tokens.extend(_tool.answer(self._model._handle, name, s))
        return self

    # --- Generate (ContextExt) ---

    async def generate(
        self,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        constrain: Schema | Constraint | None = None,
        logit_mask: list[int] | None = None,
        stop_tokens: Iterable[int] | None = None,
        decode: bool = False,
        reasoning: bool = False,
        tool_use: bool = False,
        auto_flush: bool = True,
    ) -> TokenStream | EventStream:
        """Start generating tokens.

        **Auto-flush behavior**: when ``auto_flush=True`` (the default),
        this method calls ``self.cue()`` followed by ``self.flush()`` before
        starting the generation stream — any pending instruct fills are
        committed and the model is cued to begin generating. Set
        ``auto_flush=False`` if you want to manage flushing yourself
        (e.g., to inspect the buffer before generation).

        Args:
            sampler: The sampling strategy to use.
            max_tokens: Maximum tokens to generate.
            constrain: Either a :class:`Schema` (compiled into a stateful
                grammar matcher) or any object implementing the
                :class:`Constraint` protocol. Mutually exclusive with
                ``logit_mask``.
            logit_mask: Static BRLE logit mask applied every step (for
                stateless constraints — banned tokens, etc.). Mutually
                exclusive with ``constrain``.
            stop_tokens: Override stop token IDs. Defaults to the model's
                chat stop tokens.
            decode: If ``True``, return an ``EventStream`` instead of ``TokenStream``.
            reasoning: Enable reasoning/thinking decoding (requires ``decode=True``).
            tool_use: Enable tool-use decoding (requires ``decode=True``).
            auto_flush: Auto-call ``cue()`` + ``flush()`` before generation (default ``True``).

        Returns:
            ``TokenStream`` (default) or ``EventStream`` (when ``decode=True``).
        """
        if constrain is not None and logit_mask is not None:
            raise ValueError(
                "constrain and logit_mask are mutually exclusive — "
                "wrap the static mask in a Constraint implementation if you "
                "need to combine both"
            )

        if auto_flush:
            self.cue()
            await self.flush()

        constraint: Constraint | None = None
        if constrain is not None:
            if isinstance(constrain, Schema):
                constraint = constrain._build(self._model)
            else:
                constraint = constrain
        elif logit_mask is not None:
            constraint = _StaticMaskConstraint(logit_mask)

        stream = TokenStream(
            self,
            sampler,
            max_tokens=max_tokens,
            constraint=constraint,
            stop_tokens=stop_tokens,
            pending_tokens=self._pending_tokens,
        )
        self._pending_tokens = []

        if decode or reasoning or tool_use:
            return EventStream(stream, reasoning=reasoning, tool_use=tool_use)

        return stream

    async def generate_text(
        self,
        sampler: Sampler,
        *,
        max_tokens: int | None = None,
        constrain: Schema | Constraint | None = None,
        logit_mask: list[int] | None = None,
        stop_tokens: Iterable[int] | None = None,
        auto_flush: bool = True,
    ) -> str:
        """Generate and return the full response text (one-shot).

        Convenience wrapper that calls ``generate(decode=True)`` and
        collects all ``Event.Text`` chunks into a single string. See
        :meth:`generate` for the auto-flush behavior.
        """
        parts: list[str] = []
        async for event in await self.generate(
            sampler,
            max_tokens=max_tokens,
            constrain=constrain,
            logit_mask=logit_mask,
            stop_tokens=stop_tokens,
            decode=True,
            auto_flush=auto_flush,
        ):
            if isinstance(event, Event.Text):
                parts.append(event.text)
            elif isinstance(event, Event.Done):
                break
        return "".join(parts)

    async def generate_json(
        self,
        sampler: Sampler,
        *,
        schema: str | None = None,
        max_tokens: int | None = None,
        stop_tokens: Iterable[int] | None = None,
        auto_flush: bool = True,
    ) -> Any:
        """Generate JSON-constrained output and parse it.

        If ``schema`` is provided, the output is constrained to be valid
        JSON conforming to that JSON Schema. Otherwise, output is
        constrained to any valid JSON.

        Returns the parsed Python value (``dict`` / ``list`` / primitive).
        For typed deserialization with pydantic, see :meth:`generate_pydantic`.
        For msgspec / dataclasses-json / etc., parse the return value yourself.

        ::

            data = await ctx.generate_json(
                Sampler.argmax(),
                schema=PERSON_SCHEMA,
            )
        """
        s = Schema.json_schema(schema) if schema is not None else Schema.json()
        text = await self.generate_text(
            sampler,
            max_tokens=max_tokens,
            constrain=s,
            stop_tokens=stop_tokens,
            auto_flush=auto_flush,
        )
        return _json.loads(text)

    if _PYDANTIC_AVAILABLE:
        async def generate_pydantic(
            self,
            model_class: type[BaseModel],
            *,
            sampler: Sampler,
            max_tokens: int | None = None,
            stop_tokens: Iterable[int] | None = None,
            auto_flush: bool = True,
        ) -> BaseModel:
            """Generate JSON conforming to a pydantic model and validate it.

            Requires pydantic v2.

            ::

                from pydantic import BaseModel

                class Person(BaseModel):
                    name: str
                    age: int

                person = await ctx.generate_pydantic(
                    Person, sampler=Sampler.argmax(),
                )

            Returns an instance of ``model_class``.
            """
            schema_dict = model_class.model_json_schema()
            schema_str = _json.dumps(schema_dict)
            text = await self.generate_text(
                sampler,
                max_tokens=max_tokens,
                constrain=Schema.json_schema(schema_str),
                stop_tokens=stop_tokens,
                auto_flush=auto_flush,
            )
            return model_class.model_validate_json(text)

    # --- Lifecycle ---

    def fork(self) -> Context:
        """Fork this context into a new anonymous one.

        The forked context shares committed KV pages with the parent.
        """
        raw = self._handle.fork()
        obj = object.__new__(Context)
        obj._handle = raw
        obj._model = self._model
        obj._pending_tokens = []
        return obj

    def save(self, name: str) -> None:
        """Save this context with a name, making it persistent."""
        self._handle.save(name)

    def release(self) -> None:
        """Explicitly release this context's resources."""
        self._handle.destroy()

    def __enter__(self) -> Context:
        return self

    def __exit__(self, *args) -> None:
        self.release()

    def __repr__(self) -> str:
        return f"Context({id(self._handle):#x})"
