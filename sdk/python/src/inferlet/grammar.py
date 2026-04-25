"""
Grammar, Matcher, GrammarConstraint, and Schema for constrained decoding.

For most use cases, build a :class:`Schema` and pass it as ``constrain=`` to
:meth:`Context.generate` — the SDK compiles the schema into a stateful
matcher and drives it per generated token::

    text = await ctx.generate_text(
        Sampler.argmax(),
        constrain=Schema.ebnf(grammar),
        max_tokens=512,
    )

For custom logic (banned tokens, learned constraints, etc.), implement the
:class:`Constraint` protocol and pass an instance directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from wit_world.imports.inference import Grammar as _Grammar
from wit_world.imports.inference import Matcher as _Matcher

from .model import Tokenizer


# =============================================================================
# Grammar / Matcher (raw resource wrappers)
# =============================================================================


class Grammar:
    """Describes the structure that LLM output must conform to.

    Use the static factories::

        Grammar.from_json_schema('{"type": "object", ...}')
        Grammar.json()
        Grammar.from_regex(r"\\d{3}-\\d{4}")
        Grammar.from_ebnf('root ::= "hello" | "world"')
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: _Grammar) -> None:
        self._handle = handle

    @classmethod
    def from_json_schema(cls, schema: str) -> Grammar:
        """Construct from a JSON Schema string."""
        return cls(_Grammar.from_json_schema(schema))

    @classmethod
    def json(cls) -> Grammar:
        """Construct a grammar for any valid JSON."""
        return cls(_Grammar.json())

    @classmethod
    def from_regex(cls, pattern: str) -> Grammar:
        """Construct from a regular expression."""
        return cls(_Grammar.from_regex(pattern))

    @classmethod
    def from_ebnf(cls, ebnf: str) -> Grammar:
        """Construct from an EBNF grammar string."""
        return cls(_Grammar.from_ebnf(ebnf))


class Matcher:
    """Stateful grammar matcher producing token masks.

    Most callers should reach for :class:`Schema` or :class:`GrammarConstraint`
    instead — Matcher is the lower-level resource wrapper.
    """

    __slots__ = ("_handle",)

    def __init__(self, grammar: Grammar, tokenizer: Tokenizer) -> None:
        self._handle = _Matcher(grammar._handle, tokenizer._handle)

    def accept_tokens(self, token_ids: list[int]) -> None:
        """Accept tokens, advancing the matcher state."""
        self._handle.accept_tokens(token_ids)

    def next_token_logit_mask(self) -> list[int]:
        """Get the BRLE-encoded bitmask of valid next tokens."""
        return list(self._handle.next_token_logit_mask())

    @property
    def is_terminated(self) -> bool:
        """Whether the matcher has reached a terminal state."""
        return self._handle.is_terminated()

    def reset(self) -> None:
        """Reset to initial state for reuse."""
        self._handle.reset()


# =============================================================================
# Constraint protocol + GrammarConstraint
# =============================================================================


@runtime_checkable
class Constraint(Protocol):
    """Stateful sampling constraint protocol.

    On each generation step, the :class:`TokenStream` passes any newly
    accepted tokens (or ``[]`` on the first step) and gets back the
    BRLE-encoded logit mask for the next position.

    Returning ``[]`` means "no restriction".
    """

    def step(self, accepted: list[int]) -> list[int]:
        """Advance internal state with ``accepted`` tokens, then return the
        mask for the next position."""
        ...


class GrammarConstraint:
    """Grammar-driven :class:`Constraint` backed by a host :class:`Matcher`.

    Most callers should reach for :class:`Schema` instead — `GrammarConstraint`
    is the lower-level type for callers that want to keep a constraint
    instance around (e.g., to compose with custom constraints)."""

    __slots__ = ("_matcher",)

    def __init__(self, matcher: Matcher) -> None:
        self._matcher = matcher

    @classmethod
    def from_grammar(cls, grammar: Grammar, model) -> GrammarConstraint:
        """Build from a pre-compiled grammar (compile once, reuse)."""
        return cls(Matcher(grammar, model.tokenizer()))

    @classmethod
    def from_json_schema(cls, schema: str, model) -> GrammarConstraint:
        """Build from a JSON Schema string."""
        return cls.from_grammar(Grammar.from_json_schema(schema), model)

    @classmethod
    def json(cls, model) -> GrammarConstraint:
        """Build a constraint that accepts any valid JSON."""
        return cls.from_grammar(Grammar.json(), model)

    @classmethod
    def from_regex(cls, pattern: str, model) -> GrammarConstraint:
        """Build from a regular expression pattern."""
        return cls.from_grammar(Grammar.from_regex(pattern), model)

    @classmethod
    def from_ebnf(cls, ebnf: str, model) -> GrammarConstraint:
        """Build from an EBNF grammar string."""
        return cls.from_grammar(Grammar.from_ebnf(ebnf), model)

    def step(self, accepted: list[int]) -> list[int]:
        if accepted:
            self._matcher.accept_tokens(accepted)
        return self._matcher.next_token_logit_mask()


class _StaticMaskConstraint:
    """Wraps a static BRLE mask as a :class:`Constraint` (returned every step)."""

    __slots__ = ("_mask",)

    def __init__(self, mask: list[int]) -> None:
        self._mask = mask

    def step(self, accepted: list[int]) -> list[int]:
        return self._mask


# =============================================================================
# Schema — declarative constraint description
# =============================================================================


SchemaKind = Literal["json_schema", "json", "regex", "ebnf", "grammar"]


@dataclass(frozen=True, slots=True)
class Schema:
    """Declarative description of a constraint.

    Use with ``Context.generate(..., constrain=Schema.*)``. The SDK compiles
    the schema into a :class:`GrammarConstraint` internally.

    Frozen and hashable — safe to use as a dict key or in a set.

    ::

        Schema.json_schema('{"type": "object", ...}')
        Schema.json()                          # any valid JSON
        Schema.regex(r"\\d{3}-\\d{4}")
        Schema.ebnf('root ::= "hello" | "world"')
        Schema.grammar(precompiled_grammar)
    """

    kind: SchemaKind
    value: Any = None

    @classmethod
    def json_schema(cls, schema: str) -> Schema:
        """Constrain to JSON valid against the given JSON Schema string."""
        return cls("json_schema", schema)

    @classmethod
    def json(cls) -> Schema:
        """Constrain to any valid JSON."""
        return cls("json", None)

    @classmethod
    def regex(cls, pattern: str) -> Schema:
        """Constrain to strings matching the regular expression."""
        return cls("regex", pattern)

    @classmethod
    def ebnf(cls, ebnf: str) -> Schema:
        """Constrain to a custom EBNF grammar."""
        return cls("ebnf", ebnf)

    @classmethod
    def grammar(cls, grammar: Grammar) -> Schema:
        """Constrain to a pre-compiled :class:`Grammar` (compile once, reuse)."""
        return cls("grammar", grammar)

    def _build(self, model) -> GrammarConstraint:
        if self.kind == "json_schema":
            return GrammarConstraint.from_json_schema(self.value, model)
        if self.kind == "json":
            return GrammarConstraint.json(model)
        if self.kind == "regex":
            return GrammarConstraint.from_regex(self.value, model)
        if self.kind == "ebnf":
            return GrammarConstraint.from_ebnf(self.value, model)
        if self.kind == "grammar":
            return GrammarConstraint.from_grammar(self.value, model)
        raise ValueError(f"Unknown schema kind: {self.kind}")
