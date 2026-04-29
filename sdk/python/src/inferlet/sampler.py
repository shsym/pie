"""
Sampler presets mapping to the WIT sampler variant.

The WIT sampler is a discriminated union:
  - multinomial(temperature, seed)
  - top-k(temperature, k)
  - top-p(temperature, p)
  - min-p(temperature, p)
  - top-k-top-p(temperature, k, p)
  - embedding
  - dist(temperature, top_k)
  - raw-logits
  - logprob(token_id)
  - logprobs(token_ids)
  - entropy
"""

from __future__ import annotations

from wit_world.imports.inference import (
    Sampler_Dist,
    Sampler_Embedding,
    Sampler_Entropy,
    Sampler_Logprob,
    Sampler_Logprobs,
    Sampler_MinP,
    Sampler_Multinomial,
    Sampler_RawLogits,
    Sampler_TopK,
    Sampler_TopKTopP,
    Sampler_TopP,
)
from wit_world.imports.inference import Sampler as WitSampler


class Sampler:
    """Sampler presets.

    Usage::

        Sampler.greedy()
        Sampler.top_p(temperature=0.8, top_p=0.9)
        Sampler.top_k(temperature=0.7, top_k=40)
    """

    __slots__ = ("_variant",)

    def __init__(self, variant: WitSampler) -> None:
        self._variant = variant

    # --- Presets ---

    @classmethod
    def greedy(cls) -> Sampler:
        """Deterministic (greedy / argmax) sampling.

        Recommended default for grammar-constrained generation: most masked
        positions have only a handful of valid tokens and stochastic
        sampling rarely improves quality.
        """
        return cls(Sampler_Multinomial((0.0, 1)))

    @classmethod
    def argmax(cls) -> Sampler:
        """Argmax sampling. Alias for :meth:`greedy`."""
        return cls.greedy()

    @classmethod
    def top_p(cls, temperature: float = 0.6, top_p: float = 0.95) -> Sampler:
        """Nucleus (top-p) sampling."""
        return cls(Sampler_TopP((temperature, top_p)))

    @classmethod
    def top_k(cls, temperature: float = 0.6, top_k: int = 50) -> Sampler:
        """Top-k sampling."""
        return cls(Sampler_TopK((temperature, top_k)))

    @classmethod
    def min_p(cls, temperature: float = 0.6, min_p: float = 0.1) -> Sampler:
        """Min-p sampling."""
        return cls(Sampler_MinP((temperature, min_p)))

    @classmethod
    def top_k_top_p(
        cls,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> Sampler:
        """Combined top-k + top-p sampling."""
        return cls(Sampler_TopKTopP((temperature, top_k, top_p)))

    @classmethod
    def multinomial(cls, temperature: float = 1.0, seed: int = 0) -> Sampler:
        """Plain multinomial sampling."""
        return cls(Sampler_Multinomial((temperature, seed)))

    @classmethod
    def embedding(cls) -> Sampler:
        """Embedding output mode (no sampling — returns hidden states)."""
        return cls(Sampler_Embedding())

    @classmethod
    def distribution(cls, temperature: float = 1.0, top_k: int = 0) -> Sampler:
        """Distribution output mode.

        Returns the top-``top_k`` token IDs with their probabilities instead
        of a sampled token. Useful for tree search, best-of-n, or external
        samplers. ``top_k=0`` returns the full distribution.
        """
        return cls(Sampler_Dist((temperature, top_k)))

    @classmethod
    def raw_logits(cls) -> Sampler:
        """Raw logits output mode.

        Returns the model's pre-softmax, untemperatured logits as a packed
        little-endian f32 byte buffer (length = ``vocab_size * 4``) per
        requested position. Decode in Python via
        ``np.frombuffer(buf, dtype=np.float32)``.
        """
        return cls(Sampler_RawLogits())

    @classmethod
    def logprob(cls, token_id: int) -> Sampler:
        """Single-label logprob.

        Returns ``log p(token_id | context)`` at this position via
        ``Output_Logprobs`` (a length-1 inner list per slot). Computed via
        log-softmax with no temperature scaling.
        """
        return cls(Sampler_Logprob(token_id))

    @classmethod
    def logprobs(cls, token_ids: list[int]) -> Sampler:
        """Multi-label logprobs at one position.

        Returns ``log p(t | context)`` for each ``t`` in ``token_ids`` via
        ``Output_Logprobs`` (a length-K inner list per slot, parallel to
        ``token_ids``). Useful for yes/no, multiple-choice, and reranking.
        """
        return cls(Sampler_Logprobs(list(token_ids)))

    @classmethod
    def entropy(cls) -> Sampler:
        """Shannon entropy of the unscaled distribution at this position.

        Returns ``H(p) = -sum(p * log p)`` via ``Output_Entropies``.
        Useful for uncertainty / confidence-based decisions.
        """
        return cls(Sampler_Entropy())

    def __repr__(self) -> str:
        return f"Sampler({self._variant})"
