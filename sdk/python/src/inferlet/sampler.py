"""
Sampler configuration for token generation.
"""

from dataclasses import dataclass


@dataclass
class Sampler:
    """
    Sampling configuration for token generation.

    Attributes:
        temperature: Controls randomness. 0 = deterministic, higher = more random.
        top_p: Nucleus sampling threshold (0-1). Only consider tokens with
               cumulative probability <= top_p.
        top_k: Only consider the top k most probable tokens.
        min_p: Minimum probability threshold (0-1).
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    @classmethod
    def greedy(cls) -> "Sampler":
        """
        Create a greedy sampler (always picks most probable token).
        """
        return cls(temperature=0.0)

    @classmethod
    def default(cls) -> "Sampler":
        """
        Create a sampler with balanced defaults.
        """
        return cls(temperature=0.7, top_p=0.95)

    @classmethod
    def creative(cls) -> "Sampler":
        """
        Create a more creative/diverse sampler.
        """
        return cls(temperature=1.0, top_p=0.95)

    @classmethod
    def top_p_sampling(cls, p: float = 0.95, temperature: float = 0.7) -> "Sampler":
        """
        Create a top-p (nucleus) sampler.

        Args:
            p: Cumulative probability threshold
            temperature: Temperature for sampling
        """
        return cls(top_p=p, temperature=temperature)

    @classmethod
    def top_k_sampling(cls, k: int = 40, temperature: float = 0.7) -> "Sampler":
        """
        Create a top-k sampler.

        Args:
            k: Number of top tokens to consider
            temperature: Temperature for sampling
        """
        return cls(top_k=k, temperature=temperature)

    @classmethod
    def min_p_sampling(cls, min_p: float = 0.1, temperature: float = 0.7) -> "Sampler":
        """
        Create a min-p sampler.

        Min-p sampling filters out tokens with probability below min_p * max_prob,
        where max_prob is the probability of the most likely token.

        Args:
            min_p: Minimum probability threshold (relative to max probability)
            temperature: Temperature for sampling
        """
        return cls(min_p=min_p, temperature=temperature)

    @classmethod
    def reasoning(cls) -> "Sampler":
        """
        Create a sampler optimized for reasoning tasks.

        Uses conservative sampling settings (top_k=20, top_p=0.95, temperature=0.6)
        to produce coherent, focused outputs suitable for chain-of-thought reasoning.
        """
        return cls(temperature=0.6, top_k=20, top_p=0.95)

    def with_temperature(self, temperature: float) -> "Sampler":
        """Return a new sampler with updated temperature."""
        return Sampler(
            temperature=temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )

    def with_top_p(self, top_p: float) -> "Sampler":
        """Return a new sampler with updated top_p."""
        return Sampler(
            temperature=self.temperature,
            top_p=top_p,
            top_k=self.top_k,
            min_p=self.min_p,
        )

    def with_top_k(self, top_k: int) -> "Sampler":
        """Return a new sampler with updated top_k."""
        return Sampler(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=top_k,
            min_p=self.min_p,
        )

    def with_min_p(self, min_p: float) -> "Sampler":
        """Return a new sampler with updated min_p."""
        return Sampler(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=min_p,
        )
