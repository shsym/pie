"""
Drafter interface for speculative decoding.

Speculative decoding accelerates text generation by using a small, fast "drafter"
model to propose candidate tokens. These are verified by the main model.
"""

from abc import ABC, abstractmethod


class Drafter(ABC):
    """
    Abstract base class for drafters.

    Drafters propose candidate token sequences that are verified by the main model.
    The drafter produces a forest of Tries (prefix trees) representing multiple
    speculative token sequences.

    Workflow:
        1. Before generation starts, `update` is called with the full token history to
           synchronize the drafter with the main model's state.
        2. In each iteration:
           - `update` is called with the newly accepted tokens from the previous step.
           - `draft` is called to produce candidate token sequences.
    """

    @abstractmethod
    def update(self, context: list[int]) -> None:
        """
        Update the drafter's internal state with new context tokens.

        This method is called to keep the drafter synchronized with the main model's
        token history. It is invoked:
        - Once at the start of generation with the full token history.
        - After each verification step with the tokens that were accepted.

        Args:
            context: Token IDs to append to the drafter's internal context.
                     These are tokens that have been confirmed by the main model.
        """
        pass

    @abstractmethod
    def draft(self) -> tuple[list[int], list[int]]:
        """
        Generate draft token sequences for speculative verification.

        Returns a Trie forest encoded as two parallel lists in depth-first search (DFS)
        order. This encoding allows the verifier to efficiently traverse and verify
        multiple speculative paths.

        Returns:
            Tuple of (draft_tokens, draft_pos_ids) where:
            - draft_tokens: The proposed token IDs in DFS traversal order.
            - draft_pos_ids: The depth of each token in the trie, starting at 1 for
              root nodes. These positions are relative to the last context token.

        Invariants:
            The returned lists must satisfy the following invariants:
            1. Equal length: Both lists must have the same length.
            2. Valid depths: All position IDs must be >= 1 (depth 1 = immediate
               continuation of the context).
            3. Valid DFS order: The sequence must represent a valid DFS traversal
               of a Trie forest.

        Example:
            A draft representing multiple alternative continuations with branching:

            Context: [  ...  tokens  ...  ]
                                          |
                      +-------------------+--------------+
                      |                   |              |
                    [A] (1)            [F] (1)        [J] (1)
                      |                   |
                +-----+-----+          +--+--+
                |     |     |          |     |
              [B](2) [D](2) [E](2)   [G](2) [I](2)
                |                      |
              [C](3)                 [H](3)

            draft_tokens:  [A, B, C, D, E, F, G, H, I, J]
            draft_pos_ids: [1, 2, 3, 2, 2, 1, 2, 3, 2, 1]
        """
        pass


class EmptyDrafter(Drafter):
    """Empty drafter that produces no drafts (disables speculative decoding)."""

    def update(self, context: list[int]) -> None:
        """No-op: Empty drafter does not track context."""
        pass

    def draft(self) -> tuple[list[int], list[int]]:
        """Return empty drafts, effectively disabling speculative decoding."""
        return ([], [])
