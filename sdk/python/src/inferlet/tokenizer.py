"""
Tokenizer class for encoding and decoding text.
"""

from wit_world.imports import tokenize as _tokenize


class Tokenizer:
    """
    Tokenizer for encoding text to tokens and decoding tokens to text.
    """

    def __init__(self, inner: _tokenize.Tokenizer) -> None:
        self._inner = inner

    def encode(self, text: str) -> list[int]:
        """
        Encode text to a list of token IDs.

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        return list(self._inner.tokenize(text))

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        return self._inner.detokenize(tokens)

    def __call__(self, text: str) -> list[int]:
        """
        Shorthand for encode.

        Example:
            tokens = tokenizer('hello')  # -> [1, 2, 3]
        """
        return self.encode(text)

    @property
    def vocabs(self) -> tuple[list[int], list[bytes]]:
        """
        Get the tokenizer's vocabulary.

        Returns:
            Tuple of (token IDs, token byte sequences)
        """
        ids, tokens = self._inner.get_vocabs()
        return list(ids), list(tokens)

    @property
    def split_regex(self) -> str:
        """Get the split regular expression used by the tokenizer."""
        return self._inner.get_split_regex()

    @property
    def special_tokens(self) -> tuple[list[int], list[bytes]]:
        """
        Get the special tokens recognized by the tokenizer.

        Returns:
            Tuple of (token IDs, token byte sequences)
        """
        ids, tokens = self._inner.get_special_tokens()
        return list(ids), list(tokens)
