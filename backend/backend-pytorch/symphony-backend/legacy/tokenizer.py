import numpy as np
from transformers import AutoTokenizer


class Tokenizer:

    def __init__(self):
        ...

    @staticmethod
    def from_huggingface(model_name: str):
        return AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: list[int]) -> str:
        ...

    def eos_token_id(self) -> int:
        ...


class DummyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def from_huggingface(model_name: str):
        return DummyTokenizer()

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(i) for i in token_ids)

    def eos_token_id(self) -> int:
        return 10


def load_tokenizer(model_name: str) -> Tokenizer:
    if model_name == "dummy":
        return DummyTokenizer()
    else:
        return Tokenizer.from_huggingface(model_name)
