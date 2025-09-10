import msgspec

from typing import Optional


# ==============================================================================
# 1. DATA STRUCTURES (using msgspec.Struct and modern type hints)
# ==============================================================================


class HandshakeRequest(msgspec.Struct, gc=False):
    version: str


class HandshakeResponse(msgspec.Struct, gc=False):
    version: str
    model_name: str
    model_traits: list[str]  # Use built-in list
    model_description: str
    prompt_template: str
    prompt_template_type: str
    prompt_stop_tokens: list[str]
    kv_page_size: int
    resources: dict[int, int]  # Use built-in list and tuple
    tokenizer_merge_table: dict[int, bytes]
    tokenizer_special_tokens: dict[str, int]
    tokenizer_split_regex: str
    tokenizer_escape_non_printable: bool


class QueryRequest(msgspec.Struct, gc=False):
    query: str


class QueryResponse(msgspec.Struct, gc=False):
    value: str


class ForwardPassRequest(msgspec.Struct, gc=False):
    input_tokens: list[int]
    input_token_positions: list[int]
    input_embed_ptrs: list[int]
    input_embed_positions: list[int]
    adapter: Optional[int]
    adapter_seed: Optional[int]
    mask: list[list[int]]
    kv_page_ptrs: list[int]
    kv_page_last_len: int
    output_token_indices: list[int]
    output_token_samplers: list[dict]
    output_embed_ptrs: list[int]
    output_embed_indices: list[int]


class ForwardPassResponse(msgspec.Struct, gc=False):
    tokens: list[int]
    dists: list[tuple[list[int], list[float]]]


class EmbedImageRequest(msgspec.Struct, gc=False):
    embed_ptrs: list[int]
    image_blob: bytes
    position_offset: int


class InitializeAdapterRequest(msgspec.Struct, gc=False):
    adapter_ptr: int
    rank: int
    alpha: float
    population_size: int
    mu_fraction: float
    initial_sigma: float


class UpdateAdapterRequest(msgspec.Struct, gc=False):
    adapter_ptr: int
    scores: list[float]
    seeds: list[int]
    max_sigma: float
