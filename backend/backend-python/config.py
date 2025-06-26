import tomllib
from dataclasses import dataclass

from typing import Any

import torch


# Assuming L4maConfig is defined elsewhere, as in the original code.
# For this example to be self-contained, we'll create a placeholder.
@dataclass
class L4maConfig:
    type: str
    num_layers: int
    num_query_heads: int
    num_key_value_heads: int
    head_size: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    use_qkv_bias: bool
    rms_norm_eps: float
    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16


@dataclass
class Tokenizer:
    """Struct for tokenizer configuration."""
    type: str
    vocabulary_file: str
    split_regex: str
    special_tokens: dict[str, str]


@dataclass
class ModelMetadata:
    """Top-level struct to hold the entire parsed model configuration."""
    name: str
    description: str
    version: str
    parameters: list[str]
    architecture: L4maConfig
    tokenizer: Tokenizer
    template_type: str
    template_content: str


def parse_model_metadata(path: str) -> ModelMetadata:
    """
    Parses a dictionary (from loaded TOML data) into the ModelMetadata struct,
    with improved and informative error handling.
    """

    # Helper function to safely get required keys from a dictionary
    def get_required_key(data_dict: dict, key: str, section_name: str) -> Any:
        """Gets a key from a dict, raising a contextual KeyError if it's missing."""
        try:
            return data_dict[key]
        except KeyError:
            # Provide a much more informative error message
            raise KeyError(
                f"Missing required key '{key}' in the '[{section_name}]' section "
                f"of the TOML file: {path}"
            )

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at path: {path}")
    except tomllib.TOMLDecodeError as e:
        raise tomllib.TOMLDecodeError(f"Error parsing TOML file at {path}: {e}")

    # --- Parse Architecture ---
    arch_data = get_required_key(data, 'architecture', 'top-level')
    rope_data = get_required_key(arch_data, 'rope', 'architecture')

    architecture = L4maConfig(
        type=get_required_key(arch_data, 'type', 'architecture'),
        num_layers=get_required_key(arch_data, 'num_layers', 'architecture'),
        num_query_heads=get_required_key(arch_data, 'num_query_heads', 'architecture'),
        num_key_value_heads=get_required_key(arch_data, 'num_key_value_heads', 'architecture'),
        head_size=get_required_key(arch_data, 'head_size', 'architecture'),
        hidden_size=get_required_key(arch_data, 'hidden_size', 'architecture'),
        intermediate_size=get_required_key(arch_data, 'intermediate_size', 'architecture'),
        vocab_size=get_required_key(arch_data, 'vocab_size', 'architecture'),
        use_qkv_bias=False,  # Still hardcoded as in the original
        rms_norm_eps=1e-05,  # Still hardcoded as in the original
        rope_factor=get_required_key(rope_data, 'factor', 'architecture.rope'),
        rope_high_frequency_factor=get_required_key(rope_data, 'high_frequency_factor', 'architecture.rope'),
        rope_low_frequency_factor=get_required_key(rope_data, 'low_frequency_factor', 'architecture.rope'),
        rope_theta=get_required_key(rope_data, 'theta', 'architecture.rope'),
        device="cuda:0",
    )

    # --- Parse Tokenizer ---
    tokenizer_data = get_required_key(data, 'tokenizer', 'top-level')
    tokenizer = Tokenizer(
        type=get_required_key(tokenizer_data, 'type', 'tokenizer'),
        vocabulary_file=get_required_key(tokenizer_data, 'vocabulary_file', 'tokenizer'),
        split_regex=get_required_key(tokenizer_data, 'split_regex', 'tokenizer'),
        special_tokens=get_required_key(tokenizer_data, 'special_tokens', 'tokenizer')
    )

    # --- Parse Template ---
    template_data = get_required_key(data, 'template', 'top-level')

    # --- Construct Final Metadata Object ---
    return ModelMetadata(
        name=get_required_key(data, 'name', 'top-level'),
        description=get_required_key(data, 'description', 'top-level'),
        version=get_required_key(data, 'version', 'top-level'),
        parameters=get_required_key(data, 'parameters', 'top-level'),
        architecture=architecture,
        tokenizer=tokenizer,
        template_type=get_required_key(template_data, 'type', 'template'),
        template_content=get_required_key(template_data, 'content', 'template'),
    )
