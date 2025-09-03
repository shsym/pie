import os
import tomllib
from dataclasses import dataclass

from typing import Any

import torch
import base64


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
    # Qwen3 specific parameters
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28


@dataclass
class Tokenizer:
    """Struct for tokenizer configuration."""

    type: str
    merge_table: dict[int, bytes]
    split_regex: str
    special_tokens: dict[int, str]
    escape_non_printable: bool


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
    metadata_dir = os.path.dirname(os.path.abspath(path))

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
    arch_data = get_required_key(data, "architecture", "top-level")
    rope_data = get_required_key(arch_data, "rope", "architecture")

    # Get optional Qwen3-specific parameters
    attention_bias = arch_data.get("attention_bias", False)
    use_sliding_window = arch_data.get("use_sliding_window", False)
    sliding_window = arch_data.get("sliding_window", 4096)
    max_window_layers = arch_data.get("max_window_layers", 28)

    # Get architecture type to determine RoPE parameter handling
    arch_type = get_required_key(arch_data, "type", "architecture").lower()

    # Handle RoPE parameters differently based on architecture type
    if arch_type == "qwen3":
        # Qwen3 uses simpler RoPE configuration with only theta
        rope_factor = 1.0  # Default for Qwen3
        rope_high_frequency_factor = 4.0  # Default for Qwen3
        rope_low_frequency_factor = 1.0  # Default for Qwen3
        rope_theta = rope_data.get("theta", 10000.0)
    else:
        # Llama-style RoPE with factor-based scaling
        rope_factor = get_required_key(rope_data, "factor", "architecture.rope")
        rope_high_frequency_factor = get_required_key(
            rope_data, "high_frequency_factor", "architecture.rope"
        )
        rope_low_frequency_factor = get_required_key(
            rope_data, "low_frequency_factor", "architecture.rope"
        )
        rope_theta = get_required_key(rope_data, "theta", "architecture.rope")

    architecture = L4maConfig(
        type=arch_type,
        num_layers=get_required_key(arch_data, "num_layers", "architecture"),
        num_query_heads=get_required_key(arch_data, "num_query_heads", "architecture"),
        num_key_value_heads=get_required_key(
            arch_data, "num_key_value_heads", "architecture"
        ),
        head_size=get_required_key(arch_data, "head_size", "architecture"),
        hidden_size=get_required_key(arch_data, "hidden_size", "architecture"),
        intermediate_size=get_required_key(
            arch_data, "intermediate_size", "architecture"
        ),
        vocab_size=get_required_key(arch_data, "vocab_size", "architecture"),
        use_qkv_bias=arch_data.get("use_qkv_bias", False),
        rms_norm_eps=arch_data.get("rms_norm_eps", 1e-05),
        rope_factor=rope_factor,
        rope_high_frequency_factor=rope_high_frequency_factor,
        rope_low_frequency_factor=rope_low_frequency_factor,
        rope_theta=rope_theta,
        device="cuda:0",
        attention_bias=attention_bias,
        use_sliding_window=use_sliding_window,
        sliding_window=sliding_window,
        max_window_layers=max_window_layers,
    )

    # --- Parse Tokenizer ---
    tokenizer_data = get_required_key(data, "tokenizer", "top-level")
    vocabulary_file = get_required_key(tokenizer_data, "vocabulary_file", "tokenizer")
    model_name = get_required_key(data, "name", "top-level")

    vocabulary_full_path = os.path.join(metadata_dir, model_name, vocabulary_file)

    try:
        merge_rules = load_merge_rules(vocabulary_full_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load vocabulary file '{vocabulary_file}' at path '{vocabulary_full_path}': {e}"
        ) from e

    tokenizer = Tokenizer(
        type=get_required_key(tokenizer_data, "type", "tokenizer"),
        merge_table=merge_rules,
        split_regex=get_required_key(tokenizer_data, "split_regex", "tokenizer"),
        special_tokens=get_required_key(tokenizer_data, "special_tokens", "tokenizer"),
        escape_non_printable=get_required_key(
            tokenizer_data, "escape_non_printable", "tokenizer"
        ),
    )

    # --- Parse Template ---
    template_data = get_required_key(data, "template", "top-level")

    # --- Construct Final Metadata Object ---
    return ModelMetadata(
        name=get_required_key(data, "name", "top-level"),
        description=get_required_key(data, "description", "top-level"),
        version=get_required_key(data, "version", "top-level"),
        parameters=get_required_key(data, "parameters", "top-level"),
        architecture=architecture,
        tokenizer=tokenizer,
        template_type=get_required_key(template_data, "type", "template"),
        template_content=get_required_key(template_data, "content", "template"),
    )


def load_merge_rules(path: str) -> dict[int, bytes]:
    """
    Loads merge rules from a file.

    The file is expected to contain lines where each line has a
    base64-encoded token and an integer rank, separated by whitespace.
    Empty lines are skipped.

    Args:
        path: The path to the file to read.

    Returns:
        A dictionary mapping each rank (int) to its corresponding
        decoded token (bytes).

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        ValueError: If a line in the file is malformed (e.g., incorrect
                    number of parts, invalid base64, or non-integer rank).
    """
    merge_rules: dict[int, bytes] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty or blank lines
                if not line:
                    continue

                # Expect two parts: base64-encoded token and rank
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Error on line {line_number}: expected 2 parts, "
                        f"but found {len(parts)} (line: '{line}')"
                    )

                b64_token, rank_str = parts

                # 1. Decode base64 token
                try:
                    decoded_token = base64.b64decode(b64_token)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to decode base64 token. {e}"
                    ) from e

                # 2. Parse rank into an integer
                try:
                    rank = int(rank_str)
                except ValueError as e:
                    raise ValueError(
                        f"Error on line {line_number}: failed to parse rank '{rank_str}' as an integer."
                    ) from e

                # Insert into the dictionary
                merge_rules[rank] = decoded_token

    except FileNotFoundError:
        # Re-raise the exception to be handled by the caller
        raise

    return merge_rules
