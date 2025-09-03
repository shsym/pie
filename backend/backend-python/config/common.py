"""
Base configuration module containing common classes and utilities for model configurations.
This module provides shared functionality for different model architectures.
"""

import os
import tomllib
import base64
from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class Tokenizer:
    """Struct for tokenizer configuration."""

    type: str
    merge_table: dict[int, bytes]
    split_regex: str
    special_tokens: dict[int, str]
    escape_non_printable: bool


@dataclass
class CommonArchConfig:
    """Base architecture configuration with common fields."""

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
    device: str
    dtype: torch.dtype


@dataclass
class ModelMetadata:
    """Top-level struct to hold the entire parsed model configuration."""

    name: str
    description: str
    version: str
    parameters: list[str]
    architecture: CommonArchConfig
    tokenizer: Tokenizer
    template_type: str
    template_content: str


def get_required_key(data_dict: dict, key: str) -> Any:
    """Gets a key from a dict, raising a contextual KeyError if it's missing."""
    try:
        return data_dict[key]
    except KeyError:
        raise KeyError(
            f"Missing required key '{key}' in the '[{data_dict['#section_name']}]' section "
            f"of the TOML file: {data_dict['#file_path']}"
        ) from None


def load_toml_file(path: str) -> dict:
    """Load and parse a TOML file."""
    try:
        with open(path, "rb") as f:
            conf = tomllib.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at path: {path}"
        ) from None
    except tomllib.TOMLDecodeError as e:
        raise tomllib.TOMLDecodeError(f"Error parsing TOML file at {path}: {e}") from e

    # For each subdictionary, i.e., each subsection in the TOML file,
    # add its section name and file path to the dictionary, to get clearer error messages.
    def recur_add_sec_name_and_path(conf: dict, sec_name: str, path: str):
        conf["#section_name"] = sec_name
        conf["#file_path"] = path

        for key, value in conf.items():
            if isinstance(value, dict) and isinstance(key, str):
                recur_add_sec_name_and_path(value, key, path)

    recur_add_sec_name_and_path(conf, "top-level", path)

    return conf


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
                    f"Error on line {line_number}: failed to decode base64 token."
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

    return merge_rules


def parse_common_architecture_fields(arch_data: dict) -> dict:
    """Parse common architecture fields that are shared across all models."""

    def get_required_arch_key(key: str):
        return get_required_key(arch_data, key)

    return {
        "num_layers": get_required_arch_key("num_layers"),
        "num_query_heads": get_required_arch_key("num_query_heads"),
        "num_key_value_heads": get_required_arch_key("num_key_value_heads"),
        "head_size": get_required_arch_key("head_size"),
        "hidden_size": get_required_arch_key("hidden_size"),
        "intermediate_size": get_required_arch_key("intermediate_size"),
        "vocab_size": get_required_arch_key("vocab_size"),
        "use_qkv_bias": get_required_arch_key("use_qkv_bias"),
        "rms_norm_eps": get_required_arch_key("rms_norm_eps"),
        "device": "cuda:0",
        "dtype": torch.bfloat16,
    }


def parse_tokenizer_section(data: dict, metadata_dir: str) -> Tokenizer:
    """Parse the tokenizer section of the configuration."""
    tokenizer_data = get_required_key(data, "tokenizer")

    def get_required_tokenizer_key(key: str):
        return get_required_key(tokenizer_data, key)

    vocabulary_file = get_required_tokenizer_key("vocabulary_file")
    model_name = get_required_key(data, "name")

    vocabulary_full_path = os.path.join(metadata_dir, model_name, vocabulary_file)

    try:
        merge_rules = load_merge_rules(vocabulary_full_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load vocabulary file '{vocabulary_file}' at path '{vocabulary_full_path}'"
        ) from e

    # Remove the section name and file path from the special tokens dictionary.
    # These two keys are helper keys added by the load_toml_file function for error messages.
    # They are not part of the special tokens dictionary.
    # Remove them so that protobuf serialization works.
    special_tokens = get_required_tokenizer_key("special_tokens")
    del special_tokens["#section_name"]
    del special_tokens["#file_path"]

    return Tokenizer(
        type=get_required_tokenizer_key("type"),
        merge_table=merge_rules,
        split_regex=get_required_tokenizer_key("split_regex"),
        special_tokens=special_tokens,
        escape_non_printable=get_required_tokenizer_key("escape_non_printable"),
    )


def parse_metadata_fields(data: dict) -> dict:
    """Parse common metadata fields."""

    def get_required_top_level_key(key: str):
        return get_required_key(data, key)

    return {
        "name": get_required_top_level_key("name"),
        "description": get_required_top_level_key("description"),
        "version": get_required_top_level_key("version"),
        "parameters": get_required_top_level_key("parameters"),
    }


def parse_template_section(data: dict) -> dict:
    """Parse the template section of the configuration."""
    template_data = get_required_key(data, "template")

    def get_required_template_key(key: str):
        return get_required_key(template_data, key)

    return {
        "template_type": get_required_template_key("type"),
        "template_content": get_required_template_key("content"),
    }
