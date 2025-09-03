"""
L4MA/Llama model configuration module.
"""

import os
from dataclasses import dataclass

from .common import (
    CommonArchConfig,
    ModelMetadata,
    load_toml_file,
    get_required_key,
    parse_common_architecture_fields,
    parse_tokenizer_section,
    parse_metadata_fields,
    parse_template_section,
)


@dataclass
class L4maArchConfig(CommonArchConfig):
    """L4MA/Llama specific architecture configuration."""

    rope_factor: float
    rope_high_frequency_factor: float
    rope_low_frequency_factor: float
    rope_theta: float


def parse_l4ma_architecture(arch_data: dict) -> L4maArchConfig:
    """Parse L4MA/Llama-specific architecture configuration."""
    # Get architecture type and validate it's L4MA/Llama
    arch_type = get_required_key(arch_data, "type").lower()
    if arch_type != "l4ma":
        raise ValueError(
            f"Expected architecture type 'l4ma', but got '{arch_type}' in {arch_data['#file_path']}"
        )

    # Parse common fields
    common_fields = parse_common_architecture_fields(arch_data)

    # Parse RoPE configuration (Llama-style with factor-based scaling)
    rope_data = get_required_key(arch_data, "rope")
    rope_factor = get_required_key(rope_data, "factor")
    rope_high_frequency_factor = get_required_key(rope_data, "high_frequency_factor")
    rope_low_frequency_factor = get_required_key(rope_data, "low_frequency_factor")
    rope_theta = get_required_key(rope_data, "theta")

    return L4maArchConfig(
        # Common fields
        type=arch_type,
        num_layers=common_fields["num_layers"],
        num_query_heads=common_fields["num_query_heads"],
        num_key_value_heads=common_fields["num_key_value_heads"],
        head_size=common_fields["head_size"],
        hidden_size=common_fields["hidden_size"],
        intermediate_size=common_fields["intermediate_size"],
        vocab_size=common_fields["vocab_size"],
        use_qkv_bias=common_fields["use_qkv_bias"],
        rms_norm_eps=common_fields["rms_norm_eps"],
        device=common_fields["device"],
        dtype=common_fields["dtype"],
        # L4MA-specific fields
        rope_factor=rope_factor,
        rope_high_frequency_factor=rope_high_frequency_factor,
        rope_low_frequency_factor=rope_low_frequency_factor,
        rope_theta=rope_theta,
    )


def parse_model_metadata(path: str) -> ModelMetadata:
    """
    Parses a dictionary (from loaded TOML data) into the ModelMetadata struct,
    with improved and informative error handling.
    """
    metadata_dir = os.path.dirname(os.path.abspath(path))

    # Load and parse TOML file
    toml_dict = load_toml_file(path)

    # Parse architecture section
    arch_dict = get_required_key(toml_dict, "architecture")
    arch_cfg = parse_l4ma_architecture(arch_dict)

    # Parse tokenizer section
    tokenizer = parse_tokenizer_section(toml_dict, metadata_dir)

    # Parse metadata fields
    metadata_dict = parse_metadata_fields(toml_dict)

    # Parse template section
    template_dict = parse_template_section(toml_dict)

    # Construct final metadata object
    return ModelMetadata(
        architecture=arch_cfg,
        tokenizer=tokenizer,
        **metadata_dict,
        **template_dict,
    )
