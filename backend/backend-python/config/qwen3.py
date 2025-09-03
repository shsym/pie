"""
Qwen3 model configuration module.
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
class Qwen3ArchConfig(CommonArchConfig):
    """Qwen3 specific architecture configuration."""

    rope_theta: float
    attention_bias: bool
    use_sliding_window: bool
    sliding_window: int
    max_window_layers: int


def parse_qwen3_architecture(arch_data: dict) -> Qwen3ArchConfig:
    """Parse Qwen3-specific architecture configuration."""
    # Get architecture type and validate it's Qwen3
    arch_type = get_required_key(arch_data, "type").lower()
    if arch_type != "qwen3":
        raise ValueError(
            f"Expected architecture type 'qwen3', but got '{arch_type}' "
            f"in {arch_data['#file_path']}"
        )

    # Parse common fields
    common_fields = parse_common_architecture_fields(arch_data)

    # Parse RoPE configuration (Qwen3 uses simpler RoPE with only theta)
    rope_data = get_required_key(arch_data, "rope")
    rope_theta = rope_data.get("theta", 10000.0)

    # Get Qwen3-specific parameters
    attention_bias = arch_data.get("attention_bias", False)
    use_sliding_window = arch_data.get("use_sliding_window", False)
    sliding_window = arch_data.get("sliding_window", 4096)
    max_window_layers = arch_data.get("max_window_layers", 28)

    return Qwen3ArchConfig(
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
        # Qwen3-specific fields
        rope_theta=rope_theta,
        attention_bias=attention_bias,
        use_sliding_window=use_sliding_window,
        sliding_window=sliding_window,
        max_window_layers=max_window_layers,
    )


def parse_model_metadata(path: str) -> ModelMetadata:
    """
    Parses a dictionary (from loaded TOML data) into the ModelMetadata struct,
    with improved and informative error handling.
    """
    metadata_dir = os.path.dirname(os.path.abspath(path))

    # Load the TOML file as a dictionary
    toml_dict = load_toml_file(path)

    # Parse architecture section
    arch_dict = get_required_key(toml_dict, "architecture")
    arch_cfg = parse_qwen3_architecture(arch_dict)

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
