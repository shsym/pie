"""Configuration utilities for Pie."""

import torch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def get_default_device() -> str:
    """Get the default device based on the platform."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_default_config_content() -> str:
    """Create the default configuration file content."""
    device = get_default_device()
    formatted_device = f'"{device}"'

    return f"""\
# Pie Server Configuration

# Network settings
host = "127.0.0.1"
port = 8080

# Authentication
enable_auth = false

# Model configuration (can have multiple [[model]] sections)
[[model]]
# HuggingFace model repository
hf_repo = "{DEFAULT_MODEL}"

# Device assignment (single GPU or list for tensor parallel)
device = [{formatted_device}]

# Precision settings
activation_dtype = "bfloat16"
weight_dtype = "bfloat16"

# KV cache configuration  
kv_page_size = 16

# Batch size limits
max_batch_tokens = 10240

# Distribution/sampling
max_dist_size = 32

# Embedding limits
max_num_embeds = 128

# Adapter (LoRA) settings
max_num_adapters = 32
max_adapter_rank = 8

# Memory management
gpu_mem_utilization = 0.9

# CUDA graphs (experimental)
use_cuda_graphs = false

# Debug options
enable_profiling = false
"""
