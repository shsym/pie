"""Configuration utilities for Pie."""


def create_default_config_content() -> str:
    """Create the default configuration file content."""
    return """\
# Pie Server Configuration

# Network settings
host = "127.0.0.1"
port = 8080

# Authentication
enable_auth = true

# Model configuration (can have multiple [[model]] sections)
[[model]]
# HuggingFace model repository
hf_repo = "Qwen/Qwen3-0.6B"

# Device assignment (single GPU or list for tensor parallel)
device = ["cuda:0"]

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
