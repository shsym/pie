"""HuggingFace utilities for PIE backend.

This module provides utilities for:
- Resolving HuggingFace cache paths
- Loading model config from HuggingFace's config.json
- Loading tokenizer from HuggingFace's tokenizer.json
- Architecture mapping from HuggingFace to PIE
"""

import json
from pathlib import Path


# Mapping from HuggingFace model_type to PIE architecture
HF_TO_PIE_ARCH = {
    "llama": "llama3",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "gptoss": "gpt_oss",
}


def get_hf_cache_dir() -> Path:
    """Get the HuggingFace cache directory."""
    return Path.home() / ".cache" / "huggingface" / "hub"


def get_hf_snapshot_dir(repo_id: str) -> Path:
    """Get the snapshot directory for a HuggingFace model.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        
    Returns:
        Path to the snapshot directory containing model files
        
    Raises:
        ValueError: If the model is not found in cache
    """
    cache_dir = get_hf_cache_dir()
    
    # Convert repo_id to cache dirname format: org/repo -> models--org--repo
    dirname = "models--" + repo_id.replace("/", "--")
    model_cache = cache_dir / dirname
    
    if not model_cache.exists():
        raise ValueError(
            f"Model '{repo_id}' not found in HuggingFace cache. "
            f"Run 'pie model download {repo_id}' first."
        )
    
    # Find snapshot directory 
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.exists():
        raise ValueError(f"No snapshots found for '{repo_id}'")
    
    # Get the most recent snapshot (usually there's only one)
    # HF uses commit hashes for snapshot dirs, we take the first one
    snapshots = sorted(snapshots_dir.iterdir())
    if not snapshots:
        raise ValueError(f"No snapshots found for '{repo_id}'")
    
    return snapshots[0]


def load_hf_config(snapshot_dir: Path) -> dict:
    """Load and parse config.json from HuggingFace model snapshot.
    
    Args:
        snapshot_dir: Path to the HuggingFace snapshot directory
        
    Returns:
        Parsed config dictionary with normalized field names
    """
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"config.json not found in {snapshot_dir}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    return config


def normalize_hf_config(config: dict) -> dict:
    """Normalize HuggingFace config fields to PIE format.
    
    Maps HuggingFace field names to PIE's expected field names.
    
    Args:
        config: Raw config from config.json
        
    Returns:
        Normalized config dictionary
    """
    normalized = {}
    
    # Map HuggingFace -> PIE field names
    field_map = {
        # Architecture type
        "model_type": "type",  # Will be converted via HF_TO_PIE_ARCH
        
        # Layer dimensions
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        
        # Attention heads
        "num_attention_heads": "num_query_heads",
        "num_key_value_heads": "num_key_value_heads",
        "head_dim": "head_size",
        
        # Normalization
        "rms_norm_eps": "rms_norm_eps",
        
        # RoPE
        "rope_theta": "rope_theta",
    }
    
    for hf_name, pie_name in field_map.items():
        if hf_name in config:
            normalized[pie_name] = config[hf_name]
    
    # Special handling for model type
    if "type" in normalized:
        model_type = normalized["type"]
        normalized["type"] = HF_TO_PIE_ARCH.get(model_type, model_type)
    
    # Calculate head_size if not present
    if "head_size" not in normalized and "hidden_size" in normalized and "num_query_heads" in normalized:
        normalized["head_size"] = normalized["hidden_size"] // normalized["num_query_heads"]
    
    # Handle RoPE scaling
    rope = config.get("rope_scaling")
    if rope is not None:
        normalized["rope"] = {
            "theta": config.get("rope_theta", 10000.0),
            "factor": rope.get("factor", 1.0),
            "high_frequency_factor": rope.get("high_freq_factor", 1.0),
            "low_frequency_factor": rope.get("low_freq_factor", 1.0),
            "original_max_position_embeddings": rope.get("original_max_position_embeddings", 8192),
        }
    else:
        # Default RoPE config
        normalized["rope"] = {
            "theta": config.get("rope_theta", 10000.0),
            "factor": 1.0,
        }
    
    # Add defaults for missing fields
    if "rms_norm_eps" not in normalized:
        normalized["rms_norm_eps"] = 1e-5
    
    # QKV bias (some models have it, some don't)
    normalized["use_qkv_bias"] = config.get("attention_bias", False)
    
    return normalized


def load_hf_tokenizer(snapshot_dir: Path) -> dict:
    """Load tokenizer from HuggingFace format.
    
    Parses tokenizer.json and tokenizer_config.json to construct
    the PIE tokenizer format:
    - merge_table: dict[int, bytes] (rank -> token bytes)
    - special_tokens: dict[str, int]
    - split_regex: str
    - chat_template: str (Jinja format)
    
    Args:
        snapshot_dir: Path to the HuggingFace snapshot directory
        
    Returns:
        Tokenizer configuration dict in PIE format
    """
    result = {
        "type": "bpe",
        "num_vocab": 0,
        "merge_table": {},
        "special_tokens": {},
        "split_regex": "",
        "escape_non_printable": False,
        "chat_template": "",
    }
    
    # Load tokenizer.json for vocabulary
    tokenizer_path = snapshot_dir / "tokenizer.json"
    if tokenizer_path.exists():
        with open(tokenizer_path) as f:
            tokenizer_data = json.load(f)
        
        # Extract vocabulary from model.vocab
        model = tokenizer_data.get("model", {})
        vocab = model.get("vocab", {})
        
        # Build merge_table: rank -> bytes
        # HF vocab is {token_string: rank}, we need {rank: bytes}
        merge_table = {}
        for token_str, rank in vocab.items():
            try:
                # Encode token string to bytes
                token_bytes = token_str.encode("utf-8")
                merge_table[rank] = token_bytes
            except UnicodeEncodeError:
                # Skip tokens that can't be encoded
                continue
        
        # Auto-detect escape_non_printable
        # If the vocabulary contains U+0100 (Ā), it likely uses the byte-level mapping
        # where non-printable bytes are mapped to unicode characters starting at U+0100.
        # This is common in GPT-2, RoBERTa, and Qwen tokenizers.
        if "Ā" in vocab or "\u0100" in vocab:
             result["escape_non_printable"] = True
        
        result["merge_table"] = merge_table
        
        # Calculate true vocab size based on max index to avoid "id out of range" errors
        # Start with base vocab
        max_id = 0
        if vocab:
            max_id = max(vocab.values())
            
        # Extract added tokens for special tokens and update max_id
        added_tokens = tokenizer_data.get("added_tokens", [])
        for token_info in added_tokens:
            tid = token_info.get("id")
            if tid is not None:
                max_id = max(max_id, tid)
                
            # Add ALL added tokens to special_tokens, not just those marked special=True
            # This ensures we handle tokens like <think> (which might be special=False) correctly
            # during detokenization.
            content = token_info.get("content", "")
            if content and tid is not None:
                result["special_tokens"][content] = tid
        
        result["num_vocab"] = max_id + 1
        
        # Extract pre_tokenizer pattern (split_regex)
        pre_tokenizer = tokenizer_data.get("pre_tokenizer", {})
        if pre_tokenizer.get("type") == "Sequence":
            pretokenizers = pre_tokenizer.get("pretokenizers", [])
            for pt in pretokenizers:
                if pt.get("type") == "Split" and "pattern" in pt:
                    pattern = pt["pattern"]
                    if isinstance(pattern, dict) and "Regex" in pattern:
                        result["split_regex"] = pattern["Regex"]
                        break
    
    
    # Load tokenizer_config.json for chat template
    config_path = snapshot_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_data = json.load(f)
        
        # Get chat template (already in Jinja format)
        chat_template = config_data.get("chat_template", "")
        result["chat_template"] = _sanitize_chat_template(chat_template)
        
        # Additional special tokens from added_tokens_decoder
        decoder = config_data.get("added_tokens_decoder", {})
        for token_id_str, token_info in decoder.items():
            if token_info.get("special", False):
                content = token_info.get("content", "")
                if content:
                    result["special_tokens"][content] = int(token_id_str)
    
    return result


def _sanitize_chat_template(template: str) -> str:
    """Sanitize Jinja2 template for Minijinja compatibility.
    
    Minijinja (Rust) doesn't support Python string methods like .startswith(), 
    .endswith(), .strip(), .split() which are common in HF templates.
    """
    if not template:
        return ""
        
    sanitized = template
    
    # Replace .startswith() and .endswith() with slicing
    # Targeted replacements for known patterns in Qwen/Llama templates
    sanitized = sanitized.replace(".startswith('<tool_response>')", "[:15] == '<tool_response>'")
    sanitized = sanitized.replace(".endswith('</tool_response>')", "[-16:] == '</tool_response>'")
    
    # Replace .strip() variations with | trim filter
    # Note: | trim in Minijinja removes whitespace from start and end. 
    # It takes no arguments, so we lose specific char stripping, but it's usually fine.
    sanitized = sanitized.replace(".strip('\\n')", "| trim")
    sanitized = sanitized.replace(".lstrip('\\n')", "| trim")
    sanitized = sanitized.replace(".rstrip('\\n')", "| trim")
    sanitized = sanitized.replace(".strip()", "| trim")
    
    # Disable "thinking" logic which uses .split() - highly specific to DeepSeek/Qwen code
    # We'll just skip the split logic and treat content as a whole
    if ".split('</think>')" in sanitized:
        # We crudely disable the block that tries to split reasoning
        # This regex matches the if block that does the splitting
        import re
        # Pattern to find the thinking parsing block
        pattern = r"\{%- if '</think>' in content %\}.*?\{%- endif %\}"
        sanitized = re.sub(pattern, "", sanitized, flags=re.DOTALL)
    
    return sanitized



def get_safetensor_files(snapshot_dir: Path) -> list[str]:
    """Get list of safetensor files in a snapshot directory.
    
    Returns the filenames (not full paths) of all *.safetensors files.
    """
    files = []
    for f in snapshot_dir.iterdir():
        if f.suffix == ".safetensors":
            files.append(f.name)
    return sorted(files)
