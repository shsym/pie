"""Tests for HuggingFace utilities.

These tests verify that the HuggingFace config and tokenizer loading
produces output compatible with the legacy pie format.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
from pie_backend import hf_utils


# =============================================================================
# FIXTURES - Reference Data from HuggingFace Models
# =============================================================================

@pytest.fixture
def llama_config_json():
    """Sample config.json from meta-llama/Llama-3.2-1B-Instruct."""
    return {
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128008, 128009],
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "max_position_embeddings": 131072,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "rope_theta": 500000.0,
        "torch_dtype": "bfloat16",
        "vocab_size": 128256,
    }


@pytest.fixture
def qwen3_config_json():
    """Sample config.json for Qwen3 model."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
    }


@pytest.fixture
def sample_tokenizer_json():
    """Sample tokenizer.json structure with BPE vocab."""
    return {
        "model": {
            "vocab": {
                "hello": 0,
                "world": 1,
                " ": 2,
                "!": 3,
            },
        },
        "added_tokens": [
            {"id": 128000, "content": "<|begin_of_text|>", "special": True},
            {"id": 128001, "content": "<|end_of_text|>", "special": True},
            {"id": 128009, "content": "<|eot_id|>", "special": True},
        ],
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)"}},
            ],
        },
    }


@pytest.fixture
def sample_tokenizer_config_json():
    """Sample tokenizer_config.json with chat template."""
    return {
        "added_tokens_decoder": {
            "128000": {"content": "<|begin_of_text|>", "special": True},
            "128001": {"content": "<|end_of_text|>", "special": True},
            "128009": {"content": "<|eot_id|>", "special": True},
        },
        "chat_template": "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}",
    }


@pytest.fixture
def temp_snapshot_dir(tmp_path, llama_config_json, sample_tokenizer_json, sample_tokenizer_config_json):
    """Create a temporary snapshot directory with test files."""
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    
    # Write config.json
    (snapshot / "config.json").write_text(json.dumps(llama_config_json))
    
    # Write tokenizer.json
    (snapshot / "tokenizer.json").write_text(json.dumps(sample_tokenizer_json))
    
    # Write tokenizer_config.json
    (snapshot / "tokenizer_config.json").write_text(json.dumps(sample_tokenizer_config_json))
    
    # Create a dummy safetensor file
    (snapshot / "model.safetensors").write_bytes(b"dummy")
    
    return snapshot


# =============================================================================
# TEST: Architecture Mapping
# =============================================================================

class TestArchitectureMapping:
    """Test HuggingFace model_type to PIE architecture mapping."""
    
    def test_llama_mapping(self):
        """Llama model_type maps to llama3 architecture."""
        assert hf_utils.HF_TO_PIE_ARCH["llama"] == "llama3"
    
    def test_qwen2_mapping(self):
        """Qwen2 model_type maps to qwen2 architecture."""
        assert hf_utils.HF_TO_PIE_ARCH["qwen2"] == "qwen2"
    
    def test_qwen3_mapping(self):
        """Qwen3 model_type maps to qwen3 architecture."""
        assert hf_utils.HF_TO_PIE_ARCH["qwen3"] == "qwen3"
    
    def test_gptoss_mapping(self):
        """GPT-OSS model_type maps to gpt_oss architecture."""
        assert hf_utils.HF_TO_PIE_ARCH["gptoss"] == "gpt_oss"


# =============================================================================
# TEST: Config Normalization
# =============================================================================

class TestConfigNormalization:
    """Test HuggingFace config normalization to PIE format."""
    
    def test_llama_config_normalization(self, llama_config_json):
        """Verify Llama config is normalized correctly."""
        normalized = hf_utils.normalize_hf_config(llama_config_json)
        
        # Check core architecture fields are mapped correctly
        assert normalized["type"] == "llama3"
        assert normalized["num_layers"] == 16
        assert normalized["hidden_size"] == 2048
        assert normalized["intermediate_size"] == 8192
        assert normalized["vocab_size"] == 128256
        
        # Check attention head configuration
        assert normalized["num_query_heads"] == 32
        assert normalized["num_key_value_heads"] == 8
        
        # Check normalization epsilon
        assert normalized["rms_norm_eps"] == 1e-5
        
        # Check RoPE configuration
        assert "rope" in normalized
        assert normalized["rope"]["theta"] == 500000.0
        assert normalized["rope"]["factor"] == 32.0
    
    def test_qwen3_config_normalization(self, qwen3_config_json):
        """Verify Qwen3 config is normalized correctly."""
        normalized = hf_utils.normalize_hf_config(qwen3_config_json)
        
        assert normalized["type"] == "qwen3"
        assert normalized["num_layers"] == 28
        assert normalized["num_query_heads"] == 16
        assert normalized["num_key_value_heads"] == 8
        assert normalized["hidden_size"] == 1024
        assert normalized["intermediate_size"] == 3072
    
    def test_head_size_calculation(self, llama_config_json):
        """Verify head_size is calculated from hidden_size / num_query_heads."""
        normalized = hf_utils.normalize_hf_config(llama_config_json)
        
        expected_head_size = 2048 // 32  # hidden_size / num_attention_heads
        assert normalized["head_size"] == expected_head_size
    
    def test_qkv_bias_default(self, llama_config_json):
        """Verify use_qkv_bias defaults correctly."""
        normalized = hf_utils.normalize_hf_config(llama_config_json)
        assert normalized["use_qkv_bias"] is False
    
    def test_rope_scaling_fields(self, llama_config_json):
        """Verify RoPE scaling fields are extracted correctly."""
        normalized = hf_utils.normalize_hf_config(llama_config_json)
        
        rope = normalized["rope"]
        assert rope["high_frequency_factor"] == 4.0
        assert rope["low_frequency_factor"] == 1.0
        assert rope["original_max_position_embeddings"] == 8192


# =============================================================================
# TEST: Config Loading
# =============================================================================

class TestConfigLoading:
    """Test loading config from HuggingFace snapshot directory."""
    
    def test_load_hf_config(self, temp_snapshot_dir):
        """Verify config.json is loaded correctly."""
        config = hf_utils.load_hf_config(temp_snapshot_dir)
        
        assert config["model_type"] == "llama"
        assert config["hidden_size"] == 2048
        assert config["num_hidden_layers"] == 16
    
    def test_load_hf_config_missing_file(self, tmp_path):
        """Verify error when config.json is missing."""
        with pytest.raises(ValueError, match="config.json not found"):
            hf_utils.load_hf_config(tmp_path)


# =============================================================================
# TEST: Tokenizer Loading
# =============================================================================

class TestTokenizerLoading:
    """Test tokenizer loading from HuggingFace format."""
    
    def test_tokenizer_type(self, temp_snapshot_dir):
        """Verify tokenizer type is BPE by default."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        assert tokenizer["type"] == "bpe"
    
    def test_merge_table_structure(self, temp_snapshot_dir):
        """Verify merge_table is dict[int, bytes] format."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        merge_table = tokenizer["merge_table"]
        assert isinstance(merge_table, dict)
        
        # Check that keys are integers (ranks)
        for rank, token_bytes in merge_table.items():
            assert isinstance(rank, int)
            assert isinstance(token_bytes, bytes)
    
    def test_vocab_size(self, temp_snapshot_dir):
        """Verify num_vocab matches the vocabulary size."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        # Our sample has 4 tokens in vocab
        assert tokenizer["num_vocab"] == 4
    
    def test_special_tokens_extracted(self, temp_snapshot_dir):
        """Verify special tokens are extracted correctly."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        special_tokens = tokenizer["special_tokens"]
        assert "<|begin_of_text|>" in special_tokens
        assert "<|end_of_text|>" in special_tokens
        assert "<|eot_id|>" in special_tokens
        
        # Check token IDs
        assert special_tokens["<|begin_of_text|>"] == 128000
        assert special_tokens["<|end_of_text|>"] == 128001
    
    def test_chat_template_extracted(self, temp_snapshot_dir):
        """Verify chat_template is loaded from tokenizer_config.json."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        assert "chat_template" in tokenizer
        assert "{% for message in messages %}" in tokenizer["chat_template"]
    
    def test_split_regex_extracted(self, temp_snapshot_dir):
        """Verify split_regex is extracted from pre_tokenizer."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        # Our sample has a regex pattern in the Split pre_tokenizer
        assert tokenizer["split_regex"] == "(?i:'s|'t|'re|'ve|'m|'ll|'d)"


# =============================================================================
# TEST: Safetensor File Discovery
# =============================================================================

class TestSafetensorDiscovery:
    """Test discovery of safetensor files in snapshot."""
    
    def test_single_safetensor_file(self, temp_snapshot_dir):
        """Verify single model.safetensors is found."""
        files = hf_utils.get_safetensor_files(temp_snapshot_dir)
        
        assert len(files) == 1
        assert "model.safetensors" in files
    
    def test_multiple_safetensor_files(self, tmp_path):
        """Verify multiple safetensor files are found."""
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        
        # Create multiple sharded files
        for i in range(3):
            (snapshot / f"model-{i:05d}-of-00003.safetensors").write_bytes(b"dummy")
        
        files = hf_utils.get_safetensor_files(snapshot)
        
        assert len(files) == 3
        assert all(f.endswith(".safetensors") for f in files)
    
    def test_no_safetensor_files(self, tmp_path):
        """Verify empty list when no safetensor files."""
        files = hf_utils.get_safetensor_files(tmp_path)
        assert files == []


# =============================================================================
# TEST: Snapshot Directory Resolution
# =============================================================================

class TestSnapshotResolution:
    """Test HuggingFace cache directory resolution."""
    
    def test_get_hf_cache_dir(self):
        """Verify HF cache dir is ~/.cache/huggingface/hub/."""
        cache_dir = hf_utils.get_hf_cache_dir()
        
        assert cache_dir.name == "hub"
        assert "huggingface" in str(cache_dir)
    
    def test_get_hf_snapshot_dir_not_found(self):
        """Verify error when model not in cache."""
        with pytest.raises(ValueError, match="not found in HuggingFace cache"):
            hf_utils.get_hf_snapshot_dir("nonexistent/model-that-doesnt-exist")


# =============================================================================
# TEST: Legacy Format Compatibility
# =============================================================================

class TestLegacyCompatibility:
    """Test that output format is compatible with legacy pie expectations.
    
    The legacy pie format expected these fields in the tokenizer:
    - type: str (e.g., "bpe")
    - num_vocab: int
    - merge_table: dict[int, bytes]
    - split_regex: str
    - special_tokens: dict[str, int]
    - escape_non_printable: bool
    """
    
    def test_tokenizer_output_has_all_required_fields(self, temp_snapshot_dir):
        """Verify all legacy-required fields are present."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        required_fields = [
            "type",
            "num_vocab",
            "merge_table",
            "split_regex",
            "special_tokens",
            "escape_non_printable",
            "chat_template",
        ]
        
        for field in required_fields:
            assert field in tokenizer, f"Missing required field: {field}"
    
    def test_tokenizer_field_types(self, temp_snapshot_dir):
        """Verify field types match legacy expectations."""
        tokenizer = hf_utils.load_hf_tokenizer(temp_snapshot_dir)
        
        assert isinstance(tokenizer["type"], str)
        assert isinstance(tokenizer["num_vocab"], int)
        assert isinstance(tokenizer["merge_table"], dict)
        assert isinstance(tokenizer["split_regex"], str)
        assert isinstance(tokenizer["special_tokens"], dict)
        assert isinstance(tokenizer["escape_non_printable"], bool)
        assert isinstance(tokenizer["chat_template"], str)
    
    def test_config_output_has_required_architecture_fields(self, llama_config_json):
        """Verify normalized config has all fields needed by model loaders."""
        normalized = hf_utils.normalize_hf_config(llama_config_json)
        
        required_fields = [
            "type",
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_query_heads",
            "num_key_value_heads",
            "head_size",
            "rms_norm_eps",
            "use_qkv_bias",
            "rope",
        ]
        
        for field in required_fields:
            assert field in normalized, f"Missing required config field: {field}"


# =============================================================================
# INTEGRATION TEST: Real HuggingFace Cache
# =============================================================================

class TestRealHuggingFaceCache:
    """Integration tests using real HuggingFace cache if available.
    
    These tests are skipped if the model is not cached locally.
    """
    
    @pytest.fixture
    def llama_snapshot_dir(self):
        """Get the snapshot dir for Llama-3.2-1B-Instruct if cached."""
        try:
            return hf_utils.get_hf_snapshot_dir("meta-llama/Llama-3.2-1B-Instruct")
        except ValueError:
            pytest.skip("meta-llama/Llama-3.2-1B-Instruct not in cache")
    
    def test_real_llama_config(self, llama_snapshot_dir):
        """Test loading real Llama config from HF cache."""
        config = hf_utils.load_hf_config(llama_snapshot_dir)
        
        assert config["model_type"] == "llama"
        assert config["num_hidden_layers"] == 16
        assert config["hidden_size"] == 2048
    
    def test_real_llama_config_normalization(self, llama_snapshot_dir):
        """Test normalizing real Llama config."""
        config = hf_utils.load_hf_config(llama_snapshot_dir)
        normalized = hf_utils.normalize_hf_config(config)
        
        # Verify key fields
        assert normalized["type"] == "llama3"
        assert normalized["num_layers"] == 16
        assert normalized["head_size"] == 64
    
    def test_real_llama_tokenizer(self, llama_snapshot_dir):
        """Test loading real Llama tokenizer from HF cache."""
        tokenizer = hf_utils.load_hf_tokenizer(llama_snapshot_dir)
        
        # Llama-3.2-1B has 128256 vocab size
        assert tokenizer["num_vocab"] > 100000
        
        # Should have special tokens
        assert len(tokenizer["special_tokens"]) > 0
        
        # Should have a chat template
        assert len(tokenizer["chat_template"]) > 0
    
    def test_real_llama_safetensor_files(self, llama_snapshot_dir):
        """Test finding safetensor files in real Llama cache."""
        files = hf_utils.get_safetensor_files(llama_snapshot_dir)
        
        # Llama-3.2-1B has a single model.safetensors
        assert len(files) >= 1
        assert any("safetensors" in f for f in files)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
