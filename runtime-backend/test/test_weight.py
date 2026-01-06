"""
Unit tests for the WeightSchema module.
"""

from __future__ import annotations

import pytest
import torch

from src.weight import Schema, Source, LoadConfig, WeightStore, Definition


class TestSource:
    """Tests for Source class."""

    def test_single_pattern(self):
        source = Source("model.embed.weight")
        assert source.patterns == ["model.embed.weight"]
        assert not source.is_fused
        assert source.sharding is None
        assert not source.should_quantize

    def test_fuse_multiple_patterns(self):
        source = Source.fuse(
            [
                "model.layers.0.q_proj.weight",
                "model.layers.0.k_proj.weight",
                "model.layers.0.v_proj.weight",
            ],
            dim=0,
        )

        assert len(source.patterns) == 3
        assert source.is_fused
        assert source.fuse_dim == 0

    def test_method_chaining(self):
        source = Source("model.weight").shard("column").quantize()

        assert source.sharding == "column"
        assert source.should_quantize

    def test_fuse_with_chaining(self):
        source = Source.fuse(["a", "b"], dim=1).shard("row").quantize()

        assert source.is_fused
        assert source.fuse_dim == 1
        assert source.sharding == "row"
        assert source.should_quantize


class TestDefinition:
    """Tests for Definition class."""

    def test_has_layer_pattern(self):
        defn = Definition(name="layers.*.proj", source=Source("x"))
        assert defn.has_layer_pattern()

        defn2 = Definition(name="embed", source=Source("x"))
        assert not defn2.has_layer_pattern()

    def test_expand_for_layer(self):
        defn = Definition(
            name="layers.*.proj_qkv",
            source=Source.fuse(
                [
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                ],
                dim=0,
            ),
        )

        assert defn.expand_for_layer(5) == "layers.5.proj_qkv"
        assert defn.expand_source_for_layer(5) == [
            "model.layers.5.self_attn.q_proj.weight",
            "model.layers.5.self_attn.k_proj.weight",
        ]


class TestWeightStore:
    """Tests for WeightStore class."""

    def test_put_get(self):
        store = WeightStore()
        t = torch.randn(4, 4)
        store.put("weight", t)

        assert "weight" in store
        assert torch.equal(store.get("weight"), t)
        assert len(store) == 1

    def test_get_list(self):
        store = WeightStore()
        for i in range(3):
            store.put(f"layers.{i}.proj", torch.randn(2, 2))

        tensors = store.get_list("layers.*.proj", 3)
        assert len(tensors) == 3

    def test_get_missing_raises(self):
        store = WeightStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")


class TestSchema:
    """Tests for Schema class."""

    def test_define_basic(self):
        schema = Schema("test").define("embed", Source("model.embed_tokens.weight"))

        assert len(schema._definitions) == 1
        assert schema._definitions[0].name == "embed"

    def test_define_chaining(self):
        schema = (
            Schema("test")
            .define("a", Source("x"))
            .define("b", Source("y"))
            .define("c", Source("z"))
        )

        assert len(schema._definitions) == 3

    def test_load_single_tensor(self):
        """Test loading a single non-layer tensor."""
        mock_tensors = {
            "model.embed.weight": torch.randn(100, 64),
        }

        def mock_reader(name: str) -> torch.Tensor:
            return mock_tensors[name]

        schema = Schema("test").define("embed", Source("model.embed.weight"))

        config = LoadConfig(device=torch.device("cpu"))
        store = schema.load(mock_reader, config, num_layers=0)

        assert "embed" in store
        assert store.get("embed").shape == (100, 64)

    def test_load_per_layer_tensors(self):
        """Test loading tensors with layer patterns."""
        mock_tensors = {
            f"model.layers.{i}.weight": torch.randn(32, 32) for i in range(4)
        }

        def mock_reader(name: str) -> torch.Tensor:
            return mock_tensors[name]

        schema = Schema("test").define(
            "layers.*.weight", Source("model.layers.*.weight")
        )

        config = LoadConfig(device=torch.device("cpu"))
        store = schema.load(mock_reader, config, num_layers=4)

        assert len(store) == 4
        for i in range(4):
            assert f"layers.{i}.weight" in store

    def test_load_fused_tensors(self):
        """Test loading and fusing multiple tensors."""
        mock_tensors = {
            "model.a": torch.ones(2, 4),
            "model.b": torch.ones(2, 4) * 2,
            "model.c": torch.ones(2, 4) * 3,
        }

        def mock_reader(name: str) -> torch.Tensor:
            return mock_tensors[name]

        schema = Schema("test").define(
            "fused", Source.fuse(["model.a", "model.b", "model.c"], dim=0)
        )

        config = LoadConfig(device=torch.device("cpu"))
        store = schema.load(mock_reader, config, num_layers=0)

        fused = store.get("fused")
        assert fused.shape == (6, 4)  # 3 tensors of shape (2, 4) fused on dim=0
        assert fused[0, 0] == 1  # From model.a
        assert fused[2, 0] == 2  # From model.b
        assert fused[4, 0] == 3  # From model.c

    def test_load_with_sharding(self):
        """Test tensor sharding across ranks."""
        mock_tensors = {
            "weight": torch.randn(8, 16),
        }

        def mock_reader(name: str) -> torch.Tensor:
            return mock_tensors[name]

        # Column sharding (dim=0)
        schema = Schema("test").define("col_shard", Source("weight").shard("column"))

        config = LoadConfig(
            device=torch.device("cpu"),
            world_size=2,
            rank=0,
        )
        store = schema.load(mock_reader, config, num_layers=0)

        # Should get first half (4 rows out of 8)
        assert store.get("col_shard").shape == (4, 16)

        # Row sharding (dim=1)
        schema2 = Schema("test").define("row_shard", Source("weight").shard("row"))

        store2 = schema2.load(mock_reader, config, num_layers=0)

        # Should get first half (8 rows out of 16)
        assert store2.get("row_shard").shape == (8, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
