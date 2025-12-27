"""Tests for pie_cli.model module."""

import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import toml
from typer.testing import CliRunner

from pie_cli.cli import app

runner = CliRunner()


class TestModelList:
    """Tests for model list command."""

    def test_list_empty(self, tmp_path):
        """Shows message when no models."""
        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".cache" / "pie")}):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "No models found" in result.stdout

    def test_list_models(self, tmp_path):
        """Lists downloaded models."""
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "model-a").mkdir()
        (models_dir / "model-b").mkdir()

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path)}):
            result = runner.invoke(app, ["model", "list"])

        assert result.exit_code == 0
        assert "model-a" in result.stdout
        assert "model-b" in result.stdout


class TestModelRemove:
    """Tests for model remove command."""

    def test_remove_model(self, tmp_path):
        """Removes downloaded model."""
        models_dir = tmp_path / "models"
        model_dir = models_dir / "test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "weights.bin").write_bytes(b"fake weights")
        (models_dir / "test-model.toml").write_text("name = 'test-model'")

        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path)}):
            result = runner.invoke(app, ["model", "remove", "test-model"])

        assert result.exit_code == 0
        assert not model_dir.exists()
        assert not (models_dir / "test-model.toml").exists()

    def test_remove_nonexistent(self, tmp_path):
        """Returns error when model doesn't exist."""
        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path)}):
            result = runner.invoke(app, ["model", "remove", "nonexistent"])

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()


class TestModelSearch:
    """Tests for model search command."""

    def test_search_models(self, tmp_path):
        """Searches models with mocked API."""
        mock_response = [
            {"name": "llama-7b.toml", "type": "file"},
            {"name": "llama-13b.toml", "type": "file"},
            {"name": "qwen-3.toml", "type": "file"},
            {"name": "traits.toml", "type": "file"},  # Should be excluded
            {"name": "some-dir", "type": "dir"},  # Should be excluded
        ]

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value.json.return_value = mock_response
            mock_instance.get.return_value.raise_for_status = MagicMock()

            result = runner.invoke(app, ["model", "search"])

        assert result.exit_code == 0
        assert "llama-7b" in result.stdout
        assert "llama-13b" in result.stdout
        assert "qwen-3" in result.stdout
        assert "traits" not in result.stdout

    def test_search_with_pattern(self, tmp_path):
        """Filters models by regex pattern."""
        mock_response = [
            {"name": "llama-7b.toml", "type": "file"},
            {"name": "llama-13b.toml", "type": "file"},
            {"name": "qwen-3.toml", "type": "file"},
        ]

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value.json.return_value = mock_response
            mock_instance.get.return_value.raise_for_status = MagicMock()

            result = runner.invoke(app, ["model", "search", "llama"])

        assert result.exit_code == 0
        assert "llama-7b" in result.stdout
        assert "llama-13b" in result.stdout
        assert "qwen-3" not in result.stdout


class TestModelInfo:
    """Tests for model info command."""

    def test_info_displays_architecture(self, tmp_path):
        """Displays model architecture information."""
        mock_toml = """
[architecture]
type = "llama"
hidden_size = 4096
num_layers = 32
num_attention_heads = 32
"""

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value.text = mock_toml
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.raise_for_status = MagicMock()

            with patch.dict(os.environ, {"PIE_HOME": str(tmp_path)}):
                result = runner.invoke(app, ["model", "info", "test-model"])

        assert result.exit_code == 0
        assert "Architecture" in result.stdout
        assert "llama" in result.stdout
        assert "4096" in result.stdout

    def test_info_shows_download_status(self, tmp_path):
        """Shows whether model is downloaded."""
        mock_toml = "[architecture]\ntype = 'test'\n"

        # Create downloaded model
        models_dir = tmp_path / "models"
        model_dir = models_dir / "downloaded-model"
        model_dir.mkdir(parents=True)
        (models_dir / "downloaded-model.toml").write_text("name = 'test'")

        with patch("httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.get.return_value.text = mock_toml
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.raise_for_status = MagicMock()

            with patch.dict(os.environ, {"PIE_HOME": str(tmp_path)}):
                result = runner.invoke(app, ["model", "info", "downloaded-model"])

        assert result.exit_code == 0
        assert "Downloaded locally" in result.stdout
