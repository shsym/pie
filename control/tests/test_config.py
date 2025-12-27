"""Tests for pie_cli.config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import toml
from typer.testing import CliRunner

from pie_cli import config
from pie_cli.cli import app

runner = CliRunner()


class TestCreateDefaultConfigContent:
    """Tests for create_default_config_content function."""

    def test_python_backend_default(self):
        """Creates config with Python backend by default."""
        content = config.create_default_config_content()
        parsed = toml.loads(content)

        assert parsed["host"] == "127.0.0.1"
        assert parsed["port"] == 8080
        assert parsed["enable_auth"] is True
        assert len(parsed["backend"]) == 1
        assert parsed["backend"][0]["backend_type"] == "python"
        assert parsed["backend"][0]["exec_path"] == "pie-backend"
        assert parsed["backend"][0]["model"] == "qwen-3-0.6b"
        assert parsed["backend"][0]["device"] == "cuda:0"
        assert parsed["backend"][0]["activation_dtype"] == "bfloat16"
        assert parsed["backend"][0]["kv_page_size"] == 16
        assert parsed["backend"][0]["max_batch_tokens"] == 10240
        assert parsed["backend"][0]["max_dist_size"] == 32
        assert parsed["backend"][0]["max_num_embeds"] == 128
        assert parsed["backend"][0]["max_num_adapters"] == 32
        assert parsed["backend"][0]["max_adapter_rank"] == 8
        assert parsed["backend"][0]["gpu_mem_utilization"] == 0.9
        assert parsed["backend"][0]["enable_profiling"] is False

    def test_dummy_backend(self):
        """Creates minimal config with dummy backend."""
        content = config.create_default_config_content("dummy")
        parsed = toml.loads(content)

        assert parsed["backend"][0]["backend_type"] == "dummy"
        # Dummy backend should not have exec_path
        assert "exec_path" not in parsed["backend"][0]


class TestConfigInit:
    """Tests for config init command."""

    def test_init_creates_config_file(self, tmp_path):
        """Creates config file at specified path."""
        config_path = tmp_path / "config.toml"

        result = runner.invoke(app, ["config", "init", "--path", str(config_path)])

        assert result.exit_code == 0
        assert config_path.exists()
        assert "Configuration file created" in result.stdout

    def test_init_with_dummy_backend(self, tmp_path):
        """Creates config with dummy backend when --dummy is specified."""
        config_path = tmp_path / "config.toml"

        result = runner.invoke(
            app, ["config", "init", "--dummy", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        parsed = toml.loads(config_path.read_text())
        assert parsed["backend"][0]["backend_type"] == "dummy"


class TestConfigShow:
    """Tests for config show command."""

    def test_show_displays_config(self, tmp_path):
        """Displays config file content."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('host = "localhost"\nport = 9000\n')

        result = runner.invoke(app, ["config", "show", "--path", str(config_path)])

        assert result.exit_code == 0
        assert "localhost" in result.stdout
        assert "9000" in result.stdout

    def test_show_error_when_missing(self, tmp_path):
        """Returns error when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"

        result = runner.invoke(app, ["config", "show", "--path", str(config_path)])

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()


class TestConfigUpdate:
    """Tests for config update command."""

    def test_update_engine_host(self, tmp_path):
        """Updates host in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[backend]]\nbackend_type = "dummy"\n'
        )

        result = runner.invoke(
            app, ["config", "update", "--host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["host"] == "0.0.0.0"

    def test_update_engine_port(self, tmp_path):
        """Updates port in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[backend]]\nbackend_type = "dummy"\n'
        )

        result = runner.invoke(
            app, ["config", "update", "--port", "9090", "--path", str(config_path)]
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["port"] == 9090

    def test_update_backend_model(self, tmp_path):
        """Updates backend model in config file."""
        config_path = tmp_path / "config.toml"
        config_path.write_text(
            'host = "127.0.0.1"\nport = 8080\n[[backend]]\nbackend_type = "python"\nmodel = "old-model"\n'
        )

        result = runner.invoke(
            app,
            [
                "config",
                "update",
                "--backend-model",
                "new-model",
                "--path",
                str(config_path),
            ],
        )

        assert result.exit_code == 0
        updated = toml.loads(config_path.read_text())
        assert updated["backend"][0]["model"] == "new-model"

    def test_update_no_options_warning(self, tmp_path):
        """Shows warning when no options provided."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('host = "127.0.0.1"\nport = 8080\n')

        result = runner.invoke(app, ["config", "update", "--path", str(config_path)])

        assert result.exit_code == 0
        assert "No configuration options provided" in result.stdout

    def test_update_error_when_missing(self, tmp_path):
        """Returns error when config file doesn't exist."""
        config_path = tmp_path / "nonexistent.toml"

        result = runner.invoke(
            app, ["config", "update", "--host", "0.0.0.0", "--path", str(config_path)]
        )

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()
