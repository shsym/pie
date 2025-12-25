"""Tests for pie_server CLI main commands."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from pie_server.cli import app

runner = CliRunner()


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self):
        """Shows all main commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "serve" in result.stdout
        assert "run" in result.stdout
        assert "config" in result.stdout
        assert "model" in result.stdout
        assert "auth" in result.stdout

    def test_serve_help(self):
        """Shows serve command options."""
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--no-auth" in result.stdout
        assert "--verbose" in result.stdout
        assert "--log" in result.stdout
        assert "--interactive" in result.stdout

    def test_run_help(self):
        """Shows run command options."""
        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0
        assert "INFERLET" in result.stdout.upper()
        assert "--config" in result.stdout
        assert "--log" in result.stdout

    def test_config_help(self):
        """Shows config subcommands."""
        result = runner.invoke(app, ["config", "--help"])

        assert result.exit_code == 0
        assert "init" in result.stdout
        assert "show" in result.stdout
        assert "update" in result.stdout

    def test_config_init_help(self):
        """Shows config init options."""
        result = runner.invoke(app, ["config", "init", "--help"])

        assert result.exit_code == 0
        assert "--dummy" in result.stdout
        assert "--path" in result.stdout

    def test_config_update_help(self):
        """Shows config update options - all backend options."""
        result = runner.invoke(app, ["config", "update", "--help"])

        assert result.exit_code == 0
        # Engine options
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--enable-auth" in result.stdout
        assert "--cache-dir" in result.stdout
        assert "--verbose" in result.stdout
        assert "--log" in result.stdout
        # Backend options
        assert "--backend-type" in result.stdout
        assert "--backend-exec-path" in result.stdout
        assert "--backend-model" in result.stdout
        assert "--backend-device" in result.stdout
        assert "--backend-activation-dtype" in result.stdout
        assert "--backend-weight-dtype" in result.stdout
        assert "--backend-kv-page-size" in result.stdout
        assert "--backend-max-batch-tokens" in result.stdout
        assert "--backend-max-dist-size" in result.stdout
        assert "--backend-max-num-embeds" in result.stdout
        assert "--backend-max-num-adapters" in result.stdout
        assert "--backend-max-adapter-rank" in result.stdout
        assert "--backend-gpu-mem-utilization" in result.stdout
        assert "--backend-enable-profiling" in result.stdout

    def test_model_help(self):
        """Shows model subcommands."""
        result = runner.invoke(app, ["model", "--help"])

        assert result.exit_code == 0
        assert "list" in result.stdout
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "search" in result.stdout
        assert "info" in result.stdout

    def test_auth_help(self):
        """Shows auth subcommands."""
        result = runner.invoke(app, ["auth", "--help"])

        assert result.exit_code == 0
        assert "add" in result.stdout
        assert "remove" in result.stdout
        assert "list" in result.stdout


class TestServeCommand:
    """Tests for serve command."""

    def test_serve_missing_config(self, tmp_path):
        """Returns error when config file doesn't exist."""
        with patch.dict(os.environ, {"PIE_HOME": str(tmp_path / ".pie")}):
            result = runner.invoke(app, ["serve"])

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()


class TestRunCommand:
    """Tests for run command."""

    def test_run_missing_inferlet(self, tmp_path):
        """Returns error when inferlet doesn't exist."""
        result = runner.invoke(app, ["run", str(tmp_path / "nonexistent.wasm")])

        assert result.exit_code == 1
        # Error messages go to stderr, check combined output
        assert "not found" in result.output.lower()
