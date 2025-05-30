#!/usr/bin/env python3
"""
Unit tests for the Symphony Management CLI using pytest.
"""

import os
import sys
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import zmq

# Add parent directory to path to import management_cli
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from management_cli import ManagementCLI


@pytest.fixture
def real_config():
    """Use the real config file for testing."""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    # Verify the config file exists
    if not os.path.exists(config_file):
        pytest.skip(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)
    
    yield config


@pytest.fixture
def cli():
    """Create a ManagementCLI instance for testing using real config."""
    return ManagementCLI()


@pytest.fixture
def mock_zmq_socket():
    """Create a mock ZMQ socket."""
    mock_socket = Mock()
    mock_socket.recv.return_value = json.dumps({
        "success": True,
        "correlation_id": "test-id",
        "data": {}
    }).encode('utf-8')
    return mock_socket


class TestManagementCLI:
    """Test the ManagementCLI class."""
    
    def test_cli_initialization(self, real_config):
        """Test CLI initialization."""
        cli = ManagementCLI()
        assert cli.service_endpoint == real_config["endpoints"]["cli_management"]
        assert cli.context is not None
        
    def test_cli_default_endpoint(self, real_config):
        """Test CLI with default endpoint from config."""
        cli = ManagementCLI()
        assert cli.service_endpoint == real_config["endpoints"]["cli_management"]
        
    @patch('management_cli.zmq.Context')
    def test_send_command_success(self, mock_context, cli, mock_zmq_socket):
        """Test successful command sending."""
        mock_context.return_value.socket.return_value = mock_zmq_socket
        cli.context = mock_context.return_value
        
        response = cli._send_command("status", {"param": "value"})
        
        assert response["success"] is True
        assert "correlation_id" in response
        mock_zmq_socket.connect.assert_called_once()
        mock_zmq_socket.send.assert_called_once()
        mock_zmq_socket.recv.assert_called_once()
        
    @patch('management_cli.zmq.Context')
    def test_send_command_timeout(self, mock_context, cli):
        """Test command timeout handling."""
        mock_socket = Mock()
        mock_socket.recv.side_effect = zmq.Again()
        mock_context.return_value.socket.return_value = mock_socket
        cli.context = mock_context.return_value
        
        response = cli._send_command("status")
        
        assert response["success"] is False
        assert "timeout" in response["error"].lower()
        
    @patch('management_cli.zmq.Context')
    def test_send_command_exception(self, mock_context, cli):
        """Test command exception handling."""
        mock_socket = Mock()
        mock_socket.recv.side_effect = Exception("Connection error")
        mock_context.return_value.socket.return_value = mock_socket
        cli.context = mock_context.return_value
        
        response = cli._send_command("status")
        
        assert response["success"] is False
        assert "Communication error" in response["error"]
        
    def test_is_service_running_true(self, cli):
        """Test service running check when service is up."""
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": True}
            
            assert cli._is_service_running() is True
            mock_send.assert_called_once_with("status")
            
    def test_is_service_running_false(self, cli):
        """Test service running check when service is down."""
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": False}
            
            assert cli._is_service_running() is False
            
    def test_is_service_running_exception(self, cli):
        """Test service running check with exception."""
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.side_effect = Exception("Error")
            
            assert cli._is_service_running() is False
            
    def test_get_service_script_path(self, cli):
        """Test getting service script path."""
        path = cli._get_service_script_path()
        assert path.endswith("management_service.py")
        
    @patch('management_cli.subprocess.Popen')
    @patch('management_cli.os.path.exists')
    def test_start_service_success(self, mock_exists, mock_popen, cli, capsys):
        """Test successful service start."""
        mock_exists.return_value = True
        
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.side_effect = [False, True]  # Not running, then running
            
            result = cli.start_service(daemonize=True)
            
            assert result is True
            captured = capsys.readouterr()
            assert "Management service started successfully" in captured.out
            
    def test_start_service_already_running(self, cli, capsys):
        """Test starting service when already running."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
            result = cli.start_service()
            
            assert result is True
            captured = capsys.readouterr()
            assert "already running" in captured.out
            
    @patch('management_cli.os.path.exists')
    def test_start_service_script_not_found(self, mock_exists, cli, capsys):
        """Test starting service with missing script."""
        mock_exists.return_value = False
        
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = False
            
            result = cli.start_service()
            
            assert result is False
            captured = capsys.readouterr()
            assert "script not found" in captured.out
            
    def test_stop_service_success(self, cli, capsys):
        """Test successful service stop."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": True}
            
            result = cli.stop_service()
            
            assert result is True
            captured = capsys.readouterr()
            assert "stop requested" in captured.out
            
    def test_stop_service_not_running(self, cli, capsys):
        """Test stopping service when not running."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = False
            
            result = cli.stop_service()
            
            assert result is True
            captured = capsys.readouterr()
            assert "not running" in captured.out
            
    def test_stop_service_error(self, cli, capsys):
        """Test service stop with error."""
        with patch.object(cli, '_is_service_running') as mock_running, \
             patch.object(cli, '_send_command') as mock_send:
            mock_running.return_value = True
            mock_send.return_value = {"success": False, "error": "Stop failed"}
            
            result = cli.stop_service()
            
            assert result is False
            captured = capsys.readouterr()
            assert "Stop failed" in captured.out
            
    def test_status_success(self, cli, capsys):
        """Test successful status command."""
        status_data = {
            "service_status": "running",
            "endpoint": "ipc:///tmp/test",
            "models": [
                {
                    "name": "Llama-3.1-8B-Instruct",
                    "type": "llama3",
                    "endpoint": "ipc:///tmp/model-123",
                    "uptime": 300
                }
            ]
        }
        
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": True, "data": status_data}
            
            result = cli.status()
            
            assert result is True
            captured = capsys.readouterr()
            assert "Service Status: running" in captured.out
            assert "Llama-3.1-8B-Instruct" in captured.out
            assert "5 minutes" in captured.out
            
    def test_status_no_models(self, cli, capsys):
        """Test status with no models loaded."""
        status_data = {
            "service_status": "running",
            "endpoint": "ipc:///tmp/test",
            "models": []
        }
        
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": True, "data": status_data}
            
            result = cli.status()
            
            assert result is True
            captured = capsys.readouterr()
            assert "No models currently loaded" in captured.out
            
    def test_status_service_not_running(self, cli, capsys):
        """Test status when service not running."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = False
            
            result = cli.status()
            
            assert result is False
            captured = capsys.readouterr()
            assert "not running" in captured.out
            
    def test_load_model_success(self, cli, capsys):
        """Test successful model loading."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {
                "success": True, 
                "data": {"endpoint": "ipc:///tmp/model-123"}
            }
            
            result = cli.load_model("Llama-3.1-8B-Instruct")
            
            assert result is True
            captured = capsys.readouterr()
            assert "Model loaded successfully" in captured.out
            assert "ipc:///tmp/model-123" in captured.out
            
    def test_load_model_with_config(self, cli):
        """Test loading model with config file."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {
                "success": True,
                "data": {"endpoint": "ipc:///tmp/model-123"}
            }
            
            result = cli.load_model("Llama-3.1-8B-Instruct", "/path/to/config.json")
            
            assert result is True
            # Check that config path was passed in parameters
            call_args = mock_send.call_args
            assert call_args[0][1]["config_path"] == "/path/to/config.json"
            
    def test_load_model_service_not_running(self, cli, capsys):
        """Test loading model when service not running."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = False
            
            result = cli.load_model("test-model")
            
            assert result is False
            captured = capsys.readouterr()
            assert "not running" in captured.out
            
    def test_load_model_error(self, cli, capsys):
        """Test model loading with error."""
        with patch.object(cli, '_is_service_running') as mock_running, \
             patch.object(cli, '_send_command') as mock_send:
            mock_running.return_value = True
            mock_send.return_value = {
                "success": False,
                "error": "Model not found"
            }
            
            result = cli.load_model("test-model")
            
            assert result is False
            captured = capsys.readouterr()
            assert "Model not found" in captured.out
            
    def test_unload_model_success(self, cli, capsys):
        """Test successful model unloading."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = True
            
        with patch.object(cli, '_send_command') as mock_send:
            mock_send.return_value = {"success": True}
            
            result = cli.unload_model("Llama-3.1-8B-Instruct")
            
            assert result is True
            captured = capsys.readouterr()
            assert "Model unloaded successfully" in captured.out
            
    def test_unload_model_service_not_running(self, cli, capsys):
        """Test unloading model when service not running."""
        with patch.object(cli, '_is_service_running') as mock_running:
            mock_running.return_value = False
            
            result = cli.unload_model("test-model")
            
            assert result is False
            captured = capsys.readouterr()
            assert "not running" in captured.out

    def test_unload_model_error(self, cli, capsys):
        """Test model unloading with error."""
        with patch.object(cli, '_is_service_running') as mock_running, \
             patch.object(cli, '_send_command') as mock_send:
            mock_running.return_value = True
            mock_send.return_value = {
                "success": False,
                "error": "Model not found"
            }
            
            result = cli.unload_model("test-model")
            
            assert result is False
            captured = capsys.readouterr()
            assert "Model not found" in captured.out

    def test_cli_missing_config_file(self, tmp_path):
        """Test CLI with missing configuration file."""
        missing_config_path = str(tmp_path / "missing_config.json")
        
        with pytest.raises(SystemExit):
            ManagementCLI(config_path=missing_config_path)
    
    def test_cli_invalid_json_config(self, tmp_path):
        """Test CLI with invalid JSON configuration file."""
        invalid_config_path = tmp_path / "invalid_config.json"
        invalid_config_path.write_text("{ invalid json")
        
        with pytest.raises(SystemExit):
            ManagementCLI(config_path=str(invalid_config_path))
    
    def test_cli_missing_endpoint_in_config(self, tmp_path):
        """Test CLI with configuration file missing endpoint."""
        config_data = {
            "model_backends": {
                "llama3": "l4m_backend.py"
            }
            # Missing endpoints section
        }
        config_path = tmp_path / "no_endpoint_config.json"
        config_path.write_text(json.dumps(config_data))
        
        with pytest.raises(SystemExit):
            ManagementCLI(config_path=str(config_path))

    def test_cli_custom_config_path(self, tmp_path):
        """Test CLI with custom configuration file path."""
        config_data = {
            "endpoints": {
                "client_handshake": "ipc:///tmp/test-symphony-ipc",
                "cli_management": "ipc:///tmp/test-symphony-cli"
            },
            "model_backends": {
                "llama3": "l4m_backend.py"
            }
        }
        config_path = tmp_path / "custom_config.json"
        config_path.write_text(json.dumps(config_data))
        
        cli = ManagementCLI(config_path=str(config_path))
        assert cli.service_endpoint == "ipc:///tmp/test-symphony-cli"