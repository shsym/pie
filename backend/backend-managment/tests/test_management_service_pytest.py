#!/usr/bin/env python3
"""
Unit tests for the Symphony Management Service using pytest.
"""

import os
import sys
import json
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory to path to import management_service
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import management_service
from management_service import ManagementService, ModelInstance, ManagementCommand


@pytest.fixture
def mock_process():
    """Create a mock process fixture."""
    mock_proc = Mock()
    mock_proc.poll.return_value = None  # Process is alive
    return mock_proc


@pytest.fixture
def real_config():
    """Use the real config file for testing."""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    # Verify the config file exists
    if not os.path.exists(config_file):
        pytest.skip(f"Config file not found: {config_file}")
    
    with open(config_file) as f:
        config = json.load(f)
    
    yield config_file, config


class TestModelInstance:
    """Test the ModelInstance dataclass."""
    
    def test_model_instance_creation(self, mock_process):
        """Test ModelInstance creation."""
        instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3", 
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        
        assert instance.model_name == "Llama-3.1-8B-Instruct"
        assert instance.model_type == "llama3"
        assert instance.endpoint == "ipc:///tmp/test"
        assert instance.started_at is not None
        
    def test_is_alive(self, mock_process):
        """Test is_alive method."""
        instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        
        # Process is alive
        mock_process.poll.return_value = None
        assert instance.is_alive() is True
        
        # Process is dead
        mock_process.poll.return_value = 1
        assert instance.is_alive() is False
        
    def test_terminate(self, mock_process):
        """Test terminate method."""
        instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        
        instance.terminate()
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once_with(timeout=10)


class TestManagementCommand:
    """Test the ManagementCommand dataclass."""
    
    def test_command_creation(self):
        """Test ManagementCommand creation."""
        cmd = ManagementCommand(
            command="status",
            params={"key": "value"}
        )
        
        assert cmd.command == "status"
        assert cmd.params == {"key": "value"}
        assert cmd.correlation_id is not None
        
    def test_correlation_id_auto_generation(self):
        """Test that correlation_id is auto-generated."""
        cmd1 = ManagementCommand(command="test", params={})
        cmd2 = ManagementCommand(command="test", params={})
        
        assert cmd1.correlation_id != cmd2.correlation_id


class TestManagementService:
    """Test the ManagementService class."""
        
    def test_service_initialization(self, real_config):
        """Test service initialization."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        assert service.client_endpoint == config["endpoints"]["client_handshake"]
        assert service.cli_endpoint == config["endpoints"]["cli_management"]
        assert service.model_backends["llama3"] == "l4m_backend.py"
        
    def test_load_config_file_not_found(self):
        """Test config loading with non-existent file."""
        with pytest.raises(SystemExit):
            ManagementService(config_path="/nonexistent/config.json")
            
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON."""
        import tempfile
        temp_dir = tempfile.mkdtemp()
        bad_config_file = os.path.join(temp_dir, "bad_config.json")
        
        try:
            with open(bad_config_file, 'w') as f:
                f.write("invalid json {")
                
            with pytest.raises(SystemExit):
                ManagementService(config_path=bad_config_file)
        finally:
            if os.path.exists(bad_config_file):
                os.unlink(bad_config_file)
            os.rmdir(temp_dir)
        
    def test_generate_unique_endpoint(self, real_config):
        """Test unique endpoint generation."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        endpoint1 = service._generate_unique_endpoint()
        endpoint2 = service._generate_unique_endpoint()
        
        assert endpoint1 != endpoint2
        assert endpoint1.startswith("ipc:///tmp/symphony-model-")
        assert endpoint2.startswith("ipc:///tmp/symphony-model-")
        
    def test_get_model_type(self, real_config):
        """Test model type determination using configuration mapping."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # Test exact matches from config (case-sensitive)
        assert service._get_model_type("Llama-3.1-8B-Instruct") == "llama3"
        assert service._get_model_type("Llama-3.2-1B-Instruct") == "llama3"
        assert service._get_model_type("DeepSeek-V3-0324") == "deepseek"

        # Test similar names with different capitalization raise helpful error
        with pytest.raises(ValueError, match=r"Similar model name found. Do you mean: Llama-3.1-8B-Instruct"):
            service._get_model_type("llama-3.1-8b-instruct")
        with pytest.raises(ValueError, match=r"Similar model name found. Do you mean: DeepSeek-V3-0324"):
            service._get_model_type("DEEPSEEK-V3-0324")

        # Test unknown model raises ValueError (no similar name)
        with pytest.raises(ValueError, match="Unknown model 'unknown-model' - not found in configuration"):
            service._get_model_type("unknown-model")
        
    def test_model_type_mapping_loaded_from_config(self, real_config):
        """Test that model type mapping is correctly loaded from configuration."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # Verify mapping contains expected models from config
        expected_models = {
            "Llama-3.1-8B-Instruct": "llama3",
            "Llama-3.2-1B-Instruct": "llama3", 
            "DeepSeek-V3-0324": "deepseek",
        }
        
        for model_name, expected_type in expected_models.items():
            assert model_name in service.model_type_mapping
            assert service.model_type_mapping[model_name] == expected_type
        
    @patch('management_service.subprocess.Popen')
    def test_create_model_instance_success(self, mock_popen, real_config):
        """Test successful model instance creation."""
        import tempfile
        config_file, config = real_config
        with tempfile.TemporaryDirectory() as tmp_backend_dir:
            service = ManagementService(config_path=config_file, backend_base_path=tmp_backend_dir)

            # Mock successful process with proper stdout
            mock_process = Mock()
            mock_process.poll.return_value = None

            # Mock stdout to return empty strings (simulating no output)
            mock_stdout = Mock()
            mock_stdout.readline.side_effect = ['', '', '']  # Empty strings to end iteration
            mock_process.stdout = mock_stdout

            mock_popen.return_value = mock_process

            # Create a mock backend script in the temp dir
            backend_path = os.path.join(tmp_backend_dir, "l4m_backend.py")
            with open(backend_path, 'w') as f:
                f.write("#!/usr/bin/env python3\nprint('test backend')")

            endpoint = service._create_model_instance("Llama-3.1-8B-Instruct")

            assert endpoint is not None
            assert endpoint.startswith("ipc:///tmp/symphony-model-")
            assert "Llama-3.1-8B-Instruct" in service.model_instances

            instance = service.model_instances["Llama-3.1-8B-Instruct"]
            assert instance.model_name == "Llama-3.1-8B-Instruct"
            assert instance.endpoint == endpoint
        
    def test_create_model_instance_script_not_found(self, real_config):
        """Test model instance creation when backend script is not found.
        
        This test explicitly sets a non-existent backend path to ensure
        the 'script not found' error handling works correctly.
        """
        config_file, config = real_config
        # Explicitly set a non-existent backend path to test error handling
        service = ManagementService(config_path=config_file, backend_base_path="/nonexistent/path/to/backend")
        
        # Use a valid model name so we don't hit the unknown model error first
        endpoint = service._create_model_instance("Llama-3.1-8B-Instruct")
        assert endpoint is None
        
    def test_create_model_instance_unknown_model(self, real_config):
        """Test model instance creation with unknown model name."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # This should return None due to ValueError being caught
        endpoint = service._create_model_instance("nonexistent-model")
        assert endpoint is None
        
    def test_handle_status_command(self, real_config, mock_process):
        """Test status command handling."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # Add a mock model instance
        mock_process.poll.return_value = None
        instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        service.model_instances["Llama-3.1-8B-Instruct"] = instance
        
        command = ManagementCommand(command="status", params={})
        response = service._handle_status_command(command)
        
        assert response["success"] is True
        assert response["correlation_id"] == command.correlation_id
        assert len(response["data"]["models"]) == 1
        assert response["data"]["models"][0]["name"] == "Llama-3.1-8B-Instruct"
        
    def test_handle_load_model_command(self, real_config):
        """Test load-model command handling."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(
            command="load-model",
            params={"model_name": "Llama-3.1-8B-Instruct"}
        )
        
        with patch.object(service, '_get_or_create_model_instance') as mock_get_or_create:
            mock_get_or_create.return_value = "ipc:///tmp/test-endpoint"
            
            response = service._handle_load_model_command(command)
            
            assert response["success"] is True
            assert response["data"]["endpoint"] == "ipc:///tmp/test-endpoint"
            mock_get_or_create.assert_called_once_with("Llama-3.1-8B-Instruct", None)
            
    def test_handle_load_model_command_missing_name(self, real_config):
        """Test load-model command with missing model name."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(command="load-model", params={})
        response = service._handle_load_model_command(command)
        
        assert response["success"] is False
        assert "model_name parameter required" in response["error"]
        
    def test_handle_load_model_command_invalid_model(self, real_config):
        """Test load-model command with invalid model name."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(
            command="load-model",
            params={"model_name": "invalid-model"}
        )
        
        response = service._handle_load_model_command(command)
        
        assert response["success"] is False
        assert "Unknown model 'invalid-model' - not found in configuration" in response["error"]
        
    def test_handle_load_model_command_case_mismatch(self, real_config):
        """Test load-model command with case mismatch model name."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(
            command="load-model",
            params={"model_name": "llama-3.1-8b-instruct"}
        )
        
        response = service._handle_load_model_command(command)
        
        assert response["success"] is False
        assert "Similar model name found. Do you mean: Llama-3.1-8B-Instruct" in response["error"]
        
    def test_handle_load_model_command_ignores_config_path(self, real_config):
        """Test load-model command ignores config_path parameter."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(
            command="load-model",
            params={"model_name": "Llama-3.1-8B-Instruct", "config_path": "/some/path"}
        )
        
        with patch.object(service, '_get_or_create_model_instance') as mock_get_or_create:
            mock_get_or_create.return_value = "ipc:///tmp/test-endpoint"
            
            response = service._handle_load_model_command(command)
            
            assert response["success"] is True
            # Verify config_path is ignored (passed as None)
            mock_get_or_create.assert_called_once_with("Llama-3.1-8B-Instruct", None)

    def test_handle_unload_model_command(self, real_config):
        """Test unload-model command handling."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # Add a mock model instance
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is alive
        instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3",
            endpoint="ipc:///tmp/test",
            process=mock_process
        )
        service.model_instances["Llama-3.1-8B-Instruct"] = instance
        
        command = ManagementCommand(
            command="unload-model",
            params={"model_name": "Llama-3.1-8B-Instruct"}
        )
        
        response = service._handle_unload_model_command(command)
        
        assert response["success"] is True
        assert "Llama-3.1-8B-Instruct" not in service.model_instances
        mock_process.terminate.assert_called_once()
        
    def test_handle_unload_model_command_not_found(self, real_config):
        """Test unload-model command with non-existent model."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(
            command="unload-model",
            params={"model_name": "nonexistent-model"}
        )
        
        response = service._handle_unload_model_command(command)
        
        assert response["success"] is False
        assert "not found" in response["error"]
        
    def test_handle_stop_service_command(self, real_config):
        """Test stop-service command handling."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        command = ManagementCommand(command="stop-service", params={})
        response = service._handle_stop_service_command(command)
        
        assert response["success"] is True
        assert service.shutdown_requested is True
        
    def test_check_model_instances_health(self, real_config):
        """Test model instance health checking."""
        config_file, config = real_config
        service = ManagementService(config_path=config_file)
        
        # Add alive and dead model instances
        alive_process = Mock()
        alive_process.poll.return_value = None
        
        dead_process = Mock()
        dead_process.poll.return_value = 1
        
        alive_instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct",
            model_type="llama3",
            endpoint="ipc:///tmp/alive",
            process=alive_process
        )
        
        dead_instance = ModelInstance(
            model_name="Llama-3.1-8B-Instruct", 
            model_type="llama3",
            endpoint="ipc:///tmp/dead",
            process=dead_process
        )
        
        service.model_instances["Llama-3.1-8B-Instruct"] = alive_instance
        service.model_instances["l4m"] = dead_instance
        
        service._check_model_instances_health()
        
        # Only alive instance should remain
        assert "Llama-3.1-8B-Instruct" in service.model_instances
        assert "l4m" not in service.model_instances
