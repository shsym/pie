#!/usr/bin/env python3
"""
Integration tests for Symphony Management Service and CLI using pytest.
"""

import os
import sys
import json
import time
import threading
import pytest
from unittest.mock import Mock, patch
import zmq

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from management_service import ManagementService
from management_cli import ManagementCLI


@pytest.fixture
def real_config():
    """Use the real config file for integration testing."""
    config_file = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    
    # Verify the config file exists
    if not os.path.exists(config_file):
        pytest.skip(f"Config file not found: {config_file}")
    
    yield config_file


class TestIntegration:
    """Integration tests for Management Service and CLI."""
    
    def test_service_cli_status_integration(self, real_config):
        """Test service and CLI status integration."""
        service = ManagementService(config_path=real_config)
        cli = ManagementCLI(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                result = service.handle_single_message(timeout=1000)  # Increase timeout
                if result:
                    print(f"Handled message: {result}")
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service more time to initialize
        time.sleep(0.5)
        
        try:
            # Test CLI status command with debugging
            print("Testing if service is running...")
            is_running = cli._is_service_running()
            print(f"Service running status: {is_running}")
            assert is_running is True
            
            result = cli.status()
            assert result is True
            
        finally:
            service.shutdown_requested = True
            time.sleep(0.2)  # Give time for cleanup
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.2)
    
    @patch('management_service.subprocess.Popen')
    def test_service_cli_load_model_integration(self, mock_popen, real_config):
        """Test service and CLI load model integration."""
        # Mock the subprocess for backend creation
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        
        # Mock stdout to return empty strings (simulating no output)
        mock_stdout = Mock()
        mock_stdout.readline.side_effect = ['', '', '']  # Empty strings to end iteration
        mock_process.stdout = mock_stdout
        
        mock_popen.return_value = mock_process
        
        service = ManagementService(config_path=real_config)
        cli = ManagementCLI(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Test CLI load model command using a backend from real config
            result = cli.load_model("Llama-3.1-8B-Instruct")
            assert result is True
            
            # Verify model was loaded in service
            assert "Llama-3.1-8B-Instruct" in service.model_instances
            
            # Test CLI status shows loaded model
            result = cli.status()
            assert result is True
            
            # Test CLI unload model command
            result = cli.unload_model("Llama-3.1-8B-Instruct")
            assert result is True
            
            # Verify model was unloaded
            assert "Llama-3.1-8B-Instruct" not in service.model_instances
            
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
    
    @patch('management_service.subprocess.Popen')
    def test_service_cli_load_invalid_model_integration(self, mock_popen, real_config):
        """Test service and CLI load model integration with invalid model name."""
        service = ManagementService(config_path=real_config)
        cli = ManagementCLI(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Test CLI load model command with invalid model name
            result = cli.load_model("invalid-model")
            assert result is False
            
            # Test CLI load model command with similar but wrong case
            result = cli.load_model("llama-3.1-8b-instruct")
            assert result is False
            
            # Verify no models were loaded
            assert len(service.model_instances) == 0
            
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
    
    def test_service_cli_stop_integration(self, real_config):
        """Test service and CLI stop integration."""
        service = ManagementService(config_path=real_config)
        cli = ManagementCLI(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        # Verify service is running
        assert cli._is_service_running() is True
        
        # Stop service via CLI
        result = cli.stop_service()
        assert result is True
        
        # Give time for shutdown
        time.sleep(0.5)
        
        # Verify service is stopped
        assert service.shutdown_requested is True
    
    def test_multiple_cli_commands(self, real_config):
        """Test multiple CLI commands in sequence."""
        service = ManagementService(config_path=real_config)
        cli = ManagementCLI(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Test multiple status calls
            for _ in range(3):
                result = cli.status()
                assert result is True
                time.sleep(0.1)
            
            # Test unknown command handling (should not crash service)
            response = cli._send_command("unknown-command")
            assert response["success"] is False
            assert "Unknown command" in response.get("error", "")
            
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
    
    def test_service_resilience(self, real_config):
        """Test service resilience to malformed CLI requests."""
        # Load config to get CLI endpoint
        with open(real_config) as f:
            config = json.load(f)
        
        service = ManagementService(config_path=real_config)
        
        # Initialize service sockets without starting main loop
        assert service.initialize_sockets() is True
        
        # Create a background thread to handle messages
        def message_handler():
            while not service.shutdown_requested:
                service.handle_single_message(timeout=100)
        
        handler_thread = threading.Thread(target=message_handler, daemon=True)
        handler_thread.start()
        
        # Give service time to initialize
        time.sleep(0.1)
        
        try:
            # Send malformed JSON to CLI endpoint
            context = zmq.Context()
            req_socket = context.socket(zmq.REQ)
            req_socket.setsockopt(zmq.RCVTIMEO, 500)
            
            try:
                req_socket.connect(config["endpoints"]["cli_management"])
                
                # Send invalid JSON
                req_socket.send(b"invalid json")
                
                # Service should handle gracefully and not crash
                time.sleep(0.1)
                
                # Verify service is still running by sending valid request
                cli = ManagementCLI(config_path=real_config)
                result = cli.status()
                assert result is True
                
            except zmq.Again:
                # Timeout is expected for malformed request
                pass
            finally:
                req_socket.close()
                context.term()
                
        finally:
            # Stop service
            service.shutdown_requested = True
            time.sleep(0.1)
