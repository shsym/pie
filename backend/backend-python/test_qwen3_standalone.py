#!/usr/bin/env python3
"""
Test script for the updated Qwen3 backend with integrated backend agent.

This script demonstrates how to start the Qwen3 backend in standalone mode
and shows the new management capabilities.
"""

import subprocess
import time
import requests
import json
import sys
import os

def test_qwen3_backend_standalone():
    """Test the Qwen3 backend in standalone mode."""
    
    print("=== Qwen3 Backend Standalone Test ===")
    
    # Backend configuration - use the model that's actually cached locally
    model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"  # Use the locally cached model
    ipc_endpoint = "ipc:///tmp/symphony-ipc-test"
    management_service_url = "http://localhost:9000"  # This may not exist yet
    backend_host = "127.0.0.1"
    backend_api_port = 8083  # Use a different port to avoid conflicts
    
    print(f"Model: {model_name}")
    print(f"IPC Endpoint: {ipc_endpoint}")
    print(f"Management API: http://{backend_host}:{backend_api_port}")
    print(f"Management Service URL: {management_service_url}")
    
    # Start the backend process
    cmd = [
        sys.executable, "qwen3_backend.py",
        "--model-name", model_name,
        "--ipc-endpoint", ipc_endpoint,
        "--management-service-url", management_service_url,
        "--backend-host", backend_host,
        "--backend-api-port", str(backend_api_port)
    ]
    
    print(f"\nStarting backend with command: {' '.join(cmd)}")
    
    try:
        # Start the backend process
        process = subprocess.Popen(
            cmd,
            cwd="/home/sslee/Workspace/symphony/backend/backend-python",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Give it some time to start
        print("Waiting for backend to start...")
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is not None:
            print(f"Backend process exited early with code: {process.poll()}")
            stdout, _ = process.communicate()
            print("Output:")
            print(stdout)
            return False
        
        # Test the management API
        print("\n=== Testing Management API ===")
        
        # Test health check
        try:
            response = requests.get(f"http://{backend_host}:{backend_api_port}/manage/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✓ Health check successful: {health_data}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Health check error: {e}")
        
        # Test model loading (this will load the model)
        try:
            load_request = {
                "model_name": model_name,
                "model_path": None,
                "model_type": "qwen3",
                "additional_params": {}
            }
            response = requests.post(
                f"http://{backend_host}:{backend_api_port}/manage/models/load", 
                json=load_request,
                timeout=60  # Model loading can take time
            )
            if response.status_code == 200:
                load_data = response.json()
                print(f"✓ Model load successful: {load_data}")
            else:
                print(f"✗ Model load failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ Model load error: {e}")
        
        # Test listing models
        try:
            response = requests.get(f"http://{backend_host}:{backend_api_port}/manage/models", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                print(f"✓ List models successful: {models_data}")
            else:
                print(f"✗ List models failed: {response.status_code}")
        except Exception as e:
            print(f"✗ List models error: {e}")
        
        print("\n=== Test completed ===")
        print("Backend is running. Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
                if process.poll() is not None:
                    print(f"Backend process exited with code: {process.poll()}")
                    break
        except KeyboardInterrupt:
            print("\nStopping backend...")
            
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False
        
    finally:
        # Clean up
        if 'process' in locals():
            try:
                # Try to terminate gracefully first
                response = requests.post(f"http://{backend_host}:{backend_api_port}/manage/terminate", timeout=5)
                print("Sent termination request")
                time.sleep(2)
            except:
                pass
            
            # Force terminate if still running
            if process.poll() is None:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
            
            # Get final output
            try:
                stdout, _ = process.communicate(timeout=5)
                if stdout:
                    print("\nFinal output:")
                    print(stdout)
            except:
                pass

if __name__ == "__main__":
    success = test_qwen3_backend_standalone()
    sys.exit(0 if success else 1)
