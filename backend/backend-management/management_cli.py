#!/usr/bin/env python3
"""
Symphony Management CLI Tool

A command-line interface for interacting with the Symphony Management Service.
"""

import os
import sys
import json
import time
import uuid
import argparse
import subprocess
from typing import Dict, Any, Optional

import zmq


class ManagementCLI:
    """CLI tool for interacting with the Symphony Management Service."""
    
    def __init__(self, config_path: str = None):
        # Load endpoint from config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            service_endpoint = config["endpoints"]["cli_management"]
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            print("Please provide a valid configuration file using --config")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Missing required configuration key {e} in {config_path}")
            print("Please ensure the configuration file contains 'endpoints.cli_management'")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in configuration file {config_path}: {e}")
            sys.exit(1)
        
        self.service_endpoint = service_endpoint
        self.context = zmq.Context()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup ZMQ context."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up ZMQ context."""
        if hasattr(self, 'context') and self.context:
            try:
                # Use a short linger to avoid hanging
                self.context.setsockopt(zmq.LINGER, 100)  # 100ms linger
                self.context.term()
            except Exception as e:
                # Log the error but don't raise it to avoid breaking cleanup
                print(f"Warning: Error terminating ZMQ context: {e}")
            finally:
                self.context = None
        
    def _send_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a command to the management service and get response."""
        if params is None:
            params = {}
            
        # Create command
        cmd_data = {
            "command": command,
            "params": params,
            "correlation_id": str(uuid.uuid4())
        }
        
        # Connect to service
        dealer_socket = self.context.socket(zmq.DEALER)
        dealer_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout
        dealer_socket.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout
        dealer_socket.setsockopt(zmq.LINGER, 0)       # Don't wait for pending messages on close
        
        try:
            dealer_socket.connect(self.service_endpoint)
            
            # Send command (DEALER sends clean frames)
            dealer_socket.send(json.dumps(cmd_data).encode('utf-8'))
            
            # Receive response
            response_data = dealer_socket.recv()
            response = json.loads(response_data.decode('utf-8'))
            
            return response
            
        except zmq.Again:
            return {"success": False, "error": "Service not responding (timeout)"}
        except Exception as e:
            return {"success": False, "error": f"Communication error: {e}"}
        finally:
            dealer_socket.close()
    
    def _is_service_running(self) -> bool:
        """Check if the management service is running."""
        try:
            response = self._send_command("status")
            return response.get("success", False)
        except:
            return False
    
    def _get_service_script_path(self) -> str:
        """Get the path to the management service script."""
        current_dir = os.path.dirname(__file__)
        return os.path.join(current_dir, "management_service.py")
    
    def start_service(self, daemonize: bool = False) -> bool:
        """Start the management service if not already running."""
        if self._is_service_running():
            print("Management service is already running")
            return True
        
        service_script = self._get_service_script_path()
        if not os.path.exists(service_script):
            print(f"Error: Management service script not found at {service_script}")
            return False
        
        print("Starting management service...")
        
        try:
            if daemonize:
                # Start as background process
                with open(os.devnull, 'w') as devnull:
                    subprocess.Popen([sys.executable, service_script],
                                   stdout=devnull, stderr=devnull)
            else:
                # Start in foreground
                subprocess.run([sys.executable, service_script])
            
            # Give it a moment to start
            time.sleep(2)
            
            if self._is_service_running():
                print("Management service started successfully")
                return True
            else:
                print("Failed to start management service")
                return False
                
        except Exception as e:
            print(f"Error starting management service: {e}")
            return False
    
    def stop_service(self) -> bool:
        """Stop the management service."""
        if not self._is_service_running():
            print("Management service is not running")
            return True
        
        print("Stopping management service...")
        response = self._send_command("stop-service")
        
        if response.get("success"):
            print("Management service stop requested")
            return True
        else:
            print(f"Error stopping service: {response.get('error', 'Unknown error')}")
            return False
    
    def status(self) -> bool:
        """Get status of the management service and loaded models."""
        if not self._is_service_running():
            print("Management service is not running")
            return False
        
        response = self._send_command("status")
        
        if not response.get("success"):
            print(f"Error getting status: {response.get('error', 'Unknown error')}")
            return False
        
        data = response.get("data", {})
        
        print(f"Service Status: {data.get('service_status', 'unknown')}")
        print(f"Client Endpoint: {data.get('client_endpoint', 'unknown')}")
        print(f"CLI Endpoint: {data.get('cli_endpoint', 'unknown')}")
        print()
        
        models = data.get("models", [])
        if models:
            print("Loaded Models:")
            print("-" * 60)
            for model in models:
                uptime_mins = int(model.get('uptime', 0) / 60)
                print(f"  Name: {model.get('name')}")
                print(f"  Type: {model.get('type')}")
                print(f"  Endpoint: {model.get('endpoint')}")
                print(f"  Uptime: {uptime_mins} minutes")
                print()
        else:
            print("No models currently loaded")
        
        return True
    
    def load_model(self, model_name: str, config_path: Optional[str] = None) -> bool:
        """Load a model."""
        if not self._is_service_running():
            print("Management service is not running. Start it first with 'start-service'")
            return False
        
        params = {"model_name": model_name}
        if config_path:
            params["config_path"] = config_path
        
        print(f"Loading model: {model_name}")
        response = self._send_command("load-model", params)
        
        if response.get("success"):
            endpoint = response.get("data", {}).get("endpoint")
            print(f"Model loaded successfully on endpoint: {endpoint}")
            return True
        else:
            print(f"Error loading model: {response.get('error', 'Unknown error')}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        if not self._is_service_running():
            print("Management service is not running")
            return False
        
        params = {"model_name": model_name}
        
        print(f"Unloading model: {model_name}")
        response = self._send_command("unload-model", params)
        
        if response.get("success"):
            print(f"Model unloaded successfully")
            return True
        else:
            print(f"Error unloading model: {response.get('error', 'Unknown error')}")
            return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Symphony Management CLI")
    parser.add_argument("--config", 
                       help="Path to configuration file (default: config.json in script directory)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start-service command
    start_parser = subparsers.add_parser("start-service", 
                                        help="Start the management service")
    start_parser.add_argument("--daemonize", action="store_true",
                             help="Run service in background", default=True)
    
    # stop-service command
    subparsers.add_parser("stop-service", 
                         help="Stop the management service")
    
    # status command
    subparsers.add_parser("status", 
                         help="Get service status and loaded models")
    
    # load-model command
    load_parser = subparsers.add_parser("load-model", 
                                       help="Load a model")
    load_parser.add_argument("model_name", 
                            help="Name or type of model to load")
    load_parser.add_argument("--config", 
                            help="Path to model configuration file")
    
    # unload-model command
    unload_parser = subparsers.add_parser("unload-model", 
                                         help="Unload a model")
    unload_parser.add_argument("model_name", 
                              help="Name of model to unload")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create CLI instance
    with ManagementCLI(config_path=args.config) as cli:
        # Execute command
        success = True
        
        try:
            if args.command == "start-service":
                success = cli.start_service(daemonize=args.daemonize)
            elif args.command == "stop-service":
                success = cli.stop_service()
            elif args.command == "status":
                success = cli.status()
            elif args.command == "load-model":
                success = cli.load_model(args.model_name, args.config)
            elif args.command == "unload-model":
                success = cli.unload_model(args.model_name)
            else:
                print(f"Unknown command: {args.command}")
                success = False
                
        except KeyboardInterrupt:
            print("\nOperation cancelled")
            success = False
        except Exception as e:
            print(f"Error: {e}")
            success = False
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
