#!/usr/bin/env python3
"""
Simple CLI helper for interacting with Qwen3 backend management API.

This script demonstrates how the pie controller or CLI tools would interact
with the backend management API.
"""

import argparse
import requests
import json
import sys
import time

def make_request(method, url, data=None, timeout=30):
    """Make HTTP request with error handling."""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")

        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)

def cmd_health(backend_url):
    """Check backend health."""
    print("Checking backend health...")
    success, result = make_request('GET', f"{backend_url}/manage/health")

    if success:
        print("✓ Backend is healthy")
        print(f"  Backend ID: {result.get('backend_id')}")
        print(f"  Service: {result.get('service_name')}")
        print(f"  Type: {result.get('backend_type')}")
        print(f"  Loaded models: {result.get('loaded_models', [])}")
        print(f"  IPC endpoint: {result.get('ipc_endpoint')}")
        return True
    else:
        print(f"✗ Health check failed: {result}")
        return False

def cmd_load_model(backend_url, model_name, model_path=None, model_type=None):
    """Load a model."""
    print(f"Loading model: {model_name}")

    data = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type or "qwen3",
        "additional_params": {}
    }

    success, result = make_request('POST', f"{backend_url}/manage/models/load", data, timeout=120)

    if success:
        if result.get('success'):
            print(f"✓ Model loaded successfully: {result.get('message')}")
            print(f"  Device: {result.get('device')}")
            return True
        else:
            print(f"✗ Model load failed: {result.get('error')}")
            return False
    else:
        print(f"✗ Request failed: {result}")
        return False

def cmd_unload_model(backend_url, model_name):
    """Unload a model."""
    print(f"Unloading model: {model_name}")

    data = {"model_name": model_name}
    success, result = make_request('POST', f"{backend_url}/manage/models/unload", data)

    if success:
        if result.get('success'):
            print(f"✓ Model unloaded successfully: {result.get('message')}")
            return True
        else:
            print(f"✗ Model unload failed: {result.get('error')}")
            return False
    else:
        print(f"✗ Request failed: {result}")
        return False

def cmd_list_models(backend_url):
    """List loaded models."""
    print("Listing models...")

    success, result = make_request('GET', f"{backend_url}/manage/models")

    if success:
        loaded = result.get('loaded_models', {})
        supported = result.get('supported_models', [])

        print(f"✓ Models retrieved")
        print(f"  Loaded models ({len(loaded)}):")
        for name, info in loaded.items():
            load_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.get('load_time', 0)))
            print(f"    - {name} (loaded: {load_time})")
            if info.get('model_path'):
                print(f"      Path: {info['model_path']}")

        print(f"  Supported models ({len(supported)}):")
        for model in supported:
            print(f"    - {model}")

        return True
    else:
        print(f"✗ Request failed: {result}")
        return False

def cmd_terminate(backend_url):
    """Terminate the backend."""
    print("Terminating backend...")

    success, result = make_request('POST', f"{backend_url}/manage/terminate")

    if success:
        print(f"✓ Termination initiated: {result.get('message')}")
        return True
    else:
        print(f"✗ Termination failed: {result}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Qwen3 Backend Management CLI')
    parser.add_argument('--backend-url', default='http://localhost:8081',
                       help='Backend management API URL')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Health command
    subparsers.add_parser('health', help='Check backend health')

    # Load model command
    load_parser = subparsers.add_parser('load', help='Load a model')
    load_parser.add_argument('model_name', help='Model name to load')
    load_parser.add_argument('--model-path', help='Optional model path')
    load_parser.add_argument('--model-type', default='qwen3', help='Model type')

    # Unload model command
    unload_parser = subparsers.add_parser('unload', help='Unload a model')
    unload_parser.add_argument('model_name', help='Model name to unload')

    # List models command
    subparsers.add_parser('list', help='List models')

    # Terminate command
    subparsers.add_parser('terminate', help='Terminate backend')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    backend_url = args.backend_url.rstrip('/')

    # Execute command
    success = False

    if args.command == 'health':
        success = cmd_health(backend_url)
    elif args.command == 'load':
        success = cmd_load_model(backend_url, args.model_name, args.model_path, args.model_type)
    elif args.command == 'unload':
        success = cmd_unload_model(backend_url, args.model_name)
    elif args.command == 'list':
        success = cmd_list_models(backend_url)
    elif args.command == 'terminate':
        success = cmd_terminate(backend_url)
    else:
        print(f"Unknown command: {args.command}")
        return 1

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
