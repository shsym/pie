# Symphony Management CLI

A command-line interface tool for managing Pie's model backend services. The CLI provides a convenient way to start/stop the management service and load/unload AI models dynamically.

## Overview

The Symphony Management CLI (`management_cli.py`) is designed to interact with the Symphony Management Service, allowing users to:

- Start and stop the management service
- Check service status and view loaded models
- Load and unload AI models dynamically
- Monitor model uptime and endpoints

## Features

- **Service Management**: Start/stop the management service with optional daemonization
- **Model Management**: Load and unload models on-demand
- **Status Monitoring**: View service status and detailed model information

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `pyzmq` - ZeroMQ Python bindings for communication
- `python-json-logger` - For structured logging

## Configuration

The CLI uses a JSON configuration file (`config.json` by default) that defines:

- **Model Backends**: Mapping of model types to backend scripts
- **Endpoints**: ZMQ endpoints for client and CLI communication
- **Logging**: Log level and format configuration
- **Supported Models**: List of available models with their configurations

Example configuration can be found in `config.json`.

## Usage

### Basic Command Structure

```bash
python management_cli.py [--config CONFIG_FILE] <command> [command_options]
```

### Available Commands

#### Start the Management Service

```bash
# Start service in background (default)
python management_cli.py start-service

# Start service in foreground
python management_cli.py start-service --no-daemonize
```

#### Stop the Management Service

```bash
python management_cli.py stop-service
```

#### Check Service Status

```bash
python management_cli.py status
```

This displays:
- Service status and endpoints
- List of loaded models with details (name, type, endpoint, uptime)

#### Load a Model

```bash
# Load a model by name
python management_cli.py load-model "Llama-3.1-8B-Instruct"

# Load a model with custom configuration
python management_cli.py load-model "custom-model" --config /path/to/model_config.json
```

#### Unload a Model

```bash
python management_cli.py unload-model "Llama-3.1-8B-Instruct"
```

### Example Workflow

1. **Start the management service:**
   ```bash
   python management_cli.py start-service
   ```

2. **Check status to verify service is running:**
   ```bash
   python management_cli.py status
   ```

3. **Load a model:**
   ```bash
   python management_cli.py load-model "Llama-3.1-8B-Instruct"
   ```

4. **Check status to see loaded model:**
   ```bash
   python management_cli.py status
   ```

5. **Unload the model when done:**
   ```bash
   python management_cli.py unload-model "Llama-3.1-8B-Instruct"
   ```

6. **Stop the service:**
   ```bash
   python management_cli.py stop-service
   ```

## Testing

The CLI includes comprehensive unit tests using pytest:

```bash
# Run all tests
python -m pytest tests/ -v

# Run CLI-specific tests
python -m pytest tests/test_management_cli_pytest.py
```
_Note that the tests may take some time to run, especially if they involve starting/stopping the management service (e.g., integration tests)._
