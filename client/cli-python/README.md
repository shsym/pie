# Pie CLI (Python)

A Python port of the Pie command-line interface for managing inferlets.

## Installation

First, install the `pie-client` dependency:

```bash
# From the pie repository root
pip install -e client/python
```

Then install the CLI:

```bash
pip install -e client/cli-python
```

## Usage

```bash
# Initialize configuration
pie-cli config init

# Check server status
pie-cli ping

# Create a new inferlet project
pie-cli create my-inferlet

# Build an inferlet
pie-cli build my-inferlet -o my-inferlet.wasm

# Submit an inferlet
pie-cli submit my-inferlet.wasm

# List running instances
pie-cli list

# Attach to an instance
pie-cli attach <instance-id-prefix>

# Terminate an instance
pie-cli abort <instance-id-prefix>
```

## Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new JS/TS inferlet project |
| `build` | Build JS/TS inferlet into WASM component |
| `submit` | Submit inferlet to running Pie engine |
| `ping` | Check if Pie engine is alive |
| `list` | List running inferlet instances |
| `attach` | Attach to running instance & stream output |
| `abort` | Terminate running instance |
| `config init` | Create default configuration file |
| `config update` | Update configuration fields |
| `config show` | Display configuration contents |

## Requirements

- Python 3.10+
- Node.js 18+ (for `build` command)
- `pie-client` Python package

## Signal Handling

When attached to an instance:
- **Ctrl-C**: Terminates the inferlet on the server
- **Ctrl-D**: Detaches from the inferlet (continues running on server)
