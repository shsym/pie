# Text Completion Example

A simple text completion inferlet using inferlet-py.

## Setup (One-Time)

From the pie repository root:

```bash
cd inferlet-py
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Build

From the pie repository root (with venv activated):

```bash
source inferlet-py/.venv/bin/activate
pie-cli build --python example-apps-py/text-completion -o text-completion.wasm
```

## Run

Requires a running Pie engine:

```bash
# Check engine is running
pie-cli ping

# Submit inferlet (arguments go after --)
pie-cli submit text-completion.wasm -- --prompt "What is Python?"
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `prompt` | "Hello, world!" | User prompt |
| `max_tokens` | 256 | Maximum tokens to generate |
| `system` | "You are a helpful assistant." | System prompt |
