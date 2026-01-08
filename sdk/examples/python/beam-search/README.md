# Beam Search Example

Demonstrates beam search decoding for higher-quality text generation.

## What is Beam Search?

Beam search is a decoding strategy that maintains multiple candidate sequences (beams) at each step, selecting the most likely overall sequences rather than greedily choosing the best token at each position.

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
pie-cli build --python example-apps-py/beam-search -o beam-search.wasm
```

## Run

Requires a running Pie engine:

```bash
# Check engine is running
pie-cli ping

# Submit with default settings (beam_size=4)
pie-cli submit beam-search.wasm -- --prompt "What is 2 + 2?"

# Submit with custom beam size
pie-cli submit beam-search.wasm -- \
    --prompt "What is 2 + 2?" \
    --beam_size 8 \
    --max_tokens 256
```

## Arguments

- `prompt`: The input prompt (default: "Explain the LLM decoding process ELI5.")
- `beam_size`: Number of candidate sequences to maintain (default: 4)
- `max_tokens`: Maximum tokens to generate (default: 128)
- `system`: System prompt (default: "You are a helpful, respectful and honest assistant.")
