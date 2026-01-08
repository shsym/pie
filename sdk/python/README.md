# inferlet-py

Python SDK for writing Pie inferlets.

## Setup (One-Time)

From the pie repository root:

```bash
# Create and activate venv
cd inferlet-py
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify
componentize-py --version
```

## Building Python Inferlets

**Always activate the venv first:**

```bash
source /path/to/pie/inferlet-py/.venv/bin/activate
```

Then build:

```bash
pie-cli build --python <input> -o <output.wasm>
```

### Example

```bash
# From pie root, with venv activated
pie-cli build --python example-apps-py/text-completion -o text-completion.wasm

# Run it (requires running Pie engine; check the README in the repo root)
pie-cli ping    # Check engine is up
pie-cli submit text-completion.wasm -- --prompt "What is Python?"
```

## Writing an Inferlet

Create `main.py`:

```python
from inferlet_py import Context, get_auto_model, get_arguments, send

args = get_arguments()
prompt = args.get("prompt", "Hello!")

model = get_auto_model()

with Context(model) as ctx:
    ctx.system("You are a helpful assistant.")
    ctx.user(prompt)

    for token in ctx.generate(stream=True):
        send(token, streaming=True)
```

### Beam Search Example

```python
from inferlet_py import Context, get_auto_model, get_arguments, set_return

args = get_arguments()
prompt = args.get("prompt", "Hello!")

model = get_auto_model()

with Context(model) as ctx:
    ctx.system("You are a helpful assistant.")
    ctx.user(prompt)

    # Generate with beam search for higher quality output
    result = ctx.generate_with_beam(beam_size=4, max_tokens=256)
    set_return(result)
```

## API Reference

### Runtime
- `get_version()` - Get Pie runtime version
- `get_instance_id()` - Get unique instance ID
- `get_arguments()` - Get CLI arguments as dict
- `set_return(value)` - Set return value

### Messaging
- `send(message, streaming=False)` - Send output
- `receive()` - Receive input
- `broadcast(topic, message)` - Broadcast to topic

### Model
- `get_auto_model()` - Get default model
- `get_model(service_id)` - Get specific model
- `get_all_models()` - List available models

### Context
- `system(content)` - Add system message
- `user(content)` - Add user message
- `assistant(content)` - Add assistant message
- `generate(...)` - Generate text (supports streaming)
- `generate_with_beam(beam_size, max_tokens, stop)` - Generate with beam search

## Limitations

Python inferlets run in WASM. These are **not available**:
- Network libraries (requests, httpx)
- Native extensions (numpy, pandas)
- Threading/multiprocessing
- File system (limited)

Check compatibility:

```bash
python scripts/validate_imports.py <your-app>/
```
