<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/img/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/img/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/img/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>


**Pie** is a high-performance, programmable LLM serving system that empowers you to design and deploy custom inference logic and optimization strategies.

> **Note** 🧪
>
> This software is in a **pre-release** stage and under active development. It's recommended for testing and research purposes only.


## Getting Started

Install the Pie directly from PyPI:

```bash
# For NVIDIA GPUs
pip install "pie-server[cuda]"

# For Apple Silicon
pip install "pie-server[metal]"
```

Verify the installation:

```bash
pie doctor
```

Quick start:

```bash
# Initialize config and download a model
pie config init
pie model download Qwen/Qwen3-0.6B

# Start the engine
pie serve -i

# In the interactive shell:
# pie> run --path ./my_inferlet.wasm -- --prompt "Hello!"
```

See [control/README.md](control/README.md) for full CLI documentation.

---

### Manual Installation

#### Prerequisites

- **Install Pie Control:**
  
  ```bash
  cd control
  pip install -e ".[cuda]"   # or [metal] for macOS
  ```
  
  Verify the installation:
  ```bash
  pie doctor
  ```

- **Add Wasm Target:**
  Install the WebAssembly target for Rust:

  ```bash
  rustup target add wasm32-wasip2
  ```
  This is required to compile Rust-based inferlets in the `sdk/inferlet-examples` directory.


#### Step 1: Build Inferlets

Build the example inferlets:

```bash
cd sdk/examples && cargo build --target wasm32-wasip2 --release
```

#### Step 2: Configure Engine and Backend

1. **Create default configuration file:**

   ```bash
   pie config init
   ```

2. **Download the model:**

   ```bash
   pie model download Qwen/Qwen3-0.6B
   ```

3. **Test the engine:**

   Run an inferlet directly with the engine. Due to JIT compilation of FlashInfer kernels, the first run will have **very long** latency.
   ```bash
   pie run \
       --path sdk/examples/target/wasm32-wasip2/release/text_completion.wasm \
       -- \
       --prompt "Where is the capital of France?"
   ```

#### Step 3: Run an Inferlet from a User Client

1. **Create User Public Key:**

   If you don't already have a key pair in `~/.ssh`, generate one:
   ```bash
   ssh-keygen -t ed25519
   ```

2. **Create default user client configuration file:**

   ```bash
   pie-cli config init
   ```

3. **Register the user on the engine:**

   ```bash
   cat ~/.ssh/id_ed25519.pub | pie auth add $(whoami) my-first-key
   ```

4. **Start the Engine:**

   ```bash
   pie serve
   ```

5. **Run an Inferlet:**

   From another terminal window:
   ```bash
   pie-cli submit \
       sdk/examples/target/wasm32-wasip2/release/text_completion.wasm \
       -- \
       --prompt "Where is the capital of France?"
   ```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `pie serve` | Start the Pie engine |
| `pie run` | Run an inferlet (one-shot) |
| `pie doctor` | Environment health check |
| `pie config` | Manage configuration |
| `pie model` | Manage models |
| `pie auth` | Manage authenticated users |

For detailed usage, see [control/README.md](control/README.md).

---

## Project Structure

```
pie/
├── control/         # Pie Control CLI (Python + Rust bindings)
├── runtime/         # Core engine (Rust)
├── client/          # Client libraries (Rust CLI, Python)
├── runtime-backend/ # Python backend (PyTorch)
└── sdk/             # Inferlet SDK and examples
```
