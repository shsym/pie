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

### Docker Installation

The easiest way to run Pie with CUDA support is using our pre-built Docker image.

**Prerequisites:**
- NVIDIA GPU (SM 8.0+), NVIDIA, and Docker
- Tested on Ubuntu 24.04, CUDA 12.7
- Install NVIDIA Container Toolkit

**Step 1: Pull Image and Download Model**

```bash
docker pull pieproject/pie:latest
mkdir -p ~/.cache
docker run --rm --gpus all -it -v ~/.cache:/root/.cache pieproject/pie:latest \
  pie model add "llama-3.2-1b-instruct"
```

- Models are downloaded into `~/.cache/pie/models/` and persist across container runs.

- FlashInfer's JIT-compiled kernels are cached in `~/.cache/flashinfer/` to avoid recompilation.


**Step 2: Run an Inferlet**
```bash
docker run --gpus all --rm -it -v ~/.cache:/root/.cache pieproject/pie:latest \
  pie run --config /workspace/pie/docker_config.toml \
  /workspace/example-apps/text_completion.wasm -- --prompt "What is the capital of France?"
```

Note that the very first inferlet response may take a few minutes due to the JIT compilation of FlashInfer.

### Manual Installation

#### Prerequisites

- **Configure a Backend:**
  Navigate to a backend directory and follow its `README.md` for setup:
    - [Python Backend](backend/backend-python/README.md)


- **Add Wasm Target:**
  Install the WebAssembly target for Rust:

  ```bash
  rustup target add wasm32-wasip2
  ```
  This is required to compile Rust-based inferlets in the `example-apps` directory.


#### Step 1: Build

Build the **CLIs** and the example inferlets.

1. **Build the engine `pie` and the client CLI `picli`:**
   From the repository root, run

   ```bash
   cd pie && cargo install --path .
   ```

   Also, from the repository root, run
   ```bash
   cd client/cli && cargo install --path .
   ```

2. **Build the Examples:**

   ```bash
   cd example-apps && cargo build --target wasm32-wasip2 --release
   ```

#### Step 2: Configure engine and backend

1. Create default configuration file (substitute `$REPO` to the actual cloned repository path)
   ```bash
   pie config init python $REPO/backend/backend-python/server.py
   ```

2. Download the model
   ```bash
   pie model add qwen-3-0.6b
   ```

#### Step 3: Run an Inferlet

1. **Start the Engine:**
   Launch the Pie engine with the default configuration

   ```bash
   pie
   ```

2. **Run an Inferlet:**
   From another terminal window, run

   ```bash
   picli submit $REPO/example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "What is the capital of France?"
   ```
