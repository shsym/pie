<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://pie-project.org/images/pie-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://pie-project.org/images/pie-light.svg">
    <img alt="Pie: Programmable serving system for emerging LLM applications"
         src="https://pie-project.org/images/pie-light.svg"
         width="30%">
    <p></p>
  </picture>
</div>


**Pie** is a high-performance, programmable LLM serving system that empowers you to design and deploy custom inference logic and optimization strategies.

> **Note** 🧪
>
> This software is in a **pre-release** stage and under active development. It's recommended for testing and research purposes only.


## Getting Started

### Docker (Recommended for CUDA)

The easiest way to run Pie with CUDA support is using our pre-built Docker image.

**Prerequisites:**
- NVIDIA GPU (SM 8.0+), NVIDIA, and Docker
- Tested on Ubuntu 24.04, CUDA 12.7

**Step 1: Install NVIDIA Container Toolkit** (one-time setup)

```bash
# Run the installation script
./scripts/install_nvidia_container_toolkit.sh
```

This script will:
- Verify NVIDIA drivers are installed
- Install NVIDIA Container Toolkit
- Configure Docker to use NVIDIA runtime
- Test GPU access

**Step 2: Pull Image and Download Model**

```bash
# Option A: Pull pre-built image from Docker Hub (when available)
docker pull sslee0cs/pie:latest

# Download a model into pie-models volume
docker run --rm --gpus all -v pie-models:/root/.cache/pie sslee0cs/pie:latest \
  /workspace/pie-cli/target/release/pie model add "llama-3.2-1b-instruct"
```

**Step 3: Start PIE Engine**

```bash
# Start PIE with interactive shell (uses Python backend with flashinfer)
docker run --gpus all -it -v pie-models:/root/.cache/pie sslee0cs/pie:latest
```

This opens the PIE interactive shell with:
- **Python backend** using flashinfer for GPU acceleration
- **Model cache** mounted for persistence

**Step 4: Run Inferlets**

From within the PIE shell:

```bash
pie> run example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "What is the capital of France?"
```

### Local Installation

### 1. Prerequisites

- **Configure a Backend:**
  Navigate to a backend directory and follow its `README.md` for setup:
    - [Python Backend](backend/backend-python/README.md)


- **Add Wasm Target:**
  Install the WebAssembly target for Rust:

  ```bash
  rustup target add wasm32-wasip2
  ```
  This is required to compile Rust-based inferlets in the `example-apps` directory.


### 2. Build

Build the **PIE CLI** and the example inferlets.

- **Build the PIE CLI:**
  From the repository root, run:

  ```bash
  cd pie-cli && cargo install --path .
  ```

- **Build the Examples:**

  ```bash
  cd example-apps && cargo build --target wasm32-wasip2 --release
  ```


### 3. Run an Inferlet

Download a model, start the engine, and run an inferlet.

1. **Download a Model:**
   Use the PIE CLI to add a model from the [model index](https://github.com/pie-project/model-index):

   ```bash
   pie model add "llama-3.2-1b-instruct"
   ```

2. **Start the Engine:**
   Launch the PIE engine with an example configuration. This opens the interactive PIE shell:

   ```bash
   cd pie-cli
   pie start --config ./example_config.toml
   ```

3. **Run an Inferlet:**
   From within the PIE shell, execute a compiled inferlet:

   ```bash
   pie> run ../example-apps/target/wasm32-wasip2/release/text_completion.wasm -- --prompt "What is the capital of France?"
   ```


## Building from Source

### Building the Docker Image

Build verified CUDA/PyTorch combinations:

```bash
# Build all verified configurations
./scripts/build_docker_images.sh

# Or build manually
docker build -t pie:latest .
```

**Verified Configurations:**

| CUDA Version | PyTorch | Flashinfer | Status |
|--------------|---------|------------|--------|
| 12.6.1       | cu126   | 0.3.1      | ✅ Tested |

**Requirements:**
- NVIDIA driver supporting CUDA 12.6+ (check with `nvidia-smi`)
- Docker with NVIDIA Container Toolkit installed

**To add more CUDA versions:**
Edit `scripts/build_docker_images.sh` and add to `VERIFIED_CONFIGS` array after testing the combination.

The build process:
1. Installs CUDA development toolkit
2. Compiles CUDA backend with CMake/Ninja
3. Builds PIE CLI and inferlets with Rust/Cargo
4. Sets up Python environment with flashinfer, PyTorch, and Triton

**Note:** Initial build may take 30-60 minutes depending on your system.

### Building Locally Without Docker

Follow the [Local Installation](#local-installation) instructions above to build PIE components individually on your host system.
