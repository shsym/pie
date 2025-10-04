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
- NVIDIA GPU (SM 8.0+): RTX 30/40 series, A100, H100, etc.
- NVIDIA drivers installed
- Docker installed

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

**Step 2: Run Pie**

```bash
# Pull the image from Docker Hub
docker pull pieproject/pie:latest

# Run PIE CLI interactively
docker run --gpus all -it pieproject/pie:latest

# Download a model
docker run --gpus all -v pie-models:/root/.cache/pie pieproject/pie:latest \
  /workspace/pie-cli/target/release/pie model add "llama-3.2-1b-instruct"

# Run an inferlet
docker run --gpus all -v pie-models:/root/.cache/pie pieproject/pie:latest \
  /workspace/pie-cli/target/release/pie run /workspace/example-apps/target/wasm32-wasip2/release/text_completion.wasm \
  -- --prompt "What is the capital of France?"
```

The Docker image includes:
- Pre-compiled CUDA backend (SM 80, 86, 89, 90)
- Python backend with flashinfer JIT compilation support
- PIE CLI and example inferlets
- Full CUDA development toolkit for runtime kernel compilation

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

To build the Docker image locally:

```bash
# Build the image (includes CUDA backend, Python backend with flashinfer, PIE CLI)
docker build -t pie:latest .

# Run it
docker run --gpus all -it pie:latest
```

The build process:
1. Installs CUDA 12.8 development toolkit
2. Compiles CUDA backend with CMake/Ninja
3. Builds PIE CLI and inferlets with Rust/Cargo
4. Sets up Python environment with flashinfer, PyTorch, and Triton

**Note:** Initial build may take 30-60 minutes depending on your system.

### Building Locally Without Docker

Follow the [Local Installation](#local-installation) instructions above to build PIE components individually on your host system.

