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

Run Pie server in Docker and connect to it using the `pie-cli` client.

**Prerequisites:**
- NVIDIA GPU with Docker and NVIDIA Container Toolkit
- SSH key pair (generate with `ssh-keygen -t ed25519` if needed)
- `pie-cli` binary ([download from GitHub releases](https://github.com/pie-project/pie/releases))

**Step 1: Start Pie Server**

```bash
docker run --rm --gpus all -p 8080:8080 \
  --name pie-server \
  -e PIE_AUTH_USER="$(whoami)" \
  -e PIE_AUTH_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  -v ~/.cache:/root/.cache \
  pieproject/pie:latest
```

The server will start with the name `pie-server` and authenticate using your SSH public key. Models are cached in `~/.cache/` for persistence.

**Step 2: Configure and Test Connection**

In a new terminal:

```bash
# Configure pie-cli (uses localhost:8080 by default)
pie-cli config init --enable-auth true

# Test connection
pie-cli ping
```

**Step 3: Run Text Completion**

Copy the example inferlet and run it:

```bash
# Copy inferlet from container (one-time)
docker cp pie-server:/workspace/sdk/inferlet-examples/text_completion.wasm ./

# Submit for execution
pie-cli submit ./text_completion.wasm -- --prompt "What is the capital of France?"
```

**Note:** The first run may take a few minutes for model download and kernel compilation. Subsequent runs are much faster thanks to caching.

### Manual Installation

#### Prerequisites

- **Configure a Backend (Python):**
  
  Navigate to `engine/backend-python` and install:
  ```bash
  cd engine/backend-python
  pip install -e .
  ```
  
  For CUDA support:
  ```bash
  pip install -e ".[cuda]"
  ```
  
  For Metal (macOS) support:
  ```bash
  pip install -e ".[metal]"
  ```

  To verify the installation and environment compatibility, run:
  ```bash
  pie-backend --doctor
  ```


- **Add Wasm Target:**
  Install the WebAssembly target for Rust:

  ```bash
  rustup target add wasm32-wasip2
  ```
  This is required to compile Rust-based inferlets in the `sdk/inferlet-examples` directory.


#### Step 1: Build

Build the **CLIs** and the example inferlets.

1. **Build the engine `pie` and the client CLI `pie-cli`:**

   From the repository root, run

   ```bash
   cd runtime && cargo install --path .
   ```

   Also, from the repository root, run
   ```bash
   cd client/cli && cargo install --path .
   ```

2. **Build the Examples:**

   From the repository root, run
   ```bash
   cd sdk/inferlet-examples && cargo build --target wasm32-wasip2 --release
   ```

#### Step 2: Configure Engine and Backend

1. **Create default configuration file:**

   Substitute `$REPO` to the actual repository root and run
   ```bash
   pie config init
   ```

2. **Download the model:**

   The default config file specifies the expected model. Run the following command to download it.
   ```bash
   pie model add qwen-3-0.6b
   ```

3. **Test the engine:**

   Run an inferlet directly with the engine. Due to JIT compilation of FlashInfer kernels, the first run will have **very long** latency.
   ```bash
   pie run \
       $REPO/sdk/inferlet-examples/target/wasm32-wasip2/release/text_completion.wasm \
       -- \
       --prompt "Where is the capital of France?"
   ```

#### Step 3: Run an Inferlet from a User Client

1. **Create User Public Key:**

   If you don't already have a key pair in `~/.ssh`, generate one with the following command.
   By default, the private key will be generated in `~/.ssh/id_ed25519` and the public key in `~/.ssh/id_ed25519.pub`.
   Please make sure the passphrase is *empty*.
   ```bash
   ssh-keygen
   ```

   In addition to ED25519, you can also use RSA or ECDSA keys.

2. **Create default user client configuration file:**

   The following command creates a default user client configuration file using the current Unix username and the private key in `~/.ssh`.
   ```bash
   pie-cli config init
   ```

3. **Register the user on the engine:**

   Run the following command to register the current user on the engine.
   `my-first-key` is the name of the key and can be any string.
   `cat` reads the public key from `~/.ssh/id_ed25519.pub` and pipes it to `pie auth add`.
   ```bash
   cat ~/.ssh/id_ed25519.pub | pie auth add $(whoami) my-first-key
   ```

4. **Start the Engine:**

   Launch the Pie engine with the default configuration.
   ```bash
   pie serve
   ```

5. **Run an Inferlet:**

   From another terminal window, run
   ```bash
   pie-cli submit \
       $REPO/sdk/inferlet-examples/target/wasm32-wasip2/release/text_completion.wasm \
       -- \
       --prompt "Where is the capital of France?"
   ```
