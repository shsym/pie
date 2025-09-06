
# Piggybacked Evolution (Pevo) on Pie

This summarizes the steps to set up and run Piggybacked Evolution (Pevo) using the Pie inference engine.

### Step 1: Install the FlashInfer Library


Go to `backend/backend-python` directory for the following steps.



1.  **Determine your GPU's CUDA Compute Capability.**
    Run the following command to find your GPU's compute capability version. You will need this for the next step.

    ```bash
    nvidia-smi --query-gpu=compute_cap --format=csv
    ```

    This will output a version number, for example, `8.6`.

2.  **Clone, Configure, and Build FlashInfer.**
    The following script clones the FlashInfer repository and builds it. **Important:** Before running, replace `"8.6"` in the `export TORCH_CUDA_ARCH_LIST` line with the compute capability you found in the previous step.

    ```bash
    pip install torch # Ensure you have the PyTorch installed
    
    # Clone the repository and its submodules
    git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
    cd flashinfer

    # IMPORTANT: Set your target CUDA architecture below
    export TORCH_CUDA_ARCH_LIST="8.6" # Replace "8.6" with your compute_cap value

    # Build the Ahead-of-Time (AOT) kernels. This will take some time (~5-10 minutes).
    python -m flashinfer.aot

    # Build the wheel package
    python -m pip install --no-build-isolation --verbose .

    cd ../
    ```

### Step 2: Compile Protobuf Definitions

This step uses the Protocol Buffers compiler (`protoc`) to generate the necessary Python code for data serialization and communication within the backend.

```bash
# Make the script executable
chmod +x build_proto.sh

# Run the script
./build_proto.sh
```

### Step 3: Install the PIE Torch Backend

Now, you can install the dependencies.

```bash
pip install -r requirements.txt
```

### Step 4: Install the PIE Client

Go to `client/python` directory for the following steps.

```bash
pip install -e .
```


### Step 5. Install the CLI 

The pie-cli is the primary tool for managing the PIE system. If you don't have Rust installed, please follow the [Rust installation guide](https://www.rust-lang.org/tools/install).


```
cd pie-cli
cargo install --path .
```

Verify the installation by checking the help message:

```bash
pie --help
```

### Step 6: Download and Register LLMs

Next, download the models used in our examples
Download all the models will quite a bit of time and ~70 GB of disk space.

```bash
pie model add "llama-3.2-1b-instruct"
pie model add "llama-3.2-3b-instruct"
pie model add "llama-3.1-8b-instruct"
pie model add "qwen-3-1.7b"
pie model add "qwen-3-4b"
pie model add "qwen-3-8b"
```


You can list all registered models to confirm they were added correctly:

```bash
pie model list
```

### Step 7: Add the WebAssembly Target to compile Inferlets


1.  **Add the `wasm32-wasip2` target to your Rust toolchain:**
    ```bash
    rustup target add wasm32-wasip2
    ```


### Step 8: Setup the wandb for logging

```
pip install wandb
wandb login
```

### Step 9. Quick Sanity Check for Multi-GPU Training

For a quick sanity check, run the training with a single GPU:

```
cd pie-cli
pie start --config ./example_config.toml
```

Set the `SERVER_URIS` in `main.py` to point to this server:
```
SERVER_URIS = [
    "ws://127.0.0.1:8080",
]
```


and launch the script:
```
python main.py
```


For multi-GPU training, please refer to the `start_pie.sh` for an example of launching a multi-GPU server, and `stop_pie.sh` for stopping the server.

You can simply add more SERVER_URIS in `main.py` to use more GPUs.

### Step 10. Run your own training

I will provide a more documentation for how to customize the Pevo training.

For now, you can refer to the `main.py` for the main training loop, and 
read `es-init.rs` for the ES initiaization, `es-rollout.rs` for the rollout, and `es-update.rs` for the update logic.

The actual CMA-ES implementation is located at `backend/backend-python/adapter.py`.

