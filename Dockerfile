# Dockerfile for Pie with CUDA support
# Supports specific verified CUDA/PyTorch combinations only
# See scripts/build_docker_images.sh for supported versions

ARG CUDA_VERSION=12.6
ARG CUDA_MINOR=1
ARG PYTORCH_CUDA=cu126

FROM nvidia/cuda:${CUDA_VERSION}.${CUDA_MINOR}-devel-ubuntu24.04

# Re-declare args after FROM
ARG CUDA_VERSION
ARG PYTORCH_CUDA

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PYTHONUNBUFFERED=1 \
    PIE_HOME=/root/.cache/pie \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    PATH="/workspace/backend/backend-python/.venv/bin:/usr/local/cargo/bin:/root/.local/bin:${PATH}"

# Install all dependencies
RUN apt-get update && apt-get install -y \
    git cmake ninja-build curl wget build-essential pkg-config \
    libzmq3-dev libcbor-dev libzstd-dev \
    libssl-dev \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable \
    && . $CARGO_HOME/env \
    && rustup target add wasm32-wasip2 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace
COPY . .

# Build CUDA backend
RUN cd backend/backend-cuda && mkdir -p build && cd build \
    && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    && ninja

# Install PIE CLI globally
RUN cd pie-cli && cargo install --path .

# Build example inferlets
RUN cd example-apps && cargo build --target wasm32-wasip2 --release

# Setup Python backend with flashinfer (using verified PyTorch CUDA version)
RUN cd backend/backend-python \
    && uv venv \
    && . .venv/bin/activate \
    && uv pip install flashinfer-python==0.3.1 \
    && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA} --force-reinstall \
    && uv pip install triton \
    && uv pip install -e ".[cuda,debug]" \
    && uv pip install ninja

# Set default working directory for runtime
WORKDIR /workspace

# Use docker_config.toml with Python backend and absolute paths
CMD ["pie", "serve", "--config", "/workspace/pie-cli/docker_config.toml"]
