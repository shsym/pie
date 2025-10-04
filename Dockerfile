# Dockerfile for Pie with CUDA and Python backend support
# Build: docker build -t pie:latest .
# Run:   docker run --gpus all -it pie:latest

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH="/usr/local/cargo/bin:/root/.local/bin:/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PIE_HOME=/root/.cache/pie

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

# Build PIE CLI
RUN cd pie-cli && cargo build --release

# Build example inferlets
RUN cd example-apps && cargo build --target wasm32-wasip2 --release

# Setup Python backend with flashinfer
RUN cd backend/backend-python \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && uv pip install flashinfer-python==0.3.1 \
    && uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall \
    && uv pip install triton \
    && uv pip install -e ".[cuda,debug]"

CMD ["/workspace/pie-cli/target/release/pie", "--help"]
