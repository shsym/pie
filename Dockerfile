# Dockerfile for Pie with CUDA support (Multi-stage build)
# Supports specific verified CUDA/PyTorch combinations only
# See scripts/build_docker_images.sh for supported versions

ARG CUDA_VERSION=12.6
ARG CUDA_MINOR=1
ARG PYTORCH_CUDA=cu126

# ============================================================================
# Stage 1: Builder - Build all components with full development toolchain
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}.${CUDA_MINOR}-devel-ubuntu24.04 AS builder

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

# Install all build dependencies
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

# Build CUDA backend (disabled - using Python backend only)
# RUN cd backend/backend-cuda && mkdir -p build && cd build \
#     && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
#     && ninja

# Install PIE CLI globally
RUN cd pie && cargo install --path .

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

# ============================================================================
# Stage 2: Development - Keep full builder stage for development
# ============================================================================
FROM builder AS development

# Copy entrypoint script for auth setup
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set entrypoint to handle auth setup before starting server
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command: start PIE server for development (allows pie-cli connections)
CMD ["pie", "serve", "--config", "/workspace/pie/docker_config.toml"]

# ============================================================================
# Stage 3: Runtime - Use devel image for FlashInfer JIT compilation support
# ============================================================================
# Note: FlashInfer requires CUDA development tools (nvcc, headers) for runtime
# JIT compilation. Using devel base is simpler, reliable, and easy to maintain,
# comapred with manually installing specific -dev packages which have complex
# version dependencies.
FROM nvidia/cuda:${CUDA_VERSION}.${CUDA_MINOR}-devel-ubuntu24.04

# Re-declare args after FROM
ARG CUDA_VERSION
ARG PYTORCH_CUDA

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIE_HOME=/root/.cache/pie \
    TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    PATH="/workspace/backend/backend-python/.venv/bin:/usr/local/bin:${PATH}"

# Install only runtime dependencies (CUDA dev tools already in devel base)
RUN apt-get update && apt-get install -y \
    python3.12 python3-pip python3.12-venv \
    libzmq5 libcbor0.10 libzstd1 libssl3 \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy PIE CLI binary and uv from builder
COPY --from=builder /usr/local/cargo/bin/pie /usr/local/bin/pie
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv

# Copy CUDA backend binary (disabled - using Python backend only)
# COPY --from=builder /workspace/backend/backend-cuda/build/bin/pie_cuda_be /workspace/backend/backend-cuda/build/bin/pie_cuda_be

# Copy Python virtual environment
COPY --from=builder /workspace/backend/backend-python/.venv /workspace/backend/backend-python/.venv

# Copy Python backend source code (exclude cache, build, and temp files)
COPY --from=builder /workspace/backend/backend-python/ /workspace/backend/backend-python/
RUN find /workspace/backend/backend-python -name "__pycache__" -type d -exec rm -rf {} + || true && \
    find /workspace/backend/backend-python -name "*.pyc" -delete || true && \
    rm -rf /workspace/backend/backend-python/build || true

# Copy example inferlets
COPY --from=builder /workspace/example-apps/target/wasm32-wasip2/release/*.wasm /workspace/example-apps/

# Copy configuration file
COPY --from=builder /workspace/pie/docker_config.toml /workspace/pie/docker_config.toml

# Copy entrypoint script for auth setup
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set entrypoint to handle auth setup before starting server
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command: start PIE server
CMD ["pie", "serve", "--config", "/workspace/pie/docker_config.toml"]
