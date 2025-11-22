#!/bin/bash
# Build Pie Docker images for verified CUDA/PyTorch combinations
# Only specific tested versions are supported

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Verified CUDA/PyTorch combinations
# Format: "CUDA_VERSION:CUDA_MINOR:PYTORCH_CUDA:TAG"
declare -a VERIFIED_CONFIGS=(
    "12.6:1:cu126:cuda12.6"
)

echo "=================================================="
echo "Building Pie Docker Images"
echo "=================================================="
echo ""
echo "Verified configurations:"
for config in "${VERIFIED_CONFIGS[@]}"; do
    IFS=':' read -r cuda_ver cuda_minor torch_cuda tag <<< "$config"
    echo "  - CUDA ${cuda_ver}.${cuda_minor} + PyTorch ${torch_cuda}"
    echo "    → pie:${tag}-latest"
    echo "    → pie:${tag}-dev"
done
echo ""

cd "$PROJECT_ROOT"

# Build each verified configuration
for config in "${VERIFIED_CONFIGS[@]}"; do
    IFS=':' read -r cuda_ver cuda_minor torch_cuda tag <<< "$config"

    echo "Building pie:${tag}-latest..."
    echo "→ CUDA: ${cuda_ver}.${cuda_minor}"
    echo "→ PyTorch: ${torch_cuda}"

    $SUDO docker build \
        --build-arg CUDA_VERSION=${cuda_ver} \
        --build-arg CUDA_MINOR=${cuda_minor} \
        --build-arg PYTORCH_CUDA=${torch_cuda} \
        -t pie:${tag} \
        -t pie:latest \
        .

    echo "✓ Built pie:${tag}-latest"
    echo ""

    echo "Building pie:${tag}-dev..."
    echo "→ CUDA: ${cuda_ver}.${cuda_minor}"
    echo "→ PyTorch: ${torch_cuda}"

    $SUDO docker build \
        --build-arg CUDA_VERSION=${cuda_ver} \
        --build-arg CUDA_MINOR=${cuda_minor} \
        --build-arg PYTORCH_CUDA=${torch_cuda} \
        --target development \
        -t pie:${tag}-dev \
        -t pie:dev \
        .

    echo "✓ Built pie:${tag}-dev"
    echo ""
done

echo "=================================================="
echo "Build Summary"
echo "=================================================="
echo ""
echo "Available images:"
$SUDO docker images | grep -E "^pie" || echo "No pie images found"
echo ""
echo "To run:"
echo "  Latest: $SUDO docker run --gpus all -d -p 8080:8080 -v ~/.cache:/root/.cache pie:latest"
echo "  Development: $SUDO docker run --gpus all -d -p 8080:8080 -v ~/.cache:/root/.cache pie:dev"
echo ""
echo "With authentication setup (pass SSH public key):"
echo "  $SUDO docker run --gpus all -d -p 8080:8080 \\"
echo "    -e PIE_AUTH_USER=\"myuser\" \\"
echo "    -e PIE_AUTH_KEY=\"\$(cat ~/.ssh/id_ed25519.pub)\" \\"
echo "    -v ~/.cache:/root/.cache \\"
echo "    pie:latest"
echo ""
echo "Or mount key file:"
echo "  $SUDO docker run --gpus all -d -p 8080:8080 \\"
echo "    -e PIE_AUTH_USER=\"myuser\" \\"
echo "    -e PIE_AUTH_KEY_FILE=\"/keys/id_ed25519.pub\" \\"
echo "    -v ~/.ssh/id_ed25519.pub:/keys/id_ed25519.pub:ro \\"
echo "    -v ~/.cache:/root/.cache \\"
echo "    pie:latest"
echo ""
echo "Note: Mount ~/.cache (not just ~/.cache/pie) to persist both models and FlashInfer JIT cache"
echo ""
echo "To download a model first:"
echo "  $SUDO docker run --rm --gpus all -v ~/.cache:/root/.cache pie:latest pie model add \"llama-3.2-1b-instruct\""
echo ""
echo "Build complete!"
