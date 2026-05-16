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

# Verified CUDA combinations.
# driver-cuda has no PyTorch/Python dep (flashinfer AOT, no JIT runtime),
# so the historical "PYTORCH_CUDA" field is unused; left as `none` for
# the existing parser. Format: "CUDA_VERSION:CUDA_MINOR:PYTORCH_CUDA:TAG"
declare -a VERIFIED_CONFIGS=(
    "12.9:0:none:cuda12.9"
)

echo "=================================================="
echo "Building Pie Docker Images"
echo "=================================================="
echo ""
echo "Verified configurations:"
for config in "${VERIFIED_CONFIGS[@]}"; do
    IFS=':' read -r cuda_ver cuda_minor torch_cuda tag <<< "$config"
    echo "  - CUDA ${cuda_ver}.${cuda_minor}"
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

    $SUDO docker build \
        -f Dockerfile.cuda \
        --target runtime \
        -t pie:${tag} \
        -t pie:latest \
        .

    echo "✓ Built pie:${tag}-latest"
    echo ""

    echo "Building pie:${tag}-dev..."
    echo "→ CUDA: ${cuda_ver}.${cuda_minor}"

    $SUDO docker build \
        -f Dockerfile.cuda \
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
echo "To run (NVIDIA Container Toolkit on the host + --shm-size>=2g required;"
echo "default 64 MiB /dev/shm SIGBUSes the engine↔driver shmem buffer."
echo "Use --gpus device=N to expose exactly one GPU; it maps to cuda:0 inside"
echo "the container, matching the baked config's device = [\"cuda:0\"]):"
echo "  Latest: $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 -v ~/.cache:/root/.cache pie:latest"
echo "  Development: $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 -v ~/.cache:/root/.cache pie:dev"
echo ""
echo "With authentication setup (pass SSH public key):"
echo "  $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 \\"
echo "    -e PIE_AUTH_USER=\"myuser\" \\"
echo "    -e PIE_AUTH_KEY=\"\$(cat ~/.ssh/id_ed25519.pub)\" \\"
echo "    -v ~/.cache:/root/.cache \\"
echo "    pie:latest"
echo ""
echo "Or mount key file:"
echo "  $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 \\"
echo "    -e PIE_AUTH_USER=\"myuser\" \\"
echo "    -e PIE_AUTH_KEY_FILE=\"/keys/id_ed25519.pub\" \\"
echo "    -v ~/.ssh/id_ed25519.pub:/keys/id_ed25519.pub:ro \\"
echo "    -v ~/.cache:/root/.cache \\"
echo "    pie:latest"
echo ""
echo "Note: Mount ~/.cache (not just ~/.cache/pie) so the HuggingFace cache persists across runs"
echo ""
echo "To download a model first:"
echo "  $SUDO docker run --rm --gpus device=0 --shm-size=2g -v ~/.cache:/root/.cache pie:latest pie model add \"Qwen/Qwen3-0.6B\""
echo ""
echo "Build complete!"
