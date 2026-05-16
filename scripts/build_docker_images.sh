#!/bin/bash
# Build Pie Docker images.
#
# Variants:
#   portable — CPU/ggml-backed; small, runs anywhere; ~/.cache/pie cache vol.
#   cuda     — TBD (the previous Dockerfile referenced deleted paths and was
#              removed pending a rewrite for the new driver/ architecture).
#
# Usage:
#   scripts/build_docker_images.sh              # build all variants
#   scripts/build_docker_images.sh portable     # build only the portable variant
#   PIE_IMAGE_REPO=ghcr.io/pie-project/pie scripts/build_docker_images.sh
#
# Tags produced (REPO defaults to "pieproject/pie"):
#   $REPO:portable        — runtime stage (slim)
#   $REPO:portable-dev    — development stage (full builder toolchain)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="${PIE_IMAGE_REPO:-pieproject/pie}"

if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

# Variants requested on the command line. Default = all known variants.
if [ $# -gt 0 ]; then
    VARIANTS=("$@")
else
    VARIANTS=(portable)
fi

build_portable() {
    echo "=================================================="
    echo "Building ${REPO}:portable (runtime stage, slim)"
    echo "=================================================="
    $SUDO docker build \
        -f "$PROJECT_ROOT/Dockerfile.portable" \
        --target runtime \
        -t "${REPO}:portable" \
        "$PROJECT_ROOT"
    echo "✓ ${REPO}:portable"
    echo ""

    echo "=================================================="
    echo "Building ${REPO}:portable-dev (development stage)"
    echo "=================================================="
    $SUDO docker build \
        -f "$PROJECT_ROOT/Dockerfile.portable" \
        --target development \
        -t "${REPO}:portable-dev" \
        "$PROJECT_ROOT"
    echo "✓ ${REPO}:portable-dev"
    echo ""
}

for variant in "${VARIANTS[@]}"; do
    case "$variant" in
        portable) build_portable ;;
        cuda)
            echo "ERROR: cuda variant is TBD — needs Dockerfile.cuda rewrite for the new driver/ architecture." >&2
            exit 2
            ;;
        *)
            echo "ERROR: unknown variant '$variant' (expected: portable | cuda)" >&2
            exit 2
            ;;
    esac
done

echo "=================================================="
echo "Build Summary"
echo "=================================================="
$SUDO docker images | grep -E "^${REPO//\//\\/}\s+(portable|cuda)" || true
echo ""
echo "To run portable variant:"
echo "  docker run --rm --shm-size=4g -p 8080:8080 \\"
echo "    -v ~/.cache/pie:/root/.cache/pie \\"
echo "    ${REPO}:portable"
echo ""
echo "  ↑ --shm-size=4g is required. The portable driver places KV cache"
echo "    on /dev/shm; Docker's 64 MiB default crashes pie with SIGBUS."
echo ""
echo "To download a model first:"
echo "  docker run --rm --shm-size=4g -v ~/.cache/pie:/root/.cache/pie ${REPO}:portable \\"
echo "    pie model add \"Qwen/Qwen3-0.6B\""
echo ""
echo "Auth setup (pass SSH public key via env or file):"
echo "  -e PIE_AUTH_USER=myuser -e PIE_AUTH_KEY=\"\$(cat ~/.ssh/id_ed25519.pub)\""
echo ""
