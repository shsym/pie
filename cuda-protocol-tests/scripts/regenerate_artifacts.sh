#!/usr/bin/env bash
# Avoid 'set -e' to prevent abrupt exits on expected non-zero codes; keep -u and pipefail for safety.
set -uo pipefail

# Regenerate all metal protocol test artifacts (cleans old ones first)
#
# Usage:
#   scripts/regenerate_artifacts.sh [CASE_ID]
#
# Env vars:
#   GPU=<id>                 GPU index for CUDA_VISIBLE_DEVICES (default: 0)
#   SKIP_BUILD=1             Skip building the metal-protocol-tests binary
#   DRY_RUN=1                Print commands without executing them
#   CASE_SUFFIX=<suffix>     Optional suffix appended to the case id (e.g., _debug)
#
# Example:
#   GPU=1 scripts/regenerate_artifacts.sh production
#   scripts/regenerate_artifacts.sh validation

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)
PROJECT_DIR="$ROOT_DIR/metal-protocol-tests"
BUILD_DIR="$PROJECT_DIR/build"
BIN="$BUILD_DIR/metal_protocol_tests"
GPU=${GPU:-0}
CASE_ID=${1:-production}
CASE_SUFFIX=${CASE_SUFFIX:-}
FULL_CASE_ID="${CASE_ID}${CASE_SUFFIX}"

ART_DIR="$PROJECT_DIR/tests/artifacts"

# Help mode
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<EOF
Regenerate all metal protocol test artifacts (cleans old ones first)

Usage:
  GPU=<id> SKIP_BUILD=1 DRY_RUN=1 CASE_SUFFIX=_foo \\
    bash metal-protocol-tests/scripts/regenerate_artifacts.sh [CASE_ID]

Defaults:
  CASE_ID: production
  GPU: 0

Artifacts directory:
  $ART_DIR

Environment variables:
  GPU=<id>             GPU index for CUDA_VISIBLE_DEVICES (default: 0)
  SKIP_BUILD=1         Skip building the metal-protocol-tests binary
  DRY_RUN=1            Print commands without executing them
  CASE_SUFFIX=<suffix> Optional suffix appended to the case id (e.g., _debug)
EOF
  exit 0
fi

run() {
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "+ $*"
    return 0
  fi
  echo "+ $*" >&2
  "$@"
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "Command failed with status $status: $*" >&2
    exit $status
  fi
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Error: required command '$1' not found in PATH" >&2; exit 127; }; }

# 0) Basic tooling checks
need_cmd cmake
need_cmd make

# 1) Clean old artifacts
echo "==> Cleaning old artifacts"
run rm -rf "$ART_DIR" "$BUILD_DIR/tests/artifacts"
run mkdir -p "$ART_DIR"

# 2) No CUDA dependencies are required for the Metal harness

# 3) Build the metal_protocol_tests binary (unless skipped)
if [[ -z "${SKIP_BUILD:-}" ]]; then
  echo "==> Building metal_protocol_tests"
  run mkdir -p "$BUILD_DIR"
  # Configure with explicit source/build args to avoid cd/quoting issues
  run cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  # Portable CPU core detection: prefer nproc, fallback to sysctl (macOS), else 1
  CORES=1
  if command -v nproc >/dev/null 2>&1; then
    CORES=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    CORES=$(sysctl -n hw.ncpu)
  fi
  run make -C "$BUILD_DIR" -j"$CORES"
else
  echo "==> Skipping build (SKIP_BUILD=1)"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found at $BIN" >&2
  exit 1
fi

# 4) Generate artifacts (Metal-implemented ops only)
echo "==> Generating artifacts into $ART_DIR (case: $FULL_CASE_ID)"
export CUDA_VISIBLE_DEVICES="$GPU"

# Force canonical output into project tests/artifacts, then mirror to build
export PIE_WRITE_ARTIFACTS=1
export PIE_ARTIFACTS_DIR="$ART_DIR"

cd "$BUILD_DIR" || { echo "Failed to cd to $BUILD_DIR" >&2; exit 1; }

run "$BIN" --op gemm --case "$FULL_CASE_ID" --m 128 --n 4096 --k 4096
run "$BIN" --op embedding_lookup --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096 --vocab_size 32000
run "$BIN" --op extract_k_values --case "$FULL_CASE_ID" --M 128 --N 4096 --k 50
run "$BIN" --op rms_norm --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096
run "$BIN" --op silu_and_mul --case "$FULL_CASE_ID" --num_tokens 128 --intermediate_size 11008
run "$BIN" --op softmax --case "$FULL_CASE_ID" --batch_size 2 --vocab_size 32000 --temperature 1.0

# Summary
echo "==> Done. Artifacts written under: $ART_DIR"
