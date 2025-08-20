#!/usr/bin/env bash
# Generate CUDA golden reference artifacts for Metal protocol tests (Linux)
#
# Usage:
#   scripts/generate_cuda_artifacts.sh [CASE_ID]
#
# Env vars:
#   GPU=<id>                 GPU index for CUDA_VISIBLE_DEVICES (default: 0)
#   SKIP_BUILD=1             Skip building the CUDA harness binary
#   DRY_RUN=1                Print commands without executing them
#   CASE_SUFFIX=<suffix>     Optional suffix appended to the case id (e.g., _debug)
#
# Examples:
#   GPU=1 scripts/generate_cuda_artifacts.sh production
#   scripts/generate_cuda_artifacts.sh validation

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)
CUDA_PROJECT_DIR="$ROOT_DIR/cuda-protocol-tests"
BUILD_DIR="$CUDA_PROJECT_DIR/build"
BIN="$BUILD_DIR/cuda_protocol_tests"
ART_DIR="$ROOT_DIR/metal-protocol-tests/tests/artifacts"
CUDA_BACKEND_DIR="$ROOT_DIR/backend/backend-cuda"
GPU=${GPU:-0}
CASE_ID=${1:-production}
CASE_SUFFIX=${CASE_SUFFIX:-}
FULL_CASE_ID="${CASE_ID}${CASE_SUFFIX}"

run() {
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "+ $*"
    return 0
  fi
  echo "+ $*" >&2
  "$@"
}

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Error: required command '$1' not found in PATH" >&2; exit 127; }; }

# Tooling checks
need_cmd cmake
need_cmd make

# Check CUDA backend static libs exist (as required by cuda-protocol-tests/CMakeLists.txt)
if [[ ! -f "$CUDA_BACKEND_DIR/build/lib/libprefill_kernels.a" || ! -f "$CUDA_BACKEND_DIR/build/lib/libdecode_kernels.a" ]]; then
  echo "Error: CUDA backend libraries not found. Please build backend-cuda first:" >&2
  echo "  cd backend/backend-cuda && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j\"\${NPROC:-$(command -v nproc >/dev/null && nproc || echo 1)}\"" >&2
  exit 1
fi

# Ensure artifacts directory exists
run mkdir -p "$ART_DIR"

# Build harness (unless skipped)
if [[ -z "${SKIP_BUILD:-}" ]]; then
  run mkdir -p "$BUILD_DIR"
  run cmake -S "$CUDA_PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  CORES=1
  if command -v nproc >/dev/null 2>&1; then CORES=$(nproc); elif command -v sysctl >/dev/null 2>&1; then CORES=$(sysctl -n hw.ncpu); fi
  run make -C "$BUILD_DIR" -j"$CORES"
else
  echo "==> Skipping build (SKIP_BUILD=1)"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found at $BIN" >&2
  exit 1
fi

# Configure environment for artifact writing
export CUDA_VISIBLE_DEVICES="$GPU"
export PIE_WRITE_ARTIFACTS=1
export PIE_ARTIFACTS_DIR="$ART_DIR"

cd "$BUILD_DIR"

echo "==> Generating CUDA artifacts into $ART_DIR (case: $FULL_CASE_ID)"

# Core matmul/activations/normalization
run "$BIN" --op gemm               --case "$FULL_CASE_ID" --m 128 --n 4096 --k 4096
run "$BIN" --op embedding_lookup   --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096 --vocab_size 32000
run "$BIN" --op extract_k_values   --case "$FULL_CASE_ID" --M 128 --N 4096 --k 50
run "$BIN" --op rms_norm           --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096
run "$BIN" --op silu_and_mul       --case "$FULL_CASE_ID" --num_tokens 128 --intermediate_size 11008

# Attention-related
run "$BIN" --op rope               --case "$FULL_CASE_ID" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
run "$BIN" --op topk_mask_logits   --case "$FULL_CASE_ID" --num_tokens 128 --vocab_size 32000 --k 50
run "$BIN" --op softmax            --case "$FULL_CASE_ID" --batch_size 2 --vocab_size 32000 --temperature 1.0

# Prefill + paged KV cache ops
run "$BIN" --op batch_prefill_attention --case "$FULL_CASE_ID" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128 --kv_len 2048 --page_size 16
run "$BIN" --op append_paged_kv_cache  --case "$FULL_CASE_ID" --num_tokens 128 --num_kv_heads 32 --head_size 128 --page_size 16 --max_num_pages 8 --batch_size 2

# Optional: grouped GEMM
run "$BIN" --op grouped_gemm --case "$FULL_CASE_ID" --m 128 --n 4096 --k 4096 --num_groups 4

# Multi-dtype variants for selected ops (creates *_bf16/*_fp32 cases under each op)
run "$BIN" --op gemm_all_dtypes             --case "${FULL_CASE_ID}"
run "$BIN" --op embedding_lookup_all_dtypes --case "${FULL_CASE_ID}"
run "$BIN" --op extract_k_values_all_dtypes --case "${FULL_CASE_ID}"

# Post-process: generate/update manifest if available
MANIFEST_GENERATOR="$ROOT_DIR/scripts/generate_artifact_manifest.py"
if [[ -f "$MANIFEST_GENERATOR" ]]; then
  (cd "$ROOT_DIR/scripts" && python3 generate_artifact_manifest.py generate ../metal-protocol-tests/tests/artifacts) || true
fi

echo "==> Done. CUDA artifacts written under: $ART_DIR"
