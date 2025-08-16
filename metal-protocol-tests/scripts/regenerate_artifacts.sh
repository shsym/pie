#!/usr/bin/env bash
set -euo pipefail

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
  else
  echo "+ $*" >&2
  "$@"
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

# 2) Ensure backend-cuda libs exist (required by this test binary)
CUDA_LIB_DIR="$ROOT_DIR/backend/backend-cuda/build/lib"
if [[ ! -d "$CUDA_LIB_DIR" ]] || [[ ! -f "$CUDA_LIB_DIR/libprefill_kernels.a" ]] || [[ ! -f "$CUDA_LIB_DIR/libdecode_kernels.a" ]]; then
  echo "Error: backend-cuda libraries not found at $CUDA_LIB_DIR" >&2
  echo "Please build the CUDA backend first (see backend/backend-cuda/README.md)." >&2
  exit 1
fi

# 3) Build the metal_protocol_tests binary (unless skipped)
if [[ -z "${SKIP_BUILD:-}" ]]; then
  echo "==> Building metal_protocol_tests"
  run mkdir -p "$BUILD_DIR"
  # Configure with explicit source/build args to avoid cd/quoting issues
  run cmake -S "$PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  run make -C "$BUILD_DIR" -j"$(nproc)"
else
  echo "==> Skipping build (SKIP_BUILD=1)"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: binary not found at $BIN" >&2
  exit 1
fi

# 4) Generate artifacts
echo "==> Generating artifacts into $ART_DIR (case: $FULL_CASE_ID)"
export CUDA_VISIBLE_DEVICES="$GPU"

# Force canonical output into project tests/artifacts, then mirror to build
export PIE_WRITE_ARTIFACTS=1
export PIE_ARTIFACTS_DIR="$ART_DIR"

cd "$BUILD_DIR"

# Multi-dtype operations
run "$BIN" --op gemm_all_dtypes --case "$FULL_CASE_ID" --m 128 --n 4096 --k 4096
run "$BIN" --op embedding_lookup_all_dtypes --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096 --vocab_size 32000
run "$BIN" --op extract_k_values_all_dtypes --case "$FULL_CASE_ID" --M 128 --N 4096 --k 50

# Single-dtype operations
run "$BIN" --op rms_norm --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096
run "$BIN" --op silu_and_mul --case "$FULL_CASE_ID" --num_tokens 128 --intermediate_size 11008
run "$BIN" --op rope --case "$FULL_CASE_ID" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
run "$BIN" --op topk_mask_logits --case "$FULL_CASE_ID" --num_tokens 128 --vocab_size 32000 --k 50
run "$BIN" --op softmax --case "$FULL_CASE_ID" --vocab_size 32000 --temperature 1.0
run "$BIN" --op grouped_gemm --case "$FULL_CASE_ID" --num_groups 4 --m 128 --n 4096 --k 4096
run "$BIN" --op batch_prefill_attention --case "$FULL_CASE_ID" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128 --kv_len 2048 --page_size 16
run "$BIN" --op append_paged_kv_cache --case "$FULL_CASE_ID" --num_tokens 128 --num_kv_heads 32 --head_size 128 --page_size 16 --max_num_pages 8 --batch_size 2
run "$BIN" --op add_residual --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096

# Example cast_type (conversion) sample
run "$BIN" --op cast_type --case "$FULL_CASE_ID" --num_elements 1024 --input_dtype fp32 --output_dtype fp16

# Optional: also generate single-dtype baselines for gemm and embedding_lookup (native dtype runs)
run "$BIN" --op gemm --case "$FULL_CASE_ID" --m 128 --n 4096 --k 4096
run "$BIN" --op embedding_lookup --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096 --vocab_size 32000

# Summary
echo "==> Done. Artifacts written under: $ART_DIR"
