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
#   USE_LLAMA_CONFIG=1       Use Llama 3.1 configuration and generate llama31_* cases
#   LLAMA_MODEL=<model>      Llama model size (8B, 70B, 405B, default: 8B)
#   GENERIC_ONLY=1           Generate only generic test cases (skip Llama cases)
#   LLAMA_ONLY=1             Generate only Llama test cases (skip generic cases)
#
# Examples:
#   scripts/generate_cuda_artifacts.sh production          # Generates both generic + Llama cases
#   GENERIC_ONLY=1 scripts/generate_cuda_artifacts.sh      # Only generic cases
#   LLAMA_ONLY=1 scripts/generate_cuda_artifacts.sh        # Only Llama cases
#   LLAMA_MODEL=70B scripts/generate_cuda_artifacts.sh     # Both modes with 70B model

# Detect workspace root more robustly
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "$SCRIPT_DIR" == */cuda-protocol-tests/scripts ]]; then
    # Running from cuda-protocol-tests/scripts
    ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
else
    # Try to find workspace root by looking for key directories
    SEARCH_DIR="$SCRIPT_DIR"
    while [[ "$SEARCH_DIR" != "/" ]]; do
        if [[ -d "$SEARCH_DIR/cuda-protocol-tests" && -d "$SEARCH_DIR/metal-protocol-tests" ]]; then
            ROOT_DIR="$SEARCH_DIR"
            break
        fi
        SEARCH_DIR=$(dirname "$SEARCH_DIR")
    done
    if [[ -z "$ROOT_DIR" ]]; then
        echo "Error: Could not find workspace root containing both cuda-protocol-tests and metal-protocol-tests" >&2
        exit 1
    fi
fi

CUDA_PROJECT_DIR="$ROOT_DIR/cuda-protocol-tests"
BUILD_DIR="$CUDA_PROJECT_DIR/build"
BIN="$BUILD_DIR/cuda_protocol_tests"
INTEGRATION_BIN="$BUILD_DIR/cuda_integration_tests"
ART_DIR="$ROOT_DIR/cuda-protocol-tests/tests/artifacts"
CUDA_BACKEND_DIR="$ROOT_DIR/backend/backend-cuda"
CONFIG_FILE="$CUDA_PROJECT_DIR/llama31_configs.json"
GPU=${GPU:-0}
CASE_ID=${1:-production}
CASE_SUFFIX=${CASE_SUFFIX:-}
FULL_CASE_ID="${CASE_ID}${CASE_SUFFIX}"
USE_LLAMA_CONFIG=${USE_LLAMA_CONFIG:-1}  # Default to enabled
LLAMA_MODEL=${LLAMA_MODEL:-8B}
GENERIC_ONLY=${GENERIC_ONLY:-0}
LLAMA_ONLY=${LLAMA_ONLY:-0}

# Handle mode logic
if [[ "$GENERIC_ONLY" == "1" && "$LLAMA_ONLY" == "1" ]]; then
  echo "Error: Cannot specify both GENERIC_ONLY=1 and LLAMA_ONLY=1" >&2
  exit 1
fi

if [[ "$GENERIC_ONLY" == "1" ]]; then
  USE_LLAMA_CONFIG=0
elif [[ "$LLAMA_ONLY" == "1" ]]; then
  USE_LLAMA_CONFIG=1
fi

# Determine what modes to run
RUN_GENERIC=1
RUN_LLAMA=1
if [[ "$GENERIC_ONLY" == "1" ]]; then
  RUN_LLAMA=0
elif [[ "$LLAMA_ONLY" == "1" ]]; then
  RUN_GENERIC=0
fi

run() {
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "+ $*"
    return 0
  fi
  echo "+ $*" >&2
  "$@"
}

# Validate operation output for non-zero values
validate_operation() {
  local op_name="$1"
  local case_id="$2"
  local op_dir="$ART_DIR/$op_name/$case_id"

  if [[ ! -d "$op_dir" ]]; then
    echo "❌ VALIDATION FAILED: No artifacts directory for $op_name/$case_id" >&2
    return 1
  fi

  # Find output files (different ops have different output file patterns)
  local output_files=()
  while IFS= read -r -d '' file; do
    output_files+=("$file")
  done < <(find "$op_dir" \( -name "output.bin" -o -name "*output*.bin" -o -name "masked_logits.bin" -o -name "q_output.bin" -o -name "k_output.bin" -o -name "V.bin" -o -name "C.bin" -o -name "Y.bin" -o -name "embeddings.bin" -o -name "*cache*.bin" -o -name "A.bin" -o -name "B.bin" -o -name "gate.bin" -o -name "up.bin" \) -print0 2>/dev/null)

  if [[ ${#output_files[@]} -eq 0 ]]; then
    echo "⚠️  WARNING: No recognizable output files for $op_name/$case_id" >&2
    return 0  # Not necessarily an error - some ops might have different patterns
  fi

  local all_zero_files=()
  local total_files=${#output_files[@]}
  local nonzero_files=0

  for output_file in "${output_files[@]}"; do
    if [[ -f "$output_file" && -s "$output_file" ]]; then
      # Check if file has any non-zero bytes in first 1KB
      if hexdump -n 1024 -v -e '1/1 "%02x\n"' "$output_file" | grep -q '^[1-9a-f][0-9a-f]*$\|^0[1-9a-f]$'; then
        nonzero_files=$((nonzero_files + 1))
      else
        all_zero_files+=("$(basename "$output_file")")
      fi
    fi
  done

  if [[ $nonzero_files -eq 0 && ${#all_zero_files[@]} -gt 0 ]]; then
    echo "❌ VALIDATION FAILED: $op_name/$case_id - all output files contain only zeros: ${all_zero_files[*]}" >&2
    return 1
  elif [[ ${#all_zero_files[@]} -gt 0 ]]; then
    echo "⚠️  WARNING: $op_name/$case_id - some output files are all-zero: ${all_zero_files[*]}" >&2
  else
    echo "✅ VALIDATED: $op_name/$case_id - $nonzero_files/$total_files output files contain meaningful data" >&2
  fi

  return 0
}

# Run operation with validation
run_and_validate() {
  local op_name="$1"
  local case_id="$2"
  local use_config="${3:-}"
  shift 3

  echo "==> Running $op_name with case $case_id"
  local cmd_args=("$BIN" --op "$op_name" --case "$case_id")

  # Add Llama config if requested
  if [[ "$use_config" == "with_config" ]]; then
    cmd_args+=(--config "$CONFIG_FILE" --model "$LLAMA_MODEL")
  fi

  cmd_args+=("$@")

  if run "${cmd_args[@]}"; then
    sleep 0.5  # Brief delay to ensure files are fully written
    validate_operation "$op_name" "$case_id"
    return $?
  else
    echo "❌ EXECUTION FAILED: $op_name --case $case_id $*" >&2
    return 1
  fi
}

# Run operation without validation (for multi-dtype variants)
run_operation() {
  local op_name="$1"
  local case_id="$2"
  local use_config="${3:-}"
  shift 3

  echo "==> Running $op_name with case $case_id"
  local cmd_args=("$BIN" --op "$op_name" --case "$case_id")

  # Add Llama config if requested
  if [[ "$use_config" == "with_config" ]]; then
    cmd_args+=(--config "$CONFIG_FILE" --model "$LLAMA_MODEL")
  fi

  cmd_args+=("$@")

  run "${cmd_args[@]}"
}

# Run integration test for layer-by-layer artifact generation
run_integration_test() {
  local test_case="$1"

  echo "==> Running layer-by-layer integration test: $test_case"

  if [[ ! -x "$INTEGRATION_BIN" ]]; then
    echo "❌ Integration test binary not found at $INTEGRATION_BIN" >&2
    return 1
  fi

  if run "$INTEGRATION_BIN" "$test_case"; then
    echo "✅ Integration test completed successfully"

    # Validate that layer artifacts were created
    local integration_dir="$ART_DIR/forward_pass_integration/$test_case/layer_artifacts"
    if [[ -d "$integration_dir" ]]; then
      local layer_count=$(find "$integration_dir" -name "layer_*" -type d | wc -l)
      local tensor_count=$(find "$integration_dir" -name "*.bin" | wc -l)
      echo "✅ Generated layer-by-layer artifacts: $layer_count layers, $tensor_count tensor files"
      return 0
    else
      echo "⚠️  Integration test completed but no layer artifacts found at $integration_dir" >&2
      return 1
    fi
  else
    echo "❌ Integration test failed" >&2
    return 1
  fi
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

# Check Llama config file exists if running Llama cases
if [[ "$RUN_LLAMA" == "1" && ! -f "$CONFIG_FILE" ]]; then
  echo "Error: Llama config file not found at $CONFIG_FILE" >&2
  exit 1
fi

# Clean up any recursive artifacts directories first
if [[ -d "$ART_DIR/artifacts" ]]; then
    echo "==> Cleaning up recursive artifacts directory..."
    run rm -rf "$ART_DIR/artifacts"
fi

# Ensure artifacts directory exists
run mkdir -p "$ART_DIR"

# Build harness (unless skipped)
if [[ -z "${SKIP_BUILD:-}" ]]; then
  run mkdir -p "$BUILD_DIR"
  run cmake -S "$CUDA_PROJECT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  CORES=1
  if command -v nproc >/dev/null 2>&1; then
    CORES=$(($(nproc) - 1))
    if [[ $CORES -lt 1 ]]; then CORES=1; fi
  elif command -v sysctl >/dev/null 2>&1; then
    CORES=$(($(sysctl -n hw.ncpu) - 1))
    if [[ $CORES -lt 1 ]]; then CORES=1; fi
  fi
  run make -C "$BUILD_DIR" -j"$CORES"
else
  echo "==> Skipping build (SKIP_BUILD=1)"
fi

if [[ ! -x "$BIN" ]]; then
  echo "Error: protocol tests binary not found at $BIN" >&2
  exit 1
fi

if [[ ! -x "$INTEGRATION_BIN" ]]; then
  echo "Error: integration tests binary not found at $INTEGRATION_BIN" >&2
  exit 1
fi

# Configure environment for artifact writing
export CUDA_VISIBLE_DEVICES="$GPU"
export PIE_WRITE_ARTIFACTS=1
export PIE_ARTIFACTS_DIR="$ART_DIR"

cd "$BUILD_DIR"

echo "==> Generating CUDA artifacts into $ART_DIR"
if [[ "$RUN_GENERIC" == "1" && "$RUN_LLAMA" == "1" ]]; then
  echo "    Modes: Generic + Llama $LLAMA_MODEL (case: $FULL_CASE_ID)"
elif [[ "$RUN_LLAMA" == "1" ]]; then
  echo "    Mode: Llama $LLAMA_MODEL only (case: $FULL_CASE_ID)"
else
  echo "    Mode: Generic only (case: $FULL_CASE_ID)"
fi

# Track validation results
FAILED_OPS=()
WARNED_OPS=()
TOTAL_OPS=0

# Generate Llama-specific test cases
if [[ "$RUN_LLAMA" == "1" ]]; then
  echo "==> Generating Llama-specific test cases..."

  # Llama 3.1 production cases with config-driven parameters
  run_and_validate rms_norm "llama31_prod" with_config --num_tokens 512 || FAILED_OPS+=("llama31_rms_norm")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate silu_and_mul "llama31_prod" with_config --num_tokens 512 || FAILED_OPS+=("llama31_silu_and_mul")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate rope "llama31_prod" with_config --num_tokens 512 || FAILED_OPS+=("llama31_rope")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Llama projection-specific GEMM cases
  run_and_validate gemm "llama31_qkv" with_config --m 512 --n 4096 --k 4096 || FAILED_OPS+=("llama31_gemm_qkv")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate gemm "llama31_o_proj" with_config --m 512 --n 4096 --k 4096 || FAILED_OPS+=("llama31_gemm_o_proj")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate gemm "llama31_gate_up" with_config --m 512 --n 6144 --k 4096 || FAILED_OPS+=("llama31_gemm_gate_up")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate gemm "llama31_down" with_config --m 512 --n 4096 --k 6144 || FAILED_OPS+=("llama31_gemm_down")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Note: embedding_lookup operation creates artifacts under "embedding_lookup_forward" directory
  if run "$BIN" --op embedding_lookup --case "llama31_prod" --config "$CONFIG_FILE" --model "$LLAMA_MODEL" --num_tokens 512; then
    validate_operation "embedding_lookup_forward" "llama31_prod" || FAILED_OPS+=("llama31_embedding_lookup")
  else
    echo "❌ EXECUTION FAILED: embedding_lookup --case llama31_prod --num_tokens 512" >&2
    FAILED_OPS+=("llama31_embedding_lookup")
  fi
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate softmax "llama31_prod" with_config --batch_size 8 || FAILED_OPS+=("llama31_softmax")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate topk_mask_logits "llama31_prod" with_config --num_tokens 4 --k 50 || FAILED_OPS+=("llama31_topk_mask_logits")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate extract_k_values "llama31_prod" with_config --M 4 --k 50 || FAILED_OPS+=("llama31_extract_k_values")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Attention and KV cache with Llama 3.1 8B parameters
  run_and_validate batch_prefill_attention "llama31_prod" with_config --num_tokens 128 --num_query_heads 32 --num_kv_heads 8 --head_size 128 --kv_len 2048 --page_size 16 || FAILED_OPS+=("llama31_batch_prefill_attention")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate append_paged_kv_cache "llama31_prod" with_config --num_tokens 128 --num_kv_heads 8 --head_size 128 --page_size 16 --batch_size 2 || FAILED_OPS+=("llama31_append_paged_kv_cache")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate add_residual "llama31_prod" with_config --num_tokens 512 || FAILED_OPS+=("llama31_add_residual")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Layer-by-layer integration test with real model (generates individual tensor artifacts)
  echo "==> Running layer-by-layer integration test..."
  run_integration_test "real_model_forward_pass" || FAILED_OPS+=("layer_by_layer_integration")
  TOTAL_OPS=$((TOTAL_OPS + 1))
fi

# Generate generic test cases
if [[ "$RUN_GENERIC" == "1" ]]; then
  # Original generic test cases with hardcoded parameters
  echo "==> Generating generic test cases..."

  # Core matmul/activations/normalization
  run_and_validate gemm "$FULL_CASE_ID" "" --m 128 --n 4096 --k 4096 || FAILED_OPS+=("generic_gemm")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Note: embedding_lookup operation creates artifacts under "embedding_lookup_forward" directory
  if run "$BIN" --op embedding_lookup --case "$FULL_CASE_ID" --num_tokens 128 --hidden_size 4096 --vocab_size 32000; then
    validate_operation "embedding_lookup_forward" "$FULL_CASE_ID" || FAILED_OPS+=("generic_embedding_lookup")
  else
    echo "❌ EXECUTION FAILED: embedding_lookup --case $FULL_CASE_ID" >&2
    FAILED_OPS+=("generic_embedding_lookup")
  fi
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate extract_k_values "$FULL_CASE_ID" "" --M 128 --N 4096 --k 50 || FAILED_OPS+=("generic_extract_k_values")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate rms_norm "$FULL_CASE_ID" "" --num_tokens 128 --hidden_size 4096 || FAILED_OPS+=("generic_rms_norm")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate silu_and_mul "$FULL_CASE_ID" "" --num_tokens 128 --intermediate_size 11008 || FAILED_OPS+=("generic_silu_and_mul")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Attention-related
  run_and_validate rope "$FULL_CASE_ID" "" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128 || FAILED_OPS+=("generic_rope")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate topk_mask_logits "$FULL_CASE_ID" "" --num_tokens 128 --vocab_size 32000 --k 50 || FAILED_OPS+=("generic_topk_mask_logits")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate softmax "$FULL_CASE_ID" "" --batch_size 2 --vocab_size 32000 --temperature 1.0 || FAILED_OPS+=("generic_softmax")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Prefill + paged KV cache ops (using corrected Llama 3.1 config)
  run_and_validate batch_prefill_attention "$FULL_CASE_ID" "" --num_tokens 128 --num_query_heads 32 --num_kv_heads 8 --head_size 128 --kv_len 2048 --page_size 16 || FAILED_OPS+=("generic_batch_prefill_attention")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  run_and_validate append_paged_kv_cache "$FULL_CASE_ID" "" --num_tokens 128 --num_kv_heads 8 --head_size 128 --page_size 16 --batch_size 2 || FAILED_OPS+=("generic_append_paged_kv_cache")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Optional: grouped GEMM (remove unsupported --num_groups flag for now)
  run_and_validate grouped_gemm "$FULL_CASE_ID" "" --m 128 --n 4096 --k 4096 || FAILED_OPS+=("generic_grouped_gemm")
  TOTAL_OPS=$((TOTAL_OPS + 1))

  # Add missing add_residual operation
  run_and_validate add_residual "$FULL_CASE_ID" "" --num_tokens 128 --hidden_size 4096 || FAILED_OPS+=("generic_add_residual")
  TOTAL_OPS=$((TOTAL_OPS + 1))
fi

# Multi-dtype variants for all supported ops (creates *_fp32/*_f16/*_bf16 cases under each op)
echo "==> Generating multi-dtype variants for all operations..."
echo "    This generates f16, bf16, and f32 variants for each operation"

# Generate multi-dtype variants for Llama cases
if [[ "$RUN_LLAMA" == "1" ]]; then
  echo "==> Multi-dtype variants (Llama config mode)..."
  run_operation gemm_all_dtypes             "llama31_prod" with_config --m 128 --n 2048 --k 2048
  run_operation embedding_lookup_all_dtypes "llama31_prod" with_config --num_tokens 128
  run_operation extract_k_values_all_dtypes "llama31_prod" with_config --M 128 --k 50
  run_operation rms_norm_all_dtypes         "llama31_prod" with_config --num_tokens 128
  run_operation silu_and_mul_all_dtypes     "llama31_prod" with_config --num_tokens 128
  run_operation rope_all_dtypes             "llama31_prod" with_config --num_tokens 128
  run_operation softmax_all_dtypes          "llama31_prod" with_config --batch_size 2
  run_operation topk_mask_logits_all_dtypes "llama31_prod" with_config --num_tokens 128 --k 50
  run_operation batch_prefill_attention_all_dtypes "llama31_prod" with_config --num_tokens 128 --kv_len 2048 --page_size 16
  run_operation append_paged_kv_cache_all_dtypes "llama31_prod" with_config --num_tokens 128 --page_size 16 --batch_size 2
  run_operation add_residual_all_dtypes     "llama31_prod" with_config --num_tokens 128
fi

# Generate multi-dtype variants for generic cases
if [[ "$RUN_GENERIC" == "1" ]]; then
  echo "==> Multi-dtype variants (generic mode)..."
  run_operation gemm_all_dtypes             "${FULL_CASE_ID}" "" --m 128 --n 2048 --k 2048
  run_operation embedding_lookup_all_dtypes "${FULL_CASE_ID}" "" --num_tokens 128 --hidden_size 4096 --vocab_size 32000
  run_operation extract_k_values_all_dtypes "${FULL_CASE_ID}" "" --M 128 --N 4096 --k 50
  run_operation rms_norm_all_dtypes         "${FULL_CASE_ID}" "" --num_tokens 128 --hidden_size 4096
  run_operation silu_and_mul_all_dtypes     "${FULL_CASE_ID}" "" --num_tokens 128 --intermediate_size 11008
  run_operation rope_all_dtypes             "${FULL_CASE_ID}" "" --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128
  run_operation softmax_all_dtypes          "${FULL_CASE_ID}" "" --batch_size 2 --vocab_size 32000 --temperature 1.0
  run_operation topk_mask_logits_all_dtypes "${FULL_CASE_ID}" "" --num_tokens 128 --vocab_size 32000 --k 50
  run_operation batch_prefill_attention_all_dtypes "${FULL_CASE_ID}" "" --num_tokens 128 --num_query_heads 32 --num_kv_heads 8 --head_size 128 --kv_len 2048 --page_size 16
  run_operation append_paged_kv_cache_all_dtypes "${FULL_CASE_ID}" "" --num_tokens 128 --num_kv_heads 8 --head_size 128 --page_size 16 --batch_size 2
  run_operation add_residual_all_dtypes     "${FULL_CASE_ID}" "" --num_tokens 128 --hidden_size 4096
fi

# Post-process: generate/update manifest if available
MANIFEST_GENERATOR="$ROOT_DIR/scripts/generate_artifact_manifest.py"
if [[ -f "$MANIFEST_GENERATOR" ]]; then
  echo "==> Generating artifact manifest..."
  (cd "$ROOT_DIR/scripts" && python3 generate_artifact_manifest.py generate "$ART_DIR") || true
fi

# Validation Summary Report
echo ""
echo "======================================"
echo "🎯 CUDA Artifact Generation Summary"
echo "======================================"
echo "Case ID: $FULL_CASE_ID"
echo "Artifacts Directory: $ART_DIR"
echo "Total Operations: $TOTAL_OPS"

if [[ ${#FAILED_OPS[@]} -eq 0 ]]; then
  echo "✅ SUCCESS: All operations completed with valid outputs!"
else
  echo "❌ FAILED OPERATIONS (${#FAILED_OPS[@]}/$TOTAL_OPS): ${FAILED_OPS[*]}"
fi

# Count artifacts generated
ARTIFACT_DIRS=$(find "$ART_DIR" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
ARTIFACT_FILES=$(find "$ART_DIR" -name "*.bin" 2>/dev/null | wc -l)

echo "📊 Generated: $ARTIFACT_DIRS test cases, $ARTIFACT_FILES artifact files"

echo "======================================"

if [[ ${#FAILED_OPS[@]} -gt 0 ]]; then
  echo "⚠️  Some operations failed validation. Check logs above for details."
  exit 1
else
  echo "==> Done. All CUDA artifacts successfully generated and validated under: $ART_DIR"
fi
