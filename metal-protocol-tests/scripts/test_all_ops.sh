#!/bin/bash
# Test all implemented Metal protocol operations
# Usage: ./scripts/test_all_ops.sh [--case CASE_ID] [--no-compare] [--llama-only] [--generic-only]

set -e

CASE_ID=""
EXTRA_FLAGS=""
RUN_LLAMA=1
RUN_GENERIC=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --case)
            CASE_ID="$2"
            shift 2
            ;;
        --no-compare)
            EXTRA_FLAGS="--no-compare"
            shift
            ;;
        --llama-only)
            RUN_LLAMA=1
            RUN_GENERIC=0
            shift
            ;;
        --generic-only)
            RUN_LLAMA=0
            RUN_GENERIC=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--case CASE_ID] [--no-compare] [--llama-only] [--generic-only]"
            exit 1
            ;;
    esac
done

# Resolve paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BIN="${PROJECT_ROOT}/build/metal_protocol_tests"
# Fallbacks if using multi-config generators (e.g., Xcode) or different build dir layout
if [[ ! -x "$BIN" ]]; then
    if [[ -x "${PROJECT_ROOT}/build/Release/metal_protocol_tests" ]]; then
        BIN="${PROJECT_ROOT}/build/Release/metal_protocol_tests"
    elif [[ -x "${PROJECT_ROOT}/metal_protocol_tests" ]]; then
        BIN="${PROJECT_ROOT}/metal_protocol_tests"
    fi
fi

echo "=== Testing All Metal Protocol Operations ==="
if [[ -n "$CASE_ID" ]]; then
    echo "Case ID: $CASE_ID"
else
    echo "Case: auto-select"
fi
if [[ -n "$EXTRA_FLAGS" ]]; then
    echo "Mode: Testing only (no comparison)"
else
    echo "Mode: Testing with CUDA comparison"
fi

if [[ "$RUN_LLAMA" == "1" && "$RUN_GENERIC" == "1" ]]; then
    echo "Test Suite: Generic + Llama cases"
elif [[ "$RUN_LLAMA" == "1" ]]; then
    echo "Test Suite: Llama cases only"
else
    echo "Test Suite: Generic cases only"
fi
echo

# Check if python is available and numpy is installed
if ! command -v python &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 to run the tests."
    exit 1
fi
if ! python -c "import numpy" &> /dev/null; then
    echo "Numpy is not installed. Please install numpy (e.g., via 'pip install numpy') to run the tests."
    exit 1
fi

# Define test operation arrays for both generic and Llama cases

# Generic test operations (original parameters)
generic_operations=(
    "gemm --case production --m 128 --n 4096 --k 4096"
    "embedding_lookup --case production --num_tokens 128 --hidden_size 4096 --vocab_size 32000"
    "silu_and_mul --case production --num_tokens 128 --intermediate_size 11008"
    "extract_k_values --case production --M 128 --N 4096 --k 50"
    "softmax --case production --batch_size 2 --vocab_size 32000 --temperature 1.0"
    "rms_norm --case production --num_tokens 128 --hidden_size 4096 --eps 1e-5"
    "rope --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 32 --head_size 128"
    "topk_mask_logits --case production --num_tokens 128 --vocab_size 32000 --k 50"
    "grouped_gemm --case production --m 128 --n 4096 --k 4096"
    "batch_prefill_attention --case production --num_tokens 128 --num_query_heads 32 --num_kv_heads 8 --head_size 128 --kv_len 2048 --page_size 16"
    "append_paged_kv_cache --case production --num_tokens 128 --num_kv_heads 8 --head_size 128 --page_size 16 --batch_size 2"
    "add_residual --case production --num_tokens 128 --hidden_size 4096"
)

# Llama-specific test operations (Llama 3.1 8B parameters)
llama_operations=(
    "rms_norm --case llama31_prod --num_tokens 512 --hidden_size 4096 --eps 1e-5"
    "silu_and_mul --case llama31_prod --num_tokens 512 --intermediate_size 6144"
    "rope --case llama31_prod --num_tokens 512 --num_query_heads 32 --num_kv_heads 8 --head_size 128"
    "gemm --case llama31_qkv --m 512 --n 4096 --k 4096"
    "gemm --case llama31_o_proj --m 512 --n 4096 --k 4096"
    "gemm --case llama31_gate_up --m 512 --n 6144 --k 4096"
    "gemm --case llama31_down --m 512 --n 4096 --k 6144"
    "embedding_lookup --case llama31_prod --num_tokens 512 --hidden_size 4096 --vocab_size 128000"
    "softmax --case llama31_prod --batch_size 8 --vocab_size 128000 --temperature 1.0"
    "topk_mask_logits --case llama31_prod --num_tokens 4 --vocab_size 128000 --k 50"
    "extract_k_values --case llama31_prod --M 4 --N 128000 --k 50"
    "batch_prefill_attention --case llama31_prod --num_tokens 128 --num_query_heads 32 --num_kv_heads 8 --head_size 128 --kv_len 2048 --page_size 16"
    "append_paged_kv_cache --case llama31_prod --num_tokens 128 --num_kv_heads 8 --head_size 128 --page_size 16 --batch_size 2"
    "add_residual --case llama31_prod --num_tokens 512 --hidden_size 4096"
)

passed=0
failed=0

# Function to run test operations
run_test_suite() {
    local suite_name="$1"
    shift
    local operations=("$@")
    
    echo "--- $suite_name Test Suite ---"
    
    for op_cmd in "${operations[@]}"; do
        op_name=$(echo "$op_cmd" | cut -d' ' -f1)
        case_name=$(echo "$op_cmd" | grep -o -- '--case [^ ]*' | cut -d' ' -f2 || echo "default")
        echo -n "🔄 Testing $op_name ($case_name)... "

        # Build command - the op_cmd already contains all necessary args including --case
        CMD=("$BIN" --op)
        # Split op command into args safely
        read -r -a OP_ARGS <<< "$op_cmd"
        CMD+=("${OP_ARGS[@]}")
        
        # Override case if explicitly provided
        if [[ -n "$CASE_ID" ]]; then
            # Remove existing --case from command and add new one
            local filtered_cmd=()
            local skip_next=false
            for arg in "${CMD[@]}"; do
                if [[ "$skip_next" == true ]]; then
                    skip_next=false
                    continue
                fi
                if [[ "$arg" == "--case" ]]; then
                    skip_next=true
                    continue
                fi
                filtered_cmd+=("$arg")
            done
            CMD=("${filtered_cmd[@]}" --case "$CASE_ID")
        fi
        
        if [[ -n "$EXTRA_FLAGS" ]]; then
            read -r -a EXTRA_ARR <<< "$EXTRA_FLAGS"
            CMD+=("${EXTRA_ARR[@]}")
        fi

        if "${CMD[@]}" >/dev/null 2>&1; then
            echo "✅ PASS"
            ((passed++))
        else
            echo "❌ FAIL"
            ((failed++))
        fi
    done
    echo
}

# Run test suites based on flags
if [[ "$RUN_GENERIC" == "1" ]]; then
    run_test_suite "Generic" "${generic_operations[@]}"
fi

if [[ "$RUN_LLAMA" == "1" ]]; then
    run_test_suite "Llama" "${llama_operations[@]}"
fi

echo
echo "=== Summary ==="
echo "✅ Passed: $passed"
echo "❌ Failed: $failed"
echo "📊 Total: $((passed + failed))"

if [ $failed -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "💥 $failed test(s) failed"
    exit 1
fi