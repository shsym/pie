#!/bin/bash
# Test all implemented Metal protocol operations
# Usage: ./scripts/test_all_ops.sh [--case CASE_ID] [--no-compare]

set -e

CASE_ID="test1"
EXTRA_FLAGS=""

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--case CASE_ID] [--no-compare]"
            exit 1
            ;;
    esac
done

echo "=== Testing All Metal Protocol Operations ==="
echo "Case ID: $CASE_ID"
if [[ -n "$EXTRA_FLAGS" ]]; then
    echo "Mode: Testing only (no comparison)"
else
    echo "Mode: Testing with CUDA comparison"
fi
echo

# Test all 10 implemented operations
operations=(
    "gemm --m 16 --n 32 --k 24"
    "embedding_lookup --num_tokens 8 --hidden_size 128 --vocab_size 1000"
    "silu_and_mul --num_tokens 8 --intermediate_size 64"
    "extract_k_values --M 4 --N 32 --k 5"
    "softmax --batch_size 2 --vocab_size 100 --temperature 1.0"
    "rms_norm --num_tokens 8 --hidden_size 128 --eps 1e-5"
    "rope --num_tokens 4 --num_heads 4 --head_size 16"
    "topk_mask_logits --num_tokens 4 --vocab_size 50 --k 10"
    "grouped_gemm"
    "batch_prefill_attention --num_tokens 4 --num_query_heads 2 --num_kv_heads 2 --head_size 8 --kv_len 16 --page_size 128"
)

passed=0
failed=0

for op_cmd in "${operations[@]}"; do
    op_name=$(echo $op_cmd | cut -d' ' -f1)
    echo -n "🔄 Testing $op_name... "
    
    if ./metal_protocol_tests --op $op_cmd --case $CASE_ID $EXTRA_FLAGS >/dev/null 2>&1; then
        echo "✅ PASS"
        ((passed++))
    else
        echo "❌ FAIL"
        ((failed++))
    fi
done

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