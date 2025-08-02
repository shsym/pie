#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

NUM_INSTANCES=(1 8 16 32 64 128)

BASE_CMD="python3 ./test_4_agent_case_study_pie.py"

# --- Case 1: No extra flags ---
echo "--- Running Case 1: No extra flags ---"
for i in "${NUM_INSTANCES[@]}"; do
  echo "Testing with --num-instances=$i"
  $BASE_CMD --num-instances="$i"
  echo "------------------------------------------"
done
echo "--- Case 1 Complete ---"
echo ""

# --- Case 2: --use-prefix-cache ---
echo "--- Running Case 2: --use-prefix-cache ---"
for i in "${NUM_INSTANCES[@]}"; do
  echo "Testing with --num-instances=$i and --use-prefix-cache"
  $BASE_CMD --num-instances="$i" --use-prefix-cache
  echo "------------------------------------------"
done
echo "--- Case 2 Complete ---"
echo ""


# --- Case 3: --use-prefix-cache and --concurrent-calls ---
echo "--- Running Case 3: --use-prefix-cache and --concurrent-calls ---"
for i in "${NUM_INSTANCES[@]}"; do
  echo "Testing with --num-instances=$i, --use-prefix-cache, and --concurrent-calls"
  $BASE_CMD --num-instances="$i" --use-prefix-cache --concurrent-calls
  echo "------------------------------------------"
done
echo "--- Case 3 Complete ---"
echo ""


# --- Case 4: --use-prefix-cache, --concurrent-calls, and --drop-tool-cache ---
echo "--- Running Case 4: --use-prefix-cache, --concurrent-calls, and --drop-tool-cache ---"
for i in "${NUM_INSTANCES[@]}"; do
  echo "Testing with --num-instances=$i, --use-prefix-cache, --concurrent-calls, and --drop-tool-cache"
  $BASE_CMD --num-instances="$i" --use-prefix-cache --concurrent-calls --drop-tool-cache
  echo "------------------------------------------"
done
echo "--- Case 4 Complete ---"

echo ""
echo "✅ All experiments finished."
