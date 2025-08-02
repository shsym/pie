#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PYTHON_SCRIPT="microbench_spawn_time.py"
SERVER_URI="ws://127.0.0.1:8080"

# --- Define the test parameters ---
NUM_INSTANCES=(1 8 16 32 64 128 256 384 512 640 768 896 1024)

# --- Main execution ---
# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at '$PYTHON_SCRIPT'"
    echo "Please make sure the script is in the same directory or update the PYTHON_SCRIPT variable."
    exit 1
fi

echo "Starting benchmark automation..."

# Loop through each number of instances
for num in "${NUM_INSTANCES[@]}"; do
  echo "=> Running with $num instances..."

  python3 "$PYTHON_SCRIPT" \
    --server-uri "$SERVER_URI" \
    --num-instances "$num" \

  echo "=> Completed run with $num instances."
  echo ""
  sleep 1
done

echo "--- ✅ All benchmarks complete. ---"
