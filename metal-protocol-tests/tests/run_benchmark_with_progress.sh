#!/bin/bash

# Script to run attention performance benchmark with progress bar and log file
# Usage: ./run_benchmark_with_progress.sh

set -euo pipefail

LOG_FILE="attn_bench.log"
TEMP_LOG="/tmp/benchmark_output.tmp"
# Resolve script directory to be robust when invoked from elsewhere
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_PATH="$SCRIPT_DIR/bin/test_attention_performance_benchmark"
BUILD_DIR="$SCRIPT_DIR/build"

# Ensure GPU config path is available to the benchmark (can be overridden by the user)
DEFAULT_GPU_CONFIG_PATH="$SCRIPT_DIR/../apple_gpu_configs.json"
export PIE_GPU_CONFIG_PATH="${PIE_GPU_CONFIG_PATH:-$DEFAULT_GPU_CONFIG_PATH}"

# Cleanup on exit
cleanup() {
    [[ -n "${TAIL_PID:-}" ]] && kill "$TAIL_PID" 2>/dev/null || true
    [[ -f "$TEMP_LOG" ]] && rm -f "$TEMP_LOG" || true
}
trap cleanup EXIT

# Attempt to build the benchmark if the binary is missing
maybe_build_binary() {
    if [[ -x "$BIN_PATH" ]]; then
        return 0
    fi

    echo "ℹ️  Benchmark binary not found at $BIN_PATH"
    echo "🔧 Attempting to build it via CMake..."

    # Discover cmake binary with fallbacks
    CMAKE_BIN=""
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_BIN="$(command -v cmake)"
    elif [[ -x "/opt/homebrew/bin/cmake" ]]; then
        CMAKE_BIN="/opt/homebrew/bin/cmake"
    elif [[ -x "/usr/local/bin/cmake" ]]; then
        CMAKE_BIN="/usr/local/bin/cmake"
    elif [[ -x "/usr/bin/cmake" ]]; then
        CMAKE_BIN="/usr/bin/cmake"
    fi

    if [[ -z "$CMAKE_BIN" ]]; then
        echo "❌ CMake is not installed or not on PATH. Please install CMake and retry."
        echo "   macOS: brew install cmake"
        exit 127
    fi

    mkdir -p "$BUILD_DIR"
    "$CMAKE_BIN" -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >/dev/null
    "$CMAKE_BIN" --build "$BUILD_DIR" --config Release --target test_attention_performance_benchmark --parallel >/dev/null

    if [[ ! -x "$BIN_PATH" ]]; then
        echo "❌ Build completed but binary still not found at: $BIN_PATH"
        echo "   Please ensure backend libraries are built (see backend/backend-metal/README.md) and retry."
        exit 127
    fi

    echo "✅ Built: $BIN_PATH"
}

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local operation=$3
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))

    printf "\r["
    for ((i=0; i<width; i++)); do
        if [ $i -lt $completed ]; then
            printf "█"
        elif [ $i -eq $completed ]; then
            printf "▌"
        else
            printf " "
        fi
    done
    printf "] %d%% - %s" $percentage "$operation"
}

# Error collection
errors=()

echo -e "${BLUE}🏁 Metal FlashAttention Performance Benchmark${NC}"
echo "================================================"
echo ""
echo "📝 Detailed logs will be saved to: ${LOG_FILE}"
echo "🛠️ Using GPU config: ${PIE_GPU_CONFIG_PATH}"
echo ""

# Remove old log file
rm -f "$LOG_FILE"

# Ensure the binary exists (build if necessary)
maybe_build_binary

# Start the benchmark in background and pipe output to temp file
"$BIN_PATH" > "$TEMP_LOG" 2>&1 &
BENCHMARK_PID=$!

# Monitor the output and show progress
test_count=0
total_tests=7  # Number of benchmark configurations

echo -e "${YELLOW}Starting benchmark...${NC}"
show_progress 0 $total_tests "Initializing..."

# Monitor output file
tail -f "$TEMP_LOG" | while IFS= read -r line; do
    # Log everything to file
    echo "$(date '+%H:%M:%S.%3N') $line" >> "$LOG_FILE"

    # Parse output for progress updates
    if [[ "$line" =~ "🔄 Benchmarking:" ]]; then
        test_count=$((test_count + 1))
        config=$(echo "$line" | sed 's/🔄 Benchmarking: //')
        show_progress $test_count $total_tests "Testing: $config"
    elif [[ "$line" =~ "📋 Testing baseline kernel" ]]; then
        show_progress $test_count $total_tests "Running baseline kernel..."
    elif [[ "$line" =~ "⚡ Testing simdgroup kernel" ]]; then
        show_progress $test_count $total_tests "Running SIMD kernel..."
    elif [[ "$line" =~ "FAILED" ]] || [[ "$line" =~ "ERROR" ]] || [[ "$line" =~ "❌" ]]; then
        # Collect errors
        errors+=("$line")
    fi
done &
TAIL_PID=$!

# Wait for benchmark to complete
wait $BENCHMARK_PID
BENCHMARK_EXIT_CODE=$?

# Give a moment for tail to catch up with final output
sleep 2

# Stop tail process
kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

# Final progress
show_progress $total_tests $total_tests "Completed!"
echo ""
echo ""

# Copy temp log to final log and ensure all output is captured
cat "$TEMP_LOG" >> "$LOG_FILE"
rm -f "$TEMP_LOG"

# Show summary
echo -e "${GREEN}✅ Benchmark completed!${NC}"
echo ""

# Parse and show summary from log
if [ -f "$LOG_FILE" ]; then
    echo -e "${BLUE}📊 SUMMARY:${NC}"

    # Count successful tests from the final summary
    if grep -q "PRIORITY 0 VALIDATION RESULTS:" "$LOG_FILE"; then
        successful=$(grep "✅ Successful tests:" "$LOG_FILE" | tail -1 | grep -o "[0-9]\+/[0-9]\+" | head -1)
        improved=$(grep "🚀 Improved performance:" "$LOG_FILE" | tail -1 | grep -o "[0-9]\+/[0-9]\+" | head -1)
        avg_speedup=$(grep "📈 Average speedup:" "$LOG_FILE" | tail -1 | grep -o "[0-9]\+\.[0-9]\+x" | head -1)
        validation_result=$(grep -A 1 "PRIORITY 0 VALIDATION:" "$LOG_FILE" | tail -1 | sed 's/^[^a-zA-Z]*//')

        echo "   Successful tests: ${successful:-N/A}"
        echo "   Improved performance: ${improved:-N/A}"
        echo "   Average speedup: ${avg_speedup:-N/A}"
        echo "   Validation result: ${validation_result:-N/A}"
    else
        # Fallback to basic parsing
        successful=$(grep -c "✅ GOOD\|⚠️ MODEST\|➖ NEUTRAL" "$LOG_FILE" || echo "0")
        failed=$(grep -c "❌\|FAILED" "$LOG_FILE" || echo "0")
        echo "   Successful tests: $successful"
        echo "   Failed tests: $failed"
    fi

    # Show individual speedups from final summary
    echo ""
    echo -e "${BLUE}🚀 INDIVIDUAL RESULTS:${NC}"
    if grep -q "📊 PERFORMANCE VALIDATION SUMMARY" "$LOG_FILE"; then
        # Extract results from the performance summary section
        sed -n '/📊 PERFORMANCE VALIDATION SUMMARY/,/🎯 PRIORITY 0 VALIDATION RESULTS:/p' "$LOG_FILE" | \
        grep -E "Baseline.*Speedup:" | while read -r line; do
            # Extract test description and speedup
            speedup=$(echo "$line" | grep -o "[0-9]\+\.[0-9]\+x" | head -1)
            status=$(echo "$line" | grep -o "✅\|⚠️\|❌\|➖" | head -1)
            echo "   Speedup: $speedup $status"
        done
    else
        # Fallback to parsing individual speedup lines
        grep "🚀 Speedup:" "$LOG_FILE" | tail -7 | while read -r line; do
            speedup=$(echo "$line" | grep -o "[0-9]\+\.[0-9]\+x" | head -1)
            status=$(echo "$line" | grep -o "✅ GOOD\|⚠️ MODEST\|❌ SLOWER\|➖ NEUTRAL" | head -1)
            echo "   Speedup: $speedup $status"
        done
    fi
fi

echo ""

# Show errors if any
if [ ${#errors[@]} -gt 0 ]; then
    echo -e "${RED}⚠️  ERRORS ENCOUNTERED:${NC}"
    for error in "${errors[@]}"; do
        echo -e "   ${RED}❌${NC} $error"
    done
    echo ""
fi

echo -e "${YELLOW}📋 Full details available in: ${LOG_FILE}${NC}"
echo ""

exit $BENCHMARK_EXIT_CODE