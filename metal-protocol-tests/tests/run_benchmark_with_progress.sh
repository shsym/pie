#!/bin/bash

# Script to run attention performance benchmark with progress bar and log file
# Usage: ./run_benchmark_with_progress.sh

LOG_FILE="attn_bench.log"
TEMP_LOG="/tmp/benchmark_output.tmp"
BIN_PATH="./bin/test_attention_performance_benchmark"

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
echo ""

# Remove old log file
rm -f "$LOG_FILE"

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

# Stop tail process
kill $TAIL_PID 2>/dev/null

# Final progress
show_progress $total_tests $total_tests "Completed!"
echo ""
echo ""

# Copy temp log to final log
cat "$TEMP_LOG" >> "$LOG_FILE"
rm -f "$TEMP_LOG"

# Show summary
echo -e "${GREEN}✅ Benchmark completed!${NC}"
echo ""

# Parse and show summary from log
if [ -f "$LOG_FILE" ]; then
    echo -e "${BLUE}📊 SUMMARY:${NC}"
    
    # Count successful tests
    successful=$(grep -c "✅ GOOD\|⚠️ MODEST\|➖ NEUTRAL" "$LOG_FILE" || echo "0")
    failed=$(grep -c "❌\|FAILED" "$LOG_FILE" || echo "0")
    
    echo "   Successful tests: $successful"
    echo "   Failed tests: $failed"
    
    # Show speedups
    echo ""
    echo -e "${BLUE}🚀 SPEEDUPS:${NC}"
    grep "🚀 Speedup:" "$LOG_FILE" | while read -r line; do
        config=$(echo "$line" | grep -o "Small:\|Medium:\|Large:\|Memory stress:" | head -1)
        speedup=$(echo "$line" | grep -o "[0-9]\+\.[0-9]\+x" | head -1)
        status=$(echo "$line" | grep -o "✅ GOOD\|⚠️ MODEST\|❌ SLOWER\|➖ NEUTRAL" | head -1)
        echo "   $config $speedup $status"
    done
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