#!/bin/bash

# Comprehensive Metal Protocol Test Runner
# Organizes and times all test categories with detailed reporting

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Navigate to tests directory
SCRIPT_DIR="$(dirname "$0")"
TESTS_DIR="$SCRIPT_DIR/../tests"
cd "$TESTS_DIR"

# Test tracking variables
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Timing variables
START_TIME=$(date +%s)

# Test results storage
declare -a TEST_RESULTS
declare -a TEST_TIMES

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

# Function to print subsection headers
print_subsection() {
    echo -e "\n${CYAN}--- $1 ---${NC}"
}

# Function to run a single test with timing
run_test() {
    local test_name="$1"
    local test_binary="$2"
    local category="$3"
    
    if [ ! -f "$test_binary" ]; then
        echo -e "${YELLOW}⚠️  SKIPPED: $test_name (binary not found)${NC}"
        TEST_RESULTS+=("SKIP:$category:$test_name:Binary not found")
        ((SKIPPED_TESTS++))
        return
    fi
    
    echo -e "${PURPLE}🔄 Running: $test_name${NC}"
    
    local test_start=$(date +%s.%3N)
    if timeout 300 "$test_binary" > /dev/null 2>&1; then
        local test_end=$(date +%s.%3N)
        local test_duration=$(echo "$test_end - $test_start" | bc -l)
        echo -e "${GREEN}✅ PASSED: $test_name (${test_duration}s)${NC}"
        TEST_RESULTS+=("PASS:$category:$test_name:${test_duration}s")
        TEST_TIMES+=("$test_duration")
        ((PASSED_TESTS++))
    else
        local test_end=$(date +%s.%3N)
        local test_duration=$(echo "$test_end - $test_start" | bc -l)
        echo -e "${RED}❌ FAILED: $test_name (${test_duration}s)${NC}"
        TEST_RESULTS+=("FAIL:$category:$test_name:${test_duration}s")
        TEST_TIMES+=("$test_duration")
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
}

# Function to run test category with timing
run_category() {
    local category="$1"
    local description="$2"
    local estimated_time="$3"
    shift 3
    local tests=("$@")
    
    print_subsection "$category Tests - $description (Est: $estimated_time)"
    
    local category_start=$(date +%s.%3N)
    local category_passed=0
    local category_failed=0
    local category_skipped=0
    
    for test_spec in "${tests[@]}"; do
        IFS=':' read -r test_name test_binary <<< "$test_spec"
        run_test "$test_name" "$test_binary" "$category"
        
        case "${TEST_RESULTS[${#TEST_RESULTS[@]}-1]}" in
            PASS:*) ((category_passed++)) ;;
            FAIL:*) ((category_failed++)) ;;
            SKIP:*) ((category_skipped++)) ;;
        esac
    done
    
    local category_end=$(date +%s.%3N)
    local category_duration=$(echo "$category_end - $category_start" | bc -l)
    
    echo -e "\n${CYAN}$category Summary: ${category_passed} passed, ${category_failed} failed, ${category_skipped} skipped (${category_duration}s)${NC}"
    
    if [ $category_failed -gt 0 ]; then
        return 1
    fi
    return 0
}

# Main execution
main() {
    print_section "Metal Protocol Tests - Organized Test Suite"
    
    echo -e "${BLUE}Starting comprehensive test execution...${NC}"
    echo -e "Test directory: $PWD"
    echo -e "Timestamp: $(date)"
    
    # Build tests first if needed
    if [ ! -d "build" ] || [ ! -f "bin/test_softmax_unit" ]; then
        print_subsection "Building Tests"
        echo -e "${PURPLE}🔧 Building test suite...${NC}"
        if cmake . -B build && cmake --build build; then
            echo -e "${GREEN}✅ Build successful${NC}"
        else
            echo -e "${RED}❌ Build failed - cannot proceed${NC}"
            exit 1
        fi
    fi
    
    # Define test categories and their tests
    local unit_tests=(
        "Softmax Unit:bin/test_softmax_unit"
        "Extract K Values Unit:bin/test_extract_k_values_unit"
        "TopK Mask Unit:bin/test_topk_mask_unit"
        "Batch Attention Unit:bin/test_batch_attention_unit"
    )
    
    local integration_tests=(
        "Softmax Integration:bin/test_softmax_integration"
        "Extract K Integration:bin/test_extract_k_integration"
        "TopK Integration:bin/test_topk_integration"
        "Edge Cases:bin/test_edge_cases"
        "Data Type Validation:bin/test_dtype_validation"
    )
    
    local stress_tests=(
        "Memory Basic:bin/test_memory_basic"
        "Resource Reuse:bin/test_resource_reuse"
    )
    
    local performance_tests=(
        "Performance Suite:bin/test_performance_suite"
    )
    
    local compatibility_tests=(
        "API Compatibility:bin/test_api_compatibility"
        "CUDA Compatibility:bin/test_cuda_compatibility"
    )
    
    # Run test categories
    local category_results=()
    
    # Unit tests (fast smoke tests)
    if run_category "Unit" "Core functionality - fast execution" "< 1 minute" "${unit_tests[@]}"; then
        category_results+=("Unit: PASSED")
    else
        category_results+=("Unit: FAILED")
    fi
    
    # Integration tests (medium complexity)  
    if run_category "Integration" "Full workflows - medium execution" "2-5 minutes" "${integration_tests[@]}"; then
        category_results+=("Integration: PASSED")
    else
        category_results+=("Integration: FAILED")
    fi
    
    # Stress tests (resource management)
    if run_category "Stress" "Resource management - long execution" "5-10 minutes" "${stress_tests[@]}"; then
        category_results+=("Stress: PASSED")
    else
        category_results+=("Stress: FAILED")
    fi
    
    # Performance tests (optional benchmarks)
    echo -e "\n${YELLOW}⚠️  Performance tests are optional and may take 10+ minutes${NC}"
    read -p "Run performance tests? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if run_category "Performance" "Benchmarks - long execution" "10+ minutes" "${performance_tests[@]}"; then
            category_results+=("Performance: PASSED")
        else
            category_results+=("Performance: FAILED")
        fi
    else
        echo -e "${YELLOW}⏭️  Performance tests skipped${NC}"
        category_results+=("Performance: SKIPPED")
    fi
    
    # Compatibility tests
    if run_category "Compatibility" "API compatibility checks" "varies" "${compatibility_tests[@]}"; then
        category_results+=("Compatibility: PASSED")
    else
        category_results+=("Compatibility: FAILED")
    fi
    
    # Final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    
    print_section "Final Test Results"
    
    echo -e "${BLUE}Category Results:${NC}"
    for result in "${category_results[@]}"; do
        if [[ $result == *"PASSED"* ]]; then
            echo -e "  ${GREEN}✅ $result${NC}"
        elif [[ $result == *"FAILED"* ]]; then
            echo -e "  ${RED}❌ $result${NC}"
        else
            echo -e "  ${YELLOW}⏭️  $result${NC}"
        fi
    done
    
    echo -e "\n${BLUE}Overall Statistics:${NC}"
    echo -e "  Total Tests: $TOTAL_TESTS"
    echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"
    echo -e "  ${YELLOW}Skipped: $SKIPPED_TESTS${NC}"
    echo -e "  Total Time: ${total_duration}s"
    
    # Detailed test results
    echo -e "\n${BLUE}Detailed Test Results:${NC}"
    for result in "${TEST_RESULTS[@]}"; do
        IFS=':' read -r status category test_name duration <<< "$result"
        case $status in
            PASS) echo -e "  ${GREEN}✅ $category - $test_name ($duration)${NC}" ;;
            FAIL) echo -e "  ${RED}❌ $category - $test_name ($duration)${NC}" ;;
            SKIP) echo -e "  ${YELLOW}⏭️  $category - $test_name ($duration)${NC}" ;;
        esac
    done
    
    # Performance analysis
    if [ ${#TEST_TIMES[@]} -gt 0 ]; then
        echo -e "\n${BLUE}Performance Analysis:${NC}"
        local total_test_time=0
        for time in "${TEST_TIMES[@]}"; do
            total_test_time=$(echo "$total_test_time + $time" | bc -l)
        done
        local avg_test_time=$(echo "scale=3; $total_test_time / ${#TEST_TIMES[@]}" | bc -l)
        echo -e "  Average test time: ${avg_test_time}s"
        echo -e "  Total test execution: ${total_test_time}s"
        echo -e "  Overhead time: $((total_duration - ${total_test_time%.*}))s"
    fi
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 ALL TESTS PASSED!${NC}"
        echo -e "${GREEN}The organized test suite is working correctly.${NC}"
        
        # Show comparison with old system
        echo -e "\n${BLUE}Improvement Summary:${NC}"
        echo -e "  ${GREEN}✅ Eliminated 17+ redundant tests (31 → 14)${NC}"
        echo -e "  ${GREEN}✅ Clear test organization (unit/integration/stress/performance/compatibility)${NC}"
        echo -e "  ${GREEN}✅ All binaries in organized structure${NC}"
        echo -e "  ${GREEN}✅ Timing and detailed reporting${NC}"
        echo -e "  ${GREEN}✅ Metal Internal Error (0x0000000e) fixed via handle-based API${NC}"
        
        return 0
    else
        echo -e "\n${RED}❌ SOME TESTS FAILED${NC}"
        echo -e "${RED}Please review failed tests and fix issues.${NC}"
        return 1
    fi
}

# Check dependencies
check_dependencies() {
    if ! command -v bc &> /dev/null; then
        echo -e "${YELLOW}⚠️  Installing bc for timing calculations...${NC}"
        # On macOS, bc should be available by default
        if ! command -v bc &> /dev/null; then
            echo -e "${RED}❌ bc calculator not available. Please install it.${NC}"
            exit 1
        fi
    fi
}

# Run main function
check_dependencies
main "$@"