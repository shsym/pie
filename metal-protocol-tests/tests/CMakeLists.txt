cmake_minimum_required(VERSION 3.20)
project(MetalProtocolTestsOrganized)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required frameworks
find_library(METAL_LIBRARY Metal REQUIRED)
find_library(METALPERFORMANCESHADERS_LIBRARY MetalPerformanceShaders REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)

# Set backend source paths
set(BACKEND_SRC_PATH "../../backend/backend-metal/src")
set(BACKEND_BUILD_PATH "../../backend/backend-metal/build")

# Common include directories
include_directories(${BACKEND_SRC_PATH})
include_directories(../src)
include_directories(src)
include_directories(src/performance)

# Link to backend build directory for libraries
link_directories(${BACKEND_BUILD_PATH})

# Common libraries and frameworks
set(COMMON_LIBRARIES 
    ${METAL_LIBRARY} 
    ${METALPERFORMANCESHADERS_LIBRARY} 
    ${FOUNDATION_LIBRARY}
)

# Backend source files needed for tests
set(BACKEND_SOURCES
    ${BACKEND_SRC_PATH}/metal_batch_prefill_attention.mm
    ${BACKEND_SRC_PATH}/metal_softmax.mm
    ${BACKEND_SRC_PATH}/metal_extract_k_values.mm
    ${BACKEND_SRC_PATH}/metal_topk_mask_logits.mm
    ${BACKEND_SRC_PATH}/metal_attention_naive.mm
)

# Additional backend libraries to link against
set(BACKEND_LIBS
    metal_softmax
    metal_extract_k_values  
    metal_topk_mask_logits
    metal_batch_prefill_attention
    metal_dtype_conversion
    metal_common
)

# Set output directory for all targets
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")

# ============================================================================
# Unit Tests (Core functionality - fast execution)
# ============================================================================

add_executable(test_softmax_unit 
    src/unit/test_softmax.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_softmax_unit ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_extract_k_values_unit
    src/unit/test_extract_k_values.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_extract_k_values_unit ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_topk_mask_unit
    src/unit/test_topk_mask.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_topk_mask_unit ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# Our new handle/workspace test
add_executable(test_batch_attention_unit
    src/stress/test_handle_stress.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_batch_attention_unit ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# ============================================================================
# Integration Tests (Full workflows - medium execution time)
# ============================================================================

add_executable(test_softmax_integration
    src/integration/test_softmax_integration.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_softmax_integration ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_extract_k_integration
    src/integration/test_extract_k_integration.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_extract_k_integration ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_topk_integration
    src/integration/test_topk_integration.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_topk_integration ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_edge_cases
    src/integration/test_edge_cases.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_edge_cases ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_dtype_validation
    src/integration/test_dtype_validation.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_dtype_validation ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# ============================================================================
# Stress Tests (Resource management - long execution time)
# ============================================================================

add_executable(test_memory_basic
    src/stress/test_memory_basic.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_memory_basic ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_resource_reuse
    src/stress/test_resource_reuse.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_resource_reuse ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# Note: test_memory_stress.mm disabled due to MetalContext dependency
# add_executable(test_memory_stress
#     src/stress/test_memory_stress.mm
#     ${BACKEND_SOURCES}
# )
# target_link_libraries(test_memory_stress ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# ============================================================================
# Performance Tests (Benchmarks - optional, long execution time)
# ============================================================================

add_executable(test_performance_suite
    src/performance/test_performance_suite.cpp
    ${BACKEND_SOURCES}
)
target_link_libraries(test_performance_suite ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_attention_algorithm_comparison
    src/performance/test_attention_algorithm_comparison.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_attention_algorithm_comparison ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# ============================================================================
# Compatibility Tests (API compatibility checks)
# ============================================================================

add_executable(test_api_compatibility
    src/compatibility/test_api_compatibility.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_api_compatibility ${COMMON_LIBRARIES} ${BACKEND_LIBS})

add_executable(test_cuda_compatibility
    src/compatibility/test_cuda_compatibility.mm
    ${BACKEND_SOURCES}
)
target_link_libraries(test_cuda_compatibility ${COMMON_LIBRARIES} ${BACKEND_LIBS})

# ============================================================================
# Custom targets for organized test execution
# ============================================================================

# Unit tests (fast smoke tests)
add_custom_target(run_unit_tests
    COMMAND echo "=== Running Unit Tests (< 1 minute) ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_softmax_unit
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_extract_k_values_unit
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_topk_mask_unit
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_batch_attention_unit
    COMMAND echo "✅ Unit tests completed"
    DEPENDS test_softmax_unit test_extract_k_values_unit test_topk_mask_unit test_batch_attention_unit
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Integration tests (medium complexity)
add_custom_target(run_integration_tests
    COMMAND echo "=== Running Integration Tests (2-5 minutes) ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_softmax_integration
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_extract_k_integration
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_topk_integration
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_edge_cases
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_dtype_validation
    COMMAND echo "✅ Integration tests completed"
    DEPENDS test_softmax_integration test_extract_k_integration test_topk_integration test_edge_cases test_dtype_validation
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Stress tests (resource management)
add_custom_target(run_stress_tests
    COMMAND echo "=== Running Stress Tests (5-10 minutes) ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_memory_basic
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_resource_reuse
    COMMAND echo "✅ Stress tests completed"
    DEPENDS test_memory_basic test_resource_reuse
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Performance tests (benchmarks)
add_custom_target(run_performance_tests
    COMMAND echo "=== Running Performance Tests (10+ minutes) ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_performance_suite
    COMMAND echo "✅ Performance tests completed"
    DEPENDS test_performance_suite
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Algorithmic optimization validation
add_custom_target(run_algorithm_validation
    COMMAND echo "=== Running Algorithmic Optimization Validation ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_attention_algorithm_comparison
    COMMAND echo "✅ Algorithm validation completed"
    DEPENDS test_attention_algorithm_comparison
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compatibility tests
add_custom_target(run_compatibility_tests
    COMMAND echo "=== Running Compatibility Tests ==="
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_api_compatibility
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bin/test_cuda_compatibility
    COMMAND echo "✅ Compatibility tests completed"
    DEPENDS test_api_compatibility test_cuda_compatibility
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Essential tests (unit + integration + stress)
add_custom_target(run_essential_tests
    COMMAND echo "=== Running Essential Tests (unit + integration + stress) ==="
    DEPENDS run_unit_tests run_integration_tests run_stress_tests
)

# Full test suite (all tests)
add_custom_target(run_all_organized_tests
    COMMAND echo "=== Running Complete Organized Test Suite ==="
    DEPENDS run_unit_tests run_integration_tests run_stress_tests run_compatibility_tests
)

# Quick smoke test (just unit tests)
add_custom_target(run_smoke_tests
    COMMAND echo "=== Running Smoke Tests (unit only) ==="
    DEPENDS run_unit_tests
)

# Clean binaries
add_custom_target(clean_organized_tests
    COMMAND echo "Cleaning organized test binaries..."
    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/bin/*
)

# Test statistics
add_custom_target(test_stats
    COMMAND echo "=== Organized Test Suite Statistics ==="
    COMMAND echo "Unit Tests:         4 tests  (under 1 minute)"
    COMMAND echo "Integration Tests:  5 tests  (2-5 minutes)"
    COMMAND echo "Stress Tests:       2 tests  (5-10 minutes)"
    COMMAND echo "Performance Tests:  1 test   (10+ minutes)"
    COMMAND echo "Compatibility Tests: 2 tests (varies)"
    COMMAND echo "Total:             14 tests"
    COMMAND echo ""
    COMMAND echo "Available targets:"
    COMMAND echo "  run_smoke_tests        - Unit tests only (fastest)"
    COMMAND echo "  run_essential_tests    - Unit + Integration + Stress"
    COMMAND echo "  run_all_organized_tests - All organized tests"
    COMMAND echo "  run_unit_tests"
    COMMAND echo "  run_integration_tests"
    COMMAND echo "  run_stress_tests" 
    COMMAND echo "  run_performance_tests"
    COMMAND echo "  run_algorithm_validation"
    COMMAND echo "  run_compatibility_tests"
)

message(STATUS "Organized Metal Protocol Test Suite configured")
message(STATUS "Total tests: 14 (reduced from 31+ redundant tests)")
message(STATUS "Structure: unit(4) + integration(5) + stress(2) + performance(1) + compatibility(2)")
message(STATUS "Use 'make test_stats' to see available targets")