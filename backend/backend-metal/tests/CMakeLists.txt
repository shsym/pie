# Metal Backend Test Suite
cmake_minimum_required(VERSION 3.23)
project(MetalBackendTests)

# Enable testing
enable_testing()

# Find the Metal framework
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Add the parent build directory for libraries
link_directories(../build)

# Common test dependencies
set(METAL_TEST_LIBS
    metal_softmax
    metal_extract_k_values  
    metal_topk_mask_logits
    metal_batch_prefill_attention
    metal_dtype_conversion
    metal_common
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Common compile flags for all tests
set(METAL_TEST_COMPILE_FLAGS "-fobjc-arc -std=c++17")

# Include the parent src directory for header files
include_directories(../src)

# Helper function to create a test executable
function(add_metal_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} ${METAL_TEST_LIBS})
    set_target_properties(${test_name} PROPERTIES
        COMPILE_FLAGS ${METAL_TEST_COMPILE_FLAGS}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME ${test_name} COMMAND ${CMAKE_BINARY_DIR}/bin/${test_name})
endfunction()

# Helper function for Objective-C++ tests
function(add_metal_objcpp_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} ${METAL_TEST_LIBS})
    set_target_properties(${test_name} PROPERTIES
        COMPILE_FLAGS ${METAL_TEST_COMPILE_FLAGS}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME ${test_name} COMMAND ${CMAKE_BINARY_DIR}/bin/${test_name})
endfunction()

# Unit Tests
add_metal_test(test_metal_unit_softmax src/unit/test_metal_unit_softmax.cpp)
add_metal_test(test_metal_unit_extract_k_values src/unit/test_metal_unit_extract_k_values.cpp)
add_metal_test(test_metal_unit_topk_mask src/unit/test_metal_unit_topk_mask.cpp)

# Data Type Tests
add_metal_test(test_metal_dtype_working src/integration/test_metal_dtype_working.cpp)
add_metal_test(test_metal_dtype_validation src/integration/test_metal_dtype_validation.cpp)

# Edge Case Tests
add_metal_test(test_metal_edge_cases src/integration/test_metal_edge_cases.cpp)
add_metal_test(test_metal_edge_cases_robust src/integration/test_metal_edge_cases_robust.cpp)

# Performance Tests
add_metal_test(test_metal_performance src/performance/test_metal_performance.cpp)
add_metal_test(test_metal_softmax_benchmark src/performance/test_metal_softmax_benchmark.cpp)
add_metal_test(test_tiled_performance_validation src/performance/test_tiled_performance_validation.cpp)

# Attention Performance Tests (Objective-C++ since they use Metal APIs)
add_metal_objcpp_test(test_attention_benchmark src/performance/test_attention_benchmark.mm)

# Core Integration Tests
add_metal_test(test_metal_extract_k_values src/integration/test_metal_extract_k_values.cpp)
add_metal_test(test_metal_softmax src/integration/test_metal_softmax.cpp)
add_metal_test(test_metal_softmax_tiled src/integration/test_metal_softmax_tiled.cpp)
add_metal_test(test_metal_topk_mask_logits src/integration/test_metal_topk_mask_logits.cpp)

# Full Pipeline Tests (Objective-C++)
add_metal_objcpp_test(test_metal_inference_basic src/integration/test_metal_inference_basic.mm)
add_metal_objcpp_test(test_metal_model_weight_loading src/integration/test_metal_model_weight_loading.mm)
add_metal_objcpp_test(test_metal_memory_basic src/integration/test_metal_memory_basic.mm)
add_metal_objcpp_test(test_metal_memory_stress src/integration/test_metal_memory_stress.mm)
add_metal_objcpp_test(test_metal_real_model_weights src/integration/test_metal_real_model_weights.mm)
# TODO: Re-enable when MetalModel implementation is complete
# add_metal_objcpp_test(test_metal_full_inference_pipeline src/integration/test_metal_full_inference_pipeline.mm)
# add_metal_objcpp_test(test_end_to_end_inference src/integration/test_end_to_end_inference.mm)

# Compatibility Tests (Objective-C++)
add_metal_objcpp_test(test_api_compatibility src/compatibility/test_api_compatibility.mm)
add_metal_objcpp_test(test_cuda_compatibility src/compatibility/test_cuda_compatibility.mm)
add_metal_objcpp_test(test_compilation_validation src/compatibility/test_compilation_validation.mm)

# Debug Tests (not part of main test suite)
add_metal_test(test_debug_validation src/debug/test_debug_validation.cpp)
add_metal_test(test_simple_dtype_debug src/debug/test_simple_dtype_debug.cpp)
add_metal_test(test_metal_unit_extract_k_values_debug src/unit/test_metal_unit_extract_k_values_debug.cpp)
add_metal_test(test_softmax_precision_debug src/debug/test_softmax_precision_debug.cpp)

# Test groups for organized execution
set_tests_properties(
    test_metal_unit_softmax
    test_metal_unit_extract_k_values  
    test_metal_unit_topk_mask
    PROPERTIES LABELS "unit"
)

set_tests_properties(
    test_metal_dtype_working
    test_metal_dtype_validation
    PROPERTIES LABELS "dtype"
)

set_tests_properties(
    test_metal_edge_cases
    test_metal_edge_cases_robust
    PROPERTIES LABELS "edge_cases"
)

set_tests_properties(
    test_metal_performance
    test_metal_softmax_benchmark
    test_tiled_performance_validation
    test_attention_benchmark
    PROPERTIES LABELS "performance"
)

set_tests_properties(
    test_metal_extract_k_values
    test_metal_softmax
    test_metal_topk_mask_logits
    test_metal_inference_basic
    test_metal_model_weight_loading
    test_metal_memory_basic
    test_metal_memory_stress
    test_metal_real_model_weights
    # TODO: Re-enable when MetalModel implementation is complete
    # test_metal_full_inference_pipeline
    # test_end_to_end_inference
    PROPERTIES LABELS "integration"
)

set_tests_properties(
    test_api_compatibility
    test_cuda_compatibility
    test_compilation_validation
    PROPERTIES LABELS "compatibility"
)

# Custom test targets for different categories
add_custom_target(test_unit
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --verbose
    COMMENT "Running unit tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_dtype
    COMMAND ${CMAKE_CTEST_COMMAND} -L dtype --verbose
    COMMENT "Running data type tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_edge_cases_suite
    COMMAND ${CMAKE_CTEST_COMMAND} -L edge_cases --verbose
    COMMENT "Running edge case tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_performance_suite
    COMMAND ${CMAKE_CTEST_COMMAND} -L performance --verbose
    COMMENT "Running performance tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_integration_suite
    COMMAND ${CMAKE_CTEST_COMMAND} -L integration --verbose
    COMMENT "Running integration tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(test_compatibility_suite
    COMMAND ${CMAKE_CTEST_COMMAND} -L compatibility --verbose
    COMMENT "Running compatibility tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Main test target (all tests except debug)
add_custom_target(test_all
    COMMAND ${CMAKE_CTEST_COMMAND} --exclude-regex "debug" --verbose
    COMMENT "Running all production tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print test configuration summary
message(STATUS "Metal Backend Test Suite Configuration:")
message(STATUS "  Unit Tests: test_metal_unit_*")
message(STATUS "  Data Type Tests: test_metal_dtype_*") 
message(STATUS "  Edge Case Tests: test_metal_edge_cases*")
message(STATUS "  Performance Tests: test_metal_performance, test_*_benchmark, test_attention_*")
message(STATUS "  Integration Tests: Core Metal operations + full pipeline tests")
message(STATUS "  Compatibility Tests: test_*_compatibility, test_compilation_validation")
message(STATUS "  Debug Tests: test_*debug* (excluded from main test suite)")
message(STATUS "  Build directory: ${CMAKE_BINARY_DIR}")
message(STATUS "  Test binaries: ${CMAKE_BINARY_DIR}/bin/")
message(STATUS "  Metal shaders: ${CMAKE_BINARY_DIR}/shaders/")
message(STATUS "  Test execution: Use 'make test_all' or 'ctest' commands")