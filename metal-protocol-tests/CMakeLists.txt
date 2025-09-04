cmake_minimum_required(VERSION 3.23)

# Metal-only testing framework for macOS
project(metal_protocol_tests CXX)

# Enable Objective-C++ for Metal support on macOS
if(APPLE)
    enable_language(OBJCXX)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    message(STATUS "Metal support enabled on macOS")
else()
    message(FATAL_ERROR "This Metal-only build requires macOS")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C++ standard library for both C++ and Objective-C++
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -stdlib=libc++")

# Compute repo root
get_filename_component(REPO_ROOT ${CMAKE_SOURCE_DIR} DIRECTORY)

# Define Metal-only sources (unified test driver + ops wrappers)
set(METAL_SOURCES
    src/main.cpp
    src/ops_embedding_lookup.mm
    src/ops_silu_and_mul.mm
    src/ops_extract_k_values.mm
    src/ops_softmax.mm
    src/ops_rms_norm.mm
    src/ops_rope.mm
    src/ops_grouped_gemm.mm
    src/ops_batch_prefill_attention.mm
    src/ops_topk_mask_logits.mm
    src/ops_gemm.mm
    src/ops_add_residual.mm
    src/ops_append_paged_kv_cache.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_gemm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_embedding.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_silu_and_mul.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_extract_k_values.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_softmax.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_rmsnorm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_rope.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_topk_mask_logits.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_grouped_gemm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_batch_prefill_attention.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_attention_naive.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_add_residual.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_append_paged_kv_cache.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_common.mm
)

add_executable(metal_protocol_tests ${METAL_SOURCES})

set_target_properties(metal_protocol_tests PROPERTIES
    CXX_STANDARD 17
    OBJCXX_STANDARD 17
)

# Include directories
target_include_directories(metal_protocol_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${REPO_ROOT}/backend/backend-metal/src"
)

# Detect C++ standard library include path automatically
# This handles cases where the system has broken symlinks or missing headers
if(APPLE)
    # Find the actual C++ headers location
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDK_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Check if C++ headers exist in the standard location
    set(CXX_STDLIB_PATH "${SDK_ROOT}/usr/include/c++/v1")
    if(EXISTS "${CXX_STDLIB_PATH}/cstdint")
        target_include_directories(metal_protocol_tests SYSTEM PRIVATE ${CXX_STDLIB_PATH})
        message(STATUS "Using C++ stdlib from: ${CXX_STDLIB_PATH}")
    else()
        message(WARNING "C++ stdlib headers not found at expected location: ${CXX_STDLIB_PATH}")
        message(WARNING "You may need to install Xcode Command Line Tools: xcode-select --install")
    endif()
endif()

# Set Objective-C++ compiler flags for Metal sources
set_source_files_properties(
    src/ops_embedding_lookup.mm
    src/ops_silu_and_mul.mm
    src/ops_extract_k_values.mm
    src/ops_softmax.mm
    src/ops_rms_norm.mm
    src/ops_rope.mm
    src/ops_grouped_gemm.mm
    src/ops_batch_prefill_attention.mm
    src/ops_topk_mask_logits.mm
    src/ops_gemm.mm
    src/ops_add_residual.mm
    src/ops_append_paged_kv_cache.mm  # TODO: Implement backend kernel first
    ${REPO_ROOT}/backend/backend-metal/src/metal_gemm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_embedding.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_silu_and_mul.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_extract_k_values.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_softmax.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_rmsnorm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_rope.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_topk_mask_logits.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_grouped_gemm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_batch_prefill_attention.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_add_residual.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_append_paged_kv_cache.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_common.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# Link libraries
target_link_libraries(metal_protocol_tests PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Metal shader files are loaded directly from backend/backend-metal/src/ using __FILE__ path resolution

# Find nlohmann/json from Homebrew for organized tests
find_package(PkgConfig REQUIRED)
pkg_check_modules(NLOHMANN_JSON REQUIRED nlohmann_json)

# Find zstd for organized tests
find_library(ZSTD_LIBRARY zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR zstd.h)

# Set backend source paths for organized tests
set(BACKEND_SRC_PATH "../backend/backend-metal/src")
set(BACKEND_BUILD_PATH "../backend/backend-metal/build")

# Common include directories for organized tests
include_directories(${BACKEND_SRC_PATH})
include_directories(../backend/backend-cuda/src)  # For ztensor.hpp
include_directories(tests/src)
include_directories(${NLOHMANN_JSON_INCLUDE_DIRS})
include_directories(${ZSTD_INCLUDE_DIR})

# Link to backend build directory for libraries
link_directories(${BACKEND_BUILD_PATH})

# Backend source files needed for organized tests
set(BACKEND_SOURCES_ORGANIZED
    ${BACKEND_SRC_PATH}/metal_batch_prefill_attention.mm
    ${BACKEND_SRC_PATH}/metal_softmax.mm
    ${BACKEND_SRC_PATH}/metal_extract_k_values.mm
    ${BACKEND_SRC_PATH}/metal_topk_mask_logits.mm
    ${BACKEND_SRC_PATH}/metal_attention_naive.mm
    ${BACKEND_SRC_PATH}/metal_common.mm
    ${BACKEND_SRC_PATH}/metal_l4ma.mm
    ${BACKEND_SRC_PATH}/metal_rmsnorm.mm
    ${BACKEND_SRC_PATH}/metal_rmsnorm_wrapper.mm
    ${BACKEND_SRC_PATH}/metal_rope.mm
    ${BACKEND_SRC_PATH}/metal_rope_wrapper.mm
    ${BACKEND_SRC_PATH}/metal_silu_and_mul.mm
    ${BACKEND_SRC_PATH}/metal_silu_and_mul_wrapper.mm
    ${BACKEND_SRC_PATH}/metal_gemm.mm
    ${BACKEND_SRC_PATH}/metal_gemm_wrapper.mm
    ${BACKEND_SRC_PATH}/metal_embedding.mm
    ${BACKEND_SRC_PATH}/metal_buffer.mm
    ${BACKEND_SRC_PATH}/metal_tensor.mm
    ${BACKEND_SRC_PATH}/metal_kv_cache.mm
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

# Common libraries for organized tests
set(COMMON_LIBRARIES_ORGANIZED
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${ZSTD_LIBRARY}
)

# Set output directory for organized test executables
set(ORGANIZED_TESTS_BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/bin)
file(MAKE_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR})

# ============================================================================
# Organized Unit Tests (Core functionality - fast execution)
# ============================================================================

add_executable(test_softmax_unit
    tests/src/unit/test_softmax.cpp
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_softmax_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
set_target_properties(test_softmax_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

add_executable(test_extract_k_values_unit
    tests/src/unit/test_extract_k_values.cpp
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_extract_k_values_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
set_target_properties(test_extract_k_values_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

add_executable(test_topk_mask_unit
    tests/src/unit/test_topk_mask.cpp
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_topk_mask_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
set_target_properties(test_topk_mask_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# Note: test_batch_attention_unit disabled due to missing metal_batch_prefill_handle.hpp
# add_executable(test_batch_attention_unit
#     tests/src/stress/test_handle_stress.mm
#     ${BACKEND_SOURCES_ORGANIZED}
# )
# target_link_libraries(test_batch_attention_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
# set_target_properties(test_batch_attention_unit PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
# )

add_executable(test_kernel_selection_unit
    tests/src/unit/test_kernel_selection.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_kernel_selection_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
set_target_properties(test_kernel_selection_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

add_executable(test_memory_management_unit
    tests/src/unit/test_memory_management.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_memory_management_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
set_target_properties(test_memory_management_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# zTensor reader test (for weight loading infrastructure)
add_executable(test_ztensor_reader_unit
    tests/src/unit/test_ztensor_reader.cpp
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_ztensor_reader_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
target_compile_definitions(test_ztensor_reader_unit PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_ztensor_reader_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# Real zTensor file test (for weight loading with actual model)
add_executable(test_real_ztensor_unit
    tests/src/unit/test_real_ztensor.cpp
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_real_ztensor_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
target_compile_definitions(test_real_ztensor_unit PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_real_ztensor_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# MetalTensor loading test (for weight loading from zTensor files)
add_executable(test_metal_tensor_loading_unit
    tests/src/unit/test_metal_tensor_loading.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_metal_tensor_loading_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
target_compile_definitions(test_metal_tensor_loading_unit PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_metal_tensor_loading_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

add_executable(test_memory_optimization_unit
    tests/src/unit/test_memory_optimization.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_memory_optimization_unit ${COMMON_LIBRARIES_ORGANIZED} ${BACKEND_LIBS})
target_compile_definitions(test_memory_optimization_unit PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_memory_optimization_unit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# Model weight loading integration test
add_executable(test_model_weight_loading_integration
    tests/src/integration/test_model_weight_loading.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_model_weight_loading_integration ${COMMON_LIBRARIES_ORGANIZED})
target_compile_definitions(test_model_weight_loading_integration PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_model_weight_loading_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# Forward pass integration test
add_executable(test_forward_pass_integration
    tests/src/integration/test_forward_pass.mm
    ${BACKEND_SOURCES_ORGANIZED}
)
target_link_libraries(test_forward_pass_integration ${COMMON_LIBRARIES_ORGANIZED})
target_compile_definitions(test_forward_pass_integration PRIVATE ${NLOHMANN_JSON_CFLAGS_OTHER})
set_target_properties(test_forward_pass_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${ORGANIZED_TESTS_BIN_DIR}
)

# ============================================================================
# Custom targets for organized test execution
# ============================================================================

# Unit tests (fast smoke tests)
add_custom_target(run_unit_tests
    COMMAND echo "=== Running Unit Tests - under 1 minute ==="
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_softmax_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_extract_k_values_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_topk_mask_unit
# COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_batch_attention_unit  # Disabled
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_kernel_selection_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_memory_management_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_ztensor_reader_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_real_ztensor_unit
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_metal_tensor_loading_unit
    COMMAND echo "=== Running Integration Tests ==="
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_model_weight_loading_integration
    COMMAND ${ORGANIZED_TESTS_BIN_DIR}/test_forward_pass_integration
    COMMAND echo "✅ All tests completed"
    DEPENDS test_softmax_unit test_extract_k_values_unit test_topk_mask_unit test_kernel_selection_unit test_memory_management_unit test_ztensor_reader_unit test_real_ztensor_unit test_metal_tensor_loading_unit test_model_weight_loading_integration test_forward_pass_integration
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Test statistics
add_custom_target(test_stats
    COMMAND echo "=== Metal Protocol Tests Statistics ==="
    COMMAND echo "Main Protocol Test:     1 executable - metal_protocol_tests"
    COMMAND echo "Organized Unit Tests:   8 tests  - under 1 minute"
    COMMAND echo "Integration Tests:      2 tests  - model weight loading, forward pass"
    COMMAND echo ""
    COMMAND echo "Available targets:"
    COMMAND echo "  metal_protocol_tests   - Main protocol test executable"
    COMMAND echo "  run_unit_tests         - All organized unit + integration tests"
    COMMAND echo "  test_stats             - This statistics summary"
    COMMAND echo "  clean_all              - Clean all build artifacts"
)

# Clean all build artifacts
add_custom_target(clean_all
    COMMAND echo "Cleaning all Metal protocol test build artifacts..."
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND rm -rf ${CMAKE_BINARY_DIR}
    COMMAND echo "✅ All build artifacts cleaned"
    COMMENT "Clean all build artifacts for metal protocol tests"
)