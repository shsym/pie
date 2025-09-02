cmake_minimum_required(VERSION 3.23)

project(metal_backend CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C++ for Metal integration
enable_language(OBJCXX)

# Use libc++ explicitly and ensure headers are discoverable on macOS
if(APPLE)
    # Prefer libc++ for both C++ and Objective-C++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} -stdlib=libc++")

    # Detect the SDK C++ headers path (works with Command Line Tools and full Xcode)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE SDK_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(CXX_STDLIB_PATH "${SDK_ROOT}/usr/include/c++/v1")
    if(EXISTS "${CXX_STDLIB_PATH}/cstdint")
        # Add as a system include to avoid warnings from standard headers
        include_directories(SYSTEM ${CXX_STDLIB_PATH})
        message(STATUS "Using C++ stdlib from: ${CXX_STDLIB_PATH}")
    else()
        message(WARNING "C++ stdlib headers not found at expected location: ${CXX_STDLIB_PATH}")
        message(WARNING "You may need to install Xcode Command Line Tools: xcode-select --install")
    endif()
endif()

# Optimization options
option(USE_SIMDGROUP_REDUCTIONS "Enable Metal simdgroup optimizations for attention kernels" OFF)

# Find required frameworks on macOS
if(APPLE)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
else()
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

# Include directories
include_directories(src)

# Metal libraries (define libraries first, then executables)

# Metal GEMM library
add_library(metal_gemm
    src/metal_gemm.mm
)

target_link_libraries(metal_gemm
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_gemm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal Softmax library
add_library(metal_softmax
    src/metal_softmax.mm
)

target_link_libraries(metal_softmax
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_softmax PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal Embedding library
add_library(metal_embedding
    src/metal_embedding.mm
)

target_link_libraries(metal_embedding
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_embedding PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal SiLU and Mul library
add_library(metal_silu_and_mul
    src/metal_silu_and_mul.mm
)

target_link_libraries(metal_silu_and_mul
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_silu_and_mul PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal Extract K Values library
add_library(metal_extract_k_values
    src/metal_extract_k_values.mm
)

target_link_libraries(metal_extract_k_values
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_extract_k_values PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal TopK Mask Logits library
add_library(metal_topk_mask_logits
    src/metal_topk_mask_logits.mm
)

target_link_libraries(metal_topk_mask_logits
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_topk_mask_logits PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal Batch Prefill Attention library
add_library(metal_batch_prefill_attention
    src/metal_batch_prefill_attention.mm
)

target_link_libraries(metal_batch_prefill_attention
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_batch_prefill_attention PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Add simdgroup optimization flag to Metal shaders
if(USE_SIMDGROUP_REDUCTIONS)
    target_compile_definitions(metal_batch_prefill_attention PRIVATE USE_SIMDGROUP_REDUCTIONS=1)
endif()

# Metal Data Type Conversion library
add_library(metal_dtype_conversion
    src/metal_dtype_conversion.mm
)

target_link_libraries(metal_dtype_conversion
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_dtype_conversion PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Metal Common library
add_library(metal_common
    src/metal_common.mm
)

target_link_libraries(metal_common
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

set_target_properties(metal_common PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Create shaders directory and copy Metal shaders
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_gemm.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_gemm.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_softmax.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_softmax.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_embedding.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_embedding.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_silu_and_mul.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_silu_and_mul.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_extract_k_values.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_extract_k_values.metal
    COPYONLY
)

# Copy Metal attention shaders
configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_batch_prefill_attention.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_batch_prefill_attention.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_attention_common.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_attention_common.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_attention_baseline.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_attention_baseline.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_attention_simdgroup_opt.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_attention_simdgroup_opt.metal
    COPYONLY
)

# Commented out - these files no longer exist after cleanup
# configure_file(
#     ${CMAKE_SOURCE_DIR}/src/metal_batch_prefill_attention_optimized.metal
#     ${CMAKE_BINARY_DIR}/shaders/metal_batch_prefill_attention_optimized.metal
#     COPYONLY
# )

# configure_file(
#     ${CMAKE_SOURCE_DIR}/src/metal_batch_prefill_attention_unified.metal
#     ${CMAKE_BINARY_DIR}/shaders/metal_batch_prefill_attention_unified.metal
#     COPYONLY
# )

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_topk_mask_logits.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_topk_mask_logits.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_rmsnorm.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_rmsnorm.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_rope.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_rope.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_grouped_gemm.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_grouped_gemm.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_add_residual.metal
    ${CMAKE_BINARY_DIR}/shaders/metal_add_residual.metal
    COPYONLY
)

# Include tests subdirectory only if present (optional local tests/executables)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()