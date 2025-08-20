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
    src/ops_metal.mm
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
)

add_executable(metal_protocol_tests ${METAL_SOURCES})

# Include directories
target_include_directories(metal_protocol_tests PRIVATE
    ${REPO_ROOT}/backend/backend-metal/src
    src
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
    src/ops_metal.mm
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
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# Link libraries
target_link_libraries(metal_protocol_tests PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Metal shader files are loaded directly from backend/backend-metal/src/ using __FILE__ path resolution