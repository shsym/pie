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
)

add_executable(metal_protocol_tests ${METAL_SOURCES})

# Include directories
target_include_directories(metal_protocol_tests PRIVATE
    ${REPO_ROOT}/backend/backend-metal/src
    src
)

# Set Objective-C++ compiler flags for Metal sources
set_source_files_properties(
    src/ops_metal.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_gemm.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_embedding.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_silu_and_mul.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_extract_k_values.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_softmax.mm
    ${REPO_ROOT}/backend/backend-metal/src/metal_rmsnorm.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

# Link libraries
target_link_libraries(metal_protocol_tests PRIVATE
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Add Metal support compilation definition
target_compile_definitions(metal_protocol_tests PRIVATE METAL_SUPPORT_ENABLED)

# Metal shader files are loaded directly from backend/backend-metal/src/ using __FILE__ path resolution