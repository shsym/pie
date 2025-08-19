cmake_minimum_required(VERSION 3.23)

project(metal_backend CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Objective-C++ for Metal integration
enable_language(OBJCXX)

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

# Test executables

# Final validation test for softmax
add_executable(final_test_softmax
    src/final_test_softmax.cpp
)

target_link_libraries(final_test_softmax 
    metal_softmax
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Copy Metal shaders to build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_gemm.metal
    ${CMAKE_BINARY_DIR}/metal_gemm.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_softmax.metal
    ${CMAKE_BINARY_DIR}/metal_softmax.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_embedding.metal
    ${CMAKE_BINARY_DIR}/metal_embedding.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_silu_and_mul.metal
    ${CMAKE_BINARY_DIR}/metal_silu_and_mul.metal
    COPYONLY
)

configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_extract_k_values.metal
    ${CMAKE_BINARY_DIR}/metal_extract_k_values.metal
    COPYONLY
)