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

# Metal GEMM library
add_library(metal_gemm 
    src/metal_gemm.mm
)

target_link_libraries(metal_gemm 
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Set Objective-C++ compiler flags
set_target_properties(metal_gemm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Test executable
add_executable(test_metal_gemm
    src/test_metal_gemm.cpp
)

target_link_libraries(test_metal_gemm 
    metal_gemm
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Copy Metal shader to build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/src/metal_gemm.metal
    ${CMAKE_BINARY_DIR}/metal_gemm.metal
    COPYONLY
)