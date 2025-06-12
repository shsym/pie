# 1. Set minimum CMake version and define the project
# This enables CUDA as a first-class language.
cmake_minimum_required(VERSION 3.23)
project(PIE_BACKEND CUDA CXX)


# 2. Set C++ Standard and output directories
# Ensures C++20 is used, matching the ninja file's flag.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Define output directories for binaries, matching the ninja file structure.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_CUDA_ARCHITECTURES "89")

find_package(CUDAToolkit REQUIRED)

# 3. Include CPM.cmake for package management
# This downloads the CPM script if it's not already present.
include(cmake/CPM.cmake)

# 4. Declare and fetch all dependencies using CPM
# CPM downloads, configures, and makes targets available for all dependencies.
# This replaces manual submodule management.
CPMAddPackage(
    URI "gh:flashinfer-ai/flashinfer@0.2.6"
    OPTIONS "FLASHINFER_ENABLE_FP8 OFF" "FLASHINFER_ENABLE_FP8_E4M3 OFF" "FLASHINFER_ENABLE_FP8_E5M2 OFF"
)

CPMAddPackage(
    URI "gh:NVIDIA/cuembed#main@0.1.0"
    OPTIONS "BUILD_TESTS OFF" "BUILD_BENCHMARKS OFF"
)

CPMAddPackage(
    URI "gh:zeromq/cppzmq@4.10.0"
)

CPMAddPackage(
    URI "gh:nlohmann/json@3.12.0"
    OPTIONS "JSON_BuildTests OFF"
)

CPMAddPackage(
    URI "gh:jbeder/yaml-cpp#0.8.0"
    OPTIONS "YAML_CPP_BUILD_TESTS OFF"
)

# 5. Define the main executable target
# Add all your source files here.
add_executable(cuda_app
    src/main.cu
    src/l4ma.cu
    src/bpe.cpp
    src/common.cu
)

# 6. Set compiler flags and options for the target
# These flags are applied specifically to 'cuda_app'.
# This mirrors the 'cxxflags' from your ninja file.
target_compile_options(cuda_app PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        --extended-lambda
        -g # Enable host debug symbols
        -G # Enable device-side debug symbols
        -gencode=arch=compute_89,code=sm_89
    >
)

# 7. Link libraries and include directories
# CPM automatically handles include directories for its packages when you link them.
# We just need to add our own 'include' directory and link the required libraries.
target_include_directories(cuda_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${flashinfer_SOURCE_DIR}/include 
    ${cuembed_SOURCE_DIR}
)

target_link_libraries(cuda_app PRIVATE
    # Link against targets provided by CPM packages
    cppzmq
    yaml-cpp
    
    nlohmann_json::nlohmann_json

    # Link against CUDA libraries provided by CMake's FindCUDA module
    CUDA::cublas
    CUDA::cublasLt
)

# 8. (Optional) Add a custom 'run' target
# This allows you to build and run the application with a single command.
# e.g., 'ninja run' or 'make run'
add_custom_target(run
    COMMAND $<TARGET_FILE:cuda_app>
    DEPENDS cuda_app
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Running cuda_app..."
)

# --- End of CMakeLists.txt ---