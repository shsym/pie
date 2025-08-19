# CUDA Protocol Tests - CUDA-only Reference Generation
cmake_minimum_required(VERSION 3.23)

project(cuda_protocol_tests CUDA CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compute repo root
get_filename_component(REPO_ROOT ${CMAKE_SOURCE_DIR} DIRECTORY)
set(CUDA_BACKEND_DIR "${REPO_ROOT}/backend/backend-cuda")

# Find CUDA toolkit
find_package(CUDA REQUIRED)

# Check for backend CUDA libraries
if(NOT EXISTS "${CUDA_BACKEND_DIR}/build/lib/libprefill_kernels.a")
    message(FATAL_ERROR "CUDA backend libraries not found. Please build backend-cuda first.")
endif()

# CUDA architecture selection
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_VERSION_MAJOR EQUAL 11)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
    elseif(CUDA_VERSION_MAJOR EQUAL 12)
        set(CMAKE_CUDA_ARCHITECTURES "70;80;86;89;90")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "70;80;86")
    endif()
endif()

# Sources
set(CUDA_SOURCES
    src/main.cpp
    src/ops_cuda.cu
)

add_executable(cuda_protocol_tests ${CUDA_SOURCES})

# Include directories
target_include_directories(cuda_protocol_tests PRIVATE
    src
    ${CUDA_BACKEND_DIR}/src
    ${CUDA_BACKEND_DIR}/third_party/flashinfer/include
    ${CUDA_BACKEND_DIR}/third_party/cutlass/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link libraries
target_link_libraries(cuda_protocol_tests PRIVATE
    ${CUDA_BACKEND_DIR}/build/lib/libprefill_kernels.a
    ${CUDA_BACKEND_DIR}/build/lib/libdecode_kernels.a
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cublasLt_LIBRARY}
    ${CUDA_curand_LIBRARY}
    pthread
)

# Add CUDA backend compilation definitions
target_compile_definitions(cuda_protocol_tests PRIVATE CUDA_SUPPORT_ENABLED)

# CUDA compiler flags
set_target_properties(cuda_protocol_tests PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)