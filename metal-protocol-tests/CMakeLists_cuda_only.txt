cmake_minimum_required(VERSION 3.23)

# Mirror CUDA detection from backend/backend-cuda/CMakeLists.txt
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  if(EXISTS "/usr/local/cuda/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
    message(STATUS "Auto-detected nvcc at /usr/local/cuda/bin/nvcc")
  elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
    message(STATUS "Auto-detected nvcc at $ENV{CUDA_HOME}/bin/nvcc")
  elseif(DEFINED ENV{CUDA_PATH} AND EXISTS "$ENV{CUDA_PATH}/bin/nvcc")
    set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
    message(STATUS "Auto-detected nvcc at $ENV{CUDA_PATH}/bin/nvcc")
  endif()
endif()

# Compute repo root (parent of this project dir)
get_filename_component(REPO_ROOT ${CMAKE_SOURCE_DIR} DIRECTORY)

# Detect CUDA architecture using the same helper from backend-cuda
include(${REPO_ROOT}/backend/backend-cuda/cmake/DetectCudaArchitecture.cmake)
detect_cuda_architectures()
message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")

project(metal_protocol_tests CUDA CXX)

# Enable Objective-C++ for Metal support on macOS
if(APPLE)
    enable_language(OBJCXX)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    set(METAL_SUPPORT ON)
    message(STATUS "Metal support enabled on macOS")
else()
    set(METAL_SUPPORT OFF)
    message(STATUS "Metal support disabled (not on macOS)")
endif()

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Define source files
set(COMMON_SOURCES
    src/main.cpp
    src/ops_cuda.cu
    ${REPO_ROOT}/backend/backend-cuda/src/common.cu
)

# Add Metal sources if supported
if(METAL_SUPPORT)
    list(APPEND COMMON_SOURCES
        src/ops_metal.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_gemm.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_embedding.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_silu_and_mul.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_extract_k_values.mm
    )
endif()

add_executable(metal_protocol_tests ${COMMON_SOURCES})

# Include directories
target_include_directories(metal_protocol_tests PRIVATE
    ${REPO_ROOT}/backend/backend-cuda/src
    ${REPO_ROOT}/backend/backend-cuda/build/_deps/flashinfer-src/include
    ${REPO_ROOT}/backend/backend-cuda/build/_deps/flashinfer-src/src/generated
)

# Add Metal include directories if supported
if(METAL_SUPPORT)
    target_include_directories(metal_protocol_tests PRIVATE
        ${REPO_ROOT}/backend/backend-metal/src
    )
endif()

# Link directories for FlashInfer kernel libs produced by backend-cuda build
target_link_directories(metal_protocol_tests PRIVATE
  ${REPO_ROOT}/backend/backend-cuda/build/lib
)

set_target_properties(metal_protocol_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add CUDA compiler flags for FlashInfer compatibility
target_compile_options(metal_protocol_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)

target_compile_options(metal_protocol_tests PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)
if (MSVC)
  target_compile_options(metal_protocol_tests PRIVATE /W4 /permissive-)
else()
endif()

# Link libraries
target_link_libraries(metal_protocol_tests PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    prefill_kernels
    decode_kernels
)

# Add Metal frameworks if supported
if(METAL_SUPPORT)
    target_link_libraries(metal_protocol_tests PRIVATE
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )
    
    # Set Objective-C++ compiler flags for Metal sources
    set_source_files_properties(
        src/ops_metal.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_gemm.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_embedding.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_silu_and_mul.mm
        ${REPO_ROOT}/backend/backend-metal/src/metal_extract_k_values.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
    
    # Add Metal support compilation definition
    target_compile_definitions(metal_protocol_tests PRIVATE METAL_SUPPORT_ENABLED)
endif()
