# CUDA Protocol Tests - CUDA-only Reference Generation
cmake_minimum_required(VERSION 3.23)

# --- CUDA compiler auto-detection (must run before project()) ---
# Mirror backend/backend-cuda logic so configuring this subtree works standalone.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED CUDACXX)
    if(EXISTS "/usr/local/cuda/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
        message(STATUS "Auto-detected nvcc at /usr/local/cuda/bin/nvcc")
    elseif(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc" CACHE FILEPATH "Path to nvcc compiler" FORCE)
        message(STATUS "Auto-detected nvcc at $ENV{CUDA_HOME}/bin/nvcc")
    endif()
endif()

project(cuda_protocol_tests CUDA CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compute repo root
get_filename_component(REPO_ROOT ${CMAKE_SOURCE_DIR} DIRECTORY)
set(CUDA_BACKEND_DIR "${REPO_ROOT}/backend/backend-cuda")

# Find CUDA toolkit (modern package name)
find_package(CUDAToolkit REQUIRED)

# Enforce a minimum CUDA Toolkit version (12.0+) for modern features used by FlashInfer
if(CUDAToolkit_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "CUDA Toolkit 12.0+ is required. Found ${CUDAToolkit_VERSION}.")
endif()

# Check for backend CUDA libraries
set(BACKEND_LIB_DIR "${CUDA_BACKEND_DIR}/build/lib")
if(NOT EXISTS "${BACKEND_LIB_DIR}/libprefill_kernels.a")
    message(FATAL_ERROR "CUDA backend libraries not found (libprefill_kernels.a). Please build backend-cuda first.")
endif()

# CUDA architecture policy
# We primarily target Ampere+ (sm_80 3090, sm_89 4090). Ban < 80 to avoid unsupported paths.
set(CUDA_REQUIRED_ARCHS "80;86;89")
# Force the global cache var so parent or prior cache values don't leak in
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_REQUIRED_ARCHS}" CACHE STRING "CUDA architectures" FORCE)

# Validate that all requested archs are >= 80
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 80)
        message(FATAL_ERROR "Unsupported GPU architecture sm_${arch}. Please use sm_80 or newer (e.g., 80, 86, 89).")
    endif()
endforeach()
message(STATUS "cuda_protocol_tests using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Sources
set(CUDA_SOURCES
    src/main.cpp
    src/ops_cuda.cu
    src/ops/ops_common.cuh
    ${CUDA_BACKEND_DIR}/src/common.cu
    src/ops/gemm.cu
    src/ops/embedding_lookup.cu
    src/ops/extract_k_values.cu
    src/ops/rms_norm.cu
    src/ops/silu_and_mul.cu
    src/ops/add_residual.cu
    src/ops/rope.cu
    src/ops/topk_mask_logits.cu
    src/ops/softmax.cu
    src/ops/batch_prefill_attention.cu
    src/ops/grouped_gemm.cu
    src/ops/append_paged_kv_cache.cu
    src/test_kernels/embed_fp16.cu
    src/test_kernels/extract_k_values_fp16.cu
    src/test_kernels/gemm_fp16.cu
)

add_executable(cuda_protocol_tests ${CUDA_SOURCES})

# Enforce architectures on the target
set_property(TARGET cuda_protocol_tests PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# NVCC tuning and warnings control
target_compile_options(
    cuda_protocol_tests PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Wno-deprecated-gpu-targets
    -Xcompiler=-Wno-sign-compare
        -diag-suppress=177
    >
)

# Include directories
target_include_directories(cuda_protocol_tests PRIVATE
    src
    ${CUDA_BACKEND_DIR}/src
    # Use backend's CPM-fetched flashinfer headers
    ${CUDA_BACKEND_DIR}/build/_deps/flashinfer-src/include
    ${CUDA_BACKEND_DIR}/build/_deps/flashinfer-src/src
    ${CUDA_BACKEND_DIR}/build/_deps/flashinfer-src/src/generated
    ${CUDA_BACKEND_DIR}/third_party/cutlass/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(cuda_protocol_tests PRIVATE
    ${BACKEND_LIB_DIR}/libprefill_kernels.a
    ${BACKEND_LIB_DIR}/libdecode_kernels.a
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
    pthread
)

# Add CUDA backend compilation definitions
target_compile_definitions(cuda_protocol_tests PRIVATE CUDA_SUPPORT_ENABLED)

# CUDA compiler flags
set_target_properties(cuda_protocol_tests PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
)

# Match backend CUDA compile flags needed by FlashInfer headers
target_compile_options(
    cuda_protocol_tests
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                    -Xcompiler=-Wall
                    -Xcompiler=-Wextra
                    -Wno-unused-parameter
                    --extended-lambda
                    --expt-relaxed-constexpr
                    >)