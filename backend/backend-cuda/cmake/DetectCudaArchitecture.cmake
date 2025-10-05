# DetectCudaArchitecture.cmake CMake module for auto-detecting CUDA architecture
# based on available GPU

# Auto-detect CUDA architecture based on available GPU
function(detect_cuda_architectures)
  # Query GPU compute capability
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
    OUTPUT_VARIABLE GPU_COMPUTE_CAPS
    ERROR_VARIABLE NVIDIA_SMI_ERROR
    RESULT_VARIABLE NVIDIA_SMI_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NVIDIA_SMI_RESULT EQUAL 0 AND GPU_COMPUTE_CAPS)
    # Parse the compute capabilities (handle multiple GPUs)
    string(REPLACE "\n" ";" GPU_COMPUTE_LIST ${GPU_COMPUTE_CAPS})
    list(GET GPU_COMPUTE_LIST 0 FIRST_GPU_COMPUTE)

    # Convert compute capability to architecture number Remove the dot from
    # compute capability (e.g., "8.9" -> "89")
    string(REPLACE "." "" CUDA_ARCH ${FIRST_GPU_COMPUTE})

    message(
      STATUS
        "Detected CUDA compute capability: ${FIRST_GPU_COMPUTE} (architecture: ${CUDA_ARCH})"
    )
    set(CMAKE_CUDA_ARCHITECTURES
        "${CUDA_ARCH}"
        PARENT_SCOPE)
    return()
  endif()

  # If we reach here, nvidia-smi exists but failed to get GPU info
  message(
    FATAL_ERROR
      "CUDA Architecture Detection Failed: nvidia-smi is available but failed to query GPU information.\n"
  )
endfunction()
