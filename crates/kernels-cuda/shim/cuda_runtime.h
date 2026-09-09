

#ifndef PIE_NVRTC_CUDA_RUNTIME_H_
#define PIE_NVRTC_CUDA_RUNTIME_H_

#include <cstdint>

typedef unsigned short ushort;
typedef unsigned char uchar;

#if !defined(__CUDACC_RDC__)
__device__ __forceinline__ void cudaGridDependencySynchronize() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void cudaTriggerProgrammaticLaunchCompletion() {
  asm volatile("griddepcontrol.launch_dependents;");
}
#endif

#endif
