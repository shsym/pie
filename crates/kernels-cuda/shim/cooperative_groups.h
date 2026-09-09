
#pragma once


namespace cooperative_groups {

#if defined(__CUDACC_RDC__)
#define PIE_CG_SCOPE_GRID (::cudaCGScopeGrid)
#else
extern "C" __device__ unsigned long long cudaCGGetIntrinsicHandle(unsigned int scope);
extern "C" __device__ unsigned int cudaCGSynchronize(unsigned long long handle, unsigned int flags);
#define PIE_CG_SCOPE_GRID (1u)
#endif

class thread_block {
public:

    __device__ __forceinline__ void sync() const { __syncthreads(); }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    }

    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ dim3 group_index() const { return blockIdx; }

    __device__ __forceinline__ dim3 thread_index() const { return threadIdx; }
};

__device__ __forceinline__ thread_block this_thread_block() { return thread_block{}; }

class grid_group {
public:

    __device__ __forceinline__ void sync() const {
        cudaCGSynchronize(cudaCGGetIntrinsicHandle(PIE_CG_SCOPE_GRID), 0u);
    }

    __device__ __forceinline__ unsigned long long thread_rank() const {
        const unsigned long long block =
            blockIdx.x + (unsigned long long)gridDim.x * (blockIdx.y + (unsigned long long)gridDim.y * blockIdx.z);
        const unsigned long long within =
            threadIdx.x + (unsigned long long)blockDim.x * (threadIdx.y + (unsigned long long)blockDim.y * threadIdx.z);
        return block * (blockDim.x * (unsigned long long)blockDim.y * blockDim.z) + within;
    }

    __device__ __forceinline__ unsigned long long size() const {
        return (unsigned long long)gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y *
               blockDim.z;
    }

    __device__ __forceinline__ unsigned int cluster_rank() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        unsigned int x, y, z, nx, ny;
        asm("mov.u32 %0, %%clusterid.x;" : "=r"(x));
        asm("mov.u32 %0, %%clusterid.y;" : "=r"(y));
        asm("mov.u32 %0, %%clusterid.z;" : "=r"(z));
        asm("mov.u32 %0, %%nclusterid.x;" : "=r"(nx));
        asm("mov.u32 %0, %%nclusterid.y;" : "=r"(ny));
        return x + nx * (y + ny * z);
#else
        return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
#endif
    }

    __device__ __forceinline__ unsigned int num_clusters() const {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        unsigned int x, y, z;
        asm("mov.u32 %0, %%nclusterid.x;" : "=r"(x));
        asm("mov.u32 %0, %%nclusterid.y;" : "=r"(y));
        asm("mov.u32 %0, %%nclusterid.z;" : "=r"(z));
        return x * y * z;
#else
        return gridDim.x * gridDim.y * gridDim.z;
#endif
    }
};

class cluster_group {
public:

    __device__ __forceinline__ unsigned int block_rank() const {
        unsigned int r;
        asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
        return r;
    }

    __device__ __forceinline__ unsigned int num_blocks() const {
        unsigned int r;
        asm("mov.u32 %0, %%cluster_nctarank;" : "=r"(r));
        return r;
    }

    __device__ __forceinline__ unsigned int thread_rank() const {
        const unsigned int within =
            threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
        return block_rank() * (blockDim.x * blockDim.y * blockDim.z) + within;
    }

    __device__ __forceinline__ unsigned int size() const {
        return num_blocks() * (blockDim.x * blockDim.y * blockDim.z);
    }

    __device__ __forceinline__ void sync() const {
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }

    template <typename T>
    __device__ __forceinline__ T* map_shared_rank(T* addr, int rank) const {
        void* out;
        asm("mapa.u64 %0, %1, %2;"
            : "=l"(out)
            : "l"((void*)addr), "r"((unsigned int)rank));
        return (T*)out;
    }
};

__device__ __forceinline__ cluster_group this_cluster() { return cluster_group{}; }

__device__ __forceinline__ grid_group this_grid() { return grid_group{}; }

}

namespace cg = cooperative_groups;
