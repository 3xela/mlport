#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

struct ReductionCtx{
    const float* x;
    float* block_sums;
    int n;
};

__global__ void reduction_kernel_v0(const float* x, float* block_sums , int n) {
    __shared__ float smem[256];
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;

    float sum = 0.0f;

    for (int i = global_tid; i < n ; i+=grid_stride){
        sum+=x[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x/2 ; stride > 0 ; stride /= 2){
        if (tid < stride){
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0){
        block_sums[blockIdx.x] = smem[0];
    }
}

__global__ void reduction_kernel_v1(const float* x, float* block_sums , int n) {
    __shared__ float smem[256];
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = global_tid; i < n ; i+=grid_stride){
        sum+=x[i];
    }
    
    smem[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x/2 ; stride > 32 ; stride /= 2){
        if (tid < stride){
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    // reduction within warp 0
    volatile float* vsmem = smem;

    for (int stride = 32 ; stride > 0 ; stride /= 2){
        if (tid < stride){
            vsmem[tid] += vsmem[tid + stride];
        }
    }    
    if (tid == 0){
        block_sums[blockIdx.x] = smem[0];
    }
}

__global__ void reduction_kernel_v2(const float* x, float* block_sums , int n) {
    __shared__ float smem[256];
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = global_tid; i < n ; i+=grid_stride){
        sum+=x[i];
    }
    
    smem[tid] = sum;
    __syncthreads();
    
       for (int stride = blockDim.x/2 ; stride > 32 ; stride /= 2){
        if (tid < stride){
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    // reduction within warp 0
    volatile float* vsmem = smem;

    for (int stride = 32 ; stride > 0 ; stride /= 2){
        if (tid < stride){
            vsmem[tid] += vsmem[tid + stride];
        }
    }    
    if (tid == 0){
        block_sums[blockIdx.x] = smem[0];
    }
}

static inline void reduction_launch_v0(void* p) {
    auto* c = (ReductionCtx*)p;
    int grid = 152;
    reduction_kernel_v0<<<grid, 256>>>(c->x, c->block_sums ,c->n);
}

static inline void reduction_launch_v1(void* p) {
    auto* c = (ReductionCtx*)p;
    int grid = 152;
    reduction_kernel_v1<<<grid, 256>>>(c->x, c->block_sums ,c->n);
}

static inline void reduction_launch_v2(void* p) {
    auto* c = (ReductionCtx*)p;
    int grid = 152;
    reduction_kernel_v2<<<grid, 256>>>(c->x, c->block_sums ,c->n);
}

using LaunchFn = void(*)(void*);
