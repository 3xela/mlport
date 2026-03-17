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
            smem[tid] += smem[tid+stride];
        }
        __syncthreads();
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

using LaunchFn = void(*)(void*);
