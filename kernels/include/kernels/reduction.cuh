#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

struct ReductionCtx{
    const float* x;
    float* block_sums;
    float* final_sum;
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
    // total # of warps: #blocks * #threads/block / 32 = 152 * 256 / 32 = 1216
    __shared__ float smem[8];
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    float sum = 0.0f;

    for (int i = global_tid; i < n ; i+=grid_stride){
        sum+=x[i];
    }

    for (int offset = 16; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // check  lane id = 0 
    if ( lane_id ==  0){
        smem[warp_id] = sum;
    }
    __syncthreads();

    // load sum as register
    float tmp = 0.0f;
    if (warp_id == 0){
        if (lane_id < num_warps){
            tmp = smem[lane_id];
        }
        for (int offset = 16; offset > 0; offset /= 2){
            tmp += __shfl_down_sync(0xffffffff, tmp, offset);
        }
    }

    if (tid == 0){
        block_sums[blockIdx.x] = tmp;
    }

}

__global__ void reduction_kernel_v4(const float* x, float* block_sums , int n) {
    // total # of warps: #blocks * #threads/block / 32 = 152 * 256 / 32 = 1216
    __shared__ float smem[8];
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int grid_stride = gridDim.x * blockDim.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    const int num_warps = blockDim.x / 32;

    float sum = 0.0f;

    for (int i = global_tid; i < n ; i+= 2 * grid_stride){
        sum+=x[i];
        if (i + grid_stride < n){
            sum+=x[i + grid_stride];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2){
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // check  lane id = 0 
    if ( lane_id ==  0){
        smem[warp_id] = sum;
    }
    __syncthreads();

    // load sum as register
    float tmp = 0.0f;
    if (warp_id == 0){
        if (lane_id < num_warps){
            tmp = smem[lane_id];
        }
        for (int offset = 16; offset > 0; offset /= 2){
            tmp += __shfl_down_sync(0xffffffff, tmp, offset);
        }
    }

    if (tid == 0){
        block_sums[blockIdx.x] = tmp;
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

static inline void reduction_launch_v4(void* p) {
    auto* c = (ReductionCtx*)p;
    int grid = 152;
    reduction_kernel_v4<<<grid, 256>>>(c->x, c->block_sums ,c->n);
}

static inline void full_reduction_launch(void* p){
    auto* c = (ReductionCtx*)p;
    int grid = 152;

    reduction_kernel_v2<<<grid, 256>>>(c->x, c->block_sums ,c->n);
    reduction_kernel_v2<<<1, 256>>>(c->block_sums, c->final_sum, 152);
}

using LaunchFn = void(*)(void*);
