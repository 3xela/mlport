#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

struct AxpyCtx {
    const float* x;
    float* y;
    float a;
    int n;
    int block;
};

__global__ void axpy_kernel_v0(const float* x, float* y, float a, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n){ 
        y[i] = a * x[i] + y[i];
    }
}

static inline void axpy_launch_v0(void* p) {
    auto* c = (AxpyCtx*)p;
    int grid = (c->n + c->block - 1) / c->block;
    axpy_kernel_v0<<<grid, c->block>>>(c->x, c->y, c->a, c->n);
}

__global__ void axpy_kernel_v1(const float* x, float* y, float a, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride){ 
        y[i] = a * x[i] + y[i];
    }
}

static inline void axpy_launch_v1(void* p) {
    auto* c = (AxpyCtx*)p;
    int cap = 152;
    int grid_full = (c->n  + c->block -1)/ c->block;
    int grid = std::min(grid_full, cap);
    axpy_kernel_v1<<<grid, c->block>>>(c->x, c->y, c->a, c->n);
}
