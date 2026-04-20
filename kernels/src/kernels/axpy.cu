#include "kernels/axpy.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

static __global__ void axpy_kernel_v0(const float* x, float* y, float a, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n){ 
        y[i] = a * x[i] + y[i];
    }
}

void axpy_launch_v0(void* p) {
    auto* c = (AxpyCtx*)p;
    int grid = (c->n + c->block - 1) / c->block;
    axpy_kernel_v0<<<grid, c->block>>>(c->x, c->y, c->a, c->n);
}

static __global__ void axpy_kernel_v1(const float* x, float* y, float a, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride){ 
        y[i] = a * x[i] + y[i];
    }
}

void axpy_launch_v1(void* p) {
    auto* c = (AxpyCtx*)p;
    int cap = 304;
    int grid_full = (c->n  + c->block -1)/ c->block;
    int grid = std::min(grid_full, cap);
    axpy_kernel_v1<<<grid, c->block>>>(c->x, c->y, c->a, c->n);
}

static __global__ void axpy_kernel_v2(const float* x, float* y, float a, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int n4 = n/4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    for (int i = tid; i < n4; i += stride){ 
        float4 xv = x4[i];
        float4 yv = y4[i];

        yv.x = a * xv.x + yv.x;
        yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z;
        yv.w = a * xv.w + yv.w;

        y4[i] = yv;
    }
    for (int i = 4 * n4 + tid; i < n ; i+= stride){
        y[i] = a * x[i] + y[i];
    }
}

void axpy_launch_v2(void* p) {
    auto* c = (AxpyCtx*)p;
    int cap = 304;
    int grid_full = (c->n  + c->block -1)/ c->block;
    int grid = std::min(grid_full, cap);
    axpy_kernel_v2<<<grid, c->block>>>(c->x, c->y, c->a, c->n);
}
