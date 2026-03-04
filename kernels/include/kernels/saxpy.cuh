#pragma once
#include <cuda_runtime.h>

__global__ void saxpy_kernel(const float* x, float* y, float a, float b, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] = a * x[i] + b;
}

struct SaxpyCtx {
    const float* x;
    float* y;
    float a;
    float b;
    int n;
    int block;
};

static inline void saxpy_launch(void* p) {
    auto* c = (SaxpyCtx*)p;
    int grid = (c->n + c->block - 1) / c->block;
    saxpy_kernel<<<grid, c->block>>>(c->x, c->y, c->a, c->b, c->n);
}
