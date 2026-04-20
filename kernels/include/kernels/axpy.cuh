#pragma once

#include "launch.cuh"

struct AxpyCtx {
    const float* x;
    float* y;
    float a;
    int n;
    int block;
};

void axpy_launch_v0(void* p);

void axpy_launch_v1(void* p);

void axpy_launch_v2(void* p);
