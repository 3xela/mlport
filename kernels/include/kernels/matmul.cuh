#pragma once

#include "launch.cuh"

struct MatMulCtx {
    int M;
    int K;
    int N;
    const float* A;
    const float* B; 
    float* C;
    int block;
}

void matmul_launch_v0(void* p);