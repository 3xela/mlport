#pragma once

#include "launch.cuh"

struct MatMulCtx {
    int N;
    int M;
    int K;
    float* A;
    float* B; 
    float* C;
    int block;
}

void matmul_launch_v0(void* p);