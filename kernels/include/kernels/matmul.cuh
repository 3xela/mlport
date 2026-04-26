#pragma once

#include "launch.cuh"

struct MatMulCtx {
    int const M;
    int const K;
    int const N;
    float* const A;
    float* const B; 
    float* C;
    int block;
};

void matmul_launch_v0(void* p);