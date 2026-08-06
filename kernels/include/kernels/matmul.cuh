#pragma once
#include <cuda_fp16.h>   
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

struct WMMAMulCtx {
    int const M;
    int const K;
    int const N;
    half* const A;
    half* const B; 
    float* C;
    int block;
};

void matmul_launch_v0(void* p);
void matmul_launch_v1(void* p);
void matmul_launch_v2(void* p);
void matmul_launch_v3(void* p);
void matmul_launch_v4(void* p);
void matmul_launch_v5(void* p);
void matmul_launch_v6(void* p);
void matmul_launch_v7(void* p);
