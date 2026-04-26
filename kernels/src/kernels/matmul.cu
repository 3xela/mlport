#include "kernels/matmul.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

static __global__ void matmul_kernel_v0(const int M, const int K, const int N, const float* A, const float* B, float* C){

}

void matmul_launch_v0(void* p){
    auto* c = (MatMulCtx*)p;
    int grid = 152;
    matmul_kernel_v0<<<grid, 256>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}