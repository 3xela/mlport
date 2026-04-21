#include "kernels/matmul.h"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

static __global__ void matmul_kernel_v0(int N, int M, int K, float* A, float* B, float* C){

}

void matmul_launch_v0(void* p){
    auto* c = (MatMulCtx*)p;
    int grid = 152;
    matmul_kernel_v0<<<grid, 256>>>(c->N, c->M, c->K, c->A, c->B, c->C);
}