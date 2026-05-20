#include "kernels/matmul.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

static __global__ void matmul_kernel_v0(const int M, const int K, const int N, const float* A, const float* B, float* C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N){
        float acc = 0.0f;
        for (int k = 0; k < K; k++){
            acc += A[ i * K + k ] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }

}

static __global__ void matmul_kernel_v1(const int M, const int K, const int N, const float* A, const float* B, float* C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    constexpr int BM = 32, BN = 32, BK = 32;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    if (i < M && j < N){
        float acc = 0.0f;
        for (int k_tile = 0; k_tile < K; k_tile += BK){
            As[ty * BK + tx] = A[i * K + k_tile + tx]; 
            Bs[ty * BN + tx] = B[(k_tile + ty) * N + j];
            __syncthreads();
            for (int k = 0; k < BK; k++){
                acc += As[ty * BK + k] * Bs[k * BN + tx ];
            }
            __syncthreads();
        }
        C[i * N + j] = acc;
    }

}

static __global__ void matmul_kernel_v2(const int M, const int K, const int N, const float* A, const float* B, float* C){
    constexpr int BM = 128, BN = 128, BK = 32;
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int thread_row = ty * 8;
    int thread_col = tx * 8;

    int tid = ty * blockDim.x + tx;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float acc[64];

    for (int k_tile = 0; k_tile < K; k_tile += BK){
        As[ty * BK + tx] = A[ty * K + k_tile + tx]; 
        Bs[ty * BN + tx] = B[(k_tile + ty) * N + tx];
        __syncthreads();
        for (int k = 0; k < BK; k++){
            float a_frag[8];
            float b_frag[8];

            for (int j = 0; j < 8; j++){
                a_frag[j] = As[ty * j + tx]
                b_frag[j] = Bs[ty * BN + j]
            }
        }
        __syncthreads();
    }

}

void matmul_launch_v0(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(16,16);
    dim3 grid((c->N + 15) / 16, (c->M + 15) / 16);
    matmul_kernel_v0<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v1(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(32,32);
    dim3 grid((c->N + 31) / 32, (c->M + 31) / 32);
    matmul_kernel_v1<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v2(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(16,16);
    dim3 grid((c->N + 31) / 32, (c->M + 31) / 32);
    matmul_kernel_v2<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}