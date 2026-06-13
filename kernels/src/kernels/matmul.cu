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
    constexpr int BM = 128, BN = 128, BK = 32; // TODO make sure M, N, K are multiples of 128, 128, 32 resp. 
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int thread_row = ty * 8;
    int thread_col = tx * 8;

    int tid = ty * blockDim.x + tx;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    float acc[64] = {0.0f};

    for (int k_tile = 0; k_tile < K; k_tile += BK){
        for (int load_iter = 0; load_iter < 16; load_iter++){
            int flat = load_iter * 256 + tid;

            int A_tile_row = flat / BK; 
            int A_tile_col = flat % BK;

            int B_tile_row = flat / BN;
            int B_tile_col = flat % BN;

            int A_global_row = block_row + A_tile_row;
            int A_global_col = k_tile + A_tile_col;

            int B_global_row = k_tile + B_tile_row;
            int B_global_col = block_col + B_tile_col;

            As[flat] = A[A_global_row * K + A_global_col];
            Bs[flat] = B[B_global_row * N + B_global_col];
        }
        __syncthreads();
        for (int k = 0; k < BK; k++){
            float a_frag[8];
            float b_frag[8];
            for(int l = 0; l < 8; l++){
                a_frag[l] = As[(thread_row + l) * BK + k];
                b_frag[l] = Bs[k * BN + (thread_col + l)];
            }
            for (int i = 0; i < 8; i++){ //row
                for (int j = 0; j < 8; j++){ //col
                    acc[i * 8 + j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0 ; i < 8 ; i++){
        for (int j = 0; j < 8; j++){
            C[(block_row + thread_row + i) * N + (block_col + thread_col + j)]= acc[i * 8 + j];
        }
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
    dim3 grid((c->N + 127) / 128, (c->M + 127) / 128);
    matmul_kernel_v2<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}