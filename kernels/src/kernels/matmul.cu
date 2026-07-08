#include "kernels/matmul.cuh"

#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cuda_fp16.h>   
#include <algorithm>

using namespace nvcuda;

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
    constexpr int BM = 128, BN = 128, BK = 32, pad = 1; // TODO make sure M, N, K are multiples of 128, 128, 32 resp. 
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int thread_row = ty * 8;
    int thread_col = tx * 8;

    int tid = ty * blockDim.x + tx;

    __shared__ float As[BK * (BM + pad)];
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

            As[A_tile_col * (BM + pad) + A_tile_row] = A[A_global_row * K + A_global_col];
            Bs[flat] = B[B_global_row * N + B_global_col];
        }
        __syncthreads();
        for (int k = 0; k < BK; k++){
            float a_frag[8];
            float b_frag[8];
            for(int l = 0; l < 8; l++){
                a_frag[l] = As[k * (BM + pad)+ (thread_row + l)]; 
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

static __global__ void wmma_matmul_kernel_v0(const int M, const int K, const int N, const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int ld_A = K;
    const int ld_B = N;
    const int ld_C = N;
    
    wmma::load_matrix_sync(a_frag, A, ld_A);
    wmma::load_matrix_sync(b_frag, B, ld_B);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(C, c_frag, ld_C, wmma::mem_row_major);

}

static __global__ void wmma_matmul_kernel_v1(const int M, const int K, const int N, const half* A, const half* B, float* C){
    int warp_row = blockIdx.y * 16;
    int warp_col = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int ld_A = K;
    const int ld_B = N;
    const int ld_C = N;
    
    for ( int i = 0; i< K/16; i++){
        wmma::load_matrix_sync(a_frag, A + warp_row  * K + i * 16, ld_A);
        wmma::load_matrix_sync(b_frag, B + i * 16 * N + warp_col, ld_B);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, ld_C, wmma::mem_row_major);

}

static __global__ void wmma_matmul_kernel_v2(const int M, const int K, const int N, const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    constexpr int BM = 32, BN = 64, BK = 32;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int wr = warp_id / 4;
    int wc = warp_id % 4;
    int warp_row = blockIdx.y * BM + wr * 16;
    int warp_col = blockIdx.x * BN + wc * 16;

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];
    
    const int ld_A = K;   // global strides (cooperative load + store)
    const int ld_B = N;
    const int ld_C = N;
            wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += BK){
        for (int i = 0; i < 4; i++){
            int flat = tid + 256 * i;
            int tile_row = flat / BK;
            int tile_col = flat % BK;
            As[flat] = A[(blockIdx.y * BM + tile_row) * K + (k_tile + tile_col)];
        }
        for (int i = 0; i < 8; i++){
            int flat = tid + 256 * i;
            int tile_row = flat / BN;
            int tile_col = flat % BN;
            Bs[flat] = B[(k_tile + tile_row) * N + (blockIdx.x * BN + tile_col)];
        }
        __syncthreads();
        for ( int ks = 0; ks < BK/16; ks++){
            wmma::load_matrix_sync(a_frag, As + (wr * 16) * BK + ks * 16, BK);
            wmma::load_matrix_sync(b_frag, Bs + (ks * 16) * BN + (wc * 16), BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, ld_C, wmma::mem_row_major);
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

void matmul_launch_v3(void* p){
    auto* c = (WMMAMulCtx*)p;
    wmma_matmul_kernel_v0<<<1,32>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v4(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/16, c->M/16);
    wmma_matmul_kernel_v1<<<grid,32>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v5(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/64, c->M/32);
    wmma_matmul_kernel_v2<<<grid,256>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}