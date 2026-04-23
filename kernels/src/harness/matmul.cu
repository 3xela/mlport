#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "launch.cuh"

#include "kernels/matmul.cuh"

LaunchFn matmul_launchers[] = {
    matmul_launch_v0,
}

struct MatMulRunConfig {
    int M, K, N;
    int block;
    int grid; 
    int warmup; 
    int iters; 
    bool test; 
    int v;
}

int run_matmul(const MatMulRunConfig& cfg){
    int M = cfg->M; 
    int K = cfg->K;
    int N = cfg->N;
    int block = cfg->block;
    int warmup = cfg->warmup;
    int iters = cfg->iters;
    int v = cfg->v;
    bool test = cfg->test;


    int num_variants = sizeof(matmul_launchers) / sizeof(matmul_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }
    
    auto launch = matmul_launchers[v];

    size_t A_bytes = (size_t)M*K * sizeof(floats);
    size_t B_bytes = (size_t)K*N * sizeof(floats);
    size_t C_bytes = (size_t)M*N * sizeof(floats);

    float* h_A, h_B, h_C;
    float* ref_C;

    ck(cudaMallocHost(&h_A, A_bytes));
    ck(cudaMallocHost(&h_B, B_bytes));
    ck(cudaMallocHost(&h_C, C_bytes));
    ck(cudaMallocHost(&ref_C, C_bytes));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10, 10);

    // fill out A, B from 
    for (int i = 0; i < M*K; i++){
        h_A[i] = dist(gen);
    }

    for (int i = 0; i < K*N; i++){
        h_B[i] = dist(gen);
    }

    // USE ROW MAJOR A_ij = A[i * K + j] in array space
    // C_[ij] = C[i * N + j] = A row i cdot B col j
    // get row i from A by looping over k in A[i * K + k]
    // get col j from B by looping over k in B[k * N + j]
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float acc = 0.0f;
            for (int k = 0; k < K; k++){
                acc += A[i * K + k] * B[k * N + j];
            }
            ref_C[i * N + j] = acc;
        }
    }
    float* d_A, d_B, d_C; 
    ck(cudaMalloc(&d_A, A_bytes));
    ck(cudaMalloc(&d_B, B_bytes));
    ck(cudaMalloc(&d_C, C_bytes));
    
    ck(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_B, h_B, A_bytes, cudaMemcpyHostToDevice));

    MatMulCtx ctx{M, K, N, d_A, d_B, d_C, block};


    // test loop


    ck(cudaFree(d_A));
    ck(cudaFree(d_B));
    ck(cudaFree(d_C));

    ck(cudaFreeHost(h_A));
    ck(cudaFreeHost(h_B));
    ck(cudaFreeHost(ref_C));
}