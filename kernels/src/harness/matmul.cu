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
    matmul_launch_v1,
};

struct MatMulRunConfig{
    int const M;
    int const K;
    int const N;
    int block;
    int grid;
    int warmup;
    int iters;
    bool test;
    int v;
};

int run_matmul(const MatMulRunConfig& cfg){
    const int M = cfg.M; 
    const int K = cfg.K;
    const int N = cfg.N;
    int block = cfg.block;
    int grid = cfg.grid;
    int warmup = cfg.warmup;
    int iters = cfg.iters;
    int v = cfg.v;
    bool test = cfg.test;


    int num_variants = sizeof(matmul_launchers) / sizeof(matmul_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }
    
    auto launch = matmul_launchers[v];

    size_t A_bytes = (size_t)M*K * sizeof(float);
    size_t B_bytes = (size_t)K*N * sizeof(float);
    size_t C_bytes = (size_t)M*N * sizeof(float);

    float* h_A;
    float* h_B;
    float* h_C;
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
                acc += h_A[i * K + k] * h_B[k * N + j];
            }
            ref_C[i * N + j] = acc;
        }
    }
    float* d_A;
    float* d_B;
    float* d_C; 
    ck(cudaMalloc(&d_A, A_bytes));
    ck(cudaMalloc(&d_B, B_bytes));
    ck(cudaMalloc(&d_C, C_bytes));
    
    ck(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));

    MatMulCtx ctx{M, K, N, d_A, d_B, d_C, block};


    // test loop

    if (test){
        launch(&ctx);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));

        float max_rel = 0.0f;
        for (int i = 0; i < M*N; i++) {
            float err = fabsf(h_C[i] - ref_C[i]);
            float rel = err / (fabsf(ref_C[i]) + 1e-6f); 
            max_rel = fmaxf(max_rel, rel);
        }
        std::printf("kernel=matmul:%i error:%.2f\n", v, max_rel);
    }
    else{
        float ms = bench_kernel_ms(launch, &ctx, warmup, iters);
        ck(cudaGetLastError());
        double flops = 2.0 * (double)M * N * K; 
        double gflops = flops / (ms * 1e-3) / 1e9;
        double tflops = gflops / 1000.0;
        double pct = tflops / 35.58 * 100.0;
        std::printf("kernel=matmul:%i M=%d N=%d K=%d gflops=%.1f tflops=%.2f pct_peak=%.1f%%\n",
            v, M, N, K, gflops, tflops, pct);
    }


    ck(cudaFree(d_A));
    ck(cudaFree(d_B));
    ck(cudaFree(d_C));

    ck(cudaFreeHost(h_A));
    ck(cudaFreeHost(h_B));
    ck(cudaFreeHost(h_C));

    ck(cudaFreeHost(ref_C));
    return 0;
}