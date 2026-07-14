#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>          // __float2half / __half2float, half

#include "ck.cuh"
#include "bench.cuh"
#include "launch.cuh"
#include "kernels/matmul.cuh"

LaunchFn matmul_launchers[] = {
    matmul_launch_v0,
    matmul_launch_v1,
    matmul_launch_v2,
    matmul_launch_v3,          // WMMA
    matmul_launch_v4,
    matmul_launch_v5,
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
    int block  = cfg.block;
    int warmup = cfg.warmup;
    int iters  = cfg.iters;
    int v      = cfg.v;
    bool test  = cfg.test;

    int num_variants = sizeof(matmul_launchers) / sizeof(matmul_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }

    const bool is_wmma = (v > 2);
    if (is_wmma) printf("using tensor cores\n");

    auto launch = matmul_launchers[v];

    // ---- host float source (always) ----
    size_t A_bytes = (size_t)M*K * sizeof(float);
    size_t B_bytes = (size_t)K*N * sizeof(float);
    size_t C_bytes = (size_t)M*N * sizeof(float);

    float *h_A, *h_B, *h_C, *ref_C = nullptr;
    if (test){
        ck(cudaMallocHost(&ref_C, C_bytes));
    }
    ck(cudaMallocHost(&h_A, A_bytes));
    ck(cudaMallocHost(&h_B, B_bytes));
    ck(cudaMallocHost(&h_C, C_bytes));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10, 10);
    for (int i = 0; i < M*K; i++) h_A[i] = dist(gen);
    for (int i = 0; i < K*N; i++) h_B[i] = dist(gen);

    // ---- half staging (WMMA path only) ----
    size_t A_half_bytes = (size_t)M*K * sizeof(half);
    size_t B_half_bytes = (size_t)K*N * sizeof(half);
    half *h_A_half = nullptr, *h_B_half = nullptr;
    if (is_wmma) {
        ck(cudaMallocHost(&h_A_half, A_half_bytes));
        ck(cudaMallocHost(&h_B_half, B_half_bytes));
        for (int i = 0; i < M*K; i++) h_A_half[i] = __float2half(h_A[i]);
        for (int i = 0; i < K*N; i++) h_B_half[i] = __float2half(h_B[i]);
    }

    // ---- reference: WMMA sees fp16-rounded inputs, fp32 accumulate ----
    if (test){
        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
                float acc = 0.0f;
                for (int k = 0; k < K; k++){
                    if (is_wmma) {
                        acc += __half2float(h_A_half[i*K + k]) * __half2float(h_B_half[k*N + j]);
                    } else {
                        acc += h_A[i*K + k] * h_B[k*N + j];
                    }
                }
                ref_C[i*N + j] = acc;
            }
        }
    }

    // ---- device buffers (C always float) ----
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    half  *d_A_half = nullptr, *d_B_half = nullptr;
    ck(cudaMalloc(&d_C, C_bytes));

    if (is_wmma) {
        ck(cudaMalloc(&d_A_half, A_half_bytes));
        ck(cudaMalloc(&d_B_half, B_half_bytes));
        ck(cudaMemcpy(d_A_half, h_A_half, A_half_bytes, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_B_half, h_B_half, B_half_bytes, cudaMemcpyHostToDevice));
    } else {
        ck(cudaMalloc(&d_A, A_bytes));
        ck(cudaMalloc(&d_B, B_bytes));
        ck(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));
    }

    // ---- pick the ctx the active launcher expects ----
    MatMulCtx      ctx{M, K, N, d_A, d_B, d_C, block};
    WMMAMulCtx  wmma_ctx{M, K, N, d_A_half, d_B_half, d_C, block};
    void* launch_ctx = is_wmma ? (void*)&wmma_ctx : (void*)&ctx;

    if (test){
        launch(launch_ctx);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(h_C, d_C, C_bytes, cudaMemcpyDeviceToHost));

        // pass 1: find the magnitude scale of the reference
        float ref_max = 0.0f;
        for (int i = 0; i < M*N; i++) {
            ref_max = fmaxf(ref_max, fabsf(ref_C[i]));
        }
        float floor = ref_max * 1e-3f;

        // pass 2: floored relative error
        float max_rel = 0.0f;
        for (int i = 0; i < M*N; i++) {
            float err   = fabsf(h_C[i] - ref_C[i]);
            float denom = fmaxf(fabsf(ref_C[i]), floor);   // never divide by near-zero
            float rel   = err / denom;
            max_rel = fmaxf(max_rel, rel);
        }
        std::printf("kernel=matmul:%i error:%.5f ref_max:%.1f floor:%.3f\n",
                    v, max_rel, ref_max, floor);
    }
    else{
        float ms = bench_kernel_ms(launch, launch_ctx, warmup, iters);
        ck(cudaGetLastError());
        double flops  = 2.0 * (double)M * N * K;
        double gflops = flops / (ms * 1e-3) / 1e9;
        double tflops = gflops / 1000.0;
        // NOTE: fp16 tensor peak on 3090 is NOT 35.58 — verify against datasheet before trusting this
        double peak_tflops = is_wmma ? 71.0 : 35.58;
        double pct = tflops / peak_tflops * 100.0;
        std::printf("kernel=matmul:%i M=%d N=%d K=%d gflops=%.1f tflops=%.2f pct_peak=%.1f%%\n",
            v, M, N, K, gflops, tflops, pct);
    }

    // ---- cleanup (guarded; only the active path allocated its A/B) ----
    if (d_A) ck(cudaFree(d_A));
    if (d_B) ck(cudaFree(d_B));
    if (d_A_half) ck(cudaFree(d_A_half));
    if (d_B_half) ck(cudaFree(d_B_half));
    ck(cudaFree(d_C));

    ck(cudaFreeHost(h_A));
    ck(cudaFreeHost(h_B));
    ck(cudaFreeHost(h_C));
    if (test){
        ck(cudaFreeHost(ref_C));
    }
    if (h_A_half) ck(cudaFreeHost(h_A_half));
    if (h_B_half) ck(cudaFreeHost(h_B_half));

    return 0;
}