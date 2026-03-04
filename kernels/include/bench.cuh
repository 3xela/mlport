#pragma once
#include "ck.cuh"
#include <cuda_runtime.h>

static inline float bench_kernel_ms(void (*launch)(void*), void* ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) launch(ctx);
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start));
    ck(cudaEventCreate(&stop));

    ck(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) launch(ctx);
    ck(cudaEventRecord(stop));
    ck(cudaGetLastError());
    ck(cudaEventSynchronize(stop));

    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, start, stop));

    ck(cudaEventDestroy(start));
    ck(cudaEventDestroy(stop));
    return ms / (float)iters;
}

static inline double gbps_mem(double bytes_moved, double ms_per_iter) {
    double sec = ms_per_iter * 1e-3;
    return (bytes_moved / sec) / 1e9;
}
