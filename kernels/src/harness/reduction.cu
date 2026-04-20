#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "launch.cuh"

#include "kernels/reduction.cuh"


LaunchFn reduction_launchers[] = {
    reduction_launch_v0,
    reduction_launch_v1,
    reduction_launch_v2,
    reduction_launch_v3,    
    full_reduction_launch,
};

struct ReductionRunConfig {
    int n;
    int block;
    int grid;
    int warmup;
    int iters;
    bool test;
    int v;
};

int run_reduction(const ReductionRunConfig& cfg) {

    int n = cfg.n;
    int block = cfg.block;
    int grid = cfg.grid;
    int warmup = cfg.warmup;
    int iters = cfg.iters;
    int v = cfg.v;
    bool test = cfg.test;

    int num_variants = sizeof(reduction_launchers) / sizeof(reduction_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }

    auto launch = reduction_launchers[v];

    size_t bytes = (size_t)n * sizeof(float);
    float* h_final_sum;
    float* h_block_sums;
    float* h_x;
    double h_ref= 0.0f;

    ck(cudaMallocHost(&h_x, bytes));
    ck(cudaMallocHost(&h_final_sum, sizeof(float)));
    ck(cudaMallocHost(&h_block_sums, grid * sizeof(float)));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10, 10);

    for (int i = 0; i < n; i++){
        h_x[i] = 1.0f;
    }

    for (int i = 0 ; i < n; i++){
        h_ref += h_x[i];
    }

    float *d_x, *d_block_sums, *d_final_sum;
    ck(cudaMalloc(&d_x, bytes));
    ck(cudaMalloc(&d_final_sum, sizeof(float)));
    ck(cudaMalloc(&d_block_sums, grid * sizeof(float)));
    ck(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    ReductionCtx ctx{d_x, d_block_sums, d_final_sum, n};

    if (test){
        launch(&ctx);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(h_block_sums, d_block_sums, grid * sizeof(float), cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(h_final_sum, d_final_sum, sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_sum = 0.0f;
        float err = 0;

        if (v != 3){
            for (int i = 0; i < grid; i++){
                gpu_sum += h_block_sums[i];
            }
            std::printf("gpu sum:%.2f ref host sum: %.2f\n" ,gpu_sum, h_ref);
            err = std::fabs(h_ref - gpu_sum);
        }
        else{
            err = std::fabs(h_ref - h_final_sum[0]);
        }
        std::printf("kernel=reduction:%i error:%.2f\n", v, err );
    }
    else {
        float ms = bench_kernel_ms(launch, &ctx, warmup, iters);
        ck(cudaGetLastError());
        double bytes_moved = n *sizeof(float);
        double bw = gbps_mem(bytes_moved, (double)ms);

        std::printf("kernel=reduction:%i n=%d block=%d time_ms=%.6f bw_gbps=%.2f version=%d\n", v, n, block, ms, bw, v);

    }

    ck(cudaFree(d_x));
    ck(cudaFree(d_block_sums));
    ck(cudaFree(d_final_sum));
    ck(cudaFreeHost(h_x));
    ck(cudaFreeHost(h_block_sums));
    ck(cudaFreeHost(h_final_sum));
    return 0;
}