#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "kernels/reduction.cuh"

LaunchFn reduction_launchers[] = {
    reduction_launch_v0,
};

struct ReductionRunConfig {
    int n;
    int block;
    int grid;
    int warmup;
    int iters;
    bool test;
    int v;

    ReductionRunConfig(int n_, int block_, int grid_, int warmup_, int iters_, bool test_, int v_)
        : n(n_), block(block_), grid(grid_), warmup(warmup_), iters(iters_), test(test_), v(v_) {}
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

    float* h_block_sums;
    float* h_x;
    float h_ref= 0.0f;

    ck(cudaMallocHost(&h_x, bytes));
    ck(cudaMallocHost(&h_block_sums, grid * sizeof(float)));


    for (int i = 0; i < n; i++){
        h_x[i] = i * 0.001f;
    }

    for (int i = 0 ; i < n; i++){
        h_ref += h_x[i];
    }

    float *d_x, *d_block_sums;
    ck(cudaMalloc(&d_x, bytes));
    ck(cudaMalloc(&d_block_sums, grid * sizeof(float)));
    ck(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    ReductionCtx ctx{d_x, d_block_sums, n};


    if (test){
        launch(&ctx);
        ck(cudaGetLastError());
        ck(cudaDeviceSynchronize());
        ck(cudaMemcpy(h_block_sums, d_block_sums, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float gpu_sum = 0.0f;

        for (int i = 0; i < grid; i++){
            gpu_sum += h_block_sums[i];
        }

        float err = std::fabs(h_ref - gpu_sum);

        std::printf("kernel=reduction:%i error:%.2f\n", v, err );
    }
    else {
        float ms = bench_kernel_ms(launch, &ctx, warmup, iters);
        ck(cudaGetLastError());
        double bytes_moved = (double)n * 12.0;
        double bw = gbps_mem(bytes_moved, (double)ms);

        std::printf("kernel=reduction:%i n=%d block=%d time_ms=%.6f bw_gbps=%.2f version=%d\n", v, n, block, ms, bw, v);

    }

    ck(cudaFree(d_x));
    ck(cudaFree(d_block_sums));
    ck(cudaFreeHost(h_x));
    ck(cudaFreeHost(h_block_sums));
    return 0;
}