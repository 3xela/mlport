#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "kernels/axpy.cuh"

LaunchFn axpy_launchers[] = {
    axpy_launch_v0,
    axpy_launch_v1,
    axpy_launch_v2
};

struct AxpyRunConfig {
    int n;
    int block;
    int warmup;
    int iters;
    bool test;
    int v;

    AxpyRunConfig(int n_, int block_, int warmup_, int iters_, bool test_, int v_)
        : n(n_), block(block_), warmup(warmup_), iters(iters_), test(test_), v(v_) {}
};

int run_axpy(const AxpyRunConfig& cfg) {

    int n = cfg.n;
    int block = cfg.block;
    int warmup = cfg.warmup;
    int iters = cfg.iters;
    int v = cfg.v;
    bool test = cfg.test;

    int num_variants = sizeof(axpy_launchers) / sizeof(axpy_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }

    auto launch = axpy_launchers[v];

    size_t bytes = (size_t)n * sizeof(float);

    float *h_x, *h_y, *h_ref;
        ck(cudaMallocHost(&h_x, bytes));
        ck(cudaMallocHost(&h_y, bytes));
        ck(cudaMallocHost(&h_ref, bytes));

        for (int i = 0; i < n; i++){
            h_x[i] = (float)i * 0.001f;
            h_y[i] = (float)2 * i * 0.001f;
        }

        for (int i = 0 ; i < n; i++){
            h_ref[i] = 2.0* h_x[i] + h_y[i];
        }

        float *d_x, *d_y;
        ck(cudaMalloc(&d_x, bytes));
        ck(cudaMalloc(&d_y, bytes));
        ck(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

        AxpyCtx ctx{d_x, d_y, 2.0f, n, block};


        if (test){
            launch(&ctx);
            ck(cudaGetLastError());
            ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
            float max_err = 0.0f;
            for (int i = 0; i < n; i++){
                float diff = std::fabs(h_ref[i] - h_y[i]);
                if (diff > max_err) max_err = diff;
            }
        
            std::printf("kernel=axpy:%i error:%.2f\n", v, max_err );

            for (int i = 0; i < 3; i++) {
                float ref = h_ref[i];
                std::printf("y[%d]=%.6f ref=%.6f\n", i, h_y[i], ref);
            }
        }
        else {
            float ms = bench_kernel_ms(launch, &ctx, warmup, iters);
            ck(cudaGetLastError());
            ck(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
            double bytes_moved = (double)n * 12.0;
            double bw = gbps_mem(bytes_moved, (double)ms);

            std::printf("kernel=axpy:%i n=%d block=%d time_ms=%.6f bw_gbps=%.2f version=%d\n", v, n, block, ms, bw, v);

        }

        ck(cudaFree(d_x));
        ck(cudaFree(d_y));
        ck(cudaFreeHost(h_x));
        ck(cudaFreeHost(h_y));
        ck(cudaFreeHost(h_ref));
        return 0;
}