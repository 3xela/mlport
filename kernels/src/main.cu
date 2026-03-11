#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "kernels/axpy.cuh"

static int argi(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

int main(int argc, char** argv) {
    int n = argi(argc, argv, "--n", 1 << 26);
    int block = argi(argc, argv, "--block", 256);
    int warmup = argi(argc, argv, "--warmup", 100);
    int iters = argi(argc, argv, "--iters", 1000);
    int test = argi(argc, argv, "--test", 0);
    int v = argi(argc, argv, "--v", 0);

    size_t bytes = (size_t)n * sizeof(float);

    auto launch = (v==0) ? axpy_launch_v0 : axpy_launch_v1;

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
