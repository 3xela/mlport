#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>

#include "ck.cuh"
#include "bench.cuh"
#include "kernels/saxpy.cuh"

static int argi(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

int main(int argc, char** argv) {
    int n = argi(argc, argv, "--n", 1 << 26);
    int block = argi(argc, argv, "--block", 256);
    int warmup = argi(argc, argv, "--warmup", 100);
    int iters = argi(argc, argv, "--iters", 1000);

    size_t bytes = (size_t)n * sizeof(float);

    float *h_x, *h_y;
    ck(cudaMallocHost(&h_x, bytes));
    ck(cudaMallocHost(&h_y, bytes));

    for (int i = 0; i < n; i++) h_x[i] = (float)i * 0.001f;

    float *d_x, *d_y;
    ck(cudaMalloc(&d_x, bytes));
    ck(cudaMalloc(&d_y, bytes));
    ck(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    SaxpyCtx ctx{d_x, d_y, 2.0f, 1.0f, n, block};

    float ms = bench_kernel_ms(&saxpy_launch, &ctx, warmup, iters);

    double bytes_moved = (double)n * 8.0;
    double bw = gbps_mem(bytes_moved, (double)ms);

    std::printf("kernel=saxpy n=%d block=%d time_ms=%.6f bw_gbps=%.2f\n", n, block, ms, bw);

    ck(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 3; i++) {
        float ref = 2.0f * h_x[i] + 1.0f;
        std::printf("y[%d]=%.6f ref=%.6f\n", i, h_y[i], ref);
    }

    ck(cudaFree(d_x));
    ck(cudaFree(d_y));
    ck(cudaFreeHost(h_x));
    ck(cudaFreeHost(h_y));
}
