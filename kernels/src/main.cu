#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "test_axpy.cpp"
#include "test_reduction.cpp"
#include "ck.cuh"
#include "bench.cuh"

static int argi(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

static std::string args(int argc, char** argv, const char* key, const std::string& def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

int main(int argc, char** argv) {

    int n = argi(argc, argv, "--n", 1 << 26);
    int block = argi(argc, argv, "--block", 256);
    int grid = argi(argc, argv, "--grid", 152);
    int warmup = argi(argc, argv, "--warmup", 100);
    int iters = argi(argc, argv, "--iters", 1000);
    int test = argi(argc, argv, "--test", 0);
    int v = argi(argc, argv, "--v", 0);
    std::string kernel = args(argc, argv, "--kernel", "axpy");

    int num_variants = sizeof(axpy_launchers) / sizeof(axpy_launchers[0]);
    if (v < 0 || v >= num_variants) {
        printf("invalid kernel version\n");
        exit(1);
    }

    if (kernel == "axpy"){
        AxpyRunConfig cfg(n, block, warmup, iters, test, v);
        run_axpy(cfg);
    }
    else if (kernel == "reduction"){
        ReductionRunConfig cfg(n, block, grid, warmup, iters, test, v);
        run_reduction(cfg);
    }

    return 0;
}
