#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

#include "harness/axpy.h"
#include "harness/reduction.h"
#include "harness/matmul.h"
#include "ck.cuh"
#include "bench.cuh"

static int argi(int argc, char** argv, const char* key, int def) {
    for (int i = 1; i + 1 < argc; i++) if (std::string(argv[i]) == key) return std::atoi(argv[i + 1]);
    return def;
}

static std::string args(int argc, char** argv, const char* key, const std::string& def) {\
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

// parse a 0/1 as boolean. throw if not either. 
static bool argb(int argc, char** argv, const char* key, bool def){
    for (int i = 1; i + 1 < argc; i++) {
        if (std::string(argv[i]) == key){
            if (std::string(argv[i + 1]) == "1" || std::string(argv[i + 1]) == "0"){
                return std::string(argv[i + 1]) == "1";
            }
            else {
                throw std::invalid_argument("--test must be 0 or 1");
            }
        }
    }
    return def;
}

int main(int argc, char** argv) {

    int n = argi(argc, argv, "--n", 1 << 26);
    int M_dim = argi(argc, argv, "--M", 1024);
    int K_dim = argi(argc, argv, "--K", 1024);
    int N_dim = argi(argc, argv, "--N", 1024);
    int block = argi(argc, argv, "--block", 256);
    int grid = argi(argc, argv, "--grid", 152);
    int warmup = argi(argc, argv, "--warmup", 100);
    int iters = argi(argc, argv, "--iters", 1000);
    bool test = argb(argc, argv, "--test", false);
    int v = argi(argc, argv, "--v", 0);
    std::string kernel = args(argc, argv, "--kernel", "axpy");

    if (kernel == "axpy"){
        AxpyRunConfig cfg{n, block, warmup, iters, test, v};
        run_axpy(cfg);
    }
    else if (kernel == "reduction"){
        ReductionRunConfig cfg {n, block, grid, warmup, iters, test, v};
        run_reduction(cfg);
    }
    else if (kernel == "matmul"){
        MatMulRunConfig cfg {M_dim, K_dim, N_dim, block, grid, warmup, iters, test, v};
        run_matmul(cfg);
    }
    
    return 0;
}
