#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
