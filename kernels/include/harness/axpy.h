#pragma once

struct AxpyRunConfig {
    int n, block, warmup, iters, test, v;
};

void run_axpy(const AxpyRunConfig& cfg);