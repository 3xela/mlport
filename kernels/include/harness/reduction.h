#pragma once

struct ReductionRunConfig {
    int n;
    int block;
    int grid;
    int warmup;
    int iters;
    bool test;
    int v;
};

int run_reduction(const ReductionRunConfig& cfg);