#pragma once

struct MatMulRunConfig{
    int const M;
    int const K;
    int const N;
    int block;
    int grid;
    int warmup;
    int iters;
    bool test;
    int v;
};

int run_matmul(const MatMulRunConfig& cfg);