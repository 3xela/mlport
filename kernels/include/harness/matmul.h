#pragma once

struct MatMulRunConfig{
    int N;
    int M;
    int K;
    int block;
    int grid;
    int iters;
    bool test;
    int v;
};

int run_matmul(const MatMulRunConfig& cfg);