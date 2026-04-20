#pragma once

#include "launch.cuh"

struct ReductionCtx{
    const float* x;
    float* block_sums;
    float* final_sum;
    int n;
};

void reduction_launch_v0(void* p);
void reduction_launch_v1(void* p);
void reduction_launch_v2(void* p);
void reduction_launch_v3(void* p) ;
void full_reduction_launch(void* p);
