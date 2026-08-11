#include "kernels/matmul.cuh"

#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>
#include <cmath>
#include <cuda_fp16.h>   
#include <algorithm>

using namespace nvcuda;

static __global__ void matmul_kernel_v0(const int M, const int K, const int N, const float* A, const float* B, float* C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N){
        float acc = 0.0f;
        for (int k = 0; k < K; k++){
            acc += A[ i * K + k ] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }

}

static __global__ void matmul_kernel_v1(const int M, const int K, const int N, const float* A, const float* B, float* C){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    constexpr int BM = 32, BN = 32, BK = 32;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];


    if (i < M && j < N){
        float acc = 0.0f;
        for (int k_tile = 0; k_tile < K; k_tile += BK){
            As[ty * BK + tx] = A[i * K + k_tile + tx]; 
            Bs[ty * BN + tx] = B[(k_tile + ty) * N + j];
            __syncthreads();
            for (int k = 0; k < BK; k++){
                acc += As[ty * BK + k] * Bs[k * BN + tx ];
            }
            __syncthreads();
        }
        C[i * N + j] = acc;
    }

}

static __global__ void matmul_kernel_v2(const int M, const int K, const int N, const float* A, const float* B, float* C){
    constexpr int BM = 128, BN = 128, BK = 32; // TODO make sure M, N, K are multiples of 128, 128, 32 resp. 
    
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int thread_row = ty * 8;
    int thread_col = tx * 8;

    int tid = ty * blockDim.x + tx;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];
    float acc[64] = {0.0f};

    for (int k_tile = 0; k_tile < K; k_tile += BK){
        for (int load_iter = 0; load_iter < 16; load_iter++){
            int flat = load_iter * 256 + tid;

            int A_tile_row = flat / BK; 
            int A_tile_col = flat % BK;

            int B_tile_row = flat / BN;
            int B_tile_col = flat % BN;

            int A_global_row = block_row + A_tile_row;
            int A_global_col = k_tile + A_tile_col;

            int B_global_row = k_tile + B_tile_row;
            int B_global_col = block_col + B_tile_col;

            As[A_tile_col * BM + A_tile_row] = A[A_global_row * K + A_global_col];
            Bs[flat] = B[B_global_row * N + B_global_col];
        }
        __syncthreads();
        for (int k = 0; k < BK; k++){
            float a_frag[8];
            float b_frag[8];
            for(int l = 0; l < 8; l++){
                a_frag[l] = As[k * BM + (thread_row + l)]; 
                b_frag[l] = Bs[k * BN + (thread_col + l)];
            }
            for (int i = 0; i < 8; i++){ //row
                for (int j = 0; j < 8; j++){ //col
                    acc[i * 8 + j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0 ; i < 8 ; i++){
        for (int j = 0; j < 8; j++){
            C[(block_row + thread_row + i) * N + (block_col + thread_col + j)]= acc[i * 8 + j];
        }
    }
}

static __global__ void wmma_matmul_kernel_v0(const int M, const int K, const int N, const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int ld_A = K;
    const int ld_B = N;
    const int ld_C = N;
    
    wmma::load_matrix_sync(a_frag, A, ld_A);
    wmma::load_matrix_sync(b_frag, B, ld_B);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    store_matrix_sync(C, c_frag, ld_C, wmma::mem_row_major);

}

static __global__ void wmma_matmul_kernel_v1(const int M, const int K, const int N, const half* A, const half* B, float* C){
    int warp_row = blockIdx.y * 16;
    int warp_col = blockIdx.x * 16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int ld_A = K;
    const int ld_B = N;
    const int ld_C = N;
    
    for ( int i = 0; i< K/16; i++){
        wmma::load_matrix_sync(a_frag, A + warp_row  * K + i * 16, ld_A);
        wmma::load_matrix_sync(b_frag, B + i * 16 * N + warp_col, ld_B);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, ld_C, wmma::mem_row_major);

}

static __global__ void wmma_matmul_kernel_v2(const int M, const int K, const int N, const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    constexpr int BN_stride = 64 + 8;
    constexpr int BM = 32, BN = 64, BK = 32;

    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int wr = warp_id / 4;
    int wc = warp_id % 4;
    int warp_row = blockIdx.y * BM + wr * 16;
    int warp_col = blockIdx.x * BN + wc * 16;

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN_stride];
    
    const int ld_A = K;   // global strides (cooperative load + store)
    const int ld_B = N;
    const int ld_C = N;
    
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < K; k_tile += BK){
        for (int i = 0; i < 4; i++){
            int flat = tid + 256 * i;
            int tile_row = flat / BK;
            int tile_col = flat % BK;
            As[flat] = A[(blockIdx.y * BM + tile_row) * K + (k_tile + tile_col)];
        }
        for (int i = 0; i < 8; i++){
            int flat = tid + 256 * i;
            int tile_row = flat / BN;
            int tile_col = flat % BN;
            Bs[tile_row * BN_stride + tile_col] = B[(k_tile + tile_row) * N + (blockIdx.x * BN + tile_col)];
        }
        __syncthreads();
        for ( int ks = 0; ks < BK/16; ks++){
            wmma::load_matrix_sync(a_frag, As + (wr * 16) * BK + ks * 16, BK);
            wmma::load_matrix_sync(b_frag, Bs + (ks * 16) * BN_stride + (wc * 16), BN_stride);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col, c_frag, ld_C, wmma::mem_row_major);
}

static __global__ void wmma_matmul_kernel_v3(const int M, const int K, const int N, const half* A, const half* B, float* C){
    constexpr int FRAG_N = 4;
    constexpr int FRAG_M = 2;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[FRAG_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[FRAG_N];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[FRAG_M][FRAG_N];

    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int BM = 128;


    
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    constexpr int NUM_THREADS = 256;
    constexpr int a_load_count = BM * BK / NUM_THREADS;
    constexpr int b_load_count = BK * BN / NUM_THREADS;

    const int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int warp_id = local_tid / 32;

    // warps are 4 rows 2 cols. so warp_id = wr * 2 + wc
    const int wr = warp_id / 2;
    const int wc = warp_id % 2;
    // frags are 16 by 16, so frag (a,b) takes place of (16a, 16b) within warp tile. 
    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    #pragma unroll
    for (int m = 0; m < FRAG_M; m++) {
        #pragma unroll
        for (int n = 0; n < FRAG_N; n++) {
            wmma::fill_fragment(c_frag[m][n], 0.0f);
        }
    }
    for (int k_tile = 0; k_tile < K; k_tile += BK){
        // game plan is to identify where in smem a given thread copies to, then in the global access we just offset by block index and k_tile
        for (int i = 0; i < a_load_count; i++){
            // should be this i believe? we give each row warp the 32 rows that belong to it. each thread within the warp hits its own lane + wr *32 since each warp will hit a 32 row chunk, and copies 16 elements in that row.
            // the warp col just handles along the BK dimension, since its 32 wide we can assign the second col of warps to hit hte second col of BK. main idea is just flatten along row axis
            As[(wr * 32 + local_tid % 32) * BK + (16 * wc + i)] = A[(block_row + wr * 32 + local_tid % 32) * K  + (k_tile + 16 * wc + i)];
        }
        for (int i = 0; i < b_load_count; i++){
            // this one is much more fucked up, dont reindex warps though that is way more cancerous
            // Bs is 32 by 128. maybe each lane does a row?
            // one warp should copy to 8 rows of Bs. 
            // each warp ids go 0 thru 7
            // warp row 0 : rows 0 to 7
            // warp row 1: 7 to 15 etc
            Bs[(local_tid / 8 ) * BN + (16 * (local_tid % 8) + i) ] = B[(local_tid / 8 + k_tile ) * N+ (16 * (local_tid % 8) + i + block_col)];
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BK / 16; ks ++){
            #pragma unroll
            for (int m = 0; m < FRAG_M; m++){
                wmma::load_matrix_sync(a_frag[m], As + (wr * 32 + m * 16) * BK + (ks * 16), BK);
            }
            #pragma unroll
            for (int n = 0; n < FRAG_N; n++){
                wmma::load_matrix_sync(b_frag[n], Bs + (ks * 16) * BN + (wc * 64 + n * 16), BN);
            }
            #pragma unroll
            for (int m= 0; m < FRAG_M; m++){
                #pragma unroll
                for (int n = 0; n < FRAG_N; n++){
                    wmma::mma_sync(c_frag[m][n], a_frag[m], b_frag[n], c_frag[m][n]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < FRAG_M; m++){
        #pragma unroll
        for (int n = 0; n < FRAG_N; n++){
            wmma::store_matrix_sync(C + (block_row + wr * 32 + m * 16) * N + (block_col + wc * 64 + n * 16), c_frag[m][n], N, wmma::mem_row_major);
        }
    }

}


// next level, we do BK = 32, BN = 128, BM = 128. 
__device__ __forceinline__ int swiz_a(int row, int col, int width){
    return row * width + (col ^ ((row >> 1) & 3));
}

__device__ __forceinline__ int swiz_b(int row, int col, int width){
    return row * width + (col ^ (row & 7));
}

static __global__ void wmma_matmul_kernel_v4(const int M, const int K, const int N, const half* A, const half* B, float* C){
    constexpr int BM = 128;
    constexpr int BK = 32;
    constexpr int BN = 128;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    int ty = threadIdx.y; // 0 to 7
    int tx = threadIdx.x; // 0 to 31

    int flat_tid = ty * blockDim.x + tx;
    int warp = flat_tid / 32;

    int threads = blockDim.x * blockDim.y;

    int groups_per_row_A = BK/8;
    int groups_per_row_B = BN/8;

    int row_per_pass_A =  threads / groups_per_row_A;
    int row_per_pass_B =  threads / groups_per_row_B;

    int iters_A = BM / row_per_pass_A;
    int iters_B = BK / row_per_pass_B;

    // As is 128 by 32
    // Bs is 32 by 128
    
    // 4 by 2 warps. warp = wy * 2 + wx

    int warp_row = warp / 2;
    int warp_col = warp % 2;

    // 256 threads per block. a thread copies 16 elts of As and 16 of Bs

    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    const uint4* A4  = reinterpret_cast<const uint4*>(A);
    uint4* As4 = reinterpret_cast<uint4*>(As);

    const uint4* B4  = reinterpret_cast<const uint4*>(B);
    uint4* Bs4 = reinterpret_cast<uint4*>(Bs);

    for (int k_tile = 0; k_tile < K; k_tile += BK){

        for (int i = 0; i < iters_A; i++){
            int row = (flat_tid / 4) +  i * row_per_pass_A;
            int col = flat_tid % 4;
            As4[swiz_a(row, col, groups_per_row_A) ] = A4[(row + block_row ) * (K/8) + (col + (k_tile/8))];
        }
        for (int i = 0; i < iters_B; i++){
            int row = (flat_tid / 16) + i * row_per_pass_B;
            int col = flat_tid % 16;
            Bs4[swiz_b(row, col, groups_per_row_B)] = B4[(row + k_tile) * (N/8) + (col + block_col / 8)];
        }

        __syncthreads();
        #ifdef TILE_CHECK
        if (k_tile == 0 && blockIdx.x + blockIdx.y == 0) {
            // As: logical r in [0,BM), c in [0,BK), half-space
            for (int s = flat_tid; s < BM * BK; s += threads) {
                int r = s / BK, c = s % BK;
                half want = A[(size_t)(block_row + r) * K + (k_tile + c)];
                half got = As[ swiz_a(r, c / 8, groups_per_row_A) * 8 + (c % 8) ];
                if (__half_as_ushort(got) != __half_as_ushort(want))
                    printf("As blk(%d,%d) r=%d c=%d got=%f want=%f\n",
                           blockIdx.x, blockIdx.y, r, c,
                           __half2float(got), __half2float(want));
            }
            // Bs: logical r in [0,BK), c in [0,BN), half-space
            for (int s = flat_tid; s < BK * BN; s += threads) {
                int r = s / BN, c = s % BN;
                half want = B[(size_t)(k_tile + r) * N + (block_col + c)];
                half got  = Bs[swiz_b(r, c/8, groups_per_row_B) * 8 + (c % 8)];
                if (__half_as_ushort(got) != __half_as_ushort(want))
                    printf("Bs blk(%d,%d) r=%d c=%d got=%f want=%f\n",
                           blockIdx.x, blockIdx.y, r, c,
                           __half2float(got), __half2float(want));
            }
        }
                // thread 0 only, one slab at a time — it's 512 iterations, trivial
        if (flat_tid == 0 && k_tile == 0 && blockIdx.x + blockIdx.y == 0) {
            for (int r = 0; r < 128; r++)
                for (int g = 0; g < 4; g++)
                    As4_test[swz_a(r, g)] = r * 4 + g;      // write pattern through swizzle
            for (int r = 0; r < 128; r++)
                for (int g = 0; g < 4; g++)
                    if (As4_test[swz_a(r, g)] != r * 4 + g) // read back through same fn
                        printf("swz_a broken at r=%d g=%d\n", r, g);
        }
                __syncthreads();   // nobody starts the next k_tile's loads mid-check
        #endif
    
    }
}

void matmul_launch_v0(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(16,16);
    dim3 grid((c->N + 15) / 16, (c->M + 15) / 16);
    matmul_kernel_v0<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v1(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(32,32);
    dim3 grid((c->N + 31) / 32, (c->M + 31) / 32);
    matmul_kernel_v1<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v2(void* p){
    auto* c = (MatMulCtx*)p;
    dim3 block(16,16);
    dim3 grid((c->N + 127) / 128, (c->M + 127) / 128);
    matmul_kernel_v2<<<grid, block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v3(void* p){
    auto* c = (WMMAMulCtx*)p;
    if (c->M != 16 || c->N != 16 || c->K != 16){ 
        std::fprintf(stderr, "M,N,K must be equal to 16\n");
        std::exit(1);
    }
    wmma_matmul_kernel_v0<<<1,32>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v4(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/16, c->M/16);
    wmma_matmul_kernel_v1<<<grid,32>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v5(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/64, c->M/32);
    wmma_matmul_kernel_v2<<<grid,256>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v6(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/128, c->M/128);
    dim3 block(32,8);
    wmma_matmul_kernel_v3<<<grid,256>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}

void matmul_launch_v7(void* p){
    auto* c = (WMMAMulCtx*)p;
    dim3 grid(c->N/128, c->M/128);
    dim3 block(32,8);
    wmma_matmul_kernel_v4<<<grid,block>>>(c->M, c->K, c->N, c->A, c->B, c->C);
}