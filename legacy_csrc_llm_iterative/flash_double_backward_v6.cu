#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include <cmath>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                        \
                    cudaGetErrorString(err));                                   \
    } while (0)

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

// WMMA tile dimensions (TF32)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

// V6 block parameters: BC=32 (matching V5 double backward)
#define V6_WARPS_PER_BLOCK 4
#define V6_BLOCK_THREADS (WARP_SIZE * V6_WARPS_PER_BLOCK)  // 128
#define V6_BR 64   // rows per block = 4 warps * 16
#define V6_BC 32   // column tile width = 2 wmma tiles

// Padding
#define V6_SMEM_PAD 0
#define V6_SCORE_PAD 2  // Score tile stride must be even for sm_120 store_matrix_sync

// ===========================================================================
// Device helpers
// ===========================================================================
__device__ __forceinline__ float warp_reduce_sum_v6(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_v6(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

__device__ __forceinline__ void scale_fragment_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag,
    float s
) {
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] *= s;
}

// ===========================================================================
// Wide score matmul: A[16][D] @ B[32][D]^T -> [16][32] (two 16x16 halves)
// ===========================================================================
__device__ __forceinline__ void wmma_score_matmul_wide_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_left,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_right,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    wmma::fill_fragment(acc_left, 0.0f);
    wmma::fill_fragment(acc_right, 0.0f);
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag_left;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag_right;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_frag_left, B_smem + kk, B_stride);
        wmma::load_matrix_sync(b_frag_right, B_smem + WMMA_N * B_stride + kk, B_stride);
        wmma::mma_sync(acc_left, a_frag, b_frag_left, acc_left);
        wmma::mma_sync(acc_right, a_frag, b_frag_right, acc_right);
    }
}

__device__ __forceinline__ void wmma_score_matmul_wide_acc_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_left,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_right,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag_left;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag_right;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_frag_left, B_smem + kk, B_stride);
        wmma::load_matrix_sync(b_frag_right, B_smem + WMMA_N * B_stride + kk, B_stride);
        wmma::mma_sync(acc_left, a_frag, b_frag_left, acc_left);
        wmma::mma_sync(acc_right, a_frag, b_frag_right, acc_right);
    }
}

__device__ __forceinline__ void store_wide_score_v6(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& left,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& right
) {
    wmma::store_matrix_sync(tile, left, score_stride, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + WMMA_N, right, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Tall score matmul: A[32][D] @ B[16][D]^T -> [32][16] (two stacked 16x16)
// ===========================================================================
__device__ __forceinline__ void wmma_score_matmul_tall_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_top,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_bot,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    wmma::fill_fragment(acc_top, 0.0f);
    wmma::fill_fragment(acc_bot, 0.0f);
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag_top;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag_bot;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag_top, A_smem + kk, A_stride);
        wmma::load_matrix_sync(a_frag_bot, A_smem + WMMA_M * A_stride + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc_top, a_frag_top, b_frag, acc_top);
        wmma::mma_sync(acc_bot, a_frag_bot, b_frag, acc_bot);
    }
}

__device__ __forceinline__ void wmma_score_matmul_tall_acc_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_top,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc_bot,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag_top;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag_bot;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag_top, A_smem + kk, A_stride);
        wmma::load_matrix_sync(a_frag_bot, A_smem + WMMA_M * A_stride + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc_top, a_frag_top, b_frag, acc_top);
        wmma::mma_sync(acc_bot, a_frag_bot, b_frag, acc_bot);
    }
}

__device__ __forceinline__ void store_tall_score_v6(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& top,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& bot
) {
    wmma::store_matrix_sync(tile, top, score_stride, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + WMMA_M * score_stride, bot, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Output matmul for BC=32: score[16][32] @ data[32][D] -> out[16][D]
// ===========================================================================
__device__ __forceinline__ void wmma_output_matmul_acc_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    int k_steps = V6_BC / WMMA_K;  // 32/8 = 4

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, score_smem + kk * WMMA_K, score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * WMMA_K * data_stride + cb * WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// ===========================================================================
// Transposed tall output matmul: score[32][16]^T @ data[32][D] -> out[16][D]
// ===========================================================================
__device__ __forceinline__ void wmma_output_matmul_acc_tallT_v6(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    int k_steps = V6_BC / WMMA_K;  // 4

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, score_smem + kk * WMMA_K * score_stride, score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * WMMA_K * data_stride + cb * WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// ===========================================================================
// V6 Flash Forward Kernel (2-pass, WMMA, BC=32, causal mask)
//
// Pass 1: L = logsumexp per row
// Pass 2: O = sum_j P_ij * V_j via WMMA output matmuls
// ===========================================================================
__global__ void v6_flash_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    float* __restrict__ L_out,
    int B, int H, int N, int D, float scale,
    bool causal
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V6_SMEM_PAD;
    const int SCORE_STRIDE = V6_BC + V6_SCORE_PAD;  // 34

    float* Q_block    = smem;
    float* K_tile     = Q_block    + V6_BR * D_PAD;
    float* V_tile     = K_tile     + V6_BC * D_PAD;
    float* score_base = V_tile     + V6_BC * D_PAD;
    float* m_vec      = score_base + V6_WARPS_PER_BLOCK * WMMA_M * SCORE_STRIDE;
    float* l_vec      = m_vec      + V6_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing
    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int bh = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V6_BR;
    int block_row_start = bh * N + n_start;

    float* S_w    = score_base + warp_id * WMMA_M * SCORE_STRIDE;
    float* Q_warp = Q_block    + warp_id * WMMA_M * D_PAD;

    int rows_to_load = min(V6_BR, N - n_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Zero-fill and load Q_block
    for (int idx = threadIdx.x; idx < V6_BR * D_PAD; idx += V6_BLOCK_THREADS)
        Q_block[idx] = 0.0f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        Q_block[r * D_PAD + d] = Q[(block_row_start + r) * D + d];
    }

    for (int idx = threadIdx.x; idx < V6_BR; idx += V6_BLOCK_THREADS) {
        m_vec[idx] = -1e30f;
        l_vec[idx] = 0.0f;
    }
    __syncthreads();

    // ==== Pass 1: Compute L = logsumexp ====
    for (int j_start = 0; j_start < N; j_start += V6_BC) {
        // Causal tile skip: if all cols are after all rows in this block
        if (causal && j_start > n_start + rows_to_load - 1) break;

        int tile_cols = min(V6_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS)
            K_tile[idx] = 0.0f;
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V6_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            K_tile[c * D_PAD + d] = K[(bh * N + j_start + c) * D + d];
        }
        __syncthreads();

        // WMMA wide score: S[16][32] = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide_v6(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v6(S_L, scale);
        scale_fragment_v6(S_R, scale);
        store_wide_score_v6(S_w, SCORE_STRIDE, S_L, S_R);
        __syncwarp();

        // Per-row max and sum update
        for (int r = 0; r < WMMA_M; r++) {
            if (warp_id * WMMA_M + r >= rows_to_load) continue;
            int row_idx = n_start + warp_id * WMMA_M + r;

            float local_max = -1e30f;
            for (int c = lane; c < tile_cols; c += WARP_SIZE) {
                float s = S_w[r * SCORE_STRIDE + c];
                if (causal && row_idx < j_start + c) s = -1e30f;
                local_max = fmaxf(local_max, s);
            }
            local_max = warp_reduce_max_v6(local_max);
            float tile_max = __shfl_sync(FULL_MASK, local_max, 0);

            float m_old = m_vec[warp_id * WMMA_M + r];
            float m_new = fmaxf(m_old, tile_max);
            float correction = __expf(m_old - m_new);

            float p_sum = 0.0f;
            for (int c = lane; c < tile_cols; c += WARP_SIZE) {
                float s = S_w[r * SCORE_STRIDE + c];
                if (causal && row_idx < j_start + c) s = -1e30f;
                p_sum += __expf(s - m_new);
            }
            p_sum = warp_reduce_sum_v6(p_sum);

            if (lane == 0) {
                l_vec[warp_id * WMMA_M + r] = l_vec[warp_id * WMMA_M + r] * correction + p_sum;
                m_vec[warp_id * WMMA_M + r] = m_new;
            }
        }
        __syncwarp();
    }

    // Finalize L = log(l) + m
    __syncthreads();
    for (int idx = threadIdx.x; idx < V6_BR; idx += V6_BLOCK_THREADS) {
        if (idx < rows_to_load) {
            int global_row = block_row_start + idx;
            float L_val = logf(l_vec[idx]) + m_vec[idx];
            L_out[global_row] = L_val;
            l_vec[idx] = L_val;
        }
    }
    __syncthreads();

    // ==== Pass 2: O += P @ V using WMMA ====
    int n_col_blocks = D / WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++)
        wmma::fill_fragment(O_frags[cb], 0.0f);

    for (int j_start = 0; j_start < N; j_start += V6_BC) {
        if (causal && j_start > n_start + rows_to_load - 1) break;

        int tile_cols = min(V6_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V6_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d] = K[src];
            V_tile[c * D_PAD + d] = V[src];
        }
        __syncthreads();

        // Recompute S via WMMA
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide_v6(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v6(S_L, scale);
        scale_fragment_v6(S_R, scale);
        store_wide_score_v6(S_w, SCORE_STRIDE, S_L, S_R);
        __syncwarp();

        // Element-wise: P = exp(S - L), with causal mask
        for (int r = 0; r < WMMA_M; r++) {
            int row_idx = n_start + warp_id * WMMA_M + r;
            for (int c = lane; c < V6_BC; c += WARP_SIZE) {
                int si = r * SCORE_STRIDE + c;
                if (c < tile_cols && warp_id * WMMA_M + r < rows_to_load) {
                    float L_val = l_vec[warp_id * WMMA_M + r];
                    float p_val = __expf(S_w[si] - L_val);
                    if (causal && row_idx < j_start + c) p_val = 0.0f;
                    S_w[si] = p_val;
                } else {
                    S_w[si] = 0.0f;
                }
            }
        }
        __syncwarp();

        // O += P @ V
        wmma_output_matmul_acc_v6(O_frags, S_w, SCORE_STRIDE, V_tile, D_PAD, D);
    }

    // Store O via shared memory staging
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            Q_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            O_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        O_out[(block_row_start + r) * D + d] = Q_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V6 Flash Backward Row Kernel: D_i and dQ (BC=32, causal)
// ===========================================================================
__global__ void v6_flash_bwd_row_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ O,
    const float* __restrict__ L,
    float* __restrict__ D_out,
    float* __restrict__ dQ_out,
    int B, int H, int N, int D, float scale,
    bool causal
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V6_SMEM_PAD;
    const int SCORE_STRIDE = V6_BC + V6_SCORE_PAD;  // 34

    float* Q_block    = smem;
    float* dO_block   = Q_block    + V6_BR * D_PAD;
    float* K_tile     = dO_block   + V6_BR * D_PAD;
    float* V_tile     = K_tile     + V6_BC * D_PAD;
    float* score_base = V_tile     + V6_BC * D_PAD;
    float* L_vec      = score_base + V6_WARPS_PER_BLOCK * 2 * WMMA_M * SCORE_STRIDE;
    float* Di_vec     = L_vec      + V6_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing
    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int bh = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V6_BR;
    int block_row_start = bh * N + n_start;

    float* S_w     = score_base + warp_id * 2 * WMMA_M * SCORE_STRIDE;
    float* dP_w    = S_w + WMMA_M * SCORE_STRIDE;
    float* Q_warp  = Q_block  + warp_id * WMMA_M * D_PAD;
    float* dO_warp = dO_block + warp_id * WMMA_M * D_PAD;

    int rows_to_load = min(V6_BR, N - n_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Zero-fill and load
    for (int idx = threadIdx.x; idx < V6_BR * D_PAD; idx += V6_BLOCK_THREADS) {
        Q_block[idx] = 0.0f;
        dO_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        Q_block[r * D_PAD + d]  = Q[global_row * D + d];
        dO_block[r * D_PAD + d] = dO[global_row * D + d];
    }

    for (int idx = threadIdx.x; idx < V6_BR; idx += V6_BLOCK_THREADS) {
        int global_row = block_row_start + idx;
        L_vec[idx] = (idx < rows_to_load) ? L[global_row] : 0.0f;
        Di_vec[idx] = 0.0f;
    }
    __syncthreads();

    // Phase 0: D_i = sum_d dO[i,d] * O[i,d]
    for (int r = 0; r < WMMA_M; r++) {
        if (warp_id * WMMA_M + r >= rows_to_load) {
            if (lane == 0) Di_vec[warp_id * WMMA_M + r] = 0.0f;
            continue;
        }
        int global_row = block_row_start + warp_id * WMMA_M + r;
        float partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE)
            partial += dO_block[(warp_id * WMMA_M + r) * D_PAD + d] * O[global_row * D + d];
        partial = warp_reduce_sum_v6(partial);
        if (lane == 0) {
            Di_vec[warp_id * WMMA_M + r] = partial;
            D_out[global_row] = partial;
        }
    }
    __syncthreads();

    // dQ accumulators
    int n_col_blocks = D / WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dQ_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++)
        wmma::fill_fragment(dQ_frags[cb], 0.0f);

    // Main loop over j-tiles
    for (int j_start = 0; j_start < N; j_start += V6_BC) {
        if (causal && j_start > n_start + rows_to_load - 1) break;

        int tile_cols = min(V6_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V6_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d] = K[src];
            V_tile[c * D_PAD + d] = V[src];
        }
        __syncthreads();

        // WMMA wide: S = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide_v6(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v6(S_L, scale);
        scale_fragment_v6(S_R, scale);
        store_wide_score_v6(S_w, SCORE_STRIDE, S_L, S_R);

        // WMMA wide: dP = dO_warp @ V_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_L, dP_R;
        wmma_score_matmul_wide_v6(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD, D);
        store_wide_score_v6(dP_w, SCORE_STRIDE, dP_L, dP_R);
        __syncwarp();

        // Element-wise: P, dS with causal mask
        for (int r = 0; r < WMMA_M; r++) {
            int row_idx = n_start + warp_id * WMMA_M + r;
            for (int c = lane; c < V6_BC; c += WARP_SIZE) {
                int si = r * SCORE_STRIDE + c;
                if (c < tile_cols && warp_id * WMMA_M + r < rows_to_load) {
                    float s_val = S_w[si];
                    float l_val = L_vec[warp_id * WMMA_M + r];
                    float p_val = __expf(s_val - l_val);
                    if (causal && row_idx < j_start + c) p_val = 0.0f;
                    float dp_val = dP_w[si];
                    float d_val = Di_vec[warp_id * WMMA_M + r];
                    float ds_val = p_val * (dp_val - d_val);
                    dP_w[si] = scale * ds_val;
                } else {
                    dP_w[si] = 0.0f;
                }
            }
        }
        __syncwarp();

        // dQ += (scale*dS) @ K
        wmma_output_matmul_acc_v6(dQ_frags, dP_w, SCORE_STRIDE, K_tile, D_PAD, D);
    }

    // Store dQ via shared memory staging
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            Q_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            dQ_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        dQ_out[(block_row_start + r) * D + d] = Q_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V6 Flash Backward Col Kernel: dK and dV (BC=32 tall tiles, causal)
// ===========================================================================
__global__ void v6_flash_bwd_col_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vec_in,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out,
    int B, int H, int N, int D, float scale,
    bool causal
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V6_SMEM_PAD;
    const int B_SCORE_STRIDE = WMMA_N + V6_SCORE_PAD;  // 18

    float* K_block    = smem;
    float* V_block    = K_block    + V6_BR * D_PAD;
    float* Q_tile     = V_block    + V6_BR * D_PAD;
    float* dO_tile    = Q_tile     + V6_BC * D_PAD;
    float* score_base = dO_tile    + V6_BC * D_PAD;
    float* tile_L     = score_base + V6_WARPS_PER_BLOCK * 2 * V6_BC * B_SCORE_STRIDE;
    float* tile_D     = tile_L     + V6_BC;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing
    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int bh = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V6_BR;
    int block_col_start = bh * N + n_start;

    // Per-warp tall score tiles (2 per warp: S and dP, each 32×18)
    float* S_w  = score_base + warp_id * 2 * V6_BC * B_SCORE_STRIDE;
    float* dP_w = S_w + V6_BC * B_SCORE_STRIDE;

    float* K_warp = K_block + warp_id * WMMA_M * D_PAD;
    float* V_warp = V_block + warp_id * WMMA_M * D_PAD;

    int cols_to_load = min(V6_BR, N - n_start);
    if (cols_to_load < 0) cols_to_load = 0;

    // Load K_block and V_block
    for (int idx = threadIdx.x; idx < V6_BR * D_PAD; idx += V6_BLOCK_THREADS) {
        K_block[idx] = 0.0f;
        V_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V6_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        K_block[c * D_PAD + d] = K[global_col * D + d];
        V_block[c * D_PAD + d] = V[global_col * D + d];
    }
    __syncthreads();

    // Output accumulators
    int n_col_blocks = D / WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dK_frags[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dV_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(dK_frags[cb], 0.0f);
        wmma::fill_fragment(dV_frags[cb], 0.0f);
    }

    // Main loop over row tiles
    for (int i_start = 0; i_start < N; i_start += V6_BC) {
        int tile_rows = min(V6_BC, N - i_start);

        // Causal tile skip: if all rows in tile are before all cols in block
        if (causal && i_start + tile_rows <= n_start) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            Q_tile[idx] = 0.0f;
            dO_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_rows * D; idx += V6_BLOCK_THREADS) {
            int r = idx / D;
            int d = idx % D;
            int src = (bh * N + i_start + r) * D + d;
            Q_tile[r * D_PAD + d]  = Q[src];
            dO_tile[r * D_PAD + d] = dO[src];
        }
        for (int idx = threadIdx.x; idx < V6_BC; idx += V6_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh * N + i_start + idx;
                tile_L[idx] = L[global_row];
                tile_D[idx] = D_vec_in[global_row];
            } else {
                tile_L[idx] = 0.0f;
                tile_D[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Tall score: S = Q_tile[32] @ K_warp[16]^T -> 32×16
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_top, S_bot;
        wmma_score_matmul_tall_v6(S_top, S_bot, Q_tile, D_PAD, K_warp, D_PAD, D);
        scale_fragment_v6(S_top, scale);
        scale_fragment_v6(S_bot, scale);
        store_tall_score_v6(S_w, B_SCORE_STRIDE, S_top, S_bot);

        // Tall score: dP = dO_tile[32] @ V_warp[16]^T -> 32×16
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_top, dP_bot;
        wmma_score_matmul_tall_v6(dP_top, dP_bot, dO_tile, D_PAD, V_warp, D_PAD, D);
        store_tall_score_v6(dP_w, B_SCORE_STRIDE, dP_top, dP_bot);
        __syncwarp();

        // Element-wise: P, dS with causal mask (32×16)
        for (int idx = lane; idx < V6_BC * WMMA_N; idx += WARP_SIZE) {
            int i_local = idx / WMMA_N;
            int j_local = idx % WMMA_N;
            int si = i_local * B_SCORE_STRIDE + j_local;

            int i_global = i_start + i_local;
            int j_global = n_start + warp_id * WMMA_M + j_local;

            float s_val = S_w[si];
            float l_val = tile_L[i_local];
            float p_val = (i_local < tile_rows) ? __expf(s_val - l_val) : 0.0f;
            if (causal && i_global < j_global) p_val = 0.0f;
            float dp_val = dP_w[si];
            float d_val = tile_D[i_local];
            float ds_val = p_val * (dp_val - d_val);

            S_w[si]  = p_val;           // P
            dP_w[si] = scale * ds_val;  // scale * dS
        }
        __syncwarp();

        // dK += (scale*dS)^T @ Q_tile, dV += P^T @ dO_tile
        wmma_output_matmul_acc_tallT_v6(dK_frags, dP_w, B_SCORE_STRIDE, Q_tile, D_PAD, D);
        wmma_output_matmul_acc_tallT_v6(dV_frags, S_w, B_SCORE_STRIDE, dO_tile, D_PAD, D);
    }

    // Store dK and dV
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            K_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            dK_frags[cb], D_PAD, wmma::mem_row_major);
    }
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            V_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            dV_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V6_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        dK_out[global_col * D + d] = K_block[c * D_PAD + d];
        dV_out[global_col * D + d] = V_block[c * D_PAD + d];
    }
}

// ===========================================================================
// V6 Double Backward Kernel A (row kernel): 2 passes, BC=32, causal
//
// Pass 1: D_i, dot2, A, E -> dot3
// Pass 2: g_Q, g_dO
//
// Per-head block indexing fixes cross-(b,h) boundary bug from V5.
// ===========================================================================
__global__ void v6_kernel_A_row(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ O,
    const float* __restrict__ g_dQ,
    const float* __restrict__ g_dK,
    const float* __restrict__ g_dV,
    const float* __restrict__ L,
    float* __restrict__ D_out,
    float* __restrict__ dot2_out,
    float* __restrict__ dot3_out,
    float* __restrict__ g_Q_out,
    float* __restrict__ g_dO_out,
    int B, int H, int N, int D, float scale,
    bool causal
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V6_SMEM_PAD;
    const int SCORE_STRIDE = V6_BC + V6_SCORE_PAD;  // 34

    float* Q_block    = smem;
    float* dO_block   = Q_block    + V6_BR * D_PAD;
    float* g_dQ_block = dO_block   + V6_BR * D_PAD;
    float* K_tile     = g_dQ_block + V6_BR * D_PAD;
    float* V_tile     = K_tile     + V6_BC * D_PAD;
    float* g_dK_tile  = V_tile     + V6_BC * D_PAD;
    float* g_dV_tile  = g_dK_tile  + V6_BC * D_PAD;
    float* score_base = g_dV_tile  + V6_BC * D_PAD;
    float* L_vec      = score_base + V6_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE;
    float* Di_vec     = L_vec      + V6_BR;
    float* dot2_smem  = Di_vec     + V6_BR;
    float* dot3_smem  = dot2_smem  + V6_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing (fixes cross-(b,h) bug from V5)
    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int bh = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V6_BR;
    int block_row_start = bh * N + n_start;
    int total_rows = B * H * N;

    float* S_w     = score_base + warp_id * 5 * WMMA_M * SCORE_STRIDE;
    float* gdS_w   = S_w     + WMMA_M * SCORE_STRIDE;
    float* dP_w    = gdS_w   + WMMA_M * SCORE_STRIDE;
    float* gPdV_w  = dP_w    + WMMA_M * SCORE_STRIDE;
    float* extra_w = gPdV_w  + WMMA_M * SCORE_STRIDE;

    float* Q_warp    = Q_block    + warp_id * WMMA_M * D_PAD;
    float* dO_warp   = dO_block   + warp_id * WMMA_M * D_PAD;
    float* g_dQ_warp = g_dQ_block + warp_id * WMMA_M * D_PAD;

    int warp_row_start = block_row_start + warp_id * WMMA_M;

    int rows_to_load = min(V6_BR, N - n_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Load row data
    for (int idx = threadIdx.x; idx < V6_BR * D_PAD; idx += V6_BLOCK_THREADS) {
        Q_block[idx] = 0.0f;
        dO_block[idx] = 0.0f;
        g_dQ_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        int src = global_row * D + d;
        Q_block[r * D_PAD + d]    = Q[src];
        dO_block[r * D_PAD + d]   = dO[src];
        g_dQ_block[r * D_PAD + d] = g_dQ[src];
    }

    for (int idx = threadIdx.x; idx < V6_BR; idx += V6_BLOCK_THREADS) {
        int global_row = block_row_start + idx;
        L_vec[idx] = (global_row < total_rows && idx < rows_to_load) ? L[global_row] : 0.0f;
        Di_vec[idx] = 0.0f;
        dot2_smem[idx] = 0.0f;
        dot3_smem[idx] = 0.0f;
    }
    __syncthreads();

    // Phase 0: D_i = sum_d dO[i,d] * O[i,d]
    for (int r = 0; r < WMMA_M; r++) {
        int global_row = warp_row_start + r;
        if (global_row >= total_rows || warp_id * WMMA_M + r >= rows_to_load) {
            if (lane == 0) Di_vec[warp_id * WMMA_M + r] = 0.0f;
            continue;
        }
        float partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE)
            partial += dO_block[(warp_id * WMMA_M + r) * D_PAD + d] * O[global_row * D + d];
        partial = warp_reduce_sum_v6(partial);
        if (lane == 0) {
            Di_vec[warp_id * WMMA_M + r] = partial;
            D_out[global_row] = partial;
        }
    }
    __syncthreads();

    // ====================================================================
    // Pass 1: dot2, A, E -> dot3 using WMMA wide score matmuls
    // ====================================================================
    float dot2_acc[WMMA_M];
    float A_acc[WMMA_M];
    float E_acc[WMMA_M];
    for (int r = 0; r < WMMA_M; r++) {
        dot2_acc[r] = 0.0f;
        A_acc[r] = 0.0f;
        E_acc[r] = 0.0f;
    }

    for (int j_start = 0; j_start < N; j_start += V6_BC) {
        if (causal && j_start > n_start + rows_to_load - 1) break;

        int tile_cols = min(V6_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V6_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // WMMA wide score matmuls
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide_v6(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v6(S_L, scale);
        scale_fragment_v6(S_R, scale);
        store_wide_score_v6(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_L, gdS_R;
        wmma_score_matmul_wide_v6(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_wide_acc_v6(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment_v6(gdS_L, scale);
        scale_fragment_v6(gdS_R, scale);
        store_wide_score_v6(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_L, dP_R;
        wmma_score_matmul_wide_v6(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD, D);
        store_wide_score_v6(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_L, gPdV_R;
        wmma_score_matmul_wide_v6(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        store_wide_score_v6(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);

        __syncwarp();

        // Element-wise reductions: 16 rows × 32 cols, 1 col/lane
        for (int r = 0; r < WMMA_M; r++) {
            float p_dot2 = 0.0f, p_A = 0.0f, p_E = 0.0f;
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;
                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                int row_idx = n_start + warp_id * WMMA_M + r;
                int col_idx = j_start + lane;
                float p_val = __expf(s_val - l_val);
                if (causal && row_idx < col_idx) p_val = 0.0f;
                float gds_val = gdS_w[si];
                float dp_val = dP_w[si];
                float gpdv_val = gPdV_w[si];

                float pg = p_val * gds_val;
                p_dot2 = pg;
                p_A    = pg * dp_val;
                p_E    = p_val * gpdv_val;
            }
            p_dot2 = warp_reduce_sum_v6(p_dot2);
            p_A    = warp_reduce_sum_v6(p_A);
            p_E    = warp_reduce_sum_v6(p_E);
            if (lane == 0) {
                dot2_acc[r] += p_dot2;
                A_acc[r]    += p_A;
                E_acc[r]    += p_E;
            }
        }
        __syncwarp();
    }

    // Finalize dot3 = A - 2*D*dot2 + E
    if (lane == 0) {
        for (int r = 0; r < WMMA_M; r++) {
            int global_row = warp_row_start + r;
            if (global_row >= total_rows || warp_id * WMMA_M + r >= rows_to_load) continue;
            float d_val = Di_vec[warp_id * WMMA_M + r];
            float d2 = dot2_acc[r];
            float d3 = A_acc[r] - 2.0f * d_val * d2 + E_acc[r];
            dot2_out[global_row] = d2;
            dot3_out[global_row] = d3;
            dot2_smem[warp_id * WMMA_M + r] = d2;
            dot3_smem[warp_id * WMMA_M + r] = d3;
        }
    }
    __syncthreads();

    // ====================================================================
    // Pass 2: g_Q and g_dO using WMMA output matmuls
    // ====================================================================
    int n_col_blocks = D / WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_Q_frags[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_dO_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_Q_frags[cb], 0.0f);
        wmma::fill_fragment(g_dO_frags[cb], 0.0f);
    }

    for (int j_start = 0; j_start < N; j_start += V6_BC) {
        if (causal && j_start > n_start + rows_to_load - 1) break;

        int tile_cols = min(V6_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V6_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // Recompute wide scores
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide_v6(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v6(S_L, scale);
        scale_fragment_v6(S_R, scale);
        store_wide_score_v6(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_L, gdS_R;
        wmma_score_matmul_wide_v6(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_wide_acc_v6(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment_v6(gdS_L, scale);
        scale_fragment_v6(gdS_R, scale);
        store_wide_score_v6(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_L, dP_R;
        wmma_score_matmul_wide_v6(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD, D);
        store_wide_score_v6(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_L, gPdV_R;
        wmma_score_matmul_wide_v6(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        store_wide_score_v6(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);

        __syncwarp();

        // Element-wise: compute dS, g_dP, g_S, P with causal mask
        for (int r = 0; r < WMMA_M; r++) {
            int row_idx = n_start + warp_id * WMMA_M + r;
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;

                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                int col_idx = j_start + lane;
                float p_val = __expf(s_val - l_val);
                if (causal && row_idx < col_idx) p_val = 0.0f;
                float gds_val = gdS_w[si];
                float dp_val = dP_w[si];
                float gpdv_val = gPdV_w[si];
                float d_val = Di_vec[warp_id * WMMA_M + r];
                float dot2_val = dot2_smem[warp_id * WMMA_M + r];
                float dot3_val = dot3_smem[warp_id * WMMA_M + r];

                float ds_val = p_val * (dp_val - d_val);
                float g_dp_val = p_val * (gds_val - dot2_val);
                float g_P_soft = gds_val * (dp_val - d_val) - dp_val * dot2_val;
                float g_P_val = g_P_soft + gpdv_val;
                float g_S_val = p_val * (g_P_val - dot3_val);

                dP_w[si]    = scale * ds_val;
                extra_w[si] = scale * g_S_val;
                S_w[si]     = p_val;
                gPdV_w[si]  = g_dp_val;
            } else {
                int si = r * SCORE_STRIDE + lane;
                if (lane < V6_BC) {
                    dP_w[si]    = 0.0f;
                    extra_w[si] = 0.0f;
                    S_w[si]     = 0.0f;
                    gPdV_w[si]  = 0.0f;
                }
            }
        }
        __syncwarp();

        // Output matmuls with BC=32
        wmma_output_matmul_acc_v6(g_Q_frags, dP_w, SCORE_STRIDE, g_dK_tile, D_PAD, D);
        wmma_output_matmul_acc_v6(g_Q_frags, extra_w, SCORE_STRIDE, K_tile, D_PAD, D);
        wmma_output_matmul_acc_v6(g_dO_frags, S_w, SCORE_STRIDE, g_dV_tile, D_PAD, D);
        wmma_output_matmul_acc_v6(g_dO_frags, gPdV_w, SCORE_STRIDE, V_tile, D_PAD, D);
    }

    // Store g_Q and g_dO
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            Q_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            g_Q_frags[cb], D_PAD, wmma::mem_row_major);
    }
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            dO_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            g_dO_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V6_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        g_Q_out[global_row * D + d]  = Q_block[r * D_PAD + d];
        g_dO_out[global_row * D + d] = dO_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V6 Double Backward Kernel B (column kernel): 1 pass, BC=32, causal
//
// Per-head block indexing fixes cross-(b,h) boundary bug from V5.
// ===========================================================================
__global__ void v6_kernel_B_col(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ g_dQ,
    const float* __restrict__ g_dK,
    const float* __restrict__ g_dV,
    const float* __restrict__ L,
    const float* __restrict__ D_vec_in,
    const float* __restrict__ dot2_in,
    const float* __restrict__ dot3_in,
    float* __restrict__ g_K_out,
    float* __restrict__ g_V_out,
    int B, int H, int N, int D, float scale,
    bool causal
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V6_SMEM_PAD;
    const int B_SCORE_STRIDE = WMMA_N + V6_SCORE_PAD;  // 18

    float* K_block    = smem;
    float* V_block    = K_block    + V6_BR * D_PAD;
    float* g_dK_block = V_block    + V6_BR * D_PAD;
    float* g_dV_block = g_dK_block + V6_BR * D_PAD;
    float* Q_tile     = g_dV_block + V6_BR * D_PAD;
    float* dO_tile    = Q_tile     + V6_BC * D_PAD;
    float* g_dQ_tile  = dO_tile    + V6_BC * D_PAD;
    float* score_base = g_dQ_tile  + V6_BC * D_PAD;
    float* tile_L     = score_base + V6_WARPS_PER_BLOCK * 5 * V6_BC * B_SCORE_STRIDE;
    float* tile_D     = tile_L     + V6_BC;
    float* tile_dot2  = tile_D     + V6_BC;
    float* tile_dot3  = tile_dot2  + V6_BC;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing (fixes cross-(b,h) bug from V5)
    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int bh = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V6_BR;
    int block_col_start = bh * N + n_start;

    float* S_w     = score_base + warp_id * 5 * V6_BC * B_SCORE_STRIDE;
    float* gdS_w   = S_w     + V6_BC * B_SCORE_STRIDE;
    float* dP_w    = gdS_w   + V6_BC * B_SCORE_STRIDE;
    float* gPdV_w  = dP_w    + V6_BC * B_SCORE_STRIDE;
    float* extra_w = gPdV_w  + V6_BC * B_SCORE_STRIDE;

    float* K_warp    = K_block    + warp_id * WMMA_M * D_PAD;
    float* V_warp    = V_block    + warp_id * WMMA_M * D_PAD;
    float* g_dK_warp = g_dK_block + warp_id * WMMA_M * D_PAD;
    float* g_dV_warp = g_dV_block + warp_id * WMMA_M * D_PAD;

    int cols_to_load = min(V6_BR, N - n_start);
    if (cols_to_load < 0) cols_to_load = 0;

    // Load column data
    for (int idx = threadIdx.x; idx < V6_BR * D_PAD; idx += V6_BLOCK_THREADS) {
        K_block[idx] = 0.0f;
        V_block[idx] = 0.0f;
        g_dK_block[idx] = 0.0f;
        g_dV_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V6_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        int src = global_col * D + d;
        K_block[c * D_PAD + d]    = K[src];
        V_block[c * D_PAD + d]    = V[src];
        g_dK_block[c * D_PAD + d] = g_dK[src];
        g_dV_block[c * D_PAD + d] = g_dV[src];
    }
    __syncthreads();

    // Output accumulators
    int n_col_blocks = D / WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_K_frags[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_V_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_K_frags[cb], 0.0f);
        wmma::fill_fragment(g_V_frags[cb], 0.0f);
    }

    // Single pass over row tiles (BC=32 rows per tile)
    for (int i_start = 0; i_start < N; i_start += V6_BC) {
        int tile_rows = min(V6_BC, N - i_start);

        // Causal tile skip
        if (causal && i_start + tile_rows <= n_start) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V6_BC * D_PAD; idx += V6_BLOCK_THREADS) {
            Q_tile[idx] = 0.0f;
            dO_tile[idx] = 0.0f;
            g_dQ_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_rows * D; idx += V6_BLOCK_THREADS) {
            int r = idx / D;
            int d = idx % D;
            int src = (bh * N + i_start + r) * D + d;
            Q_tile[r * D_PAD + d]    = Q[src];
            dO_tile[r * D_PAD + d]   = dO[src];
            g_dQ_tile[r * D_PAD + d] = g_dQ[src];
        }
        for (int idx = threadIdx.x; idx < V6_BC; idx += V6_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh * N + i_start + idx;
                tile_L[idx]    = L[global_row];
                tile_D[idx]    = D_vec_in[global_row];
                tile_dot2[idx] = dot2_in[global_row];
                tile_dot3[idx] = dot3_in[global_row];
            } else {
                tile_L[idx] = 0.0f;
                tile_D[idx] = 0.0f;
                tile_dot2[idx] = 0.0f;
                tile_dot3[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Tall score matmuls: 32×16 each
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_top, S_bot;
        wmma_score_matmul_tall_v6(S_top, S_bot, Q_tile, D_PAD, K_warp, D_PAD, D);
        scale_fragment_v6(S_top, scale);
        scale_fragment_v6(S_bot, scale);
        store_tall_score_v6(S_w, B_SCORE_STRIDE, S_top, S_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_top, gdS_bot;
        wmma_score_matmul_tall_v6(gdS_top, gdS_bot, g_dQ_tile, D_PAD, K_warp, D_PAD, D);
        wmma_score_matmul_tall_acc_v6(gdS_top, gdS_bot, Q_tile, D_PAD, g_dK_warp, D_PAD, D);
        scale_fragment_v6(gdS_top, scale);
        scale_fragment_v6(gdS_bot, scale);
        store_tall_score_v6(gdS_w, B_SCORE_STRIDE, gdS_top, gdS_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_top, dP_bot;
        wmma_score_matmul_tall_v6(dP_top, dP_bot, dO_tile, D_PAD, V_warp, D_PAD, D);
        store_tall_score_v6(dP_w, B_SCORE_STRIDE, dP_top, dP_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_top, gPdV_bot;
        wmma_score_matmul_tall_v6(gPdV_top, gPdV_bot, dO_tile, D_PAD, g_dV_warp, D_PAD, D);
        store_tall_score_v6(gPdV_w, B_SCORE_STRIDE, gPdV_top, gPdV_bot);

        __syncwarp();

        // Element-wise: 32 rows × 16 cols with causal mask
        for (int idx = lane; idx < V6_BC * WMMA_N; idx += WARP_SIZE) {
            int i_local = idx / WMMA_N;
            int j_local = idx % WMMA_N;
            int si = i_local * B_SCORE_STRIDE + j_local;

            int i_global = i_start + i_local;
            int j_global = n_start + warp_id * WMMA_M + j_local;

            float s_val = S_w[si];
            float l_val = tile_L[i_local];
            float p_val = (i_local < tile_rows) ? __expf(s_val - l_val) : 0.0f;
            if (causal && i_global < j_global) p_val = 0.0f;
            float gds_val = gdS_w[si];
            float dp_val = dP_w[si];
            float gpdv_val = gPdV_w[si];
            float d_val = tile_D[i_local];
            float dot2_val = tile_dot2[i_local];
            float dot3_val = tile_dot3[i_local];

            float ds_val = p_val * (dp_val - d_val);
            float g_dp_val = p_val * (gds_val - dot2_val);
            float g_P_soft = gds_val * (dp_val - d_val) - dp_val * dot2_val;
            float g_P_val = g_P_soft + gpdv_val;
            float g_S_val = p_val * (g_P_val - dot3_val);

            dP_w[si]    = scale * ds_val;
            extra_w[si] = scale * g_S_val;
            gPdV_w[si]  = g_dp_val;
        }
        __syncwarp();

        // Output matmuls (transposed tall scores)
        wmma_output_matmul_acc_tallT_v6(g_K_frags, dP_w, B_SCORE_STRIDE, g_dQ_tile, D_PAD, D);
        wmma_output_matmul_acc_tallT_v6(g_K_frags, extra_w, B_SCORE_STRIDE, Q_tile, D_PAD, D);
        wmma_output_matmul_acc_tallT_v6(g_V_frags, gPdV_w, B_SCORE_STRIDE, dO_tile, D_PAD, D);
    }

    // Store g_K and g_V
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            K_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            g_K_frags[cb], D_PAD, wmma::mem_row_major);
    }
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            V_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            g_V_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V6_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        g_K_out[global_col * D + d] = K_block[c * D_PAD + d];
        g_V_out[global_col * D + d] = V_block[c * D_PAD + d];
    }
}

// ===========================================================================
// Forward declarations of v1 fallbacks
// ===========================================================================
extern std::vector<at::Tensor> flash_double_backward_cuda(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L);

extern std::vector<at::Tensor> flash_forward_cuda(
    at::Tensor Q, at::Tensor K, at::Tensor V);

extern std::vector<at::Tensor> flash_backward_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L);

// ===========================================================================
// Host wrapper: V6 forward
// ===========================================================================
std::vector<at::Tensor> v6_forward_cuda(
    at::Tensor Q, at::Tensor K, at::Tensor V, bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(Q.dtype() == at::kFloat, "Q must be float32");

    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    // Fallback to v1 for small N or D not multiple of 16
    if (N < 16 || D % 16 != 0) {
        TORCH_CHECK(!causal, "Causal mask requires N>=16 and D%16==0 (WMMA kernels)");
        return flash_forward_cuda(Q, K, V);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int D_PAD = D + V6_SMEM_PAD;
    int SCORE_STRIDE = V6_BC + V6_SCORE_PAD;

    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int grid = BH * blocks_per_head;

    int smem_fwd = (V6_BR * D_PAD + 2 * V6_BC * D_PAD
                    + V6_WARPS_PER_BLOCK * WMMA_M * SCORE_STRIDE
                    + 2 * V6_BR) * sizeof(float);

    if (smem_fwd > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v6_flash_fwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_fwd));
    }

    auto O = at::empty({B, H, N, D}, opts);
    auto L = at::empty({B, H, N}, opts);

    v6_flash_fwd_kernel<<<grid, V6_BLOCK_THREADS, smem_fwd>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        B, H, N, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());

    return {O, L};
}

// ===========================================================================
// Host wrapper: V6 backward
// ===========================================================================
std::vector<at::Tensor> v6_backward_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L, bool causal
) {
    TORCH_CHECK(dO.is_cuda(), "dO must be CUDA");
    TORCH_CHECK(dO.dtype() == at::kFloat, "dO must be float32");

    dO = dO.contiguous();
    Q  = Q.contiguous();
    K  = K.contiguous();
    V  = V.contiguous();
    O  = O.contiguous();
    L  = L.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    if (N < 16 || D % 16 != 0) {
        TORCH_CHECK(!causal, "Causal mask requires N>=16 and D%16==0 (WMMA kernels)");
        return flash_backward_cuda(dO, Q, K, V, O, L);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int D_PAD = D + V6_SMEM_PAD;
    int A_SCORE_STRIDE = V6_BC + V6_SCORE_PAD;  // 34
    int B_SCORE_STRIDE = WMMA_N + V6_SCORE_PAD;  // 18

    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int grid = BH * blocks_per_head;

    int smem_row = (2 * V6_BR * D_PAD + 2 * V6_BC * D_PAD
                    + V6_WARPS_PER_BLOCK * 2 * WMMA_M * A_SCORE_STRIDE
                    + 2 * V6_BR) * sizeof(float);

    int smem_col = (2 * V6_BR * D_PAD + 2 * V6_BC * D_PAD
                    + V6_WARPS_PER_BLOCK * 2 * V6_BC * B_SCORE_STRIDE
                    + 2 * V6_BC) * sizeof(float);

    if (smem_row > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v6_flash_bwd_row_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_row));
    }
    if (smem_col > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v6_flash_bwd_col_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_col));
    }

    auto D_vec = at::empty({B, H, N}, opts);
    auto dQ    = at::empty({B, H, N, D}, opts);
    auto dK    = at::empty({B, H, N, D}, opts);
    auto dV    = at::empty({B, H, N, D}, opts);

    v6_flash_bwd_row_kernel<<<grid, V6_BLOCK_THREADS, smem_row>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(), L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dQ.data_ptr<float>(),
        B, H, N, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());

    v6_flash_bwd_col_kernel<<<grid, V6_BLOCK_THREADS, smem_col>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(),
        B, H, N, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());

    return {dQ, dK, dV};
}

// ===========================================================================
// Host wrapper: V6 double backward
// ===========================================================================
std::vector<at::Tensor> v6_double_backward_cuda(
    at::Tensor g_dQ,
    at::Tensor g_dK,
    at::Tensor g_dV,
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L,
    bool causal
) {
    TORCH_CHECK(g_dQ.is_cuda(), "g_dQ must be CUDA");
    TORCH_CHECK(g_dQ.dtype() == at::kFloat, "g_dQ must be float32");

    g_dQ = g_dQ.contiguous();
    g_dK = g_dK.contiguous();
    g_dV = g_dV.contiguous();
    dO   = dO.contiguous();
    Q    = Q.contiguous();
    K    = K.contiguous();
    V    = V.contiguous();
    O    = O.contiguous();
    L    = L.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    // Fallback to v1 for small N or D not multiple of 16
    if (N < 16 || D % 16 != 0) {
        TORCH_CHECK(!causal, "Causal mask requires N>=16 and D%16==0 (WMMA kernels)");
        return flash_double_backward_cuda(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int D_PAD = D + V6_SMEM_PAD;
    int A_SCORE_STRIDE = V6_BC + V6_SCORE_PAD;  // 34
    int B_SCORE_STRIDE_val = WMMA_N + V6_SCORE_PAD;  // 18

    int blocks_per_head = (N + V6_BR - 1) / V6_BR;
    int grid = BH * blocks_per_head;

    int smem_A = (3 * V6_BR * D_PAD + 4 * V6_BC * D_PAD
                  + V6_WARPS_PER_BLOCK * 5 * WMMA_M * A_SCORE_STRIDE
                  + 4 * V6_BR) * sizeof(float);

    int smem_B = (4 * V6_BR * D_PAD + 3 * V6_BC * D_PAD
                  + V6_WARPS_PER_BLOCK * 5 * V6_BC * B_SCORE_STRIDE_val
                  + 4 * V6_BC) * sizeof(float);

    // Query device max shared memory per SM and fall back if needed
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int max_smem_per_sm;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_per_sm,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));

    if (smem_A > max_smem_per_sm || smem_B > max_smem_per_sm) {
        // D too large for this GPU's shared memory — fall back to V1 scalar flash
        TORCH_CHECK(!causal, "Causal double backward requires more shared memory ("
            + std::to_string(std::max(smem_A, smem_B) / 1024)
            + " KB) than this GPU supports ("
            + std::to_string(max_smem_per_sm / 1024)
            + " KB). Use a GPU with more shared memory (e.g. A100).");
        return flash_double_backward_cuda(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L);
    }

    if (smem_A > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v6_kernel_A_row,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_A));
    }
    if (smem_B > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v6_kernel_B_col,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_B));
    }

    auto D_vec    = at::empty({B, H, N}, opts);
    auto dot2     = at::empty({B, H, N}, opts);
    auto dot3     = at::empty({B, H, N}, opts);
    auto g_Q      = at::zeros({B, H, N, D}, opts);
    auto g_dO_out = at::zeros({B, H, N, D}, opts);
    auto g_K      = at::zeros({B, H, N, D}, opts);
    auto g_V      = at::zeros({B, H, N, D}, opts);

    v6_kernel_A_row<<<grid, V6_BLOCK_THREADS, smem_A>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_Q.data_ptr<float>(), g_dO_out.data_ptr<float>(),
        B, H, N, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());

    v6_kernel_B_col<<<grid, V6_BLOCK_THREADS, smem_B>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_K.data_ptr<float>(), g_V.data_ptr<float>(),
        B, H, N, D, scale, causal);
    CUDA_CHECK(cudaGetLastError());

    return {g_dO_out, g_Q, g_K, g_V};
}
