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

// Block structure: 4 warps, each handling 16 rows/cols => Br=64
#define V3_WARPS_PER_BLOCK 4
#define V3_BLOCK_THREADS (WARP_SIZE * V3_WARPS_PER_BLOCK)  // 128
#define V3_BR 64   // rows per block = 4 warps * 16
#define V3_BC 16   // column tile width = 1 wmma tile

// Padding
#define V3_SMEM_PAD 0  // No input padding (saves smem; WMMA loads are stride-agnostic)
#define V3_SCORE_PAD 2  // Score tile stride = V3_BC + V3_SCORE_PAD = 18 (must be even for sm_120 store_matrix_sync)

// ===========================================================================
// Warp-level sum reduction
// ===========================================================================
__device__ __forceinline__ float warp_reduce_sum_v3(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ===========================================================================
// Warp-level max reduction
// ===========================================================================
__device__ __forceinline__ float warp_reduce_max_v3(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}

// ===========================================================================
// Helper: compute 16x16 score matmul via WMMA
//
// Computes acc = A_smem[16][D_pad] @ B_smem[16][D_pad]^T  (zeroes acc first)
// A is row-major (16xD), B is row-major (16xD) => B^T via col_major load
// ===========================================================================
__device__ __forceinline__ void wmma_score_matmul(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    wmma::fill_fragment(acc, 0.0f);
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
}

// Accumulate version (doesn't zero acc first)
__device__ __forceinline__ void wmma_score_matmul_acc(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& acc,
    const float* A_smem, int A_stride,
    const float* B_smem, int B_stride,
    int D_dim
) {
    for (int kk = 0; kk < D_dim; kk += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
}

// ===========================================================================
// Helper: output matmul via WMMA
//
// Computes out_frags[cb] += score_smem[16][16] @ data_smem[16][D_pad]
// score_smem: 16x16 row-major, stride=score_stride
// data_smem:  16xD  row-major, stride=data_stride
// ===========================================================================
__device__ __forceinline__ void wmma_output_matmul_acc(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    int k_steps = V3_BC / WMMA_K;  // 16/8 = 2

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
            // A: score rows 0..15, cols kk*8..(kk+1)*8
            wmma::load_matrix_sync(a_frag, score_smem + kk * WMMA_K, score_stride);
            // B: data rows kk*8..(kk+1)*8, cols cb*16..(cb+1)*16
            wmma::load_matrix_sync(b_frag, data_smem + kk * WMMA_K * data_stride + cb * WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// Transposed score version: score_smem^T @ data_smem
// score_smem is [i][j] row-major. Col-major load gives score^T[j][i].
// Base offset for each k-step: kk * WMMA_K * score_stride (row offset, not column).
__device__ __forceinline__ void wmma_output_matmul_acc_scoreT(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    int k_steps = V3_BC / WMMA_K;  // 2

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
            // Col-major load: a_frag[j][k] = score_smem[(kk*8+k)*score_stride + j] = score[kk*8+k][j]
            // So a_frag represents score^T with rows j, cols kk*8..kk*8+7
            wmma::load_matrix_sync(a_frag, score_smem + kk * WMMA_K * score_stride, score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * WMMA_K * data_stride + cb * WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// Scale all elements in an accumulator fragment
__device__ __forceinline__ void scale_fragment(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag,
    float s
) {
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] *= s;
}

// ===========================================================================
// Kernel A (row kernel): 2 passes
//   Pass 1: D_i, dot2_i, A_i, E_i -> dot3_i
//   Pass 2: g_Q[i,:], g_dO[i,:]
//
// Block: Br=64 rows (4 warps x 16 rows each)
// Each warp processes 16 rows via WMMA against Bc=16 column tiles
//
// Shared memory layout (D_PAD = D+1, SCORE_STRIDE = 17):
//   Q_block    [BR][D_PAD]           3 * 64 * (D+1)  row data
//   dO_block   [BR][D_PAD]
//   g_dQ_block [BR][D_PAD]
//   K_tile     [BC][D_PAD]           4 * 16 * (D+1)  column tiles
//   V_tile     [BC][D_PAD]
//   g_dK_tile  [BC][D_PAD]
//   g_dV_tile  [BC][D_PAD]
//   score_base [4*5][16][17]         5440             per-warp score tiles
//   L_vec      [BR]                  64               scalars
//   Di_vec     [BR]                  64
//   dot2_smem  [BR]                  64               cached for pass 2
//   dot3_smem  [BR]                  64
// ===========================================================================
__global__ void v3_kernel_A_row(
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
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V3_SMEM_PAD;
    const int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    // -- Shared memory layout --
    float* Q_block    = smem;
    float* dO_block   = Q_block    + V3_BR * D_PAD;
    float* g_dQ_block = dO_block   + V3_BR * D_PAD;
    float* K_tile     = g_dQ_block + V3_BR * D_PAD;
    float* V_tile     = K_tile     + V3_BC * D_PAD;
    float* g_dK_tile  = V_tile     + V3_BC * D_PAD;
    float* g_dV_tile  = g_dK_tile  + V3_BC * D_PAD;
    float* score_base = g_dV_tile  + V3_BC * D_PAD;
    float* L_vec      = score_base + V3_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE;
    float* Di_vec     = L_vec      + V3_BR;
    float* dot2_smem  = Di_vec     + V3_BR;
    float* dot3_smem  = dot2_smem  + V3_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_row_start = blockIdx.x * V3_BR;
    int BH = B * H;
    int bh_block = block_row_start / N;

    // Per-warp score tiles
    float* S_w     = score_base + warp_id * 5 * WMMA_M * SCORE_STRIDE;
    float* gdS_w   = S_w     + WMMA_M * SCORE_STRIDE;
    float* dP_w    = gdS_w   + WMMA_M * SCORE_STRIDE;
    float* gPdV_w  = dP_w    + WMMA_M * SCORE_STRIDE;
    float* extra_w = gPdV_w  + WMMA_M * SCORE_STRIDE;

    // Per-warp row range in shared memory
    float* Q_warp    = Q_block    + warp_id * WMMA_M * D_PAD;
    float* dO_warp   = dO_block   + warp_id * WMMA_M * D_PAD;
    float* g_dQ_warp = g_dQ_block + warp_id * WMMA_M * D_PAD;

    int warp_row_start = block_row_start + warp_id * WMMA_M;
    int total_rows = BH * N;

    // ---- Load row data into shared memory (block-cooperative) ----
    int rows_to_load = min(V3_BR, total_rows - block_row_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Zero-fill (for partial blocks / padding columns)
    for (int idx = threadIdx.x; idx < V3_BR * D_PAD; idx += V3_BLOCK_THREADS) {
        Q_block[idx] = 0.0f;
        dO_block[idx] = 0.0f;
        g_dQ_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        int src = global_row * D + d;
        Q_block[r * D_PAD + d]    = Q[src];
        dO_block[r * D_PAD + d]   = dO[src];
        g_dQ_block[r * D_PAD + d] = g_dQ[src];
    }

    // Load L values and zero-init scalar arrays
    for (int idx = threadIdx.x; idx < V3_BR; idx += V3_BLOCK_THREADS) {
        int global_row = block_row_start + idx;
        L_vec[idx] = (global_row < total_rows) ? L[global_row] : 0.0f;
        Di_vec[idx] = 0.0f;
        dot2_smem[idx] = 0.0f;
        dot3_smem[idx] = 0.0f;
    }
    __syncthreads();

    // ---- Phase 0: D_i = sum_d dO[i,d] * O[i,d] ----
    for (int r = 0; r < WMMA_M; r++) {
        int global_row = warp_row_start + r;
        if (global_row >= total_rows) {
            if (lane == 0) Di_vec[warp_id * WMMA_M + r] = 0.0f;
            continue;
        }
        float partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE)
            partial += dO_block[(warp_id * WMMA_M + r) * D_PAD + d] * O[global_row * D + d];
        partial = warp_reduce_sum_v3(partial);
        if (lane == 0) {
            Di_vec[warp_id * WMMA_M + r] = partial;
            D_out[global_row] = partial;
        }
    }
    __syncthreads();

    // ====================================================================
    // Pass 1: Compute dot2, A, E -> dot3 using WMMA score matmuls
    //
    // Per-warp register accumulators (only lane 0 accumulates via reductions)
    // ====================================================================
    float dot2_acc[WMMA_M];
    float A_acc[WMMA_M];
    float E_acc[WMMA_M];
    for (int r = 0; r < WMMA_M; r++) {
        dot2_acc[r] = 0.0f;
        A_acc[r] = 0.0f;
        E_acc[r] = 0.0f;
    }

    for (int j_start = 0; j_start < N; j_start += V3_BC) {
        int tile_cols = min(V3_BC, N - j_start);

        // Load column tile (block-cooperative)
        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V3_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // WMMA score matmuls (4 matmuls per tile)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_frag;
        wmma_score_matmul(gdS_frag, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_acc(gdS_frag, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment(gdS_frag, scale);
        wmma::store_matrix_sync(gdS_w, gdS_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_frag;
        wmma_score_matmul(dP_frag, dO_warp, D_PAD, V_tile, D_PAD, D);
        wmma::store_matrix_sync(dP_w, dP_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_frag;
        wmma_score_matmul(gPdV_frag, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        wmma::store_matrix_sync(gPdV_w, gPdV_frag, SCORE_STRIDE, wmma::mem_row_major);

        __syncwarp();

        // Element-wise: per-row warp reduction for dot2, A, E
        // 16 rows x tile_cols columns. With tile_cols<=16 and WARP_SIZE=32,
        // each lane handles at most 1 column per row.
        for (int r = 0; r < WMMA_M; r++) {
            float p_dot2 = 0.0f, p_A = 0.0f, p_E = 0.0f;
            for (int c = lane; c < tile_cols; c += WARP_SIZE) {
                int si = r * SCORE_STRIDE + c;
                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                float p_val = __expf(s_val - l_val);
                float gds_val = gdS_w[si];
                float dp_val = dP_w[si];
                float gpdv_val = gPdV_w[si];

                float pg = p_val * gds_val;
                p_dot2 += pg;
                p_A    += pg * dp_val;
                p_E    += p_val * gpdv_val;
            }
            p_dot2 = warp_reduce_sum_v3(p_dot2);
            p_A    = warp_reduce_sum_v3(p_A);
            p_E    = warp_reduce_sum_v3(p_E);
            if (lane == 0) {
                dot2_acc[r] += p_dot2;
                A_acc[r]    += p_A;
                E_acc[r]    += p_E;
            }
        }
        __syncwarp();
    }

    // Finalize dot3 = A - 2*D*dot2 + E, write to global and shared memory
    if (lane == 0) {
        for (int r = 0; r < WMMA_M; r++) {
            int global_row = warp_row_start + r;
            if (global_row >= total_rows) continue;
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
    // Pass 2: g_Q[i,:] and g_dO[i,:] using WMMA output matmuls
    // ====================================================================
    int n_col_blocks = D / WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_Q_frags[4];  // max D=64 => 4
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_dO_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_Q_frags[cb], 0.0f);
        wmma::fill_fragment(g_dO_frags[cb], 0.0f);
    }

    for (int j_start = 0; j_start < N; j_start += V3_BC) {
        int tile_cols = min(V3_BC, N - j_start);

        // Load column tile
        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V3_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // Recompute scores via WMMA
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_frag;
        wmma_score_matmul(gdS_frag, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_acc(gdS_frag, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment(gdS_frag, scale);
        wmma::store_matrix_sync(gdS_w, gdS_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_frag;
        wmma_score_matmul(dP_frag, dO_warp, D_PAD, V_tile, D_PAD, D);
        wmma::store_matrix_sync(dP_w, dP_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_frag;
        wmma_score_matmul(gPdV_frag, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        wmma::store_matrix_sync(gPdV_w, gPdV_frag, SCORE_STRIDE, wmma::mem_row_major);

        __syncwarp();

        // Element-wise: compute dS, g_dP, g_S, P and write to score tiles
        // After: dP_w=dS, extra_w=g_S, S_w=P, gPdV_w=g_dP
        for (int idx = lane; idx < WMMA_M * V3_BC; idx += WARP_SIZE) {
            int r = idx / V3_BC;
            int c = idx % V3_BC;
            int si = r * SCORE_STRIDE + c;

            if (c < tile_cols) {
                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                float p_val = __expf(s_val - l_val);
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

                dP_w[si]    = scale * ds_val;    // scale * dS
                extra_w[si] = scale * g_S_val;   // scale * g_S
                S_w[si]     = p_val;             // P
                gPdV_w[si]  = g_dp_val;          // g_dP
            } else {
                dP_w[si]    = 0.0f;
                extra_w[si] = 0.0f;
                S_w[si]     = 0.0f;
                gPdV_w[si]  = 0.0f;
            }
        }
        __syncwarp();

        // Output matmuls:
        // g_Q  += (scale*dS) @ g_dK   +  (scale*g_S) @ K
        // g_dO += P @ g_dV             +  g_dP @ V
        wmma_output_matmul_acc(g_Q_frags, dP_w, SCORE_STRIDE, g_dK_tile, D_PAD, D);
        wmma_output_matmul_acc(g_Q_frags, extra_w, SCORE_STRIDE, K_tile, D_PAD, D);
        wmma_output_matmul_acc(g_dO_frags, S_w, SCORE_STRIDE, g_dV_tile, D_PAD, D);
        wmma_output_matmul_acc(g_dO_frags, gPdV_w, SCORE_STRIDE, V_tile, D_PAD, D);
    }

    // Store g_Q and g_dO to global memory via shared memory staging
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

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        g_Q_out[global_row * D + d]  = Q_block[r * D_PAD + d];
        g_dO_out[global_row * D + d] = dO_block[r * D_PAD + d];
    }
}

// ===========================================================================
// Kernel B (column kernel): 1 pass
//   g_K[j,:] and g_V[j,:]
//
// Block: 64 columns (4 warps x 16 cols each)
// Each warp processes 16 columns via WMMA against Bc=16 row tiles
//
// Shared memory layout:
//   K_block, V_block, g_dK_block, g_dV_block: [BR][D_PAD]  (column data)
//   Q_tile, dO_tile, g_dQ_tile:               [BC][D_PAD]  (row tiles)
//   score_base: [4*5][16][17]                               (per-warp scores)
//   tile_L, tile_D, tile_dot2, tile_dot3:     [BC]          (row scalars)
// ===========================================================================
__global__ void v3_kernel_B_col(
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
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V3_SMEM_PAD;
    const int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    float* K_block    = smem;
    float* V_block    = K_block    + V3_BR * D_PAD;
    float* g_dK_block = V_block    + V3_BR * D_PAD;
    float* g_dV_block = g_dK_block + V3_BR * D_PAD;
    float* Q_tile     = g_dV_block + V3_BR * D_PAD;
    float* dO_tile    = Q_tile     + V3_BC * D_PAD;
    float* g_dQ_tile  = dO_tile    + V3_BC * D_PAD;
    float* score_base = g_dQ_tile  + V3_BC * D_PAD;
    float* tile_L     = score_base + V3_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE;
    float* tile_D     = tile_L     + V3_BC;
    float* tile_dot2  = tile_D     + V3_BC;
    float* tile_dot3  = tile_dot2  + V3_BC;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * V3_BR;
    int BH = B * H;
    int bh_block = block_col_start / N;
    int total_cols = BH * N;

    float* S_w     = score_base + warp_id * 5 * WMMA_M * SCORE_STRIDE;
    float* gdS_w   = S_w     + WMMA_M * SCORE_STRIDE;
    float* dP_w    = gdS_w   + WMMA_M * SCORE_STRIDE;
    float* gPdV_w  = dP_w    + WMMA_M * SCORE_STRIDE;
    float* extra_w = gPdV_w  + WMMA_M * SCORE_STRIDE;

    float* K_warp    = K_block    + warp_id * WMMA_M * D_PAD;
    float* V_warp    = V_block    + warp_id * WMMA_M * D_PAD;
    float* g_dK_warp = g_dK_block + warp_id * WMMA_M * D_PAD;
    float* g_dV_warp = g_dV_block + warp_id * WMMA_M * D_PAD;

    // ---- Load column data (block-cooperative) ----
    int cols_to_load = min(V3_BR, total_cols - block_col_start);
    if (cols_to_load < 0) cols_to_load = 0;

    for (int idx = threadIdx.x; idx < V3_BR * D_PAD; idx += V3_BLOCK_THREADS) {
        K_block[idx] = 0.0f;
        V_block[idx] = 0.0f;
        g_dK_block[idx] = 0.0f;
        g_dV_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V3_BLOCK_THREADS) {
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

    // ---- Single pass over row tiles ----
    for (int i_start = 0; i_start < N; i_start += V3_BC) {
        int tile_rows = min(V3_BC, N - i_start);

        // Load row tile (block-cooperative)
        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            Q_tile[idx] = 0.0f;
            dO_tile[idx] = 0.0f;
            g_dQ_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_rows * D; idx += V3_BLOCK_THREADS) {
            int r = idx / D;
            int d = idx % D;
            int src = (bh_block * N + i_start + r) * D + d;
            Q_tile[r * D_PAD + d]    = Q[src];
            dO_tile[r * D_PAD + d]   = dO[src];
            g_dQ_tile[r * D_PAD + d] = g_dQ[src];
        }
        for (int idx = threadIdx.x; idx < V3_BC; idx += V3_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh_block * N + i_start + idx;
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

        // Score matmuls: S = Q_tile @ K_warp^T (16x16)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_tile, D_PAD, K_warp, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_frag;
        wmma_score_matmul(gdS_frag, g_dQ_tile, D_PAD, K_warp, D_PAD, D);
        wmma_score_matmul_acc(gdS_frag, Q_tile, D_PAD, g_dK_warp, D_PAD, D);
        scale_fragment(gdS_frag, scale);
        wmma::store_matrix_sync(gdS_w, gdS_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_frag;
        wmma_score_matmul(dP_frag, dO_tile, D_PAD, V_warp, D_PAD, D);
        wmma::store_matrix_sync(dP_w, dP_frag, SCORE_STRIDE, wmma::mem_row_major);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_frag;
        wmma_score_matmul(gPdV_frag, dO_tile, D_PAD, g_dV_warp, D_PAD, D);
        wmma::store_matrix_sync(gPdV_w, gPdV_frag, SCORE_STRIDE, wmma::mem_row_major);

        __syncwarp();

        // Element-wise: compute dS, g_S, g_dP and write to score tiles
        // Score layout: S_w[i_local][j_local], i from Q_tile, j from K_warp
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
            int i_local = idx / WMMA_N;
            int j_local = idx % WMMA_N;
            int si = i_local * SCORE_STRIDE + j_local;

            float s_val = S_w[si];
            float l_val = tile_L[i_local];
            float p_val = (i_local < tile_rows) ? __expf(s_val - l_val) : 0.0f;
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

            // Store with scale pre-applied for dS and g_S
            dP_w[si]    = scale * ds_val;   // scale * dS
            extra_w[si] = scale * g_S_val;  // scale * g_S
            gPdV_w[si]  = g_dp_val;         // g_dP
        }
        __syncwarp();

        // Output matmuls (transposed scores):
        // g_K += (scale*dS)^T @ g_dQ_tile  +  (scale*g_S)^T @ Q_tile
        // g_V += g_dP^T @ dO_tile
        wmma_output_matmul_acc_scoreT(g_K_frags, dP_w, SCORE_STRIDE, g_dQ_tile, D_PAD, D);
        wmma_output_matmul_acc_scoreT(g_K_frags, extra_w, SCORE_STRIDE, Q_tile, D_PAD, D);
        wmma_output_matmul_acc_scoreT(g_V_frags, gPdV_w, SCORE_STRIDE, dO_tile, D_PAD, D);
    }

    // Store g_K and g_V to global memory via shared memory staging
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

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V3_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        g_K_out[global_col * D + d] = K_block[c * D_PAD + d];
        g_V_out[global_col * D + d] = V_block[c * D_PAD + d];
    }
}

// ===========================================================================
// V3 Flash Forward Kernel (2-pass, WMMA tensor cores)
//
// Pass 1: Compute L = logsumexp per row (running max + sum of exp)
// Pass 2: Compute O = sum_j P_ij * V_j using WMMA output matmuls
//
// Block: BR=64 rows (4 warps x 16), BC=16 col tiles
//
// Shared memory layout:
//   Q_block  [BR][D_PAD]
//   K_tile   [BC][D_PAD]
//   V_tile   [BC][D_PAD]
//   score    [4][16][SCORE_STRIDE]  (1 per warp)
//   m_vec    [BR]
//   l_vec    [BR]
// ===========================================================================
__global__ void v3_flash_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    float* __restrict__ L_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V3_SMEM_PAD;
    const int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    float* Q_block    = smem;
    float* K_tile     = Q_block    + V3_BR * D_PAD;
    float* V_tile     = K_tile     + V3_BC * D_PAD;
    float* score_base = V_tile     + V3_BC * D_PAD;
    float* m_vec      = score_base + V3_WARPS_PER_BLOCK * WMMA_M * SCORE_STRIDE;
    float* l_vec      = m_vec      + V3_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing: avoid cross-(b,h) boundary issues
    int blocks_per_head = (N + V3_BR - 1) / V3_BR;
    int bh_block = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V3_BR;
    int block_row_start = bh_block * N + n_start;

    float* S_w    = score_base + warp_id * WMMA_M * SCORE_STRIDE;
    float* Q_warp = Q_block    + warp_id * WMMA_M * D_PAD;

    int rows_to_load = min(V3_BR, N - n_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Zero-fill and load Q_block
    for (int idx = threadIdx.x; idx < V3_BR * D_PAD; idx += V3_BLOCK_THREADS)
        Q_block[idx] = 0.0f;
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        Q_block[r * D_PAD + d] = Q[global_row * D + d];
    }

    // Initialize m and l
    for (int idx = threadIdx.x; idx < V3_BR; idx += V3_BLOCK_THREADS) {
        m_vec[idx] = -1e30f;
        l_vec[idx] = 0.0f;
    }
    __syncthreads();

    // ==== Pass 1: Compute L = logsumexp ====
    for (int j_start = 0; j_start < N; j_start += V3_BC) {
        int tile_cols = min(V3_BC, N - j_start);

        // Load K_tile
        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS)
            K_tile[idx] = 0.0f;
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V3_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            K_tile[c * D_PAD + d] = K[(bh_block * N + j_start + c) * D + d];
        }
        __syncthreads();

        // WMMA: S = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);
        __syncwarp();

        // Per-row max and sum update
        for (int r = 0; r < WMMA_M; r++) {
            if (warp_id * WMMA_M + r >= rows_to_load) continue;

            float local_max = -1e30f;
            for (int c = lane; c < tile_cols; c += WARP_SIZE)
                local_max = fmaxf(local_max, S_w[r * SCORE_STRIDE + c]);
            local_max = warp_reduce_max_v3(local_max);
            float tile_max = __shfl_sync(FULL_MASK, local_max, 0);

            float m_old = m_vec[warp_id * WMMA_M + r];
            float m_new = fmaxf(m_old, tile_max);
            float correction = __expf(m_old - m_new);

            float p_sum = 0.0f;
            for (int c = lane; c < tile_cols; c += WARP_SIZE)
                p_sum += __expf(S_w[r * SCORE_STRIDE + c] - m_new);
            p_sum = warp_reduce_sum_v3(p_sum);

            if (lane == 0) {
                l_vec[warp_id * WMMA_M + r] = l_vec[warp_id * WMMA_M + r] * correction + p_sum;
                m_vec[warp_id * WMMA_M + r] = m_new;
            }
        }
        __syncwarp();
    }

    // Finalize L = log(l) + m, store to global and overwrite l_vec for pass 2
    __syncthreads();
    for (int idx = threadIdx.x; idx < V3_BR; idx += V3_BLOCK_THREADS) {
        if (idx < rows_to_load) {
            int global_row = block_row_start + idx;
            float L_val = logf(l_vec[idx]) + m_vec[idx];
            L_out[global_row] = L_val;
            l_vec[idx] = L_val;
        }
    }
    __syncthreads();

    // ==== Pass 2: Compute O += P @ V using WMMA ====
    int n_col_blocks = D / WMMA_N;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> O_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++)
        wmma::fill_fragment(O_frags[cb], 0.0f);

    for (int j_start = 0; j_start < N; j_start += V3_BC) {
        int tile_cols = min(V3_BC, N - j_start);

        // Load K_tile and V_tile
        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V3_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d] = K[src];
            V_tile[c * D_PAD + d] = V[src];
        }
        __syncthreads();

        // WMMA: S = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);
        __syncwarp();

        // Element-wise: P = exp(S - L)
        for (int idx = lane; idx < WMMA_M * V3_BC; idx += WARP_SIZE) {
            int r = idx / V3_BC;
            int c = idx % V3_BC;
            int si = r * SCORE_STRIDE + c;
            if (c < tile_cols) {
                float L_val = l_vec[warp_id * WMMA_M + r];
                S_w[si] = __expf(S_w[si] - L_val);
            } else {
                S_w[si] = 0.0f;
            }
        }
        __syncwarp();

        // WMMA: O += P @ V
        wmma_output_matmul_acc(O_frags, S_w, SCORE_STRIDE, V_tile, D_PAD, D);
    }

    // Store O via shared memory staging (reuse Q_block)
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            Q_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            O_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        O_out[global_row * D + d] = Q_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V3 Flash Backward Row Kernel: Compute D_i and dQ
//
// Phase 0: D_i = sum_d dO[i,d] * O[i,d]
// Main loop: For each j-tile, WMMA score + dP, element-wise dS,
//            WMMA output matmul dQ += (scale*dS) @ K
//
// Shared memory layout:
//   Q_block  [BR][D_PAD]     (persists, reused as staging)
//   dO_block [BR][D_PAD]     (persists)
//   K_tile   [BC][D_PAD]     (reloaded)
//   V_tile   [BC][D_PAD]     (reloaded)
//   score    [4*2][16][SCORE_STRIDE]  (S + dP per warp)
//   L_vec    [BR]
//   Di_vec   [BR]
// ===========================================================================
__global__ void v3_flash_bwd_row_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ O,
    const float* __restrict__ L,
    float* __restrict__ D_out,
    float* __restrict__ dQ_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V3_SMEM_PAD;
    const int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    float* Q_block    = smem;
    float* dO_block   = Q_block    + V3_BR * D_PAD;
    float* K_tile     = dO_block   + V3_BR * D_PAD;
    float* V_tile     = K_tile     + V3_BC * D_PAD;
    float* score_base = V_tile     + V3_BC * D_PAD;
    float* L_vec      = score_base + V3_WARPS_PER_BLOCK * 2 * WMMA_M * SCORE_STRIDE;
    float* Di_vec     = L_vec      + V3_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing: avoid cross-(b,h) boundary issues
    int blocks_per_head = (N + V3_BR - 1) / V3_BR;
    int bh_block = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V3_BR;
    int block_row_start = bh_block * N + n_start;

    float* S_w     = score_base + warp_id * 2 * WMMA_M * SCORE_STRIDE;
    float* dP_w    = S_w + WMMA_M * SCORE_STRIDE;
    float* Q_warp  = Q_block  + warp_id * WMMA_M * D_PAD;
    float* dO_warp = dO_block + warp_id * WMMA_M * D_PAD;

    int rows_to_load = min(V3_BR, N - n_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Zero-fill and load row data
    for (int idx = threadIdx.x; idx < V3_BR * D_PAD; idx += V3_BLOCK_THREADS) {
        Q_block[idx] = 0.0f;
        dO_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        Q_block[r * D_PAD + d]  = Q[global_row * D + d];
        dO_block[r * D_PAD + d] = dO[global_row * D + d];
    }

    for (int idx = threadIdx.x; idx < V3_BR; idx += V3_BLOCK_THREADS) {
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
        partial = warp_reduce_sum_v3(partial);
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
    for (int j_start = 0; j_start < N; j_start += V3_BC) {
        int tile_cols = min(V3_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V3_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d] = K[src];
            V_tile[c * D_PAD + d] = V[src];
        }
        __syncthreads();

        // WMMA: S = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);

        // WMMA: dP = dO_warp @ V_tile^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_frag;
        wmma_score_matmul(dP_frag, dO_warp, D_PAD, V_tile, D_PAD, D);
        wmma::store_matrix_sync(dP_w, dP_frag, SCORE_STRIDE, wmma::mem_row_major);
        __syncwarp();

        // Element-wise: P = exp(S - L), dS = P * (dP - D_i), store scale*dS
        for (int idx = lane; idx < WMMA_M * V3_BC; idx += WARP_SIZE) {
            int r = idx / V3_BC;
            int c = idx % V3_BC;
            int si = r * SCORE_STRIDE + c;
            if (c < tile_cols) {
                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                float p_val = __expf(s_val - l_val);
                float dp_val = dP_w[si];
                float d_val = Di_vec[warp_id * WMMA_M + r];
                float ds_val = p_val * (dp_val - d_val);
                dP_w[si] = scale * ds_val;
            } else {
                dP_w[si] = 0.0f;
            }
        }
        __syncwarp();

        // WMMA: dQ += (scale*dS) @ K
        wmma_output_matmul_acc(dQ_frags, dP_w, SCORE_STRIDE, K_tile, D_PAD, D);
    }

    // Store dQ via shared memory staging (reuse Q_block)
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            Q_block + warp_id * WMMA_M * D_PAD + cb * WMMA_N,
            dQ_frags[cb], D_PAD, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V3_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        dQ_out[global_row * D + d] = Q_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V3 Flash Backward Col Kernel: Compute dK and dV
//
// Main loop: For each i-tile, WMMA score + dP matmuls, element-wise P/dS,
//            WMMA transposed output matmuls for dK and dV
//
// Shared memory layout:
//   K_block  [BR][D_PAD]     (persists, reused as staging)
//   V_block  [BR][D_PAD]     (persists, reused as staging)
//   Q_tile   [BC][D_PAD]     (reloaded)
//   dO_tile  [BC][D_PAD]     (reloaded)
//   score    [4*2][16][SCORE_STRIDE]  (S + dP per warp)
//   tile_L   [BC]
//   tile_D   [BC]
// ===========================================================================
__global__ void v3_flash_bwd_col_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vec_in,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    const int D_PAD = D + V3_SMEM_PAD;
    const int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    float* K_block    = smem;
    float* V_block    = K_block    + V3_BR * D_PAD;
    float* Q_tile     = V_block    + V3_BR * D_PAD;
    float* dO_tile    = Q_tile     + V3_BC * D_PAD;
    float* score_base = dO_tile    + V3_BC * D_PAD;
    float* tile_L     = score_base + V3_WARPS_PER_BLOCK * 2 * WMMA_M * SCORE_STRIDE;
    float* tile_D     = tile_L     + V3_BC;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Per-head block indexing: avoid cross-(b,h) boundary issues
    int blocks_per_head = (N + V3_BR - 1) / V3_BR;
    int bh_block = blockIdx.x / blocks_per_head;
    int n_start = (blockIdx.x % blocks_per_head) * V3_BR;
    int block_col_start = bh_block * N + n_start;

    float* S_w  = score_base + warp_id * 2 * WMMA_M * SCORE_STRIDE;
    float* dP_w = S_w + WMMA_M * SCORE_STRIDE;

    float* K_warp = K_block + warp_id * WMMA_M * D_PAD;
    float* V_warp = V_block + warp_id * WMMA_M * D_PAD;

    int cols_to_load = min(V3_BR, N - n_start);
    if (cols_to_load < 0) cols_to_load = 0;

    // Load K_block and V_block
    for (int idx = threadIdx.x; idx < V3_BR * D_PAD; idx += V3_BLOCK_THREADS) {
        K_block[idx] = 0.0f;
        V_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V3_BLOCK_THREADS) {
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
    for (int i_start = 0; i_start < N; i_start += V3_BC) {
        int tile_rows = min(V3_BC, N - i_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V3_BC * D_PAD; idx += V3_BLOCK_THREADS) {
            Q_tile[idx] = 0.0f;
            dO_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_rows * D; idx += V3_BLOCK_THREADS) {
            int r = idx / D;
            int d = idx % D;
            int src = (bh_block * N + i_start + r) * D + d;
            Q_tile[r * D_PAD + d]  = Q[src];
            dO_tile[r * D_PAD + d] = dO[src];
        }
        for (int idx = threadIdx.x; idx < V3_BC; idx += V3_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh_block * N + i_start + idx;
                tile_L[idx] = L[global_row];
                tile_D[idx] = D_vec_in[global_row];
            } else {
                tile_L[idx] = 0.0f;
                tile_D[idx] = 0.0f;
            }
        }
        __syncthreads();

        // WMMA: S = Q_tile @ K_warp^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_frag;
        wmma_score_matmul(S_frag, Q_tile, D_PAD, K_warp, D_PAD, D);
        scale_fragment(S_frag, scale);
        wmma::store_matrix_sync(S_w, S_frag, SCORE_STRIDE, wmma::mem_row_major);

        // WMMA: dP = dO_tile @ V_warp^T
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_frag;
        wmma_score_matmul(dP_frag, dO_tile, D_PAD, V_warp, D_PAD, D);
        wmma::store_matrix_sync(dP_w, dP_frag, SCORE_STRIDE, wmma::mem_row_major);
        __syncwarp();

        // Element-wise: P = exp(S - L), dS = P * (dP - D_i)
        // Store P to S_w, scale*dS to dP_w
        for (int idx = lane; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
            int i_local = idx / WMMA_N;
            int j_local = idx % WMMA_N;
            int si = i_local * SCORE_STRIDE + j_local;

            float s_val = S_w[si];
            float l_val = tile_L[i_local];
            float p_val = (i_local < tile_rows) ? __expf(s_val - l_val) : 0.0f;
            float dp_val = dP_w[si];
            float d_val = tile_D[i_local];
            float ds_val = p_val * (dp_val - d_val);

            S_w[si]  = p_val;           // P
            dP_w[si] = scale * ds_val;  // scale * dS
        }
        __syncwarp();

        // WMMA output matmuls (transposed scores):
        // dK += (scale*dS)^T @ Q_tile
        // dV += P^T @ dO_tile
        wmma_output_matmul_acc_scoreT(dK_frags, dP_w, SCORE_STRIDE, Q_tile, D_PAD, D);
        wmma_output_matmul_acc_scoreT(dV_frags, S_w, SCORE_STRIDE, dO_tile, D_PAD, D);
    }

    // Store dK and dV via shared memory staging
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

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V3_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        dK_out[global_col * D + d] = K_block[c * D_PAD + d];
        dV_out[global_col * D + d] = V_block[c * D_PAD + d];
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
// Host wrapper
// ===========================================================================
std::vector<at::Tensor> flash_double_backward_v3_cuda(
    at::Tensor g_dQ,
    at::Tensor g_dK,
    at::Tensor g_dV,
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L
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

    // Fall back to v1 for small N or D not multiple of 16
    if (N < 16 || D % 16 != 0) {
        return flash_double_backward_cuda(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int total_rows = BH * N;
    int D_PAD = D + V3_SMEM_PAD;
    int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    int grid = (total_rows + V3_BR - 1) / V3_BR;

    // Shared memory: Kernel A has 4 extra scalar arrays (L, Di, dot2, dot3)
    int smem_A = (3 * V3_BR * D_PAD + 4 * V3_BC * D_PAD
                  + V3_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE
                  + 4 * V3_BR) * sizeof(float);

    int smem_B = (4 * V3_BR * D_PAD + 3 * V3_BC * D_PAD
                  + V3_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE
                  + 4 * V3_BC) * sizeof(float);

    if (smem_A > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v3_kernel_A_row,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_A));
    }
    if (smem_B > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v3_kernel_B_col,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_B));
    }

    auto D_vec    = at::empty({B, H, N}, opts);
    auto dot2     = at::empty({B, H, N}, opts);
    auto dot3     = at::empty({B, H, N}, opts);
    auto g_Q      = at::zeros({B, H, N, D}, opts);
    auto g_dO_out = at::zeros({B, H, N, D}, opts);
    auto g_K      = at::zeros({B, H, N, D}, opts);
    auto g_V      = at::zeros({B, H, N, D}, opts);

    v3_kernel_A_row<<<grid, V3_BLOCK_THREADS, smem_A>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_Q.data_ptr<float>(), g_dO_out.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    v3_kernel_B_col<<<grid, V3_BLOCK_THREADS, smem_B>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_K.data_ptr<float>(), g_V.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {g_dO_out, g_Q, g_K, g_V};
}

// ===========================================================================
// Host wrapper — V3 flash forward
// ===========================================================================
std::vector<at::Tensor> flash_forward_v3_cuda(
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V
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

    // Fall back to v1 for small N or D not multiple of 16
    if (N < 16 || D % 16 != 0) {
        return flash_forward_cuda(Q, K, V);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int D_PAD = D + V3_SMEM_PAD;
    int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    int blocks_per_head = (N + V3_BR - 1) / V3_BR;
    int grid = BH * blocks_per_head;

    // smem: Q_block + K_tile + V_tile + score + m_vec + l_vec
    int smem_fwd = (V3_BR * D_PAD + 2 * V3_BC * D_PAD
                    + V3_WARPS_PER_BLOCK * WMMA_M * SCORE_STRIDE
                    + 2 * V3_BR) * sizeof(float);

    if (smem_fwd > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v3_flash_fwd_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_fwd));
    }

    auto O = at::empty({B, H, N, D}, opts);
    auto L = at::empty({B, H, N}, opts);

    v3_flash_fwd_kernel<<<grid, V3_BLOCK_THREADS, smem_fwd>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {O, L};
}

// ===========================================================================
// Host wrapper — V3 flash backward
// ===========================================================================
std::vector<at::Tensor> flash_backward_v3_cuda(
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L
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

    // Fall back to v1 for small N or D not multiple of 16
    if (N < 16 || D % 16 != 0) {
        return flash_backward_cuda(dO, Q, K, V, O, L);
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int D_PAD = D + V3_SMEM_PAD;
    int SCORE_STRIDE = V3_BC + V3_SCORE_PAD;

    int blocks_per_head = (N + V3_BR - 1) / V3_BR;
    int grid = BH * blocks_per_head;

    // Row kernel: Q_block + dO_block + K_tile + V_tile + 2 score tiles/warp + L + Di
    int smem_row = (2 * V3_BR * D_PAD + 2 * V3_BC * D_PAD
                    + V3_WARPS_PER_BLOCK * 2 * WMMA_M * SCORE_STRIDE
                    + 2 * V3_BR) * sizeof(float);

    // Col kernel: K_block + V_block + Q_tile + dO_tile + 2 score tiles/warp + tile_L + tile_D
    int smem_col = (2 * V3_BR * D_PAD + 2 * V3_BC * D_PAD
                    + V3_WARPS_PER_BLOCK * 2 * WMMA_M * SCORE_STRIDE
                    + 2 * V3_BC) * sizeof(float);

    if (smem_row > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v3_flash_bwd_row_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_row));
    }
    if (smem_col > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v3_flash_bwd_col_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_col));
    }

    auto D_vec = at::empty({B, H, N}, opts);
    auto dQ    = at::empty({B, H, N, D}, opts);
    auto dK    = at::empty({B, H, N, D}, opts);
    auto dV    = at::empty({B, H, N, D}, opts);

    v3_flash_bwd_row_kernel<<<grid, V3_BLOCK_THREADS, smem_row>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(), L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dQ.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    v3_flash_bwd_col_kernel<<<grid, V3_BLOCK_THREADS, smem_col>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {dQ, dK, dV};
}
