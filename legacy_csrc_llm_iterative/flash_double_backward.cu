#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                        \
                    cudaGetErrorString(err));                                   \
    } while (0)

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

// Tile size for column/row blocking in shared memory.
#define TILE_SIZE 32

// Number of warps per block — each warp handles one row (or column).
#define WARPS_PER_BLOCK 4
#define BLOCK_THREADS (WARP_SIZE * WARPS_PER_BLOCK)  // 128

// Padding for shared memory tiles to eliminate bank conflicts.
// Without padding, K_tile[lane * D + d] with D % 32 == 0 means all lanes
// hit the same bank. With +1 padding, bank = (lane*(D+1)+d) % 32 = (lane+d) % 32
// which spreads across all 32 banks.
#define SMEM_PAD 1

// ===========================================================================
// Optimized flash-style double backward for scaled dot-product attention.
//
// Two-kernel architecture:
//   Kernel A (row kernel): 2 passes (fused from original 3)
//     Pass 1: D_i, dot2_i, dot3_i (fused via algebraic identity)
//     Pass 2: g_Q[i,:], g_dO[i,:]
//   Kernel B (col kernel): 1 pass
//     g_K[j,:], g_V[j,:]
//
// Optimizations:
//   1. 3 passes → 2 passes: dot3 = A - 2*D*dot2 + E (eliminates a full pass)
//   2. Row/column vectors cached in shared memory
//   3. Shared memory padding eliminates bank conflicts
//   4. __expf() for faster exponentiation
// ===========================================================================

// ---------------------------------------------------------------------------
// Warp-level sum reduction
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Kernel A (row kernel): Compute D_i, dot2_i, dot3_i, g_Q[i,:], g_dO[i,:]
//
// Pass 1 (fused): Computes D_i, dot2_i, A_i, E_i in one pass over j-tiles.
//   dot3_i = A_i - 2*D_i*dot2_i + E_i  (algebraic identity)
//
// Pass 2: Computes g_Q[i,:] and g_dO[i,:] using dot2_i and dot3_i.
//
// Shared memory layout (tile_stride = D + SMEM_PAD):
//   K_tile    [TILE_SIZE * tile_stride]   \
//   V_tile    [TILE_SIZE * tile_stride]    | tiles (reloaded each iteration)
//   g_dK_tile [TILE_SIZE * tile_stride]    |
//   g_dV_tile [TILE_SIZE * tile_stride]   /
//   Q_rows    [WARPS_PER_BLOCK * D]       \
//   dO_rows   [WARPS_PER_BLOCK * D]        | row vectors (loaded once)
//   g_dQ_rows [WARPS_PER_BLOCK * D]       /
// ---------------------------------------------------------------------------
__global__ void flash_kernel_A_row(
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
    int tile_stride = D + SMEM_PAD;

    // Tile data (reloaded each tile iteration, shared by all warps)
    float* K_tile    = smem;
    float* V_tile    = K_tile    + TILE_SIZE * tile_stride;
    float* g_dK_tile = V_tile    + TILE_SIZE * tile_stride;
    float* g_dV_tile = g_dK_tile + TILE_SIZE * tile_stride;

    // Per-warp cached row vectors (loaded once, persist across tiles)
    float* row_base  = g_dV_tile + TILE_SIZE * tile_stride;
    // Layout: row_base[warp * 3 * D + vec_id * D + d]
    //   vec 0 = Q_i, vec 1 = dO_i, vec 2 = g_dQ_i

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_rows = B * H * N;
    bool active = (row_idx < total_rows);

    // Block-level bh for tile loading (same for all warps when N >= WARPS_PER_BLOCK)
    int bh_block = (blockIdx.x * WARPS_PER_BLOCK) / N;

    // Per-warp row info
    int i = 0, bh = bh_block;
    int row_off = 0;
    float l_val = 0.0f;
    if (active) {
        i  = row_idx % N;
        bh = row_idx / N;
        row_off = (bh * N + i) * D;
        l_val = L[bh * N + i];
    }

    int w = warp_in_block;
    float* Q_row    = row_base + w * 3 * D;
    float* dO_row   = Q_row + D;
    float* g_dQ_row = dO_row + D;

    // ---- Load row vectors into shared memory (one-time) ----
    if (active) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            Q_row[d]    = Q[row_off + d];
            dO_row[d]   = dO[row_off + d];
            g_dQ_row[d] = g_dQ[row_off + d];
        }
    }

    // ---- Phase 0: D_i = sum_d dO[i,d] * O[i,d] ----
    float d_val = 0.0f;
    if (active) {
        float d_partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE)
            d_partial += dO_row[d] * O[row_off + d];
        d_val = warp_reduce_sum(d_partial);
        d_val = __shfl_sync(FULL_MASK, d_val, 0);
        if (lane == 0) D_out[bh * N + i] = d_val;
    }

    // ====================================================================
    // Pass 1 (fused): dot2, A, E in a single pass over all j-tiles
    //
    // dot2_i = sum_j P_ij * g_dS_ij
    // A_i    = sum_j P_ij * g_dS_ij * dP_ij
    // E_i    = sum_j P_ij * dot(dO_i, g_dV_j)
    //
    // Then: dot3_i = A_i - 2 * D_i * dot2_i + E_i
    // ====================================================================
    float dot2_partial = 0.0f;
    float A_partial = 0.0f;
    float E_partial = 0.0f;

    for (int j_start = 0; j_start < N; j_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - j_start);

        // Cooperative tile load — all threads in block load together
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int tj = idx / D;
            int td = idx % D;
            int src = (bh_block * N + j_start + tj) * D + td;
            K_tile   [tj * tile_stride + td] = K[src];
            V_tile   [tj * tile_stride + td] = V[src];
            g_dK_tile[tj * tile_stride + td] = g_dK[src];
            g_dV_tile[tj * tile_stride + td] = g_dV[src];
        }
        __syncthreads();

        if (active) {
            // Each warp processes its row against this tile
            for (int tj = lane; tj < tile_len; tj += WARP_SIZE) {
                // Compute 5 dot products in one fused d-loop
                float s_val = 0.0f;
                float gds_part1 = 0.0f;  // dot(g_dQ_i, K_j)
                float gds_part2 = 0.0f;  // dot(Q_i, g_dK_j)
                float dp_val = 0.0f;     // dot(dO_i, V_j)
                float gPdV_val = 0.0f;   // dot(dO_i, g_dV_j)

                for (int d = 0; d < D; d++) {
                    float q_d  = Q_row[d];
                    float k_d  = K_tile[tj * tile_stride + d];
                    float gq_d = g_dQ_row[d];
                    float gk_d = g_dK_tile[tj * tile_stride + d];
                    float do_d = dO_row[d];
                    float v_d  = V_tile[tj * tile_stride + d];
                    float gv_d = g_dV_tile[tj * tile_stride + d];

                    s_val     += q_d * k_d;
                    gds_part1 += gq_d * k_d;
                    gds_part2 += q_d * gk_d;
                    dp_val    += do_d * v_d;
                    gPdV_val  += do_d * gv_d;
                }

                s_val *= scale;
                float p_val = __expf(s_val - l_val);
                float g_ds_val = scale * (gds_part1 + gds_part2);

                dot2_partial += p_val * g_ds_val;
                A_partial    += p_val * g_ds_val * dp_val;
                E_partial    += p_val * gPdV_val;
            }
        }
    }

    float dot2_val = 0.0f, dot3_val = 0.0f;
    if (active) {
        dot2_val = warp_reduce_sum(dot2_partial);
        dot2_val = __shfl_sync(FULL_MASK, dot2_val, 0);
        if (lane == 0) dot2_out[bh * N + i] = dot2_val;

        float A_val = warp_reduce_sum(A_partial);
        A_val = __shfl_sync(FULL_MASK, A_val, 0);
        float E_val = warp_reduce_sum(E_partial);
        E_val = __shfl_sync(FULL_MASK, E_val, 0);

        // Algebraic identity: dot3 = A - 2*D*dot2 + E
        dot3_val = A_val - 2.0f * d_val * dot2_val + E_val;
        if (lane == 0) dot3_out[bh * N + i] = dot3_val;
    }

    // ====================================================================
    // Pass 2: g_Q[i,:] and g_dO[i,:] (tiled output pass)
    // ====================================================================
    float g_Q_acc[128];
    float g_dO_acc[128];
    for (int d = 0; d < D; d++) {
        g_Q_acc[d] = 0.0f;
        g_dO_acc[d] = 0.0f;
    }

    for (int j_start = 0; j_start < N; j_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int tj = idx / D;
            int td = idx % D;
            int src = (bh_block * N + j_start + tj) * D + td;
            K_tile   [tj * tile_stride + td] = K[src];
            V_tile   [tj * tile_stride + td] = V[src];
            g_dK_tile[tj * tile_stride + td] = g_dK[src];
            g_dV_tile[tj * tile_stride + td] = g_dV[src];
        }
        __syncthreads();

        if (active) {
            for (int tj = lane; tj < tile_len; tj += WARP_SIZE) {
                // Recompute all 5 dot products
                float s_val = 0.0f;
                float gds_part1 = 0.0f;
                float gds_part2 = 0.0f;
                float dp_val = 0.0f;
                float gPdV_val = 0.0f;

                for (int d = 0; d < D; d++) {
                    float q_d  = Q_row[d];
                    float k_d  = K_tile[tj * tile_stride + d];
                    float gq_d = g_dQ_row[d];
                    float gk_d = g_dK_tile[tj * tile_stride + d];
                    float do_d = dO_row[d];
                    float v_d  = V_tile[tj * tile_stride + d];
                    float gv_d = g_dV_tile[tj * tile_stride + d];

                    s_val     += q_d * k_d;
                    gds_part1 += gq_d * k_d;
                    gds_part2 += q_d * gk_d;
                    dp_val    += do_d * v_d;
                    gPdV_val  += do_d * gv_d;
                }

                s_val *= scale;
                float p_val = __expf(s_val - l_val);
                float g_ds_val = scale * (gds_part1 + gds_part2);

                float ds_val = p_val * (dp_val - d_val);
                float g_dp_val = p_val * (g_ds_val - dot2_val);
                float g_P_soft = g_ds_val * (dp_val - d_val) - dp_val * dot2_val;
                float g_P_val = g_P_soft + gPdV_val;
                float g_S_val = p_val * (g_P_val - dot3_val);

                // Accumulate vector outputs
                for (int d = 0; d < D; d++) {
                    float k_d  = K_tile[tj * tile_stride + d];
                    float gk_d = g_dK_tile[tj * tile_stride + d];
                    float gv_d = g_dV_tile[tj * tile_stride + d];
                    float v_d  = V_tile[tj * tile_stride + d];

                    g_Q_acc[d]  += scale * ds_val * gk_d + scale * g_S_val * k_d;
                    g_dO_acc[d] += p_val * gv_d + g_dp_val * v_d;
                }
            }
        }
    }

    if (active) {
        // Warp reduction for D-element accumulators
        for (int d = 0; d < D; d++) {
            g_Q_acc[d] = warp_reduce_sum(g_Q_acc[d]);
            g_dO_acc[d] = warp_reduce_sum(g_dO_acc[d]);
        }
        if (lane == 0) {
            for (int d = 0; d < D; d++) {
                g_Q_out[row_off + d] = g_Q_acc[d];
                g_dO_out[row_off + d] = g_dO_acc[d];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel B (column kernel): Compute g_K[j,:] and g_V[j,:]
//
// One warp per column j. Iterates over row tiles (i).
//
// Shared memory layout (tile_stride = D + SMEM_PAD):
//   Q_tile    [TILE_SIZE * tile_stride]   \
//   dO_tile   [TILE_SIZE * tile_stride]    | tiles
//   g_dQ_tile [TILE_SIZE * tile_stride]   /
//   row_L     [TILE_SIZE]                  \
//   row_D     [TILE_SIZE]                   | row scalars
//   row_dot2  [TILE_SIZE]                   |
//   row_dot3  [TILE_SIZE]                  /
//   K_cols    [WARPS_PER_BLOCK * D]        \
//   V_cols    [WARPS_PER_BLOCK * D]         | column vectors (loaded once)
//   g_dK_cols [WARPS_PER_BLOCK * D]         |
//   g_dV_cols [WARPS_PER_BLOCK * D]        /
// ---------------------------------------------------------------------------
__global__ void flash_kernel_B_col(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ g_dQ,
    const float* __restrict__ g_dK,
    const float* __restrict__ g_dV,
    const float* __restrict__ L,
    const float* __restrict__ D_vec,
    const float* __restrict__ dot2,
    const float* __restrict__ dot3,
    float* __restrict__ g_K_out,
    float* __restrict__ g_V_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    int tile_stride = D + SMEM_PAD;

    // Tile data (reloaded each iteration)
    float* Q_tile    = smem;
    float* dO_tile   = Q_tile    + TILE_SIZE * tile_stride;
    float* g_dQ_tile = dO_tile   + TILE_SIZE * tile_stride;

    // Row scalars
    float* row_L     = g_dQ_tile + TILE_SIZE * tile_stride;
    float* row_D     = row_L    + TILE_SIZE;
    float* row_dot2  = row_D    + TILE_SIZE;
    float* row_dot3  = row_dot2 + TILE_SIZE;

    // Per-warp cached column vectors (loaded once)
    float* col_base  = row_dot3 + TILE_SIZE;
    // Layout: col_base[warp * 4 * D + vec_id * D + d]
    //   vec 0 = K_j, vec 1 = V_j, vec 2 = g_dK_j, vec 3 = g_dV_j

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_cols = B * H * N;
    bool active = (col_idx < total_cols);

    int bh_block = (blockIdx.x * WARPS_PER_BLOCK) / N;

    int j = 0, bh = bh_block;
    int col_off = 0;
    if (active) {
        j  = col_idx % N;
        bh = col_idx / N;
        col_off = (bh * N + j) * D;
    }

    int w = warp_in_block;
    float* K_col    = col_base + w * 4 * D;
    float* V_col    = K_col + D;
    float* g_dK_col = V_col + D;
    float* g_dV_col = g_dK_col + D;

    // ---- Load column vectors into shared memory (one-time) ----
    if (active) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            K_col[d]    = K[col_off + d];
            V_col[d]    = V[col_off + d];
            g_dK_col[d] = g_dK[col_off + d];
            g_dV_col[d] = g_dV[col_off + d];
        }
    }

    float g_K_acc[128];
    float g_V_acc[128];
    for (int d = 0; d < D; d++) {
        g_K_acc[d] = 0.0f;
        g_V_acc[d] = 0.0f;
    }

    for (int i_start = 0; i_start < N; i_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - i_start);

        // Cooperative tile load
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int ti = idx / D;
            int td = idx % D;
            int src = (bh_block * N + i_start + ti) * D + td;
            Q_tile   [ti * tile_stride + td] = Q[src];
            dO_tile  [ti * tile_stride + td] = dO[src];
            g_dQ_tile[ti * tile_stride + td] = g_dQ[src];
        }
        // Load row scalars
        for (int idx = threadIdx.x; idx < tile_len; idx += BLOCK_THREADS) {
            int si = bh_block * N + i_start + idx;
            row_L[idx]    = L[si];
            row_D[idx]    = D_vec[si];
            row_dot2[idx] = dot2[si];
            row_dot3[idx] = dot3[si];
        }
        __syncthreads();

        if (active) {
            for (int ti = lane; ti < tile_len; ti += WARP_SIZE) {
                float l_val = row_L[ti];
                float d_val = row_D[ti];
                float dot2_val = row_dot2[ti];
                float dot3_val = row_dot3[ti];

                // Compute dot products using cached column vectors
                float s_val = 0.0f;
                float gds_part1 = 0.0f;
                float gds_part2 = 0.0f;
                float dp_val = 0.0f;
                float gPdV_val = 0.0f;

                for (int d = 0; d < D; d++) {
                    float q_d  = Q_tile[ti * tile_stride + d];
                    float k_d  = K_col[d];
                    float gq_d = g_dQ_tile[ti * tile_stride + d];
                    float gk_d = g_dK_col[d];
                    float do_d = dO_tile[ti * tile_stride + d];
                    float v_d  = V_col[d];
                    float gv_d = g_dV_col[d];

                    s_val     += q_d * k_d;
                    gds_part1 += gq_d * k_d;
                    gds_part2 += q_d * gk_d;
                    dp_val    += do_d * v_d;
                    gPdV_val  += do_d * gv_d;
                }

                s_val *= scale;
                float p_val = __expf(s_val - l_val);
                float g_ds_val = scale * (gds_part1 + gds_part2);

                float ds_val = p_val * (dp_val - d_val);
                float g_dp_val = p_val * (g_ds_val - dot2_val);

                float g_P_soft = g_ds_val * (dp_val - d_val) - dp_val * dot2_val;
                float g_P_val = g_P_soft + gPdV_val;
                float g_S_val = p_val * (g_P_val - dot3_val);

                // Accumulate vector outputs using cached column + tile data
                for (int d = 0; d < D; d++) {
                    float gq_d = g_dQ_tile[ti * tile_stride + d];
                    float q_d  = Q_tile[ti * tile_stride + d];
                    float do_d = dO_tile[ti * tile_stride + d];

                    g_K_acc[d] += scale * ds_val * gq_d + scale * g_S_val * q_d;
                    g_V_acc[d] += g_dp_val * do_d;
                }
            }
        }
    }

    if (active) {
        // Warp reduction
        for (int d = 0; d < D; d++) {
            g_K_acc[d] = warp_reduce_sum(g_K_acc[d]);
            g_V_acc[d] = warp_reduce_sum(g_V_acc[d]);
        }
        if (lane == 0) {
            for (int d = 0; d < D; d++) {
                g_K_out[col_off + d] = g_K_acc[d];
                g_V_out[col_off + d] = g_V_acc[d];
            }
        }
    }
}

// ===========================================================================
// Flash forward kernel: compute O and L without materializing S or P
//
// Single-pass online softmax (1 warp per row):
//   Loads K+V tiles once. Uses online softmax to maintain running
//   max m, sum l, and output O_acc with incremental corrections.
//
// Memory: O(ND) — no N×N intermediates
//
// Shared memory layout (tile_stride = D + SMEM_PAD):
//   K_tile  [TILE_SIZE * tile_stride]
//   V_tile  [TILE_SIZE * tile_stride]
//   Q_rows  [WARPS_PER_BLOCK * D]
// ===========================================================================
__global__ void flash_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O_out,
    float* __restrict__ L_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    int tile_stride = D + SMEM_PAD;

    float* K_tile = smem;
    float* V_tile = K_tile + TILE_SIZE * tile_stride;
    float* row_base = V_tile + TILE_SIZE * tile_stride;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_rows = B * H * N;
    bool active = (row_idx < total_rows);
    int bh_block = (blockIdx.x * WARPS_PER_BLOCK) / N;

    int i = 0, bh = bh_block, row_off = 0;
    if (active) {
        i = row_idx % N;
        bh = row_idx / N;
        row_off = (bh * N + i) * D;
    }

    int w = warp_in_block;
    float* Q_row = row_base + w * D;

    if (active) {
        for (int d = lane; d < D; d += WARP_SIZE)
            Q_row[d] = Q[row_off + d];
    }

    // ================================================================
    // Single pass: online softmax with running (m, l, O_acc)
    //
    // For each j:
    //   s = Q_i · K_j * scale
    //   m_new = max(m, s)
    //   correction = exp(m - m_new)
    //   p = exp(s - m_new)
    //   l = l * correction + p
    //   O_acc = O_acc * correction + p * V_j
    //   m = m_new
    //
    // After all j: O = O_acc / l, L = log(l) + m
    // ================================================================
    float m_val = -1e30f;
    float l_val = 0.0f;
    float O_acc[128];
    for (int d = 0; d < D; d++) O_acc[d] = 0.0f;

    for (int j_start = 0; j_start < N; j_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int tj = idx / D;
            int td = idx % D;
            int src = (bh_block * N + j_start + tj) * D + td;
            K_tile[tj * tile_stride + td] = K[src];
            V_tile[tj * tile_stride + td] = V[src];
        }
        __syncthreads();

        if (active) {
            for (int tj = lane; tj < tile_len; tj += WARP_SIZE) {
                float s_val = 0.0f;
                for (int d = 0; d < D; d++)
                    s_val += Q_row[d] * K_tile[tj * tile_stride + d];
                s_val *= scale;

                // Online softmax update
                float m_new = fmaxf(m_val, s_val);
                float correction = __expf(m_val - m_new);
                float p_val = __expf(s_val - m_new);
                l_val = l_val * correction + p_val;
                for (int d = 0; d < D; d++)
                    O_acc[d] = O_acc[d] * correction + p_val * V_tile[tj * tile_stride + d];
                m_val = m_new;
            }
        }
    }

    // Warp-level merge of (m, l, O_acc) across lanes
    if (active) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float m_other = __shfl_down_sync(FULL_MASK, m_val, offset);
            float l_other = __shfl_down_sync(FULL_MASK, l_val, offset);
            float m_max = fmaxf(m_val, m_other);
            float c1 = __expf(m_val - m_max);
            float c2 = __expf(m_other - m_max);
            l_val = l_val * c1 + l_other * c2;
            for (int d = 0; d < D; d++) {
                float O_other = __shfl_down_sync(FULL_MASK, O_acc[d], offset);
                O_acc[d] = O_acc[d] * c1 + O_other * c2;
            }
            m_val = m_max;
        }

        if (lane == 0) {
            float inv_l = 1.0f / l_val;
            for (int d = 0; d < D; d++)
                O_out[row_off + d] = O_acc[d] * inv_l;
            L_out[bh * N + i] = logf(l_val) + m_val;
        }
    }
}

// ===========================================================================
// Host wrapper — flash forward
// ===========================================================================
std::vector<at::Tensor> flash_forward_cuda(
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
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    int rows = B * H * N;
    int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int tile_stride = D + SMEM_PAD;

    // 2 padded tiles + 1 row vector per warp
    int smem_fwd = (2 * TILE_SIZE * tile_stride + WARPS_PER_BLOCK * D) * sizeof(float);

    auto O = at::empty({B, H, N, D}, opts);
    auto L = at::empty({B, H, N}, opts);

    flash_fwd_kernel<<<grid, BLOCK_THREADS, smem_fwd>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {O, L};
}

// ===========================================================================
// Flash backward kernels: compute dQ, dK, dV without materializing P
//
// Replaces the naive backward (which needs P as input) with a flash-style
// approach that recomputes P tile-by-tile from Q, K, L.
//
// Kernel C (row): compute D_i and dQ[i,:]
// Kernel D (col): compute dK[j,:] and dV[j,:]
// ===========================================================================

// ---------------------------------------------------------------------------
// Kernel C (row kernel): Compute D_i and dQ[i,:]
//
// dQ[i,d] = scale * sum_j dS_ij * K[j,d]
//   where dS_ij = P_ij * (dP_ij - D_i)
//         P_ij  = exp(Q_i · K_j * scale - L_i)
//         dP_ij = dO_i · V_j
//
// Shared memory (tile_stride = D + SMEM_PAD):
//   K_tile  [TILE_SIZE * tile_stride]
//   V_tile  [TILE_SIZE * tile_stride]
//   Q_rows  [WARPS_PER_BLOCK * D]
//   dO_rows [WARPS_PER_BLOCK * D]
// ---------------------------------------------------------------------------
__global__ void flash_bwd_row_kernel(
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
    int tile_stride = D + SMEM_PAD;

    float* K_tile  = smem;
    float* V_tile  = K_tile + TILE_SIZE * tile_stride;
    float* row_base = V_tile + TILE_SIZE * tile_stride;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int row_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_rows = B * H * N;
    bool active = (row_idx < total_rows);
    int bh_block = (blockIdx.x * WARPS_PER_BLOCK) / N;

    int i = 0, bh = bh_block, row_off = 0;
    float l_val = 0.0f;
    if (active) {
        i = row_idx % N;
        bh = row_idx / N;
        row_off = (bh * N + i) * D;
        l_val = L[bh * N + i];
    }

    int w = warp_in_block;
    float* Q_row  = row_base + w * 2 * D;
    float* dO_row = Q_row + D;

    if (active) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            Q_row[d]  = Q[row_off + d];
            dO_row[d] = dO[row_off + d];
        }
    }

    // D_i = sum_d dO[i,d] * O[i,d]
    float d_val = 0.0f;
    if (active) {
        float d_partial = 0.0f;
        for (int d = lane; d < D; d += WARP_SIZE)
            d_partial += dO_row[d] * O[row_off + d];
        d_val = warp_reduce_sum(d_partial);
        d_val = __shfl_sync(FULL_MASK, d_val, 0);
        if (lane == 0) D_out[bh * N + i] = d_val;
    }

    // dQ[i,:] accumulator
    float dQ_acc[128];
    for (int d = 0; d < D; d++) dQ_acc[d] = 0.0f;

    for (int j_start = 0; j_start < N; j_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int tj = idx / D;
            int td = idx % D;
            int src = (bh_block * N + j_start + tj) * D + td;
            K_tile[tj * tile_stride + td] = K[src];
            V_tile[tj * tile_stride + td] = V[src];
        }
        __syncthreads();

        if (active) {
            for (int tj = lane; tj < tile_len; tj += WARP_SIZE) {
                float s_val = 0.0f;
                float dp_val = 0.0f;
                for (int d = 0; d < D; d++) {
                    s_val  += Q_row[d]  * K_tile[tj * tile_stride + d];
                    dp_val += dO_row[d] * V_tile[tj * tile_stride + d];
                }
                s_val *= scale;
                float p_val = __expf(s_val - l_val);
                float ds_val = p_val * (dp_val - d_val);

                for (int d = 0; d < D; d++)
                    dQ_acc[d] += ds_val * K_tile[tj * tile_stride + d];
            }
        }
    }

    if (active) {
        for (int d = 0; d < D; d++) {
            dQ_acc[d] = warp_reduce_sum(dQ_acc[d]);
        }
        if (lane == 0) {
            for (int d = 0; d < D; d++)
                dQ_out[row_off + d] = scale * dQ_acc[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel D (column kernel): Compute dK[j,:] and dV[j,:]
//
// dK[j,d] = scale * sum_i dS_ij * Q[i,d]
// dV[j,d] = sum_i P_ij * dO[i,d]
//
// Shared memory (tile_stride = D + SMEM_PAD):
//   Q_tile   [TILE_SIZE * tile_stride]
//   dO_tile  [TILE_SIZE * tile_stride]
//   row_L    [TILE_SIZE]
//   row_D    [TILE_SIZE]
//   K_cols   [WARPS_PER_BLOCK * D]
//   V_cols   [WARPS_PER_BLOCK * D]
// ---------------------------------------------------------------------------
__global__ void flash_bwd_col_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vec,
    float* __restrict__ dK_out,
    float* __restrict__ dV_out,
    int B, int H, int N, int D, float scale
) {
    extern __shared__ float smem[];
    int tile_stride = D + SMEM_PAD;

    float* Q_tile  = smem;
    float* dO_tile = Q_tile + TILE_SIZE * tile_stride;
    float* row_L   = dO_tile + TILE_SIZE * tile_stride;
    float* row_D   = row_L + TILE_SIZE;
    float* col_base = row_D + TILE_SIZE;

    int warp_in_block = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int col_idx = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_cols = B * H * N;
    bool active = (col_idx < total_cols);
    int bh_block = (blockIdx.x * WARPS_PER_BLOCK) / N;

    int j = 0, bh = bh_block, col_off = 0;
    if (active) {
        j = col_idx % N;
        bh = col_idx / N;
        col_off = (bh * N + j) * D;
    }

    int w = warp_in_block;
    float* K_col = col_base + w * 2 * D;
    float* V_col = K_col + D;

    if (active) {
        for (int d = lane; d < D; d += WARP_SIZE) {
            K_col[d] = K[col_off + d];
            V_col[d] = V[col_off + d];
        }
    }

    float dK_acc[128], dV_acc[128];
    for (int d = 0; d < D; d++) { dK_acc[d] = 0.0f; dV_acc[d] = 0.0f; }

    for (int i_start = 0; i_start < N; i_start += TILE_SIZE) {
        int tile_len = min(TILE_SIZE, N - i_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_len * D; idx += BLOCK_THREADS) {
            int ti = idx / D;
            int td = idx % D;
            int src = (bh_block * N + i_start + ti) * D + td;
            Q_tile [ti * tile_stride + td] = Q[src];
            dO_tile[ti * tile_stride + td] = dO[src];
        }
        for (int idx = threadIdx.x; idx < tile_len; idx += BLOCK_THREADS) {
            int si = bh_block * N + i_start + idx;
            row_L[idx] = L[si];
            row_D[idx] = D_vec[si];
        }
        __syncthreads();

        if (active) {
            for (int ti = lane; ti < tile_len; ti += WARP_SIZE) {
                float l_val = row_L[ti];
                float d_val = row_D[ti];

                float s_val = 0.0f;
                float dp_val = 0.0f;
                for (int d = 0; d < D; d++) {
                    s_val  += Q_tile[ti * tile_stride + d] * K_col[d];
                    dp_val += dO_tile[ti * tile_stride + d] * V_col[d];
                }
                s_val *= scale;
                float p_val = __expf(s_val - l_val);
                float ds_val = p_val * (dp_val - d_val);

                for (int d = 0; d < D; d++) {
                    dK_acc[d] += ds_val * Q_tile[ti * tile_stride + d];
                    dV_acc[d] += p_val  * dO_tile[ti * tile_stride + d];
                }
            }
        }
    }

    if (active) {
        for (int d = 0; d < D; d++) {
            dK_acc[d] = warp_reduce_sum(dK_acc[d]);
            dV_acc[d] = warp_reduce_sum(dV_acc[d]);
        }
        if (lane == 0) {
            for (int d = 0; d < D; d++) {
                dK_out[col_off + d] = scale * dK_acc[d];
                dV_out[col_off + d] = dV_acc[d];
            }
        }
    }
}

// ===========================================================================
// Host wrapper — flash backward
// ===========================================================================

std::vector<at::Tensor> flash_backward_cuda(
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
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    int rows = B * H * N;
    int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int tile_stride = D + SMEM_PAD;

    // Kernel C: 2 padded tiles + 2 row vectors per warp
    int smem_C = (2 * TILE_SIZE * tile_stride + 2 * WARPS_PER_BLOCK * D) * sizeof(float);
    // Kernel D: 2 padded tiles + 2 scalar arrays + 2 column vectors per warp
    int smem_D = (2 * TILE_SIZE * tile_stride + 2 * TILE_SIZE + 2 * WARPS_PER_BLOCK * D) * sizeof(float);

    auto D_vec = at::empty({B, H, N}, opts);
    auto dQ    = at::empty({B, H, N, D}, opts);
    auto dK    = at::empty({B, H, N, D}, opts);
    auto dV    = at::empty({B, H, N, D}, opts);

    flash_bwd_row_kernel<<<grid, BLOCK_THREADS, smem_C>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(), L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dQ.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    flash_bwd_col_kernel<<<grid, BLOCK_THREADS, smem_D>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dK.data_ptr<float>(), dV.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {dQ, dK, dV};
}

// ===========================================================================
// Host wrapper — flash double backward
// ===========================================================================

std::vector<at::Tensor> flash_double_backward_cuda(
    at::Tensor g_dQ,   // (B,H,N,D)
    at::Tensor g_dK,   // (B,H,N,D)
    at::Tensor g_dV,   // (B,H,N,D)
    at::Tensor dO,     // (B,H,N,D)
    at::Tensor Q,      // (B,H,N,D)
    at::Tensor K,      // (B,H,N,D)
    at::Tensor V,      // (B,H,N,D)
    at::Tensor O,      // (B,H,N,D)
    at::Tensor L       // (B,H,N) — logsumexp
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
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    int rows = B * H * N;
    int grid = (rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int tile_stride = D + SMEM_PAD;

    // Shared memory sizes
    // Kernel A: 4 padded tiles + 3 row vectors per warp
    int smem_A = (4 * TILE_SIZE * tile_stride + 3 * WARPS_PER_BLOCK * D) * sizeof(float);
    // Kernel B: 3 padded tiles + 4 scalar arrays + 4 column vectors per warp
    int smem_B = (3 * TILE_SIZE * tile_stride + 4 * TILE_SIZE + 4 * WARPS_PER_BLOCK * D) * sizeof(float);

    // Outputs
    auto D_vec    = at::empty({B, H, N}, opts);
    auto dot2     = at::empty({B, H, N}, opts);
    auto dot3     = at::empty({B, H, N}, opts);
    auto g_Q      = at::empty({B, H, N, D}, opts);
    auto g_dO_out = at::empty({B, H, N, D}, opts);
    auto g_K      = at::empty({B, H, N, D}, opts);
    auto g_V      = at::empty({B, H, N, D}, opts);

    // ---- Kernel A: D, dot2, dot3, g_Q, g_dO ----
    flash_kernel_A_row<<<grid, BLOCK_THREADS, smem_A>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_Q.data_ptr<float>(), g_dO_out.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    // ---- Kernel B: g_K, g_V ----
    flash_kernel_B_col<<<grid, BLOCK_THREADS, smem_B>>>(
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
