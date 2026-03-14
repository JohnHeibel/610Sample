#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
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
#define V5_WARPS_PER_BLOCK 4
#define V5_BLOCK_THREADS (WARP_SIZE * V5_WARPS_PER_BLOCK)  // 128
#define V5_BR 64   // rows per block = 4 warps * 16
#define V5_BC 32   // column tile width = 2 wmma tiles (doubled from V3's 16)

// Padding
#define V5_SMEM_PAD 0
#define V5_SCORE_PAD 2  // Score tile stride must be even for sm_120 store_matrix_sync

// ===========================================================================
// Fused elementwise kernel: row reductions + elementwise transforms
//
// Reads P, dP, g_dS, gPdV (4 N×N buffers).
// Pass 1: row reductions for D_i, dot2, A, E → dot3
// Pass 2: overwrites dP→dS_scaled, g_dS→gS_scaled, gPdV→g_dP
//
// Eliminates ALL temporary N×N tensor allocations from elementwise phase.
// One block per (b,h,i) row, 256 threads.
// ===========================================================================
#define FUSED_BLOCK 256

__global__ void fused_dbl_bwd_elementwise(
    const float* __restrict__ P,     // [total_rows, N]
    float* __restrict__ dP,          // overwritten → dS_scaled
    float* __restrict__ g_dS,        // overwritten → gS_scaled
    float* __restrict__ gPdV,        // overwritten → g_dP
    int N, float scale
) {
    extern __shared__ float sdata[];
    // 4 arrays of blockDim.x for parallel reduction
    float* s_Di   = sdata;
    float* s_dot2 = s_Di   + blockDim.x;
    float* s_A    = s_dot2 + blockDim.x;
    float* s_E    = s_A    + blockDim.x;

    int row = blockIdx.x;
    const float* P_row    = P    + (long long)row * N;
    float* dP_row         = dP   + (long long)row * N;
    float* gds_row        = g_dS + (long long)row * N;
    float* gpdv_row       = gPdV + (long long)row * N;

    // Pass 1: partial sums for D_i, dot2, A, E
    float my_Di = 0.0f, my_dot2 = 0.0f, my_A = 0.0f, my_E = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float p    = P_row[j];
        float dp   = dP_row[j];
        float gds  = gds_row[j];
        float gpdv = gpdv_row[j];

        float p_gds = p * gds;
        my_Di   += p * dp;
        my_dot2 += p_gds;
        my_A    += p_gds * dp;
        my_E    += p * gpdv;
    }

    s_Di[threadIdx.x]   = my_Di;
    s_dot2[threadIdx.x] = my_dot2;
    s_A[threadIdx.x]    = my_A;
    s_E[threadIdx.x]    = my_E;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_Di[threadIdx.x]   += s_Di[threadIdx.x + s];
            s_dot2[threadIdx.x] += s_dot2[threadIdx.x + s];
            s_A[threadIdx.x]    += s_A[threadIdx.x + s];
            s_E[threadIdx.x]    += s_E[threadIdx.x + s];
        }
        __syncthreads();
    }

    float Di   = s_Di[0];
    float dot2 = s_dot2[0];
    float dot3 = s_A[0] - 2.0f * Di * dot2 + s_E[0];

    // Pass 2: elementwise transforms (re-read from L2 cache)
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float p    = P_row[j];
        float dp   = dP_row[j];
        float gds  = gds_row[j];
        float gpdv = gpdv_row[j];

        float ds      = p * (dp - Di);
        float g_dp    = p * (gds - dot2);
        float g_P_val = gds * (dp - Di) - dp * dot2 + gpdv;
        float g_S     = p * (g_P_val - dot3);

        dP_row[j]   = scale * ds;     // dS_scaled
        gds_row[j]  = scale * g_S;    // gS_scaled
        gpdv_row[j] = g_dp;           // g_dP
    }
}

// ===========================================================================
// Fused backward elementwise: dot_i = (P*dP).sum(-1), dS = P*(dP - dot_i)
// Overwrites dP with dS in place. One block per (b,h,i) row.
// ===========================================================================
__global__ void fused_bwd_elementwise(
    const float* __restrict__ P,   // [total_rows, N]
    float* __restrict__ dP,        // [total_rows, N] → overwritten with dS
    int N
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* P_row = P + (size_t)row * N;
    float* dP_row = dP + (size_t)row * N;

    // Pass 1: partial sum for dot = sum(P * dP)
    float partial = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        partial += P_row[j] * dP_row[j];

    sdata[threadIdx.x] = partial;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float dot = sdata[0];

    // Pass 2: dS = P * (dP - dot)
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        dP_row[j] = P_row[j] * (dP_row[j] - dot);
}

// ===========================================================================
// Component A: cuBLAS ATen Double Backward (with fused elementwise kernel)
//
// 5 matmuls → fused elementwise kernel → 7 matmuls
// Only 4 N×N buffers allocated. Zero temporary N×N tensors.
// ===========================================================================
static std::vector<at::Tensor> cublas_double_backward_v5_cuda(
    at::Tensor g_dQ,
    at::Tensor g_dK,
    at::Tensor g_dV,
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L,
    at::Tensor P_saved
) {
    at::AutoGradMode no_grad(false);
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    auto K_T = K.transpose(-2, -1);

    bool have_P = P_saved.numel() > 0;

    // N×N scratch buffers + 4 output buffers
    // If P is saved, we use it directly and only need 3 scratch buffers
    auto buf_a = have_P ? P_saved : at::empty({B, H, N, N}, opts);  // P
    auto buf_b = at::empty({B, H, N, N}, opts);  // dP → dS_scaled
    auto buf_c = at::empty({B, H, N, N}, opts);  // g_dS → gS_scaled
    auto buf_d = at::empty({B, H, N, N}, opts);  // gPdV → g_dP
    auto g_Q  = at::empty({B, H, N, D}, opts);
    auto g_dO = at::empty({B, H, N, D}, opts);
    auto g_K  = at::empty({B, H, N, D}, opts);
    auto g_V  = at::empty({B, H, N, D}, opts);

    // ---- Phase 1: matmuls to fill N×N buffers ----

    // buf_a = P (recompute only if not saved)
    if (!have_P) {
        at::matmul_out(buf_a, Q, K_T);
        buf_a.mul_(scale).sub_(L.unsqueeze(-1)).exp_();
    }

    // buf_b = dP = dO @ V^T
    at::matmul_out(buf_b, dO, V.transpose(-2, -1));

    // buf_c = g_dS = scale * (g_dQ @ K^T + Q @ g_dK^T)
    at::matmul_out(buf_c, g_dQ, K_T);
    at::matmul_out(buf_d, Q, g_dK.transpose(-2, -1));  // use buf_d as temp
    buf_c.add_(buf_d).mul_(scale);

    // buf_d = gPdV = dO @ g_dV^T
    at::matmul_out(buf_d, dO, g_dV.transpose(-2, -1));

    // ---- Phase 2: fused elementwise kernel ----
    // Row reductions (D_i, dot2, dot3) + transforms:
    //   buf_b: dP → dS_scaled
    //   buf_c: g_dS → gS_scaled
    //   buf_d: gPdV → g_dP
    //   buf_a: P (unchanged, read-only)
    {
        int total_rows = B * H * N;
        int smem = 4 * FUSED_BLOCK * sizeof(float);
        fused_dbl_bwd_elementwise<<<total_rows, FUSED_BLOCK, smem>>>(
            buf_a.data_ptr<float>(),
            buf_b.data_ptr<float>(),
            buf_c.data_ptr<float>(),
            buf_d.data_ptr<float>(),
            N, scale);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Phase 3: 7 output matmuls ----
    // After fused kernel: buf_a=P, buf_b=dS_scaled, buf_c=gS_scaled, buf_d=g_dP

    // g_Q = dS_scaled @ g_dK + gS_scaled @ K
    at::matmul_out(g_Q, buf_b, g_dK);
    g_Q.add_(at::matmul(buf_c, K));

    // g_dO = P @ g_dV + g_dP @ V
    at::matmul_out(g_dO, buf_a, g_dV);
    g_dO.add_(at::matmul(buf_d, V));

    // g_K = dS_scaled^T @ g_dQ + gS_scaled^T @ Q
    at::matmul_out(g_K, buf_b.transpose(-2, -1), g_dQ);
    g_K.add_(at::matmul(buf_c.transpose(-2, -1), Q));

    // g_V = g_dP^T @ dO
    at::matmul_out(g_V, buf_d.transpose(-2, -1), dO);

    return {g_dO, g_Q, g_K, g_V};
}

// ===========================================================================
// C++ backward for V5 (eliminates Python dispatch overhead)
// Called from FlashAttentionV5Backward.forward in Python.
// ===========================================================================
std::vector<at::Tensor> cublas_backward_v5_cuda(
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L
) {
    at::AutoGradMode no_grad(false);
    int D = Q.size(3);
    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Recompute S, P from saved L
    auto S = at::matmul(Q, K.transpose(-2, -1)) * scale;
    auto P = at::exp(S - L.unsqueeze(-1));

    // Backward GEMMs
    auto dV = at::matmul(P.transpose(-2, -1), dO);
    auto dP = at::matmul(dO, V.transpose(-2, -1));

    auto dot = (P * dP).sum(/*dim=*/-1, /*keepdim=*/true);
    auto dS = P * (dP - dot);

    auto dQ = at::matmul(dS, K) * scale;
    auto dK = at::matmul(dS.transpose(-2, -1), Q) * scale;

    return {dQ, dK, dV};
}

// ===========================================================================
// Component B: BC=32 WMMA Flash Double Backward Kernels
// ===========================================================================

__device__ __forceinline__ float warp_reduce_sum_v5(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// ===========================================================================
// Score matmul for 16×32 wide tiles: A[16][D] @ B[32][D]^T -> acc[16][32]
//
// Produces two 16×16 WMMA fragments side by side (left and right halves).
// Shares a_frag load across both halves.
// ===========================================================================
__device__ __forceinline__ void wmma_score_matmul_wide(
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
        // Left half: first 16 cols of B
        wmma::load_matrix_sync(b_frag_left, B_smem + kk, B_stride);
        // Right half: cols 16..31 of B
        wmma::load_matrix_sync(b_frag_right, B_smem + WMMA_N * B_stride + kk, B_stride);
        wmma::mma_sync(acc_left, a_frag, b_frag_left, acc_left);
        wmma::mma_sync(acc_right, a_frag, b_frag_right, acc_right);
    }
}

// Accumulate version (doesn't zero acc first)
__device__ __forceinline__ void wmma_score_matmul_wide_acc(
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

// ===========================================================================
// Helper: store wide 16×32 score to shared memory
// Stores left (16×16) and right (16×16) fragments into a 16×(32+pad) tile
// ===========================================================================
__device__ __forceinline__ void store_wide_score(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& left,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& right
) {
    // Store left half at columns 0..15
    wmma::store_matrix_sync(tile, left, score_stride, wmma::mem_row_major);
    // Store right half at columns 16..31
    wmma::store_matrix_sync(tile + WMMA_N, right, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Output matmul for BC=32: score[16][32] @ data[32][D] -> out[16][D]
//
// k_steps = 32/8 = 4 (was 2 in V3)
// ===========================================================================
__device__ __forceinline__ void wmma_output_matmul_acc_v5(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    int k_steps = V5_BC / WMMA_K;  // 32/8 = 4

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
// Kernel B score matmul: tall 32×16 score tile
// A[32][D] @ B[16][D]^T -> score[32][16]
// Produces two stacked 16×16 fragments (top and bottom halves)
// ===========================================================================
__device__ __forceinline__ void wmma_score_matmul_tall(
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
        // Top 16 rows
        wmma::load_matrix_sync(a_frag_top, A_smem + kk, A_stride);
        // Bottom 16 rows
        wmma::load_matrix_sync(a_frag_bot, A_smem + WMMA_M * A_stride + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc_top, a_frag_top, b_frag, acc_top);
        wmma::mma_sync(acc_bot, a_frag_bot, b_frag, acc_bot);
    }
}

// Accumulate version
__device__ __forceinline__ void wmma_score_matmul_tall_acc(
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

// Store tall 32×16 score to shared memory
__device__ __forceinline__ void store_tall_score(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& top,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& bot
) {
    wmma::store_matrix_sync(tile, top, score_stride, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + WMMA_M * score_stride, bot, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Kernel B output matmul (transposed tall score):
// score[32][16]^T @ data[32][D] -> out[16][D]
//
// score^T is [16][32]. k_steps = 32/8 = 4.
// We load score in col-major to transpose: top half rows 0..15 give k=0..15,
// bottom half rows 16..31 give k=16..31.
// ===========================================================================
__device__ __forceinline__ void wmma_output_matmul_acc_tallT(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    const float* data_smem, int data_stride,
    int D_dim
) {
    int n_col_blocks = D_dim / WMMA_N;
    // score is 32×16 row-major. score^T is 16×32.
    // k ranges over 32 columns of score^T = 32 rows of score.
    // k_steps = 32/8 = 4
    int k_steps = V5_BC / WMMA_K;  // 4

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
            // Col-major load: a_frag[j][k] = score[(kk*8+k)*score_stride + j]
            // This gives score^T with rows j, cols kk*8..kk*8+7
            wmma::load_matrix_sync(a_frag, score_smem + kk * WMMA_K * score_stride, score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * WMMA_K * data_stride + cb * WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

__device__ __forceinline__ void scale_fragment_v5(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag,
    float s
) {
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] *= s;
}

// ===========================================================================
// V5 Kernel A (row kernel): 2 passes, BC=32
//
// Shared memory layout (SCORE_STRIDE = V5_BC + V5_SCORE_PAD = 34):
//   Q_block    [BR][D_PAD]           3 * 64 * D
//   dO_block   [BR][D_PAD]
//   g_dQ_block [BR][D_PAD]
//   K_tile     [BC][D_PAD]           4 * 32 * D  (doubled from V3)
//   V_tile     [BC][D_PAD]
//   g_dK_tile  [BC][D_PAD]
//   g_dV_tile  [BC][D_PAD]
//   score_base [4*5][16][34]         per-warp: S(L+R), gdS(L+R), dP(L+R), gPdV(L+R), extra(L+R)
//   L_vec      [BR]
//   Di_vec     [BR]
//   dot2_smem  [BR]
//   dot3_smem  [BR]
// ===========================================================================
__global__ void v5_kernel_A_row(
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
    const int D_PAD = D + V5_SMEM_PAD;
    const int SCORE_STRIDE = V5_BC + V5_SCORE_PAD;  // 34

    float* Q_block    = smem;
    float* dO_block   = Q_block    + V5_BR * D_PAD;
    float* g_dQ_block = dO_block   + V5_BR * D_PAD;
    float* K_tile     = g_dQ_block + V5_BR * D_PAD;
    float* V_tile     = K_tile     + V5_BC * D_PAD;
    float* g_dK_tile  = V_tile     + V5_BC * D_PAD;
    float* g_dV_tile  = g_dK_tile  + V5_BC * D_PAD;
    float* score_base = g_dV_tile  + V5_BC * D_PAD;
    float* L_vec      = score_base + V5_WARPS_PER_BLOCK * 5 * WMMA_M * SCORE_STRIDE;
    float* Di_vec     = L_vec      + V5_BR;
    float* dot2_smem  = Di_vec     + V5_BR;
    float* dot3_smem  = dot2_smem  + V5_BR;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_row_start = blockIdx.x * V5_BR;
    int BH = B * H;
    int bh_block = block_row_start / N;

    // Per-warp score tiles (5 tiles of 16×SCORE_STRIDE each)
    float* S_w     = score_base + warp_id * 5 * WMMA_M * SCORE_STRIDE;
    float* gdS_w   = S_w     + WMMA_M * SCORE_STRIDE;
    float* dP_w    = gdS_w   + WMMA_M * SCORE_STRIDE;
    float* gPdV_w  = dP_w    + WMMA_M * SCORE_STRIDE;
    float* extra_w = gPdV_w  + WMMA_M * SCORE_STRIDE;

    float* Q_warp    = Q_block    + warp_id * WMMA_M * D_PAD;
    float* dO_warp   = dO_block   + warp_id * WMMA_M * D_PAD;
    float* g_dQ_warp = g_dQ_block + warp_id * WMMA_M * D_PAD;

    int warp_row_start = block_row_start + warp_id * WMMA_M;
    int total_rows = BH * N;

    // ---- Load row data into shared memory ----
    int rows_to_load = min(V5_BR, total_rows - block_row_start);
    if (rows_to_load < 0) rows_to_load = 0;

    for (int idx = threadIdx.x; idx < V5_BR * D_PAD; idx += V5_BLOCK_THREADS) {
        Q_block[idx] = 0.0f;
        dO_block[idx] = 0.0f;
        g_dQ_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V5_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        int src = global_row * D + d;
        Q_block[r * D_PAD + d]    = Q[src];
        dO_block[r * D_PAD + d]   = dO[src];
        g_dQ_block[r * D_PAD + d] = g_dQ[src];
    }

    for (int idx = threadIdx.x; idx < V5_BR; idx += V5_BLOCK_THREADS) {
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
        partial = warp_reduce_sum_v5(partial);
        if (lane == 0) {
            Di_vec[warp_id * WMMA_M + r] = partial;
            D_out[global_row] = partial;
        }
    }
    __syncthreads();

    // ====================================================================
    // Pass 1: dot2, A, E -> dot3 using WMMA wide score matmuls (16×32)
    // ====================================================================
    float dot2_acc[WMMA_M];
    float A_acc[WMMA_M];
    float E_acc[WMMA_M];
    for (int r = 0; r < WMMA_M; r++) {
        dot2_acc[r] = 0.0f;
        A_acc[r] = 0.0f;
        E_acc[r] = 0.0f;
    }

    for (int j_start = 0; j_start < N; j_start += V5_BC) {
        int tile_cols = min(V5_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V5_BC * D_PAD; idx += V5_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V5_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // WMMA wide score matmuls (16×32 each)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v5(S_L, scale);
        scale_fragment_v5(S_R, scale);
        store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_L, gdS_R;
        wmma_score_matmul_wide(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_wide_acc(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment_v5(gdS_L, scale);
        scale_fragment_v5(gdS_R, scale);
        store_wide_score(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_L, dP_R;
        wmma_score_matmul_wide(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD, D);
        store_wide_score(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_L, gPdV_R;
        wmma_score_matmul_wide(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        store_wide_score(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);

        __syncwarp();

        // Element-wise: per-row reductions for dot2, A, E
        // 16 rows × 32 cols. 32 lanes = 1 col/lane (perfect mapping)
        for (int r = 0; r < WMMA_M; r++) {
            float p_dot2 = 0.0f, p_A = 0.0f, p_E = 0.0f;
            // With BC=32 and WARP_SIZE=32, each lane handles exactly 1 column
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;
                float s_val = S_w[si];
                float l_val = L_vec[warp_id * WMMA_M + r];
                float p_val = __expf(s_val - l_val);
                float gds_val = gdS_w[si];
                float dp_val = dP_w[si];
                float gpdv_val = gPdV_w[si];

                float pg = p_val * gds_val;
                p_dot2 = pg;
                p_A    = pg * dp_val;
                p_E    = p_val * gpdv_val;
            }
            p_dot2 = warp_reduce_sum_v5(p_dot2);
            p_A    = warp_reduce_sum_v5(p_A);
            p_E    = warp_reduce_sum_v5(p_E);
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

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_Q_frags[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> g_dO_frags[4];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_Q_frags[cb], 0.0f);
        wmma::fill_fragment(g_dO_frags[cb], 0.0f);
    }

    for (int j_start = 0; j_start < N; j_start += V5_BC) {
        int tile_cols = min(V5_BC, N - j_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V5_BC * D_PAD; idx += V5_BLOCK_THREADS) {
            K_tile[idx] = 0.0f;
            V_tile[idx] = 0.0f;
            g_dK_tile[idx] = 0.0f;
            g_dV_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_cols * D; idx += V5_BLOCK_THREADS) {
            int c = idx / D;
            int d = idx % D;
            int src = (bh_block * N + j_start + c) * D + d;
            K_tile[c * D_PAD + d]    = K[src];
            V_tile[c * D_PAD + d]    = V[src];
            g_dK_tile[c * D_PAD + d] = g_dK[src];
            g_dV_tile[c * D_PAD + d] = g_dV[src];
        }
        __syncthreads();

        // Recompute wide scores
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_L, S_R;
        wmma_score_matmul_wide(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD, D);
        scale_fragment_v5(S_L, scale);
        scale_fragment_v5(S_R, scale);
        store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_L, gdS_R;
        wmma_score_matmul_wide(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD, D);
        wmma_score_matmul_wide_acc(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD, D);
        scale_fragment_v5(gdS_L, scale);
        scale_fragment_v5(gdS_R, scale);
        store_wide_score(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_L, dP_R;
        wmma_score_matmul_wide(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD, D);
        store_wide_score(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_L, gPdV_R;
        wmma_score_matmul_wide(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD, D);
        store_wide_score(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);

        __syncwarp();

        // Element-wise: compute dS, g_dP, g_S, P
        // BC=32 with WARP_SIZE=32: 1 col per lane per row
        for (int r = 0; r < WMMA_M; r++) {
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;

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
                int si = r * SCORE_STRIDE + lane;
                if (lane < V5_BC) {
                    dP_w[si]    = 0.0f;
                    extra_w[si] = 0.0f;
                    S_w[si]     = 0.0f;
                    gPdV_w[si]  = 0.0f;
                }
            }
        }
        __syncwarp();

        // Output matmuls with BC=32 score tiles:
        // g_Q  += (scale*dS) @ g_dK   +  (scale*g_S) @ K
        // g_dO += P @ g_dV             +  g_dP @ V
        wmma_output_matmul_acc_v5(g_Q_frags, dP_w, SCORE_STRIDE, g_dK_tile, D_PAD, D);
        wmma_output_matmul_acc_v5(g_Q_frags, extra_w, SCORE_STRIDE, K_tile, D_PAD, D);
        wmma_output_matmul_acc_v5(g_dO_frags, S_w, SCORE_STRIDE, g_dV_tile, D_PAD, D);
        wmma_output_matmul_acc_v5(g_dO_frags, gPdV_w, SCORE_STRIDE, V_tile, D_PAD, D);
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

    for (int idx = threadIdx.x; idx < rows_to_load * D; idx += V5_BLOCK_THREADS) {
        int r = idx / D;
        int d = idx % D;
        int global_row = block_row_start + r;
        g_Q_out[global_row * D + d]  = Q_block[r * D_PAD + d];
        g_dO_out[global_row * D + d] = dO_block[r * D_PAD + d];
    }
}

// ===========================================================================
// V5 Kernel B (column kernel): 1 pass, BC=32
//
// Block: 64 columns (4 warps × 16 cols each)
// Each warp processes 16 columns via WMMA against BC=32 row tiles
//
// Score tiles are 32×16 (tall): 2 stacked 16×16 WMMA fragments
// Score stride = 16 + V5_SCORE_PAD = 18 (even)
//
// Shared memory layout:
//   K_block, V_block, g_dK_block, g_dV_block: [BR][D_PAD]
//   Q_tile, dO_tile, g_dQ_tile:               [BC][D_PAD]  (BC=32 now)
//   score_base: [4*5][32][18]                              per-warp tall scores
//   tile_L, tile_D, tile_dot2, tile_dot3:     [BC]
// ===========================================================================
__global__ void v5_kernel_B_col(
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
    const int D_PAD = D + V5_SMEM_PAD;
    // Kernel B score tiles are 32×16 (tall), stride = 16 + pad = 18
    const int B_SCORE_STRIDE = WMMA_N + V5_SCORE_PAD;  // 18

    float* K_block    = smem;
    float* V_block    = K_block    + V5_BR * D_PAD;
    float* g_dK_block = V_block    + V5_BR * D_PAD;
    float* g_dV_block = g_dK_block + V5_BR * D_PAD;
    float* Q_tile     = g_dV_block + V5_BR * D_PAD;
    float* dO_tile    = Q_tile     + V5_BC * D_PAD;
    float* g_dQ_tile  = dO_tile    + V5_BC * D_PAD;
    float* score_base = g_dQ_tile  + V5_BC * D_PAD;
    // 5 tall score tiles per warp, each 32×18 = 32*B_SCORE_STRIDE
    float* tile_L     = score_base + V5_WARPS_PER_BLOCK * 5 * V5_BC * B_SCORE_STRIDE;
    float* tile_D     = tile_L     + V5_BC;
    float* tile_dot2  = tile_D     + V5_BC;
    float* tile_dot3  = tile_dot2  + V5_BC;

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int block_col_start = blockIdx.x * V5_BR;
    int BH = B * H;
    int bh_block = block_col_start / N;
    int total_cols = BH * N;

    // Per-warp tall score tiles (5 tiles of 32×B_SCORE_STRIDE each)
    float* S_w     = score_base + warp_id * 5 * V5_BC * B_SCORE_STRIDE;
    float* gdS_w   = S_w     + V5_BC * B_SCORE_STRIDE;
    float* dP_w    = gdS_w   + V5_BC * B_SCORE_STRIDE;
    float* gPdV_w  = dP_w    + V5_BC * B_SCORE_STRIDE;
    float* extra_w = gPdV_w  + V5_BC * B_SCORE_STRIDE;

    float* K_warp    = K_block    + warp_id * WMMA_M * D_PAD;
    float* V_warp    = V_block    + warp_id * WMMA_M * D_PAD;
    float* g_dK_warp = g_dK_block + warp_id * WMMA_M * D_PAD;
    float* g_dV_warp = g_dV_block + warp_id * WMMA_M * D_PAD;

    // ---- Load column data ----
    int cols_to_load = min(V5_BR, total_cols - block_col_start);
    if (cols_to_load < 0) cols_to_load = 0;

    for (int idx = threadIdx.x; idx < V5_BR * D_PAD; idx += V5_BLOCK_THREADS) {
        K_block[idx] = 0.0f;
        V_block[idx] = 0.0f;
        g_dK_block[idx] = 0.0f;
        g_dV_block[idx] = 0.0f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V5_BLOCK_THREADS) {
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

    // ---- Single pass over row tiles (BC=32 rows per tile) ----
    for (int i_start = 0; i_start < N; i_start += V5_BC) {
        int tile_rows = min(V5_BC, N - i_start);

        __syncthreads();
        for (int idx = threadIdx.x; idx < V5_BC * D_PAD; idx += V5_BLOCK_THREADS) {
            Q_tile[idx] = 0.0f;
            dO_tile[idx] = 0.0f;
            g_dQ_tile[idx] = 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile_rows * D; idx += V5_BLOCK_THREADS) {
            int r = idx / D;
            int d = idx % D;
            int src = (bh_block * N + i_start + r) * D + d;
            Q_tile[r * D_PAD + d]    = Q[src];
            dO_tile[r * D_PAD + d]   = dO[src];
            g_dQ_tile[r * D_PAD + d] = g_dQ[src];
        }
        for (int idx = threadIdx.x; idx < V5_BC; idx += V5_BLOCK_THREADS) {
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

        // Tall score matmuls: S = Q_tile[32×D] @ K_warp[16×D]^T -> 32×16
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> S_top, S_bot;
        wmma_score_matmul_tall(S_top, S_bot, Q_tile, D_PAD, K_warp, D_PAD, D);
        scale_fragment_v5(S_top, scale);
        scale_fragment_v5(S_bot, scale);
        store_tall_score(S_w, B_SCORE_STRIDE, S_top, S_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gdS_top, gdS_bot;
        wmma_score_matmul_tall(gdS_top, gdS_bot, g_dQ_tile, D_PAD, K_warp, D_PAD, D);
        wmma_score_matmul_tall_acc(gdS_top, gdS_bot, Q_tile, D_PAD, g_dK_warp, D_PAD, D);
        scale_fragment_v5(gdS_top, scale);
        scale_fragment_v5(gdS_bot, scale);
        store_tall_score(gdS_w, B_SCORE_STRIDE, gdS_top, gdS_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> dP_top, dP_bot;
        wmma_score_matmul_tall(dP_top, dP_bot, dO_tile, D_PAD, V_warp, D_PAD, D);
        store_tall_score(dP_w, B_SCORE_STRIDE, dP_top, dP_bot);

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> gPdV_top, gPdV_bot;
        wmma_score_matmul_tall(gPdV_top, gPdV_bot, dO_tile, D_PAD, g_dV_warp, D_PAD, D);
        store_tall_score(gPdV_w, B_SCORE_STRIDE, gPdV_top, gPdV_bot);

        __syncwarp();

        // Element-wise: 32 rows × 16 cols
        // With WARP_SIZE=32 and 32*16=512 elements, each lane handles 16 elements
        for (int idx = lane; idx < V5_BC * WMMA_N; idx += WARP_SIZE) {
            int i_local = idx / WMMA_N;
            int j_local = idx % WMMA_N;
            int si = i_local * B_SCORE_STRIDE + j_local;

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

            dP_w[si]    = scale * ds_val;   // scale * dS
            extra_w[si] = scale * g_S_val;  // scale * g_S
            gPdV_w[si]  = g_dp_val;         // g_dP
        }
        __syncwarp();

        // Output matmuls (transposed tall scores):
        // g_K += (scale*dS)^T @ g_dQ_tile  +  (scale*g_S)^T @ Q_tile
        // g_V += g_dP^T @ dO_tile
        wmma_output_matmul_acc_tallT(g_K_frags, dP_w, B_SCORE_STRIDE, g_dQ_tile, D_PAD, D);
        wmma_output_matmul_acc_tallT(g_K_frags, extra_w, B_SCORE_STRIDE, Q_tile, D_PAD, D);
        wmma_output_matmul_acc_tallT(g_V_frags, gPdV_w, B_SCORE_STRIDE, dO_tile, D_PAD, D);
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

    for (int idx = threadIdx.x; idx < cols_to_load * D; idx += V5_BLOCK_THREADS) {
        int c = idx / D;
        int d = idx % D;
        int global_col = block_col_start + c;
        g_K_out[global_col * D + d] = K_block[c * D_PAD + d];
        g_V_out[global_col * D + d] = V_block[c * D_PAD + d];
    }
}

// ===========================================================================
// BC=32 WMMA flash double backward host wrapper
// ===========================================================================
static std::vector<at::Tensor> flash_double_backward_v5_wmma(
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
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();
    int BH = B * H;
    int total_rows = BH * N;
    int D_PAD = D + V5_SMEM_PAD;
    int A_SCORE_STRIDE = V5_BC + V5_SCORE_PAD;  // 34
    int B_SCORE_STRIDE = WMMA_N + V5_SCORE_PAD;  // 18

    int grid = (total_rows + V5_BR - 1) / V5_BR;

    // Kernel A smem: 3*BR*D_PAD + 4*BC*D_PAD + 4warps*5*16*34 + 4*BR
    int smem_A = (3 * V5_BR * D_PAD + 4 * V5_BC * D_PAD
                  + V5_WARPS_PER_BLOCK * 5 * WMMA_M * A_SCORE_STRIDE
                  + 4 * V5_BR) * sizeof(float);

    // Kernel B smem: 4*BR*D_PAD + 3*BC*D_PAD + 4warps*5*32*18 + 4*BC
    int smem_B = (4 * V5_BR * D_PAD + 3 * V5_BC * D_PAD
                  + V5_WARPS_PER_BLOCK * 5 * V5_BC * B_SCORE_STRIDE
                  + 4 * V5_BC) * sizeof(float);

    if (smem_A > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v5_kernel_A_row,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_A));
    }
    if (smem_B > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(v5_kernel_B_col,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_B));
    }

    auto D_vec    = at::empty({B, H, N}, opts);
    auto dot2     = at::empty({B, H, N}, opts);
    auto dot3     = at::empty({B, H, N}, opts);
    auto g_Q      = at::zeros({B, H, N, D}, opts);
    auto g_dO_out = at::zeros({B, H, N, D}, opts);
    auto g_K      = at::zeros({B, H, N, D}, opts);
    auto g_V      = at::zeros({B, H, N, D}, opts);

    v5_kernel_A_row<<<grid, V5_BLOCK_THREADS, smem_A>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        dO.data_ptr<float>(), O.data_ptr<float>(),
        g_dQ.data_ptr<float>(), g_dK.data_ptr<float>(), g_dV.data_ptr<float>(),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        g_Q.data_ptr<float>(), g_dO_out.data_ptr<float>(),
        B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    v5_kernel_B_col<<<grid, V5_BLOCK_THREADS, smem_B>>>(
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
// Forward declarations of fallbacks
// ===========================================================================
extern std::vector<at::Tensor> flash_double_backward_cuda(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L);

extern std::vector<at::Tensor> flash_double_backward_v3_cuda(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L);

extern std::vector<at::Tensor> flash_forward_v3_cuda(
    at::Tensor Q, at::Tensor K, at::Tensor V);

extern std::vector<at::Tensor> flash_backward_v3_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L);

// ===========================================================================
// Adaptive forward: V3 flash (small N) or cuBLAS (large N)
// Single C++ call eliminates Python dispatch overhead.
// ===========================================================================
std::vector<at::Tensor> v5_forward_cuda(
    at::Tensor Q, at::Tensor K, at::Tensor V
) {
    int N = Q.size(2);
    int D = Q.size(3);

    if (N >= 16 && D % 16 == 0 && N <= 512) {
        auto result = flash_forward_v3_cuda(Q, K, V);  // {O, L}
        result.push_back(at::empty({0}, Q.options()));  // empty P placeholder
        return result;
    }

    // cuBLAS forward — also returns P for backward to avoid recomputation
    at::AutoGradMode no_grad(false);
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto S = at::matmul(Q, K.transpose(-2, -1)) * scale;
    auto L = at::logsumexp(S, -1);
    auto P = at::exp(S - L.unsqueeze(-1));
    auto O = at::matmul(P, V);
    return {O, L, P};
}

// ===========================================================================
// Adaptive backward: V3 flash (small N) or cuBLAS (large N)
// ===========================================================================
std::vector<at::Tensor> v5_backward_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K,
    at::Tensor V, at::Tensor O, at::Tensor L, at::Tensor P
) {
    int N = Q.size(2);
    int D = Q.size(3);

    if (N >= 16 && D % 16 == 0 && N <= 512) {
        return flash_backward_v3_cuda(dO, Q, K, V, O, L);
    }

    // cuBLAS backward with saved P: matmul_out + fused elementwise
    at::AutoGradMode no_grad(false);
    int B = Q.size(0), H = Q.size(1);
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    auto dV = at::empty({B, H, N, D}, opts);
    auto dP = at::empty({B, H, N, N}, opts);  // reused as dS after fused kernel
    auto dQ = at::empty({B, H, N, D}, opts);
    auto dK = at::empty({B, H, N, D}, opts);

    at::matmul_out(dV, P.transpose(-2, -1), dO);
    at::matmul_out(dP, dO, V.transpose(-2, -1));

    // Fused: dot = (P*dP).sum(-1), dP → dS = P*(dP - dot)
    {
        int total_rows = B * H * N;
        int smem = FUSED_BLOCK * sizeof(float);
        fused_bwd_elementwise<<<total_rows, FUSED_BLOCK, smem>>>(
            P.data_ptr<float>(), dP.data_ptr<float>(), N);
        CUDA_CHECK(cudaGetLastError());
    }

    // dP is now dS
    at::matmul_out(dQ, dP, K);
    dQ.mul_(scale);
    at::matmul_out(dK, dP.transpose(-2, -1), Q);
    dK.mul_(scale);

    return {dQ, dK, dV};
}

// ===========================================================================
// Component C: Adaptive Double Backward Dispatch
//   N <= 512 + WMMA-compatible: V3 WMMA flash (O(N) memory)
//   N > 512:                    cuBLAS fused (fastest at large N)
//   N > 2048 + WMMA-compatible: BC=32 WMMA flash (O(N) memory, mem-bound)
// ===========================================================================
std::vector<at::Tensor> flash_double_backward_v5_cuda(
    at::Tensor g_dQ,
    at::Tensor g_dK,
    at::Tensor g_dV,
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L,
    at::Tensor P_saved
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

    int N = Q.size(2);
    int D = Q.size(3);

    // Small N: V3 WMMA for very small N only (cuBLAS is faster for N >= 384)
    if (N <= 256 && N >= 16 && D % 16 == 0) {
        return flash_double_backward_v3_cuda(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L);
    }

    // Large N: cuBLAS fused (12 matmuls + 1 fused kernel, best throughput)
    if (N <= 2048 || D % 16 != 0) {
        return cublas_double_backward_v5_cuda(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L, P_saved);
    }

    // Very large N: BC=32 WMMA flash (O(N) memory, avoids N×N traffic)
    return flash_double_backward_v5_wmma(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L);
}
