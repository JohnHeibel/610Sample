#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <vector>
#include <cmath>

using namespace nvcuda;

#define V7_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t err = call;                                                \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                        \
                    cudaGetErrorString(err));                                   \
    } while (0)

#define V7_WARP_SIZE 32
#define V7_FULL_MASK 0xFFFFFFFF

// fp16 WMMA tile dimensions (2x throughput vs TF32 m16n16k8)
#define V7_WMMA_M 16
#define V7_WMMA_N 16
#define V7_WMMA_K 16

// Block parameters
#define V7_WARPS_PER_BLOCK 4
#define V7_BLOCK_THREADS (V7_WARP_SIZE * V7_WARPS_PER_BLOCK)  // 128
#define V7_BR 64   // rows per block = 4 warps * 16
#define V7_BC 32   // column tile width = 2 * WMMA_N

// Padding for shared memory
#define V7_SMEM_PAD 0
#define V7_SCORE_PAD 0    // BC=32, WMMA_N=16 already even for sm_120 store_matrix_sync
#define V7_HSCORE_PAD 8   // Half score staging stride must be multiple of 8 for WMMA load

// ===========================================================================
// Device helpers
// ===========================================================================
__device__ __forceinline__ float v7_warp_reduce_sum(float val) {
    for (int offset = V7_WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(V7_FULL_MASK, val, offset);
    return val;
}

__device__ __forceinline__ float v7_warp_reduce_max(float val) {
    for (int offset = V7_WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(V7_FULL_MASK, val, offset));
    return val;
}

// Scale an fp32 accumulator fragment
__device__ __forceinline__ void v7_scale_accum(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& frag,
    float s
) {
    for (int i = 0; i < frag.num_elements; i++)
        frag.x[i] *= s;
}

// ===========================================================================
// Masking helpers
// ===========================================================================

// Determine mask status for a tile: 0=all masked, 1=partial, 2=all visible
__device__ __forceinline__ int v7_tile_mask_status(
    int q_start, int q_end,   // global query row range [q_start, q_end)
    int k_start, int k_end,   // global key col range
    int window_size, int chunk_offset
) {
    // ki values: chunk_offset - N_kv + k_start ... chunk_offset - N_kv + k_end - 1
    // But we pass absolute key indices
    // Check validity: ki >= 0
    // Check causal: qi >= ki
    // Check window: qi < ki + window_size

    // All masked if: all ki < 0, or all qi < ki, or all qi >= ki + ws
    // All visible if: all ki >= 0, and all qi >= ki, and all qi < ki + ws

    int ki_min = k_start;
    int ki_max = k_end - 1;
    int qi_min = q_start;
    int qi_max = q_end - 1;

    // If all keys invalid (ki < 0)
    if (ki_max < 0) return 0;
    // If all queries before all keys (qi < ki) — causal mask eliminates all
    if (qi_max < ki_min) return 0;
    // If all queries past window for all keys (qi >= ki + ws)
    if (qi_min >= ki_max + window_size) return 0;

    // Check if all visible: all ki >= 0 AND qi_min >= ki_max (causal) AND qi_max < ki_min + ws
    if (ki_min >= 0 && qi_min >= ki_max && qi_max < ki_min + window_size)
        return 2;

    return 1;  // partial
}

// Per-element mask check
__device__ __forceinline__ bool v7_is_visible(int qi, int ki, int window_size) {
    return (ki >= 0) && (qi >= ki) && (qi < ki + window_size);
}

// ===========================================================================
// Type traits for half_t dispatch
// ===========================================================================
template<typename half_t> struct HalfTraits;

template<> struct HalfTraits<__half> {
    static __device__ __forceinline__ __half from_float(float f) { return __float2half(f); }
    static __device__ __forceinline__ float to_float(__half h) { return __half2float(h); }
    static constexpr auto torch_dtype = at::kHalf;
};

template<> struct HalfTraits<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 from_float(float f) { return __float2bfloat16(f); }
    static __device__ __forceinline__ float to_float(__nv_bfloat16 h) { return __bfloat162float(h); }
    static constexpr auto torch_dtype = at::kBFloat16;
};

// ===========================================================================
// Wide score matmul: A[16][D] @ B[32][D]^T -> [16][32] (two 16x16 halves)
// Uses fp16 WMMA: loads half_t data, fp32 accumulator
// A is in shared memory as float, B is in shared memory as float.
// We load from float smem and convert to half on the fly via fragments.
// Actually for fp16 WMMA we need half inputs. So data tiles must be half_t.
// ===========================================================================

// Score matmul from half_t shared memory tiles
// A_smem: [16][D_PAD] half_t (row-major)
// B_smem: [32][D_PAD] half_t (row-major, used as col-major for B^T)
template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_score_matmul_wide(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_left,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_right,
    const half_t* A_smem, int A_stride,
    const half_t* B_smem, int B_stride
) {
    wmma::fill_fragment(acc_left, 0.0f);
    wmma::fill_fragment(acc_right, 0.0f);
    for (int kk = 0; kk < D_CONST; kk += V7_WMMA_K) {
        wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::col_major> b_left, b_right;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_left, B_smem + kk, B_stride);
        wmma::load_matrix_sync(b_right, B_smem + V7_WMMA_N * B_stride + kk, B_stride);
        wmma::mma_sync(acc_left, a_frag, b_left, acc_left);
        wmma::mma_sync(acc_right, a_frag, b_right, acc_right);
    }
}

// Same but accumulate (don't zero first)
template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_score_matmul_wide_acc(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_left,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_right,
    const half_t* A_smem, int A_stride,
    const half_t* B_smem, int B_stride
) {
    for (int kk = 0; kk < D_CONST; kk += V7_WMMA_K) {
        wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::col_major> b_left, b_right;
        wmma::load_matrix_sync(a_frag, A_smem + kk, A_stride);
        wmma::load_matrix_sync(b_left, B_smem + kk, B_stride);
        wmma::load_matrix_sync(b_right, B_smem + V7_WMMA_N * B_stride + kk, B_stride);
        wmma::mma_sync(acc_left, a_frag, b_left, acc_left);
        wmma::mma_sync(acc_right, a_frag, b_right, acc_right);
    }
}

__device__ __forceinline__ void v7_store_wide_score(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& left,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& right
) {
    wmma::store_matrix_sync(tile, left, score_stride, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + V7_WMMA_N, right, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Tall score matmul: A[32][D] @ B[16][D]^T -> [32][16] (two stacked 16x16)
// ===========================================================================
template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_score_matmul_tall(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_top,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_bot,
    const half_t* A_smem, int A_stride,
    const half_t* B_smem, int B_stride
) {
    wmma::fill_fragment(acc_top, 0.0f);
    wmma::fill_fragment(acc_bot, 0.0f);
    for (int kk = 0; kk < D_CONST; kk += V7_WMMA_K) {
        wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> a_top, a_bot;
        wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_top, A_smem + kk, A_stride);
        wmma::load_matrix_sync(a_bot, A_smem + V7_WMMA_M * A_stride + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc_top, a_top, b_frag, acc_top);
        wmma::mma_sync(acc_bot, a_bot, b_frag, acc_bot);
    }
}

template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_score_matmul_tall_acc(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_top,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& acc_bot,
    const half_t* A_smem, int A_stride,
    const half_t* B_smem, int B_stride
) {
    for (int kk = 0; kk < D_CONST; kk += V7_WMMA_K) {
        wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> a_top, a_bot;
        wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::col_major> b_frag;
        wmma::load_matrix_sync(a_top, A_smem + kk, A_stride);
        wmma::load_matrix_sync(a_bot, A_smem + V7_WMMA_M * A_stride + kk, A_stride);
        wmma::load_matrix_sync(b_frag, B_smem + kk, B_stride);
        wmma::mma_sync(acc_top, a_top, b_frag, acc_top);
        wmma::mma_sync(acc_bot, a_bot, b_frag, acc_bot);
    }
}

__device__ __forceinline__ void v7_store_tall_score(
    float* tile, int score_stride,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& top,
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>& bot
) {
    wmma::store_matrix_sync(tile, top, score_stride, wmma::mem_row_major);
    wmma::store_matrix_sync(tile + V7_WMMA_M * score_stride, bot, score_stride, wmma::mem_row_major);
}

// ===========================================================================
// Output matmul: score[16][32](fp32) -> convert to half -> half_score[16][32]
// Then: half_score[16][32] @ data[32][D](half_t) -> out_frags[D/16](fp32 accum)
//
// We store the score tile to a half_t staging area, then use fp16 WMMA.
// ===========================================================================
template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_output_matmul_acc(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    half_t* half_score_smem, int half_score_stride,  // staging area [16][32+pad]
    const half_t* data_smem, int data_stride,
    int lane, int warp_id
) {
    // Convert score fp32 -> half_t in staging area
    // score_smem is [16][score_stride], half_score_smem is [16][half_score_stride]
    for (int idx = lane; idx < V7_WMMA_M * V7_BC; idx += V7_WARP_SIZE) {
        int r = idx / V7_BC;
        int c = idx % V7_BC;
        half_score_smem[r * half_score_stride + c] = HalfTraits<half_t>::from_float(score_smem[r * score_stride + c]);
    }
    __syncwarp();

    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    constexpr int k_steps = V7_BC / V7_WMMA_K;  // 32/16 = 2

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, half_score_smem + kk * V7_WMMA_K, half_score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * V7_WMMA_K * data_stride + cb * V7_WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// Transposed tall output matmul: score[32][16]^T(fp32) @ data[32][D](half_t) -> out[16][D]
template<typename half_t, int D_CONST>
__device__ __forceinline__ void v7_output_matmul_acc_tallT(
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float>* out_frags,
    const float* score_smem, int score_stride,
    half_t* half_score_smem, int half_score_stride,  // staging [32][16+pad]
    const half_t* data_smem, int data_stride,
    int lane, int warp_id
) {
    // Convert score fp32 -> half_t (the full 32x16 tall tile)
    for (int idx = lane; idx < V7_BC * V7_WMMA_N; idx += V7_WARP_SIZE) {
        int r = idx / V7_WMMA_N;
        int c = idx % V7_WMMA_N;
        half_score_smem[r * half_score_stride + c] = HalfTraits<half_t>::from_float(score_smem[r * score_stride + c]);
    }
    __syncwarp();

    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    constexpr int k_steps = V7_BC / V7_WMMA_K;  // 32/16 = 2

    for (int cb = 0; cb < n_col_blocks; cb++) {
        for (int kk = 0; kk < k_steps; kk++) {
            // A = score^T: load col-major from [32][half_score_stride]
            wmma::fragment<wmma::matrix_a, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::col_major> a_frag;
            wmma::fragment<wmma::matrix_b, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, half_t, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, half_score_smem + kk * V7_WMMA_K * half_score_stride, half_score_stride);
            wmma::load_matrix_sync(b_frag, data_smem + kk * V7_WMMA_K * data_stride + cb * V7_WMMA_N, data_stride);
            wmma::mma_sync(out_frags[cb], a_frag, b_frag, out_frags[cb]);
        }
    }
}

// ===========================================================================
// V7 Flash Forward Kernel (2-pass, fp16 WMMA, rectangular, sliding window)
//
// Pass 1: L = logsumexp per row
// Pass 2: O = sum_j P_ij * V_j via WMMA output matmuls
//
// Template params: D_CONST (head dim), half_t (__half or __nv_bfloat16)
// ===========================================================================
template<int D_CONST, typename half_t>
__global__ void v7_flash_fwd_kernel(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    half_t* __restrict__ O_out,
    float* __restrict__ L_out,
    int B, int H, int N_q, int N_kv, int D, float scale,
    int window_size, int chunk_offset  // chunk_offset = chunk_id * N_q
) {
    extern __shared__ char smem_raw[];
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;  // 34
    const int HALF_SCORE_STRIDE = V7_BC + V7_HSCORE_PAD;  // 40 (multiple of 8 for WMMA)

    // Shared memory layout (all half_t for data tiles, float for scores)
    half_t* Q_block = reinterpret_cast<half_t*>(smem_raw);
    half_t* K_tile  = Q_block + V7_BR * D_PAD;
    half_t* V_tile  = K_tile  + V7_BC * D_PAD;
    // Half-precision staging area for output matmul (per-warp)
    half_t* half_score_base = V_tile + V7_BC * D_PAD;
    // Align float score tiles to 16-byte boundary after half staging
    char* after_half = reinterpret_cast<char*>(
        half_score_base + V7_WARPS_PER_BLOCK * V7_WMMA_M * HALF_SCORE_STRIDE);
    after_half = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(after_half) + 15) & ~15ULL);
    float* score_base = reinterpret_cast<float*>(after_half);
    float* m_vec = score_base + V7_WARPS_PER_BLOCK * V7_WMMA_M * SCORE_STRIDE;
    float* l_vec = m_vec + V7_BR;

    int warp_id = threadIdx.x / V7_WARP_SIZE;
    int lane = threadIdx.x % V7_WARP_SIZE;

    // Per-head block indexing
    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int bh = blockIdx.x / blocks_per_head_q;
    int block_idx = blockIdx.x % blocks_per_head_q;
    int q_start = block_idx * V7_BR;  // local Q row offset

    int b_idx = bh / H;
    int h_idx = bh % H;

    half_t* Q_warp = Q_block + warp_id * V7_WMMA_M * D_PAD;
    float* S_w = score_base + warp_id * V7_WMMA_M * SCORE_STRIDE;
    half_t* hs_w = half_score_base + warp_id * V7_WMMA_M * HALF_SCORE_STRIDE;

    int rows_to_load = min(V7_BR, N_q - q_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Global query indices for masking
    int qi_base = chunk_offset + q_start;  // absolute query index

    // === Load Q_block ===
    for (int idx = threadIdx.x; idx < V7_BR * D_PAD; idx += V7_BLOCK_THREADS) {
        int r = idx / D_PAD;
        int d = idx % D_PAD;
        if (r < rows_to_load && d < D_CONST) {
            Q_block[idx] = Q[((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d];
        } else {
            Q_block[idx] = HalfTraits<half_t>::from_float(0.0f);
        }
    }

    for (int idx = threadIdx.x; idx < V7_BR; idx += V7_BLOCK_THREADS) {
        m_vec[idx] = -1e30f;
        l_vec[idx] = 0.0f;
    }
    __syncthreads();

    // Key index computation: ki_abs = chunk_offset + N_q - N_kv + j_local
    // where j_local is position in KV array [0, N_kv)
    int ki_base = chunk_offset + N_q - N_kv;  // absolute index of key at position 0

    // ==== Pass 1: Compute L = logsumexp ====
    for (int j_start = 0; j_start < N_kv; j_start += V7_BC) {
        int tile_cols = min(V7_BC, N_kv - j_start);

        // Tile-level mask check
        int ki_tile_start = ki_base + j_start;
        int ki_tile_end = ki_tile_start + tile_cols;
        int qi_tile_start = qi_base;
        int qi_tile_end = qi_base + rows_to_load;
        int mask_status = v7_tile_mask_status(qi_tile_start, qi_tile_end,
                                               ki_tile_start, ki_tile_end,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;  // fully masked, skip

        // Load K_tile
        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int c = idx / D_PAD;
            int d = idx % D_PAD;
            if (c < tile_cols && d < D_CONST) {
                K_tile[idx] = K[((b_idx * H + h_idx) * N_kv + j_start + c) * D_CONST + d];
            } else {
                K_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
            }
        }
        __syncthreads();

        // WMMA wide score: S[16][32] = Q_warp @ K_tile^T
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_L, S_R;
        v7_score_matmul_wide<half_t, D_CONST>(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD);
        v7_scale_accum(S_L, scale);
        v7_scale_accum(S_R, scale);
        v7_store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);
        __syncwarp();

        // Per-row max and sum update
        for (int r = 0; r < V7_WMMA_M; r++) {
            if (warp_id * V7_WMMA_M + r >= rows_to_load) continue;
            int qi = qi_base + warp_id * V7_WMMA_M + r;

            float local_max = -1e30f;
            for (int c = lane; c < tile_cols; c += V7_WARP_SIZE) {
                float s = S_w[r * SCORE_STRIDE + c];
                int ki = ki_base + j_start + c;
                if (mask_status == 1 && !v7_is_visible(qi, ki, window_size))
                    s = -1e30f;
                local_max = fmaxf(local_max, s);
            }
            local_max = v7_warp_reduce_max(local_max);
            float tile_max = __shfl_sync(V7_FULL_MASK, local_max, 0);

            float m_old = m_vec[warp_id * V7_WMMA_M + r];
            float m_new = fmaxf(m_old, tile_max);
            float correction = exp2f((m_old - m_new) * 1.4426950408889634f);  // M_LOG2E

            float p_sum = 0.0f;
            for (int c = lane; c < tile_cols; c += V7_WARP_SIZE) {
                float s = S_w[r * SCORE_STRIDE + c];
                int ki = ki_base + j_start + c;
                if (mask_status == 1 && !v7_is_visible(qi, ki, window_size))
                    s = -1e30f;
                p_sum += exp2f((s - m_new) * 1.4426950408889634f);
            }
            p_sum = v7_warp_reduce_sum(p_sum);

            if (lane == 0) {
                l_vec[warp_id * V7_WMMA_M + r] = l_vec[warp_id * V7_WMMA_M + r] * correction + p_sum;
                m_vec[warp_id * V7_WMMA_M + r] = m_new;
            }
        }
        __syncwarp();
    }

    // Finalize L = log(l) + m (convert from base-2 back)
    __syncthreads();
    for (int idx = threadIdx.x; idx < V7_BR; idx += V7_BLOCK_THREADS) {
        if (idx < rows_to_load) {
            int global_row = bh * N_q + q_start + idx;
            float L_val = logf(l_vec[idx]) + m_vec[idx];
            L_out[global_row] = L_val;
            l_vec[idx] = L_val;
        }
    }
    __syncthreads();

    // ==== Pass 2: O += P @ V using WMMA ====
    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> O_frags[n_col_blocks];
    for (int cb = 0; cb < n_col_blocks; cb++)
        wmma::fill_fragment(O_frags[cb], 0.0f);

    for (int j_start = 0; j_start < N_kv; j_start += V7_BC) {
        int tile_cols = min(V7_BC, N_kv - j_start);

        // Tile-level mask check
        int ki_tile_start = ki_base + j_start;
        int ki_tile_end = ki_tile_start + tile_cols;
        int qi_tile_start = qi_base;
        int qi_tile_end = qi_base + rows_to_load;
        int mask_status = v7_tile_mask_status(qi_tile_start, qi_tile_end,
                                               ki_tile_start, ki_tile_end,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        // Load K_tile and V_tile
        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int c = idx / D_PAD;
            int d = idx % D_PAD;
            if (c < tile_cols && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_kv + j_start + c) * D_CONST + d;
                K_tile[c * D_PAD + d] = K[src];
                V_tile[c * D_PAD + d] = V[src];
            } else {
                K_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
                V_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
            }
        }
        __syncthreads();

        // Recompute S via WMMA
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_L, S_R;
        v7_score_matmul_wide<half_t, D_CONST>(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD);
        v7_scale_accum(S_L, scale);
        v7_scale_accum(S_R, scale);
        v7_store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);
        __syncwarp();

        // Element-wise: P = exp(S - L), with mask
        for (int r = 0; r < V7_WMMA_M; r++) {
            int qi = qi_base + warp_id * V7_WMMA_M + r;
            for (int c = lane; c < V7_BC; c += V7_WARP_SIZE) {
                int si = r * SCORE_STRIDE + c;
                if (c < tile_cols && warp_id * V7_WMMA_M + r < rows_to_load) {
                    float L_val = l_vec[warp_id * V7_WMMA_M + r];
                    float p_val = __expf(S_w[si] - L_val);
                    int ki = ki_base + j_start + c;
                    if (mask_status == 1 && !v7_is_visible(qi, ki, window_size))
                        p_val = 0.0f;
                    S_w[si] = p_val;
                } else {
                    S_w[si] = 0.0f;
                }
            }
        }
        __syncwarp();

        // O += P @ V via half-precision WMMA output matmul
        v7_output_matmul_acc<half_t, D_CONST>(
            O_frags, S_w, SCORE_STRIDE, hs_w, HALF_SCORE_STRIDE,
            V_tile, D_PAD, lane, warp_id);
    }

    // Store O: accumulator frags (fp32) -> float staging in smem -> half_t global
    __syncthreads();
    float* O_float_staging = reinterpret_cast<float*>(smem_raw);
    const int O_STRIDE = D_CONST;

    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            O_float_staging + warp_id * V7_WMMA_M * O_STRIDE + cb * V7_WMMA_N,
            O_frags[cb], O_STRIDE, wmma::mem_row_major);
    }
    __syncthreads();

    // Convert float -> half_t and write to global
    for (int idx = threadIdx.x; idx < rows_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int r = idx / D_CONST;
        int d = idx % D_CONST;
        float val = O_float_staging[r * O_STRIDE + d];
        O_out[((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d] =
            HalfTraits<half_t>::from_float(val);
    }
}

// ===========================================================================
// Host wrapper: V7 forward (template dispatcher)
// ===========================================================================
template<int D_CONST, typename half_t>
static std::vector<at::Tensor> v7_forward_impl(
    at::Tensor Q, at::Tensor K, at::Tensor V,
    int window_size, int chunk_id
) {
    int B = Q.size(0);
    int H = Q.size(1);
    int N_q = Q.size(2);
    int N_kv = K.size(2);
    int D = Q.size(3);

    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int chunk_offset = chunk_id * N_q;

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(at::kFloat);

    int BH = B * H;
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;
    const int HALF_SCORE_STRIDE = V7_BC + V7_HSCORE_PAD;

    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int grid = BH * blocks_per_head_q;

    // Shared memory: data tiles (half_t) + half score staging + score tiles (float) + vectors
    int smem_bytes =
        (V7_BR * D_PAD + 2 * V7_BC * D_PAD) * sizeof(half_t)  // Q_block, K_tile, V_tile
        + V7_WARPS_PER_BLOCK * V7_WMMA_M * HALF_SCORE_STRIDE * sizeof(half_t)  // half score staging
        + 16  // alignment padding
        + V7_WARPS_PER_BLOCK * V7_WMMA_M * SCORE_STRIDE * sizeof(float)  // score tiles
        + 2 * V7_BR * sizeof(float);  // m_vec, l_vec

    // Also need space for O_float_staging at the end (reuses smem_raw start)
    int o_staging = V7_BR * D_CONST * sizeof(float);
    smem_bytes = max(smem_bytes, o_staging);

    if (smem_bytes > 48 * 1024) {
        V7_CUDA_CHECK(cudaFuncSetAttribute(
            v7_flash_fwd_kernel<D_CONST, half_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }

    auto O = at::empty({B, H, N_q, D}, opts_half);
    auto L = at::empty({B, H, N_q}, opts_float);

    v7_flash_fwd_kernel<D_CONST, half_t><<<grid, V7_BLOCK_THREADS, smem_bytes>>>(
        reinterpret_cast<const half_t*>(Q.data_ptr()),
        reinterpret_cast<const half_t*>(K.data_ptr()),
        reinterpret_cast<const half_t*>(V.data_ptr()),
        reinterpret_cast<half_t*>(O.data_ptr()),
        L.data_ptr<float>(),
        B, H, N_q, N_kv, D, scale,
        window_size, chunk_offset);
    V7_CUDA_CHECK(cudaGetLastError());

    return {O, L};
}

// ===========================================================================
// Public host function: v7_forward_cuda
// ===========================================================================
std::vector<at::Tensor> v7_forward_cuda(
    at::Tensor Q, at::Tensor K, at::Tensor V,
    int window_size, int chunk_id
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    int D = Q.size(3);
    bool is_bf16 = (Q.dtype() == at::kBFloat16);
    bool is_fp16 = (Q.dtype() == at::kHalf);
    TORCH_CHECK(is_bf16 || is_fp16, "Q must be float16 or bfloat16");
    TORCH_CHECK(D % 16 == 0, "D must be multiple of 16");

    if (is_bf16) {
        if (D == 64) return v7_forward_impl<64, __nv_bfloat16>(Q, K, V, window_size, chunk_id);
        if (D == 80) return v7_forward_impl<80, __nv_bfloat16>(Q, K, V, window_size, chunk_id);
        if (D == 96) return v7_forward_impl<96, __nv_bfloat16>(Q, K, V, window_size, chunk_id);
        TORCH_CHECK(false, "Unsupported D=", D, " for V7 (must be 64, 80, or 96)");
    } else {
        if (D == 64) return v7_forward_impl<64, __half>(Q, K, V, window_size, chunk_id);
        if (D == 80) return v7_forward_impl<80, __half>(Q, K, V, window_size, chunk_id);
        if (D == 96) return v7_forward_impl<96, __half>(Q, K, V, window_size, chunk_id);
        TORCH_CHECK(false, "Unsupported D=", D, " for V7 (must be 64, 80, or 96)");
    }
    return {};  // unreachable
}

// ===========================================================================
// V7 Flash Backward Row Kernel: D_i and dQ
// ===========================================================================
template<int D_CONST, typename half_t>
__global__ void v7_flash_bwd_row_kernel(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    const half_t* __restrict__ dO,
    const half_t* __restrict__ O,
    const float* __restrict__ L,
    float* __restrict__ D_out,
    half_t* __restrict__ dQ_out,
    int B, int H, int N_q, int N_kv, int D, float scale,
    int window_size, int chunk_offset
) {
    extern __shared__ char smem_raw[];
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;
    const int HALF_SCORE_STRIDE = V7_BC + V7_HSCORE_PAD;

    half_t* Q_block  = reinterpret_cast<half_t*>(smem_raw);
    half_t* dO_block = Q_block  + V7_BR * D_PAD;
    half_t* K_tile   = dO_block + V7_BR * D_PAD;
    half_t* V_tile   = K_tile   + V7_BC * D_PAD;
    half_t* half_score_base = V_tile + V7_BC * D_PAD;
    char* after_half = reinterpret_cast<char*>(
        half_score_base + V7_WARPS_PER_BLOCK * V7_WMMA_M * HALF_SCORE_STRIDE);
    after_half = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(after_half) + 15) & ~15ULL);
    float* score_base = reinterpret_cast<float*>(after_half);
    // 2 score tiles per warp: S and dP
    float* L_vec  = score_base + V7_WARPS_PER_BLOCK * 2 * V7_WMMA_M * SCORE_STRIDE;
    float* Di_vec = L_vec + V7_BR;

    int warp_id = threadIdx.x / V7_WARP_SIZE;
    int lane = threadIdx.x % V7_WARP_SIZE;

    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int bh = blockIdx.x / blocks_per_head_q;
    int block_idx = blockIdx.x % blocks_per_head_q;
    int q_start = block_idx * V7_BR;
    int b_idx = bh / H;
    int h_idx = bh % H;
    int qi_base = chunk_offset + q_start;
    int ki_base = chunk_offset + N_q - N_kv;

    float* S_w  = score_base + warp_id * 2 * V7_WMMA_M * SCORE_STRIDE;
    float* dP_w = S_w + V7_WMMA_M * SCORE_STRIDE;
    half_t* hs_w = half_score_base + warp_id * V7_WMMA_M * HALF_SCORE_STRIDE;
    half_t* Q_warp  = Q_block  + warp_id * V7_WMMA_M * D_PAD;
    half_t* dO_warp = dO_block + warp_id * V7_WMMA_M * D_PAD;

    int rows_to_load = min(V7_BR, N_q - q_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Load Q and dO
    for (int idx = threadIdx.x; idx < V7_BR * D_PAD; idx += V7_BLOCK_THREADS) {
        int r = idx / D_PAD; int d = idx % D_PAD;
        half_t val = HalfTraits<half_t>::from_float(0.0f);
        if (r < rows_to_load && d < D_CONST) {
            int src = ((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d;
            Q_block[idx] = Q[src];
            dO_block[idx] = dO[src];
        } else {
            Q_block[idx] = val;
            dO_block[idx] = val;
        }
    }
    for (int idx = threadIdx.x; idx < V7_BR; idx += V7_BLOCK_THREADS) {
        int global_row = bh * N_q + q_start + idx;
        L_vec[idx] = (idx < rows_to_load) ? L[global_row] : 0.0f;
        Di_vec[idx] = 0.0f;
    }
    __syncthreads();

    // D_i = sum_d dO[i,d] * O[i,d]
    for (int r = 0; r < V7_WMMA_M; r++) {
        if (warp_id * V7_WMMA_M + r >= rows_to_load) {
            if (lane == 0) Di_vec[warp_id * V7_WMMA_M + r] = 0.0f;
            continue;
        }
        int global_row = ((b_idx * H + h_idx) * N_q + q_start + warp_id * V7_WMMA_M + r);
        float partial = 0.0f;
        for (int d = lane; d < D_CONST; d += V7_WARP_SIZE) {
            float dO_val = HalfTraits<half_t>::to_float(dO_block[(warp_id * V7_WMMA_M + r) * D_PAD + d]);
            float O_val = HalfTraits<half_t>::to_float(O[global_row * D_CONST + d]);
            partial += dO_val * O_val;
        }
        partial = v7_warp_reduce_sum(partial);
        if (lane == 0) {
            Di_vec[warp_id * V7_WMMA_M + r] = partial;
            D_out[bh * N_q + q_start + warp_id * V7_WMMA_M + r] = partial;
        }
    }
    __syncthreads();

    // dQ accumulators
    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dQ_frags[n_col_blocks];
    for (int cb = 0; cb < n_col_blocks; cb++)
        wmma::fill_fragment(dQ_frags[cb], 0.0f);

    // Main loop over KV tiles
    for (int j_start = 0; j_start < N_kv; j_start += V7_BC) {
        int tile_cols = min(V7_BC, N_kv - j_start);
        int ki_tile_start = ki_base + j_start;
        int mask_status = v7_tile_mask_status(qi_base, qi_base + rows_to_load,
                                               ki_tile_start, ki_tile_start + tile_cols,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int c = idx / D_PAD; int d = idx % D_PAD;
            if (c < tile_cols && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_kv + j_start + c) * D_CONST + d;
                K_tile[idx] = K[src];
                V_tile[idx] = V[src];
            } else {
                K_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
                V_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
            }
        }
        __syncthreads();

        // S = Q @ K^T
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_L, S_R;
        v7_score_matmul_wide<half_t, D_CONST>(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD);
        v7_scale_accum(S_L, scale);
        v7_scale_accum(S_R, scale);
        v7_store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);

        // dP = dO @ V^T
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dP_L, dP_R;
        v7_score_matmul_wide<half_t, D_CONST>(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD);
        v7_store_wide_score(dP_w, SCORE_STRIDE, dP_L, dP_R);
        __syncwarp();

        // Element-wise: P = exp(S - L), dS = P * (dP - Di), store scale*dS to dP_w
        for (int r = 0; r < V7_WMMA_M; r++) {
            int qi = qi_base + warp_id * V7_WMMA_M + r;
            for (int c = lane; c < V7_BC; c += V7_WARP_SIZE) {
                int si = r * SCORE_STRIDE + c;
                if (c < tile_cols && warp_id * V7_WMMA_M + r < rows_to_load) {
                    float s = S_w[si];
                    float l = L_vec[warp_id * V7_WMMA_M + r];
                    float p = __expf(s - l);
                    int ki = ki_base + j_start + c;
                    if (mask_status == 1 && !v7_is_visible(qi, ki, window_size)) p = 0.0f;
                    float dp = dP_w[si];
                    float di = Di_vec[warp_id * V7_WMMA_M + r];
                    dP_w[si] = scale * p * (dp - di);
                } else {
                    dP_w[si] = 0.0f;
                }
            }
        }
        __syncwarp();

        // dQ += (scale*dS) @ K via output matmul
        v7_output_matmul_acc<half_t, D_CONST>(
            dQ_frags, dP_w, SCORE_STRIDE, hs_w, HALF_SCORE_STRIDE,
            K_tile, D_PAD, lane, warp_id);
    }

    // Store dQ
    __syncthreads();
    float* dQ_staging = reinterpret_cast<float*>(smem_raw);
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            dQ_staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            dQ_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < rows_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int r = idx / D_CONST; int d = idx % D_CONST;
        dQ_out[((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d] =
            HalfTraits<half_t>::from_float(dQ_staging[r * D_CONST + d]);
    }
}

// ===========================================================================
// V7 Flash Backward Col Kernel: dK and dV
// ===========================================================================
template<int D_CONST, typename half_t>
__global__ void v7_flash_bwd_col_kernel(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    const half_t* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_vec_in,
    half_t* __restrict__ dK_out,
    half_t* __restrict__ dV_out,
    int B, int H, int N_q, int N_kv, int D, float scale,
    int window_size, int chunk_offset
) {
    extern __shared__ char smem_raw[];
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int B_SCORE_STRIDE = V7_WMMA_N + V7_SCORE_PAD;  // 18
    const int B_HSCORE_STRIDE = V7_BC + V7_HSCORE_PAD;  // 40 (tall half staging)

    // Col kernel: blocks over N_kv (BR=64 cols), inner loop over N_q (BC=32 row tiles)
    half_t* K_block = reinterpret_cast<half_t*>(smem_raw);
    half_t* V_block = K_block + V7_BR * D_PAD;
    half_t* Q_tile  = V_block + V7_BR * D_PAD;
    half_t* dO_tile = Q_tile  + V7_BC * D_PAD;
    // Tall half staging for output matmuls (per-warp)
    half_t* half_score_base = dO_tile + V7_BC * D_PAD;
    char* after_half = reinterpret_cast<char*>(
        half_score_base + V7_WARPS_PER_BLOCK * V7_BC * B_HSCORE_STRIDE);
    after_half = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(after_half) + 15) & ~15ULL);
    float* score_base = reinterpret_cast<float*>(after_half);
    // 2 tall score tiles per warp: S[32x16], dP[32x16]
    float* tile_L = score_base + V7_WARPS_PER_BLOCK * 2 * V7_BC * B_SCORE_STRIDE;
    float* tile_D = tile_L + V7_BC;

    int warp_id = threadIdx.x / V7_WARP_SIZE;
    int lane = threadIdx.x % V7_WARP_SIZE;

    int blocks_per_head_kv = (N_kv + V7_BR - 1) / V7_BR;
    int bh = blockIdx.x / blocks_per_head_kv;
    int block_idx = blockIdx.x % blocks_per_head_kv;
    int kv_start = block_idx * V7_BR;  // local KV col offset
    int b_idx = bh / H;
    int h_idx = bh % H;
    int ki_base = chunk_offset + N_q - N_kv;

    float* S_w  = score_base + warp_id * 2 * V7_BC * B_SCORE_STRIDE;
    float* dP_w = S_w + V7_BC * B_SCORE_STRIDE;
    half_t* hs_w = half_score_base + warp_id * V7_BC * B_HSCORE_STRIDE;
    half_t* K_warp = K_block + warp_id * V7_WMMA_M * D_PAD;
    half_t* V_warp = V_block + warp_id * V7_WMMA_M * D_PAD;

    int cols_to_load = min(V7_BR, N_kv - kv_start);
    if (cols_to_load < 0) cols_to_load = 0;

    // Load K_block and V_block
    for (int idx = threadIdx.x; idx < V7_BR * D_PAD; idx += V7_BLOCK_THREADS) {
        int c = idx / D_PAD; int d = idx % D_PAD;
        if (c < cols_to_load && d < D_CONST) {
            int src = ((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d;
            K_block[idx] = K[src];
            V_block[idx] = V[src];
        } else {
            K_block[idx] = HalfTraits<half_t>::from_float(0.0f);
            V_block[idx] = HalfTraits<half_t>::from_float(0.0f);
        }
    }
    __syncthreads();

    // dK, dV accumulators
    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dK_frags[n_col_blocks];
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dV_frags[n_col_blocks];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(dK_frags[cb], 0.0f);
        wmma::fill_fragment(dV_frags[cb], 0.0f);
    }

    // Main loop over Q row tiles
    for (int i_start = 0; i_start < N_q; i_start += V7_BC) {
        int tile_rows = min(V7_BC, N_q - i_start);
        int qi_tile_start = chunk_offset + i_start;
        // Block-level mask check (all warps must agree to skip)
        int ki_block_start = ki_base + kv_start;
        int ki_block_end = ki_block_start + cols_to_load;
        int mask_status = v7_tile_mask_status(qi_tile_start, qi_tile_start + tile_rows,
                                               ki_block_start, ki_block_end,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int r = idx / D_PAD; int d = idx % D_PAD;
            if (r < tile_rows && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_q + i_start + r) * D_CONST + d;
                Q_tile[idx] = Q[src];
                dO_tile[idx] = dO[src];
            } else {
                Q_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
                dO_tile[idx] = HalfTraits<half_t>::from_float(0.0f);
            }
        }
        for (int idx = threadIdx.x; idx < V7_BC; idx += V7_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh * N_q + i_start + idx;
                tile_L[idx] = L[global_row];
                tile_D[idx] = D_vec_in[global_row];
            } else {
                tile_L[idx] = 0.0f;
                tile_D[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Tall score: S = Q_tile[32] @ K_warp[16]^T -> 32×16
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_top, S_bot;
        v7_score_matmul_tall<half_t, D_CONST>(S_top, S_bot, Q_tile, D_PAD, K_warp, D_PAD);
        v7_scale_accum(S_top, scale);
        v7_scale_accum(S_bot, scale);
        v7_store_tall_score(S_w, B_SCORE_STRIDE, S_top, S_bot);

        // Tall score: dP = dO_tile[32] @ V_warp[16]^T -> 32×16
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dP_top, dP_bot;
        v7_score_matmul_tall<half_t, D_CONST>(dP_top, dP_bot, dO_tile, D_PAD, V_warp, D_PAD);
        v7_store_tall_score(dP_w, B_SCORE_STRIDE, dP_top, dP_bot);
        __syncwarp();

        // Element-wise: P, dS with mask (32×16)
        for (int idx = lane; idx < V7_BC * V7_WMMA_N; idx += V7_WARP_SIZE) {
            int i_local = idx / V7_WMMA_N;
            int j_local = idx % V7_WMMA_N;
            int si = i_local * B_SCORE_STRIDE + j_local;

            int qi = chunk_offset + i_start + i_local;
            int ki = ki_base + kv_start + warp_id * V7_WMMA_M + j_local;

            float s = S_w[si];
            float l = tile_L[i_local];
            float p = (i_local < tile_rows) ? __expf(s - l) : 0.0f;
            if (!v7_is_visible(qi, ki, window_size)) p = 0.0f;
            float dp = dP_w[si];
            float di = tile_D[i_local];
            float ds = p * (dp - di);

            S_w[si]  = p;              // P for dV
            dP_w[si] = scale * ds;     // scale * dS for dK
        }
        __syncwarp();

        // dK += (scale*dS)^T @ Q_tile, dV += P^T @ dO_tile
        v7_output_matmul_acc_tallT<half_t, D_CONST>(
            dK_frags, dP_w, B_SCORE_STRIDE, hs_w, B_HSCORE_STRIDE,
            Q_tile, D_PAD, lane, warp_id);
        v7_output_matmul_acc_tallT<half_t, D_CONST>(
            dV_frags, S_w, B_SCORE_STRIDE, hs_w, B_HSCORE_STRIDE,
            dO_tile, D_PAD, lane, warp_id);
    }

    // Store dK and dV
    __syncthreads();
    float* staging = reinterpret_cast<float*>(smem_raw);
    // Store dK
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            dK_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < cols_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int c = idx / D_CONST; int d = idx % D_CONST;
        dK_out[((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[c * D_CONST + d]);
    }

    // Store dV
    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(
            staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            dV_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < cols_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int c = idx / D_CONST; int d = idx % D_CONST;
        dV_out[((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[c * D_CONST + d]);
    }
}

// ===========================================================================
// Host wrapper: V7 backward (template dispatcher)
// ===========================================================================
template<int D_CONST, typename half_t>
static std::vector<at::Tensor> v7_backward_impl(
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor O, at::Tensor L,
    int window_size, int chunk_id
) {
    int B = Q.size(0), H = Q.size(1), N_q = Q.size(2), D = Q.size(3);
    int N_kv = K.size(2);
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int chunk_offset = chunk_id * N_q;
    int BH = B * H;
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;
    const int HALF_SCORE_STRIDE = V7_BC + V7_HSCORE_PAD;
    const int B_SCORE_STRIDE = V7_WMMA_N + V7_SCORE_PAD;
    const int B_HSCORE_STRIDE = V7_BC + V7_HSCORE_PAD;

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(at::kFloat);

    // Row kernel: blocks over Q rows
    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int grid_row = BH * blocks_per_head_q;

    int smem_row =
        (2 * V7_BR * D_PAD + 2 * V7_BC * D_PAD) * sizeof(half_t)
        + V7_WARPS_PER_BLOCK * V7_WMMA_M * HALF_SCORE_STRIDE * sizeof(half_t)
        + 16
        + V7_WARPS_PER_BLOCK * 2 * V7_WMMA_M * SCORE_STRIDE * sizeof(float)
        + 2 * V7_BR * sizeof(float);
    int o_staging_row = V7_BR * D_CONST * sizeof(float);
    smem_row = max(smem_row, o_staging_row);

    // Col kernel: blocks over KV cols
    int blocks_per_head_kv = (N_kv + V7_BR - 1) / V7_BR;
    int grid_col = BH * blocks_per_head_kv;

    int smem_col =
        (2 * V7_BR * D_PAD + 2 * V7_BC * D_PAD) * sizeof(half_t)
        + V7_WARPS_PER_BLOCK * V7_BC * B_HSCORE_STRIDE * sizeof(half_t)
        + 16
        + V7_WARPS_PER_BLOCK * 2 * V7_BC * B_SCORE_STRIDE * sizeof(float)
        + 2 * V7_BC * sizeof(float);
    int o_staging_col = V7_BR * D_CONST * sizeof(float);
    smem_col = max(smem_col, o_staging_col);

    if (smem_row > 48 * 1024) {
        V7_CUDA_CHECK(cudaFuncSetAttribute(
            v7_flash_bwd_row_kernel<D_CONST, half_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_row));
    }
    if (smem_col > 48 * 1024) {
        V7_CUDA_CHECK(cudaFuncSetAttribute(
            v7_flash_bwd_col_kernel<D_CONST, half_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_col));
    }

    auto D_vec = at::empty({B, H, N_q}, opts_float);
    auto dQ = at::empty({B, H, N_q, D}, opts_half);
    auto dK = at::empty({B, H, N_kv, D}, opts_half);
    auto dV = at::empty({B, H, N_kv, D}, opts_half);

    v7_flash_bwd_row_kernel<D_CONST, half_t><<<grid_row, V7_BLOCK_THREADS, smem_row>>>(
        reinterpret_cast<const half_t*>(Q.data_ptr()),
        reinterpret_cast<const half_t*>(K.data_ptr()),
        reinterpret_cast<const half_t*>(V.data_ptr()),
        reinterpret_cast<const half_t*>(dO.data_ptr()),
        reinterpret_cast<const half_t*>(O.data_ptr()),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(),
        reinterpret_cast<half_t*>(dQ.data_ptr()),
        B, H, N_q, N_kv, D, scale, window_size, chunk_offset);
    V7_CUDA_CHECK(cudaGetLastError());

    v7_flash_bwd_col_kernel<D_CONST, half_t><<<grid_col, V7_BLOCK_THREADS, smem_col>>>(
        reinterpret_cast<const half_t*>(Q.data_ptr()),
        reinterpret_cast<const half_t*>(K.data_ptr()),
        reinterpret_cast<const half_t*>(V.data_ptr()),
        reinterpret_cast<const half_t*>(dO.data_ptr()),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(),
        reinterpret_cast<half_t*>(dK.data_ptr()),
        reinterpret_cast<half_t*>(dV.data_ptr()),
        B, H, N_q, N_kv, D, scale, window_size, chunk_offset);
    V7_CUDA_CHECK(cudaGetLastError());

    return {dQ, dK, dV};
}

// ===========================================================================
// Public: v7_backward_cuda
// ===========================================================================
std::vector<at::Tensor> v7_backward_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor O, at::Tensor L,
    int window_size, int chunk_id
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    dO = dO.contiguous(); Q = Q.contiguous(); K = K.contiguous();
    V = V.contiguous(); O = O.contiguous(); L = L.contiguous();

    int D = Q.size(3);
    bool is_bf16 = (Q.dtype() == at::kBFloat16);
    TORCH_CHECK(is_bf16 || Q.dtype() == at::kHalf, "Q must be float16 or bfloat16");
    TORCH_CHECK(D % 16 == 0, "D must be multiple of 16");

    if (is_bf16) {
        if (D == 64) return v7_backward_impl<64, __nv_bfloat16>(dO, Q, K, V, O, L, window_size, chunk_id);
        if (D == 80) return v7_backward_impl<80, __nv_bfloat16>(dO, Q, K, V, O, L, window_size, chunk_id);
        if (D == 96) return v7_backward_impl<96, __nv_bfloat16>(dO, Q, K, V, O, L, window_size, chunk_id);
    } else {
        if (D == 64) return v7_backward_impl<64, __half>(dO, Q, K, V, O, L, window_size, chunk_id);
        if (D == 80) return v7_backward_impl<80, __half>(dO, Q, K, V, O, L, window_size, chunk_id);
        if (D == 96) return v7_backward_impl<96, __half>(dO, Q, K, V, O, L, window_size, chunk_id);
    }
    TORCH_CHECK(false, "Unsupported D=", D);
    return {};
}

// ===========================================================================
// V7 Double Backward Kernel A (row): 2-pass
// Pass 1: D_i, dot2, A, E -> dot3
// Pass 2: g_Q, g_dO
// ===========================================================================
template<int D_CONST, typename half_t>
__global__ void v7_kernel_A_row(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    const half_t* __restrict__ dO,
    const half_t* __restrict__ O,
    const half_t* __restrict__ g_dQ,
    const half_t* __restrict__ g_dK,
    const half_t* __restrict__ g_dV,
    const float* __restrict__ L,
    float* __restrict__ D_out,
    float* __restrict__ dot2_out,
    float* __restrict__ dot3_out,
    half_t* __restrict__ g_Q_out,
    half_t* __restrict__ g_dO_out,
    int B, int H, int N_q, int N_kv, int D, float scale,
    int window_size, int chunk_offset
) {
    extern __shared__ char smem_raw[];
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;

    // Row data: Q, dO, g_dQ (BR rows each)
    half_t* Q_block     = reinterpret_cast<half_t*>(smem_raw);
    half_t* dO_block    = Q_block     + V7_BR * D_PAD;
    half_t* g_dQ_block  = dO_block    + V7_BR * D_PAD;
    // Col tiles: K, V, g_dK, g_dV (BC rows each)
    half_t* K_tile      = g_dQ_block  + V7_BR * D_PAD;
    half_t* V_tile      = K_tile      + V7_BC * D_PAD;
    half_t* g_dK_tile   = V_tile      + V7_BC * D_PAD;
    half_t* g_dV_tile   = g_dK_tile   + V7_BC * D_PAD;
    // Score tiles directly after data tiles (no separate half_score allocation)
    char* after_data = reinterpret_cast<char*>(g_dV_tile + V7_BC * D_PAD);
    after_data = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(after_data) + 15) & ~15ULL);
    float* score_base = reinterpret_cast<float*>(after_data);
    // 5 score tiles per warp: S, gdS, dP, gPdV, extra
    float* L_vec     = score_base + V7_WARPS_PER_BLOCK * 5 * V7_WMMA_M * SCORE_STRIDE;
    float* Di_vec    = L_vec + V7_BR;
    float* dot2_smem = Di_vec + V7_BR;
    float* dot3_smem = dot2_smem + V7_BR;

    int warp_id = threadIdx.x / V7_WARP_SIZE;
    int lane = threadIdx.x % V7_WARP_SIZE;

    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int bh = blockIdx.x / blocks_per_head_q;
    int block_idx = blockIdx.x % blocks_per_head_q;
    int q_start = block_idx * V7_BR;
    int b_idx = bh / H;
    int h_idx = bh % H;
    int qi_base = chunk_offset + q_start;
    int ki_base = chunk_offset + N_q - N_kv;

    float* S_w     = score_base + warp_id * 5 * V7_WMMA_M * SCORE_STRIDE;
    float* gdS_w   = S_w     + V7_WMMA_M * SCORE_STRIDE;
    float* dP_w    = gdS_w   + V7_WMMA_M * SCORE_STRIDE;
    float* gPdV_w  = dP_w    + V7_WMMA_M * SCORE_STRIDE;
    float* extra_w = gPdV_w  + V7_WMMA_M * SCORE_STRIDE;
    // half_score aliases with gdS_w (free during output matmuls)
    // stride: SCORE_STRIDE floats = SCORE_STRIDE*2 half_t per row
    const int A_HSCORE_STRIDE = SCORE_STRIDE * sizeof(float) / sizeof(half_t);
    half_t* hs_w   = reinterpret_cast<half_t*>(gdS_w);

    half_t* Q_warp    = Q_block    + warp_id * V7_WMMA_M * D_PAD;
    half_t* dO_warp   = dO_block   + warp_id * V7_WMMA_M * D_PAD;
    half_t* g_dQ_warp = g_dQ_block + warp_id * V7_WMMA_M * D_PAD;

    int rows_to_load = min(V7_BR, N_q - q_start);
    if (rows_to_load < 0) rows_to_load = 0;

    // Load row data
    for (int idx = threadIdx.x; idx < V7_BR * D_PAD; idx += V7_BLOCK_THREADS) {
        int r = idx / D_PAD; int d = idx % D_PAD;
        half_t zero = HalfTraits<half_t>::from_float(0.0f);
        if (r < rows_to_load && d < D_CONST) {
            int src = ((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d;
            Q_block[idx]    = Q[src];
            dO_block[idx]   = dO[src];
            g_dQ_block[idx] = g_dQ[src];
        } else {
            Q_block[idx] = zero; dO_block[idx] = zero; g_dQ_block[idx] = zero;
        }
    }
    for (int idx = threadIdx.x; idx < V7_BR; idx += V7_BLOCK_THREADS) {
        int global_row = bh * N_q + q_start + idx;
        L_vec[idx] = (idx < rows_to_load) ? L[global_row] : 0.0f;
        Di_vec[idx] = 0.0f; dot2_smem[idx] = 0.0f; dot3_smem[idx] = 0.0f;
    }
    __syncthreads();

    // D_i = sum_d dO * O
    for (int r = 0; r < V7_WMMA_M; r++) {
        if (warp_id * V7_WMMA_M + r >= rows_to_load) {
            if (lane == 0) Di_vec[warp_id * V7_WMMA_M + r] = 0.0f;
            continue;
        }
        int global_row = ((b_idx * H + h_idx) * N_q + q_start + warp_id * V7_WMMA_M + r);
        float partial = 0.0f;
        for (int d = lane; d < D_CONST; d += V7_WARP_SIZE) {
            float dO_val = HalfTraits<half_t>::to_float(dO_block[(warp_id * V7_WMMA_M + r) * D_PAD + d]);
            float O_val  = HalfTraits<half_t>::to_float(O[global_row * D_CONST + d]);
            partial += dO_val * O_val;
        }
        partial = v7_warp_reduce_sum(partial);
        if (lane == 0) {
            Di_vec[warp_id * V7_WMMA_M + r] = partial;
            D_out[bh * N_q + q_start + warp_id * V7_WMMA_M + r] = partial;
        }
    }
    __syncthreads();

    // ==== Pass 1: dot2, A, E -> dot3 ====
    float dot2_acc[V7_WMMA_M], A_acc[V7_WMMA_M], E_acc[V7_WMMA_M];
    for (int r = 0; r < V7_WMMA_M; r++) {
        dot2_acc[r] = 0.0f; A_acc[r] = 0.0f; E_acc[r] = 0.0f;
    }

    for (int j_start = 0; j_start < N_kv; j_start += V7_BC) {
        int tile_cols = min(V7_BC, N_kv - j_start);
        int ki_tile_start = ki_base + j_start;
        int mask_status = v7_tile_mask_status(qi_base, qi_base + rows_to_load,
                                               ki_tile_start, ki_tile_start + tile_cols,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int c = idx / D_PAD; int d = idx % D_PAD;
            half_t zero = HalfTraits<half_t>::from_float(0.0f);
            if (c < tile_cols && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_kv + j_start + c) * D_CONST + d;
                K_tile[idx] = K[src]; V_tile[idx] = V[src];
                g_dK_tile[idx] = g_dK[src]; g_dV_tile[idx] = g_dV[src];
            } else {
                K_tile[idx] = zero; V_tile[idx] = zero;
                g_dK_tile[idx] = zero; g_dV_tile[idx] = zero;
            }
        }
        __syncthreads();

        // Score matmuls: S, gdS, dP, gPdV
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_L, S_R;
        v7_score_matmul_wide<half_t, D_CONST>(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD);
        v7_scale_accum(S_L, scale); v7_scale_accum(S_R, scale);
        v7_store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gdS_L, gdS_R;
        v7_score_matmul_wide<half_t, D_CONST>(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD);
        v7_score_matmul_wide_acc<half_t, D_CONST>(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD);
        v7_scale_accum(gdS_L, scale); v7_scale_accum(gdS_R, scale);
        v7_store_wide_score(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dP_L, dP_R;
        v7_score_matmul_wide<half_t, D_CONST>(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD);
        v7_store_wide_score(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gPdV_L, gPdV_R;
        v7_score_matmul_wide<half_t, D_CONST>(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD);
        v7_store_wide_score(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);
        __syncwarp();

        // Element-wise reductions
        for (int r = 0; r < V7_WMMA_M; r++) {
            float p_dot2 = 0.0f, p_A = 0.0f, p_E = 0.0f;
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;
                float s = S_w[si];
                float l = L_vec[warp_id * V7_WMMA_M + r];
                int qi = qi_base + warp_id * V7_WMMA_M + r;
                int ki = ki_base + j_start + lane;
                float p = __expf(s - l);
                if (!v7_is_visible(qi, ki, window_size)) p = 0.0f;
                float gds = gdS_w[si];
                float dp = dP_w[si];
                float gpdv = gPdV_w[si];
                float pg = p * gds;
                p_dot2 = pg;
                p_A = pg * dp;
                p_E = p * gpdv;
            }
            p_dot2 = v7_warp_reduce_sum(p_dot2);
            p_A    = v7_warp_reduce_sum(p_A);
            p_E    = v7_warp_reduce_sum(p_E);
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
        for (int r = 0; r < V7_WMMA_M; r++) {
            if (warp_id * V7_WMMA_M + r >= rows_to_load) continue;
            int global_row = bh * N_q + q_start + warp_id * V7_WMMA_M + r;
            float d = Di_vec[warp_id * V7_WMMA_M + r];
            float d2 = dot2_acc[r];
            float d3 = A_acc[r] - 2.0f * d * d2 + E_acc[r];
            dot2_out[global_row] = d2;
            dot3_out[global_row] = d3;
            dot2_smem[warp_id * V7_WMMA_M + r] = d2;
            dot3_smem[warp_id * V7_WMMA_M + r] = d3;
        }
    }
    __syncthreads();

    // ==== Pass 2: g_Q and g_dO ====
    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> g_Q_frags[n_col_blocks];
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> g_dO_frags[n_col_blocks];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_Q_frags[cb], 0.0f);
        wmma::fill_fragment(g_dO_frags[cb], 0.0f);
    }

    for (int j_start = 0; j_start < N_kv; j_start += V7_BC) {
        int tile_cols = min(V7_BC, N_kv - j_start);
        int ki_tile_start = ki_base + j_start;
        int mask_status = v7_tile_mask_status(qi_base, qi_base + rows_to_load,
                                               ki_tile_start, ki_tile_start + tile_cols,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int c = idx / D_PAD; int d = idx % D_PAD;
            half_t zero = HalfTraits<half_t>::from_float(0.0f);
            if (c < tile_cols && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_kv + j_start + c) * D_CONST + d;
                K_tile[idx] = K[src]; V_tile[idx] = V[src];
                g_dK_tile[idx] = g_dK[src]; g_dV_tile[idx] = g_dV[src];
            } else {
                K_tile[idx] = zero; V_tile[idx] = zero;
                g_dK_tile[idx] = zero; g_dV_tile[idx] = zero;
            }
        }
        __syncthreads();

        // Recompute scores
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_L, S_R;
        v7_score_matmul_wide<half_t, D_CONST>(S_L, S_R, Q_warp, D_PAD, K_tile, D_PAD);
        v7_scale_accum(S_L, scale); v7_scale_accum(S_R, scale);
        v7_store_wide_score(S_w, SCORE_STRIDE, S_L, S_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gdS_L, gdS_R;
        v7_score_matmul_wide<half_t, D_CONST>(gdS_L, gdS_R, g_dQ_warp, D_PAD, K_tile, D_PAD);
        v7_score_matmul_wide_acc<half_t, D_CONST>(gdS_L, gdS_R, Q_warp, D_PAD, g_dK_tile, D_PAD);
        v7_scale_accum(gdS_L, scale); v7_scale_accum(gdS_R, scale);
        v7_store_wide_score(gdS_w, SCORE_STRIDE, gdS_L, gdS_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dP_L, dP_R;
        v7_score_matmul_wide<half_t, D_CONST>(dP_L, dP_R, dO_warp, D_PAD, V_tile, D_PAD);
        v7_store_wide_score(dP_w, SCORE_STRIDE, dP_L, dP_R);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gPdV_L, gPdV_R;
        v7_score_matmul_wide<half_t, D_CONST>(gPdV_L, gPdV_R, dO_warp, D_PAD, g_dV_tile, D_PAD);
        v7_store_wide_score(gPdV_w, SCORE_STRIDE, gPdV_L, gPdV_R);
        __syncwarp();

        // Element-wise: compute processed scores for output matmuls
        for (int r = 0; r < V7_WMMA_M; r++) {
            int qi = qi_base + warp_id * V7_WMMA_M + r;
            if (lane < tile_cols) {
                int si = r * SCORE_STRIDE + lane;
                float s = S_w[si];
                float l = L_vec[warp_id * V7_WMMA_M + r];
                int ki = ki_base + j_start + lane;
                float p = __expf(s - l);
                if (!v7_is_visible(qi, ki, window_size)) p = 0.0f;
                float gds = gdS_w[si];
                float dp = dP_w[si];
                float gpdv = gPdV_w[si];
                float d_val = Di_vec[warp_id * V7_WMMA_M + r];
                float dot2_val = dot2_smem[warp_id * V7_WMMA_M + r];
                float dot3_val = dot3_smem[warp_id * V7_WMMA_M + r];

                float ds = p * (dp - d_val);
                float g_dp = p * (gds - dot2_val);
                float g_P_soft = gds * (dp - d_val) - dp * dot2_val;
                float g_P = g_P_soft + gpdv;
                float g_S = p * (g_P - dot3_val);

                dP_w[si]    = scale * ds;       // for g_Q term 1
                extra_w[si] = scale * g_S;      // for g_Q term 2
                S_w[si]     = p;                // for g_dO term 1
                gPdV_w[si]  = g_dp;             // for g_dO term 2
            } else {
                int si = r * SCORE_STRIDE + lane;
                if (lane < V7_BC) {
                    dP_w[si] = 0.0f; extra_w[si] = 0.0f;
                    S_w[si] = 0.0f; gPdV_w[si] = 0.0f;
                }
            }
        }
        __syncwarp();

        // g_Q += (scale*dS) @ g_dK + (scale*g_S) @ K
        v7_output_matmul_acc<half_t, D_CONST>(g_Q_frags, dP_w, SCORE_STRIDE,
            hs_w, A_HSCORE_STRIDE, g_dK_tile, D_PAD, lane, warp_id);
        v7_output_matmul_acc<half_t, D_CONST>(g_Q_frags, extra_w, SCORE_STRIDE,
            hs_w, A_HSCORE_STRIDE, K_tile, D_PAD, lane, warp_id);
        // g_dO += P @ g_dV + g_dp @ V
        v7_output_matmul_acc<half_t, D_CONST>(g_dO_frags, S_w, SCORE_STRIDE,
            hs_w, A_HSCORE_STRIDE, g_dV_tile, D_PAD, lane, warp_id);
        v7_output_matmul_acc<half_t, D_CONST>(g_dO_frags, gPdV_w, SCORE_STRIDE,
            hs_w, A_HSCORE_STRIDE, V_tile, D_PAD, lane, warp_id);
    }

    // Store g_Q and g_dO
    __syncthreads();
    float* staging = reinterpret_cast<float*>(smem_raw);
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            g_Q_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < rows_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int r = idx / D_CONST; int d = idx % D_CONST;
        g_Q_out[((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[r * D_CONST + d]);
    }

    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            g_dO_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < rows_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int r = idx / D_CONST; int d = idx % D_CONST;
        g_dO_out[((b_idx * H + h_idx) * N_q + q_start + r) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[r * D_CONST + d]);
    }
}

// ===========================================================================
// V7 Double Backward Kernel B (col): 1-pass, computes g_K and g_V
// ===========================================================================
template<int D_CONST, typename half_t>
__global__ void v7_kernel_B_col(
    const half_t* __restrict__ Q,
    const half_t* __restrict__ K,
    const half_t* __restrict__ V,
    const half_t* __restrict__ dO,
    const half_t* __restrict__ g_dQ,
    const half_t* __restrict__ g_dK,
    const half_t* __restrict__ g_dV,
    const float* __restrict__ L,
    const float* __restrict__ D_vec_in,
    const float* __restrict__ dot2_in,
    const float* __restrict__ dot3_in,
    half_t* __restrict__ g_K_out,
    half_t* __restrict__ g_V_out,
    int B, int H, int N_q, int N_kv, int D, float scale,
    int window_size, int chunk_offset
) {
    extern __shared__ char smem_raw[];
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int B_SCORE_STRIDE = V7_WMMA_N + V7_SCORE_PAD;

    // Col data: K, V, g_dK, g_dV (BR rows each)
    half_t* K_block    = reinterpret_cast<half_t*>(smem_raw);
    half_t* V_block    = K_block    + V7_BR * D_PAD;
    half_t* g_dK_block = V_block    + V7_BR * D_PAD;
    half_t* g_dV_block = g_dK_block + V7_BR * D_PAD;
    // Row tiles: Q, dO, g_dQ (BC rows each)
    half_t* Q_tile     = g_dV_block + V7_BR * D_PAD;
    half_t* dO_tile    = Q_tile     + V7_BC * D_PAD;
    half_t* g_dQ_tile  = dO_tile    + V7_BC * D_PAD;
    // Score tiles directly after data tiles (no separate half_score allocation)
    char* after_data = reinterpret_cast<char*>(g_dQ_tile + V7_BC * D_PAD);
    after_data = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(after_data) + 15) & ~15ULL);
    float* score_base = reinterpret_cast<float*>(after_data);
    // 5 tall score tiles per warp
    float* tile_L    = score_base + V7_WARPS_PER_BLOCK * 5 * V7_BC * B_SCORE_STRIDE;
    float* tile_D    = tile_L + V7_BC;
    float* tile_dot2 = tile_D + V7_BC;
    float* tile_dot3 = tile_dot2 + V7_BC;

    int warp_id = threadIdx.x / V7_WARP_SIZE;
    int lane = threadIdx.x % V7_WARP_SIZE;

    int blocks_per_head_kv = (N_kv + V7_BR - 1) / V7_BR;
    int bh = blockIdx.x / blocks_per_head_kv;
    int block_idx = blockIdx.x % blocks_per_head_kv;
    int kv_start = block_idx * V7_BR;
    int b_idx = bh / H;
    int h_idx = bh % H;
    int ki_base = chunk_offset + N_q - N_kv;

    float* S_w     = score_base + warp_id * 5 * V7_BC * B_SCORE_STRIDE;
    float* gdS_w   = S_w     + V7_BC * B_SCORE_STRIDE;
    float* dP_w    = gdS_w   + V7_BC * B_SCORE_STRIDE;
    float* gPdV_w  = dP_w    + V7_BC * B_SCORE_STRIDE;
    float* extra_w = gPdV_w  + V7_BC * B_SCORE_STRIDE;
    // half_score aliases with gdS_w (free during output matmuls)
    const int B_HSCORE_STRIDE_ALIAS = B_SCORE_STRIDE * sizeof(float) / sizeof(half_t);
    half_t* hs_w   = reinterpret_cast<half_t*>(gdS_w);

    half_t* K_warp    = K_block    + warp_id * V7_WMMA_M * D_PAD;
    half_t* V_warp    = V_block    + warp_id * V7_WMMA_M * D_PAD;
    half_t* g_dK_warp = g_dK_block + warp_id * V7_WMMA_M * D_PAD;
    half_t* g_dV_warp = g_dV_block + warp_id * V7_WMMA_M * D_PAD;

    int cols_to_load = min(V7_BR, N_kv - kv_start);
    if (cols_to_load < 0) cols_to_load = 0;

    // Load col data
    for (int idx = threadIdx.x; idx < V7_BR * D_PAD; idx += V7_BLOCK_THREADS) {
        int c = idx / D_PAD; int d = idx % D_PAD;
        half_t zero = HalfTraits<half_t>::from_float(0.0f);
        if (c < cols_to_load && d < D_CONST) {
            int src = ((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d;
            K_block[idx] = K[src]; V_block[idx] = V[src];
            g_dK_block[idx] = g_dK[src]; g_dV_block[idx] = g_dV[src];
        } else {
            K_block[idx] = zero; V_block[idx] = zero;
            g_dK_block[idx] = zero; g_dV_block[idx] = zero;
        }
    }
    __syncthreads();

    // g_K, g_V accumulators
    constexpr int n_col_blocks = D_CONST / V7_WMMA_N;
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> g_K_frags[n_col_blocks];
    wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> g_V_frags[n_col_blocks];
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::fill_fragment(g_K_frags[cb], 0.0f);
        wmma::fill_fragment(g_V_frags[cb], 0.0f);
    }

    // Single pass over Q row tiles
    int ki_block_start = ki_base + kv_start;
    int ki_block_end = ki_block_start + cols_to_load;

    for (int i_start = 0; i_start < N_q; i_start += V7_BC) {
        int tile_rows = min(V7_BC, N_q - i_start);
        int qi_tile_start = chunk_offset + i_start;
        int mask_status = v7_tile_mask_status(qi_tile_start, qi_tile_start + tile_rows,
                                               ki_block_start, ki_block_end,
                                               window_size, chunk_offset);
        if (mask_status == 0) continue;

        __syncthreads();
        for (int idx = threadIdx.x; idx < V7_BC * D_PAD; idx += V7_BLOCK_THREADS) {
            int r = idx / D_PAD; int d = idx % D_PAD;
            half_t zero = HalfTraits<half_t>::from_float(0.0f);
            if (r < tile_rows && d < D_CONST) {
                int src = ((b_idx * H + h_idx) * N_q + i_start + r) * D_CONST + d;
                Q_tile[idx] = Q[src]; dO_tile[idx] = dO[src]; g_dQ_tile[idx] = g_dQ[src];
            } else {
                Q_tile[idx] = zero; dO_tile[idx] = zero; g_dQ_tile[idx] = zero;
            }
        }
        for (int idx = threadIdx.x; idx < V7_BC; idx += V7_BLOCK_THREADS) {
            if (idx < tile_rows) {
                int global_row = bh * N_q + i_start + idx;
                tile_L[idx] = L[global_row]; tile_D[idx] = D_vec_in[global_row];
                tile_dot2[idx] = dot2_in[global_row]; tile_dot3[idx] = dot3_in[global_row];
            } else {
                tile_L[idx] = 0.0f; tile_D[idx] = 0.0f;
                tile_dot2[idx] = 0.0f; tile_dot3[idx] = 0.0f;
            }
        }
        __syncthreads();

        // Tall score matmuls
        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> S_top, S_bot;
        v7_score_matmul_tall<half_t, D_CONST>(S_top, S_bot, Q_tile, D_PAD, K_warp, D_PAD);
        v7_scale_accum(S_top, scale); v7_scale_accum(S_bot, scale);
        v7_store_tall_score(S_w, B_SCORE_STRIDE, S_top, S_bot);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gdS_top, gdS_bot;
        v7_score_matmul_tall<half_t, D_CONST>(gdS_top, gdS_bot, g_dQ_tile, D_PAD, K_warp, D_PAD);
        v7_score_matmul_tall_acc<half_t, D_CONST>(gdS_top, gdS_bot, Q_tile, D_PAD, g_dK_warp, D_PAD);
        v7_scale_accum(gdS_top, scale); v7_scale_accum(gdS_bot, scale);
        v7_store_tall_score(gdS_w, B_SCORE_STRIDE, gdS_top, gdS_bot);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> dP_top, dP_bot;
        v7_score_matmul_tall<half_t, D_CONST>(dP_top, dP_bot, dO_tile, D_PAD, V_warp, D_PAD);
        v7_store_tall_score(dP_w, B_SCORE_STRIDE, dP_top, dP_bot);

        wmma::fragment<wmma::accumulator, V7_WMMA_M, V7_WMMA_N, V7_WMMA_K, float> gPdV_top, gPdV_bot;
        v7_score_matmul_tall<half_t, D_CONST>(gPdV_top, gPdV_bot, dO_tile, D_PAD, g_dV_warp, D_PAD);
        v7_store_tall_score(gPdV_w, B_SCORE_STRIDE, gPdV_top, gPdV_bot);
        __syncwarp();

        // Element-wise (32×16)
        for (int idx = lane; idx < V7_BC * V7_WMMA_N; idx += V7_WARP_SIZE) {
            int i_local = idx / V7_WMMA_N;
            int j_local = idx % V7_WMMA_N;
            int si = i_local * B_SCORE_STRIDE + j_local;

            int qi = chunk_offset + i_start + i_local;
            int ki = ki_base + kv_start + warp_id * V7_WMMA_M + j_local;

            float s = S_w[si];
            float l = tile_L[i_local];
            float p = (i_local < tile_rows) ? __expf(s - l) : 0.0f;
            if (!v7_is_visible(qi, ki, window_size)) p = 0.0f;
            float gds = gdS_w[si];
            float dp = dP_w[si];
            float gpdv = gPdV_w[si];
            float d_val = tile_D[i_local];
            float dot2_val = tile_dot2[i_local];
            float dot3_val = tile_dot3[i_local];

            float ds = p * (dp - d_val);
            float g_dp = p * (gds - dot2_val);
            float g_P_soft = gds * (dp - d_val) - dp * dot2_val;
            float g_P = g_P_soft + gpdv;
            float g_S = p * (g_P - dot3_val);

            dP_w[si]    = scale * ds;
            extra_w[si] = scale * g_S;
            gPdV_w[si]  = g_dp;
        }
        __syncwarp();

        // g_K += (scale*dS)^T @ g_dQ + (scale*g_S)^T @ Q
        v7_output_matmul_acc_tallT<half_t, D_CONST>(g_K_frags, dP_w, B_SCORE_STRIDE,
            hs_w, B_HSCORE_STRIDE_ALIAS, g_dQ_tile, D_PAD, lane, warp_id);
        v7_output_matmul_acc_tallT<half_t, D_CONST>(g_K_frags, extra_w, B_SCORE_STRIDE,
            hs_w, B_HSCORE_STRIDE_ALIAS, Q_tile, D_PAD, lane, warp_id);
        // g_V += g_dp^T @ dO
        v7_output_matmul_acc_tallT<half_t, D_CONST>(g_V_frags, gPdV_w, B_SCORE_STRIDE,
            hs_w, B_HSCORE_STRIDE_ALIAS, dO_tile, D_PAD, lane, warp_id);
    }

    // Store g_K and g_V
    __syncthreads();
    float* staging = reinterpret_cast<float*>(smem_raw);
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            g_K_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < cols_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int c = idx / D_CONST; int d = idx % D_CONST;
        g_K_out[((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[c * D_CONST + d]);
    }

    __syncthreads();
    for (int cb = 0; cb < n_col_blocks; cb++) {
        wmma::store_matrix_sync(staging + warp_id * V7_WMMA_M * D_CONST + cb * V7_WMMA_N,
            g_V_frags[cb], D_CONST, wmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < cols_to_load * D_CONST; idx += V7_BLOCK_THREADS) {
        int c = idx / D_CONST; int d = idx % D_CONST;
        g_V_out[((b_idx * H + h_idx) * N_kv + kv_start + c) * D_CONST + d] =
            HalfTraits<half_t>::from_float(staging[c * D_CONST + d]);
    }
}

// ===========================================================================
// Host wrapper: V7 double backward
// ===========================================================================
template<int D_CONST, typename half_t>
static std::vector<at::Tensor> v7_double_backward_impl(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor O, at::Tensor L,
    int window_size, int chunk_id
) {
    int B = Q.size(0), H = Q.size(1), N_q = Q.size(2), D = Q.size(3);
    int N_kv = K.size(2);
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    int chunk_offset = chunk_id * N_q;
    int BH = B * H;
    const int D_PAD = D_CONST + V7_SMEM_PAD;
    const int SCORE_STRIDE = V7_BC + V7_SCORE_PAD;
    const int B_SCORE_STRIDE = V7_WMMA_N + V7_SCORE_PAD;

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(at::kFloat);

    // Kernel A: blocks over Q rows
    int blocks_per_head_q = (N_q + V7_BR - 1) / V7_BR;
    int grid_A = BH * blocks_per_head_q;

    // half_score aliased into gdS_w score tile: no separate allocation needed
    int smem_A =
        (3 * V7_BR * D_PAD + 4 * V7_BC * D_PAD) * sizeof(half_t)
        + 16  // alignment
        + V7_WARPS_PER_BLOCK * 5 * V7_WMMA_M * SCORE_STRIDE * sizeof(float)
        + 4 * V7_BR * sizeof(float);
    int o_staging_A = V7_BR * D_CONST * sizeof(float);
    smem_A = max(smem_A, o_staging_A);

    // Kernel B: blocks over KV cols
    int blocks_per_head_kv = (N_kv + V7_BR - 1) / V7_BR;
    int grid_B = BH * blocks_per_head_kv;

    int smem_B =
        (4 * V7_BR * D_PAD + 3 * V7_BC * D_PAD) * sizeof(half_t)
        + 16  // alignment
        + V7_WARPS_PER_BLOCK * 5 * V7_BC * B_SCORE_STRIDE * sizeof(float)
        + 4 * V7_BC * sizeof(float);
    int o_staging_B = V7_BR * D_CONST * sizeof(float);
    smem_B = max(smem_B, o_staging_B);

    // Check shared memory limits
    int device;
    V7_CUDA_CHECK(cudaGetDevice(&device));
    int max_smem;
    V7_CUDA_CHECK(cudaDeviceGetAttribute(&max_smem,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    TORCH_CHECK(smem_A <= max_smem && smem_B <= max_smem,
        "V7 double backward needs ", max(smem_A, smem_B) / 1024,
        " KB smem but GPU has ", max_smem / 1024, " KB");

    if (smem_A > 48 * 1024) {
        V7_CUDA_CHECK(cudaFuncSetAttribute(v7_kernel_A_row<D_CONST, half_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_A));
    }
    if (smem_B > 48 * 1024) {
        V7_CUDA_CHECK(cudaFuncSetAttribute(v7_kernel_B_col<D_CONST, half_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_B));
    }

    auto D_vec = at::empty({B, H, N_q}, opts_float);
    auto dot2  = at::empty({B, H, N_q}, opts_float);
    auto dot3  = at::empty({B, H, N_q}, opts_float);
    auto g_Q   = at::zeros({B, H, N_q, D}, opts_half);
    auto g_dO  = at::zeros({B, H, N_q, D}, opts_half);
    auto g_K   = at::zeros({B, H, N_kv, D}, opts_half);
    auto g_V   = at::zeros({B, H, N_kv, D}, opts_half);

    v7_kernel_A_row<D_CONST, half_t><<<grid_A, V7_BLOCK_THREADS, smem_A>>>(
        reinterpret_cast<const half_t*>(Q.data_ptr()),
        reinterpret_cast<const half_t*>(K.data_ptr()),
        reinterpret_cast<const half_t*>(V.data_ptr()),
        reinterpret_cast<const half_t*>(dO.data_ptr()),
        reinterpret_cast<const half_t*>(O.data_ptr()),
        reinterpret_cast<const half_t*>(g_dQ.data_ptr()),
        reinterpret_cast<const half_t*>(g_dK.data_ptr()),
        reinterpret_cast<const half_t*>(g_dV.data_ptr()),
        L.data_ptr<float>(),
        D_vec.data_ptr<float>(), dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        reinterpret_cast<half_t*>(g_Q.data_ptr()),
        reinterpret_cast<half_t*>(g_dO.data_ptr()),
        B, H, N_q, N_kv, D, scale, window_size, chunk_offset);
    V7_CUDA_CHECK(cudaGetLastError());

    v7_kernel_B_col<D_CONST, half_t><<<grid_B, V7_BLOCK_THREADS, smem_B>>>(
        reinterpret_cast<const half_t*>(Q.data_ptr()),
        reinterpret_cast<const half_t*>(K.data_ptr()),
        reinterpret_cast<const half_t*>(V.data_ptr()),
        reinterpret_cast<const half_t*>(dO.data_ptr()),
        reinterpret_cast<const half_t*>(g_dQ.data_ptr()),
        reinterpret_cast<const half_t*>(g_dK.data_ptr()),
        reinterpret_cast<const half_t*>(g_dV.data_ptr()),
        L.data_ptr<float>(), D_vec.data_ptr<float>(),
        dot2.data_ptr<float>(), dot3.data_ptr<float>(),
        reinterpret_cast<half_t*>(g_K.data_ptr()),
        reinterpret_cast<half_t*>(g_V.data_ptr()),
        B, H, N_q, N_kv, D, scale, window_size, chunk_offset);
    V7_CUDA_CHECK(cudaGetLastError());

    return {g_dO, g_Q, g_K, g_V};
}

// ===========================================================================
// Public: v7_double_backward_cuda
// ===========================================================================
std::vector<at::Tensor> v7_double_backward_cuda(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor O, at::Tensor L,
    int window_size, int chunk_id
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    g_dQ = g_dQ.contiguous(); g_dK = g_dK.contiguous(); g_dV = g_dV.contiguous();
    dO = dO.contiguous(); Q = Q.contiguous(); K = K.contiguous();
    V = V.contiguous(); O = O.contiguous(); L = L.contiguous();

    int D = Q.size(3);
    bool is_bf16 = (Q.dtype() == at::kBFloat16);
    TORCH_CHECK(is_bf16 || Q.dtype() == at::kHalf, "Q must be float16 or bfloat16");
    TORCH_CHECK(D % 16 == 0, "D must be multiple of 16");

    if (is_bf16) {
        if (D == 64) return v7_double_backward_impl<64, __nv_bfloat16>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
        if (D == 80) return v7_double_backward_impl<80, __nv_bfloat16>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
        if (D == 96) return v7_double_backward_impl<96, __nv_bfloat16>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
    } else {
        if (D == 64) return v7_double_backward_impl<64, __half>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
        if (D == 80) return v7_double_backward_impl<80, __half>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
        if (D == 96) return v7_double_backward_impl<96, __half>(g_dQ,g_dK,g_dV,dO,Q,K,V,O,L,window_size,chunk_id);
    }
    TORCH_CHECK(false, "Unsupported D=", D);
    return {};
}
