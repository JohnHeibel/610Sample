#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

#include "common.cuh"

namespace flash_v9 {

using namespace cute;

// =============================================================================
// Algorithm 2 (backward).
//
// Inputs:  dO[B,H,N,D], Q,K,V[B,H,N,D], O[B,H,N,D], L[B,H,N]
// Outputs: dQ,dK,dV[B,H,N,D]
//
// Schedule: single main kernel, outer loop over kv-blocks, inner loop over
// q-blocks. dV / dK accumulated in registers and written once per kv-block.
// dQ accumulated via atomicAdd on a fp32 staging tensor (cast to the input
// dtype in a separate postprocess kernel).
//
// Per-row scalar D_i = sum_j dO_{i,j} * O_{i,j} is computed by a small
// preprocess kernel and stored in a [B,H,N] fp32 tensor used by the main
// kernel.
//
// References:
//   third_party/flash-attention/hopper/mainloop_bwd_sm80.hpp
//   FlashAttention-2 paper, Algorithm 2.
// =============================================================================

// -----------------------------------------------------------------------------
// Preprocess: D = rowsum(dO * O), per (b,h,row).
// One thread per row; threads handle Headdim contiguous dot product.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim>
__global__ void flash_v9_bwd_preprocess_kernel(
    const Dtype* __restrict__ dO,   // [B,H,N,D]
    const Dtype* __restrict__ O,    // [B,H,N,D]
    float*       __restrict__ D,    // [B,H,N]  (fp32 output)
    int B, int H, int N,
    int64_t bh_stride,        // = H * N * D for [B,H,N,D] -> bh = b*H+h
    int64_t row_stride_qkv,   // typically D
    int64_t l_bh_stride       // = N for [B,H,N]
) {
    const int bh  = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const Dtype* dO_row = dO + bh * bh_stride + row * row_stride_qkv;
    const Dtype* O_row  = O  + bh * bh_stride + row * row_stride_qkv;
    float acc = 0.f;
    CUTE_UNROLL
    for (int d = 0; d < Headdim; ++d) {
        acc += float(dO_row[d]) * float(O_row[d]);
    }
    D[bh * l_bh_stride + row] = acc;
}

// -----------------------------------------------------------------------------
// Main backward kernel.
//
// Outer loop over kv blocks. For each kv block:
//   (1) Load K[kv], V[kv] into smem.
//   (2) Initialize dK_acc, dV_acc fp32 register accumulators to 0.
//   (3) Inner loop over q blocks (covers full N for non-causal; for causal
//       skips q-blocks above the diagonal):
//       (a) Load Q[q], dO[q] into smem; load L[q], D[q] into per-row regs.
//       (b) Recompute S = (Q.K^T) * scale, apply causal mask.
//       (c) P = exp(S - L) (row-wise broadcast).
//       (d) dV_acc += P^T . dO
//       (e) dP = dO . V^T
//       (f) dS = P * (dP - D) * scale
//       (g) dK_acc += dS^T . Q
//       (h) Compute partial dQ = dS . K and atomicAdd into dQ_fp32[q].
//   (4) Write dK_acc, dV_acc back to gmem (one bf16 store per kv block).
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_bwd_kernel(
    const Dtype* __restrict__ Q_ptr,   // [B,H,N,D]
    const Dtype* __restrict__ K_ptr,   // [B,H,N,D]
    const Dtype* __restrict__ V_ptr,   // [B,H,N,D]
    const Dtype* __restrict__ O_ptr,   // [B,H,N,D]
    const Dtype* __restrict__ dO_ptr,  // [B,H,N,D]
    const float* __restrict__ L_ptr,   // [B,H,N]
    const float* __restrict__ D_ptr,   // [B,H,N]
    float*       __restrict__ dQ_ptr,  // [B,H,N,D] (fp32 staging)
    Dtype*       __restrict__ dK_ptr,  // [B,H,N,D]
    Dtype*       __restrict__ dV_ptr,  // [B,H,N,D]
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale
) {
    constexpr int NumThreads = 128;
    constexpr int NumWarps   = NumThreads / 32;

    using SmemQ_t = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemK_t = SmemLayoutK<Bc, Headdim, Dtype>;
    using SmemV_t = SmemLayoutV<Bc, Headdim, Dtype>;
    using GmemCopy_t =
        typename GmemTiledCopyTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;

    extern __shared__ char smem_buf[];
    Dtype* sQ_data  = reinterpret_cast<Dtype*>(smem_buf);
    Dtype* sK_data  = sQ_data  + cosize_v<SmemQ_t>;
    Dtype* sV_data  = sK_data  + cosize_v<SmemK_t>;
    Dtype* sdO_data = sV_data  + cosize_v<SmemV_t>;

    auto sQ  = make_tensor(make_smem_ptr(sQ_data),  SmemQ_t{});
    auto sK  = make_tensor(make_smem_ptr(sK_data),  SmemK_t{});
    auto sV  = make_tensor(make_smem_ptr(sV_data),  SmemV_t{});
    auto sdO = make_tensor(make_smem_ptr(sdO_data), SmemQ_t{});  // dO has Q-shape

    // sVt and sKt transposed views needed for B-operand of various MMAs.
    auto sVt = make_tensor(make_smem_ptr(sV_data),
                           SmemLayoutVt<Bc, Headdim, Dtype>{});
    auto sKt = make_tensor(make_smem_ptr(sK_data),
                           SmemLayoutVt<Bc, Headdim, Dtype>{});

    const int bh = blockIdx.y;
    const int b  = bh / H;
    const int h  = bh % H;
    const int kv_block = blockIdx.x;

    const Dtype* Q_bh  = Q_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* K_bh  = K_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* V_bh  = V_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* dO_bh = dO_ptr + b * qkv_b_stride + h * qkv_h_stride;
          float* dQ_bh = dQ_ptr + b * qkv_b_stride + h * qkv_h_stride;
          Dtype* dK_bh = dK_ptr + b * qkv_b_stride + h * qkv_h_stride;
          Dtype* dV_bh = dV_ptr + b * qkv_b_stride + h * qkv_h_stride;
    const float* L_bh  = L_ptr  + b * l_b_stride   + h * l_h_stride;
    const float* D_bh  = D_ptr  + b * l_b_stride   + h * l_h_stride;

    auto gQ_head  = make_tensor(make_gmem_ptr(Q_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gK_head  = make_tensor(make_gmem_ptr(K_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gV_head  = make_tensor(make_gmem_ptr(V_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gdO_head = make_tensor(make_gmem_ptr(dO_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));

    auto gQ_tiles  = local_tile(gQ_head,  Shape<Int<Br>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gK_tiles  = local_tile(gK_head,  Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gV_tiles  = local_tile(gV_head,  Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gdO_tiles = local_tile(gdO_head, Shape<Int<Br>, Int<Headdim>>{}, make_coord(_, _0{}));

    const int num_q_blocks = size<2>(gQ_tiles);

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    // -------------------------------------------------------------------------
    // (1) Load K, V for this kv block (persists across the inner q loop).
    // -------------------------------------------------------------------------
    {
        auto gK = gK_tiles(_, _, kv_block);
        auto gV = gV_tiles(_, _, kv_block);
        copy(gmem_copy_qkv, thr_copy.partition_S(gK), thr_copy.partition_D(sK));
        copy(gmem_copy_qkv, thr_copy.partition_S(gV), thr_copy.partition_D(sV));
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // (2) Allocate dK_acc, dV_acc fp32 register accumulators (Bc x Headdim).
    // The TiledMma is the same as fwd but with M-dim = Bc instead of Br.
    // -------------------------------------------------------------------------
    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // dV_acc and dK_acc are (Bc, Headdim) fp32 accumulators. Treated as the
    // C output of an MMA whose M-dim is Bc.
    auto rdV = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    auto rdK = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    clear(rdV);
    clear(rdK);

    // Per-thread row indices used for atomic dQ writes and L/D loads.
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    // Precompute scale_log2 used for the P = exp(S * scale - L) trick.
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

    // Smem copy atoms reused across the inner q loop.
    using SmemCopyAtom_QK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    using SmemCopyAtom_VK_T = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;
    auto smem_tiled_copy_Q  = make_tiled_copy_A(SmemCopyAtom_QK{},   tiled_mma);
    auto smem_tiled_copy_K  = make_tiled_copy_B(SmemCopyAtom_QK{},   tiled_mma);
    auto smem_tiled_copy_V  = make_tiled_copy_B(SmemCopyAtom_VK_T{}, tiled_mma);
    auto smem_tiled_copy_dO = make_tiled_copy_A(SmemCopyAtom_QK{},   tiled_mma);
    auto smem_thr_copy_Q  = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K  = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_V  = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_dO = smem_tiled_copy_dO.get_thread_slice(threadIdx.x);

    // -------------------------------------------------------------------------
    // (3) Inner loop over q blocks.
    // -------------------------------------------------------------------------
    for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
        // Causal: skip q blocks fully above the diagonal of this kv block,
        // i.e. q < kv (kv * Bc > q * Br + Br - 1 simplifies for Br=Bc).
        if constexpr (IsCausal) {
            if (q_block * Br + (Br - 1) < kv_block * Bc) continue;
        }

        // (a) Load Q[q], dO[q] into smem.
        {
            auto gQ  = gQ_tiles(_, _, q_block);
            auto gdO = gdO_tiles(_, _, q_block);
            copy(gmem_copy_qkv, thr_copy.partition_S(gQ),  thr_copy.partition_D(sQ));
            copy(gmem_copy_qkv, thr_copy.partition_S(gdO), thr_copy.partition_D(sdO));
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();
        }

        // (b) Recompute S = Q.K^T (no scale yet -- absorbed into exp2 below).
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rS);
        {
            auto rQ = thr_mma.partition_fragment_A(sQ);
            auto rK = thr_mma.partition_fragment_B(sK);
            auto tSsQ      = smem_thr_copy_Q.partition_S(sQ);
            auto tSsK      = smem_thr_copy_K.partition_S(sK);
            auto tSrQ_view = smem_thr_copy_Q.retile_D(rQ);
            auto tSrK_view = smem_thr_copy_K.retile_D(rK);
            CUTE_UNROLL
            for (int k = 0; k < size<2>(rQ); ++k) {
                copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_view(_, _, k));
                copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_view(_, _, k));
                gemm(tiled_mma, rQ(_, _, k), rK(_, _, k), rS);
            }
        }

        // Causal mask on the (Br x Bc) S tile.
        if constexpr (IsCausal) {
            auto cS = make_identity_tensor(Shape<Int<Br>, Int<Bc>>{});
            auto tScS    = thr_mma.partition_C(cS);
            auto tScS_rc = make_tensor(tScS.data(), convert_layout_acc_rowcol(tScS.layout()));
            auto rS_rc   = make_tensor(rS.data(),   convert_layout_acc_rowcol(rS.layout()));
            const int row_off = q_block * Br;
            const int col_off = kv_block * Bc;
            CUTE_UNROLL
            for (int m = 0; m < size<0>(rS_rc); ++m) {
                const int q_idx = row_off + get<0>(tScS_rc(m, _0{}));
                CUTE_UNROLL
                for (int n = 0; n < size<1>(rS_rc); ++n) {
                    const int k_idx = col_off + get<1>(tScS_rc(_0{}, n));
                    if (k_idx > q_idx) rS_rc(m, n) = -INFINITY;
                }
            }
        }

        // (c) P = exp2(S * scale_log2 - L_i * scale_log2)
        //   = exp(S * scale - L_i)
        // Load L[q] for the rows owned by this thread.
        float L_thread[2];
        {
            const int row_lo = warp_id * 16 + (lane_id >> 2);
            const int row_hi = row_lo + 8;
            const int q_row_lo = q_block * Br + row_lo;
            const int q_row_hi = q_block * Br + row_hi;
            L_thread[0] = (q_row_lo < N) ? L_bh[q_row_lo] : 0.f;
            L_thread[1] = (q_row_hi < N) ? L_bh[q_row_hi] : 0.f;
        }
        {
            auto rS_rc = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));
            CUTE_UNROLL
            for (int m = 0; m < size<0>(rS_rc); ++m) {
                const float L_scaled = L_thread[m] * softmax_scale_log2;
                CUTE_UNROLL
                for (int n = 0; n < size<1>(rS_rc); ++n) {
                    rS_rc(m, n) = exp2f(rS_rc(m, n) * softmax_scale_log2 - L_scaled);
                }
            }
        }
        // rS now holds P.

        // (d) dV_acc += P^T . dO
        // For the MMA: A=P^T (Bc x Br), B=dO (Br x Headdim), C=dV (Bc x Headdim).
        // Trick: since P is in registers (the C output of Q.K^T), we view it as
        // operand A (after convert_layout_acc_Aregs). But for P^T we'd need a
        // transposed view. FA's approach: store P to smem, then load smem->reg
        // partitioned as B for the dV MMA. Or equivalently: treat the (P^T . dO)
        // GEMM with operand-A-from-smem schedule.
        //
        // For simplicity, we materialize P in smem and load it back. Reuses the
        // sQ smem region (Q is no longer needed for this q iteration after S
        // was computed -- actually sQ is needed below for the dK accumulation,
        // so we use the sdO smem region instead, and load dO from gmem again
        // when needed). Hmm, sdO is also needed for dV.
        //
        // Cleanest: alloc separate sP smem of size (Br x Bc) bf16 = 8KB.
        // For now, take the simple route: store P in regs, transpose via
        // a series of shuffles. Actually, the ldmatrix patterns can do this.
        //
        // Punting on the perf-optimal path: serialize through a fp32 sP smem
        // staging area (Br x Bc * 4 = 16KB). This adds 16KB to smem.
        // (TODO 4h: register-to-register transpose to avoid sP staging.)

        // For now, do dV and dK GEMMs by partial unrolling using the rS rowcol view.
        // We approximate the gradient computation by skipping the full PT.dO
        // and dS^T.Q GEMMs and instead doing them via thread-local computation
        // restricted to the per-thread (rows, cols) of rS.
        //
        // That is: each thread holds a (rows_per_thread x cols_per_thread)
        // chunk of P. For dV[kv_row_in_block, d] = sum_q P[q, kv_row_in_block].dO[q, d],
        // we need contributions across all (q, kv_row) pairs in this tile.
        // A thread's contribution: sum over the q-rows it owns of P[q, kv]*dO[q, d]
        // for the kv columns it owns. After atomic adds across warps for the
        // kv-row dimension, we get dV.
        //
        // This is much slower than the canonical CuTe MMA approach, but it's
        // correct and unblocks dbl_bwd. Optimize in 4h.
        //
        // NOTE: the canonical MMA path requires sP staging in smem. Doing that
        // properly requires writing rS to smem, syncing, then loading back as
        // operand A or B for the next MMA. This is the TODO referenced above.
        //
        // STUB: this naive approach is too slow and not actually implemented.
        // Proper approach below uses sP smem.

        // (DEFERRED to 4b proper: sP smem staging + canonical MMAs.)
        // For 4a we just leave dV / dK / dQ at zero (which fails correctness
        // but verifies kernel launches without crashing).
        (void)rS;  // silence unused warning
    }

    // -------------------------------------------------------------------------
    // (4) Write dK_acc, dV_acc to gmem (one bf16 store per kv block).
    // For 4a: rdV and rdK are still zero, so this writes zeros.
    // -------------------------------------------------------------------------
    Tensor rdV_out = make_tensor_like<Dtype>(rdV);
    Tensor rdK_out = make_tensor_like<Dtype>(rdK);
    convert_type_out(rdV, rdV_out);
    convert_type_out(rdK, rdK_out);

    __syncthreads();
    auto sdV_smem = make_tensor(make_smem_ptr(sQ_data), SmemK_t{});
    using SmemCopyAtom_dKV = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtom_dKV{}, tiled_mma);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(threadIdx.x);

    {
        auto t = smem_thr_copy_dKV.partition_D(sdV_smem);
        auto v = smem_thr_copy_dKV.retile_S(rdV_out);
        copy(smem_tiled_copy_dKV, v, t);
    }
    __syncthreads();
    {
        auto gdV_head = make_tensor(make_gmem_ptr(dV_bh),
                                    make_shape(N, Int<Headdim>{}),
                                    make_stride(Int<Headdim>{}, _1{}));
        auto gdV_tiles = local_tile(gdV_head, Shape<Int<Bc>, Int<Headdim>>{},
                                    make_coord(_, _0{}));
        auto gdV = gdV_tiles(_, _, kv_block);
        using GmemTiledCopyO_t =
            typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
        GmemTiledCopyO_t gmem_copy_o;
        auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);
        copy(gmem_copy_o, thr_copy_o.partition_S(sdV_smem), thr_copy_o.partition_D(gdV));
    }
    __syncthreads();
    {
        auto t = smem_thr_copy_dKV.partition_D(sdV_smem);
        auto v = smem_thr_copy_dKV.retile_S(rdK_out);
        copy(smem_tiled_copy_dKV, v, t);
    }
    __syncthreads();
    {
        auto gdK_head = make_tensor(make_gmem_ptr(dK_bh),
                                    make_shape(N, Int<Headdim>{}),
                                    make_stride(Int<Headdim>{}, _1{}));
        auto gdK_tiles = local_tile(gdK_head, Shape<Int<Bc>, Int<Headdim>>{},
                                    make_coord(_, _0{}));
        auto gdK = gdK_tiles(_, _, kv_block);
        using GmemTiledCopyO_t =
            typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
        GmemTiledCopyO_t gmem_copy_o;
        auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);
        copy(gmem_copy_o, thr_copy_o.partition_S(sdV_smem), thr_copy_o.partition_D(gdK));
    }

    (void)dQ_bh; (void)D_bh;  // unused in 4a
}

// -----------------------------------------------------------------------------
// Postprocess: cast dQ from fp32 staging to the input dtype.
// -----------------------------------------------------------------------------
template <typename Dtype>
__global__ void flash_v9_bwd_postprocess_kernel(
    const float* __restrict__ dQ_fp32,
    Dtype*       __restrict__ dQ_out,
    int total_elems
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_elems) {
        dQ_out[i] = Dtype(dQ_fp32[i]);
    }
}

// -----------------------------------------------------------------------------
// Launch helpers.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, bool IsCausal>
static void launch_bwd(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    const Dtype* O, const Dtype* dO,
    const float* L, const float* D,
    float* dQ_fp32, Dtype* dK, Dtype* dV,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale, cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kNumThreads = 128;

    // Smem: Q + K + V + dO   (sQ also reused as sdV / sdK staging at end)
    constexpr int sQ_elems  = cosize_v<SmemLayoutQ<Br, Headdim, Dtype>>;
    constexpr int sK_elems  = cosize_v<SmemLayoutK<Bc, Headdim, Dtype>>;
    constexpr int sV_elems  = cosize_v<SmemLayoutV<Bc, Headdim, Dtype>>;
    constexpr int sdO_elems = sQ_elems;
    constexpr int smem_bytes = (sQ_elems + sK_elems + sV_elems + sdO_elems) * sizeof(Dtype);

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    dim3 grid(num_kv_blocks, B * H);
    dim3 block(kNumThreads);

    flash_v9_bwd_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, dO, L, D, dQ_fp32, dK, dV,
            B, H, N,
            qkv_b_stride, qkv_h_stride,
            l_b_stride,   l_h_stride,
            softmax_scale
        );
}

template <typename Dtype>
static void dispatch_bwd_headdim(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    const Dtype* O, const Dtype* dO,
    const float* L, const float* D,
    float* dQ_fp32, Dtype* dK, Dtype* dV,
    int B, int H, int N, int Hd,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    bool is_causal, float softmax_scale, cudaStream_t stream
) {
    #define LAUNCH(HD, CAUSAL) \
        launch_bwd<Dtype, HD, CAUSAL>( \
            Q, K, V, O, dO, L, D, dQ_fp32, dK, dV, \
            B, H, N, qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride, \
            softmax_scale, stream)
    if (Hd == 64) {
        if (is_causal) LAUNCH(64, true);  else LAUNCH(64, false);
    } else if (Hd == 128) {
        if (is_causal) LAUNCH(128, true); else LAUNCH(128, false);
    } else {
        TORCH_CHECK(false, "flash_v9 bwd: headdim must be 64 or 128");
    }
    #undef LAUNCH
}

template <typename Dtype, int Headdim>
static void launch_preprocess(
    const Dtype* dO, const Dtype* O, float* D,
    int B, int H, int N,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride,
    cudaStream_t stream
) {
    const int threads = 128;
    const int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    dim3 block(threads);
    flash_v9_bwd_preprocess_kernel<Dtype, Headdim>
        <<<grid, block, 0, stream>>>(
            dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride
        );
}

template <typename Dtype>
static void dispatch_preprocess_headdim(
    const Dtype* dO, const Dtype* O, float* D,
    int B, int H, int N, int Hd,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride,
    cudaStream_t stream
) {
    if      (Hd == 64)  launch_preprocess<Dtype, 64>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else if (Hd == 128) launch_preprocess<Dtype, 128>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else TORCH_CHECK(false, "flash_v9 preprocess: headdim must be 64 or 128");
}

} // namespace flash_v9


std::vector<torch::Tensor> flash_v9_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    bool is_causal, double softmax_scale
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q, K, V must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.size(2) % 64 == 0,
                "flash_v9 bwd: N must be a multiple of 64");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D = Q.size(3);

    auto dQ_fp32 = torch::zeros_like(Q, Q.options().dtype(torch::kFloat32));
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    auto D_tensor = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (Q.scalar_type() == torch::kBFloat16) {
        using T = cutlass::bfloat16_t;
        const T* Q_p  = reinterpret_cast<const T*>(Q.data_ptr());
        const T* K_p  = reinterpret_cast<const T*>(K.data_ptr());
        const T* V_p  = reinterpret_cast<const T*>(V.data_ptr());
        const T* O_p  = reinterpret_cast<const T*>(O.data_ptr());
        const T* dO_p = reinterpret_cast<const T*>(dO.data_ptr());
        T* dK_p = reinterpret_cast<T*>(dK.data_ptr());
        T* dV_p = reinterpret_cast<T*>(dV.data_ptr());

        flash_v9::dispatch_preprocess_headdim<T>(
            dO_p, O_p, D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(1), Q.stride(2), N, stream
        );
        flash_v9::dispatch_bwd_headdim<T>(
            Q_p, K_p, V_p, O_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(),
            dQ_fp32.data_ptr<float>(), dK_p, dV_p,
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1),
            is_causal, (float)softmax_scale, stream
        );
    } else if (Q.scalar_type() == torch::kHalf) {
        using T = cutlass::half_t;
        const T* Q_p  = reinterpret_cast<const T*>(Q.data_ptr());
        const T* K_p  = reinterpret_cast<const T*>(K.data_ptr());
        const T* V_p  = reinterpret_cast<const T*>(V.data_ptr());
        const T* O_p  = reinterpret_cast<const T*>(O.data_ptr());
        const T* dO_p = reinterpret_cast<const T*>(dO.data_ptr());
        T* dK_p = reinterpret_cast<T*>(dK.data_ptr());
        T* dV_p = reinterpret_cast<T*>(dV.data_ptr());

        flash_v9::dispatch_preprocess_headdim<T>(
            dO_p, O_p, D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(1), Q.stride(2), N, stream
        );
        flash_v9::dispatch_bwd_headdim<T>(
            Q_p, K_p, V_p, O_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(),
            dQ_fp32.data_ptr<float>(), dK_p, dV_p,
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1),
            is_causal, (float)softmax_scale, stream
        );
    } else {
        TORCH_CHECK(false, "flash_v9 bwd: only bf16 and fp16 supported");
    }

    // Cast dQ_fp32 -> dQ_dtype.
    auto dQ = torch::empty_like(Q);
    const int total = B * H * N * D;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;
    if (Q.scalar_type() == torch::kBFloat16) {
        flash_v9::flash_v9_bwd_postprocess_kernel<cutlass::bfloat16_t>
            <<<blocks, threads, 0, stream>>>(
                dQ_fp32.data_ptr<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(dQ.data_ptr()),
                total
            );
    } else {
        flash_v9::flash_v9_bwd_postprocess_kernel<cutlass::half_t>
            <<<blocks, threads, 0, stream>>>(
                dQ_fp32.data_ptr<float>(),
                reinterpret_cast<cutlass::half_t*>(dQ.data_ptr()),
                total
            );
    }

    return {dQ, dK, dV};
}
