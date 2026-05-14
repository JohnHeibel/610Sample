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
// Algorithm 2 (backward) -- v8-style two-kernel split.
//
//   bwd_preprocess   D[b,h,row] = sum_d dO * O                  per-row kernel
//   bwd_dQ           outer Q, inner KV.  Writes dQ.             (this commit)
//   bwd_dKV          outer KV, inner Q.  Writes dK, dV.          (commit 6c)
//
// No atomics; each output is exclusive to one kernel.
//
// References:
//   third_party/flash-attention/hopper/mainloop_bwd_sm80.hpp (FA2 reference)
//   v8: git show v8:csrc/flash_double_backward_v8.cu          (v8 proven shape)
// =============================================================================

// -----------------------------------------------------------------------------
// Preprocess: D = rowsum(dO * O), per (b, h, row). One thread per row.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim>
__global__ void flash_v9_bwd_preprocess_kernel(
    const Dtype* __restrict__ dO,
    const Dtype* __restrict__ O,
    float*       __restrict__ D,
    int B, int H, int N,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride
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
// bwd_dQ kernel: outer Q, inner KV. Accumulates dQ in registers and writes
// once at the end of the kernel. No atomics.
//
// Per inner KV step:
//   (a) cp.async load K, V tile
//   (b) MMA-S    rS = Q . K^T
//   (c) causal   mask
//   (d) P        = exp2(rS * scale_log2 - L * scale_log2)
//   (e) MMA-dP   rdP = dO . V^T   (V loaded as sVt + LDSM_T)
//   (f) dS       = P * (rdP - D) * scale   (overwrites rdP)
//   (g) R2S dS   convert rdP fp32 -> bf16 fragment, store to sPdS
//   (h) MMA-dQ   rdQ += dS . K  (A from sPdS via LDSM_N, B as sKt + LDSM_T)
//
// We stage dS through sPdS rather than using register-source A because the
// dQ MMA has a different M/K orientation than the Q.K^T MMA whose C output
// rdP is. FA2 does the same thing (mainloop_bwd_sm80.hpp lines 836-841,
// when Mma_dKV_is_RS=false).
//
// Smem at D=64:  sQ + sdO + sK + sV + sPdS = 40 KB
// Smem at D=128: sQ + sdO + sK + sV + sPdS = 72 KB
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_bwd_dQ_kernel(
    const Dtype* __restrict__ Q_ptr,
    const Dtype* __restrict__ K_ptr,
    const Dtype* __restrict__ V_ptr,
    const Dtype* __restrict__ dO_ptr,
    const float* __restrict__ L_ptr,
    const float* __restrict__ D_ptr,
    Dtype*       __restrict__ dQ_ptr,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale
) {
    constexpr int NumThreads = 128;
    constexpr int NumWarps   = NumThreads / 32;
    constexpr int kNRows     = 2;

    using SmemQ_t    = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemdO_t   = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemK_t    = SmemLayoutK<Bc, Headdim, Dtype>;
    using SmemV_t    = SmemLayoutV<Bc, Headdim, Dtype>;
    using SmemVt_t   = SmemLayoutVt<Bc, Headdim, Dtype>;
    using SmemKt_t   = SmemLayoutKt<Bc, Headdim, Dtype>;
    using SmemPdS_t  = SmemLayoutPdS<Br, Bc, Dtype>;
    using GmemCopy_t =
        typename GmemTiledCopyTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;

    extern __shared__ char smem_buf[];
    Dtype* sQ_data   = reinterpret_cast<Dtype*>(smem_buf);
    Dtype* sdO_data  = sQ_data   + cosize_v<SmemQ_t>;
    Dtype* sK_data   = sdO_data  + cosize_v<SmemdO_t>;
    Dtype* sV_data   = sK_data   + cosize_v<SmemK_t>;
    Dtype* sPdS_data = sV_data   + cosize_v<SmemV_t>;

    auto sQ   = make_tensor(make_smem_ptr(sQ_data),   SmemQ_t{});
    auto sdO  = make_tensor(make_smem_ptr(sdO_data),  SmemdO_t{});
    auto sK   = make_tensor(make_smem_ptr(sK_data),   SmemK_t{});
    auto sV   = make_tensor(make_smem_ptr(sV_data),   SmemV_t{});
    auto sVt  = make_tensor(make_smem_ptr(sV_data),   SmemVt_t{});
    auto sKt  = make_tensor(make_smem_ptr(sK_data),   SmemKt_t{});
    auto sPdS = make_tensor(make_smem_ptr(sPdS_data), SmemPdS_t{});

    const int bh = blockIdx.y;
    const int b  = bh / H;
    const int h  = bh % H;
    const int q_block = blockIdx.x;

    const Dtype* Q_bh  = Q_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* K_bh  = K_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* V_bh  = V_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* dO_bh = dO_ptr + b * qkv_b_stride + h * qkv_h_stride;
          Dtype* dQ_bh = dQ_ptr + b * qkv_b_stride + h * qkv_h_stride;
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

    auto gQ  = gQ_tiles(_, _, q_block);
    auto gdO = gdO_tiles(_, _, q_block);
    const int num_kv_blocks = size<2>(gK_tiles);

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    // Load Q and dO once.
    copy(gmem_copy_qkv, thr_copy.partition_S(gQ),  thr_copy.partition_D(sQ));
    copy(gmem_copy_qkv, thr_copy.partition_S(gdO), thr_copy.partition_D(sdO));

    // Per-thread L and D for the 2 rows this thread owns.
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    float L_arr[2], D_arr[2];
    load_L_D_per_thread<Br>(L_bh, D_bh, q_block, N, warp_id, lane_id, L_arr, D_arr);
    auto L_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
    auto D_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
    L_thread(0) = L_arr[0]; L_thread(1) = L_arr[1];
    D_thread(0) = D_arr[0]; D_thread(1) = D_arr[1];

    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto rQ    = thr_mma.partition_fragment_A(sQ);
    auto rK_S  = thr_mma.partition_fragment_B(sK);    // K as B for Q.K^T MMA  (N=Bc, K=Headdim)
    auto rdO   = thr_mma.partition_fragment_A(sdO);
    auto rV    = thr_mma.partition_fragment_B(sV);    // V as B for dO.V^T MMA (N=Bc, K=Headdim) -- sV row-major already has the right shape
    auto rdS   = thr_mma.partition_fragment_A(sPdS);  // dS from sPdS for dS.K MMA
    auto rK_Q  = thr_mma.partition_fragment_B(sKt);   // K^T as B for dS.K MMA (N=Headdim, K=Bc)

    using SmemCopyAtom_AK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    using SmemCopyAtom_BT = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;

    auto smem_tiled_copy_Q   = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_K_S = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);  // K from sK (no transpose)
    auto smem_tiled_copy_dO  = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_V   = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);  // V from sV (no transpose; K=Headdim)
    auto smem_tiled_copy_dS  = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_K_Q = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);  // K from sKt (transposed; needs LDSM_T)

    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K_S = smem_tiled_copy_K_S.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_dO  = smem_tiled_copy_dO.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_dS  = smem_tiled_copy_dS.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K_Q = smem_tiled_copy_K_Q.get_thread_slice(threadIdx.x);

    auto tSsQ        = smem_thr_copy_Q.partition_S(sQ);
    auto tSsK        = smem_thr_copy_K_S.partition_S(sK);
    auto tSsdO       = smem_thr_copy_dO.partition_S(sdO);
    auto tSsV        = smem_thr_copy_V.partition_S(sV);
    auto tSsdS_smem  = smem_thr_copy_dS.partition_S(sPdS);
    auto tSsKt       = smem_thr_copy_K_Q.partition_S(sKt);

    auto tSrQ_view    = smem_thr_copy_Q.retile_D(rQ);
    auto tSrK_S_view  = smem_thr_copy_K_S.retile_D(rK_S);
    auto tSrdO_view   = smem_thr_copy_dO.retile_D(rdO);
    auto tSrV_view    = smem_thr_copy_V.retile_D(rV);
    auto tSrdS_view   = smem_thr_copy_dS.retile_D(rdS);
    auto tSrK_Q_view  = smem_thr_copy_K_Q.retile_D(rK_Q);

    // R2S TiledCopy for staging rdP (after it becomes rdS) into sPdS.
    using SmemCopyAtom_R2S = Copy_Atom<DefaultCopy, Dtype>;
    auto r2s_tiled_copy_dS = make_tiled_copy_C(SmemCopyAtom_R2S{}, tiled_mma);
    auto r2s_thr_copy_dS   = r2s_tiled_copy_dS.get_thread_slice(threadIdx.x);
    auto tdSsPdS           = r2s_thr_copy_dS.partition_D(sPdS);

    // dQ accumulator (fp32, Br x Headdim).
    auto rdQ = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Headdim>>{});
    clear(rdQ);

    const float scale_log2 = softmax_scale * 1.4426950408889634f;

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Inner KV loop.
    for (int kv = 0; kv < num_kv_blocks; ++kv) {
        // Causal skip-tile: any q_idx in [q_block*Br, q_block*Br + Br) attends
        // to k_idx in [0, q_idx]. Skip if kv*Bc > q_block*Br + (Br - 1).
        if constexpr (IsCausal) {
            if (kv * Bc > q_block * Br + (Br - 1)) break;
        }

        // (a) Load K, V tile.
        {
            auto gK = gK_tiles(_, _, kv);
            auto gV = gV_tiles(_, _, kv);
            copy(gmem_copy_qkv, thr_copy.partition_S(gK), thr_copy.partition_D(sK));
            copy(gmem_copy_qkv, thr_copy.partition_S(gV), thr_copy.partition_D(sV));
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();
        }

        // (b) MMA-S: rS = Q . K^T
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rS);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rQ); ++k) {
            copy(smem_tiled_copy_Q,   tSsQ(_, _, k),       tSrQ_view(_, _, k));
            copy(smem_tiled_copy_K_S, tSsK(_, _, k),       tSrK_S_view(_, _, k));
            gemm(tiled_mma, rQ(_, _, k), rK_S(_, _, k), rS);
        }

        // (c) Causal mask.
        if constexpr (IsCausal) {
            causal_mask_tile<Br, Bc, TiledMma_t>(rS, q_block * Br, kv * Bc, threadIdx.x);
        }

        // (d) P = exp2(rS * scale_log2 - L * scale_log2). In-place on rS.
        {
            auto rS_rc = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));
            apply_lse_exp2(rS_rc, L_thread, scale_log2);
        }

        // (e) MMA-dP: rdP = dO . V^T
        auto rdP = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rdP);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdO); ++k) {
            copy(smem_tiled_copy_dO, tSsdO(_, _, k), tSrdO_view(_, _, k));
            copy(smem_tiled_copy_V,  tSsV(_, _, k),  tSrV_view(_, _, k));
            gemm(tiled_mma, rdO(_, _, k), rV(_, _, k), rdP);
        }

        // (f) dS = P * (dP - D) * scale. In-place on rdP.
        {
            auto rS_rc  = make_tensor(rS.data(),  convert_layout_acc_rowcol(rS.layout()));
            auto rdP_rc = make_tensor(rdP.data(), convert_layout_acc_rowcol(rdP.layout()));
            apply_dS(rS_rc, rdP_rc, D_thread, softmax_scale);
        }

        // (g) Convert rdP -> bf16 fragment, R2S to sPdS.
        Tensor rdP_bf16 = make_tensor_like<Dtype>(rdP);
        convert_type_out(rdP, rdP_bf16);
        auto tdSrdS = r2s_thr_copy_dS.retile_S(rdP_bf16);
        copy(r2s_tiled_copy_dS, tdSrdS, tdSsPdS);
        __syncthreads();

        // (h) MMA-dQ: rdQ += dS . K  (A from sPdS via LDSM_N, B from sKt via LDSM_T).
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdS); ++k) {
            copy(smem_tiled_copy_dS,  tSsdS_smem(_, _, k), tSrdS_view(_, _, k));
            copy(smem_tiled_copy_K_Q, tSsKt(_, _, k),      tSrK_Q_view(_, _, k));
            gemm(tiled_mma, rdS(_, _, k), rK_Q(_, _, k), rdQ);
        }

        // Sync before next iter overwrites sK, sV (and sPdS).
        __syncthreads();
    }

    // --- Epilogue: convert rdQ -> bf16, smem stage in sQ region, gmem write.
    Tensor rdQ_out = make_tensor_like<Dtype>(rdQ);
    convert_type_out(rdQ, rdQ_out);

    __syncthreads();
    auto sdQ = make_tensor(make_smem_ptr(sQ_data), SmemQ_t{});  // reuse sQ region
    using SmemCopyAtomDef = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(SmemCopyAtomDef{}, tiled_mma);
    auto smem_thr_copy_dQ   = smem_tiled_copy_dQ.get_thread_slice(threadIdx.x);

    auto tdQsdQ_dst = smem_thr_copy_dQ.partition_D(sdQ);
    auto tdQrdQ_view = smem_thr_copy_dQ.retile_S(rdQ_out);
    copy(smem_tiled_copy_dQ, tdQrdQ_view, tdQsdQ_dst);
    __syncthreads();

    auto gdQ_head = make_tensor(make_gmem_ptr(dQ_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gdQ_tiles = local_tile(gdQ_head, Shape<Int<Br>, Int<Headdim>>{},
                                make_coord(_, _0{}));
    auto gdQ = gdQ_tiles(_, _, q_block);

    using GmemTiledCopyO_t =
        typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
    GmemTiledCopyO_t gmem_copy_o;
    auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);
    copy(gmem_copy_o, thr_copy_o.partition_S(sdQ), thr_copy_o.partition_D(gdQ));
}

// -----------------------------------------------------------------------------
// bwd_dKV kernel: outer KV, inner Q. Accumulates dV and dK in registers
// across the inner Q loop and writes once at the end. No atomics.
//
// Per inner Q step:
//   (a) cp.async load Q, dO
//   (b) MMA-S    rS = Q . K^T              (K from sK, persistent)
//   (c) causal   mask
//   (d) P        = exp2(rS * scale_log2 - L * log2(e))   apply_lse_exp2
//   (e) R2S      rP -> sPdS (bf16)
//   (f) MMA-dV   rdV += P^T . dO   (A from sPdSt + LDSM_T, B from sdOt + LDSM_T)
//   (g) MMA-dP   rdP = dO . V^T   (B from sV, K=Headdim)
//   (h) dS       = P * (rdP - D) * scale  in-place on rdP   apply_dS
//   (i) R2S      rdP -> sPdS (overwrites P)
//   (j) MMA-dK   rdK += dS^T . Q    (A from sPdSt + LDSM_T, B from sQt + LDSM_T)
//
// Smem at D=64:  sK + sV + sQ + sdO + sPdS = 40 KB
// Smem at D=128: sK + sV + sQ + sdO + sPdS = 72 KB
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_bwd_dKV_kernel(
    const Dtype* __restrict__ Q_ptr,
    const Dtype* __restrict__ K_ptr,
    const Dtype* __restrict__ V_ptr,
    const Dtype* __restrict__ dO_ptr,
    const float* __restrict__ L_ptr,
    const float* __restrict__ D_ptr,
    Dtype*       __restrict__ dK_ptr,
    Dtype*       __restrict__ dV_ptr,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale
) {
    constexpr int NumThreads = 128;
    constexpr int NumWarps   = NumThreads / 32;
    constexpr int kNRows     = 2;

    using SmemQ_t    = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemdO_t   = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemK_t    = SmemLayoutK<Bc, Headdim, Dtype>;
    using SmemV_t    = SmemLayoutV<Bc, Headdim, Dtype>;
    using SmemQt_t   = SmemLayoutQt<Br, Headdim, Dtype>;
    using SmemdOt_t  = SmemLayoutdOt<Br, Headdim, Dtype>;
    using SmemPdS_t  = SmemLayoutPdS<Br, Bc, Dtype>;
    using SmemPdSt_t = SmemLayoutPdSt<Br, Bc, Dtype>;
    using GmemCopy_t =
        typename GmemTiledCopyTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;

    extern __shared__ char smem_buf[];
    Dtype* sK_data   = reinterpret_cast<Dtype*>(smem_buf);
    Dtype* sV_data   = sK_data   + cosize_v<SmemK_t>;
    Dtype* sQ_data   = sV_data   + cosize_v<SmemV_t>;
    Dtype* sdO_data  = sQ_data   + cosize_v<SmemQ_t>;
    Dtype* sPdS_data = sdO_data  + cosize_v<SmemdO_t>;

    auto sK    = make_tensor(make_smem_ptr(sK_data),   SmemK_t{});
    auto sV    = make_tensor(make_smem_ptr(sV_data),   SmemV_t{});
    auto sQ    = make_tensor(make_smem_ptr(sQ_data),   SmemQ_t{});
    auto sdO   = make_tensor(make_smem_ptr(sdO_data),  SmemdO_t{});
    auto sPdS  = make_tensor(make_smem_ptr(sPdS_data), SmemPdS_t{});
    auto sQt   = make_tensor(make_smem_ptr(sQ_data),   SmemQt_t{});
    auto sdOt  = make_tensor(make_smem_ptr(sdO_data),  SmemdOt_t{});
    auto sPdSt = make_tensor(make_smem_ptr(sPdS_data), SmemPdSt_t{});

    const int bh = blockIdx.y;
    const int b  = bh / H;
    const int h  = bh % H;
    const int kv_block = blockIdx.x;

    const Dtype* Q_bh  = Q_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* K_bh  = K_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* V_bh  = V_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* dO_bh = dO_ptr + b * qkv_b_stride + h * qkv_h_stride;
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

    auto gK = gK_tiles(_, _, kv_block);
    auto gV = gV_tiles(_, _, kv_block);
    const int num_q_blocks = size<2>(gQ_tiles);

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    // Load K and V once.
    copy(gmem_copy_qkv, thr_copy.partition_S(gK), thr_copy.partition_D(sK));
    copy(gmem_copy_qkv, thr_copy.partition_S(gV), thr_copy.partition_D(sV));

    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // Persistent fragment handles for K, V (used in MMA-S, MMA-dP).
    auto rK   = thr_mma.partition_fragment_B(sK);
    auto rV   = thr_mma.partition_fragment_B(sV);
    // dV (P^T . dO): A from sPdSt, B from sdOt.
    auto rP_t  = thr_mma.partition_fragment_A(sPdSt);
    auto rdOt  = thr_mma.partition_fragment_B(sdOt);
    // dK (dS^T . Q): A from sPdSt, B from sQt.
    auto rdS_t = thr_mma.partition_fragment_A(sPdSt);
    auto rQt   = thr_mma.partition_fragment_B(sQt);
    // Per-Q fragments for the inner-loop MMAs.
    auto rQ   = thr_mma.partition_fragment_A(sQ);
    auto rdO  = thr_mma.partition_fragment_A(sdO);

    using SmemCopyAtom_AK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    using SmemCopyAtom_BT = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;
    using SmemCopyAtom_AT = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;

    auto smem_tiled_copy_Q   = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_K   = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_dO  = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_V   = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tiled_copy_PdS_T = make_tiled_copy_A(SmemCopyAtom_AT{}, tiled_mma);  // for sPdSt -> A
    auto smem_tiled_copy_dOt = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);    // for sdOt -> B
    auto smem_tiled_copy_Qt  = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);    // for sQt  -> B

    auto smem_thr_copy_Q     = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K     = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_dO    = smem_tiled_copy_dO.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_V     = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_PdS_T = smem_tiled_copy_PdS_T.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_dOt   = smem_tiled_copy_dOt.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_Qt    = smem_tiled_copy_Qt.get_thread_slice(threadIdx.x);

    auto tSsQ        = smem_thr_copy_Q.partition_S(sQ);
    auto tSsK        = smem_thr_copy_K.partition_S(sK);
    auto tSsdO       = smem_thr_copy_dO.partition_S(sdO);
    auto tSsV        = smem_thr_copy_V.partition_S(sV);
    auto tdVsP_t     = smem_thr_copy_PdS_T.partition_S(sPdSt);
    auto tdVsdOt     = smem_thr_copy_dOt.partition_S(sdOt);
    auto tdKsdS_t    = smem_thr_copy_PdS_T.partition_S(sPdSt);
    auto tdKsQt      = smem_thr_copy_Qt.partition_S(sQt);

    auto tSrQ_view    = smem_thr_copy_Q.retile_D(rQ);
    auto tSrK_view    = smem_thr_copy_K.retile_D(rK);
    auto tSrdO_view   = smem_thr_copy_dO.retile_D(rdO);
    auto tSrV_view    = smem_thr_copy_V.retile_D(rV);
    auto tdVrP_view   = smem_thr_copy_PdS_T.retile_D(rP_t);
    auto tdVrdOt_view = smem_thr_copy_dOt.retile_D(rdOt);
    auto tdKrdS_view  = smem_thr_copy_PdS_T.retile_D(rdS_t);
    auto tdKrQt_view  = smem_thr_copy_Qt.retile_D(rQt);

    // R2S TiledCopy for staging P/dS into sPdS (uses C-output pattern).
    using SmemCopyAtom_R2S = Copy_Atom<DefaultCopy, Dtype>;
    auto r2s_tiled_copy = make_tiled_copy_C(SmemCopyAtom_R2S{}, tiled_mma);
    auto r2s_thr_copy   = r2s_tiled_copy.get_thread_slice(threadIdx.x);
    auto tdSsPdS        = r2s_thr_copy.partition_D(sPdS);

    // Accumulators.
    auto rdK = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    auto rdV = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    clear(rdK);
    clear(rdV);

    const float scale_log2 = softmax_scale * 1.4426950408889634f;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Inner Q loop.
    for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
        // Causal skip-tile: skip Q-blocks fully above this KV-block's diagonal.
        // For non-causal, no skip. For causal, skip if q_block*Br + Br - 1 < kv_block*Bc.
        if constexpr (IsCausal) {
            if (q_block * Br + (Br - 1) < kv_block * Bc) continue;
        }

        // (a) Load Q, dO.
        {
            auto gQ  = gQ_tiles(_, _, q_block);
            auto gdO = gdO_tiles(_, _, q_block);
            copy(gmem_copy_qkv, thr_copy.partition_S(gQ),  thr_copy.partition_D(sQ));
            copy(gmem_copy_qkv, thr_copy.partition_S(gdO), thr_copy.partition_D(sdO));
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();
        }

        // Per-thread L and D for this Q-block's rows.
        float L_arr[2], D_arr[2];
        load_L_D_per_thread<Br>(L_bh, D_bh, q_block, N, warp_id, lane_id, L_arr, D_arr);
        auto L_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
        auto D_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
        L_thread(0) = L_arr[0]; L_thread(1) = L_arr[1];
        D_thread(0) = D_arr[0]; D_thread(1) = D_arr[1];

        // (b) MMA-S: rS = Q . K^T
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rS);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rQ); ++k) {
            copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_view(_, _, k));
            copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_view(_, _, k));
            gemm(tiled_mma, rQ(_, _, k), rK(_, _, k), rS);
        }

        // (c) Causal mask.
        if constexpr (IsCausal) {
            causal_mask_tile<Br, Bc, TiledMma_t>(rS, q_block * Br, kv_block * Bc, threadIdx.x);
        }

        // (d) P = exp2(rS * scale_log2 - L * log2(e)). In-place.
        {
            auto rS_rc = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));
            apply_lse_exp2(rS_rc, L_thread, scale_log2);
        }

        // (e) Stage P (bf16) into sPdS.
        Tensor rP_bf16 = make_tensor_like<Dtype>(rS);
        convert_type_out(rS, rP_bf16);
        {
            auto tdSrP = r2s_thr_copy.retile_S(rP_bf16);
            copy(r2s_tiled_copy, tdSrP, tdSsPdS);
        }
        __syncthreads();

        // (f) MMA-dV: rdV += P^T . dO.
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rP_t); ++k) {
            copy(smem_tiled_copy_PdS_T, tdVsP_t(_, _, k), tdVrP_view(_, _, k));
            copy(smem_tiled_copy_dOt,   tdVsdOt(_, _, k), tdVrdOt_view(_, _, k));
            gemm(tiled_mma, rP_t(_, _, k), rdOt(_, _, k), rdV);
        }

        // (g) MMA-dP: rdP = dO . V^T  (B from sV, K=Headdim).
        auto rdP = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rdP);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdO); ++k) {
            copy(smem_tiled_copy_dO, tSsdO(_, _, k), tSrdO_view(_, _, k));
            copy(smem_tiled_copy_V,  tSsV(_, _, k),  tSrV_view(_, _, k));
            gemm(tiled_mma, rdO(_, _, k), rV(_, _, k), rdP);
        }

        // (h) dS = P * (rdP - D) * scale. Apply on rdP using rS (still holds P).
        {
            auto rS_rc  = make_tensor(rS.data(),  convert_layout_acc_rowcol(rS.layout()));
            auto rdP_rc = make_tensor(rdP.data(), convert_layout_acc_rowcol(rdP.layout()));
            apply_dS(rS_rc, rdP_rc, D_thread, softmax_scale);
        }

        // (i) Stage dS (bf16) into sPdS, overwriting P.
        Tensor rdS_bf16 = make_tensor_like<Dtype>(rdP);
        convert_type_out(rdP, rdS_bf16);
        __syncthreads();  // ensure dV MMA finished reading sPdS
        {
            auto tdSrdS = r2s_thr_copy.retile_S(rdS_bf16);
            copy(r2s_tiled_copy, tdSrdS, tdSsPdS);
        }
        __syncthreads();

        // (j) MMA-dK: rdK += dS^T . Q.
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdS_t); ++k) {
            copy(smem_tiled_copy_PdS_T, tdKsdS_t(_, _, k), tdKrdS_view(_, _, k));
            copy(smem_tiled_copy_Qt,    tdKsQt(_, _, k),   tdKrQt_view(_, _, k));
            gemm(tiled_mma, rdS_t(_, _, k), rQt(_, _, k), rdK);
        }

        // Sync before next iter overwrites sQ, sdO, sPdS.
        __syncthreads();
    }

    // --- Epilogue: write rdV then rdK to gmem via smem staging ---
    // Reuse sQ region as a (Bc, Headdim) staging buffer (it's the same byte
    // count as the K/V tiles since Bc=Br for our config).
    auto sdKV = make_tensor(make_smem_ptr(sQ_data), SmemK_t{});
    using SmemCopyAtomDef = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomDef{}, tiled_mma);
    auto smem_thr_copy_dKV   = smem_tiled_copy_dKV.get_thread_slice(threadIdx.x);
    using GmemTiledCopyO_t =
        typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
    GmemTiledCopyO_t gmem_copy_o;
    auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);

    auto gdV_head = make_tensor(make_gmem_ptr(dV_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gdK_head = make_tensor(make_gmem_ptr(dK_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto gdV_tiles = local_tile(gdV_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gdK_tiles = local_tile(gdK_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gdV = gdV_tiles(_, _, kv_block);
    auto gdK = gdK_tiles(_, _, kv_block);

    // Write dV
    {
        Tensor rdV_out = make_tensor_like<Dtype>(rdV);
        convert_type_out(rdV, rdV_out);
        __syncthreads();
        auto dst = smem_thr_copy_dKV.partition_D(sdKV);
        auto src = smem_thr_copy_dKV.retile_S(rdV_out);
        copy(smem_tiled_copy_dKV, src, dst);
        __syncthreads();
        copy(gmem_copy_o, thr_copy_o.partition_S(sdKV), thr_copy_o.partition_D(gdV));
    }
    __syncthreads();
    // Write dK
    {
        Tensor rdK_out = make_tensor_like<Dtype>(rdK);
        convert_type_out(rdK, rdK_out);
        auto dst = smem_thr_copy_dKV.partition_D(sdKV);
        auto src = smem_thr_copy_dKV.retile_S(rdK_out);
        copy(smem_tiled_copy_dKV, src, dst);
        __syncthreads();
        copy(gmem_copy_o, thr_copy_o.partition_S(sdKV), thr_copy_o.partition_D(gdK));
    }
}

// -----------------------------------------------------------------------------
// Launch helpers.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, bool IsCausal>
static void launch_bwd_dQ(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    const Dtype* dO,
    const float* L, const float* D,
    Dtype* dQ,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale, cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kNumThreads = 128;
    constexpr int sQ_elems   = cosize_v<SmemLayoutQ<Br, Headdim, Dtype>>;
    constexpr int sdO_elems  = sQ_elems;
    constexpr int sK_elems   = cosize_v<SmemLayoutK<Bc, Headdim, Dtype>>;
    constexpr int sV_elems   = cosize_v<SmemLayoutV<Bc, Headdim, Dtype>>;
    constexpr int sPdS_elems = cosize_v<SmemLayoutPdS<Br, Bc, Dtype>>;
    constexpr int smem_bytes =
        (sQ_elems + sdO_elems + sK_elems + sV_elems + sPdS_elems) * sizeof(Dtype);

    const int num_q_blocks = (N + Br - 1) / Br;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(kNumThreads);

    // Enable >48KB dynamic smem (default cap on most arches).
    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_v9_bwd_dQ_kernel<Dtype, Headdim, Br, Bc, IsCausal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    flash_v9_bwd_dQ_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, dO, L, D, dQ,
            B, H, N,
            qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride,
            softmax_scale
        );
}

template <typename Dtype>
static void dispatch_bwd_dQ(
    const Dtype* Q, const Dtype* K, const Dtype* V, const Dtype* dO,
    const float* L, const float* D,
    Dtype* dQ,
    int B, int H, int N, int Hd,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    bool is_causal, float softmax_scale, cudaStream_t stream
) {
    #define LAUNCH(HD, CAUSAL) \
        launch_bwd_dQ<Dtype, HD, CAUSAL>(Q, K, V, dO, L, D, dQ, B, H, N, \
            qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride, \
            softmax_scale, stream)
    if (Hd == 64)        { if (is_causal) LAUNCH(64,  true); else LAUNCH(64,  false); }
    else if (Hd == 128)  { if (is_causal) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "flash_v9 bwd: headdim must be 64 or 128");
    #undef LAUNCH
}

template <typename Dtype, int Headdim, bool IsCausal>
static void launch_bwd_dKV(
    const Dtype* Q, const Dtype* K, const Dtype* V, const Dtype* dO,
    const float* L, const float* D,
    Dtype* dK, Dtype* dV,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale, cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kNumThreads = 128;
    constexpr int sQ_elems   = cosize_v<SmemLayoutQ<Br, Headdim, Dtype>>;
    constexpr int sdO_elems  = sQ_elems;
    constexpr int sK_elems   = cosize_v<SmemLayoutK<Bc, Headdim, Dtype>>;
    constexpr int sV_elems   = cosize_v<SmemLayoutV<Bc, Headdim, Dtype>>;
    constexpr int sPdS_elems = cosize_v<SmemLayoutPdS<Br, Bc, Dtype>>;
    constexpr int smem_bytes =
        (sQ_elems + sdO_elems + sK_elems + sV_elems + sPdS_elems) * sizeof(Dtype);

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    dim3 grid(num_kv_blocks, B * H);
    dim3 block(kNumThreads);

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_v9_bwd_dKV_kernel<Dtype, Headdim, Br, Bc, IsCausal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    flash_v9_bwd_dKV_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, dO, L, D, dK, dV,
            B, H, N,
            qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride,
            softmax_scale
        );
}

template <typename Dtype>
static void dispatch_bwd_dKV(
    const Dtype* Q, const Dtype* K, const Dtype* V, const Dtype* dO,
    const float* L, const float* D,
    Dtype* dK, Dtype* dV,
    int B, int H, int N, int Hd,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    bool is_causal, float softmax_scale, cudaStream_t stream
) {
    #define LAUNCH(HD, CAUSAL) \
        launch_bwd_dKV<Dtype, HD, CAUSAL>(Q, K, V, dO, L, D, dK, dV, B, H, N, \
            qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride, \
            softmax_scale, stream)
    if (Hd == 64)        { if (is_causal) LAUNCH(64,  true); else LAUNCH(64,  false); }
    else if (Hd == 128)  { if (is_causal) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "flash_v9 bwd_dKV: headdim must be 64 or 128");
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
    flash_v9_bwd_preprocess_kernel<Dtype, Headdim>
        <<<grid, threads, 0, stream>>>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride);
}

template <typename Dtype>
static void dispatch_preprocess(
    const Dtype* dO, const Dtype* O, float* D,
    int B, int H, int N, int Hd,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride,
    cudaStream_t stream
) {
    if      (Hd == 64)  launch_preprocess<Dtype, 64>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else if (Hd == 128) launch_preprocess<Dtype, 128>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else TORCH_CHECK(false, "flash_v9 bwd preprocess: headdim must be 64 or 128");
}

} // namespace flash_v9


std::vector<torch::Tensor> flash_v9_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    bool is_causal, double softmax_scale
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(Q.dim() == 4, "Q, K, V, dO must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.size(2) % 64 == 0, "flash_v9 bwd: N must be a multiple of 64");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D = Q.size(3);

    auto dQ = torch::empty_like(Q);
    auto dK = torch::empty_like(K);
    auto dV = torch::empty_like(V);
    auto D_tensor = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (Q.scalar_type() == torch::kBFloat16) {
        using T = cutlass::bfloat16_t;
        const T* Q_p  = reinterpret_cast<const T*>(Q.data_ptr());
        const T* K_p  = reinterpret_cast<const T*>(K.data_ptr());
        const T* V_p  = reinterpret_cast<const T*>(V.data_ptr());
        const T* O_p  = reinterpret_cast<const T*>(O.data_ptr());
        const T* dO_p = reinterpret_cast<const T*>(dO.data_ptr());
        T* dQ_p = reinterpret_cast<T*>(dQ.data_ptr());
        T* dK_p = reinterpret_cast<T*>(dK.data_ptr());
        T* dV_p = reinterpret_cast<T*>(dV.data_ptr());

        flash_v9::dispatch_preprocess<T>(dO_p, O_p, D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D, Q.stride(1), Q.stride(2), N, stream);
        flash_v9::dispatch_bwd_dQ<T>(Q_p, K_p, V_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), dQ_p,
            (int)B, (int)H, (int)N, (int)D, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
        flash_v9::dispatch_bwd_dKV<T>(Q_p, K_p, V_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), dK_p, dV_p,
            (int)B, (int)H, (int)N, (int)D, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
    } else if (Q.scalar_type() == torch::kHalf) {
        using T = cutlass::half_t;
        const T* Q_p  = reinterpret_cast<const T*>(Q.data_ptr());
        const T* K_p  = reinterpret_cast<const T*>(K.data_ptr());
        const T* V_p  = reinterpret_cast<const T*>(V.data_ptr());
        const T* O_p  = reinterpret_cast<const T*>(O.data_ptr());
        const T* dO_p = reinterpret_cast<const T*>(dO.data_ptr());
        T* dQ_p = reinterpret_cast<T*>(dQ.data_ptr());
        T* dK_p = reinterpret_cast<T*>(dK.data_ptr());
        T* dV_p = reinterpret_cast<T*>(dV.data_ptr());

        flash_v9::dispatch_preprocess<T>(dO_p, O_p, D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D, Q.stride(1), Q.stride(2), N, stream);
        flash_v9::dispatch_bwd_dQ<T>(Q_p, K_p, V_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), dQ_p,
            (int)B, (int)H, (int)N, (int)D, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
        flash_v9::dispatch_bwd_dKV<T>(Q_p, K_p, V_p, dO_p,
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), dK_p, dV_p,
            (int)B, (int)H, (int)N, (int)D, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
    } else {
        TORCH_CHECK(false, "flash_v9 bwd: only bf16 and fp16 supported");
    }

    return {dQ, dK, dV};
}
