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
// Algorithm 3 (double backward) -- v8-style two-kernel split.
//
//   dblbwd_KV     outer KV, inner Q.  Writes g_K and g_V.        (this commit)
//   dblbwd_QdO    outer Q,  inner KV. Writes g_Q and g_dO.        (commit 6g)
//
// Math (saved L not used; bwd recomputes via softmax via apply_lse_exp2):
//   S    = Q . K^T
//   P    = exp(S * scale - L)                               apply_lse_exp2
//   D_i  = rowsum(dO * O)                                   from preprocess
//   dP   = dO . V^T
//   dS   = P * (dP - D) * scale                             apply_dS
//   M    = g_dQ . K^T + Q . g_dK^T                          (two MMAs)
//   N    = dO . g_dV^T                                      (one MMA)
//   dL/dS = P * N + scale * M * dS                          elementwise
//   g_K  = scale * (dL/dS)^T . Q                            (this kernel)
//   g_V  = scale * (P * M)^T . dO                           (this kernel)
//   g_Q  = scale * (dL/dS) . K                              (commit 6g)
//   g_dO = P @ g_dV + (scale * P*M) V - scale * rowsum(P*M) * O   (6g)
// =============================================================================

// Forward decl.
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_dblbwd_KV_kernel(
    const Dtype* __restrict__ Q,    const Dtype* __restrict__ K,
    const Dtype* __restrict__ V,    const Dtype* __restrict__ dO,
    const Dtype* __restrict__ g_dQ, const Dtype* __restrict__ g_dK,
    const Dtype* __restrict__ g_dV,
    const float* __restrict__ L,    const float* __restrict__ D,
    Dtype* __restrict__ g_K,        Dtype* __restrict__ g_V,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale);

// Preprocess: D = rowsum(dO * O). Reuses backward.cu's approach.
template <typename Dtype, int Headdim>
__global__ void flash_v9_dblbwd_preprocess_kernel(
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
// dblbwd_KV kernel: outer KV, inner Q. Writes g_K, g_V.
//
// Per inner Q step (single-pass design):
//   (a) cp.async load Q, dO, g_dQ
//   (b) load L[q], D[q] per thread
//   (c) MMA-S    rS = Q . K^T
//   (d) causal   mask
//   (e) P        = exp(S * scale - L)              (in-place on rS)
//   (f) MMA-dP   rdP = dO . V^T
//   (g) dS       = P * (rdP - D) * scale           (in-place on rdP)
//   (h) MMA-M1   rM  = g_dQ . K^T
//   (i) MMA-M2   rM += Q . g_dK^T
//   (j) MMA-N    rN  = dO . g_dV^T
//   (k) dL/dS    rN  = rS*rN + scale * rM * rdS    (in-place on rN)
//                Also compute (P*M) into a scratch register tensor.
//   (l) R2S      dL/dS -> sPdS
//   (m) MMA-gK   rg_K += scale * (dL/dS)^T . Q     (A from sPdSt, B from sQt)
//   (n) R2S      (P*M) -> sPdS (overwrites dL/dS)
//   (o) MMA-gV   rg_V += scale * (P*M)^T . dO      (A from sPdSt, B from sdOt)
//
// Smem at D=64: sK + sV + sg_dK + sg_dV + sQ + sdO + sg_dQ + sPdS = 64 KB
// D=128 not yet supported (would need 128 KB).
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_dblbwd_KV_kernel(
    const Dtype* __restrict__ Q_ptr,    const Dtype* __restrict__ K_ptr,
    const Dtype* __restrict__ V_ptr,    const Dtype* __restrict__ dO_ptr,
    const Dtype* __restrict__ g_dQ_ptr, const Dtype* __restrict__ g_dK_ptr,
    const Dtype* __restrict__ g_dV_ptr,
    const float* __restrict__ L_ptr,    const float* __restrict__ D_ptr,
    Dtype* __restrict__ g_K_ptr,        Dtype* __restrict__ g_V_ptr,
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
    using SmemKt_t   = SmemLayoutKt<Bc, Headdim, Dtype>;
    using SmemPdS_t  = SmemLayoutPdS<Br, Bc, Dtype>;
    using SmemPdSt_t = SmemLayoutPdSt<Br, Bc, Dtype>;
    using GmemCopy_t =
        typename GmemTiledCopyTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;

    extern __shared__ char smem_buf[];
    Dtype* sK_data    = reinterpret_cast<Dtype*>(smem_buf);
    Dtype* sV_data    = sK_data    + cosize_v<SmemK_t>;
    Dtype* sg_dK_data = sV_data    + cosize_v<SmemV_t>;
    Dtype* sg_dV_data = sg_dK_data + cosize_v<SmemK_t>;
    Dtype* sQ_data    = sg_dV_data + cosize_v<SmemV_t>;
    Dtype* sdO_data   = sQ_data    + cosize_v<SmemQ_t>;
    Dtype* sg_dQ_data = sdO_data   + cosize_v<SmemdO_t>;
    Dtype* sPdS_data  = sg_dQ_data + cosize_v<SmemQ_t>;

    auto sK     = make_tensor(make_smem_ptr(sK_data),    SmemK_t{});
    auto sV     = make_tensor(make_smem_ptr(sV_data),    SmemV_t{});
    auto sg_dK  = make_tensor(make_smem_ptr(sg_dK_data), SmemK_t{});
    auto sg_dV  = make_tensor(make_smem_ptr(sg_dV_data), SmemV_t{});
    auto sQ     = make_tensor(make_smem_ptr(sQ_data),    SmemQ_t{});
    auto sdO    = make_tensor(make_smem_ptr(sdO_data),   SmemdO_t{});
    auto sg_dQ  = make_tensor(make_smem_ptr(sg_dQ_data), SmemQ_t{});
    auto sPdS   = make_tensor(make_smem_ptr(sPdS_data),  SmemPdS_t{});
    // Transposed views.
    auto sg_dKt = make_tensor(make_smem_ptr(sg_dK_data), SmemKt_t{});
    auto sg_dVt = make_tensor(make_smem_ptr(sg_dV_data), SmemKt_t{});
    auto sQt    = make_tensor(make_smem_ptr(sQ_data),    SmemQt_t{});
    auto sdOt   = make_tensor(make_smem_ptr(sdO_data),   SmemdOt_t{});
    auto sPdSt  = make_tensor(make_smem_ptr(sPdS_data),  SmemPdSt_t{});

    const int bh = blockIdx.y;
    const int b  = bh / H;
    const int h  = bh % H;
    const int kv_block = blockIdx.x;

    const Dtype* Q_bh    = Q_ptr    + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* K_bh    = K_ptr    + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* V_bh    = V_ptr    + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* dO_bh   = dO_ptr   + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* gdQ_bh  = g_dQ_ptr + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* gdK_bh  = g_dK_ptr + b * qkv_b_stride + h * qkv_h_stride;
    const Dtype* gdV_bh  = g_dV_ptr + b * qkv_b_stride + h * qkv_h_stride;
          Dtype* gK_bh   = g_K_ptr  + b * qkv_b_stride + h * qkv_h_stride;
          Dtype* gV_bh   = g_V_ptr  + b * qkv_b_stride + h * qkv_h_stride;
    const float* L_bh    = L_ptr    + b * l_b_stride   + h * l_h_stride;
    const float* D_bh    = D_ptr    + b * l_b_stride   + h * l_h_stride;

    // Gmem heads + tiles.
    auto make_head = [&](const Dtype* p) {
        return make_tensor(make_gmem_ptr(p),
                           make_shape(N, Int<Headdim>{}),
                           make_stride(Int<Headdim>{}, _1{}));
    };
    auto gQ_head    = make_head(Q_bh);
    auto gK_head    = make_head(K_bh);
    auto gV_head    = make_head(V_bh);
    auto gdO_head   = make_head(dO_bh);
    auto ggdQ_head  = make_head(gdQ_bh);
    auto ggdK_head  = make_head(gdK_bh);
    auto ggdV_head  = make_head(gdV_bh);

    auto gQ_tiles    = local_tile(gQ_head,   Shape<Int<Br>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gK_tiles    = local_tile(gK_head,   Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gV_tiles    = local_tile(gV_head,   Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto gdO_tiles   = local_tile(gdO_head,  Shape<Int<Br>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto ggdQ_tiles  = local_tile(ggdQ_head, Shape<Int<Br>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto ggdK_tiles  = local_tile(ggdK_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto ggdV_tiles  = local_tile(ggdV_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));

    auto gK    = gK_tiles(_, _, kv_block);
    auto gV    = gV_tiles(_, _, kv_block);
    auto ggdK  = ggdK_tiles(_, _, kv_block);
    auto ggdV  = ggdV_tiles(_, _, kv_block);
    const int num_q_blocks = size<2>(gQ_tiles);

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    // Load K, V, g_dK, g_dV once (per-KV outer-loop tile).
    copy(gmem_copy_qkv, thr_copy.partition_S(gK),    thr_copy.partition_D(sK));
    copy(gmem_copy_qkv, thr_copy.partition_S(gV),    thr_copy.partition_D(sV));
    copy(gmem_copy_qkv, thr_copy.partition_S(ggdK),  thr_copy.partition_D(sg_dK));
    copy(gmem_copy_qkv, thr_copy.partition_S(ggdV),  thr_copy.partition_D(sg_dV));

    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    // Persistent fragment handles.
    auto rK     = thr_mma.partition_fragment_B(sK);
    auto rV     = thr_mma.partition_fragment_B(sV);
    auto rgdKt  = thr_mma.partition_fragment_B(sg_dKt);
    auto rgdVt  = thr_mma.partition_fragment_B(sg_dVt);
    // Per-Q fragments (rebuilt each iter is fine; the tensor types match).
    auto rQ     = thr_mma.partition_fragment_A(sQ);
    auto rdO    = thr_mma.partition_fragment_A(sdO);
    auto rgdQ   = thr_mma.partition_fragment_A(sg_dQ);
    auto rPdS_A = thr_mma.partition_fragment_A(sPdSt);
    auto rdOt   = thr_mma.partition_fragment_B(sdOt);
    auto rQt    = thr_mma.partition_fragment_B(sQt);

    using SmemCopyAtom_AK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    using SmemCopyAtom_BT = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;
    using SmemCopyAtom_AT = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;

    auto smem_tcopy_Q     = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tcopy_K     = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tcopy_dO    = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tcopy_V     = make_tiled_copy_B(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tcopy_gdQ   = make_tiled_copy_A(SmemCopyAtom_AK{}, tiled_mma);
    auto smem_tcopy_gdKt  = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);
    auto smem_tcopy_gdVt  = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);
    auto smem_tcopy_PdS_A = make_tiled_copy_A(SmemCopyAtom_AT{}, tiled_mma);
    auto smem_tcopy_dOt   = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);
    auto smem_tcopy_Qt    = make_tiled_copy_B(SmemCopyAtom_BT{}, tiled_mma);

    auto smem_thr_Q     = smem_tcopy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_K     = smem_tcopy_K.get_thread_slice(threadIdx.x);
    auto smem_thr_dO    = smem_tcopy_dO.get_thread_slice(threadIdx.x);
    auto smem_thr_V     = smem_tcopy_V.get_thread_slice(threadIdx.x);
    auto smem_thr_gdQ   = smem_tcopy_gdQ.get_thread_slice(threadIdx.x);
    auto smem_thr_gdKt  = smem_tcopy_gdKt.get_thread_slice(threadIdx.x);
    auto smem_thr_gdVt  = smem_tcopy_gdVt.get_thread_slice(threadIdx.x);
    auto smem_thr_PdS_A = smem_tcopy_PdS_A.get_thread_slice(threadIdx.x);
    auto smem_thr_dOt   = smem_tcopy_dOt.get_thread_slice(threadIdx.x);
    auto smem_thr_Qt    = smem_tcopy_Qt.get_thread_slice(threadIdx.x);

    auto tSsQ        = smem_thr_Q.partition_S(sQ);
    auto tSsK        = smem_thr_K.partition_S(sK);
    auto tSsdO       = smem_thr_dO.partition_S(sdO);
    auto tSsV        = smem_thr_V.partition_S(sV);
    auto tSsgdQ      = smem_thr_gdQ.partition_S(sg_dQ);
    auto tSsgdKt     = smem_thr_gdKt.partition_S(sg_dKt);
    auto tSsgdVt     = smem_thr_gdVt.partition_S(sg_dVt);
    auto tSsPdSt     = smem_thr_PdS_A.partition_S(sPdSt);
    auto tSsdOt      = smem_thr_dOt.partition_S(sdOt);
    auto tSsQt       = smem_thr_Qt.partition_S(sQt);

    auto tSrQ_v       = smem_thr_Q.retile_D(rQ);
    auto tSrK_v       = smem_thr_K.retile_D(rK);
    auto tSrdO_v      = smem_thr_dO.retile_D(rdO);
    auto tSrV_v       = smem_thr_V.retile_D(rV);
    auto tSrgdQ_v     = smem_thr_gdQ.retile_D(rgdQ);
    auto tSrgdKt_v    = smem_thr_gdKt.retile_D(rgdKt);
    auto tSrgdVt_v    = smem_thr_gdVt.retile_D(rgdVt);
    auto tSrPdS_A_v   = smem_thr_PdS_A.retile_D(rPdS_A);
    auto tSrdOt_v     = smem_thr_dOt.retile_D(rdOt);
    auto tSrQt_v      = smem_thr_Qt.retile_D(rQt);

    using SmemCopyAtom_R2S = Copy_Atom<DefaultCopy, Dtype>;
    auto r2s_tiled = make_tiled_copy_C(SmemCopyAtom_R2S{}, tiled_mma);
    auto r2s_thr   = r2s_tiled.get_thread_slice(threadIdx.x);
    auto tdSsPdS   = r2s_thr.partition_D(sPdS);

    // Accumulators.
    auto rg_K = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    auto rg_V = partition_fragment_C(tiled_mma, Shape<Int<Bc>, Int<Headdim>>{});
    clear(rg_K);
    clear(rg_V);

    const float scale_log2 = softmax_scale * 1.4426950408889634f;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (int q_block = 0; q_block < num_q_blocks; ++q_block) {
        if constexpr (IsCausal) {
            if (q_block * Br + (Br - 1) < kv_block * Bc) continue;
        }

        // (a) Load Q, dO, g_dQ.
        {
            auto gQ   = gQ_tiles(_, _, q_block);
            auto gdO  = gdO_tiles(_, _, q_block);
            auto ggdQ_q = ggdQ_tiles(_, _, q_block);
            copy(gmem_copy_qkv, thr_copy.partition_S(gQ),    thr_copy.partition_D(sQ));
            copy(gmem_copy_qkv, thr_copy.partition_S(gdO),   thr_copy.partition_D(sdO));
            copy(gmem_copy_qkv, thr_copy.partition_S(ggdQ_q),thr_copy.partition_D(sg_dQ));
            cp_async_fence();
            cp_async_wait<0>();
            __syncthreads();
        }

        // (b) Per-thread L and D.
        float L_arr[2], D_arr[2];
        load_L_D_per_thread<Br>(L_bh, D_bh, q_block, N, warp_id, lane_id, L_arr, D_arr);
        auto L_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
        auto D_thread = make_tensor<float>(make_shape(Int<kNRows>{}));
        L_thread(0) = L_arr[0]; L_thread(1) = L_arr[1];
        D_thread(0) = D_arr[0]; D_thread(1) = D_arr[1];

        // (c) MMA-S: rS = Q . K^T
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rS);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rQ); ++k) {
            copy(smem_tcopy_Q, tSsQ(_, _, k), tSrQ_v(_, _, k));
            copy(smem_tcopy_K, tSsK(_, _, k), tSrK_v(_, _, k));
            gemm(tiled_mma, rQ(_, _, k), rK(_, _, k), rS);
        }

        // (d) Causal mask.
        if constexpr (IsCausal) {
            causal_mask_tile<Br, Bc, TiledMma_t>(rS, q_block * Br, kv_block * Bc, threadIdx.x);
        }

        // (e) P = exp(rS * scale - L). In-place on rS.
        {
            auto rS_rc = make_tensor(rS.data(), convert_layout_acc_rowcol(rS.layout()));
            apply_lse_exp2(rS_rc, L_thread, scale_log2);
        }

        // (f) MMA-dP: rdP = dO . V^T
        auto rdP = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rdP);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdO); ++k) {
            copy(smem_tcopy_dO, tSsdO(_, _, k), tSrdO_v(_, _, k));
            copy(smem_tcopy_V,  tSsV(_, _, k),  tSrV_v(_, _, k));
            gemm(tiled_mma, rdO(_, _, k), rV(_, _, k), rdP);
        }

        // (g) dS = P * (rdP - D) * scale. In-place on rdP.
        {
            auto rS_rc  = make_tensor(rS.data(),  convert_layout_acc_rowcol(rS.layout()));
            auto rdP_rc = make_tensor(rdP.data(), convert_layout_acc_rowcol(rdP.layout()));
            apply_dS(rS_rc, rdP_rc, D_thread, softmax_scale);
        }
        // rS holds P, rdP holds dS.

        // (h+i) MMA-M: rM = g_dQ . K^T + Q . g_dK^T
        auto rM = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rM);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rgdQ); ++k) {
            copy(smem_tcopy_gdQ, tSsgdQ(_, _, k), tSrgdQ_v(_, _, k));
            // K is already loaded in rK from MMA-S; but each gemm consumes
            // rK by k-slice -- safe to re-issue copies (idempotent).
            copy(smem_tcopy_K,   tSsK(_, _, k),   tSrK_v(_, _, k));
            gemm(tiled_mma, rgdQ(_, _, k), rK(_, _, k), rM);
        }
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rQ); ++k) {
            // Q is already in registers from MMA-S. Re-copy because k-slice
            // tracking across non-adjacent uses isn't guaranteed.
            copy(smem_tcopy_Q,    tSsQ(_, _, k),    tSrQ_v(_, _, k));
            copy(smem_tcopy_gdKt, tSsgdKt(_, _, k), tSrgdKt_v(_, _, k));
            gemm(tiled_mma, rQ(_, _, k), rgdKt(_, _, k), rM);
        }

        // (j) MMA-N: rN = dO . g_dV^T
        auto rN = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rN);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rdO); ++k) {
            copy(smem_tcopy_dO,   tSsdO(_, _, k),   tSrdO_v(_, _, k));
            copy(smem_tcopy_gdVt, tSsgdVt(_, _, k), tSrgdVt_v(_, _, k));
            gemm(tiled_mma, rdO(_, _, k), rgdVt(_, _, k), rN);
        }

        // (k) Compute dL/dS = P * N + scale * M * dS.
        // We need (P * M) for g_V too -- compute it in a scratch tensor.
        auto rPM = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        {
            auto rS_rc  = make_tensor(rS.data(),  convert_layout_acc_rowcol(rS.layout()));
            auto rdP_rc = make_tensor(rdP.data(), convert_layout_acc_rowcol(rdP.layout()));  // dS
            auto rM_rc  = make_tensor(rM.data(),  convert_layout_acc_rowcol(rM.layout()));
            auto rN_rc  = make_tensor(rN.data(),  convert_layout_acc_rowcol(rN.layout()));
            auto rPM_rc = make_tensor(rPM.data(), convert_layout_acc_rowcol(rPM.layout()));
            CUTE_UNROLL
            for (int mi = 0; mi < size<0>(rN_rc); ++mi) {
                CUTE_UNROLL
                for (int ni = 0; ni < size<1>(rN_rc); ++ni) {
                    const float p  = rS_rc(mi, ni);
                    const float ds = rdP_rc(mi, ni);
                    const float m  = rM_rc(mi, ni);
                    const float n  = rN_rc(mi, ni);
                    rN_rc(mi, ni)  = p * n + softmax_scale * m * ds;  // dL/dS
                    rPM_rc(mi, ni) = p * m;
                }
            }
        }
        // rN now holds dL/dS, rPM holds P*M.

        // (l) Stage dL/dS into sPdS.
        Tensor rdLdS_bf16 = make_tensor_like<Dtype>(rN);
        convert_type_out(rN, rdLdS_bf16);
        {
            auto src = r2s_thr.retile_S(rdLdS_bf16);
            copy(r2s_tiled, src, tdSsPdS);
        }
        __syncthreads();

        // (m) MMA-gK: rg_K += scale * (dL/dS)^T . Q
        // Note: scale is folded in at the end of the kernel, not per-iter,
        // to save an elementwise rescale. But that means rg_K accumulates
        // (dL/dS)^T . Q without scale. We multiply by scale once at epilogue.
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rPdS_A); ++k) {
            copy(smem_tcopy_PdS_A, tSsPdSt(_, _, k), tSrPdS_A_v(_, _, k));
            copy(smem_tcopy_Qt,    tSsQt(_, _, k),   tSrQt_v(_, _, k));
            gemm(tiled_mma, rPdS_A(_, _, k), rQt(_, _, k), rg_K);
        }

        // (n) Stage P*M into sPdS, overwriting dL/dS.
        Tensor rPM_bf16 = make_tensor_like<Dtype>(rPM);
        convert_type_out(rPM, rPM_bf16);
        __syncthreads();
        {
            auto src = r2s_thr.retile_S(rPM_bf16);
            copy(r2s_tiled, src, tdSsPdS);
        }
        __syncthreads();

        // (o) MMA-gV: rg_V += scale * (P*M)^T . dO
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rPdS_A); ++k) {
            copy(smem_tcopy_PdS_A, tSsPdSt(_, _, k), tSrPdS_A_v(_, _, k));
            copy(smem_tcopy_dOt,   tSsdOt(_, _, k),  tSrdOt_v(_, _, k));
            gemm(tiled_mma, rPdS_A(_, _, k), rdOt(_, _, k), rg_V);
        }

        __syncthreads();
    }

    // --- Epilogue: scale rg_K and rg_V by softmax_scale, then write.
    {
        auto rgK_rc = make_tensor(rg_K.data(), convert_layout_acc_rowcol(rg_K.layout()));
        auto rgV_rc = make_tensor(rg_V.data(), convert_layout_acc_rowcol(rg_V.layout()));
        CUTE_UNROLL
        for (int mi = 0; mi < size<0>(rgK_rc); ++mi) {
            CUTE_UNROLL
            for (int ni = 0; ni < size<1>(rgK_rc); ++ni) {
                rgK_rc(mi, ni) *= softmax_scale;
                rgV_rc(mi, ni) *= softmax_scale;
            }
        }
    }

    auto sdKV = make_tensor(make_smem_ptr(sQ_data), SmemK_t{});
    using SmemCopyAtomDef = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_dKV = make_tiled_copy_C(SmemCopyAtomDef{}, tiled_mma);
    auto smem_thr_copy_dKV   = smem_tiled_copy_dKV.get_thread_slice(threadIdx.x);
    using GmemTiledCopyO_t =
        typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
    GmemTiledCopyO_t gmem_copy_o;
    auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);

    auto ggK_head = make_tensor(make_gmem_ptr(gK_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto ggV_head = make_tensor(make_gmem_ptr(gV_bh),
                                make_shape(N, Int<Headdim>{}),
                                make_stride(Int<Headdim>{}, _1{}));
    auto ggK_tiles = local_tile(ggK_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto ggV_tiles = local_tile(ggV_head, Shape<Int<Bc>, Int<Headdim>>{}, make_coord(_, _0{}));
    auto ggK = ggK_tiles(_, _, kv_block);
    auto ggV = ggV_tiles(_, _, kv_block);

    // Write g_V
    {
        Tensor rgV_out = make_tensor_like<Dtype>(rg_V);
        convert_type_out(rg_V, rgV_out);
        __syncthreads();
        auto dst = smem_thr_copy_dKV.partition_D(sdKV);
        auto src = smem_thr_copy_dKV.retile_S(rgV_out);
        copy(smem_tiled_copy_dKV, src, dst);
        __syncthreads();
        copy(gmem_copy_o, thr_copy_o.partition_S(sdKV), thr_copy_o.partition_D(ggV));
    }
    __syncthreads();
    // Write g_K
    {
        Tensor rgK_out = make_tensor_like<Dtype>(rg_K);
        convert_type_out(rg_K, rgK_out);
        auto dst = smem_thr_copy_dKV.partition_D(sdKV);
        auto src = smem_thr_copy_dKV.retile_S(rgK_out);
        copy(smem_tiled_copy_dKV, src, dst);
        __syncthreads();
        copy(gmem_copy_o, thr_copy_o.partition_S(sdKV), thr_copy_o.partition_D(ggK));
    }
}

// -----------------------------------------------------------------------------
// Launch helpers.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, bool IsCausal>
static void launch_dblbwd_KV(
    const Dtype* Q, const Dtype* K, const Dtype* V, const Dtype* dO,
    const Dtype* gdQ, const Dtype* gdK, const Dtype* gdV,
    const float* L, const float* D,
    Dtype* gK, Dtype* gV,
    int B, int H, int N,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    float softmax_scale, cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kNumThreads = 128;
    constexpr int sQ   = cosize_v<SmemLayoutQ<Br, Headdim, Dtype>>;
    constexpr int sdO  = sQ;
    constexpr int sgdQ = sQ;
    constexpr int sK   = cosize_v<SmemLayoutK<Bc, Headdim, Dtype>>;
    constexpr int sV   = sK;
    constexpr int sgdK = sK;
    constexpr int sgdV = sK;
    constexpr int sPdS = cosize_v<SmemLayoutPdS<Br, Bc, Dtype>>;
    constexpr int smem_bytes =
        (sQ + sdO + sgdQ + sK + sV + sgdK + sgdV + sPdS) * sizeof(Dtype);

    const int num_kv_blocks = (N + Bc - 1) / Bc;
    dim3 grid(num_kv_blocks, B * H);
    dim3 block(kNumThreads);

    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            flash_v9_dblbwd_KV_kernel<Dtype, Headdim, Br, Bc, IsCausal>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    }

    flash_v9_dblbwd_KV_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, dO, gdQ, gdK, gdV, L, D, gK, gV,
            B, H, N, qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride,
            softmax_scale
        );
}

template <typename Dtype>
static void dispatch_dblbwd_KV(
    const Dtype* Q, const Dtype* K, const Dtype* V, const Dtype* dO,
    const Dtype* gdQ, const Dtype* gdK, const Dtype* gdV,
    const float* L, const float* D,
    Dtype* gK, Dtype* gV,
    int B, int H, int N, int Hd,
    int64_t qkv_b_stride, int64_t qkv_h_stride,
    int64_t l_b_stride,   int64_t l_h_stride,
    bool is_causal, float softmax_scale, cudaStream_t stream
) {
    #define LAUNCH(HD, CAUSAL) \
        launch_dblbwd_KV<Dtype, HD, CAUSAL>(Q, K, V, dO, gdQ, gdK, gdV, L, D, \
            gK, gV, B, H, N, qkv_b_stride, qkv_h_stride, l_b_stride, l_h_stride, \
            softmax_scale, stream)
    if (Hd == 64) { if (is_causal) LAUNCH(64, true); else LAUNCH(64, false); }
    else TORCH_CHECK(false, "flash_v9 dblbwd_KV: only headdim=64 supported (D=128 needs 128 KB smem; reduce Bc to 32 to enable)");
    #undef LAUNCH
}

template <typename Dtype, int Headdim>
static void launch_dblbwd_preprocess(
    const Dtype* dO, const Dtype* O, float* D,
    int B, int H, int N,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride,
    cudaStream_t stream
) {
    const int threads = 128;
    const int blocks_x = (N + threads - 1) / threads;
    dim3 grid(blocks_x, B * H);
    flash_v9_dblbwd_preprocess_kernel<Dtype, Headdim>
        <<<grid, threads, 0, stream>>>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride);
}

template <typename Dtype>
static void dispatch_dblbwd_preprocess(
    const Dtype* dO, const Dtype* O, float* D,
    int B, int H, int N, int Hd,
    int64_t bh_stride, int64_t row_stride_qkv, int64_t l_bh_stride,
    cudaStream_t stream
) {
    if      (Hd == 64)  launch_dblbwd_preprocess<Dtype, 64>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else if (Hd == 128) launch_dblbwd_preprocess<Dtype, 128>(dO, O, D, B, H, N, bh_stride, row_stride_qkv, l_bh_stride, stream);
    else TORCH_CHECK(false, "flash_v9 dblbwd preprocess: headdim must be 64 or 128");
}

} // namespace flash_v9


std::vector<torch::Tensor> flash_v9_double_backward_cuda(
    torch::Tensor g_dQ, torch::Tensor g_dK, torch::Tensor g_dV,
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    bool is_causal, double softmax_scale
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(Q.dim() == 4, "Q, K, V must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.size(2) % 64 == 0, "flash_v9 dbl_bwd: N must be a multiple of 64");

    if (!dO.is_contiguous())   dO   = dO.contiguous();
    if (!Q.is_contiguous())    Q    = Q.contiguous();
    if (!K.is_contiguous())    K    = K.contiguous();
    if (!V.is_contiguous())    V    = V.contiguous();
    if (!O.is_contiguous())    O    = O.contiguous();
    if (!L.is_contiguous())    L    = L.contiguous();
    if (!g_dQ.is_contiguous()) g_dQ = g_dQ.contiguous();
    if (!g_dK.is_contiguous()) g_dK = g_dK.contiguous();
    if (!g_dV.is_contiguous()) g_dV = g_dV.contiguous();

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D_dim = Q.size(3);

    auto g_dO = torch::zeros_like(dO);   // until 6g lands
    auto g_Q  = torch::zeros_like(Q);    // until 6g lands
    auto g_K  = torch::empty_like(K);
    auto g_V  = torch::empty_like(V);
    auto D_tensor = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (Q.scalar_type() == torch::kBFloat16) {
        using T = cutlass::bfloat16_t;
        auto P = [](torch::Tensor const& t) { return reinterpret_cast<const T*>(t.data_ptr()); };
        auto W = [](torch::Tensor& t)       { return reinterpret_cast<T*>(t.data_ptr()); };
        flash_v9::dispatch_dblbwd_preprocess<T>(P(dO), P(O), D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D_dim, Q.stride(1), Q.stride(2), N, stream);
        flash_v9::dispatch_dblbwd_KV<T>(P(Q), P(K), P(V), P(dO), P(g_dQ), P(g_dK), P(g_dV),
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), W(g_K), W(g_V),
            (int)B, (int)H, (int)N, (int)D_dim, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
    } else if (Q.scalar_type() == torch::kHalf) {
        using T = cutlass::half_t;
        auto P = [](torch::Tensor const& t) { return reinterpret_cast<const T*>(t.data_ptr()); };
        auto W = [](torch::Tensor& t)       { return reinterpret_cast<T*>(t.data_ptr()); };
        flash_v9::dispatch_dblbwd_preprocess<T>(P(dO), P(O), D_tensor.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D_dim, Q.stride(1), Q.stride(2), N, stream);
        flash_v9::dispatch_dblbwd_KV<T>(P(Q), P(K), P(V), P(dO), P(g_dQ), P(g_dK), P(g_dV),
            L.data_ptr<float>(), D_tensor.data_ptr<float>(), W(g_K), W(g_V),
            (int)B, (int)H, (int)N, (int)D_dim, Q.stride(0), Q.stride(1),
            L.stride(0), L.stride(1), is_causal, (float)softmax_scale, stream);
    } else {
        TORCH_CHECK(false, "flash_v9 dbl_bwd: only bf16 and fp16 supported");
    }

    return {g_dO, g_Q, g_K, g_V};
}
