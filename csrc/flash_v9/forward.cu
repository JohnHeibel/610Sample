#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

#include "common.cuh"

namespace flash_v9 {

using namespace cute;

// -----------------------------------------------------------------------------
// Forward kernel (Algorithm 1).
// Sub-commit 3b: gmem -> smem cp.async copies for Q, K, V tiles (first K/V
// tile only). No compute. Output O and L remain zero (set by host).
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_fwd_kernel(
    const Dtype* __restrict__ Q_ptr,  // [B, H, N, D]
    const Dtype* __restrict__ K_ptr,  // [B, H, N, D]
    const Dtype* __restrict__ V_ptr,  // [B, H, N, D]
    Dtype*       __restrict__ O_ptr,
    float*       __restrict__ L_ptr,
    int B, int H, int N, int /*D*/,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t /*qkv_row_stride*/,
    int64_t o_batch_stride,   int64_t o_head_stride,   int64_t /*o_row_stride*/,
    int64_t l_batch_stride,   int64_t l_head_stride,
    float softmax_scale
) {
    constexpr int NumThreads = 128;
    using SmemQ_t = SmemLayoutQ<Br, Headdim, Dtype>;
    using SmemK_t = SmemLayoutK<Bc, Headdim, Dtype>;
    using SmemV_t = SmemLayoutV<Bc, Headdim, Dtype>;
    using GmemCopy_t =
        typename GmemTiledCopyTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;

    extern __shared__ char smem_buf[];
    Dtype* sQ_data = reinterpret_cast<Dtype*>(smem_buf);
    Dtype* sK_data = sQ_data + cosize_v<SmemQ_t>;
    Dtype* sV_data = sK_data + cosize_v<SmemK_t>;

    auto sQ = make_tensor(make_smem_ptr(sQ_data), SmemQ_t{});
    auto sK = make_tensor(make_smem_ptr(sK_data), SmemK_t{});
    auto sV = make_tensor(make_smem_ptr(sV_data), SmemV_t{});

    const int bh      = blockIdx.y;
    const int b       = bh / H;
    const int h       = bh % H;
    const int q_block = blockIdx.x;

    const Dtype* Q_bh = Q_ptr + b * qkv_batch_stride + h * qkv_head_stride;
    const Dtype* K_bh = K_ptr + b * qkv_batch_stride + h * qkv_head_stride;
    const Dtype* V_bh = V_ptr + b * qkv_batch_stride + h * qkv_head_stride;

    auto gQ_head = make_tensor(make_gmem_ptr(Q_bh),
                               make_shape(N, Int<Headdim>{}),
                               make_stride(Int<Headdim>{}, _1{}));
    auto gK_head = make_tensor(make_gmem_ptr(K_bh),
                               make_shape(N, Int<Headdim>{}),
                               make_stride(Int<Headdim>{}, _1{}));
    auto gV_head = make_tensor(make_gmem_ptr(V_bh),
                               make_shape(N, Int<Headdim>{}),
                               make_stride(Int<Headdim>{}, _1{}));

    // ------------------------------------------------------------------------
    // Build gmem tile views for Q (single block) and K/V (all blocks).
    // ------------------------------------------------------------------------
    auto gQ_tiles = local_tile(gQ_head, Shape<Int<Br>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gK_tiles = local_tile(gK_head, Shape<Int<Bc>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gV_tiles = local_tile(gV_head, Shape<Int<Bc>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gQ = gQ_tiles(_, _, q_block);
    const int num_kv_tiles = size<2>(gK_tiles);

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    // ------------------------------------------------------------------------
    // Load Q once (it's the same across all K/V tiles).
    // ------------------------------------------------------------------------
    auto tQgQ = thr_copy.partition_S(gQ);
    auto tQsQ = thr_copy.partition_D(sQ);
    copy(gmem_copy_qkv, tQgQ, tQsQ);

    // ------------------------------------------------------------------------
    // 3c/3e components reused across the K/V loop: TiledMma, smem copies.
    // ------------------------------------------------------------------------
    constexpr int NumWarps = NumThreads / 32;
    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto rQ = thr_mma.partition_fragment_A(sQ);
    auto rK = thr_mma.partition_fragment_B(sK);
    auto sVt = make_tensor(make_smem_ptr(sV_data),
                           SmemLayoutVt<Bc, Headdim, Dtype>{});
    auto rV = thr_mma.partition_fragment_B(sVt);

    using SmemCopyAtom_QK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom_QK{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom_QK{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(threadIdx.x);
    auto tSsQ      = smem_thr_copy_Q.partition_S(sQ);
    auto tSsK      = smem_thr_copy_K.partition_S(sK);
    auto tSrQ_view = smem_thr_copy_Q.retile_D(rQ);
    auto tSrK_view = smem_thr_copy_K.retile_D(rK);

    using SmemCopyAtom_V = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom_V{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto tOsV      = smem_thr_copy_V.partition_S(sVt);
    auto tOrV_view = smem_thr_copy_V.retile_D(rV);

    // rO accumulator (fp32) lives across all K/V tiles.
    auto rO = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Headdim>>{});
    clear(rO);

    // Softmax state lives across all K/V tiles.
    constexpr int kNRows = 2;
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;
    Softmax<kNRows> softmax(softmax_scale_log2);

    // ------------------------------------------------------------------------
    // 3f: outer K/V tile loop.
    // For each kv tile: cp.async K,V -> smem; Q.K^T -> rS; online softmax;
    // P.V -> rO (accumulating).
    // ------------------------------------------------------------------------
    for (int kv = 0; kv < num_kv_tiles; ++kv) {
        // Load K and V tile from gmem to smem.
        auto gK = gK_tiles(_, _, kv);
        auto gV = gV_tiles(_, _, kv);
        auto tKgK = thr_copy.partition_S(gK);
        auto tKsK = thr_copy.partition_D(sK);
        auto tVgV = thr_copy.partition_S(gV);
        auto tVsV = thr_copy.partition_D(sV);
        copy(gmem_copy_qkv, tKgK, tKsK);
        copy(gmem_copy_qkv, tVgV, tVsV);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Q . K^T MMA -> rS.
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
        clear(rS);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(rQ); ++k) {
            copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_view(_, _, k));
            copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_view(_, _, k));
            gemm(tiled_mma, rQ(_, _, k), rK(_, _, k), rS);
        }

        // Online softmax + rO rescale.
        // - First iter: initialize row_max, row_sum; no rescale.
        // - Later iters: update row_max, compute exp(prev_max - new_max) scale,
        //                rescale rO by scale, update row_sum.
        if (kv == 0) {
            softmax.template max_get_scale</*Is_first=*/true>(rS);
            softmax.template online_softmax</*Is_first=*/true>(rS);
        } else {
            auto scores_scale = softmax.template max_get_scale</*Is_first=*/false>(rS);
            softmax.rescale_o(rO, scores_scale);
            softmax.template online_softmax</*Is_first=*/false>(rS);
        }

        // Convert rS -> tOrP (bf16/fp16 A-operand layout) for P.V MMA.
        Tensor tOrP_acc = make_tensor(
            rS.data(),
            convert_layout_acc_Aregs<TiledMma_t>(rS.layout())
        );
        Tensor tOrP = make_tensor_like<Dtype>(tOrP_acc);
        convert_type_out(tOrP_acc, tOrP);

        // P . V MMA -> rO (accumulating).
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k) {
            copy(smem_tiled_copy_V, tOsV(_, _, k), tOrV_view(_, _, k));
            gemm(tiled_mma, tOrP(_, _, k), rV(_, _, k), rO);
        }

        // Sync before next iteration overwrites sK / sV.
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Epilogue: finalize softmax, write O and L. Same as in 3e.
    // ------------------------------------------------------------------------
    auto scores_scale = softmax.finalize(/*final_scale=*/1.0f);
    softmax.rescale_o(rO, scores_scale);

    Tensor rO_out = make_tensor_like<Dtype>(rO);
    convert_type_out(rO, rO_out);

    __syncthreads();
    auto sO = make_tensor(make_smem_ptr(sQ_data), SmemQ_t{});
    using SmemCopyAtom_O = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtom_O{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto tOsO_dst  = smem_thr_copy_O.partition_D(sO);
    auto tOrO_view = smem_thr_copy_O.retile_S(rO_out);
    copy(smem_tiled_copy_O, tOrO_view, tOsO_dst);
    __syncthreads();

    Dtype* O_bh = O_ptr + b * o_batch_stride + h * o_head_stride;
    auto gO_head = make_tensor(make_gmem_ptr(O_bh),
                               make_shape(N, Int<Headdim>{}),
                               make_stride(Int<Headdim>{}, _1{}));
    auto gO_tiles = local_tile(gO_head, Shape<Int<Br>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gO = gO_tiles(_, _, q_block);

    using GmemTiledCopyO_t =
        typename GmemTiledCopyOTraits<Headdim, NumThreads, Dtype>::GmemTiledCopy;
    GmemTiledCopyO_t gmem_copy_o;
    auto thr_copy_o = gmem_copy_o.get_thread_slice(threadIdx.x);
    auto tOsO_src = thr_copy_o.partition_S(sO);
    auto tOgO     = thr_copy_o.partition_D(gO);
    copy(gmem_copy_o, tOsO_src, tOgO);

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    if ((lane_id & 3) == 0) {
        int row_lo = warp_id * 16 + (lane_id >> 2);
        int row_hi = row_lo + 8;
        int q_row_lo = q_block * Br + row_lo;
        int q_row_hi = q_block * Br + row_hi;
        int64_t L_base = b * l_batch_stride + h * l_head_stride;
        if (q_row_lo < N) L_ptr[L_base + q_row_lo] = softmax.row_sum(0);
        if (q_row_hi < N) L_ptr[L_base + q_row_hi] = softmax.row_sum(1);
    }
}

template <typename Dtype, int Headdim, bool IsCausal>
static void launch_fwd(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    Dtype* O, float* L,
    int B, int H, int N, int D,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t qkv_row_stride,
    int64_t o_batch_stride,   int64_t o_head_stride,   int64_t o_row_stride,
    int64_t l_batch_stride,   int64_t l_head_stride,
    float softmax_scale, cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = 64;
    constexpr int kNumThreads = 128;

    constexpr int smem_bytes = SmemSize<Br, Bc, Headdim, Dtype>::total_bytes;

    const int num_q_blocks = (N + Br - 1) / Br;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(kNumThreads);

    flash_v9_fwd_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, L, B, H, N, D,
            qkv_batch_stride, qkv_head_stride, qkv_row_stride,
            o_batch_stride,   o_head_stride,   o_row_stride,
            l_batch_stride,   l_head_stride,
            softmax_scale
        );
}

template <typename Dtype>
static void dispatch_fwd_headdim(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    Dtype* O, float* L,
    int B, int H, int N, int D,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t qkv_row_stride,
    int64_t o_batch_stride,   int64_t o_head_stride,   int64_t o_row_stride,
    int64_t l_batch_stride,   int64_t l_head_stride,
    bool is_causal, float softmax_scale, cudaStream_t stream
) {
    #define LAUNCH(HD, CAUSAL) \
        launch_fwd<Dtype, HD, CAUSAL>( \
            Q, K, V, O, L, B, H, N, D, \
            qkv_batch_stride, qkv_head_stride, qkv_row_stride, \
            o_batch_stride,   o_head_stride,   o_row_stride, \
            l_batch_stride,   l_head_stride, \
            softmax_scale, stream)

    if (D == 64) {
        if (is_causal) LAUNCH(64, true);  else LAUNCH(64, false);
    } else if (D == 128) {
        if (is_causal) LAUNCH(128, true); else LAUNCH(128, false);
    } else {
        TORCH_CHECK(false, "flash_v9: headdim must be 64 or 128, got ", D);
    }
    #undef LAUNCH
}

} // namespace flash_v9


std::vector<torch::Tensor> flash_v9_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool is_causal, double softmax_scale
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q, K, V must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.scalar_type() == K.scalar_type() &&
                Q.scalar_type() == V.scalar_type(),
                "Q, K, V must have the same dtype");
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
                "B must match across Q, K, V");
    TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1),
                "H must match across Q, K, V");
    TORCH_CHECK(K.size(2) == V.size(2),
                "N_kv must match between K and V");
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3),
                "D must match across Q, K, V");
    TORCH_CHECK(Q.size(2) == K.size(2),
                "flash_v9: fixed-seqlen only (N_q == N_kv) in v9");
    // v9 requires N divisible by both Br=64 (Q tile size) and Bc=64 (K/V tile
    // size). Masking for partial tiles lands in 3g.
    TORCH_CHECK(Q.size(2) % 64 == 0,
                "flash_v9: N must be a multiple of 64 (until 3g adds masking)");

    const int64_t B = Q.size(0);
    const int64_t H = Q.size(1);
    const int64_t N = Q.size(2);
    const int64_t D = Q.size(3);

    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, H, N}, Q.options().dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (Q.scalar_type() == torch::kBFloat16) {
        flash_v9::dispatch_fwd_headdim<cutlass::bfloat16_t>(
            reinterpret_cast<const cutlass::bfloat16_t*>(Q.data_ptr()),
            reinterpret_cast<const cutlass::bfloat16_t*>(K.data_ptr()),
            reinterpret_cast<const cutlass::bfloat16_t*>(V.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(O.data_ptr()),
            L.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            is_causal, (float)softmax_scale, stream
        );
    } else if (Q.scalar_type() == torch::kHalf) {
        flash_v9::dispatch_fwd_headdim<cutlass::half_t>(
            reinterpret_cast<const cutlass::half_t*>(Q.data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(K.data_ptr()),
            reinterpret_cast<const cutlass::half_t*>(V.data_ptr()),
            reinterpret_cast<cutlass::half_t*>(O.data_ptr()),
            L.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            is_causal, (float)softmax_scale, stream
        );
    } else {
        TORCH_CHECK(false, "flash_v9: only bf16 and fp16 are supported");
    }

    return {O, L};
}
