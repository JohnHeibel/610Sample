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

    // Tile the head slice into (Br, Headdim) Q blocks and (Bc, Headdim) K/V
    // blocks. For 3b we load only the first K/V tile.
    auto gQ_tiles = local_tile(gQ_head, Shape<Int<Br>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gK_tiles = local_tile(gK_head, Shape<Int<Bc>, Int<Headdim>>{},
                               make_coord(_, _0{}));
    auto gV_tiles = local_tile(gV_head, Shape<Int<Bc>, Int<Headdim>>{},
                               make_coord(_, _0{}));

    auto gQ = gQ_tiles(_, _, q_block);  // (Br, Headdim)
    auto gK = gK_tiles(_, _, 0);        // (Bc, Headdim)
    auto gV = gV_tiles(_, _, 0);        // (Bc, Headdim)

    GmemCopy_t gmem_copy_qkv;
    auto thr_copy = gmem_copy_qkv.get_thread_slice(threadIdx.x);

    auto tQgQ = thr_copy.partition_S(gQ);
    auto tQsQ = thr_copy.partition_D(sQ);
    copy(gmem_copy_qkv, tQgQ, tQsQ);

    auto tKgK = thr_copy.partition_S(gK);
    auto tKsK = thr_copy.partition_D(sK);
    copy(gmem_copy_qkv, tKgK, tKsK);

    auto tVgV = thr_copy.partition_S(gV);
    auto tVsV = thr_copy.partition_D(sV);
    copy(gmem_copy_qkv, tVgV, tVsV);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // ------------------------------------------------------------------------
    // 3c: Q . K^T MMA. Result S is in registers (rS), shape (Br, Bc) split
    // across the warps of the TiledMma. No output write; correctness is
    // verified implicitly through O at commit 3e.
    // ------------------------------------------------------------------------
    constexpr int NumWarps = NumThreads / 32;
    using TiledMma_t = TiledMma_SM80<Dtype, NumWarps>;
    TiledMma_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto rS = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Bc>>{});
    auto rQ = thr_mma.partition_fragment_A(sQ);
    auto rK = thr_mma.partition_fragment_B(sK);
    clear(rS);

    using SmemCopyAtom_QK = Copy_Atom<SM75_U32x4_LDSM_N, Dtype>;
    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom_QK{}, tiled_mma);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom_QK{}, tiled_mma);
    auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(threadIdx.x);
    auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(threadIdx.x);

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

    // ------------------------------------------------------------------------
    // 3d: online softmax for the (single) K/V tile.
    // softmax_scale_log2 = softmax_scale * log2(e) so we can use exp2f.
    // ------------------------------------------------------------------------
    constexpr int kNRows = 2;  // for SM80 16x16 atom, MMA_M=1 -> 2 rows / thread
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;
    Softmax<kNRows> softmax(softmax_scale_log2);
    softmax.template max_get_scale</*Is_first=*/true>(rS);
    softmax.template online_softmax</*Is_first=*/true>(rS);

    // ------------------------------------------------------------------------
    // 3e: P . V MMA -> rO accumulator (fp32), finalize softmax, write O+L.
    // ------------------------------------------------------------------------

    // (1) Convert rS to A-operand layout for P.V MMA and cast fp32 -> bf16.
    Tensor tOrP_acc = make_tensor(
        rS.data(),
        convert_layout_acc_Aregs<TiledMma_t>(rS.layout())
    );
    Tensor tOrP = make_tensor_like<Dtype>(tOrP_acc);
    convert_type_out(tOrP_acc, tOrP);

    // (2) Allocate rO fp32 accumulator for P.V output (Br x Headdim per warp-tuple).
    auto rO = partition_fragment_C(tiled_mma, Shape<Int<Br>, Int<Headdim>>{});
    clear(rO);

    // (3) Smem->reg loads for V (operand B in P.V). The physical sV is
    // (Bc, Headdim) row-major; the MMA's B operand wants (N=Headdim, K=Bc).
    // sVt is the same bytes viewed as (Headdim, Bc); LDSM_T transpose-loads.
    auto sVt = make_tensor(make_smem_ptr(sV_data),
                           SmemLayoutVt<Bc, Headdim, Dtype>{});
    using SmemCopyAtom_V = Copy_Atom<SM75_U16x8_LDSM_T, Dtype>;
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom_V{}, tiled_mma);
    auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(threadIdx.x);
    auto rV = thr_mma.partition_fragment_B(sVt);
    auto tOsV       = smem_thr_copy_V.partition_S(sVt);
    auto tOrV_view  = smem_thr_copy_V.retile_D(rV);

    CUTE_UNROLL
    for (int k = 0; k < size<2>(tOrP); ++k) {
        copy(smem_tiled_copy_V, tOsV(_, _, k), tOrV_view(_, _, k));
        gemm(tiled_mma, tOrP(_, _, k), rV(_, _, k), rO);
    }

    // (4) Finalize softmax: quad-allreduce row_sum, divide rO by sum, and
    // overwrite softmax.row_sum with L = m + log(sum).
    auto scores_scale = softmax.finalize(/*final_scale=*/1.0f);
    softmax.rescale_o(rO, scores_scale);

    // (5) Convert rO fp32 -> bf16/fp16.
    Tensor rO_out = make_tensor_like<Dtype>(rO);
    convert_type_out(rO, rO_out);

    // (6) Store rO_out to smem (reuse sQ area; Q is no longer needed).
    __syncthreads();
    auto sO = make_tensor(make_smem_ptr(sQ_data), SmemQ_t{});
    using SmemCopyAtom_O = Copy_Atom<DefaultCopy, Dtype>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtom_O{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(threadIdx.x);
    auto tOsO_dst    = smem_thr_copy_O.partition_D(sO);
    auto tOrO_view   = smem_thr_copy_O.retile_S(rO_out);
    copy(smem_tiled_copy_O, tOrO_view, tOsO_dst);
    __syncthreads();

    // (7) Store smem -> gmem with vectorized non-async copies.
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

    // (8) Write L. After finalize(), softmax.row_sum holds L = m + log(sum)
    // per row, replicated across 4 lanes of each quad. Each thread owns 2
    // rows (lane_id/4 and lane_id/4 + 8 within its warp). Lane (lane_id%4==0)
    // is responsible for writing both rows of its quad.
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
