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
    Dtype*       __restrict__ /*O_ptr*/,
    float*       __restrict__ /*L_ptr*/,
    int B, int H, int N, int /*D*/,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t /*qkv_row_stride*/,
    int64_t /*o_batch_stride*/, int64_t /*o_head_stride*/, int64_t /*o_row_stride*/,
    int64_t /*l_batch_stride*/, int64_t /*l_head_stride*/,
    float /*softmax_scale*/
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
    // rS now holds Q . K^T for the (q_block, first_kv_tile) pair.
    // No output yet; 3d adds softmax, 3e adds P.V and writes O.
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

    // Until commit 3e fills in the kernel, fill outputs with zeros so the
    // wrapper still produces well-defined tensors.
    O.zero_();
    L.zero_();

    auto stream = at::cuda::getCurrentCUDAStream();

    if (Q.scalar_type() == torch::kBFloat16) {
        flash_v9::dispatch_fwd_headdim<__nv_bfloat16>(
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
            L.data_ptr<float>(),
            (int)B, (int)H, (int)N, (int)D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            is_causal, (float)softmax_scale, stream
        );
    } else if (Q.scalar_type() == torch::kHalf) {
        flash_v9::dispatch_fwd_headdim<__half>(
            reinterpret_cast<const __half*>(Q.data_ptr()),
            reinterpret_cast<const __half*>(K.data_ptr()),
            reinterpret_cast<const __half*>(V.data_ptr()),
            reinterpret_cast<__half*>(O.data_ptr()),
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
