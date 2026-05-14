#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

#include "common.cuh"

namespace flash_v9 {

// -----------------------------------------------------------------------------
// Kernel launch skeleton for Algorithm 1 (forward).
//
// Templated on dtype, head dim, tile sizes (Br, Bc), and is_causal so the
// compiler can specialize layouts, smem usage, and mask logic. The body is
// empty in commit 3a; subsequent commits fill in:
//   3b: cp.async gmem -> smem copies for Q, K, V tiles
//   3c: Q.K^T MMA via CuTe atoms (S = Q K^T)
//   3d: online softmax (running m, l; rescale O)
//   3e: P.V MMA, write output
//   3f: multi-tile inner loop
//   3g: causal masking variant
//   3h: tuning (occupancy, smem budget, cp.async stages)
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, int Br, int Bc, bool IsCausal>
__global__ void flash_v9_fwd_kernel(
    const Dtype* __restrict__ Q,  // [B, H, N, D]
    const Dtype* __restrict__ K,  // [B, H, N, D]
    const Dtype* __restrict__ V,  // [B, H, N, D]
    Dtype*       __restrict__ O,  // [B, H, N, D]
    float*       __restrict__ L,  // [B, H, N]   (logsumexp)
    int B, int H, int N, int /*D*/,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t qkv_row_stride,
    int64_t o_batch_stride,   int64_t o_head_stride,   int64_t o_row_stride,
    int64_t l_batch_stride,   int64_t l_head_stride,
    float softmax_scale
) {
    // Block geometry: gridDim.x indexes the query-row tile, gridDim.y the
    // (batch, head) pair. This matches FA2's launch convention.
    //
    // Currently a no-op. The .cu file compiles, links into flash_v9_cuda,
    // and the launch in flash_v9_forward_cuda produces well-formed outputs
    // (O and L are still allocated as zeros at host side until commit 3e).
    (void)Q; (void)K; (void)V; (void)O; (void)L;
    (void)B; (void)H; (void)N;
    (void)qkv_batch_stride; (void)qkv_head_stride; (void)qkv_row_stride;
    (void)o_batch_stride;   (void)o_head_stride;   (void)o_row_stride;
    (void)l_batch_stride;   (void)l_head_stride;
    (void)softmax_scale;
}

// -----------------------------------------------------------------------------
// Dispatch table: dtype x headdim x is_causal.
// Tile sizes (Br, Bc) are picked per headdim. Initial values match FA2's
// defaults for sm_80 and will be retuned in commit 3h.
// -----------------------------------------------------------------------------
template <typename Dtype, int Headdim, bool IsCausal>
static void launch_fwd(
    const Dtype* Q, const Dtype* K, const Dtype* V,
    Dtype* O, float* L,
    int B, int H, int N, int D,
    int64_t qkv_batch_stride, int64_t qkv_head_stride, int64_t qkv_row_stride,
    int64_t o_batch_stride,   int64_t o_head_stride,   int64_t o_row_stride,
    int64_t l_batch_stride,   int64_t l_head_stride,
    float softmax_scale,
    cudaStream_t stream
) {
    constexpr int Br = 64;
    constexpr int Bc = (Headdim <= 64) ? 64 : 64;  // retuned in 3h
    constexpr int kNumThreads = 128;               // 4 warps; retuned in 3h

    const int num_q_blocks = (N + Br - 1) / Br;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(kNumThreads);

    flash_v9_fwd_kernel<Dtype, Headdim, Br, Bc, IsCausal>
        <<<grid, block, /*smem=*/0, stream>>>(
            Q, K, V, O, L,
            B, H, N, D,
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
