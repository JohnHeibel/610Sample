#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "common.cuh"

// Algorithm 1 (forward) placeholder.
// Replaced in Commit 3 with the CUTLASS-based fused fwd kernel.
std::vector<torch::Tensor> flash_v9_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool /*is_causal*/, double /*softmax_scale*/
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q, K, V must be 4D [B, H, N, D]");
    const int64_t B   = Q.size(0);
    const int64_t H   = Q.size(1);
    const int64_t N_q = Q.size(2);

    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({B, H, N_q}, Q.options().dtype(torch::kFloat32));
    return {O, L};
}
