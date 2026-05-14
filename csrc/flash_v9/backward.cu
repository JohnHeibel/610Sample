#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "common.cuh"

// Algorithm 2 (backward) placeholder.
// Replaced in Commit 4 with the CUTLASS-based bwd kernel (recompute attention,
// two-pass dQ accumulation).
std::vector<torch::Tensor> flash_v9_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor /*O*/, torch::Tensor /*L*/,
    bool /*is_causal*/, double /*softmax_scale*/
) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && dO.is_cuda(),
                "Q, K, V, dO must be CUDA tensors");
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);
    return {dQ, dK, dV};
}
