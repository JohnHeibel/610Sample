#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#include "common.cuh"

// Algorithm 3 (double backward) placeholder.
// Replaced in Commit 5 with the novel IO-aware HVP kernel: no O(N^2)
// materialization; S, P, dP, dS, and their VJPs are recomputed online.
// Inputs:  upstream g_dQ, g_dK, g_dV (gradients of a downstream loss w.r.t.
//          dQ, dK, dV).
// Outputs: g_dO, g_Q, g_K, g_V (those gradients propagated back to dO, Q, K, V).
std::vector<torch::Tensor> flash_v9_double_backward_cuda(
    torch::Tensor g_dQ, torch::Tensor g_dK, torch::Tensor g_dV,
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor /*O*/, torch::Tensor /*L*/,
    bool /*is_causal*/, double /*softmax_scale*/
) {
    TORCH_CHECK(Q.is_cuda(), "tensors must be CUDA");
    auto g_dO = torch::zeros_like(dO);
    auto g_Q  = torch::zeros_like(Q);
    auto g_K  = torch::zeros_like(K);
    auto g_V  = torch::zeros_like(V);
    return {g_dO, g_Q, g_K, g_V};
}
