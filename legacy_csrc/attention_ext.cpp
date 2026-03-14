#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V);

std::vector<torch::Tensor> attention_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K,
    torch::Tensor V, torch::Tensor P);

std::vector<torch::Tensor> attention_double_backward_cuda(
    torch::Tensor g_dQ, torch::Tensor g_dK, torch::Tensor g_dV,
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K,
    torch::Tensor V, torch::Tensor P, torch::Tensor dS, torch::Tensor dP);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward_cuda, "attention forward (CUDA)");
    m.def("backward", &attention_backward_cuda, "attention backward (CUDA)");
    m.def("double_backward", &attention_double_backward_cuda, "attention double backward (CUDA)");
}
