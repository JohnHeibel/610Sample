#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> flash_v9_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    bool is_causal, double softmax_scale);

std::vector<torch::Tensor> flash_v9_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    bool is_causal, double softmax_scale);

std::vector<torch::Tensor> flash_v9_double_backward_cuda(
    torch::Tensor g_dQ, torch::Tensor g_dK, torch::Tensor g_dV,
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    bool is_causal, double softmax_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_v9_forward_cuda,
          "Flash v9 forward (CUTLASS, FA2-style, bf16/fp16, causal+non-causal)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("is_causal"), py::arg("softmax_scale"));
    m.def("backward", &flash_v9_backward_cuda,
          "Flash v9 backward (CUTLASS, FA2-style)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("is_causal"), py::arg("softmax_scale"));
    m.def("double_backward", &flash_v9_double_backward_cuda,
          "Flash v9 double backward (CUTLASS, novel IO-aware HVP)",
          py::arg("g_dQ"), py::arg("g_dK"), py::arg("g_dV"),
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("is_causal"), py::arg("softmax_scale"));
}
