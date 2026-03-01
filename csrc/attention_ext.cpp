#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> v7_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    int window_size, int chunk_id);

std::vector<torch::Tensor> v7_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    int window_size, int chunk_id);

std::vector<torch::Tensor> v7_double_backward_cuda(
    torch::Tensor g_dQ, torch::Tensor g_dK, torch::Tensor g_dV,
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor O, torch::Tensor L,
    int window_size, int chunk_id);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("v7_forward", &v7_forward_cuda,
          "V7 rectangular sliding-window flash forward (fp16/bf16 WMMA)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("window_size"), py::arg("chunk_id"));
    m.def("v7_backward", &v7_backward_cuda,
          "V7 rectangular sliding-window flash backward (fp16/bf16 WMMA)",
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("window_size"), py::arg("chunk_id"));
    m.def("v7_double_backward", &v7_double_backward_cuda,
          "V7 rectangular sliding-window flash double backward (fp16/bf16 WMMA)",
          py::arg("g_dQ"), py::arg("g_dK"), py::arg("g_dV"),
          py::arg("dO"), py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("O"), py::arg("L"),
          py::arg("window_size"), py::arg("chunk_id"));
}
