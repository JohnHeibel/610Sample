#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

__global__ void compute_dV_kernel(
    const float* P, const float* dO, float* dV,
    int B, int H, int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * D) return;

    int d = idx % D;
    int j = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int p_base  = ((b * H + h) * N) * N;
    int do_base = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int i = 0; i < N; i++)
        sum += P[p_base + i * N + j] * dO[do_base + i * D + d];

    dV[idx] = sum;
}

__global__ void compute_dP_kernel(
    const float* dO, const float* V, float* dP,
    int B, int H, int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * N) return;

    int j = idx % N;
    int i = (idx / N) % N;
    int h = (idx / (N * N)) % H;
    int b = idx / (H * N * N);

    int do_off = ((b * H + h) * N + i) * D;
    int v_off  = ((b * H + h) * N + j) * D;

    float sum = 0.f;
    for (int d = 0; d < D; d++)
        sum += dO[do_off + d] * V[v_off + d];

    dP[idx] = sum;
}

__global__ void softmax_backward_kernel(
    const float* P, const float* dP, float* dS,
    int B, int H, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;

    int off = row * N;

    float dot = 0.f;
    for (int j = 0; j < N; j++)
        dot += P[off + j] * dP[off + j];

    for (int j = 0; j < N; j++)
        dS[off + j] = P[off + j] * (dP[off + j] - dot);
}

__global__ void compute_dQ_kernel(
    const float* dS, const float* K, float* dQ,
    int B, int H, int N, int D, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * D) return;

    int d = idx % D;
    int i = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int ds_off = ((b * H + h) * N + i) * N;
    int k_base = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int j = 0; j < N; j++)
        sum += dS[ds_off + j] * K[k_base + j * D + d];

    dQ[idx] = sum * scale;
}

__global__ void compute_dK_kernel(
    const float* dS, const float* Q, float* dK,
    int B, int H, int N, int D, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * D) return;

    int d = idx % D;
    int j = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int ds_base = ((b * H + h) * N) * N;
    int q_base  = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int i = 0; i < N; i++)
        sum += dS[ds_base + i * N + j] * Q[q_base + i * D + d];

    dK[idx] = sum * scale;
}

std::vector<at::Tensor> attention_backward_cuda(
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V, at::Tensor P)
{
    TORCH_CHECK(dO.is_cuda() && Q.is_cuda() && K.is_cuda() && V.is_cuda() && P.is_cuda());
    TORCH_CHECK(dO.dtype() == at::kFloat);

    dO = dO.contiguous(); Q = Q.contiguous();
    K  = K.contiguous();  V = V.contiguous();
    P  = P.contiguous();

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    float scale = 1.f / sqrtf((float)D);

    auto opts = Q.options();
    auto dV = at::empty({B, H, N, D}, opts);
    auto dP = at::empty({B, H, N, N}, opts);
    auto dS = at::empty({B, H, N, N}, opts);
    auto dQ = at::empty({B, H, N, D}, opts);
    auto dK = at::empty({B, H, N, D}, opts);

    auto grid = [](int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; };

    compute_dV_kernel<<<grid(B*H*N*D), BLOCK_SIZE>>>(P.data_ptr<float>(), dO.data_ptr<float>(), dV.data_ptr<float>(), B, H, N, D);
    CUDA_CHECK(cudaGetLastError());

    compute_dP_kernel<<<grid(B*H*N*N), BLOCK_SIZE>>>(dO.data_ptr<float>(), V.data_ptr<float>(), dP.data_ptr<float>(), B, H, N, D);
    CUDA_CHECK(cudaGetLastError());

    softmax_backward_kernel<<<grid(B*H*N), BLOCK_SIZE>>>(P.data_ptr<float>(), dP.data_ptr<float>(), dS.data_ptr<float>(), B, H, N);
    CUDA_CHECK(cudaGetLastError());

    compute_dQ_kernel<<<grid(B*H*N*D), BLOCK_SIZE>>>(dS.data_ptr<float>(), K.data_ptr<float>(), dQ.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    compute_dK_kernel<<<grid(B*H*N*D), BLOCK_SIZE>>>(dS.data_ptr<float>(), Q.data_ptr<float>(), dK.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {dQ, dK, dV, dS, dP};
}
