#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

__global__ void matmul_ABt_kernel(
    const float* A, const float* B, float* C,
    int Batch, int H, int N, int D, float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Batch * H * N * N) return;

    int j = idx % N;
    int i = (idx / N) % N;
    int h = (idx / (N * N)) % H;
    int b = idx / (H * N * N);

    int a_off = ((b * H + h) * N + i) * D;
    int b_off = ((b * H + h) * N + j) * D;

    float sum = 0.f;
    for (int d = 0; d < D; d++)
        sum += A[a_off + d] * B[b_off + d];

    C[idx] = sum * s;
}

__global__ void matmul_AB_kernel(
    const float* A, const float* B, float* C,
    int Batch, int H, int N, int D, float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Batch * H * N * D) return;

    int d = idx % D;
    int i = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int a_off  = ((b * H + h) * N + i) * N;
    int b_base = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int j = 0; j < N; j++)
        sum += A[a_off + j] * B[b_base + j * D + d];

    C[idx] = sum * s;
}

__global__ void matmul_AtB_kernel(
    const float* A, const float* B, float* C,
    int Batch, int H, int N, int D, float s)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Batch * H * N * D) return;

    int d = idx % D;
    int j = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int a_base = ((b * H + h) * N) * N;
    int b_base = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int i = 0; i < N; i++)
        sum += A[a_base + i * N + j] * B[b_base + i * D + d];

    C[idx] = sum * s;
}

__global__ void softmax_double_bwd_kernel(
    const float* g_dS, const float* P, const float* dP,
    float* g_dP, float* g_P,
    int B, int H, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;

    int off = row * N;

    float dot = 0.f, dot2 = 0.f;
    for (int k = 0; k < N; k++) {
        float p = P[off + k];
        dot  += p * dP[off + k];
        dot2 += p * g_dS[off + k];
    }

    for (int j = 0; j < N; j++) {
        float p   = P[off + j];
        float dp  = dP[off + j];
        float gds = g_dS[off + j];

        g_dP[off + j] = p * (gds - dot2);
        g_P[off + j]  = gds * (dp - dot) - dp * dot2;
    }
}

__global__ void softmax_bwd_kernel(
    const float* P, const float* g_P, float* g_S,
    int B, int H, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;

    int off = row * N;

    float dot = 0.f;
    for (int k = 0; k < N; k++)
        dot += P[off + k] * g_P[off + k];

    for (int j = 0; j < N; j++)
        g_S[off + j] = P[off + j] * (g_P[off + j] - dot);
}

std::vector<at::Tensor> attention_double_backward_cuda(
    at::Tensor g_dQ, at::Tensor g_dK, at::Tensor g_dV,
    at::Tensor dO, at::Tensor Q, at::Tensor K, at::Tensor V,
    at::Tensor P, at::Tensor dS, at::Tensor dP)
{
    TORCH_CHECK(g_dQ.is_cuda());

    g_dQ = g_dQ.contiguous(); g_dK = g_dK.contiguous(); g_dV = g_dV.contiguous();
    dO   = dO.contiguous();   Q    = Q.contiguous();    K    = K.contiguous();
    V    = V.contiguous();    P    = P.contiguous();
    dS   = dS.contiguous();   dP   = dP.contiguous();

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    float scale = 1.f / sqrtf((float)D);
    auto opts = Q.options();

    auto grid_nn  = (B*H*N*N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto grid_nd  = (B*H*N*D + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto grid_row = (B*H*N   + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto g_dS_1 = at::empty({B, H, N, N}, opts);
    matmul_ABt_kernel<<<grid_nn, BLOCK_SIZE>>>(g_dQ.data_ptr<float>(), K.data_ptr<float>(), g_dS_1.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    auto g_dS_2 = at::empty({B, H, N, N}, opts);
    matmul_ABt_kernel<<<grid_nn, BLOCK_SIZE>>>(Q.data_ptr<float>(), g_dK.data_ptr<float>(), g_dS_2.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    auto g_dS_total = g_dS_1 + g_dS_2;

    auto g_K_1 = at::empty({B, H, N, D}, opts);
    matmul_AtB_kernel<<<grid_nd, BLOCK_SIZE>>>(dS.data_ptr<float>(), g_dQ.data_ptr<float>(), g_K_1.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    auto g_Q_1 = at::empty({B, H, N, D}, opts);
    matmul_AB_kernel<<<grid_nd, BLOCK_SIZE>>>(dS.data_ptr<float>(), g_dK.data_ptr<float>(), g_Q_1.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    auto g_dP     = at::empty({B, H, N, N}, opts);
    auto g_P_soft = at::empty({B, H, N, N}, opts);
    softmax_double_bwd_kernel<<<grid_row, BLOCK_SIZE>>>(g_dS_total.data_ptr<float>(), P.data_ptr<float>(), dP.data_ptr<float>(), g_dP.data_ptr<float>(), g_P_soft.data_ptr<float>(), B, H, N);
    CUDA_CHECK(cudaGetLastError());

    auto g_P_dV = at::empty({B, H, N, N}, opts);
    matmul_ABt_kernel<<<grid_nn, BLOCK_SIZE>>>(dO.data_ptr<float>(), g_dV.data_ptr<float>(), g_P_dV.data_ptr<float>(), B, H, N, D, 1.f);
    CUDA_CHECK(cudaGetLastError());

    auto g_P = g_P_soft + g_P_dV;

    auto g_dO_1 = at::empty({B, H, N, D}, opts);
    matmul_AB_kernel<<<grid_nd, BLOCK_SIZE>>>(P.data_ptr<float>(), g_dV.data_ptr<float>(), g_dO_1.data_ptr<float>(), B, H, N, D, 1.f);
    CUDA_CHECK(cudaGetLastError());

    auto g_dO_2 = at::empty({B, H, N, D}, opts);
    matmul_AB_kernel<<<grid_nd, BLOCK_SIZE>>>(g_dP.data_ptr<float>(), V.data_ptr<float>(), g_dO_2.data_ptr<float>(), B, H, N, D, 1.f);
    CUDA_CHECK(cudaGetLastError());

    auto g_dO = g_dO_1 + g_dO_2;

    auto g_V = at::empty({B, H, N, D}, opts);
    matmul_AtB_kernel<<<grid_nd, BLOCK_SIZE>>>(g_dP.data_ptr<float>(), dO.data_ptr<float>(), g_V.data_ptr<float>(), B, H, N, D, 1.f);
    CUDA_CHECK(cudaGetLastError());

    auto g_S = at::empty({B, H, N, N}, opts);
    softmax_bwd_kernel<<<grid_row, BLOCK_SIZE>>>(P.data_ptr<float>(), g_P.data_ptr<float>(), g_S.data_ptr<float>(), B, H, N);
    CUDA_CHECK(cudaGetLastError());

    auto g_Q_2 = at::empty({B, H, N, D}, opts);
    matmul_AB_kernel<<<grid_nd, BLOCK_SIZE>>>(g_S.data_ptr<float>(), K.data_ptr<float>(), g_Q_2.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    auto g_K_2 = at::empty({B, H, N, D}, opts);
    matmul_AtB_kernel<<<grid_nd, BLOCK_SIZE>>>(g_S.data_ptr<float>(), Q.data_ptr<float>(), g_K_2.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    return {g_dO, g_Q_1 + g_Q_2, g_K_1 + g_K_2, g_V};
}
