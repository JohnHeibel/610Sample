#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
} while (0)

__global__ void compute_scores_kernel(
    const float* Q, const float* K, float* S,
    int B, int H, int N, int D, float scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * N) return;

    int j = idx % N;
    int i = (idx / N) % N;
    int h = (idx / (N * N)) % H;
    int b = idx / (H * N * N);

    int q_off = ((b * H + h) * N + i) * D;
    int k_off = ((b * H + h) * N + j) * D;

    float sum = 0.f;
    for (int d = 0; d < D; d++)
        sum += Q[q_off + d] * K[k_off + d];

    S[idx] = sum * scale;
}

__global__ void softmax_kernel(
    const float* S, float* P,
    int B, int H, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;

    int off = row * N;

    float max_val = S[off];
    for (int j = 1; j < N; j++)
        max_val = fmaxf(max_val, S[off + j]);

    float sum = 0.f;
    for (int j = 0; j < N; j++) {
        float v = expf(S[off + j] - max_val);
        P[off + j] = v;
        sum += v;
    }

    float inv = 1.f / sum;
    for (int j = 0; j < N; j++)
        P[off + j] *= inv;
}

__global__ void compute_output_kernel(
    const float* P, const float* V, float* O,
    int B, int H, int N, int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H * N * D) return;

    int d = idx % D;
    int i = (idx / D) % N;
    int h = (idx / (N * D)) % H;
    int b = idx / (H * N * D);

    int p_off  = ((b * H + h) * N + i) * N;
    int v_base = ((b * H + h) * N) * D;

    float sum = 0.f;
    for (int j = 0; j < N; j++)
        sum += P[p_off + j] * V[v_base + j * D + d];

    O[idx] = sum;
}

std::vector<at::Tensor> attention_forward_cuda(at::Tensor Q, at::Tensor K, at::Tensor V)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda());
    TORCH_CHECK(Q.dtype() == at::kFloat && K.dtype() == at::kFloat && V.dtype() == at::kFloat);
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4);

    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous();

    int B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    float scale = 1.f / sqrtf((float)D);
    auto opts = Q.options();

    auto S = at::empty({B, H, N, N}, opts);
    auto P = at::empty({B, H, N, N}, opts);
    auto O = at::empty({B, H, N, D}, opts);

    auto grid_nn  = (B*H*N*N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto grid_row = (B*H*N   + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto grid_nd  = (B*H*N*D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_scores_kernel<<<grid_nn, BLOCK_SIZE>>>(Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(), B, H, N, D, scale);
    CUDA_CHECK(cudaGetLastError());

    softmax_kernel<<<grid_row, BLOCK_SIZE>>>(S.data_ptr<float>(), P.data_ptr<float>(), B, H, N);
    CUDA_CHECK(cudaGetLastError());

    compute_output_kernel<<<grid_nd, BLOCK_SIZE>>>(P.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), B, H, N, D);
    CUDA_CHECK(cudaGetLastError());

    return {O, S, P};
}
