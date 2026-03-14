#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <cmath>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ",                        \
                    cudaGetErrorString(err));                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,                           \
                    "cuBLAS error: ", (int)status);                            \
    } while (0)

// ============================================================================
// Element-wise kernel: D_i = (dO * O).sum(dim=-1)
// One thread per row.
// ============================================================================
__global__ void v2_compute_D_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D_vec,
    int total_rows, int D
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    float sum = 0.0f;
    const float* dO_row = dO + row * D;
    const float* O_row  = O  + row * D;
    for (int d = 0; d < D; d++) {
        sum += dO_row[d] * O_row[d];
    }
    D_vec[row] = sum;
}

// ============================================================================
// Pass 1+2 element-wise: from S,g_dS,dP,g_PdV tiles, compute P and
// accumulate dot2, A, E via atomicAdd.
//   P_ij = exp(S_ij - L_i)
//   dot2_i += P_ij * g_dS_ij
//   A_i    += P_ij * g_dS_ij * dP_ij
//   E_i    += P_ij * g_PdV_ij
// ============================================================================
__global__ void v2_ewise_pass12_kernel(
    const float* __restrict__ S_tile,
    const float* __restrict__ g_dS_tile,
    const float* __restrict__ dP_tile,
    const float* __restrict__ g_PdV_tile,
    const float* __restrict__ L,
    float* __restrict__ P_tile,
    float* __restrict__ dot2_vec,
    float* __restrict__ A_vec,
    float* __restrict__ E_vec,
    int BH, int Br, int N, int i_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BH * Br * N;
    if (idx >= total) return;

    int bh = idx / (Br * N);
    int rem = idx % (Br * N);
    int i_local = rem / N;
    int i_global = i_start + i_local;

    float l_val = L[bh * N + i_global];
    float s_val = S_tile[idx];
    float p_val = __expf(s_val - l_val);
    P_tile[idx] = p_val;

    float gds_val = g_dS_tile[idx];
    float dp_val = dP_tile[idx];
    float gpdv_val = g_PdV_tile[idx];

    float pg = p_val * gds_val;
    int row_idx = bh * N + i_global;

    atomicAdd(&dot2_vec[row_idx], pg);
    atomicAdd(&A_vec[row_idx], pg * dp_val);
    atomicAdd(&E_vec[row_idx], p_val * gpdv_val);
}

// ============================================================================
// Finalize dot3 = A - 2*D*dot2 + E
// ============================================================================
__global__ void v2_finalize_dot3_kernel(
    const float* __restrict__ A_vec,
    const float* __restrict__ D_vec,
    const float* __restrict__ dot2_vec,
    const float* __restrict__ E_vec,
    float* __restrict__ dot3_vec,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dot3_vec[idx] = A_vec[idx] - 2.0f * D_vec[idx] * dot2_vec[idx] + E_vec[idx];
}

// ============================================================================
// Compute P from S: P_ij = exp(S_ij - L_i)
// Used in pass 3 to get P_tile from recomputed S_tile.
// ============================================================================
__global__ void v2_compute_P_kernel(
    const float* __restrict__ S_tile,
    const float* __restrict__ L,
    float* __restrict__ P_tile,
    int BH, int Br, int N, int i_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BH * Br * N;
    if (idx >= total) return;

    int bh = idx / (Br * N);
    int rem = idx % (Br * N);
    int i_local = rem / N;
    int i_global = i_start + i_local;

    float l_val = L[bh * N + i_global];
    P_tile[idx] = __expf(S_tile[idx] - l_val);
}

// ============================================================================
// Pass 3 element-wise: from P,dP,g_dS,g_PdV + row vectors, compute dS,g_dP,g_S
//   dS_ij   = P_ij * (dP_ij - D_i)
//   g_dP_ij = P_ij * (g_dS_ij - dot2_i)
//   g_P_soft_ij = g_dS_ij * (dP_ij - D_i) - dP_ij * dot2_i
//   g_P_ij  = g_P_soft_ij + g_PdV_ij
//   g_S_ij  = P_ij * (g_P_ij - dot3_i)
// ============================================================================
__global__ void v2_ewise_pass3_kernel(
    const float* __restrict__ P_tile,
    const float* __restrict__ dP_tile,
    const float* __restrict__ g_dS_tile,
    const float* __restrict__ g_PdV_tile,
    const float* __restrict__ D_vec,
    const float* __restrict__ dot2_vec,
    const float* __restrict__ dot3_vec,
    float* __restrict__ dS_tile,
    float* __restrict__ g_dP_tile,
    float* __restrict__ g_S_tile,
    int BH, int Br, int N, int i_start
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BH * Br * N;
    if (idx >= total) return;

    int bh = idx / (Br * N);
    int rem = idx % (Br * N);
    int i_local = rem / N;
    int i_global = i_start + i_local;
    int row_idx = bh * N + i_global;

    float p = P_tile[idx];
    float dp = dP_tile[idx];
    float gds = g_dS_tile[idx];
    float gpdv = g_PdV_tile[idx];
    float d_val = D_vec[row_idx];
    float dot2 = dot2_vec[row_idx];
    float dot3 = dot3_vec[row_idx];

    float ds = p * (dp - d_val);
    float g_dp = p * (gds - dot2);
    float g_P_soft = gds * (dp - d_val) - dp * dot2;
    float g_P = g_P_soft + gpdv;
    float g_S = p * (g_P - dot3);

    dS_tile[idx] = ds;
    g_dP_tile[idx] = g_dp;
    g_S_tile[idx] = g_S;
}

// ============================================================================
// cuBLAS batched strided GEMM helper (row-major convention)
//
// Computes: C = alpha * op(A) @ op(B) + beta * C   [row-major]
// where op(A) is (M, K), op(B) is (K, N_out), C is (M, N_out)
//
// Row-major to col-major conversion:
//   cuBLAS sees C^T = alpha * op'(B^T) @ op'(A^T) + beta * C^T
// ============================================================================
static void gemm_strided(
    cublasHandle_t handle,
    bool transA, bool transB,
    int M, int N_out, int K_dim,
    float alpha,
    const float* A, int lda_row, long long strideA,
    const float* B, int ldb_row, long long strideB,
    float beta,
    float* C, int ldc_row, long long strideC,
    int batchCount
) {
    // Row->col swap: first cublas arg = B, second = A
    cublasOperation_t cublas_opA = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t cublas_opB = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle,
        cublas_opA, cublas_opB,
        N_out, M, K_dim,
        &alpha,
        B, ldb_row, strideB,
        A, lda_row, strideA,
        &beta,
        C, ldc_row, strideC,
        batchCount
    ));
}

// ============================================================================
// Host wrapper
// ============================================================================
std::vector<at::Tensor> flash_double_backward_v2_cuda(
    at::Tensor g_dQ,
    at::Tensor g_dK,
    at::Tensor g_dV,
    at::Tensor dO,
    at::Tensor Q,
    at::Tensor K,
    at::Tensor V,
    at::Tensor O,
    at::Tensor L,
    bool return_timing
) {
    TORCH_CHECK(g_dQ.is_cuda(), "g_dQ must be CUDA");
    TORCH_CHECK(g_dQ.dtype() == at::kFloat, "g_dQ must be float32");

    g_dQ = g_dQ.contiguous();
    g_dK = g_dK.contiguous();
    g_dV = g_dV.contiguous();
    dO   = dO.contiguous();
    Q    = Q.contiguous();
    K    = K.contiguous();
    V    = V.contiguous();
    O    = O.contiguous();
    L    = L.contiguous();

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);
    int BH = B * H;
    float scale = 1.0f / sqrtf(static_cast<float>(D));
    auto opts = Q.options();

    // Choose Br: default 256, cap to N, must divide N
    int Br = 256;
    if (Br > N) Br = N;
    while (N % Br != 0 && Br > 1) Br /= 2;
    int num_blocks = N / Br;

    // cuBLAS handle on current stream
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    // Outputs (zeros for accumulation)
    auto g_Q_out  = at::zeros({B, H, N, D}, opts);
    auto g_dO_out = at::zeros({B, H, N, D}, opts);
    auto g_K_out  = at::zeros({B, H, N, D}, opts);
    auto g_V_out  = at::zeros({B, H, N, D}, opts);

    // Row vectors
    auto D_vec = at::empty({BH, N}, opts);
    auto dot2  = at::zeros({BH, N}, opts);
    auto A_vec = at::zeros({BH, N}, opts);
    auto E_vec = at::zeros({BH, N}, opts);
    auto dot3  = at::empty({BH, N}, opts);

    // Temp tiles (BH x Br x N)
    auto S_tile     = at::empty({BH, Br, N}, opts);
    auto P_tile     = at::empty({BH, Br, N}, opts);
    auto g_dS_tile  = at::empty({BH, Br, N}, opts);
    auto dP_tile    = at::empty({BH, Br, N}, opts);
    auto g_PdV_tile = at::empty({BH, Br, N}, opts);
    auto dS_tile    = at::empty({BH, Br, N}, opts);
    auto g_dP_tile  = at::empty({BH, Br, N}, opts);
    auto g_S_tile   = at::empty({BH, Br, N}, opts);

    // Reshape to (BH, N, D) / (BH, N) for indexing
    auto Q_2d    = Q.reshape({BH, N, D});
    auto K_2d    = K.reshape({BH, N, D});
    auto V_2d    = V.reshape({BH, N, D});
    auto dO_2d   = dO.reshape({BH, N, D});
    auto O_2d    = O.reshape({BH, N, D});
    auto g_dQ_2d = g_dQ.reshape({BH, N, D});
    auto g_dK_2d = g_dK.reshape({BH, N, D});
    auto g_dV_2d = g_dV.reshape({BH, N, D});
    auto L_2d    = L.reshape({BH, N});

    auto g_Q_2d  = g_Q_out.reshape({BH, N, D});
    auto g_dO_2d = g_dO_out.reshape({BH, N, D});
    auto g_K_2d  = g_K_out.reshape({BH, N, D});
    auto g_V_2d  = g_V_out.reshape({BH, N, D});

    // Strides
    long long strideND  = (long long)N * D;
    long long strideBrN = (long long)Br * N;

    // Optional timing
    cudaEvent_t ev_start, ev_p0, ev_p12, ev_p3;
    if (return_timing) {
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_p0);
        cudaEventCreate(&ev_p12);
        cudaEventCreate(&ev_p3);
        cudaEventRecord(ev_start, stream);
    }

    // ================================================================
    // Pass 0: D_i = (dO * O).sum(-1)
    // ================================================================
    {
        int total = BH * N;
        int thr = 256;
        int blks = (total + thr - 1) / thr;
        v2_compute_D_kernel<<<blks, thr, 0, stream>>>(
            dO_2d.data_ptr<float>(), O_2d.data_ptr<float>(),
            D_vec.data_ptr<float>(), total, D);
        CUDA_CHECK(cudaGetLastError());
    }

    if (return_timing) cudaEventRecord(ev_p0, stream);

    // ================================================================
    // Pass 1+2: Compute dot2, A, E -> dot3
    // ================================================================
    for (int blk = 0; blk < num_blocks; blk++) {
        int i_start = blk * Br;
        float* Q_blk    = Q_2d.data_ptr<float>()    + i_start * D;
        float* dO_blk   = dO_2d.data_ptr<float>()   + i_start * D;
        float* g_dQ_blk = g_dQ_2d.data_ptr<float>() + i_start * D;
        float* K_ptr    = K_2d.data_ptr<float>();
        float* V_ptr    = V_2d.data_ptr<float>();
        float* g_dK_ptr = g_dK_2d.data_ptr<float>();
        float* g_dV_ptr = g_dV_2d.data_ptr<float>();

        // S = scale * Q_blk @ K^T
        gemm_strided(handle, false, true, Br, N, D, scale,
            Q_blk, D, strideND, K_ptr, D, strideND,
            0.0f, S_tile.data_ptr<float>(), N, strideBrN, BH);

        // g_dS = scale * g_dQ_blk @ K^T
        gemm_strided(handle, false, true, Br, N, D, scale,
            g_dQ_blk, D, strideND, K_ptr, D, strideND,
            0.0f, g_dS_tile.data_ptr<float>(), N, strideBrN, BH);

        // g_dS += scale * Q_blk @ g_dK^T
        gemm_strided(handle, false, true, Br, N, D, scale,
            Q_blk, D, strideND, g_dK_ptr, D, strideND,
            1.0f, g_dS_tile.data_ptr<float>(), N, strideBrN, BH);

        // dP = dO_blk @ V^T
        gemm_strided(handle, false, true, Br, N, D, 1.0f,
            dO_blk, D, strideND, V_ptr, D, strideND,
            0.0f, dP_tile.data_ptr<float>(), N, strideBrN, BH);

        // g_PdV = dO_blk @ g_dV^T
        gemm_strided(handle, false, true, Br, N, D, 1.0f,
            dO_blk, D, strideND, g_dV_ptr, D, strideND,
            0.0f, g_PdV_tile.data_ptr<float>(), N, strideBrN, BH);

        // Element-wise: P, accumulate dot2/A/E
        {
            int total = BH * Br * N;
            int thr = 256;
            int blks_ew = (total + thr - 1) / thr;
            v2_ewise_pass12_kernel<<<blks_ew, thr, 0, stream>>>(
                S_tile.data_ptr<float>(), g_dS_tile.data_ptr<float>(),
                dP_tile.data_ptr<float>(), g_PdV_tile.data_ptr<float>(),
                L_2d.data_ptr<float>(), P_tile.data_ptr<float>(),
                dot2.data_ptr<float>(), A_vec.data_ptr<float>(),
                E_vec.data_ptr<float>(), BH, Br, N, i_start);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Finalize dot3
    {
        int total = BH * N;
        int thr = 256;
        int blks_f = (total + thr - 1) / thr;
        v2_finalize_dot3_kernel<<<blks_f, thr, 0, stream>>>(
            A_vec.data_ptr<float>(), D_vec.data_ptr<float>(),
            dot2.data_ptr<float>(), E_vec.data_ptr<float>(),
            dot3.data_ptr<float>(), total);
        CUDA_CHECK(cudaGetLastError());
    }

    if (return_timing) cudaEventRecord(ev_p12, stream);

    // ================================================================
    // Pass 3: g_Q, g_dO, g_K, g_V
    // ================================================================
    for (int blk = 0; blk < num_blocks; blk++) {
        int i_start = blk * Br;
        float* Q_blk    = Q_2d.data_ptr<float>()    + i_start * D;
        float* dO_blk   = dO_2d.data_ptr<float>()   + i_start * D;
        float* g_dQ_blk = g_dQ_2d.data_ptr<float>() + i_start * D;
        float* K_ptr    = K_2d.data_ptr<float>();
        float* V_ptr    = V_2d.data_ptr<float>();
        float* g_dK_ptr = g_dK_2d.data_ptr<float>();
        float* g_dV_ptr = g_dV_2d.data_ptr<float>();
        float* g_Q_blk  = g_Q_2d.data_ptr<float>()  + i_start * D;
        float* g_dO_blk = g_dO_2d.data_ptr<float>() + i_start * D;

        // ---- Recompute S, g_dS, dP, g_PdV (5 GEMMs) ----
        gemm_strided(handle, false, true, Br, N, D, scale,
            Q_blk, D, strideND, K_ptr, D, strideND,
            0.0f, S_tile.data_ptr<float>(), N, strideBrN, BH);

        gemm_strided(handle, false, true, Br, N, D, scale,
            g_dQ_blk, D, strideND, K_ptr, D, strideND,
            0.0f, g_dS_tile.data_ptr<float>(), N, strideBrN, BH);

        gemm_strided(handle, false, true, Br, N, D, scale,
            Q_blk, D, strideND, g_dK_ptr, D, strideND,
            1.0f, g_dS_tile.data_ptr<float>(), N, strideBrN, BH);

        gemm_strided(handle, false, true, Br, N, D, 1.0f,
            dO_blk, D, strideND, V_ptr, D, strideND,
            0.0f, dP_tile.data_ptr<float>(), N, strideBrN, BH);

        gemm_strided(handle, false, true, Br, N, D, 1.0f,
            dO_blk, D, strideND, g_dV_ptr, D, strideND,
            0.0f, g_PdV_tile.data_ptr<float>(), N, strideBrN, BH);

        // ---- Compute P from S ----
        {
            int total = BH * Br * N;
            int thr = 256;
            int blks_ew = (total + thr - 1) / thr;
            v2_compute_P_kernel<<<blks_ew, thr, 0, stream>>>(
                S_tile.data_ptr<float>(), L_2d.data_ptr<float>(),
                P_tile.data_ptr<float>(), BH, Br, N, i_start);
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- Compute dS, g_dP, g_S ----
        {
            int total = BH * Br * N;
            int thr = 256;
            int blks_ew = (total + thr - 1) / thr;
            v2_ewise_pass3_kernel<<<blks_ew, thr, 0, stream>>>(
                P_tile.data_ptr<float>(), dP_tile.data_ptr<float>(),
                g_dS_tile.data_ptr<float>(), g_PdV_tile.data_ptr<float>(),
                D_vec.data_ptr<float>(), dot2.data_ptr<float>(),
                dot3.data_ptr<float>(),
                dS_tile.data_ptr<float>(), g_dP_tile.data_ptr<float>(),
                g_S_tile.data_ptr<float>(), BH, Br, N, i_start);
            CUDA_CHECK(cudaGetLastError());
        }

        // ---- Output GEMMs ----

        // g_Q_blk += scale * dS @ g_dK  (Br,N)@(N,D) -> (Br,D)
        gemm_strided(handle, false, false, Br, D, N, scale,
            dS_tile.data_ptr<float>(), N, strideBrN,
            g_dK_ptr, D, strideND,
            1.0f, g_Q_blk, D, strideND, BH);

        // g_Q_blk += scale * g_S @ K  (Br,N)@(N,D) -> (Br,D)
        gemm_strided(handle, false, false, Br, D, N, scale,
            g_S_tile.data_ptr<float>(), N, strideBrN,
            K_ptr, D, strideND,
            1.0f, g_Q_blk, D, strideND, BH);

        // g_dO_blk += P @ g_dV  (Br,N)@(N,D) -> (Br,D)
        gemm_strided(handle, false, false, Br, D, N, 1.0f,
            P_tile.data_ptr<float>(), N, strideBrN,
            g_dV_ptr, D, strideND,
            1.0f, g_dO_blk, D, strideND, BH);

        // g_dO_blk += g_dP @ V  (Br,N)@(N,D) -> (Br,D)
        gemm_strided(handle, false, false, Br, D, N, 1.0f,
            g_dP_tile.data_ptr<float>(), N, strideBrN,
            V_ptr, D, strideND,
            1.0f, g_dO_blk, D, strideND, BH);

        // g_K += scale * dS^T @ g_dQ_blk  (N,Br)@(Br,D) -> (N,D)
        gemm_strided(handle, true, false, N, D, Br, scale,
            dS_tile.data_ptr<float>(), N, strideBrN,
            g_dQ_blk, D, strideND,
            1.0f, g_K_2d.data_ptr<float>(), D, strideND, BH);

        // g_K += scale * g_S^T @ Q_blk  (N,Br)@(Br,D) -> (N,D)
        gemm_strided(handle, true, false, N, D, Br, scale,
            g_S_tile.data_ptr<float>(), N, strideBrN,
            Q_blk, D, strideND,
            1.0f, g_K_2d.data_ptr<float>(), D, strideND, BH);

        // g_V += g_dP^T @ dO_blk  (N,Br)@(Br,D) -> (N,D)
        gemm_strided(handle, true, false, N, D, Br, 1.0f,
            g_dP_tile.data_ptr<float>(), N, strideBrN,
            dO_blk, D, strideND,
            1.0f, g_V_2d.data_ptr<float>(), D, strideND, BH);
    }

    if (return_timing) cudaEventRecord(ev_p3, stream);

    CUBLAS_CHECK(cublasDestroy(handle));

    if (return_timing) {
        CUDA_CHECK(cudaEventSynchronize(ev_p3));
        float t_p0, t_p12, t_p3;
        cudaEventElapsedTime(&t_p0, ev_start, ev_p0);
        cudaEventElapsedTime(&t_p12, ev_p0, ev_p12);
        cudaEventElapsedTime(&t_p3, ev_p12, ev_p3);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_p0);
        cudaEventDestroy(ev_p12);
        cudaEventDestroy(ev_p3);

        auto timing = at::empty({3}, opts.dtype(at::kFloat).device(at::kCPU));
        timing[0] = t_p0;
        timing[1] = t_p12;
        timing[2] = t_p3;
        return {g_dO_out, g_Q_out, g_K_out, g_V_out, timing};
    }

    return {g_dO_out, g_Q_out, g_K_out, g_V_out};
}
