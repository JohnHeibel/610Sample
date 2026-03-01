"""
V7 Flash Attention with Double Backward Support
================================================
Rectangular sliding-window attention (fp16/bf16 WMMA) for TTT-E2E.
O(N) memory through forward, backward, and double backward.
"""

import torch
import attention_cuda as _attention_cuda


class FlashAttentionV7Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dO, Q, K, V, O, L, window_size, chunk_id):
        dQ, dK, dV = _attention_cuda.v7_backward(dO, Q, K, V, O, L, window_size, chunk_id)
        ctx.save_for_backward(dO, Q, K, V, O, L)
        ctx.window_size = window_size
        ctx.chunk_id = chunk_id
        return dQ, dK, dV

    @staticmethod
    def backward(ctx, g_dQ, g_dK, g_dV):
        dO, Q, K, V, O, L = ctx.saved_tensors
        if g_dQ is None:
            g_dQ = torch.zeros_like(Q)
        if g_dK is None:
            g_dK = torch.zeros_like(K)
        if g_dV is None:
            g_dV = torch.zeros_like(V)
        g_dO, g_Q, g_K, g_V = _attention_cuda.v7_double_backward(
            g_dQ, g_dK, g_dV, dO, Q, K, V, O, L,
            ctx.window_size, ctx.chunk_id
        )
        # Gradients for: dO, Q, K, V, O, L, window_size, chunk_id
        return g_dO, g_Q, g_K, g_V, None, None, None, None


class FlashAttentionV7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, window_size, chunk_id):
        O, L = _attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.window_size = window_size
        ctx.chunk_id = chunk_id
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        return (*FlashAttentionV7Backward.apply(
            grad_output, Q, K, V, O, L, ctx.window_size, ctx.chunk_id
        ), None, None)


def flash_attention_v7(Q, K, V, window_size, chunk_id):
    """Rectangular sliding-window attention with double backward support.

    Designed for TTT-E2E: Q[B,H,N_q,D] attends to K/V[B,H,N_kv,D]
    with causal + sliding window + validity mask.

    Args:
        Q: (B, H, N_q, D) query tensor (fp16 or bf16, CUDA)
        K: (B, H, N_kv, D) key tensor
        V: (B, H, N_kv, D) value tensor
        window_size: sliding window size
        chunk_id: chunk index (for mask computation)

    Returns:
        O: (B, H, N_q, D) attention output
    """
    return FlashAttentionV7.apply(Q, K, V, window_size, chunk_id)
