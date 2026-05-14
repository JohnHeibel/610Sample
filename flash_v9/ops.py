"""
Python autograd interface for Flash v9.

Two autograd Functions chained so that forward -> backward -> double_backward
compose cleanly:

    FlashV9Function       calls backward in its .backward
        -> FlashV9Backward calls double_backward in its .backward
"""

import math
import torch
import flash_v9_cuda as _ext


class FlashV9Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dO, Q, K, V, O, L, is_causal, softmax_scale):
        dQ, dK, dV = _ext.backward(dO, Q, K, V, O, L, is_causal, softmax_scale)
        ctx.save_for_backward(dO, Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
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
        g_dO, g_Q, g_K, g_V = _ext.double_backward(
            g_dQ, g_dK, g_dV, dO, Q, K, V, O, L,
            ctx.is_causal, ctx.softmax_scale,
        )
        return g_dO, g_Q, g_K, g_V, None, None, None, None


class FlashV9Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal, softmax_scale):
        O, L = _ext.forward(Q, K, V, is_causal, softmax_scale)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.softmax_scale = softmax_scale
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = FlashV9Backward.apply(
            grad_output, Q, K, V, O, L, ctx.is_causal, ctx.softmax_scale,
        )
        return dQ, dK, dV, None, None


def flash_v9_attention(Q, K, V, *, is_causal=False, softmax_scale=None):
    """Flash v9 attention.

    Args:
        Q, K, V: [B, H, N, D] tensors, bf16 or fp16, CUDA.
        is_causal: apply causal mask if True.
        softmax_scale: defaults to 1 / sqrt(D).

    Returns:
        O: [B, H, N, D].
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Q.shape[-1])
    return FlashV9Function.apply(Q, K, V, is_causal, softmax_scale)
