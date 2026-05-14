"""
Python autograd interface for Flash v9.

Two autograd Functions chained so that forward -> backward -> double_backward
compose cleanly:

    FlashV9Function       calls backward in its .backward
        -> FlashV9Backward calls double_backward in its .backward

Status:
- forward       CUDA (CUTLASS / CuTe).    Functionally complete.
- backward      Python reference (torch). Correct but slow. CUDA replacement
                planned post-dbl_bwd. The Python implementation uses the
                saved L (not Q.K^T from scratch), so memory cost stays O(N^2)
                only inside this function -- not in the autograd graph.
- double_bwd    CUDA (CUTLASS / CuTe).    The novel IO-aware piece.
"""

import math
import torch
import flash_v9_cuda as _ext


def _reference_bwd(dO, Q, K, V, O, L, is_causal, softmax_scale):
    """Python reference for v9 backward.

    Uses the saved L = m + log(sum_exp(S - m)) so we can recover
    P = exp(S * scale - L) without recomputing the rowmax.

    Returns (dQ, dK, dV) in the input dtype.
    """
    Qf, Kf, Vf, Of, dOf = Q.float(), K.float(), V.float(), O.float(), dO.float()
    S = torch.matmul(Qf, Kf.transpose(-2, -1)) * softmax_scale  # [B,H,N,N]
    if is_causal:
        N_q = Q.shape[-2]
        N_kv = K.shape[-2]
        q_idx = torch.arange(N_q, device=Q.device).unsqueeze(1)
        k_idx = torch.arange(N_kv, device=Q.device).unsqueeze(0)
        mask = q_idx + (N_kv - N_q) >= k_idx
        S = S.masked_fill(~mask, float('-inf'))
    P = torch.exp(S - L.unsqueeze(-1))                          # [B,H,N,N]
    P = torch.nan_to_num(P, nan=0.0)
    dV = torch.matmul(P.transpose(-2, -1), dOf)
    dP = torch.matmul(dOf, Vf.transpose(-2, -1))
    Di = (dOf * Of).sum(dim=-1, keepdim=True)
    dS = P * (dP - Di) * softmax_scale
    dQ = torch.matmul(dS, Kf)
    dK = torch.matmul(dS.transpose(-2, -1), Qf)
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class FlashV9Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dO, Q, K, V, O, L, is_causal, softmax_scale):
        with torch.no_grad():
            dQ, dK, dV = _reference_bwd(dO, Q, K, V, O, L, is_causal, softmax_scale)
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
