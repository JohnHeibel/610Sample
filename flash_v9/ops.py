"""
Python autograd interface for Flash v9.

Status:
- forward       CUDA (CUTLASS / CuTe).      Functionally complete.
- backward      CUDA (bwd_dQ + bwd_dKV in CuTe).  Wired here.
- double_bwd    Currently falls back to autograd-over-_reference_bwd. The
                CUDA dbl_bwd kernel is the novel paper contribution and
                lands in commit 6g/6h. After that, FlashV9Backward.backward
                will call _ext.double_backward directly.
"""

import math
import torch
import flash_v9_cuda as _ext


def _reference_bwd(dO, Q, K, V, O, L, is_causal, softmax_scale):
    """Python reference for Flash v9 backward, using torch ops.

    Used (a) as a debugging baseline (`backend='reference'`) and
         (b) as the autograd-trackable path for double-backward until
             the CUDA dbl_bwd kernel lands.

    NOTE: this reimplementation uses torch.softmax, not the saved L,
    because softmax is autograd-trackable through L (whereas exp(S - L)
    treats L as a constant and breaks the second-order chain rule).
    """
    Qf, Kf, Vf, Of, dOf = Q.float(), K.float(), V.float(), O.float(), dO.float()
    S = torch.matmul(Qf, Kf.transpose(-2, -1)) * softmax_scale
    if is_causal:
        N_q  = Q.shape[-2]
        N_kv = K.shape[-2]
        q_idx = torch.arange(N_q,  device=Q.device).unsqueeze(1)
        k_idx = torch.arange(N_kv, device=Q.device).unsqueeze(0)
        mask  = (q_idx + (N_kv - N_q) >= k_idx).to(Qf.dtype)
        S = S + (1.0 - mask) * -1.0e30
    del L
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    dV = torch.matmul(P.transpose(-2, -1), dOf)
    dP = torch.matmul(dOf, Vf.transpose(-2, -1))
    Di = (dOf * Of).sum(dim=-1, keepdim=True)
    dS = P * (dP - Di) * softmax_scale
    dQ = torch.matmul(dS, Kf)
    dK = torch.matmul(dS.transpose(-2, -1), Qf)
    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)


class FlashV9Backward(torch.autograd.Function):
    """Backward as an autograd.Function so we can hand-define dbl_bwd.

    .forward calls the CUDA bwd (bwd_dQ + bwd_dKV).
    .backward currently re-runs the Python reference under enable_grad
        and pulls second-order grads via autograd. Will be replaced by a
        direct call to _ext.double_backward once the CUDA dbl_bwd kernel
        (commit 6g) lands.
    """

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
        D = Q.shape[-1]

        if D == 64:
            # CUDA dbl_bwd (the novel piece).
            if g_dQ is None: g_dQ = torch.zeros_like(Q)
            else:            g_dQ = g_dQ.contiguous()
            if g_dK is None: g_dK = torch.zeros_like(K)
            else:            g_dK = g_dK.contiguous()
            if g_dV is None: g_dV = torch.zeros_like(V)
            else:            g_dV = g_dV.contiguous()
            g_dO, g_Q, g_K, g_V = _ext.double_backward(
                g_dQ, g_dK, g_dV, dO, Q, K, V, O, L,
                ctx.is_causal, ctx.softmax_scale,
            )
            return g_dO, g_Q, g_K, g_V, None, None, None, None

        # D != 64 (currently only D=128 is in the FA2-parity surface):
        # CUDA dblbwd kernels don't yet support headdim != 64, so fall
        # back to autograd-over-_reference_bwd. (Removed once the CUDA
        # kernels grow D=128 support.)
        with torch.enable_grad():
            dO_g = dO.detach().requires_grad_(True)
            Q_g  = Q.detach().requires_grad_(True)
            K_g  = K.detach().requires_grad_(True)
            V_g  = V.detach().requires_grad_(True)
            dQ_p, dK_p, dV_p = _reference_bwd(
                dO_g, Q_g, K_g, V_g, O, L,
                ctx.is_causal, ctx.softmax_scale,
            )
            grads_in = (
                g_dQ if g_dQ is not None else torch.zeros_like(dQ_p),
                g_dK if g_dK is not None else torch.zeros_like(dK_p),
                g_dV if g_dV is not None else torch.zeros_like(dV_p),
            )
            g_dO_out, g_Q_out, g_K_out, g_V_out = torch.autograd.grad(
                (dQ_p, dK_p, dV_p),
                [dO_g, Q_g, K_g, V_g],
                grad_outputs=grads_in,
            )
        return g_dO_out, g_Q_out, g_K_out, g_V_out, None, None, None, None


class FlashV9Function(torch.autograd.Function):
    """Flash v9 attention.

    .forward    -> CUDA fwd
    .backward   -> dispatch:
                     if grad_output is autograd-tracked (create_graph=True
                       on the outer .backward / autograd.grad), route through
                       _reference_bwd so the second-order chain is handled
                       by standard PyTorch autograd. Slow but correct.
                     otherwise, call FlashV9Backward.apply for the fast
                       CUDA-backed first-order path.
                   The CUDA dbl_bwd kernel (6g/6h) replaces the create_graph
                   branch with a direct _ext.double_backward call.
    """

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
        # FlashV9Backward.apply gives both (a) the fast CUDA bwd and (b)
        # the CUDA dbl_bwd in its .backward. Always use it.
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
