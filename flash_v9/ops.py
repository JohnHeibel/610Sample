"""
Python autograd interface for Flash v9.

Status:
- forward       CUDA (CUTLASS / CuTe).    Functionally complete.
- backward      Python reference (torch). Plain torch ops -- autograd-
                trackable, so create_graph=True works through it for free.
                Slow but correct. CUDA bwd planned post-dbl_bwd.
- double_bwd    *Currently flows through the Python backward via standard
                PyTorch autograd.* The CUDA dbl_bwd kernel is the novel
                paper contribution and lands as a follow-up to this commit.

The structure mirrors what FA2 has in flash_attn_interface.py: a single
autograd.Function whose .backward returns torch-op-computed gradients.
For create_graph=True (HVP / second-order optimizers / influence
functions / meta-learning), autograd traces through the .backward and
delivers higher-order gradients without any extra plumbing.
"""

import math
import torch
import flash_v9_cuda as _ext


def _reference_bwd(dO, Q, K, V, O, L, is_causal, softmax_scale):
    """Python reference for Flash v9 backward, using torch ops.

    Inputs are bf16/fp16; intermediates are fp32 (the math op promotes).
    Returns (dQ, dK, dV) in the input dtype.

    Uses the saved logsumexp L so we can recover P = exp(S * scale - L)
    without re-running the rowmax. Uses the saved O for D_i = rowsum(dO * O).

    All ops are differentiable -- autograd-trackable through this function,
    so .backward of FlashV9Function (which calls this) is itself
    differentiable for create_graph=True.
    """
    Qf, Kf, Vf, Of, dOf = Q.float(), K.float(), V.float(), O.float(), dO.float()
    S = torch.matmul(Qf, Kf.transpose(-2, -1)) * softmax_scale       # [B,H,N,N]
    if is_causal:
        # Additive mask: keeps the autograd graph well-defined through both
        # first and second derivatives (masked_fill(-inf) -> exp() yields NaN
        # in second-order).
        N_q  = Q.shape[-2]
        N_kv = K.shape[-2]
        q_idx = torch.arange(N_q,  device=Q.device).unsqueeze(1)
        k_idx = torch.arange(N_kv, device=Q.device).unsqueeze(0)
        mask  = (q_idx + (N_kv - N_q) >= k_idx).to(Qf.dtype)
        S = S + (1.0 - mask) * -1.0e30
    # Use softmax so L is autograd-tracked implicitly. We do *not* use the
    # saved L from the forward kernel here -- treating L as constant breaks
    # the chain rule for second-order grads. The CUDA dbl_bwd kernel
    # (future) will account for the L chain explicitly.
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


class FlashV9Function(torch.autograd.Function):
    """Flash v9 attention with autograd-trackable backward.

    .forward calls the v9 CUDA kernel for O.
    .backward uses _reference_bwd (plain torch ops). Because the ops are
    differentiable, create_graph=True automatically gives correct double
    backward via standard PyTorch autograd -- no hand-defined dbl_bwd
    autograd.Function needed.
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
        dQ, dK, dV = _reference_bwd(
            grad_output, Q, K, V, O, L,
            ctx.is_causal, ctx.softmax_scale,
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


# Kept for downstream code that imports FlashV9Backward; the dedicated
# bwd autograd.Function isn't needed in the Python-bwd flow.
FlashV9Backward = None  # placeholder
