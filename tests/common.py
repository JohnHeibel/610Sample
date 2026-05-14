"""
Shared test/bench utilities for Flash v9.

FA2-parity surface: causal + non-causal, fixed seqlen, headdim in {64, 128}.
"""

import math
import torch


# ---------------------------------------------------------------
# Shape configs (FA2-paper-shaped baselines, kept small at the top so
# local 5070 Ti can run them; large shapes appended for A100 sweeps)
# ---------------------------------------------------------------
# (label, B, H, N, D)
SHAPE_CONFIGS = [
    ('tiny',     1,  4,   256,  64),
    ('small64',  1,  8,   512,  64),
    ('small128', 1,  8,   512, 128),
    ('med64',    1, 16,  1024,  64),
    ('med128',   1, 16,  1024, 128),
    ('large64',  1, 32,  2048,  64),
    ('large128', 1, 32,  2048, 128),
    ('xl64',     1, 16,  4096,  64),
    ('xl128',    1, 16,  4096, 128),
]

# Default tolerances for bf16 fp32-accum.
ATOL_FWD = 5e-3
RTOL_FWD = 5e-3
ATOL_BWD = 5e-3
RTOL_BWD = 5e-3
ATOL_DBL = 1.5e-2
RTOL_DBL = 1.5e-2


def causal_mask(N_q, N_kv, device):
    """Standard causal mask: query i sees keys j with j <= i + (N_kv - N_q)."""
    q = torch.arange(N_q, device=device).unsqueeze(1)
    k = torch.arange(N_kv, device=device).unsqueeze(0)
    return q + (N_kv - N_q) >= k


def reference_attention(Q, K, V, *, is_causal=False, softmax_scale=None):
    """Reference fp32 attention for correctness.

    Q: [B, H, N_q, D], K: [B, H, N_kv, D], V: [B, H, N_kv, D]
    Returns O: [B, H, N_q, D] (fp32).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(Q.shape[-1])
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    S = torch.matmul(Qf, Kf.transpose(-2, -1)) * softmax_scale
    if is_causal:
        m = causal_mask(Q.shape[-2], K.shape[-2], Q.device)
        S = S.masked_fill(~m, float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    return torch.matmul(P, Vf)


def make_inputs(B, H, N, D, dtype, device, *, requires_grad=False, seed=42):
    """Build random Q, K, V at [B, H, N, D]. Same N for Q and K/V (FA2 parity)."""
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    K = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    V = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1
    if requires_grad:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
    return Q, K, V
