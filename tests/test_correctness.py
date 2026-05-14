"""
Smoke test for the v9 binding. Verifies the kernel loads and returns tensors
of the right shape. Replaced in Commit 6 with full correctness comparisons
against the reference attention (fwd, bwd, dbl_bwd).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flash_v9 import flash_v9_attention
from tests.common import make_inputs, SHAPE_CONFIGS


def test_forward_smoke():
    device = 'cuda'
    dtype = torch.bfloat16
    for label, B, H, N, D in SHAPE_CONFIGS[:2]:
        Q, K, V = make_inputs(B, H, N, D, dtype, device)
        O = flash_v9_attention(Q, K, V, is_causal=False)
        assert O.shape == Q.shape, f"{label}: expected {Q.shape}, got {O.shape}"
        assert O.dtype == Q.dtype
        print(f"  [{label}] B={B} H={H} N={N} D={D}: forward shape OK")
    print("forward smoke: PASS")


if __name__ == '__main__':
    test_forward_smoke()
