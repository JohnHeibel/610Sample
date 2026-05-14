"""
Correctness tests for v9 forward.

test_forward_smoke: shape-only sanity check at the configs in SHAPE_CONFIGS.
test_forward_single_tile: numeric correctness at N=Bc=64 (single K/V tile),
   which is the largest N a single-tile kernel can handle correctly.

Replaced in Commit 6 with full bwd / dbl_bwd correctness comparisons.
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flash_v9 import flash_v9_attention
from tests.common import (
    make_inputs, SHAPE_CONFIGS, reference_attention,
    ATOL_FWD, RTOL_FWD, ATOL_BWD, RTOL_BWD,
)


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


def test_forward_single_tile():
    """
    Single-tile correctness at N=Bc=64.

    Until commit 3f adds the outer K/V-tile loop, the kernel only processes
    the first Bc=64 K/V positions. At N=64 that's the entire sequence, so
    O should match the reference exactly (within bf16 tolerance).
    """
    device = 'cuda'
    dtype = torch.bfloat16
    # Need N=64 (one full K/V tile) for correctness at the 3e stage.
    cases = [(1, 1, 64, 64), (1, 4, 64, 64), (1, 8, 64, 128)]
    all_ok = True
    for B, H, N, D in cases:
        Q, K, V = make_inputs(B, H, N, D, dtype, device)
        O_v9  = flash_v9_attention(Q, K, V, is_causal=False).float()
        O_ref = reference_attention(Q, K, V, is_causal=False)
        diff = (O_v9 - O_ref).abs()
        ma = diff.max().item()
        mr = (diff / (O_ref.abs() + 1e-8)).max().item()
        ok = torch.allclose(O_v9, O_ref, atol=ATOL_FWD, rtol=RTOL_FWD)
        status = "PASS" if ok else "FAIL"
        print(f"  [single-tile] B={B} H={H} N={N} D={D}: {status} "
              f"(max_abs={ma:.2e}, max_rel={mr:.2e})")
        all_ok &= ok
    print(f"forward single-tile: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_forward_multi_tile():
    """
    Multi-tile correctness for non-causal attention.
    N must be a multiple of 64 (Br=Bc=64). Causal masking is added in 3g.
    """
    device = 'cuda'
    dtype = torch.bfloat16
    cases = [
        (1, 1,  128, 64),
        (1, 4,  256, 64),
        (1, 8,  512, 64),
        (1, 16, 1024, 64),
        (1, 4,  128, 128),
        (1, 8,  256, 128),
        (1, 16, 1024, 128),
    ]
    all_ok = True
    for B, H, N, D in cases:
        Q, K, V = make_inputs(B, H, N, D, dtype, device)
        O_v9  = flash_v9_attention(Q, K, V, is_causal=False).float()
        O_ref = reference_attention(Q, K, V, is_causal=False)
        diff = (O_v9 - O_ref).abs()
        ma = diff.max().item()
        mr = (diff / (O_ref.abs() + 1e-8)).max().item()
        ok = torch.allclose(O_v9, O_ref, atol=ATOL_FWD, rtol=RTOL_FWD)
        status = "PASS" if ok else "FAIL"
        print(f"  [multi-tile] B={B} H={H} N={N} D={D}: {status} "
              f"(max_abs={ma:.2e}, max_rel={mr:.2e})")
        all_ok &= ok
    print(f"forward multi-tile: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_backward():
    """Backward correctness: dQ/dK/dV from FlashV9Function vs reference autograd."""
    device = 'cuda'
    dtype = torch.bfloat16
    cases = [
        (1, 1, 64,  64, False),
        (1, 4, 128, 64, False),
        (1, 8, 256, 64, False),
        (1, 4, 128, 128, False),
        (1, 8, 256, 128, False),
        (1, 4, 128, 64, True),
        (1, 8, 256, 64, True),
    ]
    all_ok = True
    for B, H, N, D, causal in cases:
        Q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True) * 0.1
        K = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True) * 0.1
        V = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True) * 0.1
        Q = Q.detach().requires_grad_(True)
        K = K.detach().requires_grad_(True)
        V = V.detach().requires_grad_(True)

        Qref = Q.detach().float().requires_grad_(True)
        Kref = K.detach().float().requires_grad_(True)
        Vref = V.detach().float().requires_grad_(True)
        Oref = reference_attention(Qref, Kref, Vref, is_causal=causal)
        Oref.sum().backward()

        Ov9 = flash_v9_attention(Q, K, V, is_causal=causal)
        Ov9.sum().backward()

        ok = True
        for name, actual, expected in [('dQ', Q.grad, Qref.grad),
                                        ('dK', K.grad, Kref.grad),
                                        ('dV', V.grad, Vref.grad)]:
            diff = (actual.float() - expected).abs()
            ma = diff.max().item()
            ok_x = torch.allclose(actual.float(), expected, atol=ATOL_BWD, rtol=RTOL_BWD)
            print(f"  [bwd] B={B} H={H} N={N} D={D} causal={causal} {name}: "
                  f"{'PASS' if ok_x else 'FAIL'} (max_abs={ma:.2e})")
            ok &= ok_x
        all_ok &= ok
    print(f"backward: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_forward_causal():
    """Causal masking correctness."""
    device = 'cuda'
    dtype = torch.bfloat16
    cases = [
        (1, 1, 64,   64),
        (1, 4, 128,  64),
        (1, 8, 512,  64),
        (1, 16, 1024, 64),
        (1, 8, 256,  128),
        (1, 16, 1024, 128),
    ]
    all_ok = True
    for B, H, N, D in cases:
        Q, K, V = make_inputs(B, H, N, D, dtype, device)
        O_v9  = flash_v9_attention(Q, K, V, is_causal=True).float()
        O_ref = reference_attention(Q, K, V, is_causal=True)
        diff = (O_v9 - O_ref).abs()
        ma = diff.max().item()
        mr = (diff / (O_ref.abs() + 1e-8)).max().item()
        ok = torch.allclose(O_v9, O_ref, atol=ATOL_FWD, rtol=RTOL_FWD)
        status = "PASS" if ok else "FAIL"
        print(f"  [causal] B={B} H={H} N={N} D={D}: {status} "
              f"(max_abs={ma:.2e}, max_rel={mr:.2e})")
        all_ok &= ok
    print(f"forward causal: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    test_forward_smoke()
    test_forward_single_tile()
    test_forward_multi_tile()
    test_forward_causal()
    test_backward()
