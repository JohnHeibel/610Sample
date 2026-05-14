"""
Functional correctness: Flash v9 vs FlashAttention-2 (fwd + bwd only).

FA2 does not support double backward, so dbl_bwd has no FA-side baseline.
For the dbl_bwd comparison, see test_correctness.py (which compares against
PyTorch reference attention through autograd).

If flash_attn isn't importable, this whole file is skipped. That's the
expected state on Blackwell consumer (sm_120) where prebuilt wheels don't
exist yet. On A100 (sm_80) the import succeeds.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from flash_v9 import flash_v9_attention
from tests.common import ATOL_FWD, RTOL_FWD, ATOL_BWD, RTOL_BWD


# FA2's signature is flash_attn_func(Q, K, V, dropout_p, softmax_scale, causal),
# with Q/K/V laid out as [B, N, H, D] -- transpose from our [B, H, N, D].
try:
    from flash_attn import flash_attn_func  # type: ignore
    HAVE_FLASH_ATTN = True
except Exception:
    flash_attn_func = None  # type: ignore
    HAVE_FLASH_ATTN = False


def _fa2_attention(Q, K, V, is_causal):
    """Call FA2 with our [B, H, N, D] tensors transposed to [B, N, H, D]."""
    Q_t = Q.transpose(1, 2).contiguous()
    K_t = K.transpose(1, 2).contiguous()
    V_t = V.transpose(1, 2).contiguous()
    softmax_scale = 1.0 / math.sqrt(Q.shape[-1])
    O_t = flash_attn_func(Q_t, K_t, V_t, dropout_p=0.0,
                          softmax_scale=softmax_scale, causal=is_causal)
    return O_t.transpose(1, 2).contiguous()


def test_forward_vs_fa2():
    if not HAVE_FLASH_ATTN:
        print("test_forward_vs_fa2: SKIPPED (flash_attn not installed)")
        return True

    device = 'cuda'
    dtype = torch.bfloat16
    cases = [
        (1, 4,  256,  64, False),
        (1, 8,  512,  64, False),
        (1, 16, 1024, 64, False),
        (1, 8,  512, 128, False),
        (1, 16, 1024, 128, False),
        (1, 8,  512,  64, True),
        (1, 16, 1024, 64, True),
        (1, 16, 1024, 128, True),
    ]
    all_ok = True
    for B, H, N, D, causal in cases:
        torch.manual_seed(0)
        Q = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
        K = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
        V = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
        O_v9  = flash_v9_attention(Q, K, V, is_causal=causal).float()
        O_fa2 = _fa2_attention(Q, K, V, is_causal=causal).float()
        diff = (O_v9 - O_fa2).abs()
        ma = diff.max().item()
        ok = torch.allclose(O_v9, O_fa2, atol=ATOL_FWD, rtol=RTOL_FWD)
        status = "PASS" if ok else "FAIL"
        print(f"  [v9 vs FA2 fwd] B={B} H={H} N={N} D={D} causal={causal}: "
              f"{status} (max_abs={ma:.2e})")
        all_ok &= ok
    print(f"forward vs FA2: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def test_backward_vs_fa2():
    if not HAVE_FLASH_ATTN:
        print("test_backward_vs_fa2: SKIPPED (flash_attn not installed)")
        return True

    device = 'cuda'
    dtype = torch.bfloat16
    cases = [
        (1, 4,  256,  64, False),
        (1, 8,  512,  64, False),
        (1, 8,  512, 128, False),
        (1, 8,  512,  64, True),
        (1, 16, 1024, 64, True),
    ]
    all_ok = True
    for B, H, N, D, causal in cases:
        torch.manual_seed(0)
        Q = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        K = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        V = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1).requires_grad_(True)
        Q = Q.detach().requires_grad_(True)
        K = K.detach().requires_grad_(True)
        V = V.detach().requires_grad_(True)

        Q2 = Q.detach().requires_grad_(True)
        K2 = K.detach().requires_grad_(True)
        V2 = V.detach().requires_grad_(True)

        O_v9  = flash_v9_attention(Q, K, V, is_causal=causal)
        O_fa2 = _fa2_attention(Q2, K2, V2, is_causal=causal)

        O_v9.sum().backward()
        O_fa2.sum().backward()

        ok = True
        for name, a, b in [('dQ', Q.grad, Q2.grad),
                           ('dK', K.grad, K2.grad),
                           ('dV', V.grad, V2.grad)]:
            diff = (a.float() - b.float()).abs()
            ma = diff.max().item()
            ok_x = torch.allclose(a.float(), b.float(), atol=ATOL_BWD, rtol=RTOL_BWD)
            print(f"  [v9 vs FA2 bwd] B={B} H={H} N={N} D={D} causal={causal} "
                  f"{name}: {'PASS' if ok_x else 'FAIL'} (max_abs={ma:.2e})")
            ok &= ok_x
        all_ok &= ok
    print(f"backward vs FA2: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    fwd_ok = test_forward_vs_fa2()
    bwd_ok = test_backward_vs_fa2()
    sys.exit(0 if (fwd_ok and bwd_ok) else 1)
