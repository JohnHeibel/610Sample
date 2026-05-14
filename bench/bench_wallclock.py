"""
Wall-clock benchmark for Flash v9 vs PyTorch reference and FlashAttention-2.

Per (config, op) cell we report:
- v9 median wall-clock (ms)
- PyTorch reference median wall-clock (ms)
- FA2 median wall-clock (ms), if flash_attn is importable

Ops measured:
- fwd                 forward only
- fwd + bwd           forward + backward (first-order gradients)
- fwd + bwd + dbl_bwd full second-order: forward, then dQ/dK/dV, then
                      gradient-of-gradient through (dQ, dK, dV) @ random
                      vectors (a Hessian-vector product).

For FA2, dbl_bwd has no native path -- those cells are reported as N/A.

Output: stdout + optional JSON via --out.

Usage:
    python bench/bench_wallclock.py [--out file.json] [--warmup 5] [--trials 30]
"""

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from flash_v9 import flash_v9_attention
from tests.common import reference_attention


try:
    from flash_attn import flash_attn_func  # type: ignore
    HAVE_FLASH_ATTN = True
except Exception:
    flash_attn_func = None
    HAVE_FLASH_ATTN = False


CONFIGS = [
    # (label, B, H, N, D, causal)
    ('small_nc',  1,  8,   512,  64, False),
    ('small_c',   1,  8,   512,  64, True),
    ('med_nc',    1, 16,  1024,  64, False),
    ('med_c',     1, 16,  1024,  64, True),
    ('med128_nc', 1, 16,  1024, 128, False),
    ('med128_c',  1, 16,  1024, 128, True),
    ('large_nc',  1, 16,  2048,  64, False),
    ('large_c',   1, 16,  2048,  64, True),
]


def _build(B, H, N, D, dtype, device, *, requires_grad=False):
    Q = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    K = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    V = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    if requires_grad:
        Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    return Q, K, V


def _fa2_attention(Q, K, V, is_causal):
    Q_t = Q.transpose(1, 2).contiguous()
    K_t = K.transpose(1, 2).contiguous()
    V_t = V.transpose(1, 2).contiguous()
    scale = 1.0 / math.sqrt(Q.shape[-1])
    O_t = flash_attn_func(Q_t, K_t, V_t, dropout_p=0.0,
                          softmax_scale=scale, causal=is_causal)
    return O_t.transpose(1, 2).contiguous()


def _time_op(fn, n_warmup, n_trials):
    """Return median wall-clock ms for `fn`."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def bench_fwd(impl, B, H, N, D, causal, dtype, device, n_warmup, n_trials):
    if impl == 'v9':
        Q, K, V = _build(B, H, N, D, dtype, device)
        return _time_op(lambda: flash_v9_attention(Q, K, V, is_causal=causal),
                        n_warmup, n_trials)
    if impl == 'ref':
        Q, K, V = _build(B, H, N, D, dtype, device)
        return _time_op(lambda: reference_attention(Q, K, V, is_causal=causal),
                        n_warmup, n_trials)
    if impl == 'fa2':
        if not HAVE_FLASH_ATTN: return None
        Q, K, V = _build(B, H, N, D, dtype, device)
        return _time_op(lambda: _fa2_attention(Q, K, V, causal),
                        n_warmup, n_trials)
    raise ValueError(impl)


def bench_fwd_bwd(impl, B, H, N, D, causal, dtype, device, n_warmup, n_trials):
    def make_step():
        Q, K, V = _build(B, H, N, D, dtype, device, requires_grad=True)
        if impl == 'v9':
            def step():
                if Q.grad is not None: Q.grad = None
                if K.grad is not None: K.grad = None
                if V.grad is not None: V.grad = None
                O = flash_v9_attention(Q, K, V, is_causal=causal)
                O.sum().backward()
        elif impl == 'ref':
            def step():
                if Q.grad is not None: Q.grad = None
                if K.grad is not None: K.grad = None
                if V.grad is not None: V.grad = None
                O = reference_attention(Q, K, V, is_causal=causal)
                O.sum().backward()
        elif impl == 'fa2':
            if not HAVE_FLASH_ATTN: return None
            def step():
                if Q.grad is not None: Q.grad = None
                if K.grad is not None: K.grad = None
                if V.grad is not None: V.grad = None
                O = _fa2_attention(Q, K, V, causal)
                O.sum().backward()
        else: raise ValueError(impl)
        return step
    step = make_step()
    if step is None: return None
    return _time_op(step, n_warmup, n_trials)


def bench_full(impl, B, H, N, D, causal, dtype, device, n_warmup, n_trials):
    """fwd + bwd + dbl_bwd (HVP through dQ/dK/dV * random unit vectors)."""
    if impl == 'fa2':
        # FA2 doesn't support double_backward natively.
        return None

    def make_step():
        Q, K, V = _build(B, H, N, D, dtype, device, requires_grad=True)
        uQ = torch.randn_like(Q)
        uK = torch.randn_like(K)
        uV = torch.randn_like(V)
        if impl == 'v9':
            def step():
                O = flash_v9_attention(Q, K, V, is_causal=causal)
                gQ, gK, gV = torch.autograd.grad(O.sum(), [Q, K, V], create_graph=True)
                h = (gQ * uQ).sum() + (gK * uK).sum() + (gV * uV).sum()
                torch.autograd.grad(h, [Q, K, V])
        elif impl == 'ref':
            def step():
                O = reference_attention(Q, K, V, is_causal=causal)
                gQ, gK, gV = torch.autograd.grad(O.sum(), [Q, K, V], create_graph=True)
                h = (gQ * uQ).sum() + (gK * uK).sum() + (gV * uV).sum()
                torch.autograd.grad(h, [Q, K, V])
        else: raise ValueError(impl)
        return step
    step = make_step()
    return _time_op(step, n_warmup, n_trials)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    args = ap.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    fa2_status = "available" if HAVE_FLASH_ATTN else "NOT installed (FA2 cells N/A)"
    print(f"[bench] dtype={args.dtype} warmup={args.warmup} trials={args.trials} "
          f"flash_attn={fa2_status}")
    print()

    results = []
    header = f"{'config':<12} {'op':<22} {'v9':>10} {'pytorch':>10} {'fa2':>10} {'v9/pt':>8} {'v9/fa2':>8}"
    print(header)
    print("-" * len(header))

    for label, B, H, N, D, causal in CONFIGS:
        cfg = f"{label}"
        for op_name, op_fn in [
            ('fwd',                bench_fwd),
            ('fwd+bwd',            bench_fwd_bwd),
            ('fwd+bwd+dbl_bwd',    bench_full),
        ]:
            try:
                t_v9  = op_fn('v9',  B, H, N, D, causal, dtype, device, args.warmup, args.trials)
            except Exception as e:
                t_v9  = None
            try:
                t_ref = op_fn('ref', B, H, N, D, causal, dtype, device, args.warmup, args.trials)
            except Exception:
                t_ref = None
            try:
                t_fa2 = op_fn('fa2', B, H, N, D, causal, dtype, device, args.warmup, args.trials)
            except Exception:
                t_fa2 = None
            def fmt(x): return f"{x:.3f}" if x is not None else "  N/A "
            def ratio(a, b): return f"{a/b:.2f}x" if (a and b) else "    -  "
            print(f"{cfg:<12} {op_name:<22} {fmt(t_v9):>10} {fmt(t_ref):>10} {fmt(t_fa2):>10} "
                  f"{ratio(t_v9, t_ref):>8} {ratio(t_v9, t_fa2):>8}")
            results.append({
                'config': label, 'B': B, 'H': H, 'N': N, 'D': D,
                'causal': causal, 'op': op_name,
                'v9_ms': t_v9, 'ref_ms': t_ref, 'fa2_ms': t_fa2,
                'dtype': args.dtype,
            })
        print()

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'records': results, 'fa2_available': HAVE_FLASH_ATTN,
                       'timestamp': time.time()}, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == '__main__':
    main()
