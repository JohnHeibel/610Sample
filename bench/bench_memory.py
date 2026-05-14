"""
Peak GPU memory benchmark for Flash v9 vs PyTorch reference and FA2.

Measures torch.cuda.max_memory_allocated for each (config, op) combination.
The memory story is the *primary* win over PyTorch reference attention,
which materializes the (B, H, N, N) attention matrix and quickly OOMs at
large N. v9 and FA2 stay O(B*H*N*D) throughout.

Usage:
    python bench/bench_memory.py [--out file.json]
"""

import argparse
import gc
import json
import math
import os
import sys

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
    ('small',   1,  8,   512,  64),
    ('med64',   1, 16,  1024,  64),
    ('med128',  1, 16,  1024, 128),
    ('large64', 1, 16,  2048,  64),
    ('xl64',    1, 16,  4096,  64),
    ('xxl64',   1, 16,  8192,  64),
]


def _build(B, H, N, D, dtype, device, *, requires_grad=False):
    Q = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    K = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    V = (torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.1)
    if requires_grad:
        Q.requires_grad_(True); K.requires_grad_(True); V.requires_grad_(True)
    return Q, K, V


def _fa2_attention(Q, K, V, is_causal=False):
    Q_t = Q.transpose(1, 2).contiguous()
    K_t = K.transpose(1, 2).contiguous()
    V_t = V.transpose(1, 2).contiguous()
    scale = 1.0 / math.sqrt(Q.shape[-1])
    O_t = flash_attn_func(Q_t, K_t, V_t, dropout_p=0.0,
                          softmax_scale=scale, causal=is_causal)
    return O_t.transpose(1, 2).contiguous()


def _peak_mem_mb(fn):
    """Run fn(), return peak GPU memory used during fn() in MB."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except Exception:
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    args = ap.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16

    fa2_status = "available" if HAVE_FLASH_ATTN else "NOT installed"
    print(f"[mem] dtype={args.dtype} flash_attn={fa2_status}")
    print()

    results = []
    header = f"{'config':<10} {'op':<22} {'v9 MB':>10} {'pytorch MB':>12} {'fa2 MB':>10}"
    print(header)
    print("-" * len(header))

    for label, B, H, N, D in CONFIGS:
        for op_name, run in [
            ('fwd',
             lambda Q, K, V, impl: (flash_v9_attention(Q, K, V) if impl == 'v9'
                                    else reference_attention(Q, K, V) if impl == 'ref'
                                    else _fa2_attention(Q, K, V))),
            ('fwd+bwd',
             lambda Q, K, V, impl: (flash_v9_attention(Q, K, V).sum().backward() if impl == 'v9'
                                    else reference_attention(Q, K, V).sum().backward() if impl == 'ref'
                                    else _fa2_attention(Q, K, V).sum().backward())),
        ]:
            mems = {}
            for impl in ('v9', 'ref', 'fa2'):
                if impl == 'fa2' and not HAVE_FLASH_ATTN:
                    mems[impl] = None; continue
                rg = (op_name != 'fwd')
                Q, K, V = _build(B, H, N, D, dtype, device, requires_grad=rg)
                mems[impl] = _peak_mem_mb(lambda: run(Q, K, V, impl))
                del Q, K, V
                gc.collect(); torch.cuda.empty_cache()
            def fmt(x): return f"{x:.1f}" if x is not None else "OOM/N/A"
            print(f"{label:<10} {op_name:<22} {fmt(mems['v9']):>10} {fmt(mems['ref']):>12} {fmt(mems['fa2']):>10}")
            results.append({
                'config': label, 'B': B, 'H': H, 'N': N, 'D': D,
                'op': op_name,
                'v9_mb': mems['v9'], 'ref_mb': mems['ref'], 'fa2_mb': mems['fa2'],
                'dtype': args.dtype,
            })

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'records': results, 'fa2_available': HAVE_FLASH_ATTN}, f, indent=2)
        print(f"[saved] {args.out}")


if __name__ == '__main__':
    main()
