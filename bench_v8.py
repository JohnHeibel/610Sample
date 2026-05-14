"""
Bench v7 vs v8 wall time at the kernel level (fwd+bwd+double_bwd at one config).

Usage:
  python bench_v8.py --config 350M [--n-warmup 5] [--n-trials 30]
                     [--out profile_output/v8_iterations/baseline.json]
                     [--label baseline]

Times the three-call sequence (v7_forward → v7_backward → v7_double_backward)
end-to-end on CUDA events for each iteration; same for v8.

Reports median + per-iteration cv. Saves JSON record so iteration deltas are
trivially comparable.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import attention_cuda as _ac
from tests.common import MODEL_CONFIGS, N_Q, N_KV, WINDOW_SIZE


def build_inputs(B, H, N_q, N_kv, D, dtype, device):
    Q = torch.randn(B, H, N_q,  D, device=device, dtype=dtype) * 0.1
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype) * 0.1
    dO = torch.randn(B, H, N_q,  D, device=device, dtype=dtype) * 0.1
    return Q, K, V, dO


def run_seq(version, Q, K, V, dO, ws, cid):
    if version == 'v7':
        O, L = _ac.v7_forward(Q, K, V, ws, cid)
        dQ, dK, dV = _ac.v7_backward(dO, Q, K, V, O, L, ws, cid)
        gdQ = torch.zeros_like(Q); gdK = torch.zeros_like(K); gdV = torch.zeros_like(V)
        _ac.v7_double_backward(gdQ, gdK, gdV, dO, Q, K, V, O, L, ws, cid)
    else:
        O, L = _ac.v8_forward(Q, K, V, ws, cid)
        dQ, dK, dV = _ac.v8_backward(dO, Q, K, V, O, L, ws, cid)
        gdQ = torch.zeros_like(Q); gdK = torch.zeros_like(K); gdV = torch.zeros_like(V)
        _ac.v8_double_backward(gdQ, gdK, gdV, dO, Q, K, V, O, L, ws, cid)


def time_version(version, Q, K, V, dO, ws, cid, n_warmup, n_trials):
    for _ in range(n_warmup):
        run_seq(version, Q, K, V, dO, ws, cid)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_trials):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        run_seq(version, Q, K, V, dO, ws, cid)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='350M', choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--chunk-id', type=int, default=4)
    ap.add_argument('--n-warmup', type=int, default=5)
    ap.add_argument('--n-trials', type=int, default=30)
    ap.add_argument('--versions', nargs='+', default=['v7', 'v8'])
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--label', type=str, default='unlabeled')
    args = ap.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    cfg = MODEL_CONFIGS[args.config]
    H, D = cfg['H'], cfg['D']

    print(f"[bench] config={args.config} H={H} D={D} B={args.batch} "
          f"N_q={N_Q} N_kv={N_KV} ws={WINDOW_SIZE} dtype={args.dtype} "
          f"warmup={args.n_warmup} trials={args.n_trials}")

    Q, K, V, dO = build_inputs(args.batch, H, N_Q, N_KV, D, dtype, device)

    record = {
        'label': args.label,
        'config': args.config,
        'H': H, 'D': D, 'B': args.batch,
        'N_q': N_Q, 'N_kv': N_KV, 'window_size': WINDOW_SIZE,
        'dtype': args.dtype,
        'chunk_id': args.chunk_id,
        'n_warmup': args.n_warmup, 'n_trials': args.n_trials,
        'timestamp': time.time(),
        'results': {},
    }

    for ver in args.versions:
        try:
            times = time_version(ver, Q, K, V, dO, WINDOW_SIZE, args.chunk_id,
                                 args.n_warmup, args.n_trials)
            med = statistics.median(times)
            mean = statistics.fmean(times)
            stdev = statistics.stdev(times) if len(times) > 1 else 0.0
            tmin, tmax = min(times), max(times)
            cv = stdev / mean if mean > 0 else 0.0
            print(f"  [{ver}] median={med:.3f}ms  mean={mean:.3f}ms  "
                  f"min={tmin:.3f}ms  max={tmax:.3f}ms  cv={cv*100:.1f}%")
            record['results'][ver] = {
                'median_ms': med, 'mean_ms': mean, 'stdev_ms': stdev,
                'min_ms': tmin, 'max_ms': tmax, 'cv': cv,
                'n_trials': len(times), 'all_times_ms': times,
            }
        except Exception as e:
            print(f"  [{ver}] ERROR: {e}")
            record['results'][ver] = {'error': str(e)}

    if 'v7' in record['results'] and 'v8' in record['results'] \
            and 'median_ms' in record['results']['v7'] \
            and 'median_ms' in record['results']['v8']:
        v7_med = record['results']['v7']['median_ms']
        v8_med = record['results']['v8']['median_ms']
        speedup = v7_med / v8_med
        record['v8_vs_v7_speedup'] = speedup
        record['v8_vs_v7_pct_faster'] = (1 - v8_med / v7_med) * 100
        print(f"  [delta] v8/v7 speedup = {speedup:.4f}x  "
              f"({record['v8_vs_v7_pct_faster']:+.2f}% faster)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(record, f, indent=2)
        print(f"  [saved] {args.out}")


if __name__ == '__main__':
    main()
