"""
Profile driver for v8 kernels under Nsight Compute.

Mirrors profile_v7.py but invokes v8_* entry points so ncu --kernel-name
'regex:v8_' captures only the v8 launches.

Usage:
  python profile_v8.py --config 350M
"""

import argparse
import sys
import os

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


def run_once(Q, K, V, dO, window_size, chunk_id):
    O, L = _ac.v8_forward(Q, K, V, window_size, chunk_id)
    dQ, dK, dV = _ac.v8_backward(dO, Q, K, V, O, L, window_size, chunk_id)
    g_dQ = torch.zeros_like(Q); g_dK = torch.zeros_like(K); g_dV = torch.zeros_like(V)
    _ac.v8_double_backward(g_dQ, g_dK, g_dV, dO, Q, K, V, O, L, window_size, chunk_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16'])
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--chunk-id', type=int, default=4)
    args = ap.parse_args()

    device = 'cuda'
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    cfg = MODEL_CONFIGS[args.config]
    H, D = cfg['H'], cfg['D']

    print(f"[profile_v8] config={args.config}  H={H}  D={D}  "
          f"B={args.batch}  N_q={N_Q}  N_kv={N_KV}  window={WINDOW_SIZE}  "
          f"dtype={args.dtype}")

    Q, K, V, dO = build_inputs(args.batch, H, N_Q, N_KV, D, dtype, device)
    for _ in range(2):
        run_once(Q, K, V, dO, WINDOW_SIZE, args.chunk_id)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    run_once(Q, K, V, dO, WINDOW_SIZE, args.chunk_id)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[profile_v8] done")


if __name__ == '__main__':
    main()
