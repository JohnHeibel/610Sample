"""
Shared test utilities for V7 kernel test suite.
================================================
Provides model configs, chunking constants, reference implementations,
benchmarking helpers, and output utilities used across all test files.
"""

import os
import csv
import time
import math
import argparse
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# Model configurations
# =====================================================================

MODEL_CONFIGS = {
    '125M':  dict(H=12,  D=64),
    '350M':  dict(H=16,  D=64),
    '760M':  dict(H=16,  D=96),
    '1.3B':  dict(H=32,  D=64),
    '2.7B':  dict(H=32,  D=80),
    '6.7B':  dict(H=64,  D=64),
    '13B':   dict(H=64,  D=80),
    '20B':   dict(H=64,  D=96),
    '65B':   dict(H=128, D=80),
}

STANDARD_MODELS = ['125M', '350M', '760M', '1.3B', '2.7B']
LARGE_MODELS = ['6.7B', '13B', '20B', '65B']
ALL_MODELS = STANDARD_MODELS + LARGE_MODELS

# =====================================================================
# Chunking constants (fixed for all configs)
# =====================================================================

N_Q = 1024
N_KV = 9216
WINDOW_SIZE = 8192
N_CHUNKS = 8
BATCH_SIZE = 1

# =====================================================================
# Mask construction
# =====================================================================

def make_ttt_mask(chunk_id, N_q, N_kv, window_size, device):
    """Replicate TTT-E2E sw_causal_mask: (qi >= ki) & (qi < ki + ws) & (ki >= 0)"""
    starting_query_idx = chunk_id * N_q
    ending_query_idx = starting_query_idx + N_q
    ending_key_idx = ending_query_idx

    qi = (torch.arange(N_q, device=device, dtype=torch.int32) + starting_query_idx).unsqueeze(1)
    ki = (torch.arange(-N_kv, 0, device=device, dtype=torch.int32) + ending_key_idx).unsqueeze(0)

    mask = (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    return mask  # [N_q, N_kv] bool


# =====================================================================
# Reference attention implementations
# =====================================================================

def reference_attention_matmul(Q, K, V, mask, scale=None):
    """Direct matmul attention with create_graph support.
    Q: [B, H, N_q, D], K: [B, H, N_kv, D], V: [B, H, N_kv, D]
    mask: [N_q, N_kv] bool
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    O = torch.matmul(P, V)
    return O


def v7_attention(Q, K, V, chunk_id, window_size):
    """V7 custom kernel attention via autograd wrapper."""
    from attention import flash_attention_v7
    return flash_attention_v7(Q, K, V, chunk_id=chunk_id, window_size=window_size)


# =====================================================================
# Meta-step simulation
# =====================================================================

def run_meta_step(B, H, D, method, device, dtype,
                  N_q=N_Q, N_kv=N_KV, window_size=WINDOW_SIZE, n_chunks=N_CHUNKS):
    """Simulate one TTT-E2E meta-step: n_chunks chunks, fwd+bwd+double_bwd.
    Returns (W_q.grad, W_k.grad, W_v.grad).
    """
    scale = 1.0 / (D ** 0.5)

    W_q = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_k = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_v = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_q.requires_grad_(True)
    W_k.requires_grad_(True)
    W_v.requires_grad_(True)

    all_tokens = torch.randn(B, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1

    meta_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    for chunk_id in range(n_chunks):
        start = chunk_id * N_q
        end = start + N_q
        x_q = all_tokens[:, :, start:end, :]

        kv_start = max(0, end - N_kv)
        x_kv = all_tokens[:, :, kv_start:end, :]
        if x_kv.shape[2] < N_kv:
            pad_size = N_kv - x_kv.shape[2]
            padding = torch.zeros(B, H, pad_size, D, device=device, dtype=dtype)
            x_kv = torch.cat([padding, x_kv], dim=2)

        Q = torch.matmul(x_q, W_q)
        K = torch.matmul(x_kv, W_k)
        V = torch.matmul(x_kv, W_v)

        if method == 'matmul':
            mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
            O = reference_attention_matmul(Q, K, V, mask, scale)
        elif method == 'v7':
            O = v7_attention(Q, K, V, chunk_id, window_size)
        else:
            raise ValueError(f"Unknown method: {method}")

        chunk_loss = O.sum()
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    meta_loss.backward()
    return W_q.grad, W_k.grad, W_v.grad


# =====================================================================
# Benchmarking helpers
# =====================================================================

def benchmark_method(fn, n_warmup=3, n_trials=10, device='cuda'):
    """Run fn() with warmup, return (median_time_s, peak_mem_mb, error_or_None).
    Catches OOM and returns ('OOM', peak, 'OOM') on failure.
    """
    # Warmup
    for _ in range(n_warmup):
        try:
            torch.cuda.empty_cache()
            fn()
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, None, 'OOM'
        except Exception as e:
            return None, None, str(e)

    # Timed runs
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        try:
            fn()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None, None, 'OOM'
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    median_time = sorted(times)[len(times) // 2]
    return median_time, peak_mem, None


# =====================================================================
# Output utilities
# =====================================================================

def add_output_dir_arg(parser):
    """Add --output-dir argument to an argparse parser."""
    parser.add_argument('--output-dir', type=str, default='test-output',
                        help='Directory to write CSV/PNG results')
    return parser


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_csv(filepath, headers, rows):
    """Write rows to a CSV file."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  Saved CSV: {filepath}")


def save_plot(fig, filepath):
    """Save a matplotlib figure to PNG."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved PNG: {filepath}")
