"""
Test 5: Scaling Limits
=======================
Two scaling limit tests with 75GB memory budget:

A) Max Batch Size: Binary search for largest B per model config, matmul vs V7.
B) Max Heads: Fix D=64, B=1, binary search for largest H, matmul vs V7.

Outputs:
  - scaling_batch.csv + scaling_max_batch.png
  - scaling_heads.csv + scaling_max_heads.png

Run:
  python tests/test_scaling.py --output-dir test-output
"""

import sys
import os
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.common import (
    MODEL_CONFIGS, ALL_MODELS,
    N_Q, N_KV, WINDOW_SIZE, N_CHUNKS,
    make_ttt_mask, reference_attention_matmul, v7_attention,
    save_csv, save_plot, add_output_dir_arg, ensure_output_dir,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MEM_BUDGET_MB = 75 * 1024  # 75 GB


def _run_meta_step_with_batch(B, H, D, method, device, dtype):
    """Run meta-step with specified batch size. Raises on OOM."""
    scale = 1.0 / (D ** 0.5)
    N_q, N_kv, window_size, n_chunks = N_Q, N_KV, WINDOW_SIZE, N_CHUNKS

    W_q = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    W_k = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
    W_v = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
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

        chunk_loss = O.sum()
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    meta_loss.backward()
    torch.cuda.synchronize()


def find_max_batch(H, D, method, device, dtype, mem_budget_mb):
    """Binary search for max batch size within memory budget."""
    lo, hi = 1, 128
    best = 0

    # Check B=1 first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        _run_meta_step_with_batch(1, H, D, method, device, dtype)
        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.empty_cache()
        if mem > mem_budget_mb:
            return 0
        best = 1
        # Estimate upper bound
        hi = min(128, max(1, int(mem_budget_mb / mem)))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return 0

    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            _run_meta_step_with_batch(mid, H, D, method, device, dtype)
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.empty_cache()
            if mem <= mem_budget_mb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            hi = mid - 1

    return best


def find_max_heads(D, method, device, dtype, mem_budget_mb):
    """Binary search for max H with D fixed, B=1."""
    lo, hi = 1, 512
    best = 0

    # Check H=1 first
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        _run_meta_step_with_batch(1, 1, D, method, device, dtype)
        mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.empty_cache()
        if mem > mem_budget_mb:
            return 0
        best = 1
        hi = min(512, max(1, int(mem_budget_mb / mem)))
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return 0

    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            _run_meta_step_with_batch(1, mid, D, method, device, dtype)
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            torch.cuda.empty_cache()
            if mem <= mem_budget_mb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            hi = mid - 1

    return best


def make_batch_graph(results, output_dir):
    """Grouped bar chart of max batch size per model."""
    models = [m for m in ALL_MODELS if m in results]
    methods = ['matmul', 'v7']
    colors = {'matmul': '#2196F3', 'v7': '#4CAF50'}
    labels = {'matmul': 'Matmul', 'v7': 'V7'}

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
    x = np.arange(len(models))
    width = 0.35

    for i, method in enumerate(methods):
        vals = [results[m].get(method, 0) for m in models]
        ax.bar(x + i * width, vals, width, label=labels[method],
               color=colors[method], alpha=0.85)

    ax.set_xlabel('Model Config')
    ax.set_ylabel('Max Batch Size')
    ax.set_title(f'Scaling: Max Batch Size ({MEM_BUDGET_MB/1024:.0f} GB budget)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    save_plot(fig, os.path.join(output_dir, 'scaling_max_batch.png'))


def make_heads_graph(results, output_dir):
    """Two-bar chart of max heads for matmul vs V7."""
    methods = ['matmul', 'v7']
    colors = ['#2196F3', '#4CAF50']
    labels = ['Matmul', 'V7']
    vals = [results.get(m, 0) for m in methods]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Max Heads (H)')
    ax.set_title(f'Scaling: Max Attention Heads (D=64, B=1, {MEM_BUDGET_MB/1024:.0f} GB budget)')
    fig.tight_layout()
    save_plot(fig, os.path.join(output_dir, 'scaling_max_heads.png'))


def main():
    parser = argparse.ArgumentParser(description='Scaling Limits')
    add_output_dir_arg(parser)
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models for batch scaling (default: all)')
    parser.add_argument('--mem-budget-gb', type=float, default=75,
                        help='Memory budget in GB')
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    global MEM_BUDGET_MB
    MEM_BUDGET_MB = args.mem_budget_gb * 1024

    device = 'cuda'
    dtype = torch.bfloat16
    models = args.models or ALL_MODELS

    print(f"\n{'='*60}")
    print(f"Scaling Limits (memory budget: {args.mem_budget_gb:.0f} GB)")
    print(f"{'='*60}")

    # =====================================================================
    # A) Max Batch Size
    # =====================================================================
    print(f"\n--- A) Max Batch Size per Model Config ---")

    batch_results = {}
    batch_csv_rows = []

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            continue

        cfg = MODEL_CONFIGS[model_name]
        H, D = cfg['H'], cfg['D']
        print(f"\n  [{model_name}] H={H}, D={D}")
        batch_results[model_name] = {}

        for method in ['matmul', 'v7']:
            print(f"    {method}...", end='', flush=True)
            max_b = find_max_batch(H, D, method, device, dtype, MEM_BUDGET_MB)
            print(f" max_B={max_b}")
            batch_results[model_name][method] = max_b
            batch_csv_rows.append([model_name, method, str(max_b)])

    save_csv(os.path.join(output_dir, 'scaling_batch.csv'),
             ['model', 'method', 'max_batch_size'],
             batch_csv_rows)

    try:
        make_batch_graph(batch_results, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate batch graph: {e}")

    # =====================================================================
    # B) Max Heads
    # =====================================================================
    print(f"\n--- B) Max Attention Heads (D=64, B=1) ---")

    heads_results = {}
    heads_csv_rows = []

    for method in ['matmul', 'v7']:
        print(f"  {method}...", end='', flush=True)
        max_h = find_max_heads(64, method, device, dtype, MEM_BUDGET_MB)
        print(f" max_H={max_h}")
        heads_results[method] = max_h
        heads_csv_rows.append([method, str(max_h)])

    save_csv(os.path.join(output_dir, 'scaling_heads.csv'),
             ['method', 'max_heads'],
             heads_csv_rows)

    try:
        make_heads_graph(heads_results, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate heads graph: {e}")

    print(f"\n{'='*60}")
    print(f"Scaling Tests Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
