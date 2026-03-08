"""
Test 3: Memory Usage Benchmarks
================================
Peak GPU memory for the same meta-step pipeline, all 9 model configs.

Outputs:
  - memory.csv
  - memory_comparison.png

Run:
  python tests/test_memory.py --output-dir test-output
"""

import sys
import os
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.common import (
    MODEL_CONFIGS, ALL_MODELS,
    N_Q, N_KV, WINDOW_SIZE, N_CHUNKS, BATCH_SIZE,
    run_meta_step, benchmark_method,
    save_csv, save_plot, add_output_dir_arg, ensure_output_dir,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def bench_memory(model_name, method, device, dtype, n_warmup=1, n_trials=3):
    """Benchmark peak memory for one model config and method."""
    cfg = MODEL_CONFIGS[model_name]
    H, D = cfg['H'], cfg['D']

    def fn():
        run_meta_step(BATCH_SIZE, H, D, method, device, dtype)

    # Reset memory stats before benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    return benchmark_method(fn, n_warmup=n_warmup, n_trials=n_trials, device=device)


def make_memory_graph(results, output_dir):
    """Create grouped bar chart of peak memory usage."""
    models = [m for m in ALL_MODELS if m in results]
    methods = ['matmul', 'v7']
    colors = {'matmul': '#2196F3', 'v7': '#4CAF50'}
    labels = {'matmul': 'Matmul', 'v7': 'V7'}

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
    x = np.arange(len(models))
    width = 0.35

    for i, method in enumerate(methods):
        vals = []
        oom_mask = []
        for m in models:
            entry = results[m].get(method)
            if entry is None or entry[1] is None:
                vals.append(0)
                oom_mask.append(True)
            else:
                vals.append(entry[1] / 1024)  # MB -> GB
                oom_mask.append(False)

        bars = ax.bar(x + i * width, vals, width, label=labels[method],
                      color=colors[method], alpha=0.85)

        for j, (is_oom, bar) in enumerate(zip(oom_mask, bars)):
            if is_oom:
                bar.set_hatch('///')
                bar.set_alpha(0.3)
                ax.text(bar.get_x() + bar.get_width() / 2, 0.5, 'OOM',
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

    # 80GB reference line
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(models) - 0.5, 81, '80 GB (A100)', ha='right', va='bottom',
            fontsize=8, color='red', alpha=0.7)

    ax.set_xlabel('Model Config')
    ax.set_ylabel('Peak Memory (GB)')
    ax.set_title('TTT Meta-Step Peak Memory: Matmul vs V7')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    save_plot(fig, os.path.join(output_dir, 'memory_comparison.png'))


def main():
    parser = argparse.ArgumentParser(description='Memory Usage Benchmarks')
    add_output_dir_arg(parser)
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to benchmark (default: all)')
    parser.add_argument('--n-warmup', type=int, default=1)
    parser.add_argument('--n-trials', type=int, default=3)
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    device = 'cuda'
    dtype = torch.bfloat16
    models = args.models or ALL_MODELS

    print(f"\n{'='*60}")
    print(f"Memory Usage Benchmarks")
    print(f"{'='*60}")

    results = {}
    csv_rows = []

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"  Unknown model: {model_name}, skipping")
            continue

        cfg = MODEL_CONFIGS[model_name]
        print(f"\n  [{model_name}] H={cfg['H']}, D={cfg['D']}, Hidden={cfg['H']*cfg['D']}")
        results[model_name] = {}

        for method in ['matmul', 'v7']:
            print(f"    {method}...", end='', flush=True)
            torch.cuda.empty_cache()
            t, mem, err = bench_memory(model_name, method, device, dtype,
                                        n_warmup=args.n_warmup, n_trials=args.n_trials)
            if err:
                print(f" {err}")
                csv_rows.append([model_name, method, '', 'OOM' if err == 'OOM' else 'ERROR'])
                results[model_name][method] = (None, None)
            else:
                mem_gb = mem / 1024
                print(f" {mem_gb:.2f} GB")
                csv_rows.append([model_name, method, f'{mem_gb:.3f}', 'OK'])
                results[model_name][method] = (t, mem)

    # Save CSV
    save_csv(os.path.join(output_dir, 'memory.csv'),
             ['model', 'method', 'peak_memory_gb', 'status'],
             csv_rows)

    # Save graph
    try:
        make_memory_graph(results, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate graph: {e}")

    print(f"\n{'='*60}")
    print(f"Memory Benchmarks Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
