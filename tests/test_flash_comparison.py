"""
Test 4: V7 vs FlashAttention-2 Comparison
==========================================
Two parts:
  A) Functional correctness: V7 vs Flash forward & backward outputs match
  B) Performance: wall clock and memory benchmarks

Causal mask, square sequences (N_q=N_kv=1024).

Limitations (printed clearly):
  - Flash doesn't support create_graph (no double backward)
  - Flash doesn't support sliding-window + validity mask
  - Only causal masking on square sequences is comparable

Outputs:
  - flash_correctness.csv + flash_correctness.png
  - flash_comparison.csv
  - flash_wallclock.png
  - flash_memory.png

Run:
  python tests/test_flash_comparison.py --output-dir test-output
"""

import sys
import os
import time
import argparse

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.common import (
    MODEL_CONFIGS, ALL_MODELS,
    benchmark_method,
    save_csv, save_plot, add_output_dir_arg, ensure_output_dir,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

N = 1024  # Square sequence length for fair comparison
B = 1


def check_flash_available():
    """Check if FlashAttention-2 backend is available."""
    try:
        Q = torch.randn(1, 1, 32, 64, device='cuda', dtype=torch.bfloat16)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(Q, Q, Q, is_causal=True)
        return True
    except Exception:
        return False


# =====================================================================
# Correctness: V7 vs FlashAttention-2
# =====================================================================

# Flash and V7 are different implementations (different accumulation order,
# different numerics), so we use cross-implementation tolerances.
FLASH_ATOL = 1e-2
FLASH_RTOL = 1e-2

# Correctness configs: smaller sequences to avoid dominating runtime
CORRECTNESS_CONFIGS = [
    # (label, B, H, N, D)
    ('H1_D64',    1,  1,  256, 64),
    ('H4_D64',    1,  4,  256, 64),
    ('H12_D64',   1, 12,  256, 64),
    ('H12_D64_L', 1, 12, 1024, 64),
    ('H16_D64',   1, 16,  256, 64),
    ('H32_D80',   1, 32,  256, 80),
    ('H8_D96',    1,  8,  256, 96),
    ('H64_D64',   1, 64,  256, 64),
]


def test_flash_correctness_config(B, H, N_seq, D, dtype, device):
    """Compare V7 vs Flash forward and backward on one config.
    Returns list of (tensor_name, max_abs, max_rel, passed).
    """
    from attention import flash_attention_v7

    torch.manual_seed(42)
    Q = torch.randn(B, H, N_seq, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N_seq, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N_seq, D, device=device, dtype=dtype, requires_grad=True)

    # --- Flash forward+backward ---
    Q_f = Q.detach().clone().requires_grad_(True)
    K_f = K.detach().clone().requires_grad_(True)
    V_f = V.detach().clone().requires_grad_(True)

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        O_flash = F.scaled_dot_product_attention(Q_f, K_f, V_f, is_causal=True)
    O_flash.sum().backward()

    # --- V7 forward+backward ---
    # chunk_id=1, window_size=N_seq: all keys valid, equivalent to causal
    Q_v = Q.detach().clone().requires_grad_(True)
    K_v = K.detach().clone().requires_grad_(True)
    V_v = V.detach().clone().requires_grad_(True)

    O_v7 = flash_attention_v7(Q_v, K_v, V_v, window_size=N_seq, chunk_id=1)
    O_v7.sum().backward()

    # --- Compare ---
    results = []

    # Forward: O
    diff = (O_v7.float() - O_flash.float()).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (O_flash.float().abs() + 1e-8)).max().item()
    passed = torch.allclose(O_v7.float(), O_flash.float(), atol=FLASH_ATOL, rtol=FLASH_RTOL)
    results.append(('O', max_abs, max_rel, passed))

    # Backward: dQ, dK, dV
    for name, v7_g, flash_g in [('dQ', Q_v.grad, Q_f.grad),
                                  ('dK', K_v.grad, K_f.grad),
                                  ('dV', V_v.grad, V_f.grad)]:
        diff = (v7_g.float() - flash_g.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (flash_g.float().abs() + 1e-8)).max().item()
        passed = torch.allclose(v7_g.float(), flash_g.float(), atol=FLASH_ATOL, rtol=FLASH_RTOL)
        results.append((name, max_abs, max_rel, passed))

    return results


def make_flash_correctness_graph(csv_rows, output_dir):
    """Grouped bar chart of max absolute errors: V7 vs Flash by config and tensor."""
    # Collect unique configs in order
    configs = []
    seen = set()
    for row in csv_rows:
        cfg = row[0]
        if cfg not in seen:
            configs.append(cfg)
            seen.add(cfg)

    tensors = ['O', 'dQ', 'dK', 'dV']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    # max error per (config, tensor)
    data = {}
    for row in csv_rows:
        cfg, dtype_name, tensor, max_abs_str, max_rel_str, status = row
        try:
            val = float(max_abs_str)
        except ValueError:
            val = 0
        key = (cfg, tensor)
        if key not in data or val > data[key]:
            data[key] = val

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.5), 6))
    x = np.arange(len(configs))
    width = 0.2

    for i, (tensor, color) in enumerate(zip(tensors, colors)):
        vals = []
        for cfg in configs:
            v = data.get((cfg, tensor), 0)
            vals.append(v if v > 0 else 1e-10)
        ax.bar(x + i * width, vals, width, label=tensor, color=color, alpha=0.85)

    ax.set_yscale('log')
    ax.set_xlabel('Config')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('V7 vs FlashAttention-2: Functional Correctness (fwd+bwd)')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)

    ax.axhline(y=FLASH_ATOL, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(configs) - 0.5, FLASH_ATOL * 1.3, f'tol ({FLASH_ATOL})',
            ha='right', fontsize=8, color='red', alpha=0.7)

    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, os.path.join(output_dir, 'flash_correctness.png'))


def run_flash_correctness(output_dir, device, dtype):
    """Run all flash correctness configs, return True if all passed."""
    dtype_name = 'bf16' if dtype == torch.bfloat16 else 'fp16'
    all_passed = True
    csv_rows = []

    print(f"\n--- Functional Correctness: V7 vs FlashAttention-2 ---")
    print(f"  Tolerances: ATOL={FLASH_ATOL}, RTOL={FLASH_RTOL}")
    print(f"  Square causal sequences, chunk_id=1, window_size=N\n")

    for label, b, h, n_seq, d in CORRECTNESS_CONFIGS:
        print(f"  [{label}] B={b} H={h} N={n_seq} D={d}")
        try:
            results = test_flash_correctness_config(b, h, n_seq, d, dtype, device)
            for tensor_name, max_abs, max_rel, passed in results:
                status = "PASS" if passed else "FAIL"
                print(f"    {tensor_name}: {status} (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")
                csv_rows.append([label, dtype_name, tensor_name,
                                 f'{max_abs:.2e}', f'{max_rel:.2e}', status])
                all_passed &= passed
        except Exception as e:
            print(f"    ERROR: {e}")
            for t in ['O', 'dQ', 'dK', 'dV']:
                csv_rows.append([label, dtype_name, t, 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    # Save CSV
    save_csv(os.path.join(output_dir, 'flash_correctness.csv'),
             ['config', 'dtype', 'tensor', 'max_abs', 'max_rel', 'passed'],
             csv_rows)

    # Save graph
    try:
        make_flash_correctness_graph(csv_rows, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate correctness graph: {e}")

    return all_passed


# =====================================================================
# Performance benchmarks
# =====================================================================

def bench_flash(H, D, device, dtype, n_warmup, n_trials):
    """Benchmark FlashAttention-2 forward+backward."""
    def fn():
        Q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        O.sum().backward()
        torch.cuda.synchronize()

    return benchmark_method(fn, n_warmup=n_warmup, n_trials=n_trials, device=device)


def bench_v7(H, D, device, dtype, n_warmup, n_trials):
    """Benchmark V7 forward+backward on square causal sequence."""
    from attention import flash_attention_v7

    # chunk_id=1, window_size=N so all keys are valid (equivalent to causal)
    chunk_id = 1
    window_size = N

    def fn():
        Q = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        O = flash_attention_v7(Q, K, V, window_size=window_size, chunk_id=chunk_id)
        O.sum().backward()
        torch.cuda.synchronize()

    return benchmark_method(fn, n_warmup=n_warmup, n_trials=n_trials, device=device)


def make_flash_graphs(results, output_dir):
    """Create wallclock and memory comparison graphs."""
    models = [m for m in ALL_MODELS if m in results]
    methods = ['flash', 'v7']
    colors = {'flash': '#FF5722', 'v7': '#4CAF50'}
    labels = {'flash': 'FlashAttention-2', 'v7': 'V7'}

    for metric, ylabel, title, filename, convert in [
        ('time', 'Median Time (ms)', 'V7 vs FlashAttention-2: Wall Clock (fwd+bwd)',
         'flash_wallclock.png', lambda t, m: t * 1000 if t else 0),
        ('mem', 'Peak Memory (MB)', 'V7 vs FlashAttention-2: Peak Memory (fwd+bwd)',
         'flash_memory.png', lambda t, m: m if m else 0),
    ]:
        fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.5), 6))
        x = np.arange(len(models))
        width = 0.35

        for i, method in enumerate(methods):
            vals = []
            oom_mask = []
            for m in models:
                entry = results[m].get(method)
                if entry is None or entry[0] is None:
                    vals.append(0)
                    oom_mask.append(True)
                else:
                    t, mem = entry
                    if metric == 'time':
                        vals.append(t * 1000)
                    else:
                        vals.append(mem)
                    oom_mask.append(False)

            bars = ax.bar(x + i * width, vals, width, label=labels[method],
                          color=colors[method], alpha=0.85)

            for j, (is_oom, bar) in enumerate(zip(oom_mask, bars)):
                if is_oom:
                    bar.set_hatch('///')
                    bar.set_alpha(0.3)
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.5, 'OOM',
                            ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

        ax.set_xlabel('Model Config')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        fig.tight_layout()
        save_plot(fig, os.path.join(output_dir, filename))


def main():
    parser = argparse.ArgumentParser(description='V7 vs FlashAttention-2')
    add_output_dir_arg(parser)
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to benchmark (default: all)')
    parser.add_argument('--n-warmup', type=int, default=3)
    parser.add_argument('--n-trials', type=int, default=10)
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    device = 'cuda'
    dtype = torch.bfloat16
    models = args.models or ALL_MODELS

    print(f"\n{'='*60}")
    print(f"V7 vs FlashAttention-2 Comparison")
    print(f"{'='*60}")
    print(f"\nLimitations:")
    print(f"  - Flash doesn't support create_graph (no double backward)")
    print(f"  - Flash doesn't support sliding-window + validity mask")
    print(f"  - Only causal masking on square sequences (N={N}) is comparable")
    print(f"  - Mode: forward + backward only\n")

    if not check_flash_available():
        print("FlashAttention-2 not available. Requires sm_80+ and PyTorch 2.2+.")
        sys.exit(1)

    # --- Part A: Functional correctness ---
    all_correct = run_flash_correctness(output_dir, device, dtype)

    # --- Part B: Performance benchmarks ---
    print(f"\n{'='*60}")
    print(f"Performance Benchmarks: V7 vs FlashAttention-2")
    print(f"{'='*60}")

    results = {}
    csv_rows = []

    for model_name in models:
        if model_name not in MODEL_CONFIGS:
            print(f"  Unknown model: {model_name}, skipping")
            continue

        cfg = MODEL_CONFIGS[model_name]
        H, D = cfg['H'], cfg['D']
        print(f"\n  [{model_name}] H={H}, D={D}")
        results[model_name] = {}

        # Flash
        print(f"    flash...", end='', flush=True)
        t, mem, err = bench_flash(H, D, device, dtype, args.n_warmup, args.n_trials)
        if err:
            print(f" {err}")
            csv_rows.append([model_name, 'flash', '', '', 'OOM' if err == 'OOM' else 'ERROR'])
        else:
            print(f" {t*1000:.2f} ms, {mem:.0f} MB")
            results[model_name]['flash'] = (t, mem)
            csv_rows.append([model_name, 'flash', f'{t*1000:.2f}', f'{mem:.0f}', 'OK'])

        # V7
        print(f"    v7...", end='', flush=True)
        t, mem, err = bench_v7(H, D, device, dtype, args.n_warmup, args.n_trials)
        if err:
            print(f" {err}")
            csv_rows.append([model_name, 'v7', '', '', 'OOM' if err == 'OOM' else 'ERROR'])
        else:
            print(f" {t*1000:.2f} ms, {mem:.0f} MB")
            results[model_name]['v7'] = (t, mem)
            csv_rows.append([model_name, 'v7', f'{t*1000:.2f}', f'{mem:.0f}', 'OK'])

        # Speedup
        if 'flash' in results[model_name] and 'v7' in results[model_name]:
            ft, fm = results[model_name]['flash']
            vt, vm = results[model_name]['v7']
            ratio = ft / vt
            label = "faster" if ratio > 1 else "slower"
            print(f"    V7 is {ratio:.2f}x vs flash ({label})")

    # Save CSV
    save_csv(os.path.join(output_dir, 'flash_comparison.csv'),
             ['model', 'method', 'median_time_ms', 'peak_memory_mb', 'status'],
             csv_rows)

    # Save graphs
    try:
        make_flash_graphs(results, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate graphs: {e}")

    print(f"\n{'='*60}")
    correctness_status = "ALL PASSED" if all_correct else "SOME FAILED"
    print(f"Flash Correctness: {correctness_status}")
    print(f"Flash Benchmarks: Complete")
    print(f"{'='*60}")
    sys.exit(0 if all_correct else 1)


if __name__ == '__main__':
    main()
