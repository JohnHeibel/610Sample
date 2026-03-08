"""
Test 1: V7 Kernel Correctness
==============================
Compares V7 custom kernels against matmul reference for forward, backward,
and double backward on random tensors.

Outputs:
  - correctness_errors.csv
  - correctness_errors.png (grouped bar chart, log scale)

Run:
  python tests/test_correctness.py --output-dir test-output
"""

import sys
import os
import math
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.common import (
    make_ttt_mask, reference_attention_matmul, save_csv, save_plot,
    add_output_dir_arg, ensure_output_dir,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# Tolerances
# =====================================================================
ATOL_FWD = 5e-3
RTOL_FWD = 5e-3
ATOL_BWD = 5e-3
RTOL_BWD = 5e-3
ATOL_DBL = 1.5e-2
RTOL_DBL = 1.5e-2

# =====================================================================
# Test configs: (B, H, N_q, N_kv, D, window_size, chunk_id)
# =====================================================================
TEST_CONFIGS = [
    ('cfg1', 1, 1, 32, 128, 64, 96, 1),
    ('cfg2', 1, 2, 64, 256, 64, 192, 2),
    ('cfg3', 2, 4, 64, 256, 64, 192, 3),
    ('cfg4', 1, 1, 64, 256, 80, 192, 1),
    ('cfg5', 1, 1, 64, 256, 96, 192, 2),
    ('cfg6', 1, 1, 64, 256, 64, 192, 0),   # chunk_id=0 edge case
    ('cfg7', 1, 2, 32, 128, 64, 96, 0),    # chunk_id=0 edge case
    ('cfg8', 2, 2, 64, 256, 64, 192, 1),   # multi-batch
]


def reference_forward(Q, K, V, mask):
    """Reference attention forward in float32."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    O = torch.matmul(P, V)
    return O


def test_forward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 forward, return (max_abs, max_rel, passed)."""
    import attention_cuda

    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
    ref_O = reference_forward(Q.float(), K.float(), V.float(), mask)
    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)

    diff = (O.float() - ref_O).abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref_O.abs() + 1e-8)).max().item()
    passed = torch.allclose(O.float(), ref_O, atol=ATOL_FWD, rtol=RTOL_FWD)
    return max_abs, max_rel, passed


def test_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 backward, return list of (tensor_name, max_abs, max_rel, passed)."""
    import attention_cuda

    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

    Q_ref = Q.detach().float().requires_grad_(True)
    K_ref = K.detach().float().requires_grad_(True)
    V_ref = V.detach().float().requires_grad_(True)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_O.sum().backward()

    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)
    dO = torch.ones_like(O)
    dQ, dK, dV = attention_cuda.v7_backward(dO, Q, K, V, O, L, window_size, chunk_id)

    results = []
    for name, actual, expected in [('dQ', dQ, Q_ref.grad), ('dK', dK, K_ref.grad), ('dV', dV, V_ref.grad)]:
        diff = (actual.float() - expected).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (expected.abs() + 1e-8)).max().item()
        passed = torch.allclose(actual.float(), expected, atol=ATOL_BWD, rtol=RTOL_BWD)
        results.append((name, max_abs, max_rel, passed))
    return results


def test_double_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 double backward via scalar probe, return list of (name, max_abs, max_rel, passed)."""
    from attention import flash_attention_v7

    device = 'cuda'
    torch.manual_seed(42)

    Q_ref = torch.randn(B, H, N_q, D, device=device, dtype=torch.float32, requires_grad=True)
    K_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)
    V_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_grads = torch.autograd.grad(ref_O.sum(), [Q_ref, K_ref, V_ref], create_graph=True)

    uQ = torch.randn_like(Q_ref)
    uK = torch.randn_like(K_ref)
    uV = torch.randn_like(V_ref)
    h_ref = sum((g * u).sum() for g, u in zip(ref_grads, [uQ, uK, uV]))
    ref_g2 = torch.autograd.grad(h_ref, [Q_ref, K_ref, V_ref])

    Q_v7 = Q_ref.detach().clone().to(dtype).requires_grad_(True)
    K_v7 = K_ref.detach().clone().to(dtype).requires_grad_(True)
    V_v7 = V_ref.detach().clone().to(dtype).requires_grad_(True)

    v7_O = flash_attention_v7(Q_v7, K_v7, V_v7, window_size=window_size, chunk_id=chunk_id)
    v7_grads = torch.autograd.grad(v7_O.sum(), [Q_v7, K_v7, V_v7], create_graph=True)

    uQ_v7, uK_v7, uV_v7 = uQ.to(dtype), uK.to(dtype), uV.to(dtype)
    h_v7 = sum((g * u).sum() for g, u in zip(v7_grads, [uQ_v7, uK_v7, uV_v7]))
    v7_g2 = torch.autograd.grad(h_v7, [Q_v7, K_v7, V_v7])

    atol = ATOL_DBL if dtype == torch.bfloat16 else ATOL_FWD
    rtol = RTOL_DBL if dtype == torch.bfloat16 else RTOL_FWD

    results = []
    for name, v7_val, ref_val in [('g2_Q', v7_g2[0], ref_g2[0]),
                                   ('g2_K', v7_g2[1], ref_g2[1]),
                                   ('g2_V', v7_g2[2], ref_g2[2])]:
        diff = (v7_val.float() - ref_val).abs()
        max_abs = diff.max().item()
        max_rel = (diff / (ref_val.abs() + 1e-8)).max().item()
        passed = torch.allclose(v7_val.float(), ref_val, atol=atol, rtol=rtol)
        results.append((name, max_abs, max_rel, passed))
    return results


# =====================================================================
# Visualization
# =====================================================================

def make_correctness_graph(csv_rows, output_dir):
    """Create grouped bar chart of max absolute errors by config and pass type."""
    # Group by config
    configs = []
    seen = set()
    for row in csv_rows:
        cfg = row[0]
        if cfg not in seen:
            configs.append(cfg)
            seen.add(cfg)

    pass_types = ['fwd', 'bwd', 'dbl_bwd']
    pass_labels = ['Forward', 'Backward', 'Double Bwd']
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    # Aggregate: max error per (config, pass_type)
    data = {}
    for row in csv_rows:
        cfg, dtype_name, pass_type, tensor, max_abs, max_rel, passed = row
        key = (cfg, pass_type)
        val = float(max_abs)
        if key not in data or val > data[key]:
            data[key] = val

    fig, ax = plt.subplots(figsize=(max(10, len(configs) * 1.2), 6))
    x = np.arange(len(configs))
    width = 0.25

    for i, (pt, label, color) in enumerate(zip(pass_types, pass_labels, colors)):
        vals = []
        for cfg in configs:
            v = data.get((cfg, pt), 0)
            vals.append(v if v > 0 else 1e-10)
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_yscale('log')
    ax.set_xlabel('Config')
    ax.set_ylabel('Max Absolute Error')
    ax.set_title('V7 Correctness: Max Error by Config and Pass Type')
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.legend()

    # Tolerance threshold lines
    ax.axhline(y=ATOL_FWD, color='#2196F3', linestyle='--', alpha=0.5, label=f'fwd/bwd tol ({ATOL_FWD})')
    ax.axhline(y=ATOL_DBL, color='#FF9800', linestyle='--', alpha=0.5, label=f'dbl_bwd tol ({ATOL_DBL})')

    ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, os.path.join(output_dir, 'correctness_errors.png'))


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='V7 Correctness Tests')
    add_output_dir_arg(parser)
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    dtype = torch.bfloat16
    dtype_name = 'bf16'
    all_passed = True
    csv_rows = []

    print(f"\n{'='*60}")
    print(f"V7 Correctness Tests (dtype={dtype_name})")
    print(f"{'='*60}")

    # Forward tests
    print(f"\n--- Forward Tests ---")
    for cfg_name, B, H, N_q, N_kv, D, ws, cid in TEST_CONFIGS:
        label = f"{cfg_name} B={B} H={H} Nq={N_q} Nkv={N_kv} D={D} ws={ws} cid={cid}"
        try:
            max_abs, max_rel, passed = test_forward(B, H, N_q, N_kv, D, ws, cid, dtype)
            status = "PASS" if passed else "FAIL"
            print(f"  {label}: {status} (max_abs={max_abs:.2e})")
            csv_rows.append([cfg_name, dtype_name, 'fwd', 'O', f'{max_abs:.2e}', f'{max_rel:.2e}', status])
            all_passed &= passed
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            csv_rows.append([cfg_name, dtype_name, 'fwd', 'O', 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    # Backward tests
    print(f"\n--- Backward Tests ---")
    for cfg_name, B, H, N_q, N_kv, D, ws, cid in TEST_CONFIGS:
        label = f"{cfg_name} B={B} H={H} Nq={N_q} Nkv={N_kv} D={D} ws={ws} cid={cid}"
        try:
            results = test_backward(B, H, N_q, N_kv, D, ws, cid, dtype)
            for tensor_name, max_abs, max_rel, passed in results:
                status = "PASS" if passed else "FAIL"
                print(f"  {label} {tensor_name}: {status} (max_abs={max_abs:.2e})")
                csv_rows.append([cfg_name, dtype_name, 'bwd', tensor_name, f'{max_abs:.2e}', f'{max_rel:.2e}', status])
                all_passed &= passed
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            for t in ['dQ', 'dK', 'dV']:
                csv_rows.append([cfg_name, dtype_name, 'bwd', t, 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    # Double backward tests (skip D=96 — needs A100 shared memory)
    print(f"\n--- Double Backward Tests ---")
    dbl_bwd_configs = [(n, B, H, Nq, Nkv, D, ws, cid)
                       for n, B, H, Nq, Nkv, D, ws, cid in TEST_CONFIGS if D != 96]
    for cfg_name, B, H, N_q, N_kv, D, ws, cid in dbl_bwd_configs:
        label = f"{cfg_name} B={B} H={H} Nq={N_q} Nkv={N_kv} D={D} ws={ws} cid={cid}"
        try:
            results = test_double_backward(B, H, N_q, N_kv, D, ws, cid, dtype)
            for tensor_name, max_abs, max_rel, passed in results:
                status = "PASS" if passed else "FAIL"
                print(f"  {label} {tensor_name}: {status} (max_abs={max_abs:.2e})")
                csv_rows.append([cfg_name, dtype_name, 'dbl_bwd', tensor_name, f'{max_abs:.2e}', f'{max_rel:.2e}', status])
                all_passed &= passed
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            for t in ['g2_Q', 'g2_K', 'g2_V']:
                csv_rows.append([cfg_name, dtype_name, 'dbl_bwd', t, 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    # Save outputs
    csv_path = os.path.join(output_dir, 'correctness_errors.csv')
    save_csv(csv_path,
             ['config', 'dtype', 'pass_type', 'tensor', 'max_abs', 'max_rel', 'passed'],
             csv_rows)

    try:
        make_correctness_graph(csv_rows, output_dir)
    except Exception as e:
        print(f"  Warning: could not generate graph: {e}")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
