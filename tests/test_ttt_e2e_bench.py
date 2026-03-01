"""
TTT-E2E Meta-Step Benchmark
============================
Recreates the exact TTT-E2E attention pattern in PyTorch:
  - 8 sequential chunks, each: Q[B,H,1024,D], KV[B,H,9216,D]
  - Causal + sliding window + validity mask
  - Forward -> loss -> backward(create_graph=True) -> meta_loss -> backward
  - Measures wall clock + peak GPU memory

Baselines:
  1. torch SDPA with explicit mask (math backend)
  2. Direct PyTorch matmul implementation with create_graph=True

Run:
  .venv/Scripts/python.exe tests/test_ttt_e2e_bench.py
"""

import argparse
import time
import torch
import torch.nn.functional as F


def make_ttt_mask(chunk_id, N_q, N_kv, window_size, device):
    """Replicate TTT-E2E sw_causal_mask: (qi >= ki) & (qi < ki + ws) & (ki >= 0)"""
    starting_query_idx = chunk_id * N_q
    ending_query_idx = starting_query_idx + N_q
    ending_key_idx = ending_query_idx

    qi = (torch.arange(N_q, device=device, dtype=torch.int32) + starting_query_idx).unsqueeze(1)
    ki = (torch.arange(-N_kv, 0, device=device, dtype=torch.int32) + ending_key_idx).unsqueeze(0)

    mask = (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    return mask  # [N_q, N_kv] bool


def reference_attention_matmul(Q, K, V, mask, scale):
    """Direct matmul attention with create_graph support."""
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    O = torch.matmul(P, V)
    return O


def reference_attention_sdpa(Q, K, V, mask):
    """PyTorch SDPA with explicit mask (math backend for create_graph support)."""
    attn_mask = torch.zeros_like(mask, dtype=Q.dtype)
    attn_mask.masked_fill_(~mask, float('-inf'))
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N_q, N_kv]

    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        O = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    return O


def v7_attention(Q, K, V, chunk_id, window_size):
    """V7 custom kernel attention (placeholder — import when available)."""
    try:
        from attention import flash_attention_v7
        return flash_attention_v7(Q, K, V, chunk_id=chunk_id, window_size=window_size)
    except (ImportError, AttributeError):
        return None


def run_meta_step(B, H, N_q, N_kv, D, window_size, n_chunks, method, device, dtype):
    """Simulate one TTT-E2E meta-step: 8 chunks, fwd+bwd+double_bwd."""
    scale = 1.0 / (D ** 0.5)

    # Create persistent parameters
    W_q = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_k = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_v = torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02
    W_q.requires_grad_(True)
    W_k.requires_grad_(True)
    W_v.requires_grad_(True)

    # Pre-generate all input tokens
    all_tokens = torch.randn(B, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1

    meta_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    for chunk_id in range(n_chunks):
        # Current chunk queries
        start = chunk_id * N_q
        end = start + N_q
        x_q = all_tokens[:, :, start:end, :]  # [B, H, N_q, D]

        # KV cache: previous chunks + current
        kv_start = max(0, end - N_kv)
        x_kv = all_tokens[:, :, kv_start:end, :]
        # Pad if kv_start == 0 and we need more
        if x_kv.shape[2] < N_kv:
            pad_size = N_kv - x_kv.shape[2]
            padding = torch.zeros(B, H, pad_size, D, device=device, dtype=dtype)
            x_kv = torch.cat([padding, x_kv], dim=2)

        Q = torch.matmul(x_q, W_q)
        K = torch.matmul(x_kv, W_k)
        V = torch.matmul(x_kv, W_v)

        mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

        if method == 'matmul':
            O = reference_attention_matmul(Q, K, V, mask, scale)
        elif method == 'sdpa':
            O = reference_attention_sdpa(Q, K, V, mask)
        elif method == 'v7':
            O = v7_attention(Q, K, V, chunk_id, window_size)
            if O is None:
                raise RuntimeError("V7 not available")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Inner loss
        chunk_loss = O.sum()
        # First backward with create_graph for 2nd order grads
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        # Meta loss from gradients
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    # Second backward (triggers double backward)
    meta_loss.backward()

    return W_q.grad, W_k.grad, W_v.grad


def benchmark_method(method, B, H, N_q, N_kv, D, window_size, n_chunks, n_warmup, n_trials, device, dtype):
    """Benchmark a method, return median time and peak memory."""
    torch.cuda.reset_peak_memory_stats(device)

    # Warmup
    for _ in range(n_warmup):
        try:
            run_meta_step(B, H, N_q, N_kv, D, window_size, n_chunks, method, device, dtype)
        except Exception as e:
            return None, None, str(e)
        torch.cuda.synchronize(device)

    # Timed runs
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        run_meta_step(B, H, N_q, N_kv, D, window_size, n_chunks, method, device, dtype)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    median_time = sorted(times)[len(times) // 2]
    return median_time, peak_mem, None


CONFIGS = {
    '125M': dict(B=1, H=12, N_q=1024, N_kv=9216, D=64, window_size=8192, n_chunks=8),
    '760M': dict(B=1, H=16, N_q=1024, N_kv=9216, D=96, window_size=8192, n_chunks=8),
    '3B':   dict(B=1, H=32, N_q=1024, N_kv=9216, D=80, window_size=8192, n_chunks=8),
    'small': dict(B=1, H=4, N_q=64, N_kv=256, D=64, window_size=192, n_chunks=4),
}


def verify_forward_agreement(methods, B, H, N_q, N_kv, D, window_size, device, dtype):
    """Verify that all methods produce the same forward output on one chunk."""
    torch.manual_seed(0)
    scale = 1.0 / (D ** 0.5)
    chunk_id = 1

    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

    outputs = {}
    for method in methods:
        if method == 'matmul':
            outputs[method] = reference_attention_matmul(Q, K, V, mask, scale)
        elif method == 'sdpa':
            outputs[method] = reference_attention_sdpa(Q, K, V, mask)
        elif method == 'v7':
            O = v7_attention(Q, K, V, chunk_id, window_size)
            if O is None:
                continue
            outputs[method] = O

    if len(outputs) < 2:
        return True

    baseline_name = list(outputs.keys())[0]
    baseline_O = outputs[baseline_name].float()
    all_ok = True
    for method, O in outputs.items():
        if method == baseline_name:
            continue
        max_diff = (O.float() - baseline_O).abs().max().item()
        # Cross-method comparison: different backends (matmul vs SDPA vs WMMA)
        # each accumulate differently in bf16/fp16, so allow ~1%
        atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        ok = max_diff < atol
        status = "PASS" if ok else "FAIL"
        print(f"  Forward check {baseline_name} vs {method}: "
              f"{status} (max_diff={max_diff:.2e})")
        all_ok &= ok

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='TTT-E2E Meta-Step Benchmark')
    parser.add_argument('--config', type=str, default='small',
                        choices=list(CONFIGS.keys()), help='Model config')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['matmul', 'sdpa'],
                        help='Methods to benchmark')
    parser.add_argument('--n-warmup', type=int, default=3)
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'])
    args = parser.parse_args()

    device = 'cuda'
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]

    cfg = CONFIGS[args.config]
    print(f"\n{'='*60}")
    print(f"TTT-E2E Meta-Step Benchmark: {args.config}")
    print(f"B={cfg['B']}, H={cfg['H']}, N_q={cfg['N_q']}, N_kv={cfg['N_kv']}, D={cfg['D']}")
    print(f"window_size={cfg['window_size']}, n_chunks={cfg['n_chunks']}")
    print(f"dtype={args.dtype}, warmup={args.n_warmup}, trials={args.n_trials}")
    print(f"{'='*60}")

    # Cross-method forward correctness check before timing
    if len(args.methods) > 1:
        print(f"\n  --- Forward agreement check ---")
        verify_forward_agreement(
            args.methods, cfg['B'], cfg['H'], cfg['N_q'], cfg['N_kv'],
            cfg['D'], cfg['window_size'], device, dtype
        )

    results = {}
    for method in args.methods:
        print(f"\n  [{method}] Running...", end='', flush=True)
        t, mem, err = benchmark_method(
            method, **cfg, n_warmup=args.n_warmup, n_trials=args.n_trials,
            device=device, dtype=dtype
        )
        if err:
            print(f" ERROR: {err}")
            continue
        results[method] = (t, mem)
        print(f" {t*1000:.2f} ms  |  {mem:.0f} MB peak")

    if len(results) > 1:
        baseline_method = args.methods[0]
        if baseline_method in results:
            t_base = results[baseline_method][0]
            print(f"\n  Speedup vs {baseline_method}:")
            for method, (t, mem) in results.items():
                if method != baseline_method:
                    print(f"    {method}: {t_base/t:.2f}x")


if __name__ == '__main__':
    main()
