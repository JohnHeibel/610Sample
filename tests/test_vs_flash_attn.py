"""
V7 vs FlashAttention-2 — Kernel Speed & Memory Comparison
==========================================================
Compares V7 CUDA kernels against PyTorch's FlashAttention-2 backend
(torch.nn.functional.scaled_dot_product_attention with FLASH_ATTENTION backend).

NOTE: This is NOT an apples-to-apples functional comparison:
  - FlashAttention-2: causal mask, no create_graph, no double backward
  - V7: sliding-window + causal + validity mask, supports create_graph

This benchmark answers: "How does our custom kernel's raw throughput compare
to the highly-optimized FlashAttention-2 implementation?"

Two comparison modes:
  1. Forward only
  2. Forward + backward (no create_graph — flash doesn't support it)

Run:
  python tests/test_vs_flash_attn.py                    # default
  python tests/test_vs_flash_attn.py --all              # all configs
  python tests/test_vs_flash_attn.py --config 125M      # A100 scale
"""

import argparse
import time
import torch
import torch.nn.functional as F


# =====================================================================
# Configs: (B, H, N_q, N_kv, D, window_size, chunk_id)
# For flash comparison, chunk_id is set so all keys are valid
# =====================================================================

CONFIGS = {
    'tiny':    (1,   1,   64,   256,  64,   256,   2),
    'small':   (1,   4,  128,   512,  64,   512,   4),
    'med':     (1,   8,  256,  1024,  64,  1024,   4),
    'large':   (1,  12,  512,  4096,  64,  4096,   8),
    '125M':    (1,  12, 1024,  9216,  64,  9216,   9),
    'D80':     (1,   8, 1024,  9216,  80,  9216,   9),
    # Square configs (N_q == N_kv) — most comparable to flash causal
    'sq-256':  (1,  12,  256,   256,  64,   256,   1),
    'sq-512':  (1,  12,  512,   512,  64,   512,   1),
    'sq-1024': (1,  12, 1024,  1024,  64,  1024,   1),
    'sq-2048': (1,  12, 2048,  2048,  64,  2048,   1),
    'sq-4096': (1,  12, 4096,  4096,  64,  4096,   1),
}


def check_flash_available():
    """Check if FlashAttention-2 backend is available."""
    try:
        backends = torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION)
        # Quick smoke test
        Q = torch.randn(1, 1, 32, 64, device='cuda', dtype=torch.bfloat16)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(Q, Q, Q, is_causal=True)
        return True
    except Exception:
        return False


# =====================================================================
# Benchmark helpers
# =====================================================================

def _measure(fn, device, n_warmup, n_trials):
    """Warmup, then measure median time and peak memory."""
    for _ in range(n_warmup):
        try:
            fn()
            torch.cuda.empty_cache()
        except Exception as e:
            return None, None, str(e)

    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        torch.cuda.empty_cache()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    median_ms = sorted(times)[len(times) // 2] * 1000
    return median_ms, peak_mb, None


def bench_flash(B, H, N_q, N_kv, D, dtype, device, mode, n_warmup, n_trials):
    """Benchmark FlashAttention-2 via SDPA."""
    def run():
        Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))
        # Flash causal requires N_q == N_kv for is_causal=True
        # For rectangular, we use no mask (flash doesn't support sliding window)
        K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))
        V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            if N_q == N_kv:
                O = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                # Flash supports rectangular without mask (no causal)
                O = F.scaled_dot_product_attention(Q, K, V)

        if mode == 'fwd+bwd':
            O.sum().backward()
        torch.cuda.synchronize()

    return _measure(run, device, n_warmup, n_trials)


def bench_v7(B, H, N_q, N_kv, D, window_size, chunk_id, dtype, device, mode, n_warmup, n_trials):
    """Benchmark V7 kernels."""
    from attention import flash_attention_v7

    def run():
        Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))
        K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))
        V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=(mode != 'fwd'))

        O = flash_attention_v7(Q, K, V, window_size=window_size, chunk_id=chunk_id)

        if mode == 'fwd+bwd':
            O.sum().backward()
        torch.cuda.synchronize()

    return _measure(run, device, n_warmup, n_trials)


# =====================================================================
# Run benchmark
# =====================================================================

def run_config(config_name, dtype, n_warmup, n_trials):
    """Run flash vs V7 for one config."""
    cfg = CONFIGS.get(config_name)
    if cfg is None:
        print(f"  Unknown config: {config_name}")
        return None

    B, H, N_q, N_kv, D, window_size, chunk_id = cfg
    device = 'cuda'
    dtype_name = {torch.bfloat16: 'bf16', torch.float16: 'fp16'}[dtype]
    rect = "rect" if N_q != N_kv else "square"

    print(f"\n{'='*72}")
    print(f"  {config_name}: B={B} H={H} N_q={N_q} N_kv={N_kv} D={D} "
          f"ws={window_size} [{dtype_name}, {rect}]")
    if N_q != N_kv:
        print(f"  NOTE: flash uses no mask (rectangular), V7 uses sliding-window causal")
    else:
        print(f"  flash: causal mask | V7: full-window causal (equivalent)")
    print(f"{'='*72}")

    results = {}

    for mode in ['fwd', 'fwd+bwd']:
        print(f"\n  --- {mode} ---")

        # Flash
        ms, mb, err = bench_flash(B, H, N_q, N_kv, D, dtype, device, mode, n_warmup, n_trials)
        if err:
            print(f"  {'flash':>8s}:  ERROR — {err}")
        else:
            results[('flash', mode)] = (ms, mb)
            print(f"  {'flash':>8s}:  {ms:8.2f} ms  |  {mb:8.0f} MB")

        # V7
        ms, mb, err = bench_v7(B, H, N_q, N_kv, D, window_size, chunk_id,
                                dtype, device, mode, n_warmup, n_trials)
        if err:
            print(f"  {'v7':>8s}:  ERROR — {err}")
        else:
            results[('v7', mode)] = (ms, mb)
            print(f"  {'v7':>8s}:  {ms:8.2f} ms  |  {mb:8.0f} MB")

        # Comparison
        if ('flash', mode) in results and ('v7', mode) in results:
            f_ms, f_mb = results[('flash', mode)]
            v_ms, v_mb = results[('v7', mode)]
            ratio = f_ms / v_ms
            label = "faster" if ratio > 1 else "slower"
            print(f"  {'':>8s}   v7 is {ratio:.2f}x vs flash ({label})"
                  f"  |  mem: {v_mb/f_mb:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='V7 vs FlashAttention-2 — Kernel Speed & Memory')
    parser.add_argument('--config', type=str, default='small',
                        choices=list(CONFIGS.keys()))
    parser.add_argument('--all', action='store_true',
                        help='Run all configs')
    parser.add_argument('--square-only', action='store_true',
                        help='Run only square (N_q==N_kv) configs')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16'])
    parser.add_argument('--n-warmup', type=int, default=3)
    parser.add_argument('--n-trials', type=int, default=10)
    args = parser.parse_args()

    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    if not check_flash_available():
        print("\nFlashAttention-2 backend not available on this GPU/PyTorch version.")
        print("Requires: sm_80+ (A100/H100/etc) and PyTorch 2.2+")
        return

    if args.all:
        configs = list(CONFIGS.keys())
    elif args.square_only:
        configs = [k for k in CONFIGS if k.startswith('sq-')]
    else:
        configs = [args.config]

    all_results = {}
    for cfg_name in configs:
        r = run_config(cfg_name, dtype, args.n_warmup, args.n_trials)
        if r:
            all_results[cfg_name] = r

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*72}")
        print(f"  Summary (vs FlashAttention-2, {args.dtype})")
        print(f"{'='*72}")
        print(f"  {'config':<10s} | {'fwd flash':>10s} {'fwd v7':>10s} {'ratio':>7s}"
              f" | {'f+b flash':>10s} {'f+b v7':>10s} {'ratio':>7s}")
        print(f"  {'-'*10}-+-{'-'*30}-+-{'-'*30}")
        for cfg_name, res in all_results.items():
            parts = []
            for mode in ['fwd', 'fwd+bwd']:
                if ('flash', mode) in res and ('v7', mode) in res:
                    f_ms = res[('flash', mode)][0]
                    v_ms = res[('v7', mode)][0]
                    ratio = f_ms / v_ms
                    parts.append(f"{f_ms:10.2f} {v_ms:10.2f} {ratio:6.2f}x")
                else:
                    parts.append(f"{'—':>10s} {'—':>10s} {'—':>7s}")
            print(f"  {cfg_name:<10s} | {parts[0]} | {parts[1]}")
        print(f"  {'='*72}")
        print(f"\n  ratio > 1 = V7 faster than flash, ratio < 1 = V7 slower")


if __name__ == '__main__':
    main()
