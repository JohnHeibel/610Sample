#!/usr/bin/env python3
"""
Run all V7 tests: correctness, memory scaling, e2e meta-step, and benchmarks.

Usage:
  python run_all_tests.py                    # correctness only
  python run_all_tests.py --bench            # + medium benchmarks
  python run_all_tests.py --bench --large    # + 125M-scale benchmark
  python run_all_tests.py --a100             # full A100 benchmark suite
"""

import sys
import os
import subprocess
import argparse
import time

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")


def run(label, cmd):
    """Run a command and return success/failure."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}\n")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    status = "PASSED" if ok else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run all V7 tests")
    parser.add_argument("--bench", action="store_true",
                        help="Include medium benchmarks")
    parser.add_argument("--large", action="store_true",
                        help="Include 125M-scale benchmark (needs ~40GB GPU)")
    parser.add_argument("--a100", action="store_true",
                        help="Full A100 80GB benchmark suite (125M/760M/3B)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip the build step")
    args = parser.parse_args()

    # --a100 implies --bench and --large
    if args.a100:
        args.bench = True
        args.large = True

    results = []

    # --- Build ---
    if not args.skip_build:
        ok = run("Build CUDA extension",
                 [sys.executable, "setup.py", "build_ext", "--inplace"])
        results.append(("Build", ok))
        if not ok:
            print("\nBuild failed — cannot run tests.")
            sys.exit(1)

    # --- 1. Kernel correctness (forward, backward, double backward, memory) ---
    ok = run("V7 Kernel Correctness",
             [sys.executable, "tests/test_v7_correctness.py"])
    results.append(("V7 Correctness", ok))

    # --- 2. E2E PyTorch tests (component + meta-step + V7 integration) ---
    ok = run("E2E PyTorch Tests (skip benchmarks)",
             [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py", "--skip-benchmarks"])
    results.append(("E2E PyTorch", ok))

    # --- 3. Benchmark: small config with cross-method check ---
    ok = run("Benchmark: small (matmul vs v7, bf16)",
             [sys.executable, "tests/test_ttt_e2e_bench.py",
              "--config", "small", "--methods", "matmul", "v7",
              "--dtype", "bfloat16", "--n-trials", "5", "--n-warmup", "2"])
    results.append(("Bench small", ok))

    # --- 4. Medium benchmarks ---
    if args.bench:
        for cfg in ["med-H4", "med-H8", "med-H12"]:
            ok = run(f"E2E Full-Model Benchmark ({cfg})",
                     [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                      "--benchmark-only", "--benchmark-config", cfg])
            results.append((f"Bench {cfg}", ok))

        ok = run("E2E Full-Model Benchmark (large-kv)",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--benchmark-config", "large-kv"])
        results.append(("Bench large-kv", ok))

        # V7 vs FlashAttention-2 comparison (requires sm_80+ GPU)
        ok = run("V7 vs FlashAttention-2 (square configs)",
                 [sys.executable, "tests/test_vs_flash_attn.py",
                  "--square-only", "--n-trials", "10"])
        results.append(("V7 vs Flash (sq)", ok))

        ok = run("V7 vs FlashAttention-2 (rectangular configs)",
                 [sys.executable, "tests/test_vs_flash_attn.py",
                  "--config", "large", "--n-trials", "10"])
        results.append(("V7 vs Flash (rect)", ok))

    # --- 5. Large-scale benchmarks (125M) ---
    if args.large:
        ok = run("V7 vs FlashAttention-2 (125M scale)",
                 [sys.executable, "tests/test_vs_flash_attn.py",
                  "--config", "125M", "--n-trials", "10"])
        results.append(("V7 vs Flash 125M", ok))

        ok = run("E2E 125M Full Meta-Step Benchmark",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--benchmark-config", "125M"])
        results.append(("Bench 125M", ok))

        ok = run("Attention-Only 125M Meta-Step",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--attn-only", "--attn-config", "125M"])
        results.append(("AttnBench 125M", ok))

        ok = run("Max-Batch Throughput 125M",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--throughput",
                  "--throughput-config", "125M", "--mem-budget", "75000"])
        results.append(("Throughput 125M", ok))

    # --- 6. A100-only benchmarks (760M, 3B) ---
    if args.a100:
        # 760M attention-only (H=16, D=96) — ref will likely OOM
        ok = run("Attention-Only 760M Meta-Step",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--attn-only", "--attn-config", "760M"])
        results.append(("AttnBench 760M", ok))

        ok = run("Max-Batch Throughput 760M",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--throughput",
                  "--throughput-config", "760M", "--mem-budget", "75000"])
        results.append(("Throughput 760M", ok))

        # 3B attention-only (H=32, D=80) — ref will likely OOM
        ok = run("Attention-Only 3B Meta-Step",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--attn-only", "--attn-config", "3B"])
        results.append(("AttnBench 3B", ok))

        ok = run("Max-Batch Throughput 3B",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--throughput",
                  "--throughput-config", "3B", "--mem-budget", "75000"])
        results.append(("Throughput 3B", ok))

        # V7 vs Flash at larger scales
        ok = run("V7 vs FlashAttention-2 (D=80)",
                 [sys.executable, "tests/test_vs_flash_attn.py",
                  "--config", "D80", "--n-trials", "10"])
        results.append(("V7 vs Flash D80", ok))

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        all_ok &= ok

    print(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
