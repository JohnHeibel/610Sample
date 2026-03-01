#!/usr/bin/env python3
"""
Run all V7 tests: correctness, memory scaling, e2e meta-step, and benchmarks.

Usage:
  python run_all_tests.py                    # correctness only
  python run_all_tests.py --bench            # + A100-sized benchmarks
  python run_all_tests.py --bench --large    # + 125M-scale benchmark
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
                        help="Include A100-sized benchmarks")
    parser.add_argument("--large", action="store_true",
                        help="Include 125M-scale benchmark (needs ~40GB GPU)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip the build step")
    args = parser.parse_args()

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

    # --- 4. A100-sized benchmarks ---
    if args.bench:
        # Medium-scale attention-only benchmark
        ok = run("E2E Attention-Only Benchmark (med-H4)",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--benchmark-config", "med-H4"])
        results.append(("Bench med-H4", ok))

        ok = run("E2E Attention-Only Benchmark (med-H8)",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--benchmark-config", "med-H8"])
        results.append(("Bench med-H8", ok))

        ok = run("E2E Attention-Only Benchmark (med-H12)",
                 [sys.executable, "e2e/tests/test_ttt_e2e_pytorch.py",
                  "--benchmark-only", "--benchmark-config", "med-H12"])
        results.append(("Bench med-H12", ok))

        # Attention-only bench from test_ttt_e2e_bench
        for cfg in ["small", "med"]:
            ok = run(f"Attention-Only Meta-Step: {cfg}",
                     [sys.executable, "tests/test_ttt_e2e_bench.py",
                      "--config", cfg if cfg != "med" else "small",
                      "--methods", "matmul", "v7",
                      "--dtype", "bfloat16", "--n-trials", "10"])
            results.append((f"AttnBench {cfg}", ok))

    if args.large:
        # 125M scale — needs significant GPU memory
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
                  "--throughput-config", "125M", "--mem-budget", "40000"])
        results.append(("Throughput 125M", ok))

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
