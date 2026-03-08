#!/usr/bin/env python3
"""
Run all V7 tests: correctness, wall clock, memory, flash comparison, and scaling.

Usage:
  python run_all_tests.py                    # correctness only
  python run_all_tests.py --bench            # + wallclock, memory, flash comparison
  python run_all_tests.py --a100             # full A100 benchmark suite (+ scaling)
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
                        help="Include wallclock, memory, and flash benchmarks")
    parser.add_argument("--a100", action="store_true",
                        help="Full A100 80GB benchmark suite (bench + scaling)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip the build step")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write result files into")
    args = parser.parse_args()

    # --a100 implies --bench
    if args.a100:
        args.bench = True

    output_dir = args.output_dir or "test-output"
    output_args = ["--output-dir", output_dir]

    results = []

    # --- Build ---
    if not args.skip_build:
        ok = run("Build CUDA extension",
                 [sys.executable, "setup.py", "build_ext", "--inplace"])
        results.append(("Build", ok))
        if not ok:
            print("\nBuild failed — cannot run tests.")
            sys.exit(1)

    # --- 1. Correctness (always runs) ---
    ok = run("V7 Kernel Correctness",
             [sys.executable, "tests/test_correctness.py"] + output_args)
    results.append(("Correctness", ok))

    # --- 2. Wall Clock Benchmarks (--bench or --a100) ---
    if args.bench:
        ok = run("Wall Clock Benchmarks",
                 [sys.executable, "tests/test_wallclock.py"] + output_args)
        results.append(("Wall Clock", ok))

    # --- 3. Memory Benchmarks (--bench or --a100) ---
    if args.bench:
        ok = run("Memory Benchmarks",
                 [sys.executable, "tests/test_memory.py"] + output_args)
        results.append(("Memory", ok))

    # --- 4. Flash Comparison (--bench or --a100) ---
    if args.bench:
        ok = run("V7 vs FlashAttention-2",
                 [sys.executable, "tests/test_flash_comparison.py"] + output_args)
        results.append(("Flash Comparison", ok))

    # --- 5. Scaling Limits (--a100 only) ---
    if args.a100:
        ok = run("Scaling Limits",
                 [sys.executable, "tests/test_scaling.py"] + output_args)
        results.append(("Scaling", ok))

    # --- Summary ---
    summary_lines = []
    summary_lines.append(f"\n{'='*70}")
    summary_lines.append(f"  SUMMARY")
    summary_lines.append(f"{'='*70}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        summary_lines.append(f"  [{status}] {name}")
        all_ok &= ok

    summary_lines.append(f"\n  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    summary_lines.append(f"{'='*70}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Write summary to output directory
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
