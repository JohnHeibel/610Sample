#!/bin/bash
#SBATCH --job-name=v7-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=v7_test_%j.log

# ============================================================
# V7 Flash Attention — A100 Build + Test + Benchmark
# ============================================================
#
# Usage:
#   sbatch slurm_a100.sh                   # correctness only
#   sbatch slurm_a100.sh --bench           # + medium benchmarks
#   sbatch slurm_a100.sh --bench --large   # + 125M-scale benchmark
#
# Prerequisites:
#   - Python 3.12+ with PyTorch 2.x + CUDA in your environment
#   - Either: module load cuda pytorch
#     Or:     conda/venv with torch installed
#
# The script will:
#   1. Build the CUDA extension
#   2. Run correctness tests (fwd, bwd, dbl_bwd, memory scaling)
#   3. Run e2e meta-step tests with numerical comparison
#   4. Run benchmarks (if --bench or --large)
# ============================================================

set -e

echo "============================================================"
echo "  V7 Flash Attention — A100 Test Run"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date:   $(date)"
echo "============================================================"

# --- Environment setup ---
# Uncomment/modify for your cluster:
# module load cuda/12.1 pytorch/2.x
# source /path/to/your/venv/bin/activate
# conda activate your_env

echo ""
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""

cd "$(dirname "$0")"

# --- Build ---
echo "=== Building CUDA extension ==="
python setup.py build_ext --inplace
echo ""

# --- Run tests ---
# Pass through any arguments (--bench, --large, etc.)
python run_all_tests.py --skip-build "$@"
