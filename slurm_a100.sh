#!/bin/bash
#SBATCH --job-name=v7-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ============================================================
# V7 Flash Attention — A100 Build + Test + Benchmark
# ============================================================
#
# Usage:
#   sbatch slurm_a100.sh                   # correctness only
#   sbatch slurm_a100.sh --bench           # + medium benchmarks
#   sbatch slurm_a100.sh --bench --large   # + 125M-scale benchmark
#   sbatch slurm_a100.sh --a100            # full A100 suite (125M/760M/3B)
#
# Prerequisites:
#   - conda env with torch + numpy installed
#
# The script will:
#   1. Build the CUDA extension
#   2. Run correctness tests (fwd, bwd, dbl_bwd, memory scaling)
#   3. Run e2e meta-step tests with numerical comparison
#   4. Run benchmarks (if --bench, --large, or --a100)
#
# All output (logs, errors, data) is placed in a timestamped
# directory: results/test-MM-DD-HH-MM/
# ============================================================

set -e

# SLURM copies the script to a spool directory, so $0 won't point back
# to your project. Use an explicit path or SLURM_SUBMIT_DIR instead.
cd "$SLURM_SUBMIT_DIR"

# --- Create timestamped output directory ---
RUN_DIR="results/test-$(date +%m-%d-%H-%M)"
mkdir -p "$RUN_DIR"

# Redirect all stdout and stderr into the run directory
exec > >(tee "$RUN_DIR/output.log") 2> >(tee "$RUN_DIR/error.log" >&2)

echo "============================================================"
echo "  V7 Flash Attention — A100 Test Run"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date:   $(date)"
echo "  Output: $RUN_DIR"
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
echo "GPU Mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# --- Build ---
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
echo "=== Building CUDA extension (arch=${TORCH_CUDA_ARCH_LIST}) ==="
python setup.py build_ext --inplace
echo ""

# --- Run tests ---
# Pass through any arguments (--bench, --large, --a100, etc.)
python run_all_tests.py --skip-build --output-dir "$RUN_DIR" "$@"
