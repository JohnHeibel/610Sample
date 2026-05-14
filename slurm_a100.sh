#!/bin/bash
#SBATCH --job-name=v9-test
#SBATCH --partition=lowd
#SBATCH --gpus=1
#SBATCH --constraint=a100,gpu-80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# ============================================================
# Flash v9 — A100 Build + Correctness + Benchmark
# ============================================================
#
# Usage:
#   sbatch slurm_a100.sh                     correctness only
#   sbatch slurm_a100.sh --bench             correctness + wall-clock + memory
#   sbatch slurm_a100.sh --bench --profile   correctness + bench + ncu profile
#
# Output goes to results/v9-MM-DD-HH-MM/ relative to the submission dir.
# ============================================================

set -e

cd "$SLURM_SUBMIT_DIR"

RUN_DIR="results/v9-$(date +%m-%d-%H-%M)"
mkdir -p "$RUN_DIR"

exec > >(tee "$RUN_DIR/output.log") 2> >(tee "$RUN_DIR/error.log" >&2)

echo "============================================================"
echo "  Flash v9 - A100 Test Run"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Date:   $(date)"
echo "  Output: $RUN_DIR"
echo "============================================================"

module purge
module load gcc/13.1.0 miniconda-t2/20230523 python3/3.11.4 cuda/12.4.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mytorch

echo ""
echo "Python:  $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo "GPU Mem: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# --- Init submodules (CUTLASS + flash-attention reference) ---
echo "=== Initializing submodules ==="
git submodule update --init --recursive
echo ""

# --- Build v9 extension for sm_80 ---
export TORCH_CUDA_ARCH_LIST="8.0"
echo "=== Building flash_v9 extension (arch=${TORCH_CUDA_ARCH_LIST}) ==="
python setup.py build_ext --inplace
echo ""

# --- Try to install flash-attn for the FA2 baseline ---
if ! python -c "import flash_attn" 2>/dev/null; then
    echo "=== Installing flash-attn ==="
    # Prebuilt wheels available for sm_80; should be quick on A100.
    pip install --no-build-isolation flash-attn || \
        echo "WARN: flash-attn install failed; FA2 cells will be N/A"
    echo ""
fi

# --- Correctness vs PyTorch reference (fwd / bwd / dbl_bwd) ---
echo "============================================================"
echo "  Correctness vs PyTorch reference"
echo "============================================================"
python tests/test_correctness.py 2>&1 | tee "$RUN_DIR/correctness.log"
echo ""

# --- Correctness vs FA2 (fwd + bwd only) ---
echo "============================================================"
echo "  Correctness vs FlashAttention-2"
echo "============================================================"
python tests/test_flash_parity.py 2>&1 | tee "$RUN_DIR/flash_parity.log" || \
    echo "WARN: FA2 parity tests failed/skipped"
echo ""

# --- Wall-clock benchmark ---
if [[ " $* " == *" --bench "* ]] || [[ " $* " == *" --profile "* ]]; then
    echo "============================================================"
    echo "  Wall-clock benchmark"
    echo "============================================================"
    python bench/bench_wallclock.py --warmup 5 --trials 30 \
        --out "$RUN_DIR/bench_wallclock.json" \
        2>&1 | tee "$RUN_DIR/bench_wallclock.log"
    echo ""

    echo "============================================================"
    echo "  Memory benchmark"
    echo "============================================================"
    python bench/bench_memory.py \
        --out "$RUN_DIR/bench_memory.json" \
        2>&1 | tee "$RUN_DIR/bench_memory.log"
    echo ""
fi

# --- ncu profile (optional) ---
if [[ " $* " == *" --profile "* ]]; then
    echo "============================================================"
    echo "  ncu profile: v9 forward (med64 config)"
    echo "============================================================"
    # Brief, focused profile. Full profile would multiply this 20x.
    NCU=$(command -v ncu || echo "/packages/cuda/12.4.1/bin/ncu")
    if [ -x "$NCU" ]; then
        $NCU --set basic \
             --kernel-name regex:flash_v9 \
             --launch-skip 5 --launch-count 3 \
             -o "$RUN_DIR/ncu_v9_fwd" \
             python -c "
import torch
from flash_v9 import flash_v9_attention
Q = torch.randn(1, 16, 1024, 64, device='cuda', dtype=torch.bfloat16) * 0.1
K = torch.randn(1, 16, 1024, 64, device='cuda', dtype=torch.bfloat16) * 0.1
V = torch.randn(1, 16, 1024, 64, device='cuda', dtype=torch.bfloat16) * 0.1
for _ in range(10):
    O = flash_v9_attention(Q, K, V, is_causal=False)
torch.cuda.synchronize()
" 2>&1 | tee "$RUN_DIR/ncu.log"
    else
        echo "ncu not found on PATH or at default cuda/12.4.1 location; skipping"
    fi
fi

echo ""
echo "============================================================"
echo "  All done. Results in: $RUN_DIR"
echo "============================================================"
