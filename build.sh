#!/bin/bash
# Build the V7 CUDA extension on Linux
#
# Set TORCH_CUDA_ARCH_LIST if building on a node without a GPU (e.g. login node).
# Defaults to 8.0 (A100). For H100 use "9.0", for both use "8.0;9.0".
set -e

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "=== Building attention_cuda extension ==="
echo "  Target arch: ${TORCH_CUDA_ARCH_LIST}"
python setup.py build_ext --inplace

echo ""
echo "=== Build complete ==="
ls -la attention_cuda*.so 2>/dev/null || echo "WARNING: .so not found in current directory"
