#!/bin/bash
# Build the V7 CUDA extension on Linux
set -e

echo "=== Building attention_cuda extension ==="
python setup.py build_ext --inplace

echo ""
echo "=== Build complete ==="
ls -la attention_cuda*.so 2>/dev/null || echo "WARNING: .so not found in current directory"
