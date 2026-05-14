# Flash v9

IO-aware attention kernel with native double backward, FA2-parity surface.
CUTLASS 3 / CuTe implementation targeting A100 (sm_80) and Blackwell consumer
(sm_120).

## Status

v9 branch — under construction. The v7 / v8 history (sliding-window TTT
support, custom WMMA kernels) is preserved on the `v8` and `master` branches.

Tracked progress: see the task list (`TaskList`). Theory notes live in
`paper/theory/notes.md`.

## Surface

```python
from flash_v9 import flash_v9_attention

O = flash_v9_attention(Q, K, V, is_causal=True)   # standard fwd
loss = O.sum()
dQ, dK, dV = torch.autograd.grad(loss, [Q, K, V], create_graph=True)  # fwd+bwd
hvp = torch.autograd.grad((dQ * u_Q + dK * u_K + dV * u_V).sum(),
                          [Q, K, V])              # fwd+bwd+dbl_bwd
```

- Q, K, V: `[B, H, N, D]`, bf16 or fp16, CUDA.
- Headdims: 64, 128.
- Mask: causal or none.

## Build

```bash
git submodule update --init --recursive    # pulls CUTLASS into third_party/
python setup.py build_ext --inplace
```

## Layout

```
csrc/flash_v9/         CUDA kernels (forward, backward, double_backward)
csrc/attention_ext.cpp pybind bindings
flash_v9/              Python autograd Functions
tests/                 Correctness vs PyTorch reference (and FA2 fwd+bwd)
bench/                 Wall-clock + memory benchmarks
profiling/             ncu drivers
paper/theory/          Algorithm + IO-complexity notes
third_party/cutlass/   CUTLASS submodule (added in commit 2)
```
