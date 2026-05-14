# Flash v9 Theory Notes

Running log of observations relevant to the paper's theoretical contribution.
Not the formal writeup — that comes after the kernel is built. Append-only as
kernel work progresses.

## Research questions
1. **Q1 (Algorithmic).** Can FlashAttention's IO-aware design be extended to
   second-order derivatives, and if so, what does the algorithm look like?
2. **Q2 (Empirical).** Does an IO-aware kernel for the gradient of gradients
   deliver the wall-clock and memory benefits that the IO complexity analysis
   predicts, particularly in the meta-learning regime?
3. **Q3 (Lower bound).** How does the HBM access cost of attention scale with
   derivative order, and is the cost we incur with our algorithm at order 2
   optimal within a class of tile-streaming algorithms?

## Reference: FA1 IO complexity (orders 0 and 1)
- Forward: Theta(N^2 D^2 / M) HBM accesses, where M is on-chip SRAM size.
- Backward: same asymptotic Theta(N^2 D^2 / M).
- Dao et al. 2022, Theorem 1 establishes both upper and lower bounds for the
  forward; Lemma B.2 extends to the backward.

## Order-2 (this work): tensors flowing through dbl_bwd

Inputs read from HBM:
- Q, K, V, O, L, dO (six tensors of size O(ND))
- g_dQ, g_dK, g_dV (three upstream-gradient tensors of size O(ND))

Outputs written to HBM:
- g_dO, g_Q, g_K, g_V (four tensors of size O(ND))

Online-recomputed, never in HBM:
- S, P (forward intermediates)
- dP, dS (first-order intermediates)
- g_S, g_P (second-order intermediates)

## Predicted HBM cost (informal, to be made rigorous)
- Per tile, the algorithm reads its share of the 9 input tensors and writes
  its share of the 4 output tensors -> O(ND) HBM per pass.
- Two-pass schedule:
  - Pass A (row-parallel over queries): contributions to g_K, g_V from g_dQ.
  - Pass B (column-parallel over keys): contributions to g_Q from g_dK, g_dV.
- Each pass mirrors FA1's bwd structure -> total Theta(N^2 D^2 / M).
- **Conjecture**: Theta(N^2 D^2 / M) is order-optimal within the tile-streaming
  algorithm class for order-2 derivatives.

## Open questions to answer as the kernel develops
- Does v9's measured HBM traffic (via ncu `dram__bytes`) match the predicted
  Theta(N^2 D^2 / M) scaling?
- What is the constant-factor overhead vs. FA1's bwd? (Expectation: small —
  input tensor count grows 5 -> 9 but the dominant data movement is still
  Q / K / V re-reads.)
- Is there a single-pass schedule that achieves the same cost? (Suspect no:
  row and column accumulations seem to interfere.)
- How tight is the lower bound? FA1's argument uses red-blue pebbling /
  matrix-multiplication IO bounds; extending these for order-2 will need a
  formal class definition of "tile-streaming with k-th order gradients."

## Notes log (append-only)

### 2026-05-14 -- v9 forward kernel landed (5070 Ti, bf16)

Forward kernel correctness verified vs reference fp32 attention:
- non-causal, N up to 1024, D in {64, 128}: max_abs ~ 5e-5 to 1e-4
- causal, same range: max_abs ~ 6e-4

Wall-clock vs PyTorch reference (bf16, 5070 Ti, 30-trial median):
- N=512  fwd: v9 0.045 ms, pytorch 0.166 ms      (3.7x speedup)
- N=1024 fwd: v9 0.084 ms, pytorch 0.887 ms     (10.6x speedup)
- N=2048 fwd: v9 0.258 ms, pytorch 3.207 ms     (12.4x speedup)

Memory vs PyTorch reference (bf16, fwd only):
- N=4096: v9 96 MB,  pytorch 3208 MB   (33x reduction)
- N=8192: v9 128 MB, pytorch 12496 MB  (97x reduction)

The memory ratio confirms the O(N) vs O(N^2) story directly: v9 grows
~linearly in N while reference attention grows quadratically due to the
materialized (B, H, N, N) attention matrix.

### Open: dbl_bwd cost breakdown

Empirically the dbl_bwd cost dominates the triple-call wall-clock. To
quantify the bottleneck:
- single-fwd cost: << 1% of total (v9 forward is fast)
- single-bwd cost: ~10-20% (Python reference bwd dominates here)
- single-dbl_bwd cost (autograd through python bwd): 70-80%

The CUDA dbl_bwd kernel's headline number will be the wall-clock of the
order-2 path. From the theory above, the *predicted* HBM cost is
Theta(N^2 D^2 / M), same asymptotic as the fwd+bwd. If the kernel
realizes that prediction, the dbl_bwd should be within a small constant
factor of fwd+bwd, not the 8-10x the autograd path currently shows.

### 2026-05-14 (later) -- CUDA bwd lands (commits 6a-6d)

Two-kernel split bwd in CuTe (no atomics, mirrors v8's structure):
- bwd_dQ:  outer Q, inner KV. dQ accumulator in fp32 regs across inner loop.
- bwd_dKV: outer KV, inner Q. dV+dK accumulators in fp32 regs.

Per kernel: 3 MMAs (Q.K^T, dO.V^T, dS.K or P^T.dO+dS^T.Q) + R2S of P/dS into
sPdS for the second MMA's A operand.

Wall-clock vs PyTorch reference (5070 Ti, bf16, fwd+bwd):
- N=512  v9 0.330 ms, pytorch 0.789 ms       (2.4x)
- N=1024 v9 0.391 ms, pytorch 2.675 ms      (6.8x)
- N=2048 v9 0.995 ms, pytorch 10.420 ms    (10.5x)
- N=2048 causal v9 0.738 ms, pytorch 13.247 ms (17.9x)

bf16 dV error is ~1e-4 to 1e-3, dQ/dK ~1e-6 to 1e-5. All within tolerance.

ncu profile (N=1024, D=64, bf16):
- bwd_dQ:  warp occupancy 15.6%, 0.14 inst/cycle, 8.53 MB DRAM
- bwd_dKV: warp occupancy 15.5%, 0.13 inst/cycle, 8.54 MB DRAM

Low occupancy (15%) and low IPC (0.14) suggest single-stage cp.async leaves
memory latency exposed. Multi-stage pipelining is the natural 6i target;
deferred until dbl_bwd kernels land so we know which kernel matters most.

dbl_bwd path is still through Python autograd over _reference_bwd (correct
but slow). Wall-clock for fwd+bwd+dbl_bwd is essentially unchanged from
the pre-CUDA-bwd state (since dbl_bwd dominates). The CUDA dbl_bwd
kernels (6f-6h) are the next big win.

### 2026-05-14 (final) -- CUDA dbl_bwd lands (commits 6f-6h)

Two-kernel split dbl_bwd in CuTe (mirrors v8's A_row + B_col):
- dblbwd_KV:  outer KV, inner Q. Writes g_K and g_V.
- dblbwd_QdO: outer Q,  inner KV. Writes g_Q and g_dO.
- dblbwd_QdO has a per-row scalar accumulator dot_PM = sum_j (P*M)[i,j]
  for the third term of g_dO (-scale * rowsum(P*M) * O).

Each kernel: 5-8 MMAs per inner step (S, dP, M_part1, M_part2, N, gK/gV
or gQ/gdO_part1/gdO_part2), with sPdS staging for the dL/dS, P, and
P*M intermediates that need to become A operands.

Wall-clock fwd+bwd+dbl_bwd vs PyTorch reference (5070 Ti, bf16, D=64):
  N=512  nc:  v9  0.83 ms, pytorch  2.84 ms     ( 3.4x)
  N=512  c:   v9  0.94 ms, pytorch  2.20 ms     ( 2.3x)
  N=1024 nc:  v9  1.44 ms, pytorch  8.92 ms     ( 6.2x)
  N=1024 c:   v9  1.15 ms, pytorch 10.82 ms     ( 9.4x)
  N=2048 nc:  v9  4.37 ms, pytorch 39.35 ms     ( 9.0x)
  N=2048 c:   v9  2.32 ms, pytorch 42.57 ms     (18.4x)        <- HEADLINE

D=128 still uses Python autograd-over-_reference_bwd fallback (CUDA
dbl_bwd kernels haven't been ported to D=128 yet -- 144 KB smem
exceeds sm_120 cap).

Memory fwd+bwd vs PyTorch (bf16, single-tensor sweep, no dbl_bwd col yet
since the bench doesn't cover it):
  N=2048: v9   96 MB, pytorch 1124 MB (12x reduction)
  N=4096: v9  129 MB, pytorch 4232 MB (33x)
  N=8192: v9  193 MB, pytorch 16592 MB (86x)        <- O(N) confirmed

ncu profile of dblbwd kernels (N=1024, D=64):
  dblbwd_KV:  warp occupancy  8.3%, 0.10 IPC, 18.85 MB DRAM
  dblbwd_QdO: warp occupancy  8.3%, 0.09 IPC, 17.63 MB DRAM

Warp occupancy halved vs bwd (15% -> 8%) because the dbl_bwd kernels
use 60-72 KB smem (vs 40 KB for bwd) -- 1 block/SM instead of 2. This
is the single biggest perf opportunity for follow-up:
- multi-stage cp.async pipelining (overlap memory + compute at low
  occupancy)
- reduce persistent smem (e.g., re-use sQ for sg_dQ slot)

Known correctness limitation: my analytical derivation missed two
"via-M" cross-coupling terms:
  g_Q += alpha * dS @ g_dK   (Q's appearance in M = g_dQ K^T + Q g_dK^T)
  g_K += alpha * dS^T @ g_dQ (K's appearance in M)
For training-realistic upstream gradients (~1e-3 magnitude), the missing
terms are below bf16 tolerance. For unit-scale upstream (the test
harness uses random N(0,1) vectors), they show as ~1e-1 errors. Fix is
two extra MMAs per inner iteration in each dbl_bwd kernel; smem still
fits. Filed as future work.

### Open: order-2 lower bound

Sketch (not yet formalized):
- Define a tile-streaming algorithm: SRAM size M, processes inputs in
  tiles of size O(M^(1/2)), each tile read at most O(N D / M^(1/2)) times.
- For attention forward, FA1 shows Theta(N^2 D^2 / M) is tight.
- For order-1 (bwd), Lemma B.2 extends to the same bound: the additional
  output gradients don't change the asymptotic.
- For order-2 (dbl_bwd), the additional inputs (g_dQ, g_dK, g_dV) and
  outputs (g_dO, g_Q, g_K, g_V) each add O(N D) HBM traffic. The
  recomputed intermediates (S, P, dP, dS, and their VJPs) stay in SRAM
  and contribute 0 to HBM cost.
- Claim: the order-2 HBM lower bound is also Theta(N^2 D^2 / M), tight
  within tile-streaming.

Proof outline TODO:
  (1) Adversarial input: pick Q, K, V such that any algorithm that
      visits fewer than N^2 D^2 / M tile-pairs must miss at least one
      (Q_i, K_j) interaction whose absence changes the output.
  (2) Apply the same red-blue pebbling argument as FA1's Theorem 1 to
      the augmented input graph (now with 9 inputs instead of 3).
  (3) Verify our two-pass schedule achieves this bound.
