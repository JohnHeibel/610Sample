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
