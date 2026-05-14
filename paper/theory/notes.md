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
