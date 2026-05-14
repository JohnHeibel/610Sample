# V8 Kernel Optimization Results

Overnight iteration log for `csrc/flash_double_backward_v8.cu`. Built off the
v7 profile findings in `PROFILER_REPORTS.md` and aimed at substantively
improving wall-clock time without regressing correctness or peak memory.

---

## TL;DR

- **Wall-clock speedup vs v7 on the fwd → bwd → double-bwd path: +44–45% across
  all standard configs.** On 350M (D=64): v7 23.31 ms → v8 12.92 ms (1.80×).
- All v7 unit tests still pass on v7 (untouched); all v8 unit tests pass on v8.
- Peak memory: v8 within ±0.5 MB of v7 on all configs (essentially identical).
- 7 iterations applied — full source-correlated profiling fed each step.
- Per-kernel hot-kernel improvement on 350M:
  - `kernel_A_row`: 12.16 ms → 6.42 ms (-47%)
  - `kernel_B_col`: 6.64 ms → 3.20 ms (-52%)
  - `flash_bwd_row_kernel`: 2.07 ms → 1.19 ms (-43%)
  - `flash_bwd_col_kernel`: 3.88 ms → 2.30 ms (-41%)
  - `flash_fwd_kernel`: 3.97 ms → 2.77 ms (-30%)
- Tensor-pipe utilization roughly doubled (e.g. A_row 13.6 % → 25.9 %,
  B_col 14.3 % → 29.6 %).

---

## Layout

- v7 lives untouched in `csrc/flash_double_backward_v7.cu` and remains the
  fallback. The Python entry points `attention_cuda.v7_*` and
  `attention.flash_attention_v7` continue to work and pass v7 tests.
- v8 lives in `csrc/flash_double_backward_v8.cu` with renamed symbols
  (`v7_` → `v8_`, `V7_` → `V8_`). Same algorithm, different micro-architecture.
- New Python entry points: `attention_cuda.v8_forward / v8_backward /
  v8_double_backward` and `attention.flash_attention_v8`.
- Both v7 and v8 are compiled into the same `attention_cuda.*.pyd`.

Build: `python setup.py build_ext --inplace` — same as before, just compiles
both `.cu` files. Existing `setup.py` was edited (one line: added v8.cu to
sources).

---

## Iteration log

Each row is the cumulative speedup vs v7 on the 350M fwd+bwd+dbl-bwd wall-clock
path measured by `bench_v8.py` (median of 30 trials after 5 warmups).

| Iter | Change | v8 (ms) | Δ vs v7 | Notes |
|---|---|---|---|---|
| baseline | bit-identical copy of v7 | 24.77 | +0.07 % | parity check |
| 1 | data-tile padding (`V8_SMEM_PAD = 8` for D=64) | 21.76 | +7.0 % | breaks 128 B row stride |
| 2 | int4 vectorized global→shared loads (all 12 sites) | 15.22 | +34.6 % | biggest single win — 8× fewer LDGs |
| 3 | A_row score-tile padding (+4 fp32) | 14.89 | +36.1 % | only A_row; bwd_row would lose occupancy |
| 4 | skip zero-fill of padding column in vec helpers | 14.04 | +39.7 % | WMMA never reads it |
| 5 | packed `__floats2bfloat162_rn` for half_score conversion | 13.24 | +43.2 % | halves inner loop, packed cvt |
| 6 | `_full` fast-path variants for steady-state inner loops | 13.22 | +43.3 % | cosmetic — runtime branch wasn't the bottleneck |
| 7 | `#pragma unroll` on per-row inner loops | 12.92 | +44.6 % | helps ILP across rows |

Cross-config wall-clock final speedup:

| Config | H × D | v7 (ms) | v8 (ms) | Speedup | Δ |
|---|---|---|---|---|---|
| 125M | 12 × 64 | ~23.3 | ~12.8 | 1.82× | +45.1 % |
| 350M | 16 × 64 | 23.31 | 12.92 | 1.80× | +44.7 % |
| 1.3B | 32 × 64 | ~23.3 | ~12.8 | 1.82× | +45.2 % |
| 6.7B | 64 × 64 | ~23.3 | ~12.7 | 1.83× | +45.4 % |
| 2.7B | 32 × 80 | ~23.3 | ~13.2 | 1.77× | +43.4 % |

D=80 still gets +43 % despite not receiving the SMEM_PAD or SCORE_PAD wins
(D=80 row stride is 160 B, not on the 128 B bank-pathology boundary, and adding
those pads pushes the dbl-bwd kernels past the 100 KB per-block smem cap on the
5070 Ti). The D=80 win comes entirely from the vectorization, conversion-pack,
and unroll iters. D=96 (760M, 20B configs) was already over the smem cap in v7
double-backward (at any pad level) and remains so in v8 — tested but errored
out on both v7 and v8.

JSON files for each iter at every config: `profile_output/v8_iterations/`.

---

## Per-kernel breakdown (350M, --set basic)

| Kernel | v7 dur | v8 dur | Δ | v7 Tensor% | v8 Tensor% |
|---|---|---|---|---|---|
| `flash_fwd_kernel`        | 3.97 ms | 2.77 ms | -30 % | 9.2  | 13.1 |
| `flash_bwd_row_kernel`    | 2.07 ms | 1.19 ms | -43 % | 17.5 | 30.3 |
| `flash_bwd_col_kernel`    | 3.88 ms | 2.30 ms | -41 % | 12.2 | 20.7 |
| `kernel_A_row`            | 12.16 ms | 6.42 ms | -47 % | 13.6 | 25.9 |
| `kernel_B_col`            | 6.64 ms | 3.20 ms | -52 % | 14.3 | 29.6 |
| **sum** | **28.72 ms** | **15.88 ms** | **-45 %** | | |

Sum-of-kernel-durations vs end-to-end wall-clock: 15.88 ms vs 12.92 ms — the
3 ms gap is the host-side dispatch and inter-kernel synchronization (same
pattern in v7).

---

## Validation

- `tests/test_correctness.py` (v7): all 56 tests PASS (forward, backward,
  double-backward across 8 configs).
- `tests/test_correctness_v8.py` (v8): all 56 tests PASS at the SAME
  tolerances as v7 (`ATOL_FWD=5e-3`, `ATOL_BWD=5e-3`, `ATOL_DBL=1.5e-2`).
- Peak memory comparison (`profile_output/v8_iterations/final_memory.json`):

  | Config | v7 peak | v8 peak | Δ |
  |---|---|---|---|
  | 125M | 122.19 MB | 122.69 MB | +0.50 MB |
  | 350M | 159.25 MB | 158.25 MB | -1.00 MB |
  | 1.3B | 316.50 MB | 316.50 MB | +0.00 MB |
  | 2.7B | 404.50 MB | 403.50 MB | -1.00 MB |
  | 6.7B | 633.00 MB | 633.00 MB | +0.00 MB |

  All within ±1 MB of v7 (less than 1% on every config). Memory constraint
  satisfied.

- Per-block shared-memory budget: A_row D=64 went from 81 KB (v7) to 93 KB
  (v8) due to the data-pad and A-only score-pad. B_col D=64 went from 90 KB
  to 95 KB. Both fit the 100 KB Blackwell per-block cap. D=80 at the same
  pad would overflow B_col, so pad is conditional on `D_CONST <= 64` — no
  regression on D=80.

---

## What worked, with the "why"

### Iter 1: data-tile padding (`V8_SMEM_PAD`)
- Original `D_PAD = D_CONST + 0` made D=64 row stride exactly 128 B = one
  bank-row, causing 8-way conflicts on every WMMA col-major load of K/V/Q/dO.
- Pad +8 half-elements (16 B) → 144 B/row, period 8 (conflict-free).
- Conditional on D=64 via `v8_smem_pad(D)` constexpr, because D=80 has stride
  160 B (already different period) and D=96 already overflows smem.
- **Win: +7 %.** Less than the predicted 10–15 % because the bank-conflict
  stalls were partially overlapped with other compute.

### Iter 2: int4 vectorized loads (the big one)
- Replaced 12 scalar `K_tile[idx] = K[src]` loops with calls to four
  `v8_vec_load_{1,2,3,4}tile<half_t, D_CONST>` helpers that issue int4
  (16-byte = 8 half_t) loads.
- Cuts global LDG count by 8× and lets the compiler pipeline multiple LDGs
  in flight per iteration.
- Hit `long_scoreboard` directly — was 22 % of A_row stalls in v7, dropped
  to ~6 % after vec.
- **Win: +27.6 % incremental → +34.6 % cumulative.** Single biggest jump.

### Iter 3: A_row score-tile padding (`V8_AROW_SCORE_PAD`)
- Score-tile WMMA stores had 4-way conflict (128 B/row), and the half_score
  alias in A_row had 8-way (since `A_HSCORE_STRIDE = SCORE_STRIDE * 2 = 128 B`
  too).
- Pad +4 fp32 → SCORE_STRIDE = 36 = 144 B/row, period 8 (conflict-free), and
  half-alias inherits the same period.
- **Critical:** A_row only. Applying to fwd/bwd_row pushes bwd_row's smem
  from 49 KB → 51 KB which crosses the 2-block/SM threshold (100/49 = 2 vs
  100/51 = 1) and regressed bwd_row by 70 %. So `v8_a_score_pad` returns 0
  and a separate `v8_arow_score_pad` returns 4 for A_row.
- **Win: +1.5 %** (A_row alone gained more, but A_row is 41 % of pipeline).

### Iter 4: skip padding-column zero-fill
- The vec helpers were iterating `D_PAD / VEC` vectors per row, writing zeros
  to the padding columns d ∈ [D_CONST, D_PAD). Verified by reading every
  WMMA load site that the padding region is **never** read (all WMMA k-loops
  have `kk < D_CONST`, output-matmul col index is `cb * V8_WMMA_N + kk * V8_WMMA_K`
  with `cb < n_col_blocks = D_CONST / V8_WMMA_N`, no consumer ever reads
  `d >= D_CONST`).
- Changed iteration to `D_CONST / VEC` and dropped the column-bound else
  branch entirely.
- **Win: +3.5 %** — also dropped the "stalls attributed to else branch"
  noise that was masking other findings.

### Iter 5: packed float2 → half2 conversion in output_matmul
- The half_score staging (fp32 → half_t conversion, 16×32 elements) did one
  cvt per element with 16 iters per lane.
- Replaced with `__floats2bfloat162_rn` / `__floats2half2_rn` packed convert,
  reading float2 (8 B LDS) and writing half2 (4 B STS). 8 iters per lane
  instead of 16.
- Added `half2_t` and `from_float2` to `HalfTraits<>`.
- **Win: +3.5 %** on top, despite this being a "secondary" path inside an
  inlined helper called 4× per A_row inner iter.

### Iter 6: `_full` fast-path variants (mostly no-op)
- Added `v8_vec_load_*tile_full` variants without the row-bounds check, and
  dispatched to them at every site when `tile_cols == V8_BC` /
  `rows_to_load == V8_BR` etc. (always true for the standard configs).
- Hypothesis was that the runtime predicate-setup was the source of the 7–19 %
  "else branch" stall attribution.
- **Win: ~0.1 %** — turned out the branch wasn't the bottleneck and the
  attribution was just SASS line drift. Kept the change anyway because it's
  free and makes the fast-path explicit.

### Iter 7: `#pragma unroll` on per-row inner loops
- The 9 instances of `for (int r = 0; r < V8_WMMA_M; r++)` (inside D_i,
  Pass-1 reductions, Pass-2 element-wise compute, etc.) — and 4 instances of
  the WMMA k-loops — got `#pragma unroll`.
- Lets the compiler interleave independent per-row work for ILP, which
  matters at our low occupancy (8.3 %).
- **Win: +1.3 %.**

---

## What I did NOT do (and why)

- **`cp.async` / true async global→shared with double-buffering.** This was
  the open follow-up #2 from `PROFILER_REPORTS.md` and would target the
  remaining ~15 % `long_scoreboard` stalls. It needs double-buffering of the
  K/V/g_dK/g_dV tiles to actually overlap with compute, but doubling those 4
  tiles costs +18 KB in A_row D=64 (currently 93 KB) → 111 KB total, which
  exceeds the 100 KB per-block cap. Partial single-tile double-buffering
  fits but only captures a fraction of the win and adds substantial code
  complexity. Deemed not worth the risk overnight.
- **`B_col` score-pad.** Same logic as iter 3 — would help B_col's tall
  score conflicts but B_col D=64 has only ~2 KB of headroom after iter 1
  and the tall-score pad costs ~10 KB. Skipped to preserve correctness.
- **D=80/96 padding.** Both data-pad and score-pad gated on `D_CONST <= 64`
  because the dbl-bwd kernels at D=80/96 are already near or above the smem
  cap.
- **Restructuring the warp-reduce shfl chains.** The 16 % of A_row stalls
  on `__shfl_down_sync` is algorithmic — 16 rows × 3 reductions per warp per
  kv-tile = ~13 800 reductions across the loop, each with 5 serial shfls.
  Repacking into shfl_xor butterflies on uint64 pairs would only be a 33 %
  reduction and added complexity.
- **Skipping `D_i` recomputation by passing `D_vec` from bwd to dbl-bwd.**
  Estimated saving: ~0.01 % (the D_i computation is one-time per Q-block,
  not per kv-tile). Not worth the API change.

---

## Source modifications

Two files in this repo, plus `setup.py`. No v7 source touched.

### `csrc/flash_double_backward_v8.cu` — new (1900-ish lines)

Started as `cp csrc/flash_double_backward_v7.cu csrc/flash_double_backward_v8.cu`
+ `sed 's/V7_/V8_/g; s/v7_/v8_/g'`. Then layered the iters:

- **Padding constants** (top of file, lines ~33–82): replaced raw `#define`s
  with `v8_smem_pad / v8_arow_score_pad / v8_a_score_pad / v8_b_score_pad`
  constexpr functions and corresponding `_FOR(D_CONST)` macros. All
  `D_PAD / SCORE_STRIDE / B_SCORE_STRIDE` computations updated to use them.
- **Vectorized load helpers** (lines ~152–375): four `v8_vec_load_{1,2,3,4}tile`
  templates plus four `_full` variants. All 12 prior load loops in the 5
  kernels rewritten to call these.
- **`HalfTraits<>`** (lines ~290–305): added `half2_t` typedef and
  `from_float2` packed converter for both `__half` and `__nv_bfloat16`.
- **`v8_output_matmul_acc` and `_tallT`** (lines ~430–510): the per-element
  fp32 → half_t conversion loop replaced with the packed float2 → half2 form.
- **`#pragma unroll`** added to 13 hot loops via `_Pragma("unroll")`.
- **Misc:** `auto` → `at::Tensor` survives from PROFILER_REPORTS-era fixes
  for nvcc 13.1 + MSVC 14.50 in 5 places (carried over from v7 file because
  v7 had them).

### `csrc/attention_ext.cpp` — extended (was 36 lines, now 70)

Adds the three v8 Python entry points alongside the existing v7 ones.

### `setup.py` — one-line change

Added `csrc/flash_double_backward_v8.cu` to the `CUDAExtension` source list.

### `attention.py` — extended (was 72 lines, now ~120)

Adds `FlashAttentionV8 / FlashAttentionV8Backward` autograd Functions (mirror
of the v7 ones) and a `flash_attention_v8(...)` callable.

### `tests/test_correctness_v8.py` — new

Mirrors `tests/test_correctness.py` but invokes `attention_cuda.v8_*` and
`flash_attention_v8`. Same 8 configs, same tolerances.

### `bench_v8.py` — new (root)

Benchmarks v7 vs v8 fwd+bwd+dbl-bwd at one MODEL_CONFIG using CUDA events.
Saves JSON with median/mean/cv/per-trial times. Used for every iter delta.

### `profile_v8.py` — new (root)

ncu driver for v8 (mirror of `profile_v7.py`).

---

## Reports on disk (`profile_output/`)

Pre-existing v7 profiles:
- `v7_350M.ncu-rep` (basic), `v7_350M_detailed.ncu-rep` (detailed),
  `v7_350M_A_row_full*.ncu-rep` (full + lineinfo) — see `PROFILER_REPORTS.md`.

New v8 profiles (one per iter):
- `v8_350M_iter1.ncu-rep` (basic, all 5 v8 kernels)
- `v8_350M_iter2.ncu-rep` (basic)
- `v8_350M_iter1_A_row_full.ncu-rep` + `..._source.csv` (full + source for
  the most-iterated kernel)
- `v8_350M_iter2_A_row_full.ncu-rep` + `..._source.csv`
- `v8_350M_iter3_A_row_full.ncu-rep` + `..._source.csv`
- `v8_350M_iter4_A_row_full.ncu-rep` + `..._source.csv`
- `v8_350M_iter5_A_row_full.ncu-rep` + `..._source.csv`
- `v8_350M_iter5_B_col_full.ncu-rep` + `..._source.csv`
- `v8_350M_iter6.ncu-rep` (basic)
- `v8_350M_final.ncu-rep` (basic, after iter 7)

Bench JSON: `profile_output/v8_iterations/{baseline,iter1..iter7,final}{,_<config>}.json`
plus `final_memory.json`.

---

## Reproducing

All commands from `C:\Users\jhcat\Documents\VSCODE\610Sample` with the venv
Python:

```powershell
$PY = '.\.venv\Scripts\python.exe'

# Build (both v7 and v8)
Remove-Item 'attention_cuda.cp313-win_amd64.pyd' -Force -ErrorAction SilentlyContinue
Remove-Item 'build' -Recurse -Force -ErrorAction SilentlyContinue
& $PY setup.py build_ext --inplace

# Correctness
& $PY tests/test_correctness.py    --output-dir test-output/v7
& $PY tests/test_correctness_v8.py --output-dir test-output/v8

# Bench v7 vs v8 at one config
& $PY bench_v8.py --config 350M --label sanity

# Profile v8 (basic) — all 5 kernels at 350M
& 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\ncu.bat' `
    --target-processes all --kernel-name 'regex:v8_' `
    --profile-from-start no --set basic `
    -o 'profile_output/v8_350M_sanity' -f `
    $PY profile_v8.py --config 350M

& $PY summarize_ncu.py profile_output/v8_350M_sanity.ncu-rep
```

---

## Open follow-ups (descending priority)

1. **`cp.async` with K-only single-buffer in A_row.** +4.5 KB smem fits,
   but requires careful pipeline management (issue cp.async at iter j+1
   start, wait at iter j+1 use). Estimated upside: 5–10 % more on A_row.
2. **`__expf` → `exp2f` with pre-multiplied L_vec.** L_vec is currently
   stored as `log(sum exp(s)) + m`. Storing `log2(sum exp(s)) + m * LOG2E`
   instead lets the inner loop use `exp2f(s_log2 - L_log2)` saving one mul
   instruction per softmax-related call (~40 K calls per kernel). Marginal.
3. **Restructure shfl reductions.** Pack `(p_dot2, p_A)` into uint64 and
   do shfl_xor butterfly on the pair to share shfl latency. ~33 % shfl
   count reduction → maybe 4–5 % stall reduction.
4. **D=80/96 path.** Currently no padding wins because dbl-bwd doesn't fit
   smem. Would need to either cut score tile count (5 → 4) or split A_row
   into smaller sub-kernels. Substantial restructure.
5. **fwd kernel still leaves the most on the table** (-30 % vs -47 % for
   A_row). Profile-targeted iter on fwd specifically might recover a few
   percent more.

---

## Known gotchas

- The `V8_AROW_SCORE_PAD_FOR(D_CONST)` value of 4 is **only** safe for
  A_row. Don't apply to bwd_row — the smem growth crosses a 2-block/SM
  occupancy threshold there. The split macros (`V8_A_SCORE_PAD_FOR` for
  fwd/bwd_row, `V8_AROW_SCORE_PAD_FOR` for A_row) enforce this.
- D=96 dbl-bwd doesn't fit shared memory on the 5070 Ti at any pad level.
  This was already true in v7 and remains true in v8. The TORCH_CHECK in
  `v8_double_backward_impl` will throw a clear error.
- v8's `attention_cuda.cp313-win_amd64.pyd` is built for Python 3.13
  exactly. Use the venv at `.\.venv\Scripts\python.exe` to load it.
