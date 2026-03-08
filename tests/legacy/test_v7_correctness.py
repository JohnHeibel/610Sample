"""
V7 Correctness Tests
====================
Tests for the V7 rectangular sliding-window attention kernels.

Stages:
  1. Forward: compare O against PyTorch reference
  2. Backward: compare dQ, dK, dV against reference
  3. Double backward: scalar probe test for gradient-of-gradient

Run:
  .venv/Scripts/python.exe tests/test_v7_correctness.py
"""

import sys
import math
import torch
import torch.nn.functional as F

ATOL = 5e-3
RTOL = 5e-3

def make_ttt_mask(chunk_id, N_q, N_kv, window_size, device):
    """Replicate TTT-E2E sw_causal_mask."""
    starting_query_idx = chunk_id * N_q
    ending_query_idx = starting_query_idx + N_q
    ending_key_idx = ending_query_idx

    qi = (torch.arange(N_q, device=device, dtype=torch.int32) + starting_query_idx).unsqueeze(1)
    ki = (torch.arange(-N_kv, 0, device=device, dtype=torch.int32) + ending_key_idx).unsqueeze(0)

    mask = (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    return mask


def reference_forward(Q, K, V, mask):
    """Reference attention forward in PyTorch."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    O = torch.matmul(P, V)
    return O


def check_close(name, actual, expected, atol=ATOL, rtol=RTOL):
    """Check if tensors are close and print results."""
    if actual is None:
        print(f"  {name}: SKIPPED (not computed)")
        return True
    diff = (actual - expected).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (expected.abs() + 1e-8)).max().item()
    passed = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status} (max_abs={max_diff:.2e}, max_rel={rel_diff:.2e})")
    return passed


# =====================================================================
# Test: Forward
# =====================================================================
def test_forward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 forward against reference."""
    import attention_cuda

    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

    # Reference
    ref_O = reference_forward(Q.float(), K.float(), V.float(), mask)

    # V7
    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)

    return check_close("O", O.float(), ref_O)


# =====================================================================
# Test: Backward
# =====================================================================
def test_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 backward against reference."""
    import attention_cuda

    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

    # Reference backward
    Q_ref = Q.detach().float().requires_grad_(True)
    K_ref = K.detach().float().requires_grad_(True)
    V_ref = V.detach().float().requires_grad_(True)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_loss = ref_O.sum()
    ref_loss.backward()

    # V7 forward + backward
    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)
    dO = torch.ones_like(O)
    dQ, dK, dV = attention_cuda.v7_backward(dO, Q, K, V, O, L, window_size, chunk_id)

    ok = True
    ok &= check_close("dQ", dQ.float(), Q_ref.grad)
    ok &= check_close("dK", dK.float(), K_ref.grad)
    ok &= check_close("dV", dV.float(), V_ref.grad)
    return ok


# =====================================================================
# Test: Double Backward (scalar probe)
# =====================================================================
def test_double_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    """Test V7 double backward via scalar probe."""
    from attention import flash_attention_v7

    device = 'cuda'
    torch.manual_seed(42)

    # Reference (PyTorch matmul, create_graph)
    Q_ref = torch.randn(B, H, N_q, D, device=device, dtype=torch.float32, requires_grad=True)
    K_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)
    V_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_loss = ref_O.sum()
    ref_grads = torch.autograd.grad(ref_loss, [Q_ref, K_ref, V_ref], create_graph=True)

    # Random probe directions
    uQ = torch.randn_like(Q_ref)
    uK = torch.randn_like(K_ref)
    uV = torch.randn_like(V_ref)
    h_ref = sum((g * u).sum() for g, u in zip(ref_grads, [uQ, uK, uV]))
    ref_g2 = torch.autograd.grad(h_ref, [Q_ref, K_ref, V_ref])

    # V7 (uses flash_attention_v7 which goes through autograd)
    Q_v7 = Q_ref.detach().clone().to(dtype).requires_grad_(True)
    K_v7 = K_ref.detach().clone().to(dtype).requires_grad_(True)
    V_v7 = V_ref.detach().clone().to(dtype).requires_grad_(True)

    v7_O = flash_attention_v7(Q_v7, K_v7, V_v7, window_size=window_size, chunk_id=chunk_id)
    v7_loss = v7_O.sum()
    v7_grads = torch.autograd.grad(v7_loss, [Q_v7, K_v7, V_v7], create_graph=True)

    uQ_v7 = uQ.to(dtype)
    uK_v7 = uK.to(dtype)
    uV_v7 = uV.to(dtype)
    h_v7 = sum((g * u).sum() for g, u in zip(v7_grads, [uQ_v7, uK_v7, uV_v7]))
    v7_g2 = torch.autograd.grad(h_v7, [Q_v7, K_v7, V_v7])

    # Double backward compounds errors from 3 levels of differentiation;
    # bf16 needs wider tolerance than fp16/fwd/bwd
    dbl_atol = 1.5e-2 if dtype == torch.bfloat16 else ATOL
    dbl_rtol = 1.5e-2 if dtype == torch.bfloat16 else RTOL

    ok = True
    ok &= check_close("g2_Q", v7_g2[0].float(), ref_g2[0], atol=dbl_atol, rtol=dbl_rtol)
    ok &= check_close("g2_K", v7_g2[1].float(), ref_g2[1], atol=dbl_atol, rtol=dbl_rtol)
    ok &= check_close("g2_V", v7_g2[2].float(), ref_g2[2], atol=dbl_atol, rtol=dbl_rtol)
    return ok


# =====================================================================
# Test: Memory scaling (forward, backward, AND double backward)
# =====================================================================
def _measure_peak_memory(fn, device):
    """Run fn() after clearing caches, return peak memory delta in bytes."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    return peak - baseline


def test_memory_scaling(dtype):
    """Verify peak memory scales as O(N_q*D + N_kv*D), not O(N_q*N_kv).

    Tests all three passes: forward, backward, and double backward.
    For O(N) memory, a 4x increase in N should give roughly 4x memory.
    For O(N^2), a 4x increase gives 16x — we catch that.
    """
    import attention_cuda
    from attention import flash_attention_v7

    device = 'cuda'
    B, H, D = 1, 1, 64
    window_size = 192
    chunk_id = 2

    sizes = [(64, 256), (128, 512), (256, 1024)]
    all_ok = True

    for pass_name in ['forward', 'backward', 'double_backward']:
        memories = []
        for N_q, N_kv in sizes:
            def run(N_q=N_q, N_kv=N_kv):
                Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
                K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
                V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)

                if pass_name == 'forward':
                    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)
                elif pass_name == 'backward':
                    O, L = attention_cuda.v7_forward(Q, K, V, window_size, chunk_id)
                    dO = torch.ones_like(O)
                    dQ, dK, dV = attention_cuda.v7_backward(
                        dO, Q, K, V, O, L, window_size, chunk_id)
                else:  # double_backward
                    Q.requires_grad_(True)
                    K.requires_grad_(True)
                    V.requires_grad_(True)
                    O = flash_attention_v7(Q, K, V,
                                           window_size=window_size, chunk_id=chunk_id)
                    loss = O.sum()
                    grads = torch.autograd.grad(loss, [Q, K, V], create_graph=True)
                    h = sum(g.sum() for g in grads)
                    g2 = torch.autograd.grad(h, [Q, K, V])

            mem = _measure_peak_memory(run, device)
            memories.append((N_q, N_kv, mem))

        print(f"  {pass_name} memory:")
        for nq, nkv, m in memories:
            print(f"    N_q={nq}, N_kv={nkv}: {m / 1024:.1f} KB")

        # 4x N increase: O(N) -> ~4x mem, O(N^2) -> ~16x mem.
        # Threshold: must be < 8x for a 4x N increase (catches quadratic).
        ratio = memories[-1][2] / max(memories[0][2], 1)
        nq_ratio = memories[-1][0] / memories[0][0]
        passed = ratio < nq_ratio * 2
        status = "PASS" if passed else "FAIL"
        print(f"    ratio: {ratio:.1f}x for {nq_ratio:.0f}x N "
              f"(expect <{nq_ratio * 2:.0f}x if O(N)): {status}")
        all_ok &= passed

    return all_ok


# =====================================================================
# Main test runner
# =====================================================================
TEST_CONFIGS = [
    # (B, H, N_q, N_kv, D, window_size, chunk_id)
    (1, 1, 32, 128, 64, 96, 1),
    (1, 2, 64, 256, 64, 192, 2),
    (2, 4, 64, 256, 64, 192, 3),
    (1, 1, 64, 256, 80, 192, 1),
    (1, 1, 64, 256, 96, 192, 2),
    # chunk_id=0: most ki values are negative (ki = N_q - N_kv + j),
    # exercises the (ki >= 0) mask boundary and fully-masked rows
    (1, 1, 64, 256, 64, 192, 0),
    (1, 2, 32, 128, 64, 96, 0),
]


def main():
    all_passed = True
    dtypes_to_test = [torch.bfloat16, torch.float16]

    for dtype in dtypes_to_test:
        dtype_name = 'bf16' if dtype == torch.bfloat16 else 'fp16'
        print(f"\n{'='*60}")
        print(f"Testing dtype: {dtype_name}")
        print(f"{'='*60}")

        # Forward tests
        print(f"\n--- Forward Tests ({dtype_name}) ---")
        for cfg in TEST_CONFIGS:
            B, H, N_q, N_kv, D, ws, cid = cfg
            print(f"  B={B} H={H} N_q={N_q} N_kv={N_kv} D={D} ws={ws} chunk={cid}")
            try:
                ok = test_forward(B, H, N_q, N_kv, D, ws, cid, dtype)
                all_passed &= ok
            except Exception as e:
                print(f"  ERROR: {e}")
                all_passed = False

        # Backward tests
        print(f"\n--- Backward Tests ({dtype_name}) ---")
        for cfg in TEST_CONFIGS:
            B, H, N_q, N_kv, D, ws, cid = cfg
            print(f"  B={B} H={H} N_q={N_q} N_kv={N_kv} D={D} ws={ws} chunk={cid}")
            try:
                ok = test_backward(B, H, N_q, N_kv, D, ws, cid, dtype)
                all_passed &= ok
            except Exception as e:
                print(f"  ERROR: {e}")
                all_passed = False

        # Double backward tests (skip D=96 — needs A100 shared memory)
        print(f"\n--- Double Backward Tests ({dtype_name}) ---")
        dbl_bwd_configs = [cfg for cfg in TEST_CONFIGS if cfg[4] != 96]
        for cfg in dbl_bwd_configs:
            B, H, N_q, N_kv, D, ws, cid = cfg
            print(f"  B={B} H={H} N_q={N_q} N_kv={N_kv} D={D} ws={ws} chunk={cid}")
            try:
                ok = test_double_backward(B, H, N_q, N_kv, D, ws, cid, dtype)
                all_passed &= ok
            except Exception as e:
                print(f"  ERROR: {e}")
                all_passed = False

    # Memory scaling test
    print(f"\n--- Memory Scaling Test ---")
    try:
        ok = test_memory_scaling(torch.bfloat16)
        all_passed &= ok
    except Exception as e:
        print(f"  ERROR: {e}")
        all_passed = False

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
