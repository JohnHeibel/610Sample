"""
Test V8 kernel correctness (mirrors tests/test_correctness.py for V8).

Forward, backward, double backward vs reference matmul attention.
Tolerances same as V7 to gate parity.

Run:
  python tests/test_correctness_v8.py [--output-dir test-output]
"""

import sys
import os
import math
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.common import (
    make_ttt_mask, save_csv, add_output_dir_arg, ensure_output_dir,
)

ATOL_FWD = 5e-3
RTOL_FWD = 5e-3
ATOL_BWD = 5e-3
RTOL_BWD = 5e-3
ATOL_DBL = 1.5e-2
RTOL_DBL = 1.5e-2

TEST_CONFIGS = [
    ('cfg1', 1, 1, 32, 128, 64, 96, 1),
    ('cfg2', 1, 2, 64, 256, 64, 192, 2),
    ('cfg3', 2, 4, 64, 256, 64, 192, 3),
    ('cfg4', 1, 1, 64, 256, 80, 192, 1),
    ('cfg5', 1, 1, 64, 256, 96, 192, 2),
    ('cfg6', 1, 1, 64, 256, 64, 192, 0),
    ('cfg7', 1, 2, 32, 128, 64, 96, 0),
    ('cfg8', 2, 2, 64, 256, 64, 192, 1),
]


def reference_forward(Q, K, V, mask):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    return torch.matmul(P, V)


def test_forward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    import attention_cuda
    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype)

    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
    ref_O = reference_forward(Q.float(), K.float(), V.float(), mask)
    O, L = attention_cuda.v8_forward(Q, K, V, window_size, chunk_id)
    diff = (O.float() - ref_O).abs()
    return diff.max().item(), (diff / (ref_O.abs() + 1e-8)).max().item(), \
        torch.allclose(O.float(), ref_O, atol=ATOL_FWD, rtol=RTOL_FWD)


def test_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    import attention_cuda
    device = 'cuda'
    torch.manual_seed(42)
    Q = torch.randn(B, H, N_q, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, N_kv, D, device=device, dtype=dtype, requires_grad=True)
    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)

    Q_ref = Q.detach().float().requires_grad_(True)
    K_ref = K.detach().float().requires_grad_(True)
    V_ref = V.detach().float().requires_grad_(True)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_O.sum().backward()

    O, L = attention_cuda.v8_forward(Q, K, V, window_size, chunk_id)
    dO = torch.ones_like(O)
    dQ, dK, dV = attention_cuda.v8_backward(dO, Q, K, V, O, L, window_size, chunk_id)

    out = []
    for name, actual, expected in [('dQ', dQ, Q_ref.grad), ('dK', dK, K_ref.grad), ('dV', dV, V_ref.grad)]:
        diff = (actual.float() - expected).abs()
        ma = diff.max().item()
        mr = (diff / (expected.abs() + 1e-8)).max().item()
        out.append((name, ma, mr,
                    torch.allclose(actual.float(), expected, atol=ATOL_BWD, rtol=RTOL_BWD)))
    return out


def test_double_backward(B, H, N_q, N_kv, D, window_size, chunk_id, dtype):
    from attention import flash_attention_v8
    device = 'cuda'
    torch.manual_seed(42)
    Q_ref = torch.randn(B, H, N_q, D, device=device, dtype=torch.float32, requires_grad=True)
    K_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)
    V_ref = torch.randn(B, H, N_kv, D, device=device, dtype=torch.float32, requires_grad=True)
    mask = make_ttt_mask(chunk_id, N_q, N_kv, window_size, device)
    ref_O = reference_forward(Q_ref, K_ref, V_ref, mask)
    ref_grads = torch.autograd.grad(ref_O.sum(), [Q_ref, K_ref, V_ref], create_graph=True)
    uQ = torch.randn_like(Q_ref); uK = torch.randn_like(K_ref); uV = torch.randn_like(V_ref)
    h_ref = sum((g * u).sum() for g, u in zip(ref_grads, [uQ, uK, uV]))
    ref_g2 = torch.autograd.grad(h_ref, [Q_ref, K_ref, V_ref])

    Q_v = Q_ref.detach().clone().to(dtype).requires_grad_(True)
    K_v = K_ref.detach().clone().to(dtype).requires_grad_(True)
    V_v = V_ref.detach().clone().to(dtype).requires_grad_(True)
    v_O = flash_attention_v8(Q_v, K_v, V_v, window_size=window_size, chunk_id=chunk_id)
    v_grads = torch.autograd.grad(v_O.sum(), [Q_v, K_v, V_v], create_graph=True)
    uQv, uKv, uVv = uQ.to(dtype), uK.to(dtype), uV.to(dtype)
    h_v = sum((g * u).sum() for g, u in zip(v_grads, [uQv, uKv, uVv]))
    v_g2 = torch.autograd.grad(h_v, [Q_v, K_v, V_v])

    atol = ATOL_DBL if dtype == torch.bfloat16 else ATOL_FWD
    rtol = RTOL_DBL if dtype == torch.bfloat16 else RTOL_FWD
    out = []
    for name, vv, rr in [('g2_Q', v_g2[0], ref_g2[0]),
                          ('g2_K', v_g2[1], ref_g2[1]),
                          ('g2_V', v_g2[2], ref_g2[2])]:
        diff = (vv.float() - rr).abs()
        ma = diff.max().item()
        mr = (diff / (rr.abs() + 1e-8)).max().item()
        out.append((name, ma, mr, torch.allclose(vv.float(), rr, atol=atol, rtol=rtol)))
    return out


def main():
    parser = argparse.ArgumentParser(description='V8 Correctness Tests')
    add_output_dir_arg(parser)
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    dtype = torch.bfloat16
    dtype_name = 'bf16'
    all_passed = True
    csv_rows = []

    print(f"\n{'='*60}\nV8 Correctness Tests (dtype={dtype_name})\n{'='*60}")

    print("\n--- Forward Tests ---")
    for cfg_name, B, H, Nq, Nkv, D, ws, cid in TEST_CONFIGS:
        label = f"{cfg_name} B={B} H={H} Nq={Nq} Nkv={Nkv} D={D} ws={ws} cid={cid}"
        try:
            ma, mr, ok = test_forward(B, H, Nq, Nkv, D, ws, cid, dtype)
            status = "PASS" if ok else "FAIL"
            print(f"  {label}: {status} (max_abs={ma:.2e})")
            csv_rows.append([cfg_name, dtype_name, 'fwd', 'O', f'{ma:.2e}', f'{mr:.2e}', status])
            all_passed &= ok
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            csv_rows.append([cfg_name, dtype_name, 'fwd', 'O', 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    print("\n--- Backward Tests ---")
    for cfg_name, B, H, Nq, Nkv, D, ws, cid in TEST_CONFIGS:
        label = f"{cfg_name} B={B} H={H} Nq={Nq} Nkv={Nkv} D={D} ws={ws} cid={cid}"
        try:
            for name, ma, mr, ok in test_backward(B, H, Nq, Nkv, D, ws, cid, dtype):
                status = "PASS" if ok else "FAIL"
                print(f"  {label} {name}: {status} (max_abs={ma:.2e})")
                csv_rows.append([cfg_name, dtype_name, 'bwd', name, f'{ma:.2e}', f'{mr:.2e}', status])
                all_passed &= ok
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            for t in ['dQ','dK','dV']:
                csv_rows.append([cfg_name, dtype_name, 'bwd', t, 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    print("\n--- Double Backward Tests ---")
    dbl_cfgs = [(n, B, H, Nq, Nkv, D, ws, cid) for n,B,H,Nq,Nkv,D,ws,cid in TEST_CONFIGS if D != 96]
    for cfg_name, B, H, Nq, Nkv, D, ws, cid in dbl_cfgs:
        label = f"{cfg_name} B={B} H={H} Nq={Nq} Nkv={Nkv} D={D} ws={ws} cid={cid}"
        try:
            for name, ma, mr, ok in test_double_backward(B, H, Nq, Nkv, D, ws, cid, dtype):
                status = "PASS" if ok else "FAIL"
                print(f"  {label} {name}: {status} (max_abs={ma:.2e})")
                csv_rows.append([cfg_name, dtype_name, 'dbl_bwd', name, f'{ma:.2e}', f'{mr:.2e}', status])
                all_passed &= ok
        except Exception as e:
            print(f"  {label}: ERROR ({e})")
            for t in ['g2_Q','g2_K','g2_V']:
                csv_rows.append([cfg_name, dtype_name, 'dbl_bwd', t, 'ERR', 'ERR', 'ERROR'])
            all_passed = False

    save_csv(os.path.join(output_dir, 'correctness_v8_errors.csv'),
             ['config', 'dtype', 'pass_type', 'tensor', 'max_abs', 'max_rel', 'passed'],
             csv_rows)

    print(f"\n{'='*60}\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}\n{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
