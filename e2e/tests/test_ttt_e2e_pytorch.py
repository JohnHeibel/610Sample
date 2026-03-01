"""
TTT-E2E PyTorch Tests
=====================
Tests for the PyTorch reimplementation of TTT-E2E.

Tests:
  1. Component tests: shapes, no NaN, forward/backward
  2. Meta-step double backward: verify outer gradients are non-zero
  3. V7 kernel integration: correctness vs reference attention
  4. Wall clock + memory benchmarks

Run:
  .venv/Scripts/python.exe e2e/tests/test_ttt_e2e_pytorch.py
"""

import sys
import os
import time
import math
import argparse

import torch
import torch.nn.functional as F

# Add project root and e2e to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ttt_pytorch.model import (
    TTTConfig,
    RMSNorm,
    SwiGLUMLP,
    SWA,
    Block,
    CausalLM,
    cross_entropy_loss,
    meta_step,
    pretrain_step,
    precompute_freqs_cis,
    apply_rotary_emb,
    make_sw_causal_mask,
    reference_attention,
)


# =====================================================================
# Test configs
# =====================================================================

SMALL_CONFIG = TTTConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    mini_batch_size=32,
    sliding_window_size=64,
    seq_len=512,
    suffix_len=2,
    prime=True,
    inner_lr=0.01,
    tie_word_embeddings=False,
    use_v7_kernels=False,
)

TINY_CONFIG = TTTConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    mini_batch_size=16,
    sliding_window_size=32,
    seq_len=64,
    suffix_len=1,
    prime=True,
    inner_lr=0.01,
    tie_word_embeddings=False,
    use_v7_kernels=False,
)

V7_CONFIG = TTTConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=1,
    mini_batch_size=64,
    sliding_window_size=192,
    seq_len=256,
    suffix_len=1,
    prime=False,
    inner_lr=0.01,
    use_v7_kernels=True,
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  {name}: {status}"
    if detail:
        msg += f" ({detail})"
    print(msg)
    return condition


# =====================================================================
# Test 1: Component tests
# =====================================================================

def test_rmsnorm():
    """Test RMSNorm produces correct shape and no NaN."""
    print("\n--- RMSNorm ---")
    ok = True
    norm = RMSNorm(64, eps=1e-6)
    x = torch.randn(16, 64)
    y = norm(x)
    ok &= check("shape", y.shape == (16, 64), f"got {y.shape}")
    ok &= check("no NaN", not torch.isnan(y).any().item())

    # Should roughly normalize variance
    var = y.float().var(dim=-1)
    ok &= check("variance ~1", var.mean().item() < 2.0, f"mean_var={var.mean().item():.3f}")
    return ok


def test_swiglu():
    """Test SwiGLU MLP shape and backward."""
    print("\n--- SwiGLU MLP ---")
    ok = True
    config = TINY_CONFIG
    mlp = SwiGLUMLP(config)
    x = torch.randn(8, config.hidden_size, requires_grad=True)
    y = mlp(x)
    ok &= check("shape", y.shape == (8, config.hidden_size), f"got {y.shape}")
    ok &= check("no NaN", not torch.isnan(y).any().item())

    loss = y.sum()
    loss.backward()
    ok &= check("grad exists", x.grad is not None)
    ok &= check("grad no NaN", not torch.isnan(x.grad).any().item())
    return ok


def test_rope():
    """Test RoPE produces correct shape and is rotation."""
    print("\n--- RoPE ---")
    ok = True
    freqs = precompute_freqs_cis(16, 64)
    ok &= check("freqs shape", freqs.shape == (64, 8), f"got {freqs.shape}")

    x = torch.randn(32, 4, 16)  # [T, H, D]
    fc = freqs[:32]
    y = apply_rotary_emb(x, fc)
    ok &= check("output shape", y.shape == x.shape, f"got {y.shape}")
    ok &= check("no NaN", not torch.isnan(y).any().item())

    # RoPE should preserve norm (approximately)
    x_norm = x.float().norm(dim=-1)
    y_norm = y.float().norm(dim=-1)
    ratio = (y_norm / x_norm).mean().item()
    ok &= check("preserves norm", abs(ratio - 1.0) < 0.01, f"ratio={ratio:.4f}")
    return ok


def test_sw_mask():
    """Test sliding window causal mask."""
    print("\n--- SW Causal Mask ---")
    ok = True

    mask = make_sw_causal_mask(chunk_id=0, N_q=32, N_kv=64, window_size=48, device='cpu')
    ok &= check("shape", mask.shape == (32, 64), f"got {mask.shape}")

    # chunk_id=0: first 32 queries, keys from -64..0 relative to end
    # starting_query_idx=0, ending_query_idx=32, ending_key_idx=32
    # ki: [-64..0) + 32 = [-32..32)
    # qi: [0..32)
    # Valid: qi >= ki AND qi < ki + 48 AND ki >= 0
    # ki ranges from -32 to 31, only ki >= 0 are valid (ki=0..31)
    ok &= check("causal", not mask[0, 0].item(),
                "q=0 shouldn't see padded keys")

    # With chunk_id=2
    mask2 = make_sw_causal_mask(chunk_id=2, N_q=32, N_kv=96, window_size=48, device='cpu')
    ok &= check("chunk2 shape", mask2.shape == (32, 96), f"got {mask2.shape}")
    ok &= check("chunk2 has True", mask2.any().item())

    return ok


def test_swa_forward():
    """Test SWA forward produces correct shapes."""
    print("\n--- SWA Forward ---")
    ok = True
    config = TINY_CONFIG
    swa = SWA(config)

    T = config.mini_batch_size
    x = torch.randn(T, config.hidden_size)

    # Regular chunked forward
    out, kv_cache, new_cid = swa(x, chunk_id=0)
    ok &= check("output shape", out.shape == (T, config.hidden_size), f"got {out.shape}")
    ok &= check("no NaN", not torch.isnan(out).any().item())
    ok &= check("kv_cache exists", kv_cache is not None)
    ok &= check("kv_cache[0] shape",
                kv_cache[0].shape == (config.sliding_window_size, config.hidden_size))
    ok &= check("chunk_id incremented", new_cid == 1)

    # Prefix mode
    x2 = torch.randn(64, config.hidden_size)
    out2, kv2, cid2 = swa(x2, chunk_id=0, is_prefix=True)
    ok &= check("prefix shape", out2.shape == (64, config.hidden_size), f"got {out2.shape}")
    ok &= check("prefix no NaN", not torch.isnan(out2).any().item())
    return ok


def test_swa_backward():
    """Test SWA backward produces gradients."""
    print("\n--- SWA Backward ---")
    ok = True
    config = TINY_CONFIG
    swa = SWA(config)

    T = config.mini_batch_size
    x = torch.randn(T, config.hidden_size, requires_grad=True)
    out, _, _ = swa(x, chunk_id=0)
    loss = out.sum()
    loss.backward()

    ok &= check("input grad exists", x.grad is not None)
    ok &= check("input grad nonzero", x.grad.abs().sum().item() > 0)
    ok &= check("wq grad exists", swa.wq.weight.grad is not None)
    return ok


def test_block_forward():
    """Test Block forward/backward."""
    print("\n--- Block ---")
    ok = True
    config = TINY_CONFIG
    block = Block(config, has_prime_mlp=True)

    T = config.mini_batch_size
    x = torch.randn(T, config.hidden_size, requires_grad=True)
    out, kv, cid = block(x, chunk_id=0)

    ok &= check("output shape", out.shape == (T, config.hidden_size), f"got {out.shape}")
    ok &= check("no NaN", not torch.isnan(out).any().item())
    ok &= check("has prime MLP", hasattr(block, 'feed_forward_prime'))

    loss = out.sum()
    loss.backward()
    ok &= check("input grad exists", x.grad is not None)
    return ok


def test_causal_lm_forward():
    """Test CausalLM forward pass."""
    print("\n--- CausalLM Forward ---")
    ok = True
    config = TINY_CONFIG
    model = CausalLM(config)

    T = config.mini_batch_size * 2  # 2 chunks
    input_ids = torch.randint(0, config.vocab_size, (T,))
    logits, kv_caches, chunk_id = model(input_ids)

    ok &= check("logits shape", logits.shape == (T, config.vocab_size),
                f"got {logits.shape}")
    ok &= check("no NaN", not torch.isnan(logits).any().item())
    ok &= check("kv_caches count", len(kv_caches) == config.num_hidden_layers)
    ok &= check("chunk_id", chunk_id == 2, f"got {chunk_id}")
    return ok


def test_cross_entropy():
    """Test cross-entropy loss."""
    print("\n--- Cross-Entropy Loss ---")
    ok = True

    logits = torch.randn(16, 64)
    targets = torch.randint(0, 64, (16,))
    loss = cross_entropy_loss(logits, targets)

    ok &= check("scalar", loss.ndim == 0)
    ok &= check("positive", loss.item() > 0)
    ok &= check("finite", torch.isfinite(loss).item())

    # With mask
    masks = torch.ones(16)
    masks[:4] = 0
    loss_masked = cross_entropy_loss(logits, targets, masks)
    ok &= check("masked finite", torch.isfinite(loss_masked).item())

    return ok


# =====================================================================
# Test 2: Meta-step double backward
# =====================================================================

def test_meta_step_cpu():
    """Test meta_step runs and produces outer gradients (CPU)."""
    print("\n--- Meta Step (CPU) ---")
    ok = True
    config = TINY_CONFIG
    model = CausalLM(config)

    T = config.mini_batch_size * 2  # 2 chunks
    input_ids = torch.randint(0, config.vocab_size, (T,))
    targets = torch.randint(0, config.vocab_size, (T,))
    loss_masks = torch.ones(T)

    # Run meta-step
    loss = meta_step(model, input_ids, targets, loss_masks, config)

    ok &= check("loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
    ok &= check("loss positive", loss.item() > 0, f"loss={loss.item():.4f}")

    # Backward (outer gradient)
    loss.backward()

    # Check gradients on outer params (prefix blocks)
    prefix_has_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            if 'blocks.0.' in name:  # prefix block
                prefix_has_grad = True
                break

    ok &= check("prefix blocks have grad", prefix_has_grad,
                "outer gradient should flow through inner loop")

    # Check suffix block params also have gradients
    suffix_has_grad = False
    n_prefix = config.num_hidden_layers - config.suffix_len
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            if f'blocks.{n_prefix}.' in name:
                suffix_has_grad = True
                break

    ok &= check("suffix blocks have grad", suffix_has_grad)

    # Check no NaN in any gradient
    any_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any().item():
            any_nan = True
            print(f"    NaN grad: {name}")

    ok &= check("no NaN grads", not any_nan)

    return ok


def test_meta_step_cuda():
    """Test meta_step with CUDA (if available)."""
    if not torch.cuda.is_available():
        print("\n--- Meta Step (CUDA) --- SKIPPED (no CUDA)")
        return True

    print("\n--- Meta Step (CUDA) ---")
    ok = True
    config = TINY_CONFIG
    device = 'cuda'

    model = CausalLM(config).to(device)

    T = config.mini_batch_size * 2
    input_ids = torch.randint(0, config.vocab_size, (T,), device=device)
    targets = torch.randint(0, config.vocab_size, (T,), device=device)
    loss_masks = torch.ones(T, device=device)

    loss = meta_step(model, input_ids, targets, loss_masks, config)

    ok &= check("loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

    loss.backward()

    grad_norm = sum(p.grad.float().norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    ok &= check("total grad norm > 0", grad_norm > 0, f"grad_norm={grad_norm:.4f}")
    ok &= check("total grad norm finite", math.isfinite(grad_norm), f"grad_norm={grad_norm:.4f}")

    return ok


def test_pretrain_step():
    """Test pretrain_step (no meta-learning)."""
    print("\n--- Pretrain Step ---")
    ok = True
    config = TINY_CONFIG
    model = CausalLM(config)

    T = config.mini_batch_size * 2
    input_ids = torch.randint(0, config.vocab_size, (T,))
    targets = torch.randint(0, config.vocab_size, (T,))

    loss = pretrain_step(model, input_ids, targets, config=config)

    ok &= check("loss finite", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")

    loss.backward()
    grad_norm = sum(p.grad.float().norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    ok &= check("grad norm > 0", grad_norm > 0, f"grad_norm={grad_norm:.4f}")

    return ok


# =====================================================================
# Test 3: V7 kernel integration
# =====================================================================

def test_v7_correctness():
    """Test V7 kernels produce same results as reference attention."""
    if not torch.cuda.is_available():
        print("\n--- V7 Correctness --- SKIPPED (no CUDA)")
        return True

    try:
        import attention_cuda
    except ImportError:
        print("\n--- V7 Correctness --- SKIPPED (attention_cuda not built)")
        return True

    print("\n--- V7 Correctness ---")
    ok = True
    device = 'cuda'
    dtype = torch.bfloat16

    # Test configs
    configs = [
        dict(H=1, N_q=32, N_kv=128, D=64, ws=96, cid=1),
        dict(H=4, N_q=64, N_kv=256, D=64, ws=192, cid=2),
    ]

    for cfg in configs:
        H, N_q, N_kv, D, ws, cid = cfg['H'], cfg['N_q'], cfg['N_kv'], cfg['D'], cfg['ws'], cfg['cid']
        label = f"H={H} Nq={N_q} Nkv={N_kv} D={D} ws={ws} cid={cid}"
        print(f"  Config: {label}")

        torch.manual_seed(42)
        Q = torch.randn(N_q, H, D, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn(N_kv, H, D, device=device, dtype=dtype, requires_grad=True)
        V = torch.randn(N_kv, H, D, device=device, dtype=dtype, requires_grad=True)

        # Reference
        mask = make_sw_causal_mask(cid, N_q, N_kv, ws, device)
        Q_h = Q.detach().clone().float().permute(1, 0, 2).requires_grad_(True)
        K_h = K.detach().clone().float().permute(1, 0, 2).requires_grad_(True)
        V_h = V.detach().clone().float().permute(1, 0, 2).requires_grad_(True)
        ref_O = reference_attention(Q_h, K_h, V_h, mask)
        ref_O_nhd = ref_O.permute(1, 0, 2)  # [N_q, H, D]

        # V7
        from ttt_pytorch.model import v7_kernel_attention
        v7_O = v7_kernel_attention(Q, K, V, chunk_id=cid, window_size=ws)

        # Forward check
        max_diff = (v7_O.float() - ref_O_nhd.detach()).abs().max().item()
        ok &= check(f"  fwd max_diff", max_diff < 5e-3, f"{max_diff:.2e}")

        # Backward check
        ref_loss = ref_O.sum()
        ref_loss.backward()

        v7_loss = v7_O.sum()
        v7_grads = torch.autograd.grad(v7_loss, [Q, K, V])

        ref_grads = [Q_h.grad.permute(1, 0, 2), K_h.grad.permute(1, 0, 2), V_h.grad.permute(1, 0, 2)]
        for gname, v7_g, ref_g in zip(['dQ', 'dK', 'dV'], v7_grads, ref_grads):
            gdiff = (v7_g.float() - ref_g).abs().max().item()
            ok &= check(f"  {gname} max_diff", gdiff < 2e-2, f"{gdiff:.2e}")

    return ok


def test_v7_meta_step():
    """Test full meta-step with V7 kernels produces valid gradients AND
    compares them numerically against the reference (matmul) attention path.

    This catches bugs where V7 produces non-zero, finite gradients that are
    nonetheless wrong (e.g. 10x off from the correct value).
    """
    if not torch.cuda.is_available():
        print("\n--- V7 Meta Step --- SKIPPED (no CUDA)")
        return True

    try:
        import attention_cuda
    except ImportError:
        print("\n--- V7 Meta Step --- SKIPPED (attention_cuda not built)")
        return True

    print("\n--- V7 Meta Step ---")
    ok = True
    device = 'cuda'

    # Shared config template (use_v7_kernels toggled below)
    def make_config(use_v7):
        return TTTConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            mini_batch_size=64,
            sliding_window_size=192,
            seq_len=256,
            suffix_len=1,
            prime=False,
            inner_lr=0.01,
            use_v7_kernels=use_v7,
        )

    T = 64 * 4  # 4 chunks
    torch.manual_seed(123)
    input_ids = torch.randint(0, 64, (T,), device=device)
    targets = torch.randint(0, 64, (T,), device=device)
    loss_masks = torch.ones(T, device=device)

    # --- Run reference (matmul attention) ---
    ref_config = make_config(use_v7=False)
    torch.manual_seed(42)
    ref_model = CausalLM(ref_config).to(device).to(torch.bfloat16)

    ref_loss = meta_step(ref_model, input_ids, targets, loss_masks, ref_config)
    ref_loss.backward()

    ref_grads = {
        name: p.grad.float().clone()
        for name, p in ref_model.named_parameters()
        if p.grad is not None
    }

    # --- Run V7 (same init via same seed) ---
    v7_config = make_config(use_v7=True)
    torch.manual_seed(42)
    v7_model = CausalLM(v7_config).to(device).to(torch.bfloat16)

    v7_loss = meta_step(v7_model, input_ids, targets, loss_masks, v7_config)

    ok &= check("v7 loss finite", torch.isfinite(v7_loss).item(),
                f"loss={v7_loss.item():.4f}")
    ok &= check("v7 loss positive", v7_loss.item() > 0)

    v7_loss.backward()

    # --- Basic sanity ---
    v7_grad_norm = sum(
        p.grad.float().norm().item() ** 2
        for p in v7_model.parameters()
        if p.grad is not None
    ) ** 0.5
    ok &= check("v7 grad norm > 0", v7_grad_norm > 0,
                f"grad_norm={v7_grad_norm:.6f}")
    ok &= check("v7 grad norm finite", math.isfinite(v7_grad_norm))

    any_nan = any(
        torch.isnan(p.grad).any().item()
        for p in v7_model.parameters()
        if p.grad is not None
    )
    ok &= check("no NaN grads", not any_nan)

    # --- Numerical comparison: V7 vs reference ---
    # Loss values should match closely
    loss_diff = abs(v7_loss.item() - ref_loss.item())
    ok &= check("loss matches ref", loss_diff < 0.1,
                f"v7={v7_loss.item():.4f} ref={ref_loss.item():.4f} "
                f"diff={loss_diff:.4f}")

    # Gradient comparison per parameter
    # bf16 meta-step compounds errors through fwd+bwd+dbl_bwd across 4 chunks,
    # so we use a per-element tolerance of 5e-2 (5%)
    META_ATOL = 5e-2
    META_RTOL = 5e-2
    n_compared = 0
    n_passed = 0
    worst_name = ""
    worst_diff = 0.0

    for name, p in v7_model.named_parameters():
        if p.grad is None or name not in ref_grads:
            continue
        v7_g = p.grad.float()
        ref_g = ref_grads[name]
        max_diff = (v7_g - ref_g).abs().max().item()
        ref_scale = ref_g.abs().max().item()

        if max_diff > worst_diff:
            worst_diff = max_diff
            worst_name = name

        n_compared += 1
        if torch.allclose(v7_g, ref_g, atol=META_ATOL, rtol=META_RTOL):
            n_passed += 1

    ok &= check(f"grad match ({n_passed}/{n_compared} params)",
                n_passed == n_compared,
                f"worst: {worst_name} max_diff={worst_diff:.2e}")

    return ok


# =====================================================================
# Test 4: Benchmarks
# =====================================================================

def benchmark_meta_step(config_name, use_v7=False, n_warmup=2, n_trials=5):
    """Benchmark a meta-step configuration."""
    if not torch.cuda.is_available():
        print(f"\n--- Benchmark {config_name} --- SKIPPED (no CUDA)")
        return

    if use_v7:
        try:
            import attention_cuda
        except ImportError:
            print(f"\n--- Benchmark {config_name} V7 --- SKIPPED (attention_cuda not built)")
            return

    configs = {
        'tiny': TTTConfig(
            vocab_size=64, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=1,
            mini_batch_size=64, sliding_window_size=192, seq_len=256,
            suffix_len=1, prime=False, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        'small': TTTConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=1,
            mini_batch_size=64, sliding_window_size=192, seq_len=256,
            suffix_len=2, prime=True, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        # Medium configs that fit in 16GB — scale up attention dimensions
        'med-H4': TTTConfig(
            vocab_size=256, hidden_size=256, intermediate_size=512,
            num_hidden_layers=4, num_attention_heads=4,
            mini_batch_size=128, sliding_window_size=512, seq_len=512,
            suffix_len=2, prime=False, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        'med-H8': TTTConfig(
            vocab_size=256, hidden_size=512, intermediate_size=1024,
            num_hidden_layers=4, num_attention_heads=8,
            mini_batch_size=128, sliding_window_size=1024, seq_len=512,
            suffix_len=2, prime=False, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        'med-H12': TTTConfig(
            vocab_size=256, hidden_size=768, intermediate_size=2048,
            num_hidden_layers=4, num_attention_heads=12,
            mini_batch_size=128, sliding_window_size=1024, seq_len=512,
            suffix_len=2, prime=False, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        # Larger KV window — this is where V7 shines (O(N) vs O(N*N_kv) memory)
        'large-kv': TTTConfig(
            vocab_size=256, hidden_size=768, intermediate_size=2048,
            num_hidden_layers=4, num_attention_heads=12,
            mini_batch_size=256, sliding_window_size=2048, seq_len=1024,
            suffix_len=2, prime=False, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
        '125M': TTTConfig(
            vocab_size=32000, hidden_size=768, intermediate_size=2048,
            num_hidden_layers=12, num_attention_heads=12,
            mini_batch_size=1024, sliding_window_size=8192, seq_len=8192,
            suffix_len=4, prime=True, inner_lr=0.01,
            use_v7_kernels=use_v7,
        ),
    }

    config = configs.get(config_name)
    if config is None:
        print(f"\n--- Benchmark {config_name} --- SKIPPED (unknown config)")
        return

    v7_label = " (V7)" if use_v7 else " (ref)"
    print(f"\n--- Benchmark: {config_name}{v7_label} ---")
    print(f"  layers={config.num_hidden_layers}, H={config.num_attention_heads}, "
          f"D={config.hidden_size // config.num_attention_heads}, "
          f"mini_batch={config.mini_batch_size}, ws={config.sliding_window_size}")

    device = 'cuda'
    dtype = torch.bfloat16

    try:
        model = CausalLM(config).to(device).to(dtype)
    except Exception as e:
        print(f"  ERROR creating model: {e}")
        return

    T = config.seq_len
    input_ids = torch.randint(0, config.vocab_size, (T,), device=device)
    targets = torch.randint(0, config.vocab_size, (T,), device=device)
    loss_masks = torch.ones(T, device=device)

    # Warmup
    for _ in range(n_warmup):
        try:
            model.zero_grad()
            loss = meta_step(model, input_ids, targets, loss_masks, config)
            loss.backward()
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  ERROR during warmup: {e}")
            return

    # Timed runs
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(n_trials):
        model.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        loss = meta_step(model, input_ids, targets, loss_masks, config)
        loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    median_time = sorted(times)[len(times) // 2]

    print(f"  Median time: {median_time * 1000:.1f} ms")
    print(f"  Peak memory: {peak_mem:.0f} MB")
    print(f"  Loss: {loss.item():.4f}")

    return median_time, peak_mem


# =====================================================================
# Test 5: Attention-only meta-step benchmark (isolates V7 advantage)
# =====================================================================

ATTN_CONFIGS = {
    # name: (H, N_q, N_kv, D, window_size, n_chunks)
    'small':    (4,   64,   256,  64,  192,   4),
    'med':      (8,  128,  1024,  64,  896,   4),
    '125M':     (12, 1024,  9216, 64, 8192,   8),
    'D80':      (8,  1024,  9216, 80, 8192,   8),
}


def _attn_meta_step_ref(W_q, W_k, W_v, all_tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype):
    """Attention-only meta-step using reference matmul attention."""
    scale = 1.0 / math.sqrt(D)
    meta_loss = torch.tensor(0.0, device=W_q.device, dtype=torch.float32)

    for chunk_id in range(n_chunks):
        start = chunk_id * N_q
        end = start + N_q
        x_q = all_tokens[:, :, start:end, :]

        kv_start = max(0, end - N_kv)
        x_kv = all_tokens[:, :, kv_start:end, :]
        if x_kv.shape[2] < N_kv:
            pad_size = N_kv - x_kv.shape[2]
            padding = torch.zeros(1, H, pad_size, D, device=W_q.device, dtype=dtype)
            x_kv = torch.cat([padding, x_kv], dim=2)

        Q = torch.matmul(x_q, W_q)
        K = torch.matmul(x_kv, W_k)
        V = torch.matmul(x_kv, W_v)

        mask = make_sw_causal_mask(chunk_id, N_q, N_kv, window_size, W_q.device)

        S = torch.matmul(Q, K.transpose(-2, -1)) * scale
        S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        P = torch.softmax(S, dim=-1)
        P = P.masked_fill(torch.isnan(P), 0.0)
        O = torch.matmul(P, V)

        chunk_loss = O.sum()
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    return meta_loss


def _attn_meta_step_v7(W_q, W_k, W_v, all_tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype):
    """Attention-only meta-step using V7 kernels."""
    from attention import flash_attention_v7

    meta_loss = torch.tensor(0.0, device=W_q.device, dtype=torch.float32)

    for chunk_id in range(n_chunks):
        start = chunk_id * N_q
        end = start + N_q
        x_q = all_tokens[:, :, start:end, :]

        kv_start = max(0, end - N_kv)
        x_kv = all_tokens[:, :, kv_start:end, :]
        if x_kv.shape[2] < N_kv:
            pad_size = N_kv - x_kv.shape[2]
            padding = torch.zeros(1, H, pad_size, D, device=W_q.device, dtype=dtype)
            x_kv = torch.cat([padding, x_kv], dim=2)

        Q = torch.matmul(x_q, W_q)
        K = torch.matmul(x_kv, W_k)
        V = torch.matmul(x_kv, W_v)

        O = flash_attention_v7(Q, K, V, window_size=window_size, chunk_id=chunk_id)

        chunk_loss = O.sum()
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    return meta_loss


def benchmark_attention_only(config_name, n_warmup=3, n_trials=10):
    """Benchmark attention-only meta-step (no model overhead)."""
    if not torch.cuda.is_available():
        print(f"\n--- Attn Bench {config_name} --- SKIPPED (no CUDA)")
        return None

    try:
        import attention_cuda
    except ImportError:
        print(f"\n--- Attn Bench {config_name} --- SKIPPED (attention_cuda not built)")
        return None

    cfg = ATTN_CONFIGS.get(config_name)
    if cfg is None:
        print(f"\n--- Attn Bench {config_name} --- SKIPPED (unknown)")
        return None

    H, N_q, N_kv, D, window_size, n_chunks = cfg
    dtype = torch.bfloat16
    device = 'cuda'

    print(f"\n--- Attn-Only: {config_name} (H={H} Nq={N_q} Nkv={N_kv} D={D} ws={window_size} chunks={n_chunks}) ---")

    def make_inputs():
        W_q = (torch.randn(1, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        W_k = (torch.randn(1, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        W_v = (torch.randn(1, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        all_tokens = torch.randn(1, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1
        return W_q, W_k, W_v, all_tokens

    results = {}
    for method_name, fn in [('ref', _attn_meta_step_ref), ('V7', _attn_meta_step_v7)]:
        # Warmup
        ok = True
        for _ in range(n_warmup):
            try:
                W_q, W_k, W_v, tokens = make_inputs()
                loss = fn(W_q, W_k, W_v, tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype)
                loss.backward()
                torch.cuda.synchronize()
                del loss, W_q, W_k, W_v, tokens
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {method_name}: ERROR {e}")
                ok = False
                break

        if not ok:
            continue

        # Timed runs
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(n_trials):
            W_q, W_k, W_v, tokens = make_inputs()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            loss = fn(W_q, W_k, W_v, tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype)
            loss.backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            del loss, W_q, W_k, W_v, tokens

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        median_time = sorted(times)[len(times) // 2]
        results[method_name] = (median_time, peak_mem)
        print(f"  {method_name}: {median_time*1000:.1f} ms  |  {peak_mem:.0f} MB")
        torch.cuda.empty_cache()

    if 'ref' in results and 'V7' in results:
        t_ref, m_ref = results['ref']
        t_v7, m_v7 = results['V7']
        print(f"  Speedup: {t_ref/t_v7:.2f}x  |  Mem save: {m_ref/m_v7:.2f}x")
        return results
    return None


# =====================================================================
# Test 6: Max-batch throughput benchmark
# =====================================================================

def _attn_meta_step_batched(fn, B, W_q, W_k, W_v, all_tokens,
                             H, D, N_q, N_kv, window_size, n_chunks, dtype):
    """Run meta-step with batch dimension B.

    W_q/W_k/W_v: [B, H, D, D], all_tokens: [B, H, seq, D]
    """
    meta_loss = torch.tensor(0.0, device=W_q.device, dtype=torch.float32)

    for chunk_id in range(n_chunks):
        start = chunk_id * N_q
        end = start + N_q
        x_q = all_tokens[:, :, start:end, :]

        kv_start = max(0, end - N_kv)
        x_kv = all_tokens[:, :, kv_start:end, :]
        if x_kv.shape[2] < N_kv:
            pad_size = N_kv - x_kv.shape[2]
            padding = torch.zeros(B, H, pad_size, D, device=W_q.device, dtype=dtype)
            x_kv = torch.cat([padding, x_kv], dim=2)

        Q = torch.matmul(x_q, W_q)
        K = torch.matmul(x_kv, W_k)
        V = torch.matmul(x_kv, W_v)

        O = fn(Q, K, V, chunk_id, window_size)

        chunk_loss = O.sum()
        grads = torch.autograd.grad(chunk_loss, [W_q, W_k, W_v], create_graph=True)
        meta_loss = meta_loss + sum(g.sum() for g in grads)

    return meta_loss


def _ref_attn_fn(Q, K, V, chunk_id, window_size):
    """Reference attention for batched benchmark."""
    B, H, N_q, D = Q.shape
    N_kv = K.shape[2]
    scale = 1.0 / math.sqrt(D)
    mask = make_sw_causal_mask(chunk_id, N_q, N_kv, window_size, Q.device)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    S = S.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    return torch.matmul(P, V)


def _v7_attn_fn(Q, K, V, chunk_id, window_size):
    """V7 attention for batched benchmark."""
    from attention import flash_attention_v7
    return flash_attention_v7(Q, K, V, window_size=window_size, chunk_id=chunk_id)


def find_max_batch(fn, attn_fn, H, D, N_q, N_kv, window_size, n_chunks, dtype,
                   device, mem_budget_mb):
    """Binary search for max batch size that fits in memory budget."""
    lo, hi = 1, 64
    best = 0

    # First check if B=1 even fits
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        B = 1
        W_q = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        W_k = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        W_v = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
        tokens = torch.randn(B, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1
        loss = fn(attn_fn, B, W_q, W_k, W_v, tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype)
        loss.backward()
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        del loss, W_q, W_k, W_v, tokens
        torch.cuda.empty_cache()
        if mem > mem_budget_mb:
            return 0, mem
        best = 1
        mem_per_b = mem  # rough per-sample cost
    except Exception:
        return 0, 0

    # Estimate max from linear scaling
    hi = min(64, max(1, int(mem_budget_mb / mem_per_b)))

    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            B = mid
            W_q = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            W_k = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            W_v = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            tokens = torch.randn(B, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1
            loss = fn(attn_fn, B, W_q, W_k, W_v, tokens, H, D, N_q, N_kv, window_size, n_chunks, dtype)
            loss.backward()
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            del loss, W_q, W_k, W_v, tokens
            torch.cuda.empty_cache()
            if mem <= mem_budget_mb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except Exception:
            hi = mid - 1
            torch.cuda.empty_cache()

    return best, 0


def benchmark_max_throughput(config_name, mem_budget_mb=14000, n_warmup=2, n_trials=5):
    """Find max batch size for ref vs V7, measure throughput at that batch."""
    if not torch.cuda.is_available():
        print(f"\n--- Throughput {config_name} --- SKIPPED (no CUDA)")
        return

    try:
        import attention_cuda
    except ImportError:
        print(f"\n--- Throughput {config_name} --- SKIPPED (attention_cuda not built)")
        return

    cfg = ATTN_CONFIGS.get(config_name)
    if cfg is None:
        return

    H, N_q, N_kv, D, window_size, n_chunks = cfg
    dtype = torch.bfloat16
    device = 'cuda'
    total_tokens = n_chunks * N_q

    print(f"\n--- Max-Batch Throughput: {config_name} "
          f"(H={H} Nq={N_q} Nkv={N_kv} D={D} ws={window_size}) ---")
    print(f"  Memory budget: {mem_budget_mb} MB")

    results = {}
    for method_name, attn_fn in [('ref', _ref_attn_fn), ('V7', _v7_attn_fn)]:
        print(f"\n  [{method_name}] Finding max batch size...", end='', flush=True)
        max_B, _ = find_max_batch(
            _attn_meta_step_batched, attn_fn,
            H, D, N_q, N_kv, window_size, n_chunks, dtype, device, mem_budget_mb
        )
        print(f" B={max_B}")

        if max_B == 0:
            print(f"  {method_name}: Can't fit B=1 in {mem_budget_mb} MB")
            continue

        # Benchmark at max batch
        def run_once():
            B = max_B
            W_q = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            W_k = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            W_v = (torch.randn(B, H, D, D, device=device, dtype=dtype) * 0.02).requires_grad_(True)
            tokens = torch.randn(B, H, n_chunks * N_q, D, device=device, dtype=dtype) * 0.1
            loss = _attn_meta_step_batched(
                attn_fn, B, W_q, W_k, W_v, tokens,
                H, D, N_q, N_kv, window_size, n_chunks, dtype
            )
            loss.backward()
            torch.cuda.synchronize()
            del loss, W_q, W_k, W_v, tokens

        # Warmup
        for _ in range(n_warmup):
            try:
                run_once()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  {method_name}: ERROR in warmup: {e}")
                break

        # Timed
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_once()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            torch.cuda.empty_cache()

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        median_time = sorted(times)[len(times) // 2]
        tokens_per_sec = (max_B * total_tokens) / median_time
        results[method_name] = (max_B, median_time, peak_mem, tokens_per_sec)

        print(f"  {method_name}: B={max_B}, {median_time*1000:.1f} ms, "
              f"{peak_mem:.0f} MB, {tokens_per_sec:.0f} tok/s")

    if 'ref' in results and 'V7' in results:
        _, _, _, tps_ref = results['ref']
        b_ref = results['ref'][0]
        _, _, _, tps_v7 = results['V7']
        b_v7 = results['V7'][0]
        print(f"\n  V7 batch advantage: {b_v7}x vs {b_ref}x ({b_v7/max(b_ref,1):.1f}x larger batch)")
        print(f"  V7 throughput advantage: {tps_v7/tps_ref:.2f}x tokens/sec")

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='TTT-E2E PyTorch Tests')
    parser.add_argument('--skip-benchmarks', action='store_true', help='Skip benchmarks')
    parser.add_argument('--benchmark-only', action='store_true', help='Only run benchmarks')
    parser.add_argument('--benchmark-config', type=str, default='tiny',
                        choices=['tiny', 'small', 'med-H4', 'med-H8', 'med-H12',
                                 'large-kv', '125M'],
                        help='Benchmark config')
    parser.add_argument('--all-configs', action='store_true',
                        help='Run all benchmark configs')
    parser.add_argument('--attn-only', action='store_true',
                        help='Run attention-only benchmarks (isolates V7 advantage)')
    parser.add_argument('--attn-config', type=str, default=None,
                        choices=list(ATTN_CONFIGS.keys()),
                        help='Specific attention-only config to benchmark')
    parser.add_argument('--throughput', action='store_true',
                        help='Run max-batch throughput benchmark')
    parser.add_argument('--throughput-config', type=str, default='125M',
                        choices=list(ATTN_CONFIGS.keys()),
                        help='Config for throughput benchmark')
    parser.add_argument('--mem-budget', type=int, default=14000,
                        help='Memory budget in MB for throughput benchmark')
    args = parser.parse_args()

    all_passed = True

    if not args.benchmark_only:
        print("=" * 60)
        print("TTT-E2E PyTorch Tests")
        print("=" * 60)

        # Component tests
        print("\n=== Component Tests ===")
        all_passed &= test_rmsnorm()
        all_passed &= test_swiglu()
        all_passed &= test_rope()
        all_passed &= test_sw_mask()
        all_passed &= test_swa_forward()
        all_passed &= test_swa_backward()
        all_passed &= test_block_forward()
        all_passed &= test_causal_lm_forward()
        all_passed &= test_cross_entropy()

        # Meta-step tests
        print("\n=== Meta-Step Tests ===")
        all_passed &= test_pretrain_step()
        all_passed &= test_meta_step_cpu()
        all_passed &= test_meta_step_cuda()

        # V7 integration tests
        print("\n=== V7 Integration Tests ===")
        all_passed &= test_v7_correctness()
        all_passed &= test_v7_meta_step()

    # Benchmarks
    if not args.skip_benchmarks:
        print("\n=== Benchmarks ===")
        if args.all_configs:
            bench_configs = ['tiny', 'small', 'med-H4', 'med-H8', 'med-H12', 'large-kv']
        else:
            bench_configs = [args.benchmark_config]

        results = {}
        for cfg_name in bench_configs:
            r_ref = benchmark_meta_step(cfg_name, use_v7=False)
            r_v7 = benchmark_meta_step(cfg_name, use_v7=True)
            if r_ref is not None and r_v7 is not None:
                results[cfg_name] = (r_ref, r_v7)

        if len(results) > 0:
            print(f"\n  Full-model meta-step summary:")
            print(f"  {'=' * 70}")
            print(f"  {'Config':<12} {'Ref (ms)':>10} {'V7 (ms)':>10} {'Speedup':>10} "
                  f"{'Ref MB':>10} {'V7 MB':>10} {'Mem save':>10}")
            print(f"  {'-' * 70}")
            for name, ((t_ref, m_ref), (t_v7, m_v7)) in results.items():
                speedup = t_ref / t_v7
                mem_save = m_ref / m_v7
                print(f"  {name:<12} {t_ref*1000:>10.1f} {t_v7*1000:>10.1f} "
                      f"{speedup:>9.2f}x {m_ref:>10.0f} {m_v7:>10.0f} "
                      f"{mem_save:>9.2f}x")
            print(f"  {'=' * 70}")

    # Attention-only benchmarks
    if args.attn_only or args.attn_config:
        print("\n=== Attention-Only Meta-Step Benchmarks ===")
        print("  (No model overhead — isolates V7 kernel advantage)")
        if args.attn_config:
            attn_configs = [args.attn_config]
        else:
            attn_configs = list(ATTN_CONFIGS.keys())

        attn_results = {}
        for cfg_name in attn_configs:
            r = benchmark_attention_only(cfg_name)
            if r is not None:
                attn_results[cfg_name] = r

        if attn_results:
            print(f"\n  Attention-only summary:")
            print(f"  {'=' * 70}")
            print(f"  {'Config':<10} {'Ref (ms)':>10} {'V7 (ms)':>10} {'Speedup':>10} "
                  f"{'Ref MB':>10} {'V7 MB':>10} {'Mem save':>10}")
            print(f"  {'-' * 70}")
            for name, res in attn_results.items():
                t_ref, m_ref = res['ref']
                t_v7, m_v7 = res['V7']
                print(f"  {name:<10} {t_ref*1000:>10.1f} {t_v7*1000:>10.1f} "
                      f"{t_ref/t_v7:>9.2f}x {m_ref:>10.0f} {m_v7:>10.0f} "
                      f"{m_ref/m_v7:>9.2f}x")
            print(f"  {'=' * 70}")

    # Max-batch throughput benchmark
    if args.throughput:
        print("\n=== Max-Batch Throughput Benchmark ===")
        print("  Given a fixed GPU memory budget, how much throughput can each method achieve?")
        benchmark_max_throughput(args.throughput_config, mem_budget_mb=args.mem_budget)

    print(f"\n{'=' * 60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'=' * 60}")
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
