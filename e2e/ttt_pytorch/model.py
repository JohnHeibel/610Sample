"""
TTT-E2E PyTorch Implementation
================================
Full reimplementation of TTT-E2E (Test-Time Training End-to-End) in PyTorch.
Mirrors the JAX/Equinox architecture from e2e/ttt/model/.

Key components:
  - RMSNorm, SwiGLU MLP, RoPE
  - SWA (Sliding Window Attention) with explicit KV cache
  - Block = Attention + optional prime MLP + MLP with pre/post norm + residuals
  - CausalLM = Embedding + Blocks + LM head
  - meta_step() = Inner loop (SGD) + outer loss for meta-learning

Architecture notes:
  - All modules operate on a single sequence (no batch dimension in the model).
    Batching is handled externally.
  - KV cache is passed explicitly through forward functions (not stored as state).
  - For meta-learning, torch.func.functional_call enables functional parameter
    updates while maintaining the computation graph for double backward.
  - V7 CUDA kernels integrate via attention.flash_attention_v7 (autograd Function
    with full double backward support).
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TTTConfig:
    """Model configuration matching JAX ModelConfig."""
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mini_batch_size: int = 1024
    sliding_window_size: int = 1024
    seq_len: int = 131072
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    qk_norm: bool = True
    pre_norm: bool = True
    post_norm: bool = True
    suffix_len: int = 0
    prime: bool = False
    use_v7_kernels: bool = False
    inner_lr: float = 0.01

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def output_size(self):
        return self.vocab_size


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    """Compute complex RoPE frequencies. Matches JAX precompute_freqs_cis."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """Apply rotary embeddings. x: [..., D], freqs_cis: [T, D//2].

    Matches JAX apply_rotary_emb: reshapes x into pairs, multiplies by
    complex freqs, flattens back.
    """
    input_dtype = x.dtype
    x_float = x.float()
    x_reshaped = x_float.reshape(*x_float.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshaped.contiguous())

    # Broadcast freqs_cis to match x_complex shape
    # x_complex: [T, H, D//2] or [T, D//2]
    # freqs_cis: [T, D//2]
    fc = freqs_cis
    while fc.ndim < x_complex.ndim:
        fc = fc.unsqueeze(-2)

    x_out = x_complex * fc
    x_out = torch.view_as_real(x_out).flatten(-2)
    return x_out.to(input_dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP: w2(silu(w1(x)) * w3(x))"""

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.resid_pdrop)

        nn.init.normal_(self.w1.weight, std=config.initializer_range)
        nn.init.normal_(self.w2.weight, std=config.initializer_range)
        nn.init.normal_(self.w3.weight, std=config.initializer_range)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Sliding Window Attention
# ---------------------------------------------------------------------------

def make_sw_causal_mask(chunk_id, N_q, N_kv, window_size, device):
    """Build sliding window causal mask.

    Matches JAX SWA.sw_causal_mask:
        (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    where qi are absolute query positions, ki are absolute key positions.
    """
    starting_query_idx = chunk_id * N_q
    ending_query_idx = starting_query_idx + N_q
    ending_key_idx = ending_query_idx

    qi = (torch.arange(N_q, device=device, dtype=torch.int32) + starting_query_idx).unsqueeze(1)
    ki = (torch.arange(-N_kv, 0, device=device, dtype=torch.int32) + ending_key_idx).unsqueeze(0)

    mask = (qi >= ki) & (qi < ki + window_size) & (ki >= 0)
    return mask  # [N_q, N_kv]


def reference_attention(Q, K, V, mask):
    """Reference masked attention (supports create_graph for double backward).

    Q: [H, N_q, d], K: [H, N_kv, d], V: [H, N_kv, d], mask: [N_q, N_kv]
    Returns: [H, N_q, d]
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [H, N_q, N_kv]
    S = S.masked_fill(~mask.unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    P = P.masked_fill(torch.isnan(P), 0.0)
    O = torch.matmul(P, V)  # [H, N_q, d]
    return O


def v7_kernel_attention(Q, K, V, chunk_id, window_size):
    """V7 CUDA kernel attention with double backward support.

    Q: [N_q, H, d], K: [N_kv, H, d], V: [N_kv, H, d]
    Returns: [N_q, H, d]
    """
    from attention import flash_attention_v7
    # flash_attention_v7 expects [B, H, N, D]
    Q_bhnd = Q.permute(1, 0, 2).unsqueeze(0)  # [1, H, N_q, d]
    K_bhnd = K.permute(1, 0, 2).unsqueeze(0)  # [1, H, N_kv, d]
    V_bhnd = V.permute(1, 0, 2).unsqueeze(0)  # [1, H, N_kv, d]

    O_bhnd = flash_attention_v7(
        Q_bhnd, K_bhnd, V_bhnd,
        window_size=window_size,
        chunk_id=chunk_id,
    )
    return O_bhnd.squeeze(0).permute(1, 0, 2)  # [N_q, H, d]


class SWA(nn.Module):
    """Sliding Window Attention with explicit KV cache.

    Matches JAX SWA from e2e/ttt/model/attention.py.
    """

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.mini_batch_size = config.mini_batch_size
        self.window_size = config.sliding_window_size

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)

        for m in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.normal_(m.weight, std=config.initializer_range)

        if config.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Precompute RoPE frequencies — must cover window_size + mini_batch_size
        max_pos = max(2 * config.seq_len, config.sliding_window_size + config.mini_batch_size)
        freqs = precompute_freqs_cis(self.head_dim, max_pos, theta=config.rope_theta)
        self.register_buffer('freqs_cis', freqs, persistent=False)

    def _split_heads(self, x):
        """[..., D] -> [..., H, d]"""
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def _merge_heads(self, x):
        """[..., H, d] -> [..., D]"""
        return x.reshape(*x.shape[:-2], self.num_heads * self.head_dim)

    def forward(self, hidden_states, chunk_id=0, kv_cache=None, is_prefix=False):
        """
        Args:
            hidden_states: [T, D]
            chunk_id: int, current chunk index
            kv_cache: (k_cache, v_cache) each [WS, D], or None
            is_prefix: if True, full causal attention (no chunked SWA)

        Returns:
            output: [T, D]
            new_kv_cache: (k_cache, v_cache) or None
            new_chunk_id: int
        """
        T = hidden_states.shape[0]
        device = hidden_states.device

        # Project QKV
        xq = self._split_heads(self.wq(hidden_states))  # [T, H, d]
        xk = self._split_heads(self.wk(hidden_states))  # [T, H, d]
        xv = self._split_heads(self.wv(hidden_states))  # [T, H, d]

        # QK norm
        if self.q_norm is not None:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if is_prefix:
            # Full causal attention for prefix blocks
            pos_ids = torch.arange(T, device=device)
            freqs = self.freqs_cis[pos_ids].to(device)
            xq = apply_rotary_emb(xq, freqs)
            xk = apply_rotary_emb(xk, freqs)

            # [H, T, d] for batched matmul
            Q = xq.permute(1, 0, 2)
            K = xk.permute(1, 0, 2)
            V = xv.permute(1, 0, 2)

            # Use SDPA with causal mask (MATH backend for create_graph compat)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                attn_output = F.scaled_dot_product_attention(
                    Q, K, V, is_causal=True
                )

            attn_output = attn_output.permute(1, 0, 2)  # [T, H, d]
            attn_output = self._merge_heads(attn_output)
            attn_output = self.resid_dropout(self.wo(attn_output))
            return attn_output, kv_cache, chunk_id

        # ----- Chunked SWA -----
        # Get previous KV cache
        if kv_cache is None:
            prev_k = torch.zeros(self.window_size, self.config.hidden_size,
                                 device=device, dtype=hidden_states.dtype)
            prev_v = torch.zeros(self.window_size, self.config.hidden_size,
                                 device=device, dtype=hidden_states.dtype)
        else:
            prev_k, prev_v = kv_cache

        prev_k = self._split_heads(prev_k)  # [WS, H, d]
        prev_v = self._split_heads(prev_v)  # [WS, H, d]

        # Concatenate current chunk with cache
        xk_full = torch.cat([prev_k, xk], dim=0)  # [WS+T, H, d]
        xv_full = torch.cat([prev_v, xv], dim=0)  # [WS+T, H, d]

        # Update KV cache
        new_kv_k = self._merge_heads(xk_full[-self.window_size:])  # [WS, D]
        new_kv_v = self._merge_heads(xv_full[-self.window_size:])  # [WS, D]

        # Apply RoPE
        N_total = self.window_size + T
        pos_ids = torch.arange(N_total, device=device)
        freqs_q = self.freqs_cis[pos_ids[-T:]].to(device)
        freqs_kv = self.freqs_cis[pos_ids].to(device)

        xq = apply_rotary_emb(xq, freqs_q)
        xk_full = apply_rotary_emb(xk_full, freqs_kv)

        # Compute attention
        N_q = T
        N_kv = xk_full.shape[0]

        if self.config.use_v7_kernels:
            attn_output = v7_kernel_attention(
                xq, xk_full, xv_full,
                chunk_id=chunk_id,
                window_size=self.window_size,
            )
        else:
            mask = make_sw_causal_mask(chunk_id, N_q, N_kv, self.window_size, device)
            # [H, N_q, d] format for reference_attention
            Q_h = xq.permute(1, 0, 2)
            K_h = xk_full.permute(1, 0, 2)
            V_h = xv_full.permute(1, 0, 2)
            attn_output = reference_attention(Q_h, K_h, V_h, mask)
            attn_output = attn_output.permute(1, 0, 2)  # [N_q, H, d]

        attn_output = self._merge_heads(attn_output)  # [T, D]
        attn_output = self.resid_dropout(self.wo(attn_output))

        return attn_output, (new_kv_k, new_kv_v), chunk_id + 1


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block: attention + optional prime MLP + MLP.

    Matches JAX Block from e2e/ttt/model/transformer.py.
    """

    def __init__(self, config: TTTConfig, has_prime_mlp=False):
        super().__init__()
        self.config = config

        self.seq_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.seq_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = SWA(config)

        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLUMLP(config)

        self.has_prime_mlp = has_prime_mlp
        if has_prime_mlp:
            self.ffn_prime_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.ffn_prime_post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.feed_forward_prime = SwiGLUMLP(config)

    def forward(self, hidden_states, chunk_id=0, kv_cache=None, is_prefix=False):
        """
        Args:
            hidden_states: [T, D]
        Returns:
            hidden_states: [T, D], new_kv_cache, new_chunk_id
        """
        # --- Attention ---
        attn_input = self.seq_norm(hidden_states) if self.config.pre_norm else hidden_states
        attn_output, kv_cache, chunk_id = self.attention(
            attn_input, chunk_id=chunk_id, kv_cache=kv_cache, is_prefix=is_prefix
        )
        if self.config.post_norm:
            attn_output = self.seq_post_norm(attn_output)
        hidden_states = hidden_states + attn_output

        # --- Prime MLP (optional, for suffix blocks in meta-learning) ---
        if self.has_prime_mlp:
            prime_input = self.ffn_prime_norm(hidden_states) if self.config.pre_norm else hidden_states
            prime_output = self.feed_forward_prime(prime_input)
            if self.config.post_norm:
                prime_output = self.ffn_prime_post_norm(prime_output)
            hidden_states = hidden_states + prime_output

        # --- Main MLP ---
        ffn_input = self.ffn_norm(hidden_states) if self.config.pre_norm else hidden_states
        ffn_output = self.feed_forward(ffn_input)
        if self.config.post_norm:
            ffn_output = self.ffn_post_norm(ffn_output)
        hidden_states = hidden_states + ffn_output

        return hidden_states, kv_cache, chunk_id


# ---------------------------------------------------------------------------
# Causal Language Model
# ---------------------------------------------------------------------------

class CausalLM(nn.Module):
    """Full causal language model: Embedding + Blocks + LM head.

    Matches JAX CausalLM from e2e/ttt/model/transformer.py.
    """

    def __init__(self, config: TTTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.wte.weight, std=config.initializer_range)

        self.dropout = nn.Dropout(config.embd_pdrop)

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            is_suffix = config.suffix_len > 0 and i >= config.num_hidden_layers - config.suffix_len
            has_prime = config.prime and is_suffix
            self.blocks.append(Block(config, has_prime_mlp=has_prime))

        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.output_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=config.initializer_range)
        else:
            self.lm_head = None

    @property
    def n_prefix(self):
        return self.config.num_hidden_layers - self.config.suffix_len

    @property
    def prefix_blocks(self):
        return list(self.blocks[:self.n_prefix])

    @property
    def suffix_blocks(self):
        return list(self.blocks[self.n_prefix:])

    def embed(self, input_ids):
        """input_ids: [T] -> [T, D]"""
        return self.dropout(self.wte(input_ids.long()))

    def logits_from_hidden(self, hidden_states):
        """hidden_states: [T, D] -> logits: [T, V]"""
        hidden_states = self.ln_f(hidden_states)
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        else:
            return F.linear(hidden_states, self.wte.weight)

    def forward_blocks(self, hidden_states, blocks, chunk_id=0, kv_caches=None, is_prefix=False):
        """Run hidden_states through a list of blocks."""
        new_kv_caches = []
        for i, block in enumerate(blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv, chunk_id = block(
                hidden_states, chunk_id=chunk_id, kv_cache=kv, is_prefix=is_prefix
            )
            new_kv_caches.append(new_kv)
        return hidden_states, new_kv_caches, chunk_id

    def forward(self, input_ids, chunk_id=0, kv_caches=None):
        """Full forward pass.

        Args:
            input_ids: [T] token ids

        Returns:
            logits: [T, V], new_kv_caches, new_chunk_id
        """
        hidden_states = self.embed(input_ids)
        hidden_states, kv_caches, chunk_id = self.forward_blocks(
            hidden_states, list(self.blocks), chunk_id=chunk_id, kv_caches=kv_caches
        )
        logits = self.logits_from_hidden(hidden_states)
        return logits, kv_caches, chunk_id


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits, targets, loss_masks=None):
    """Cross-entropy loss matching JAX cross_entropy_loss_and_accuracy.

    Args:
        logits: [T, V]
        targets: [T]
        loss_masks: [T] or None (1 = valid token)

    Returns:
        scalar loss
    """
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    if loss_masks is not None:
        loss_masks = loss_masks.float()
        valid_len = loss_masks.sum().clamp(min=1e-10)
        loss = -(token_log_probs * loss_masks).sum() / valid_len
    else:
        loss = -token_log_probs.mean()

    return loss


# ---------------------------------------------------------------------------
# Meta-learning step
# ---------------------------------------------------------------------------

def get_inner_param_names(model, config):
    """Get names of inner parameters (suffix block params for meta-learning).

    By default, selects all parameters in suffix blocks.
    """
    n_prefix = config.num_hidden_layers - config.suffix_len
    inner_names = []
    for name, _ in model.named_parameters():
        # Parameters in suffix blocks (index >= n_prefix)
        if name.startswith('blocks.'):
            block_idx = int(name.split('.')[1])
            if block_idx >= n_prefix:
                inner_names.append(name)
    return inner_names


def meta_step(model, input_ids, targets, loss_masks=None, config=None):
    """Perform one TTT-E2E meta-learning step.

    This implements the full meta-learning pipeline:
    1. Embed tokens
    2. Run prefix blocks (frozen) over full sequence
    3. For each chunk: run suffix blocks -> loss -> inner grad -> SGD update
    4. Return total loss (outer gradient flows through inner updates -> double backward)

    Args:
        model: CausalLM instance
        input_ids: [T] token ids
        targets: [T] target token ids
        loss_masks: [T] or None
        config: TTTConfig

    Returns:
        total_loss: scalar (call .backward() for outer gradient)
    """
    if config is None:
        config = model.config

    T = input_ids.shape[0]
    chunk_size = config.mini_batch_size
    assert T % chunk_size == 0, f"seq_len {T} must be divisible by mini_batch_size {chunk_size}"
    n_chunks = T // chunk_size
    device = input_ids.device

    # --- Step 1: Embed ---
    hidden_states = model.embed(input_ids)  # [T, D]

    # --- Step 2: Prefix blocks ---
    # Prefix blocks are "frozen" only w.r.t. inner SGD updates, but outer
    # gradients must still flow through them (the outer loss depends on
    # prefix output which depends on prefix block parameters).
    prefix_blocks = model.prefix_blocks
    if len(prefix_blocks) > 0:
        prefix_out, _, _ = model.forward_blocks(
            hidden_states, prefix_blocks, is_prefix=True
        )
    else:
        prefix_out = hidden_states

    # --- Step 3: Inner loop over chunks ---
    suffix_blocks = model.suffix_blocks
    n_suffix = len(suffix_blocks)

    if n_suffix == 0:
        # No suffix blocks — just do regular forward
        logits = model.logits_from_hidden(prefix_out)
        return cross_entropy_loss(logits, targets, loss_masks)

    # Collect inner parameter names (relative to the suffix block modules)
    inner_param_names = get_inner_param_names(model, config)

    # Get all model parameters as a dict
    all_params = dict(model.named_parameters())

    # Clone inner params. Keep gradient connection (no detach!) so that
    # the outer gradient can flow through the inner SGD updates back to
    # the original model parameters — this is the core of meta-learning.
    inner_state = {}
    for name in inner_param_names:
        inner_state[name] = all_params[name].clone()

    # Initialize suffix KV caches
    suffix_kv_caches = [None] * n_suffix
    suffix_chunk_id = 0

    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = chunk_start + chunk_size

        chunk_hidden = prefix_out[chunk_start:chunk_end]  # [chunk_size, D]
        chunk_targets = targets[chunk_start:chunk_end]
        chunk_masks = loss_masks[chunk_start:chunk_end] if loss_masks is not None else None

        # Build override param dict: inner_state for suffix blocks, original for rest
        param_override = {}
        for name, param in all_params.items():
            if name in inner_state:
                param_override[name] = inner_state[name]
            else:
                param_override[name] = param

        # Forward through suffix blocks
        h = chunk_hidden
        new_suffix_kv = []
        cid = suffix_chunk_id
        for i, block in enumerate(suffix_blocks):
            block_idx = model.n_prefix + i
            # Extract per-block params for functional_call
            block_params = {}
            block_prefix = f'blocks.{block_idx}.'
            for name, val in param_override.items():
                if name.startswith(block_prefix):
                    block_params[name[len(block_prefix):]] = val

            if block_params:
                h, kv, cid = torch.func.functional_call(
                    block, block_params,
                    args=(h,),
                    kwargs=dict(chunk_id=cid, kv_cache=suffix_kv_caches[i]),
                )
            else:
                h, kv, cid = block(h, chunk_id=cid, kv_cache=suffix_kv_caches[i])

            new_suffix_kv.append(kv)

        suffix_kv_caches = new_suffix_kv
        suffix_chunk_id = cid

        # Logits and loss
        logits = model.logits_from_hidden(h)  # [chunk_size, V]
        chunk_loss = cross_entropy_loss(logits, chunk_targets, chunk_masks)

        # Inner gradient
        inner_grad_values = torch.autograd.grad(
            chunk_loss,
            [inner_state[n] for n in inner_param_names],
            create_graph=True,
        )

        # SGD update
        for name, grad in zip(inner_param_names, inner_grad_values):
            inner_state[name] = inner_state[name] - config.inner_lr * grad

        total_loss = total_loss + chunk_loss

    return total_loss / n_chunks


# ---------------------------------------------------------------------------
# Simple forward-only meta step (no inner loop, for pretrain mode)
# ---------------------------------------------------------------------------

def pretrain_step(model, input_ids, targets, loss_masks=None, config=None):
    """Standard pretrain step with chunked forward.

    Processes the sequence in chunks through all blocks (no meta-learning).

    Args:
        model: CausalLM instance
        input_ids: [T] token ids
        targets: [T] target token ids
        loss_masks: [T] or None

    Returns:
        total_loss: scalar
    """
    if config is None:
        config = model.config

    T = input_ids.shape[0]
    chunk_size = config.mini_batch_size
    assert T % chunk_size == 0
    n_chunks = T // chunk_size
    device = input_ids.device

    hidden_states = model.embed(input_ids)  # [T, D]

    kv_caches = [None] * len(model.blocks)
    chunk_id = 0
    total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

    for c in range(n_chunks):
        chunk_start = c * chunk_size
        chunk_end = chunk_start + chunk_size

        chunk_hidden = hidden_states[chunk_start:chunk_end]
        chunk_targets = targets[chunk_start:chunk_end]
        chunk_masks = loss_masks[chunk_start:chunk_end] if loss_masks is not None else None

        new_kv = []
        h = chunk_hidden
        for i, block in enumerate(list(model.blocks)):
            h, kv, chunk_id_out = block(h, chunk_id=chunk_id, kv_cache=kv_caches[i])
            new_kv.append(kv)

        kv_caches = new_kv
        chunk_id = chunk_id_out

        logits = model.logits_from_hidden(h)
        chunk_loss = cross_entropy_loss(logits, chunk_targets, chunk_masks)
        total_loss = total_loss + chunk_loss

    return total_loss / n_chunks
