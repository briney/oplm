# OPLM Model Architecture — Implementation Plan

This plan builds `src/oplm/model/` from scratch per `docs/MODEL_ARCHITECTURE.md`.
It is organized so each phase produces independently testable code; dependencies
flow top-to-bottom. All file paths are relative to the repo root unless noted.

Conventions:

- Python ≥ 3.11, `from __future__ import annotations` at the top of every file.
- `torch ≥ 2.11`, `transformers ≥ 4.45` (any version that exposes
  `PreTrainedModel`, `PretrainedConfig`, and the `AutoModelFor*` registry API).
- Tensor shape comments follow `(B, T, D)` / `(B, H, T, d_head)` convention.
- All norms compute in fp32 internally; logits returned as fp32.
- Public classes prefixed `Oplm`; internal helpers may use short names.

Glossary of shape symbols used throughout:

- `B` = batch size
- `T` = sequence length (including `<cls>` and `<eos>`)
- `D` = `hidden_size` (residual stream dim)
- `H` = `num_attention_heads`
- `d_head` = `head_dim = D / H`
- `F` = `intermediate_size` (FFN hidden dim)
- `L` = `num_hidden_layers`
- `V` = `vocab_size` (33 for OPLM)

---

## Phase 0 — Scaffolding

- [ ] **0.1 Create empty module files** under `src/oplm/model/`:
  - `__init__.py`
  - `outputs.py`
  - `norm.py`
  - `rope.py`
  - `embedding.py`
  - `masking.py`
  - `ffn.py`
  - `conv.py`
  - `attention.py`
  - `transformer.py`
  - `configuration_oplm.py`
  - `tokenization_oplm.py`
  - `modeling_oplm.py`

  Each starts with `"""<one-line module summary>."""` and
  `from __future__ import annotations`.

- [ ] **0.2 Wire `src/oplm/__init__.py`**. Leave it minimal for now (just the
  existing `__version__`). The auto-class registration goes in here in Phase 12
  once the public classes exist; importing them now would cause circulars.

- [ ] **0.3 Create test directory `tests/model/`** with an empty `__init__.py`.
  Per-component tests will mirror the source layout (`tests/model/test_norm.py`,
  `tests/model/test_rope.py`, …).

---

## Phase 1 — `outputs.py`: dataclasses for the ESM-C-style API

- [ ] **1.1 Define `LogitsConfig`** (dataclass, frozen=False):
  ```python
  @dataclass
  class LogitsConfig:
      sequence: bool = True
      return_embeddings: bool = False
      return_hidden_states: bool = False
      return_attentions: bool = False
  ```

- [ ] **1.2 Define `LogitsOutput`** (dataclass):
  ```python
  @dataclass
  class LogitsOutput:
      sequence_logits: torch.Tensor | None
      embeddings: torch.Tensor | None
      hidden_states: tuple[torch.Tensor, ...] | None
      attentions: tuple[torch.Tensor, ...] | None
  ```

- [ ] **1.3 Re-export both names** from `src/oplm/model/__init__.py`.

---

## Phase 2 — `norm.py`: LayerNorm, RMSNorm, factory

The single `norm_type` config field selects the operator used at every norm
site in the model (pre-norm, sandwich/hybrid/post-sdpa norms, QK-norm,
post-embedding norm, final norm, MLM-head intermediate norm, pre-head norm).

- [ ] **2.1 Implement `OplmLayerNorm(nn.Module)`**:
  - `__init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-6, bias: bool = True)`.
  - Learnable `weight` (gain, init to 1) of shape `normalized_shape`; learnable
    `bias` (init to 0) only when `bias=True`.
  - Forward: up-cast input to fp32, compute mean/var over the last dim
    (`x.mean(-1, keepdim=True)`, `x.var(-1, keepdim=True, unbiased=False)`),
    normalize, multiply by `weight`, add `bias` if present, cast back to input
    dtype.
  - Numerically equivalent to `torch.nn.LayerNorm` but with forced fp32
    internals.

- [ ] **2.2 Implement `OplmRMSNorm(nn.Module)`**:
  - `__init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-6)`.
  - Learnable `weight` (init to 1); no bias.
  - Forward: up-cast to fp32, compute `rms = sqrt(mean(x**2, dim=-1, keepdim=True) + eps)`,
    output `weight * x / rms`, cast back.

- [ ] **2.3 Implement `make_norm(norm_type: str, normalized_shape, eps: float) -> nn.Module`**:
  - Returns `OplmLayerNorm` for `"layernorm"`, `OplmRMSNorm` for `"rmsnorm"`.
  - Raises `ValueError` for unknown types.
  - Used everywhere the model needs a norm so a single config flag controls all
    sites.

- [ ] **2.4 Tests** (`tests/model/test_norm.py`):
  - Output dtype matches input dtype after autocast.
  - Output statistics: per-row mean ≈ 0 (LayerNorm), per-row RMS ≈ 1 (RMSNorm)
    when `weight=1`, `bias=0`.
  - Internal compute is fp32: pass bf16 input, assert intermediate (mocked or
    via known-bad fp16 reference) doesn't accumulate the expected fp16 drift.

---

## Phase 3 — `masking.py`: mask helpers

Centralizes the three places masks are used: SDPA fallback, `flex_attention`,
and Canon pre-conv zeroing.

- [ ] **3.1 `prepare_attention_mask(attention_mask: Tensor | None, batch_size: int, seq_len: int, device, dtype=torch.long) -> Tensor`**:
  - When `None`, returns an all-ones `(B, T)` long tensor on the correct device.
  - Otherwise validates dtype/shape and returns as-is.

- [ ] **3.2 `make_flex_block_mask(attention_mask: Tensor, num_heads: int) -> BlockMask`**:
  - Builds a closure `mask_mod(b, h, q_idx, kv_idx) -> bool` that returns
    `attention_mask[b, kv_idx] == 1` (bidirectional encoder: no causal
    restriction; only KV padding is masked). The closure does not actually use
    `h`, but the signature is fixed by `create_block_mask`.
  - Calls
    `torch.nn.attention.flex_attention.create_block_mask(mask_mod, B=B,
    H=num_heads, Q_LEN=T, KV_LEN=T, device=attention_mask.device)`.
    Pass `num_heads` as an explicit positive integer (not `None`) — the
    current PyTorch documentation specifies `H: int`, and although recent
    builds tolerate `None` for head-broadcasting masks, that is undocumented
    behavior. Using the real head count keeps the contract version-stable.
  - Returns the resulting `BlockMask`. Caching by `(B, T, device)` is **not**
    required — `create_block_mask` is cheap relative to the kernel itself; do
    not introduce module-level caches (violates FSDP friendliness).

- [ ] **3.3 `zero_pad_positions(x: Tensor, attention_mask: Tensor) -> Tensor`**:
  - `x: (B, T, D)`, `attention_mask: (B, T)`.
  - Returns `x * attention_mask.unsqueeze(-1).to(x.dtype)`.
  - Used by Canon conv to prevent pad leakage into real-token channels.

- [ ] **3.4 Tests** (`tests/model/test_masking.py`):
  - `prepare_attention_mask` defaults to all-ones.
  - `zero_pad_positions` zeros out pad rows, leaves real-token rows unchanged.

---

## Phase 4 — `rope.py`: rotary positional embedding

Implements full RoPE plus the partial-RoPE / NoPE split.

- [ ] **4.1 Implement `RotaryEmbedding(nn.Module)`**:
  - `__init__(self, head_dim: int, rope_dim: int, max_position_embeddings: int,
    base: float = 10000.0)`. Stores `rope_dim` (channels per head that receive
    rotation); `nope_dim = head_dim - rope_dim` is implicit.
  - Pre-computes `inv_freq = base ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)`
    of shape `(rope_dim/2,)` and registers as a buffer (non-persistent).
  - Pre-computes `cos` and `sin` buffers of shape
    `(max_position_embeddings, rope_dim/2)` at fp32 and registers as buffers
    (non-persistent).

- [ ] **4.2 Implement on-the-fly buffer extension**:
  - Method `_maybe_extend_cache(self, seq_len: int, device) -> None`:
    if `seq_len > current_max`, recompute `cos`/`sin` for the new length and
    swap buffers. Update `self.max_position_embeddings`. No parameter update —
    these are buffers.

- [ ] **4.3 Implement `apply_rotary(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]`**:
  - Inputs `q`, `k` of shape `(B, H, T, d_head)` in whatever dtype.
  - Calls `_maybe_extend_cache(T, q.device)`.
  - Splits each tensor along the last dim: first `rope_dim` channels go through
    rotation; trailing `nope_dim` channels pass through untouched.
  - Rotation (computed in fp32):
    ```
    q_rope: (B, H, T, rope_dim)
    cos, sin: (T, rope_dim/2) -> reshape to (1, 1, T, rope_dim/2)
    q_even = q_rope[..., 0::2]
    q_odd  = q_rope[..., 1::2]
    out_even = q_even * cos - q_odd * sin
    out_odd  = q_even * sin + q_odd * cos
    q_rope_out = interleave(out_even, out_odd)
    ```
    Use `torch.stack([out_even, out_odd], dim=-1).flatten(-2)` for interleave.
  - Concatenate `q_rope_out` and `q_pass` along last dim; cast back to input
    dtype.
  - Apply the same rotation to `k`.

- [ ] **4.4 No-op fast path**: if `rope_dim == 0`, return inputs unchanged
  (pure NoPE).

- [ ] **4.5 Tests** (`tests/model/test_rope.py`):
  - Norm preservation: `‖rotated‖ == ‖original‖` per channel pair.
  - Position 0 is identity: `cos=1, sin=0` ⇒ output equals input.
  - Partial RoPE: with `rope_dim=32, nope_dim=32`, the last 32 channels match
    the input bit-for-bit.
  - Buffer extension: pass a longer-than-max input, verify it runs and the
    buffer length now equals the new `T`.

---

## Phase 5 — `embedding.py`: token embedding + pooling helpers

- [ ] **5.1 Implement `OplmEmbedding(nn.Module)`**:
  - Wraps `nn.Embedding(vocab_size, hidden_size)`.
  - If `config.post_embed_norm` is `True`, holds a `make_norm(norm_type, D)`
    instance and applies it after the lookup.
  - Forward: `(B, T) int64 -> (B, T, D)`.

- [ ] **5.2 Implement pooling helpers** (module-level functions):
  - `mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor`:
    `hidden: (B, T, D)`, `attention_mask: (B, T) {0,1}`, returns `(B, D)`.
    Compute `(hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)`.
    Cast mask to hidden dtype before the multiply.
  - `cls_pool(hidden: Tensor) -> Tensor`: returns `hidden[:, 0, :]`.

- [ ] **5.3 Tests** (`tests/model/test_embedding.py`):
  - Output shape `(B, T, D)`.
  - `mean_pool` ignores pad positions (test with a known mask).
  - `cls_pool` returns position 0.

---

## Phase 6 — `ffn.py`: SwiGLU feed-forward

- [ ] **6.1 Implement `SwiGLU(nn.Module)`**:
  - `__init__(self, hidden_size: int, intermediate_size: int, bias: bool = False)`.
  - Three linears, all `bias=bias`:
    - `gate_proj: Linear(D, F)`
    - `up_proj:   Linear(D, F)`
    - `down_proj: Linear(F, D)`
  - Forward: `x: (B, T, D) -> down_proj(silu(gate_proj(x)) * up_proj(x))`.
    Use `F.silu`.

- [ ] **6.2 Implement `round_up_to(value: int, multiple: int) -> int`**:
  - Helper for computing `intermediate_size` when not set explicitly.
  - `((value + multiple - 1) // multiple) * multiple`.

- [ ] **6.3 Implement `make_ffn(config) -> nn.Module`**:
  - Returns a `SwiGLU` when `config.ffn_activation == "swiglu"`.
  - Raises `NotImplementedError` for the reserved `"geglu"` (config validation
    still accepts it; this is the implementation gate).
  - Extensible: new activations land here.

- [ ] **6.4 Tests** (`tests/model/test_ffn.py`):
  - Output shape matches input `(B, T, D)`.
  - Gradient flows through all three linears.

---

## Phase 7 — `conv.py`: Canon depthwise convolution

- [ ] **7.1 Implement `CanonConv(nn.Module)`**:
  - `__init__(self, hidden_size: int, kernel_size: int, activation: str = "none")`.
  - Validate `kernel_size >= 2`.
  - For **odd** `kernel_size`: `nn.Conv1d(D, D, kernel_size=k, groups=D,
    padding=k//2, bias=False)`. This yields same-length symmetric padding.
  - For **even** `kernel_size`: build with `padding=0`, then apply
    `F.pad(x, (k//2, k//2 - 1))` along the time axis before the conv. Document
    this asymmetry explicitly in a one-line code comment so future readers don't
    drift it.
  - Optional pointwise activation (`F.silu`, `F.gelu`, or identity).
  - Forward:
    1. Accept `x: (B, T, D)` and `attention_mask: (B, T)`.
    2. Apply `zero_pad_positions(x, attention_mask)`.
    3. Transpose to `(B, D, T)` (`.transpose(1, 2)`).
    4. Pad if even kernel (see above), then run conv.
    5. Transpose back to `(B, T, D)`.
    6. Apply pointwise activation.

- [ ] **7.2 Implement `resolve_canon_kernel_sizes(spec, num_hidden_layers: int) -> list[int]`**:
  - Resolves the polymorphic `canon_kernel_sizes` field into a flat
    `list[int]` of length `num_hidden_layers`.
  - Accepted forms:
    - `int` → broadcast.
    - `list[int]` → must have length `num_hidden_layers`; otherwise `ValueError`.
    - `dict`: `{"schedule": "linear", "min": int, "max": int}` →
      `numpy.linspace(min, max, num_hidden_layers).round().astype(int).tolist()`.
    - `dict`: `{"schedule": "constant", "value": int}` → same as scalar.
  - Final validation: every element `>= 2`.

- [ ] **7.3 Tests** (`tests/model/test_conv.py`):
  - Output shape `(B, T, D)` for odd and even kernels.
  - Pad zeroing: feed an input with non-zero values at pad positions; verify
    the conv operates as if those rows were zero.
  - `resolve_canon_kernel_sizes` for each of scalar, list, linear schedule,
    constant schedule.

---

## Phase 8 — `attention.py`: dual-path multi-head attention

The two compute paths share **one set of parameters and one set of pre-attention
transformations**. Only the kernel differs.

- [ ] **8.1 Implement `OplmAttention(nn.Module)`** with:
  - Constructor args derived from `OplmConfig`: `hidden_size,
    num_attention_heads, head_dim, rope_dim, max_position_embeddings,
    rope_theta, norm_type, norm_eps, qk_norm, attention_dropout,
    hidden_dropout, use_flex_attention, norm_strategy`. The last field is
    used only to decide whether to construct a V-norm (see below).
  - Parameters:
    - `q_proj, k_proj, v_proj`: `Linear(D, D, bias=False)` each.
    - `o_proj`: `Linear(D, D, bias=False)`. Mark
      `o_proj._is_residual_writer = True` for the `1/sqrt(2L)` init scaling
      (see §15.1 of the architecture doc).
    - `q_norm = make_norm(norm_type, d_head, norm_eps)` when `qk_norm` else
      `nn.Identity()`.
    - `k_norm`: same construction as `q_norm`.
    - `v_norm = make_norm(norm_type, d_head, norm_eps)` when
      `norm_strategy == "hybrid"` else `nn.Identity()`. Under hybrid, V is
      also normed (the paper's "QKV-norm"); under all other strategies V is
      not normed.
    - `rotary = RotaryEmbedding(head_dim, rope_dim, max_position_embeddings, base=rope_theta)`.
  - Class-level attribute `_FLEX_AVAILABLE: bool` — set once at module load by
    `try: from torch.nn.attention.flex_attention import flex_attention,
    create_block_mask` and used by §8.3.
  - Buffers: none beyond what `RotaryEmbedding` registers.

- [ ] **8.2 Internal helpers**:
  - `_project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]`:
    projects then reshapes each to `(B, H, T, d_head)` via
    `.view(B, T, H, d_head).transpose(1, 2)`.
  - `_qk_norm(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]`:
    applies `q_norm`/`k_norm` in fp32 (cast in, cast out).
  - `_apply_rope(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]`:
    delegates to `self.rotary.apply_rotary(q, k)`.
  - `_output_projection(self, attn_out: Tensor) -> Tensor`:
    reshape `(B, H, T, d_head) -> (B, T, D)` and project through `o_proj`.

- [ ] **8.3 Fast-path guard `_use_fast_path(self, output_attentions, device)`**:
  - Returns `True` only when **all** of the following hold; otherwise `False`
    and the fallback runs:
    - `output_attentions is False` — `flex_attention` returns no weights.
    - `self.use_flex_attention is True` — debug override.
    - `self.attention_dropout == 0.0` — the fast path has no `dropout_p`
      argument; honouring the configured dropout exactly requires the
      fallback. Document this in code.
    - `device.type == "cuda"` — `flex_attention` requires CUDA.
    - The `flex_attention` symbol imports without error (catch
      `ImportError` once at module load and cache the result on the class —
      `_FLEX_AVAILABLE: bool`).
  - This list is the **contract**. Any expansion of the fast path (e.g.,
    enabling on ROCm) is a deliberate change here, not an implicit fallthrough.

- [ ] **8.4 Fast-path kernel `_flex_attention(self, q, k, v, attention_mask)`**:
  - Builds the block mask via `make_flex_block_mask(attention_mask,
    num_heads=self.num_attention_heads)`.
  - Calls `torch.nn.attention.flex_attention.flex_attention(q, k, v,
    block_mask=block_mask)`.
  - Returns `(B, H, T, d_head)`.
  - Note: `flex_attention` returns no attention weights and applies no dropout;
    the caller is responsible for honouring those constraints via §8.3.

- [ ] **8.5 Fallback-path kernel `_manual_attention(self, q, k, v, attention_mask)`**:
  - `scale = 1.0 / math.sqrt(d_head)`.
  - `scores = (q @ k.transpose(-2, -1)) * scale` → `(B, H, T, T)`.
  - If `attention_mask` provided: `mask = (attention_mask == 0)[:, None, None, :]`,
    `scores = scores.masked_fill(mask, float("-inf"))`.
  - `attn = F.softmax(scores, dim=-1, dtype=torch.float32)`.
  - `attn_dropped = F.dropout(attn, p=self.attention_dropout, training=self.training)`.
  - `out = attn_dropped.to(v.dtype) @ v` → `(B, H, T, d_head)`.
  - Return `(out, attn)`; `attn` stays fp32.

- [ ] **8.6 Forward**:
  ```python
  def forward(self, x, attention_mask, output_attentions=False):
      q, k, v = self._project_qkv(x)
      q, k = self._qk_norm(q, k)
      v = self._v_norm(v)                     # nn.Identity unless hybrid
      q, k = self._apply_rope(q, k)

      if self._use_fast_path(output_attentions, q.device):
          out = self._flex_attention(q, k, v, attention_mask)
          attn = None
      else:
          out, attn = self._manual_attention(q, k, v, attention_mask)

      out = self._output_projection(out)
      out = F.dropout(out, p=self.hidden_dropout, training=self.training)
      return out, attn
  ```

- [ ] **8.7 Tests** (`tests/model/test_attention.py`):
  - Output shape `(B, T, D)`.
  - With `output_attentions=True`, returned `attn` has shape `(B, H, T, T)`,
    is fp32, and sums to 1.0 along the last dim on real positions.
  - **Two-path equivalence**: under identical seeds/dtypes, the fast and
    fallback paths produce outputs that match within `1e-2` relative tolerance
    (loosened from typical because of `flex_attention` kernel variation).
  - Pad masking: doubling the seq length with `<pad>` tokens does not change
    the output at real positions (within float tolerance).
  - QK-norm on/off: both branches run; with `qk_norm=False`, q_norm/k_norm are
    `nn.Identity`.
  - V-norm wiring: under `norm_strategy="hybrid"`, `v_norm` is a real Norm
    (not `nn.Identity`) and its parameters appear in `model.parameters()`;
    under every other strategy, `v_norm` is `nn.Identity`.
  - Fallback guards: with `attention_dropout=0.1`, `_use_fast_path` returns
    `False` even when `output_attentions=False` and CUDA is available; on a
    CPU device, `_use_fast_path` returns `False` regardless of
    `use_flex_attention`.

---

## Phase 9 — `transformer.py`: `OplmBlock` and `OplmStack`

The block is the repeating unit. Its layout is `pre` by default, with toggles
for `sandwich`, `hybrid`, `post_sdpa`, plus Canon insertions at A/B/C/D.

- [ ] **9.1 Implement `OplmBlock(nn.Module)`**:
  - Constructor args: `config, layer_idx`. Stores `layer_idx`, `num_hidden_layers`,
    `norm_strategy`, `residual_scaling` enum.
  - `self.alpha`: float, `1.0 / math.sqrt(num_hidden_layers)` when
    `residual_scaling == "sqrt_num_layers"`, else `1.0`. Stored as a python
    float, not a buffer — multiplying a tensor by a python float is the
    cheapest path and incurs no state-dict noise.
  - Sub-modules:
    - `self.attn_norm = make_norm(...)` when `norm_strategy != "hybrid"`,
      else **omitted entirely**. Hybrid suppresses the outer attention
      pre-norm; the QKV-norms inside `OplmAttention` (§8.1) provide the only
      attention-path normalization.
    - `self.attention = OplmAttention(config)`. The attention module reads
      `config.norm_strategy` and self-configures `v_norm` (a real Norm under
      hybrid, `nn.Identity()` otherwise).
    - `self.ffn_norm = make_norm(...)`. Always present.
    - `self.ffn = make_ffn(config)`.
    - Strategy-specific extras (each created only when needed):
      - `sandwich`: `self.attn_post_norm`, `self.ffn_post_norm`.
      - `post_sdpa`: `self.attn_post_norm` only.
      - `hybrid`: **no extra norm modules at the block level** — the
        attention side relies on the QKV-norms inside `OplmAttention`, and
        the FFN-side post-add Norm is the same `ffn_norm(h)` already used as
        the FFN input (no new parameters).
    - Canon convs (only when `config.canon_enabled` and the position is in
      `config.canon_positions`):
      - `self.conv_a`, `self.conv_b`, `self.conv_c`, `self.conv_d`. Each is a
        `CanonConv(hidden_size, kernel_size=config.canon_kernel_sizes[layer_idx],
         activation=config.canon_activation)`.

- [ ] **9.2 Forward — pseudocode (handles every `norm_strategy` × Canon combo)**:
  ```python
  def forward(self, x, attention_mask, output_attentions):
      # Canon A: pre-block, on residual-stream input.
      if hasattr(self, "conv_a"):
          x = x + self.conv_a(x, attention_mask)   # additive into the stream

      # Attention sublayer.
      # Hybrid: no outer pre-norm; raw x feeds attention (QKV-norm lives
      # inside `self.attention`). All other strategies: standard pre-norm.
      if self.norm_strategy == "hybrid":
          a_in = x
      else:
          a_in = self.attn_norm(x)
      if hasattr(self, "conv_b"):
          a_in = self.conv_b(a_in, attention_mask)
      attn_out, attn_weights = self.attention(a_in, attention_mask, output_attentions)
      if hasattr(self, "conv_c"):
          attn_out = self.conv_c(attn_out, attention_mask)

      if self.norm_strategy == "sandwich":
          attn_out = self.attn_post_norm(attn_out)
          h = x + self.alpha * attn_out
      elif self.norm_strategy == "post_sdpa":
          attn_out = self.attn_post_norm(attn_out)
          h = x + self.alpha * attn_out
      else:  # "pre" or "hybrid"
          h = x + self.alpha * attn_out

      # FFN sublayer.
      # Under hybrid: Norm(h) is reused as both FFN input and FFN-side
      # residual stream (paper's `X_{l+1} = FFN(Norm(Y_l)) + Norm(Y_l)`).
      # Canon D acts on the FFN input only — never on the residual.
      h_norm = self.ffn_norm(h)
      f_in = h_norm
      if hasattr(self, "conv_d"):
          f_in = self.conv_d(f_in, attention_mask)
      ffn_out = self.ffn(f_in)

      if self.norm_strategy == "sandwich":
          ffn_out = self.ffn_post_norm(ffn_out)
          y = h + self.alpha * ffn_out
      elif self.norm_strategy == "hybrid":
          # Residual on the FFN side is h_norm, NOT h.
          y = h_norm + self.alpha * ffn_out
      else:  # "pre" or "post_sdpa"
          y = h + self.alpha * ffn_out

      return y, attn_weights
  ```

  Notes:
  - The four `norm_strategy` values produce the formulas in §5.3 of the
    architecture doc. In particular, the corrected hybrid block matches
    arXiv 2503.04598's QKV-Post main method:
    `Y_l = MHA_QKV(X_l) + X_l` and
    `X_{l+1} = FFN(Norm(Y_l)) + Norm(Y_l)`.
  - Canon at A is **additive** into the residual stream (i.e., `x = x +
    conv_a(x)`). The architecture doc shows the conv inline (`x ──► [Conv] ──►
    Norm ...`); the additive form preserves the residual identity and is
    consistent with how the paper treats it — confirm this against the
    pre-conv input intent before merging. (If the project owner intends the
    conv to *replace* the stream content rather than add to it, drop the `x +`
    and feed `conv_a(x, attention_mask)` directly into `attn_norm`. Default to
    additive unless they say otherwise.)

- [ ] **9.3 Gradient-checkpointing wrapper**:
  - Block exposes a `gradient_checkpointing: bool` attribute. When `True` and
    `self.training`, wrap the body in `torch.utils.checkpoint.checkpoint(
    self._forward_impl, x, attention_mask, output_attentions, use_reentrant=False)`.
  - Implementation: factor body into `_forward_impl(...)`; `forward` chooses
    plain vs. checkpointed dispatch.

- [ ] **9.4 Implement `OplmStack(nn.Module)`** (the backbone holder):
  - Holds `embed_tokens`, `nn.ModuleList[L]` of `OplmBlock`, and `final_norm`.
  - Forward returns `(last_hidden, hidden_states_tuple_or_None,
    attentions_tuple_or_None)`.
  - Collects intermediates only when requested:
    - `hidden_states`: tuple of `(L + 1)` tensors — the post-embedding state,
      then the output of each block (post-residual). Shapes `(B, T, D)`.
    - `attentions`: tuple of `L` tensors of shape `(B, H, T, T)` (each may be
      `None` only if a particular block returns `None`; in practice when
      `output_attentions=True` all blocks fall back).

- [ ] **9.5 Tests** (`tests/model/test_transformer.py`):
  - Output shape preserved.
  - Each `norm_strategy` value constructs without error and runs forward.
  - With `output_hidden_states=True`, returned tuple length is `L + 1`.
  - Gradient-checkpoint path produces same output (within float tolerance) as
    plain path.

---

## Phase 10 — `configuration_oplm.py`: `OplmConfig`

- [ ] **10.1 Implement `OplmConfig(PretrainedConfig)`** with `model_type = "oplm"`.
  Constructor signature:

  ```python
  class OplmConfig(PretrainedConfig):
      model_type = "oplm"

      def __init__(
          self,
          *,
          vocab_size: int = 33,
          hidden_size: int = 768,
          num_hidden_layers: int = 12,
          num_attention_heads: int = 12,
          head_dim: int | None = None,
          intermediate_size: int | None = None,
          max_position_embeddings: int = 1024,
          rope_theta: float = 10000.0,
          rope_dim: int | None = None,
          nope_dim: int = 0,
          norm_type: str = "layernorm",
          norm_eps: float = 1e-6,
          norm_strategy: str = "pre",
          qk_norm: bool = True,
          post_embed_norm: bool = False,
          residual_scaling: str = "sqrt_num_layers",
          init_scale_output_projections: bool = True,
          ffn_activation: str = "swiglu",
          ffn_bias: bool = False,
          attention_dropout: float = 0.0,
          hidden_dropout: float = 0.0,
          tie_word_embeddings: bool = False,
          mlm_head_activation: str = "gelu",
          canon_enabled: bool = False,
          canon_positions: list[str] | None = None,
          canon_kernel_sizes: int | list[int] | dict = 4,
          canon_activation: str = "none",
          initializer_range: float = 0.02,
          classifier_pool: str = "mean",
          classifier_dropout: float = 0.0,
          num_labels: int = 2,
          pre_head_norm: bool = False,
          use_flex_attention: bool = True,
          gradient_checkpointing: bool = False,
          pad_token_id: int = 1,
          bos_token_id: int = 0,
          eos_token_id: int = 2,
          unk_token_id: int = 3,
          mask_token_id: int = 32,
          **kwargs,
      ) -> None:
          # ... assign self.<field> = <field> for every OPLM field ...
          # ... run validation (§10.3) ...
          # ... resolve derived fields (§10.2) ...
          super().__init__(
              pad_token_id=pad_token_id,
              bos_token_id=bos_token_id,
              eos_token_id=eos_token_id,
              tie_word_embeddings=tie_word_embeddings,
              **kwargs,
          )
  ```

  Two non-obvious rules:

  - **Accept and forward `**kwargs`**. HuggingFace passes arbitrary
    framework-level kwargs into `PretrainedConfig.__init__` — `torch_dtype`,
    `use_cache`, `transformers_version`, `_name_or_path`, `architectures`,
    `name_or_path`, etc. Swallowing them silently means `from_pretrained`
    round-trips lose information. Forwarding them via `super().__init__(
    **kwargs)` lets the base class set them correctly.
  - Pass `pad_token_id`, `bos_token_id`, `eos_token_id`, and
    `tie_word_embeddings` explicitly into `super().__init__(...)` — these are
    base-class fields that `PretrainedConfig` looks up by name, not via
    `**kwargs`. The remaining OPLM-specific fields are stored as `self.<name>
    = <name>` before the `super().__init__` call so validation can fire on them.

  Defaults (reproduced for convenience):

  | Field | Default |
  | --- | --- |
  | `vocab_size` | `33` |
  | `hidden_size` | `768` |
  | `num_hidden_layers` | `12` |
  | `num_attention_heads` | `12` |
  | `head_dim` | derived (`D / H`) |
  | `intermediate_size` | derived (`round_up_to(8/3·D, 256)`) |
  | `max_position_embeddings` | `1024` |
  | `rope_theta` | `10000.0` |
  | `rope_dim` | `head_dim` (full RoPE) |
  | `nope_dim` | `0` |
  | `norm_type` | `"layernorm"` |
  | `norm_eps` | `1e-6` |
  | `norm_strategy` | `"pre"` |
  | `qk_norm` | `True` |
  | `post_embed_norm` | `False` |
  | `residual_scaling` | `"sqrt_num_layers"` |
  | `init_scale_output_projections` | `True` |
  | `ffn_activation` | `"swiglu"` |
  | `ffn_bias` | `False` |
  | `attention_dropout` | `0.0` |
  | `hidden_dropout` | `0.0` |
  | `tie_word_embeddings` | `False` |
  | `mlm_head_activation` | `"gelu"` |
  | `canon_enabled` | `False` |
  | `canon_positions` | `[]` |
  | `canon_kernel_sizes` | `4` |
  | `canon_activation` | `"none"` |
  | `initializer_range` | `0.02` |
  | `classifier_pool` | `"mean"` |
  | `classifier_dropout` | `0.0` |
  | `num_labels` | `2` |
  | `pre_head_norm` | `False` |
  | `use_flex_attention` | `True` |
  | `gradient_checkpointing` | `False` |
  | `pad_token_id` | `1` |
  | `bos_token_id` | `0` |
  | `eos_token_id` | `2` |
  | `unk_token_id` | `3` |
  | `mask_token_id` | `32` |

- [ ] **10.2 Derived-field computation** runs in `__init__`:
  - If `head_dim is None`: `head_dim = hidden_size // num_attention_heads`.
  - If `intermediate_size is None`: `intermediate_size = round_up_to(int(8 *
    hidden_size / 3), 256)`. Import `round_up_to` from `ffn.py`.
  - If `rope_dim is None`: `rope_dim = head_dim`. (Implies `nope_dim = 0`.)

- [ ] **10.3 Validation rules** (raise `ValueError` with explicit messages):
  - `hidden_size % num_attention_heads == 0`.
  - `head_dim * num_attention_heads == hidden_size`.
  - `rope_dim + nope_dim == head_dim`.
  - `rope_dim >= 0`, `nope_dim >= 0`, `rope_dim` is even (each RoPE rotation
    consumes a channel pair).
  - `norm_type in {"layernorm", "rmsnorm"}`.
  - `norm_strategy in {"pre", "sandwich", "hybrid", "post_sdpa"}`.
  - `residual_scaling in {"sqrt_num_layers", "none"}`.
  - `ffn_activation in {"swiglu", "geglu"}`.
  - `mlm_head_activation in {"gelu", "silu", "relu"}`.
  - `canon_activation in {"none", "silu", "gelu"}`.
  - `classifier_pool in {"mean", "cls"}`.
  - If `canon_enabled`:
    - `canon_positions` non-empty, each element ⊆ `{"A", "B", "C", "D"}`, no
      duplicates.
    - `canon_kernel_sizes` resolves to a list of length `num_hidden_layers`
      with every value `>= 2`. Resolve via
      `resolve_canon_kernel_sizes(spec, num_hidden_layers)` from `conv.py` and
      cache the resolved list back onto `self.canon_kernel_sizes` so downstream
      code can index it without re-resolving.
  - `vocab_size == 33`: emit `warnings.warn(...)` only (not raise) — custom
    vocab support is future work.

- [ ] **10.4 `auto_map` population**: do **not** override `save_pretrained` to
  hand-write `auto_map`. The supported mechanism is
  `register_for_auto_class(...)` (called once at package import time in
  `src/oplm/__init__.py`, see §13.2). Once classes are registered with the
  auto-class hook, HF populates `auto_map` automatically in `config.json` and
  `tokenizer_config.json` and copies the source `.py` files alongside the
  weights on `push_to_hub`.

  The resulting `config.json` will contain (HF writes this):
  ```json
  "auto_map": {
    "AutoConfig": "configuration_oplm.OplmConfig",
    "AutoModel": "modeling_oplm.OplmModel",
    "AutoModelForMaskedLM": "modeling_oplm.OplmForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_oplm.OplmForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_oplm.OplmForTokenClassification"
  }
  ```
  but no manual JSON construction is needed.

- [ ] **10.5 Tests** (`tests/model/test_config.py`):
  - Default construction succeeds and derives expected defaults
    (`intermediate_size`, `head_dim`, `rope_dim`).
  - Each validation rule fires `ValueError` (or a warning, for vocab_size) on
    bad input.
  - `save_pretrained` / `from_pretrained` round-trip preserves all fields and
    includes `auto_map`.
  - Canon `linear` schedule produces the expected per-layer list.

---

## Phase 11 — `tokenization_oplm.py`: `OplmTokenizerFast`

- [x] **11.1 Build the underlying `tokenizers.Tokenizer` JSON**:
  - WordLevel model with the 33-token vocab from §3.1 of the architecture doc
    (IDs 0..32 in the exact order listed). The IDs and ordering must be
    bit-identical to ESM-C — a batch tokenized by ESM-C and by OPLM must
    produce the same `input_ids`.
  - Unknown token `<unk>` (id 3).
  - Pre-tokenizer: a `Split` pre-tokenizer that splits the raw string into
    individual characters (regex `""` / `Split(behavior="isolated")` with a
    character-level regex, or build a `pre_tokenizer.Sequence` that explicitly
    splits per character). Document the exact construction in code comments —
    the goal is "one token per input character," with no merges.
  - `TemplateProcessing`:
    - single: `"<cls> $A <eos>"`.
    - pair: same as single applied to each side, joined (pair input isn't a
      production code path but be safe).
    - Special tokens: `<cls>` (0), `<eos>` (2).
  - Padding token: `<pad>` (id 1).

- [x] **11.2 Implement `OplmTokenizerFast(PreTrainedTokenizerFast)`**:
  - Constructor accepts the standard HF kwargs, plus `vocab_file=None`
    (unused), and constructs the backing `tokenizers.Tokenizer` programmatically
    when no file path is provided. Persisted form goes through `tokenizer.json`.
  - `vocab_files_names = {"tokenizer_file": "tokenizer.json"}`.
  - Class-level constants:
    - `model_input_names = ["input_ids", "attention_mask"]`.
  - Set special tokens (`cls_token, pad_token, eos_token, unk_token,
    mask_token`) with the explicit IDs from §3.1.
  - Override (or rely on the base class for) `save_pretrained` so it writes
    `tokenizer.json`, `tokenizer_config.json` (with `tokenizer_class:
    "OplmTokenizerFast"`), `special_tokens_map.json`.
  - In the saved `tokenizer_config.json`, include the `auto_map` entry so
    remote-loading via `trust_remote_code` works:
    `"auto_map": {"AutoTokenizer": ["tokenization_oplm.OplmTokenizerFast", null]}`.

- [x] **11.3 Tests** (`tests/model/test_tokenizer.py`):
  - `tokenizer("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]` — the canonical
    sanity check from §3.4.
  - Batch with padding produces equal-length sequences with `<pad>` at the end.
  - Unknown chars (e.g. `*` or a digit) map to id 3 (`<unk>`).
  - `save_pretrained` / `from_pretrained` round-trip preserves vocab and
    special tokens.
  - **ESM-C parity**: if the `esm` package is installable in the test
    environment, compare a batch's `input_ids` against `EsmTokenizer`'s output.
    Mark test `@pytest.mark.skipif(not _have_esm)` so CI without esm still
    passes.

---

## Phase 12 — `modeling_oplm.py`: every public `Oplm*` class

The architecture doc deliberately puts all public model classes into a single
file so `auto_map` + `trust_remote_code` loading works without chasing imports
across files. Internal helpers live in their own modules; `modeling_oplm.py`
imports them.

- [ ] **12.1 `OplmPreTrainedModel(PreTrainedModel)` (abstract base)**:
  - Class attributes:
    ```python
    config_class = OplmConfig
    base_model_prefix = "oplm"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OplmBlock"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    ```
  - `_init_weights(self, module)` per §15.1 of the architecture doc:
    - `nn.Linear`: trunc-normal weight with std `config.initializer_range`,
      zero bias if present.
    - `nn.Embedding`: trunc-normal weight with std `config.initializer_range`;
      do **not** zero the `<pad>` row (HF convention).
    - `OplmLayerNorm` / `OplmRMSNorm`: weight=1, bias=0 (if present).
    - `nn.Conv1d` (Canon): trunc-normal weight with std `config.initializer_range`.
    - **Residual-stream-writing projections** (attention `o_proj`, FFN
      `down_proj`): when `config.init_scale_output_projections` is `True`,
      override the std to `config.initializer_range / math.sqrt(2 *
      config.num_hidden_layers)`. Detect these projections by attribute name
      (`o_proj`, `down_proj`) on parent module, since `nn.Linear` itself
      doesn't know its role. Cleanest implementation: mark them in `__init__`
      (e.g., `module._is_residual_writer = True`) and check that flag here.

- [ ] **12.2 `OplmModel(OplmPreTrainedModel)`** — the backbone:
  - Holds `embed_tokens`, `nn.ModuleList[L]` of `OplmBlock`, and `final_norm =
    make_norm(config.norm_type, config.hidden_size, eps=config.norm_eps)`.
  - Forward signature matches HF convention:
    ```python
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple
    ```
  - Resolve defaults from `config` for `output_attentions`,
    `output_hidden_states`, `return_dict`.
  - If `inputs_embeds is None`: `x = self.embed_tokens(input_ids)`. Else
    `x = inputs_embeds`.
  - Prepare attention mask via `prepare_attention_mask(...)`.
  - Iterate blocks; accumulate `hidden_states` (post-embedding + per-layer)
    and `attentions` (per-layer) when requested.
  - Apply `final_norm` to the last hidden state.
  - Return `BaseModelOutput(last_hidden_state, hidden_states, attentions)`.

- [ ] **12.3 `OplmMLMHead(nn.Module)`** (defined in this file):
  - `dense: Linear(D, D, bias=True)`.
  - `act`: `F.gelu` / `F.silu` / `F.relu` per `config.mlm_head_activation`.
  - `norm = make_norm(config.norm_type, D, eps=config.norm_eps)`.
  - `decoder: Linear(D, V, bias=True)`. Set
    `decoder._is_residual_writer = False` (i.e., do **not** apply the
    `1/sqrt(2L)` scaling here — see §15.1).
  - Forward: `decoder(norm(act(dense(x))))`.

- [ ] **12.4 `EsmcCompatMixin`** (defined in this file):

  Tokenizer attachment policy:
  - The mixin holds an optional `self.tokenizer: PreTrainedTokenizerFast | None
    = None`.
  - `OplmPreTrainedModel.from_pretrained` (overridden) auto-attaches the
    saved tokenizer when `tokenizer.json` is present in the load directory or
    Hub repo. If not, `self.tokenizer` stays `None`.
  - Users can also assign manually: `model.tokenizer = AutoTokenizer.from_pretrained(...)`.
  - **No lazy `AutoTokenizer.from_pretrained(self.config._name_or_path)`
    fallback**. That pattern silently breaks for scratch models, offline
    workflows, and tests where tokenizer files do not exist next to the
    config.

  Methods (defined on the mixin):

  ```python
  def tokenize(self, seqs: list[str], **tokenizer_kwargs) -> "BatchEncoding":
      """Tokenize `seqs` with the attached tokenizer and move to model device."""
      tok = self._require_tokenizer()
      defaults = {"return_tensors": "pt", "padding": True}
      defaults.update(tokenizer_kwargs)
      batch = tok(seqs, **defaults)
      try:
          device = next(self.parameters()).device
      except StopIteration:
          device = torch.device("cpu")
      return batch.to(device)

  def encode(self, seqs: list[str], **tokenizer_kwargs) -> torch.Tensor:
      """ESM-C-compatible: returns the padded ``input_ids`` only.

      Warning:
          This returns input IDs **without** the attention mask. If you feed
          the result into ``forward()`` without also passing
          ``attention_mask``, pad tokens are treated as real input. Prefer
          ``tokenize()`` (returns the full ``BatchEncoding``) or
          ``logits()`` (handles the mask plumbing internally).
      """
      return self.tokenize(seqs, **tokenizer_kwargs).input_ids

  def logits(
      self,
      seqs: list[str],
      config: LogitsConfig | None = None,
  ) -> LogitsOutput:
      cfg = config or LogitsConfig()
      batch = self.tokenize(seqs)
      out = self(
          input_ids=batch.input_ids,
          attention_mask=batch.attention_mask,
          output_hidden_states=(cfg.return_hidden_states or cfg.return_embeddings),
          output_attentions=cfg.return_attentions,
          return_dict=True,
      )
      return LogitsOutput(
          sequence_logits=getattr(out, "logits", None) if cfg.sequence else None,
          embeddings=out.hidden_states[-1] if cfg.return_embeddings else None,
          hidden_states=out.hidden_states if cfg.return_hidden_states else None,
          attentions=out.attentions if cfg.return_attentions else None,
      )

  def _require_tokenizer(self):
      if getattr(self, "tokenizer", None) is None:
          raise RuntimeError(
              "No tokenizer attached. Either load via "
              "`Oplm*.from_pretrained(<path-with-tokenizer-files>)` "
              "or assign one manually with "
              "`model.tokenizer = AutoTokenizer.from_pretrained(...)`."
          )
      return self.tokenizer
  ```

  - This mixin is inherited by `OplmForMaskedLM`,
    `OplmForSequenceClassification`, and `OplmForTokenClassification`.
    `OplmModel` itself also inherits it (per §2.2) but its `logits()` returns
    `sequence_logits=None` because `OplmModel` has no head — `getattr(out,
    "logits", None)` handles that case automatically.

- [ ] **12.4b `OplmPreTrainedModel.from_pretrained` override** (auto-attach tokenizer):

  ```python
  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
      model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
      # Best-effort tokenizer attachment. Failures (no tokenizer files) are
      # silent — model.tokenizer remains None, and any call to tokenize/encode/
      # logits raises with an actionable message.
      try:
          from transformers import AutoTokenizer
          model.tokenizer = AutoTokenizer.from_pretrained(
              pretrained_model_name_or_path,
              trust_remote_code=kwargs.get("trust_remote_code", False),
          )
      except (OSError, EnvironmentError, ValueError):
          model.tokenizer = None
      return model
  ```

- [ ] **12.5 `OplmForMaskedLM(OplmPreTrainedModel, EsmcCompatMixin)`**:
  - `__init__`: instantiates `self.oplm = OplmModel(config)` and
    `self.lm_head = OplmMLMHead(config)`.
  - Calls `self.post_init()` to fire `_init_weights` and `tie_weights`.
  - Override `get_output_embeddings(self) -> nn.Module`: returns
    `self.lm_head.decoder`. This makes HF's standard weight-tying machinery
    work.
  - Override `set_output_embeddings(self, new_embeddings)`.
  - `tie_weights` (override): when `config.tie_word_embeddings`, set
    `self.lm_head.decoder.weight = self.oplm.embed_tokens.weight`. Decoder bias
    remains independent.
  - Forward:
    - Standard HF kwargs (`input_ids`, `attention_mask`, `labels`,
      `output_attentions`, `output_hidden_states`, `return_dict`).
    - Run backbone, apply `lm_head`, cast logits to fp32.
    - Compute MLM loss: `F.cross_entropy(logits.view(-1, V), labels.view(-1),
      ignore_index=-100)` when `labels is not None`.
    - Return `MaskedLMOutput`.

- [ ] **12.6 `OplmForSequenceClassification(OplmPreTrainedModel, EsmcCompatMixin)`**:
  - `__init__`: `self.oplm = OplmModel(config)`; optional `self.pre_head_norm`
    (only if `config.pre_head_norm`); `self.dropout = nn.Dropout(
    config.classifier_dropout)`; `self.classifier = nn.Linear(D,
    config.num_labels, bias=True)`.
  - Forward:
    - Run backbone.
    - Pool: `mean_pool(last_hidden, attention_mask)` or `cls_pool(last_hidden)`
      per `config.classifier_pool`.
    - Apply `pre_head_norm` if present, then dropout, then classifier.
    - Cast logits to fp32.
    - Compute `F.cross_entropy(logits, labels)` when `labels` provided (multi-
      class). When `config.num_labels == 1`, fall back to MSE (HF convention)
      — implement matching their `problem_type` detection if you want to match
      HF idioms, otherwise leave a single `cross_entropy` path and document.
    - Return `SequenceClassifierOutput`.

- [ ] **12.7 `OplmForTokenClassification(OplmPreTrainedModel, EsmcCompatMixin)`**:
  - `__init__`: `self.oplm = OplmModel(config)`; optional `self.pre_head_norm`;
    `self.dropout = nn.Dropout(config.classifier_dropout)`;
    `self.classifier = nn.Linear(D, config.num_labels, bias=True)`.
  - Forward:
    - Run backbone.
    - Optional `pre_head_norm`, then dropout, then per-token classifier.
    - Cast logits to fp32.
    - Loss: `F.cross_entropy(logits.view(-1, num_labels), labels.view(-1),
      ignore_index=-100)`.
    - Return `TokenClassifierOutput`.

- [ ] **12.8 Module re-exports**: at top of `modeling_oplm.py`, export every
  public class: `__all__ = ["OplmPreTrainedModel", "OplmModel",
  "OplmForMaskedLM", "OplmForSequenceClassification",
  "OplmForTokenClassification", "OplmMLMHead", "EsmcCompatMixin"]`.

---

## Phase 13 — Package `__init__.py` and Auto-class registration

- [ ] **13.1 Update `src/oplm/model/__init__.py`** to re-export the public
  surface so `from oplm.model import OplmForMaskedLM` works:
  ```python
  from .configuration_oplm import OplmConfig
  from .modeling_oplm import (
      OplmPreTrainedModel,
      OplmModel,
      OplmForMaskedLM,
      OplmForSequenceClassification,
      OplmForTokenClassification,
  )
  from .tokenization_oplm import OplmTokenizerFast
  from .outputs import LogitsConfig, LogitsOutput

  __all__ = [
      "OplmConfig",
      "OplmPreTrainedModel",
      "OplmModel",
      "OplmForMaskedLM",
      "OplmForSequenceClassification",
      "OplmForTokenClassification",
      "OplmTokenizerFast",
      "LogitsConfig",
      "LogitsOutput",
  ]
  ```

- [ ] **13.2 Update `src/oplm/__init__.py`** to register with HF Auto classes
  AND mark each class for auto-class custom-code copying (§13 of the
  architecture doc):

  ```python
  from transformers import (
      AutoConfig, AutoModel, AutoModelForMaskedLM,
      AutoModelForSequenceClassification, AutoModelForTokenClassification,
      AutoTokenizer,
  )
  from .model import (
      OplmConfig, OplmModel, OplmForMaskedLM,
      OplmForSequenceClassification, OplmForTokenClassification,
      OplmTokenizerFast, LogitsConfig, LogitsOutput,
  )

  # (1) In-process registration so `import oplm` plus AutoModel*.from_pretrained
  # works without trust_remote_code.
  AutoConfig.register("oplm", OplmConfig)
  AutoModel.register(OplmConfig, OplmModel)
  AutoModelForMaskedLM.register(OplmConfig, OplmForMaskedLM)
  AutoModelForSequenceClassification.register(OplmConfig, OplmForSequenceClassification)
  AutoModelForTokenClassification.register(OplmConfig, OplmForTokenClassification)
  AutoTokenizer.register(OplmConfig, fast_tokenizer_class=OplmTokenizerFast)

  # (2) Tell HF to copy the custom-code .py files when push_to_hub is called
  # and to write the matching auto_map entries into config.json /
  # tokenizer_config.json. Setting auto_map manually is NOT sufficient —
  # register_for_auto_class is the documented hook for the file-copy step.
  OplmConfig.register_for_auto_class("AutoConfig")
  OplmModel.register_for_auto_class("AutoModel")
  OplmForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
  OplmForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
  OplmForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
  OplmTokenizerFast.register_for_auto_class("AutoTokenizer")

  __all__ = [
      "OplmConfig", "OplmModel", "OplmForMaskedLM",
      "OplmForSequenceClassification", "OplmForTokenClassification",
      "OplmTokenizerFast", "LogitsConfig", "LogitsOutput",
  ]
  ```
  Keep `__version__` unchanged.

- [ ] **13.3 Verify the registration is idempotent** — re-importing `oplm`
  must not raise. (HF's `register` raises on duplicate by default; if so,
  guard with `try/except` or check the registry first.) The same guard applies
  to `register_for_auto_class` — it sets a class attribute, so the second call
  is a no-op in practice, but wrap it in a try/except if you observe spurious
  warnings under repeated import.

- [ ] **13.4 Push-to-Hub round-trip test** (`tests/model/test_push_to_hub.py`,
  mark `@pytest.mark.slow`, requires `HF_TOKEN` and a writable test repo —
  skip when unavailable):

  - `model.save_pretrained(tmpdir)` followed by inspection of `tmpdir`:
    assert that `modeling_oplm.py`, `configuration_oplm.py`, and
    `tokenization_oplm.py` (plus the helper modules they import) all landed
    in `tmpdir` next to `config.json`.
  - In a **subprocess** (so that `import oplm` does not poison the module
    cache), run `python -c "from transformers import
    AutoModelForMaskedLM; AutoModelForMaskedLM.from_pretrained('<tmpdir>',
    trust_remote_code=True)"` and assert exit code 0. This is the only way to
    detect regressions in the custom-code copy step.

---

## Phase 14 — Integration & end-to-end tests

These tests exercise the public surface and would catch wiring bugs that
component tests miss.

- [ ] **14.1 `tests/model/test_e2e_forward.py`**:
  - Build a tiny `OplmConfig` (e.g., `hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4`) and instantiate each of the four task classes.
  - Forward a `(2, 16)` input through each; assert output shapes match §14.2.
  - With `output_hidden_states=True`, assert tuple length `L+1`.
  - With `output_attentions=True`, assert each attention tensor is fp32 and
    sums to 1 along the last dim on real positions.
  - Loss is finite when `labels` are provided.

- [ ] **14.2 `tests/model/test_save_load.py`**:
  - `save_pretrained(tmpdir)` + `from_pretrained(tmpdir)` round-trip for each
    task class. Assert state-dict equality and identical forward outputs on the
    same input.
  - `OplmTokenizerFast.save_pretrained(tmpdir)` +
    `AutoTokenizer.from_pretrained(tmpdir)` resolves back to
    `OplmTokenizerFast`.

- [ ] **14.3 `tests/model/test_auto_classes.py`**:
  - After `import oplm`, `AutoConfig.from_pretrained(tmpdir)`,
    `AutoModelForMaskedLM.from_pretrained(tmpdir)`, etc. all return the
    expected concrete `Oplm*` classes (no `trust_remote_code`).

- [ ] **14.4 `tests/model/test_toggles.py`**:
  - Parametrize over the Cartesian product of:
    - `norm_type ∈ {"layernorm", "rmsnorm"}`
    - `norm_strategy ∈ {"pre", "sandwich", "hybrid", "post_sdpa"}`
    - `canon_enabled ∈ {False, True}` (with `canon_positions=["A","C","D"]`
      when enabled)
    - `rope_dim ∈ {head_dim, head_dim // 2}` (full RoPE + partial RoPE)
    - `qk_norm ∈ {True, False}`
    - `residual_scaling ∈ {"sqrt_num_layers", "none"}`
    - `tie_word_embeddings ∈ {False, True}`
    - `post_embed_norm ∈ {False, True}`
  - For each combo: instantiate a tiny model and run one forward + one
    backward (`loss.backward()`); assert no NaNs and that every parameter has
    a non-None grad.
  - Keep `num_hidden_layers=2`, `hidden_size=64` so the matrix runs fast.

- [ ] **14.5 `tests/model/test_esmc_api.py`**:
  - Build `OplmForMaskedLM`. Construct an `OplmTokenizerFast` and assign it:
    `model.tokenizer = tokenizer` (the auto-attach branch of
    `from_pretrained` is exercised separately in `test_save_load.py`).
  - **Missing tokenizer**: with a fresh `OplmForMaskedLM(config)` (no
    tokenizer assigned), `model.tokenize(["MEEPQ"])` raises with an
    actionable message referencing both `from_pretrained` and manual
    assignment.
  - `model.tokenize(["MEEPQ", "GAGT"])` returns a `BatchEncoding` with both
    `input_ids` and `attention_mask`; both are on the model's device.
  - `model.encode(["MEEPQ"])` returns only `input_ids` (asserting against
    ESM-C parity is left to the tokenizer test in §11.3); the test docstring
    explicitly notes that this API is a footgun if used without the matching
    mask.
  - `model.logits(["MEEPQ", "GAGT"], LogitsConfig(sequence=True,
    return_embeddings=True))` — assert `sequence_logits.shape == (2, T, 33)`
    and `embeddings.shape == (2, T, D)`. Compare against a direct
    `model(**model.tokenize([...]))` forward pass and require bit-identical
    logits — proves `logits()` carries the attention mask through correctly.

- [ ] **14.6 `tests/model/test_pilot_train.py`** (mark `@pytest.mark.slow`):
  - Build a 4-layer / 128-hidden / 4-head MLM, run 5 train steps on synthetic
    MLM data with an `AdamW` optimizer, assert loss decreases (or at least
    doesn't NaN). Validates that the full graph trains end-to-end.

- [ ] **14.7 `tests/model/test_gradient_checkpointing.py`**:
  - Same forward, with and without `config.gradient_checkpointing`, produces
    matching outputs (within float tolerance) and matching grads on a fixed
    seed.

---

## Phase 15 — Documentation polish

- [ ] **15.1 Update `src/oplm/model/__init__.py`'s top-level docstring** with a
  one-paragraph overview that points readers at `docs/MODEL_ARCHITECTURE.md`.

- [ ] **15.2 Add module-level docstrings** to each helper file describing the
  one role of that module (e.g., `rope.py`: "RoPE / partial RoPE applied to Q
  and K post-QK-norm.").

- [ ] **15.3 Update `CLAUDE.md`'s project-structure section** if the new file
  list materially changes the previous layout. (The model dir was previously
  populated with a different file set; the new layout matches §19 of the
  architecture doc.)

---

## Open questions to confirm before implementation

These were not unambiguously resolved by reading the architecture doc; flag
them for the project owner during Phase 9 / Phase 12 work:

1. **Canon at position A — additive or replacing?** The doc draws the conv
   inline (`x ──► [Conv_A] ──► Norm ──► ...`), which reads as a replacement,
   but the residual identity is preserved more cleanly with an additive form
   (`x = x + conv_A(x)`). Default to additive in the implementation; ask the
   owner during review.

2. **`residual_scaling` divisor**: §5.1 of the doc notes a literal `1/sqrt(L)`
   per the owner's preference, but mentions `1/sqrt(2L)` as an equally
   defensible alternative. Implementation uses `1/sqrt(num_hidden_layers)`
   exactly as written; no second toggle.

3. **`hybrid` norm strategy** — RESOLVED. The block matches arXiv 2503.04598's
   "QKV-Post" main method exactly:
   `Y_l = MHA_QKV(X_l) + X_l` and `X_{l+1} = FFN(Norm(Y_l)) + Norm(Y_l)`.
   Attention has no outer pre-norm under hybrid; Q, K, V are each normed
   inside `OplmAttention` (V via `v_norm`, auto-constructed when
   `norm_strategy == "hybrid"`). On the FFN side, `Norm(h)` is computed once
   and reused as both the FFN input and the residual stream. See §5.3 and
   §6.2 of the architecture doc and §9.2 of this plan.

4. **`Sequence classification loss for `num_labels == 1`**: HF convention is
   to switch to MSE when `problem_type == "regression"` (auto-inferred from
   `num_labels == 1`). The minimum-viable implementation can stick to
   `cross_entropy`; matching HF semantics is a small follow-up.

5. **ESM-C tokenizer parity**: confirm the JSON construction reproduces ESM-C
   token IDs exactly by running the parity test in Phase 11.3 against an
   actual `esm` install before declaring the tokenizer done.

---

## Suggested implementation order recap

```
Phase 0  → scaffolding
Phase 1  → outputs.py
Phase 2  → norm.py            (no deps)
Phase 3  → masking.py         (no deps)
Phase 4  → rope.py            (no deps beyond torch)
Phase 5  → embedding.py       (deps: norm)
Phase 6  → ffn.py             (no deps)
Phase 7  → conv.py            (deps: masking)
Phase 8  → attention.py       (deps: norm, rope, masking)
Phase 9  → transformer.py     (deps: attention, ffn, norm, conv)
Phase 10 → configuration_oplm.py (deps: ffn.round_up_to, conv.resolve_canon_kernel_sizes)
Phase 11 → tokenization_oplm.py  (deps: none beyond transformers/tokenizers)
Phase 12 → modeling_oplm.py   (deps: everything above)
Phase 13 → package __init__   (deps: all classes exist)
Phase 14 → integration tests
Phase 15 → docs
```

Each phase ends with the per-component test file landing green before moving to
the next. The Phase 14 tests are the safety net that catches cross-component
regressions.
