# OPLM Architecture

A modern encoder-only protein language model combining architectural innovations from
**Proust** (GQA, shared K/V projections, cross-layer value residuals, depthwise
convolutions) and **Attention Residuals** (learned depth-wise residual connections),
adapted for bidirectional protein sequence modeling.

**Design goals:**

- **Ablation-friendly**: Every novel feature is togglable via config with zero memory
  overhead when disabled.
- **Production-scale**: Targets 10-15B parameters across 96-192 GPUs via
  HuggingFace Accelerate / FSDP.
- **Encoder-native**: All components are adapted for bidirectional attention (no causal
  masking, no KV cache, symmetric convolutions).

**Reference implementations:**

- Proust: [arxiv:2602.01845](https://arxiv.org/abs/2602.01845),
  [code](https://github.com/Furkan9015/proust-inference)
- Attention Residuals: [arxiv:2603.15031](https://arxiv.org/abs/2603.15031),
  [code](https://github.com/MoonshotAI/Attention-Residuals)

---

## Table of Contents

1. [Module Layout](#1-module-layout)
2. [Configuration System](#2-configuration-system)
3. [Model Components](#3-model-components)
   - [3.1 RMSNorm](#31-rmsnorm)
   - [3.2 Rotary Embeddings](#32-rotary-embeddings)
   - [3.3 Token Embedding](#33-token-embedding)
   - [3.4 Bidirectional Depthwise Convolution](#34-bidirectional-depthwise-convolution)
   - [3.5 Attention](#35-attention)
   - [3.6 Feed-Forward Network](#36-feed-forward-network)
   - [3.7 Block Attention Residuals](#37-block-attention-residuals)
   - [3.8 Value Embeddings](#38-value-embeddings)
4. [Transformer Assembly](#4-transformer-assembly)
5. [Tokenizer](#5-tokenizer)
6. [CLI and Training Stub](#6-cli-and-training-stub)
7. [Encoder Adaptations from Proust](#7-encoder-adaptations-from-proust)
8. [Ablation Toggle Reference](#8-ablation-toggle-reference)
9. [Dependencies](#9-dependencies)
10. [Implementation Order](#10-implementation-order)
11. [Verification Plan](#11-verification-plan)

---

## 1. Module Layout

```
src/oplm/
├── __init__.py                 # version only
├── __main__.py                 # `python -m oplm` → cli.app()
├── cli.py                      # typer CLI: train, encode, info
├── config.py                   # structured OmegaConf configs + load_config()
├── train.py                    # stub for `accelerate launch -m oplm.train`
├── model/
│   ├── __init__.py             # exports OplmModel, OplmForMLM
│   ├── norm.py                 # RMSNorm
│   ├── rope.py                 # RotaryEmbedding, PartialRotaryEmbedding
│   ├── embedding.py            # TokenEmbedding (scaled, optional post-embed norm)
│   ├── conv.py                 # BidirectionalDepthwiseConv
│   ├── attention.py            # GQA attention (all optional sub-features)
│   ├── ffn.py                  # FFN (swiglu / relu_squared / gelu)
│   ├── residual.py             # BlockAttentionResidual
│   └── transformer.py          # TransformerBlock, OplmEncoder, MLMHead, OplmForMLM
├── data/
│   ├── __init__.py
│   └── tokenizer.py            # ESM-compatible protein tokenizer
└── configs/
    └── model/
        ├── base.yaml           # default ModelConfig values
        └── presets/
            ├── small.yaml      # ~25M  (256d / 6L / 4H)  — fast ablation
            ├── medium.yaml     # ~150M (768d / 12L / 12H) — standard ablation
            ├── base.yaml       # ~350M (1024d / 24L / 16H) — Proust-scale
            ├── large.yaml      # ~3B   (2560d / 32L / 32H)
            └── xlarge.yaml     # ~15B  (5120d / 40L / 40H) — production
```

Additionally, a project-root `configs/` directory (not inside the package) will hold
experiment-specific overrides. The in-package configs are loaded via
`importlib.resources`; project-root configs are loaded by path.

---

## 2. Configuration System

**File: `src/oplm/config.py`**

Three nested dataclasses composed into a root config:

```python
@dataclass
class ModelConfig:
    # ── Core dimensions ──────────────────────────────────────────────
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4           # == num_heads → MHA; < num_heads → GQA
    head_dim: int | None = None     # derived: hidden_dim // num_heads
    ffn_dim: int | None = None      # derived: see below
    ffn_activation: str = "swiglu"  # "swiglu" | "relu_squared" | "gelu"
    vocab_size: int = 33
    max_seq_len: int = 2048

    # ── Attention features ───────────────────────────────────────────
    shared_kv: bool = False              # K and V share a single projection
    qk_norm: bool = True                 # RMSNorm on Q and K before attention
    output_gate: bool = False            # sigmoid gate on attention output
    query_dependent_gate: bool = False   # gate is per-token (vs static per-head)
    post_sdpa_norm: bool = False         # RMSNorm after attention computation

    # ── Positional encoding ──────────────────────────────────────────
    rope_theta: float = 10000.0
    partial_rope: bool = False           # NoPE/RoPE split (Proust GQA-S2)
    nope_dim: int | None = None          # derived: head_dim - rope_dim
    rope_dim: int | None = None          # derived: 32 if partial_rope else head_dim

    # ── Cross-layer value residuals (Proust) ─────────────────────────
    value_residual: bool = False
    value_residual_lambda_init: float = 0.5

    # ── Value embeddings (Proust) ────────────────────────────────────
    num_value_embeds: int = 0            # 0 = disabled; N = first/last N layers
    value_embed_gate_dim: int = 16       # input dims for value embed gating

    # ── Depthwise convolutions ───────────────────────────────────────
    conv_positions: str = ""             # "" | "A" | "C" | "AC" | "ACD" etc.
    conv_kernel_size: int = 7            # must be odd (bidirectional)
    conv_activation: bool = True         # SiLU activation in conv

    # ── Attention residuals (depth-wise, Kimi) ───────────────────────
    attn_residual: bool = False
    attn_residual_block_size: int = 8

    # ── Normalization ────────────────────────────────────────────────
    norm_eps: float = 1e-6
    post_embed_norm: bool = False

    # ── Training features ────────────────────────────────────────────
    gradient_checkpointing: bool = False
    tie_embeddings: bool = False
    dtype: str = "bfloat16"


@dataclass
class TrainConfig:
    # Stub — minimal fields for CLI parsing
    lr: float = 1e-4
    batch_size: int = 32
    max_steps: int = 100_000
    seed: int = 42
    output_dir: str = "outputs"
    config_path: str | None = None


@dataclass
class DataConfig:
    # Stub
    dataset: str = ""
    max_length: int = 1024
    mask_prob: float = 0.15


@dataclass
class OplmConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
```

### Derived fields

Computed in `ModelConfig.__post_init__()`:

| Field | Default derivation |
|-------|--------------------|
| `head_dim` | `hidden_dim // num_heads` |
| `ffn_dim` (SwiGLU) | `round_multiple(8/3 * hidden_dim, 256)` |
| `ffn_dim` (ReLU²/GELU) | `4 * hidden_dim` |
| `rope_dim` | `32` if `partial_rope` else `head_dim` |
| `nope_dim` | `head_dim - rope_dim` |

### Validation rules

- `num_heads % num_kv_heads == 0`
- `hidden_dim % num_heads == 0`
- If `partial_rope`: `nope_dim + rope_dim == head_dim`
- `conv_kernel_size` is odd
- All chars in `conv_positions` are in `{"A", "C", "D"}`
- `attn_residual_block_size` divides `num_layers` evenly
- `ffn_activation` is one of `"swiglu"`, `"relu_squared"`, `"gelu"`

### Config loading pipeline

`load_config(argv: list[str]) -> OplmConfig`:

1. Create `OmegaConf.structured(OplmConfig)` as the base schema (provides defaults
   and type info).
2. If `--config <path>` is present in argv, load the YAML file and merge it over the
   base with `OmegaConf.merge()`.
3. Remaining argv entries are treated as dotlist overrides
   (e.g., `model.num_layers=32 model.value_residual=true`) and merged via
   `OmegaConf.from_dotlist()`.
4. Convert to a plain Python object via `OmegaConf.to_object()`, which returns nested
   dataclass instances.
5. `__post_init__()` runs on conversion, computing derived fields and raising
   `ValueError` on invalid combinations.

This gives full type safety, YAML serialization, and Hydra-style CLI overrides
without requiring Hydra itself.

### Compatibility with `accelerate launch`

Since `accelerate launch` passes extra args through to the launched script, the
pattern is:

```bash
accelerate launch -m oplm.train --config configs/my_run.yaml model.num_layers=32
```

`oplm.train` receives `["--config", "configs/my_run.yaml", "model.num_layers=32"]`
in `sys.argv[1:]` and passes them to `load_config()`.

---

## 3. Model Components

### 3.1 RMSNorm

**File: `model/norm.py`**

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Uses torch.nn.functional.rms_norm (fused by torch.compile).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None: ...

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
```

Single learnable scale parameter per dimension. No bias. Used throughout the model:
pre-attention norm, pre-FFN norm, Q/K norms, post-SDPA norm, final output norm,
and as key normalization in attention residuals.

---

### 3.2 Rotary Embeddings

**File: `model/rope.py`**

#### `RotaryEmbedding`

Standard RoPE applied to Q and K only (not V — see
[Section 7](#7-encoder-adaptations-from-proust) for rationale).

- **Initialization**: precomputes inverse frequencies `inv_freq = 1 / (theta^(2i/d))`
  and cos/sin cache up to `max_seq_len`.
- **Input modes**:
  - Padded batch: `(B, T, H, D)` with implicit `position_ids = 0..T-1`
  - Explicit positions: `position_ids: Tensor` of shape `(B, T)` or `(total_tokens,)`
    for packed sequences
- **Core rotation**:
  ```python
  q_rot = q * cos + rotate_half(q) * sin
  k_rot = k * cos + rotate_half(k) * sin
  ```
  where `rotate_half(x)` splits the last dimension in half and swaps with negation:
  `[-x2, x1]`.

#### `PartialRotaryEmbedding`

For GQA-S2 mode (`partial_rope=True`). Splits head dimensions into position-invariant
(NoPE) and position-dependent (RoPE) portions.

- **Split**: head_dim = `nope_dim` (default 96) + `rope_dim` (default 32)
- **Selective application**: RoPE applied only to the `rope_dim` portion of Q and K
  ```python
  q_rope = q[..., nope_dim:]
  q_out = cat([q[..., :nope_dim], q_rope * cos + rotate_half(q_rope) * sin], dim=-1)
  ```
- **No inverse RoPE on output**: Unlike Proust, we omit the VO-RoPE (inverse rotation
  on attention output). This is a KV-cache optimization irrelevant to bidirectional
  encoders. See [Section 7](#7-encoder-adaptations-from-proust).

The transformer block selects the appropriate embedding at init time based on the
`partial_rope` config flag.

---

### 3.3 Token Embedding

**File: `model/embedding.py`**

#### `TokenEmbedding`

```python
class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.scale = math.sqrt(config.hidden_dim)
        self.post_norm = RMSNorm(config.hidden_dim) if config.post_embed_norm else None

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids) * self.scale
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x
```

Scaling by `sqrt(hidden_dim)` follows Proust convention (prevents embedding magnitude
from being dwarfed by hidden state magnitudes in deeper layers).

---

### 3.4 Bidirectional Depthwise Convolution

**File: `model/conv.py`**

```python
class BidirectionalDepthwiseConv(nn.Module):
    """Depthwise Conv1d with symmetric (bidirectional) padding.

    Adaptation of Proust's Canon layers for the encoder: uses "same" padding
    instead of causal (left-only) padding, and odd kernel sizes for symmetry.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        activation: bool = True,
    ) -> None:
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            groups=dim,           # depthwise
            padding=kernel_size // 2,  # symmetric "same" padding
        )
        self.act = nn.SiLU() if activation else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        return self.act(self.conv(x.transpose(1, 2)).transpose(1, 2))
```

**Design notes:**

- Kernel size 7 gives +/- 3 positions of local context (the bidirectional analog of
  Proust's 4-wide causal receptive field).
- Depthwise: `groups=dim` means each channel has its own kernel — the layer has only
  `dim * kernel_size` parameters (not `dim^2`).
- No custom CUDA kernels needed. The `causal-conv1d` kernel only supports causal
  padding. Standard PyTorch `Conv1d` with `torch.compile` is efficient for
  bidirectional.
- SiLU activation is optional (configurable via `conv_activation`).

**Placement** is governed by `conv_positions` config string:

| Position | Where applied | Rationale |
|----------|--------------|-----------|
| `"A"` | Before attention (after pre-attention norm) | Local context before global attention |
| `"C"` | Before FFN (after pre-FFN norm) | Local mixing before pointwise expansion |
| `"D"` | Inside FFN (in expanded space, between activation and down-projection) | High-dimensional local mixing |

Modules are only instantiated for positions that appear in the config string.

---

### 3.5 Attention

**File: `model/attention.py`**

A single `Attention` class supports all attention modes via config flags. All optional
sub-features are resolved at `__init__` time (no dynamic branching in forward).

```python
class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None: ...

    def forward(
        self,
        x: Tensor,                          # (B, T, D)
        v_first: Tensor | None = None,      # for cross-layer value residual
        attention_mask: Tensor | None = None,
        value_embed: Tensor | None = None,  # from ValueEmbedding
        need_weights: bool = False,         # return per-head attention weights
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        # (output, v_first_out, attn_weights)
```

#### Projection strategy

Resolved at init:

- **Standard GQA** (`shared_kv=False`):
  - `q_proj: Linear(hidden_dim → num_heads * head_dim)`
  - `k_proj: Linear(hidden_dim → num_kv_heads * head_dim)`
  - `v_proj: Linear(hidden_dim → num_kv_heads * head_dim)`
- **Shared KV** (`shared_kv=True`):
  - `q_proj: Linear(hidden_dim → num_heads * head_dim)`
  - `kv_proj: Linear(hidden_dim → num_kv_heads * head_dim)` — output used as both K
    and V

All projections are bias-free.

#### Forward flow

```
1.  Project Q, K, V (shared or separate)
2.  Reshape: (B, T, D) → (B, T, H, D_head) for Q; (B, T, KV_H, D_head) for K, V
3.  [If qk_norm]   Apply RMSNorm to Q and K independently
4.  Apply RoPE to Q and K (standard or partial, based on config)
5.  [If GQA]        Expand K, V: repeat_interleave(gqa_ratio, dim=2)
6.  [If value_embed] v = v + gate * value_embed
7.  [If value_residual and layer_idx > 0]
        λ_v, λ_first = sigmoid(self.value_lambda)    # learned per-layer
        v = λ_v * v + λ_first * v_first
    [If value_residual and layer_idx == 0]
        v_first_out = v                               # store for later layers
8.  Compute attention:
        [If need_weights]  Manual: softmax(Q·K^T / √d + mask) · V → attn_weights (B, H, T, T)
        [Else]             FlashAttention or SDPA (causal=False)
9.  [If post_sdpa_norm] Apply RMSNorm to attention output
10. [If output_gate]
        gate = sigmoid(gate_params or gate_proj(q_features))
        output = gate * attn_output
11. Output projection: Linear(num_heads * head_dim → hidden_dim)
12. Return (output, v_first_out, attn_weights)
```

#### FlashAttention fallback chain

Backend selected once at import time (try/except), not per forward call:

1. **`flash_attn.flash_attn_func`** with `causal=False` — optimal for bidirectional
2. **`F.scaled_dot_product_attention`** — PyTorch native SDPA (uses FlashAttention
   or memory-efficient backend internally based on input shapes)

#### Attention weight extraction

When `need_weights=True` is passed to `forward()`, the module bypasses
FlashAttention/SDPA and computes attention manually:
`softmax(Q @ K^T / sqrt(d_k) + mask) @ V`. This returns per-head attention weights
of shape `(B, H, T, T)` as the third element of the return tuple.

This is a per-call runtime flag (not a config toggle) since it is typically False
during training and True during evaluation for metrics like precision@L. When False
(the default), the optimized FlashAttention/SDPA path is used with zero overhead.

#### GQA mechanics

When `num_kv_heads < num_heads`, the GQA ratio is `num_heads // num_kv_heads`. K and
V tensors are expanded via `repeat_interleave`:

```python
if self.gqa_ratio > 1:
    k = k.repeat_interleave(self.gqa_ratio, dim=2)  # (B, T, KV_H, D) → (B, T, H, D)
    v = v.repeat_interleave(self.gqa_ratio, dim=2)
```

When `num_kv_heads == num_heads`, this is standard multi-head attention with no
expansion overhead.

#### Output gating

When `output_gate=True`:

- **Static gate** (`query_dependent_gate=False`): learnable parameter per head,
  `nn.Parameter(torch.zeros(num_heads))` → `sigmoid(gate)` applied to attention output
- **Query-dependent gate** (`query_dependent_gate=True`): linear projection from
  hidden state → per-token, per-head gate values

#### Cross-layer value residuals

When `value_residual=True`:

- Layer 0: stores its V as `v_first`, passes it through subsequent layers
- Layers 1+: each has a learned 2-element parameter
  `value_lambda = nn.Parameter([init, -init])`. After sigmoid, these become mixing
  weights for current V and first-layer V:
  ```python
  λ_v, λ_first = sigmoid(self.value_lambda)
  v = λ_v * v + λ_first * v_first
  ```
- `v_first` is passed through the full model via the forward return value (not stored
  as module state, keeping the module stateless).

---

### 3.6 Feed-Forward Network

**File: `model/ffn.py`**

```python
class FFN(nn.Module):
    """Feed-forward network with configurable activation."""

    def __init__(self, config: ModelConfig) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
```

Three activation variants, selected at init:

#### SwiGLU (default)

```
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

- `gate_proj: Linear(hidden_dim → ffn_dim)` — gating pathway
- `up_proj: Linear(hidden_dim → ffn_dim)` — value pathway
- `down_proj: Linear(ffn_dim → hidden_dim)` — output
- `ffn_dim = round_multiple(8/3 * hidden_dim, 256)` — compensates for the extra
  gate projection so total FFN parameters are comparable to a 4x expansion

#### ReLU squared

```
output = down_proj(relu(up_proj(x))^2)
```

- `up_proj: Linear(hidden_dim → ffn_dim)`
- `down_proj: Linear(ffn_dim → hidden_dim)`
- `ffn_dim = 4 * hidden_dim`
- This is the Proust default. The squaring operation sharpens the activation
  landscape.

#### GELU

```
output = down_proj(gelu(up_proj(x)))
```

Same structure as ReLU squared but with GELU activation. Standard BERT/ESM choice.

#### Optional Canon-D convolution

If `"D"` appears in `conv_positions`, a `BidirectionalDepthwiseConv` is applied in
the expanded space between the activation and the down projection:

```
output = down_proj(conv_d(activation(up_proj(x))))
```

---

### 3.7 Block Attention Residuals

**File: `model/residual.py`**

Replaces fixed residual connections (`x + sublayer(x)`) with learned, input-dependent
softmax attention over depth. Based on the Kimi team's Attention Residuals paper,
adapted to block-level granularity for memory efficiency.

#### Core concept

Standard residuals accumulate all layer outputs with equal weight:

```
h_l = v_0 + v_1 + ... + v_{l-1}       (fixed unit weights)
```

Block Attention Residuals learn selective aggregation:

```
h_l = Σ_i  α_{i→l} · v_i              (learned softmax weights)

where α_{i→l} = softmax_i(w_l^T · RMSNorm(v_i))
```

Each layer has a pseudo-query `w_l` (a d-dimensional learned vector). The attention
is over the depth dimension (number of blocks), not the sequence dimension.

#### Block structure

Layers are partitioned into blocks of size `attn_residual_block_size`. Within each
block, standard residual accumulation builds the block representation. Across blocks,
learned attention selects which block representations to use.

With `num_layers=24` and `block_size=8`, there are 3 blocks. Each layer attends over
at most 4 entries (3 completed blocks + current partial sum). This is very cheap
compared to full AttnRes (which would attend over all 24 previous layer outputs).

#### State tracking

```python
@dataclass
class BlockAttentionResidualState:
    blocks: list[Tensor]          # completed block representations: (B, T, D) each
    partial_block: Tensor | None  # current intra-block accumulation
    step_count: int               # tracks block boundaries
```

The token embedding is the first entry in `blocks` (the "zeroth block").

#### Module structure

```python
class BlockAttentionResidual(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        n = 2 * config.num_layers  # two applications per layer (before attn, before FFN)
        self.pseudo_queries = nn.ParameterList([
            nn.Parameter(torch.randn(config.hidden_dim))
            for _ in range(n)
        ])
        self.key_norms = nn.ModuleList([
            RMSNorm(config.hidden_dim) for _ in range(n)
        ])

    def aggregate(
        self,
        state: BlockAttentionResidualState,
        step_idx: int,
    ) -> Tensor:
        """Compute depth-wise attention over block representations.

        Args:
            state: Current block accumulation state.
            step_idx: Index into pseudo_queries/key_norms (0..2*num_layers-1).

        Returns:
            Aggregated hidden state: (B, T, D).
        """
        entries = state.blocks + ([state.partial_block] if state.partial_block is not None else [])
        V = torch.stack(entries)                            # (N, B, T, D)
        K = self.key_norms[step_idx](V)                     # (N, B, T, D)
        w = self.pseudo_queries[step_idx]                    # (D,)
        logits = torch.einsum("d, n b t d -> n b t", w, K)  # (N, B, T)
        weights = logits.softmax(dim=0)                      # (N, B, T)
        return torch.einsum("n b t, n b t d -> b t d", weights, V)  # (B, T, D)
```

#### Integration with TransformerBlock

AttnRes is applied **twice per layer**: before the attention sublayer and before the
FFN sublayer. This matches the paper's pseudocode exactly.

```
# Before attention sublayer:
h = aggregate(state, step_idx=2*layer_idx)
attn_out = attention(norm(h))
state.partial_block = state.partial_block + attn_out  # (or just attn_out if None)

# Check block boundary (between attention and FFN or after FFN, depending on config)
if at_block_boundary:
    state.blocks.append(state.partial_block)
    state.partial_block = None

# Before FFN sublayer:
h = aggregate(state, step_idx=2*layer_idx + 1)
ffn_out = ffn(norm(h))
state.partial_block = state.partial_block + ffn_out
```

Block boundaries are determined by `layer_idx % (block_size // 2) == 0` for the
inter-attention boundary, matching the paper's convention where each "step" in a block
is either an attention or FFN sublayer.

---

### 3.8 Value Embeddings

**File: `model/embedding.py`** (alongside `TokenEmbedding`)

```python
class ValueEmbedding(nn.Module):
    """Separate embedding tables injected into V at selected layers.

    From Proust: applied symmetrically to the first N and last N layers,
    where N = num_value_embeds.
    """

    def __init__(self, config: ModelConfig) -> None:
        kv_dim = config.num_kv_heads * (config.head_dim or config.hidden_dim // config.num_heads)
        self.embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size, kv_dim)
            for _ in range(config.num_value_embeds)
        ])
        self.gates = nn.ModuleList([
            nn.Linear(config.value_embed_gate_dim, config.num_kv_heads)
            for _ in range(config.num_value_embeds)
        ])
        # Build layer → embed index mapping
        n = config.num_value_embeds
        self.layer_map: dict[int, int] = {}
        for i in range(n):
            self.layer_map[i] = i                              # first N layers
            self.layer_map[config.num_layers - n + i] = i      # last N layers

    def forward(self, input_ids: Tensor, x: Tensor, layer_idx: int) -> Tensor | None:
        if layer_idx not in self.layer_map:
            return None
        idx = self.layer_map[layer_idx]
        ve = self.embeds[idx](input_ids)                          # (B, T, kv_dim)
        gate_input = x[..., :self.gate_dim]                       # (B, T, gate_dim)
        gate = 2.0 * torch.sigmoid(self.gates[idx](gate_input))  # (B, T, num_kv_heads)
        return gate.unsqueeze(-1) * ve.view(*ve.shape[:-1], self.num_kv_heads, self.head_dim)
```

The gating mechanism uses the first `value_embed_gate_dim` (default 16) dimensions of
the hidden state as input, producing per-KV-head gates scaled to [0, 2]. The value
embedding is added to V inside the attention module.

---

## 4. Transformer Assembly

**File: `model/transformer.py`**

### `TransformerBlock`

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        self.attn_pre_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn_pre_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attention = Attention(config, layer_idx)
        self.ffn = FFN(config)
        self.conv_a = BidirectionalDepthwiseConv(...) if "A" in config.conv_positions else None
        self.conv_c = BidirectionalDepthwiseConv(...) if "C" in config.conv_positions else None
```

Two forward paths exist, selected at the `OplmEncoder` level (not per-call):

**Standard residual path** (`attn_residual=False`):

```
x → [conv_a(x) if enabled] → attn_pre_norm → attention → + x (residual)
  → [conv_c(x) if enabled] → ffn_pre_norm  → ffn       → + x (residual)
```

**AttnRes path** (`attn_residual=True`):

```
h = attn_res.aggregate(state, 2*i)           # depth-wise attention
h → [conv_a(h) if enabled] → attn_pre_norm → attention
state.partial_block += attn_out               # accumulate into block
[if block boundary: state.blocks.append(state.partial_block); reset]

h = attn_res.aggregate(state, 2*i+1)         # depth-wise attention
h → [conv_c(h) if enabled] → ffn_pre_norm  → ffn
state.partial_block += ffn_out                # accumulate into block
```

### `OplmEncoder`

The model backbone:

```python
class OplmEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.value_embedding = ValueEmbedding(config) if config.num_value_embeds > 0 else None
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        self.attn_residual = BlockAttentionResidual(config) if config.attn_residual else None
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        x = self.embedding(input_ids)
        v_first: Tensor | None = None
        all_attn_weights: list[Tensor] = []

        # Initialize AttnRes state with token embedding as first "block"
        state = None
        if self.attn_residual is not None:
            state = BlockAttentionResidualState(
                blocks=[x], partial_block=None, step_count=0,
            )

        for i, block in enumerate(self.blocks):
            # Get value embedding for this layer (if any)
            ve = self.value_embedding(input_ids, x, i) if self.value_embedding else None

            if state is not None:
                x, v_first, state = block.forward_with_attn_res(
                    x, v_first, attention_mask, ve, self.attn_residual, state,
                )
            else:
                x, v_first, layer_weights = block(
                    x, v_first, attention_mask, ve, need_weights,
                )
                if layer_weights is not None:
                    all_attn_weights.append(layer_weights)

        hidden = self.final_norm(x)  # (B, T, D)
        return hidden, all_attn_weights if need_weights else None
```

### `MLMHead`

```python
class MLMHead(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.activation = nn.GELU()
        self.projection = nn.Linear(config.hidden_dim, config.vocab_size)
        # If tie_embeddings, self.projection.weight = encoder.embedding.embed.weight

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = self.activation(self.norm(self.dense(hidden_states)))
        return self.projection(x)
```

### `OplmForMLM`

```python
class OplmForMLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        self.encoder = OplmEncoder(config)
        self.mlm_head = MLMHead(config)
        if config.tie_embeddings:
            self.mlm_head.projection.weight = self.encoder.embedding.embed.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        need_weights: bool = False,
    ) -> dict[str, Tensor]:
        hidden, attn_weights = self.encoder(input_ids, attention_mask, need_weights)
        logits = self.mlm_head(hidden)

        loss = None
        if labels is not None:
            # Only compute loss on masked positions (labels != -100)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        result = {"logits": logits, "loss": loss}
        if attn_weights is not None:
            result["attention_weights"] = attn_weights
        return result
```

### FSDP wrapping strategy

Each `TransformerBlock` is a natural FSDP wrapping unit. Accelerate's
`fsdp_auto_wrap_policy` can use a size-based or transformer-layer-based policy:

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)
```

### Gradient checkpointing

When `config.gradient_checkpointing=True`, each `TransformerBlock.forward()` is
wrapped with `torch.utils.checkpoint.checkpoint()`. This trades compute for memory,
critical at 10B+ scale.

---

## 5. Tokenizer

**File: `data/tokenizer.py`**

ESM-compatible protein tokenizer with 33 tokens:

```python
VOCAB = {
    "<cls>": 0, "<pad>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4,
    "L": 5, "A": 6, "G": 7, "V": 8, "S": 9,
    "E": 10, "R": 11, "T": 12, "I": 13, "D": 14,
    "P": 15, "K": 16, "Q": 17, "N": 18, "F": 19,
    "Y": 20, "M": 21, "H": 22, "W": 23, "C": 24,
    "B": 25, "U": 26, "Z": 27, "O": 28, "X": 29,
    ".": 30, "-": 31, "<null>": 32,
}
```

```python
class ProteinTokenizer:
    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]: ...
    def decode(self, token_ids: list[int] | Tensor) -> str: ...
    def batch_encode(
        self,
        sequences: list[str],
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Tensor]:
        """Returns {"input_ids": Tensor, "attention_mask": Tensor}."""
        ...

    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def mask_token_id(self) -> int: ...
    @property
    def cls_token_id(self) -> int: ...
    @property
    def eos_token_id(self) -> int: ...
```

No external dependencies (dict-based lookup, not sentencepiece/BPE).

---

## 6. CLI and Training Stub

### CLI (`cli.py` + `__main__.py`)

```python
# cli.py
import typer
from rich.console import Console

app = typer.Typer(name="oplm", help="Open Protein Language Model")
console = Console()

@app.command()
def train(
    config: str = typer.Option(None, "--config", "-c", help="Path to YAML config"),
    overrides: list[str] = typer.Argument(None, help="Config overrides (key=value)"),
) -> None:
    """Launch training. For distributed: accelerate launch -m oplm.train"""
    ...

@app.command()
def encode(
    sequences: list[str] = typer.Argument(..., help="Protein sequences to encode"),
    model_path: str = typer.Option(..., "--model", "-m"),
    output: str = typer.Option("embeddings.pt", "--output", "-o"),
) -> None:
    """Encode protein sequences to embeddings."""
    ...

@app.command()
def info(
    config: str = typer.Option(None, "--config", "-c"),
    overrides: list[str] = typer.Argument(None),
) -> None:
    """Print model config and parameter count."""
    ...
```

```python
# __main__.py
from oplm.cli import app

if __name__ == "__main__":
    app()
```

### Training stub (`train.py`)

```python
"""Training entry point.

Run directly:     python -m oplm.train --config configs/my_run.yaml
Run distributed:  accelerate launch -m oplm.train --config configs/my_run.yaml model.num_layers=32
"""
from __future__ import annotations

import sys

from oplm.config import load_config


def main() -> None:
    cfg = load_config(sys.argv[1:])
    # TODO: instantiate model, optimizer, dataloader, training loop
    raise NotImplementedError("Training loop not yet implemented")


if __name__ == "__main__":
    main()
```

---

## 7. Encoder Adaptations from Proust

Proust is a decoder-only causal model. Three key adaptations are needed for the
bidirectional encoder:

### 7.1 No VO-RoPE (inverse rotation on attention output)

**Proust behavior**: Applies RoPE to Q, K, *and* V, then applies inverse RoPE
(rotation by negative theta) to the attention output. This ensures that the output
at position `i` represents a linear combination of V vectors in a
position-independent space, which is important for KV-cache correctness in
autoregressive inference.

**Encoder adaptation**: In a bidirectional encoder with no KV cache, standard
Q/K-only RoPE is mathematically equivalent for the attention output (the relative
position information is fully captured by the Q-K dot product). We omit VO-RoPE
entirely — it adds complexity and compute for no benefit.

### 7.2 No key offset

**Proust behavior**: In GQA-S2, the NoPE portion of K is shifted forward by one
position (`k_shifted[1:, :, :nope_dim] = k[:-1, :, :nope_dim]`). This enables the
model to learn single-layer induction heads (bigram pattern detection) which are
important for next-token prediction.

**Encoder adaptation**: In bidirectional attention, every position already attends
to every other position. The key offset would break the symmetric position treatment
that the encoder relies on. We omit it.

### 7.3 Bidirectional convolutions

**Proust behavior**: Canon layers use causal Conv1d with left-only padding
(kernel=4, padding=3 on left). This ensures the output at position `t` depends only
on positions `0..t`.

**Encoder adaptation**: Symmetric padding with odd kernel sizes. The default
kernel_size=7 gives +/- 3 positions of local context. We cannot use the
`causal-conv1d` CUDA kernel (it only supports causal mode), but standard PyTorch
Conv1d with `torch.compile` is efficient.

---

## 8. Ablation Toggle Reference

Every novel feature can be independently enabled or disabled via config, with zero
memory overhead when off.

| Feature | Config field(s) | Disabled state | Effect when disabled |
|---------|----------------|----------------|---------------------|
| GQA | `num_kv_heads < num_heads` | `num_kv_heads == num_heads` | Standard MHA, no K/V expansion |
| Shared K/V | `shared_kv` | `False` | Separate K, V projections |
| Partial RoPE | `partial_rope` | `False` | Full RoPE on entire head_dim |
| Q/K norm | `qk_norm` | `False` | Raw Q, K to attention |
| Output gate | `output_gate` | `False` | No gate params, direct output |
| Post-SDPA norm | `post_sdpa_norm` | `False` | No norm after attention |
| Value residual | `value_residual` | `False` | No lambda params, V passes through |
| Value embeddings | `num_value_embeds` | `0` | No embed tables created |
| Depthwise conv | `conv_positions` | `""` | No Conv1d modules instantiated |
| Attn residuals | `attn_residual` | `False` | Standard `x + sublayer(x)` |
| FFN activation | `ffn_activation` | N/A (always one) | Switches activation fn |
| Post-embed norm | `post_embed_norm` | `False` | No norm after embedding |
| Gradient ckpt | `gradient_checkpointing` | `False` | No checkpointing |
| Tied embeddings | `tie_embeddings` | `False` | Separate MLM head weights |
| Attention weights | `need_weights` (runtime) | `False` (default) | FlashAttn/SDPA, no weight matrix |

**Zero-overhead principle**: When a feature is disabled, no `nn.Module` or
`nn.Parameter` is created for it. The forward-path check `if self.module is not None`
is a single pointer check that `torch.compile` / JIT eliminates. This ensures
ablation experiments are fair (disabled features don't consume memory or add
overhead) and production runs carry no penalty for unused features.

**Runtime flags**: `need_weights` is not a config toggle but a per-call parameter on
`Attention.forward()`. It is listed here for completeness. When False, the optimized
attention backend is used with no overhead.

---

## 9. Dependencies

```toml
[project]
dependencies = [
    "torch>=2.4.0",
    "omegaconf>=2.3",
    "typer>=0.9",
    "rich>=13.0",
]

[project.optional-dependencies]
train = [
    "accelerate>=0.30",
    "wandb>=0.16",
    "datasets>=2.18",
]
flash = [
    "flash-attn>=2.5",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.8",
    "mypy>=1.13",
]

[project.scripts]
oplm = "oplm.cli:app"
```

---

## 10. Implementation Order

### Phase 1: Foundation

1. **`config.py`** — structured configs, `load_config()`, validation, YAML I/O
2. **`model/norm.py`** — RMSNorm
3. **`model/rope.py`** — RotaryEmbedding, PartialRotaryEmbedding
4. **`model/embedding.py`** — TokenEmbedding, ValueEmbedding
5. **`data/tokenizer.py`** — ProteinTokenizer

### Phase 2: Core modules

6. **`model/conv.py`** — BidirectionalDepthwiseConv
7. **`model/ffn.py`** — FFN (SwiGLU, ReLU², GELU, optional Canon-D)
8. **`model/attention.py`** — Attention (all modes and optional features)
9. **`model/residual.py`** — BlockAttentionResidual + state tracking

### Phase 3: Assembly

10. **`model/transformer.py`** — TransformerBlock, OplmEncoder, MLMHead, OplmForMLM
11. **`model/__init__.py`** — public exports

### Phase 4: CLI & config files

12. **`cli.py`** — typer app with train, encode, info subcommands
13. **`__main__.py`** — module entry point
14. **`train.py`** — accelerate-compatible stub
15. **YAML configs** — base.yaml + size presets
16. **`pyproject.toml`** — updated dependencies and `[project.scripts]`

### Phase 5: Tests

17. **`tests/model/test_norm.py`** — RMSNorm correctness
18. **`tests/model/test_rope.py`** — rotation properties, partial RoPE
19. **`tests/model/test_attention.py`** — GQA shapes, shared KV, value residual
20. **`tests/model/test_ffn.py`** — activation variants, output shapes
21. **`tests/model/test_conv.py`** — bidirectional padding, output shapes
22. **`tests/model/test_residual.py`** — block boundaries, aggregation
23. **`tests/model/test_transformer.py`** — full forward pass, all config combos
24. **`tests/test_config.py`** — validation, merging, CLI overrides
25. **`tests/data/test_tokenizer.py`** — encode/decode roundtrip
26. **`tests/test_e2e.py`** — small model trains a few steps, loss decreases

---

## 11. Verification Plan

1. **Unit tests**: Each module tested with known input shapes and expected outputs.
2. **Ablation matrix**: Parametrize over all boolean feature flags; verify forward
   pass succeeds and produces correct output shapes for every combination.
3. **Parameter counting**: Verify that disabled features add exactly zero parameters.
4. **Gradient flow**: Verify gradients reach all parameters (no dead params) via a
   single backward pass.
5. **E2E pilot**: Instantiate the `small` preset, run 10 MLM training steps on a
   batch of real protein sequences, verify loss decreases.
6. **torch.compile**: Verify model compiles without graph breaks (at least with the
   standard feature set).
7. **Config roundtrip**: `YAML → OmegaConf → dataclass → YAML` produces identical
   output.
8. **CLI smoke test**: `oplm info`, `oplm train --help` execute without error.
