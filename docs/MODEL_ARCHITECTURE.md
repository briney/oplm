# OPLM Model Architecture

> Founding architectural reference for the OPLM model. This document specifies
> the model — its layers, math, public API, configuration schema, and HuggingFace
> integration — in enough detail to implement it directly. The trainer, eval
> harness, data pipeline, and CLI are out of scope and live in separate documents.

---

## 1. Scope and design philosophy

### 1.1 What this document covers

This document specifies the OPLM model itself. Concretely:

- The PyTorch modules under `src/oplm/model/`.
- The tokenizer (`OplmTokenizerFast`) — included here because `from_pretrained()`
  returns it alongside the model and the two are co-versioned.
- The model-facing public API: HuggingFace `Auto*` classes and the ESM-C-style
  convenience methods.
- The `OplmConfig` schema — the contract between YAML config and model.

### 1.2 What this document does not cover

- The trainer, optimizer, learning-rate schedules, FLOPs accounting,
  gradient-clipping, gradient-accumulation, checkpoint format. → future
  `docs/TRAINER.md`.
- The eval harness and downstream benchmarks. → [`EVAL_HARNESS.md`](EVAL_HARNESS.md).
- The data pipeline, dataset formats, masking strategy. → `docs/DATA_TOOLING.md`.
- The CLI. → future `docs/CLI.md`.
- The full YAML config schema beyond the `model:` block (i.e., the `train:`,
  `data:`, `eval:` blocks). → future docs above.
- Exact numerical recipes for the size presets (small/medium/large/...). → see
  `src/oplm/configs/model/presets/*.yaml`.

### 1.3 Design philosophy

OPLM is a **vanilla pre-norm encoder transformer with RoPE and QK-norm**. The
default config reproduces a textbook modern PLM: LayerNorm everywhere (RMSNorm
available as a toggle), SwiGLU FFN, untied input/output embeddings, standard
multi-head attention with `num_heads == num_kv_heads`, residual streams scaled
by `1/sqrt(L)` at each sublayer for stability at depth, and a BERT-style MLP
MLM head.

A small, curated set of **research toggles** layers on top, each switchable
independently from config:

- Canon-style bidirectional depthwise conv sublayers (arXiv 2512.17351)
- Partial RoPE / NoPE split (arXiv 2502.14837v1)
- Hybrid norm (arXiv 2503.04598)
- Sandwich norm
- Post-SDPA norm

When every toggle is off (the default), the model is the textbook baseline. The
toggles are not "configurable variants of every component" — they are specific,
named ablation hypotheses with literature backing. New ablations are added by
following the same pattern, not by extending existing toggles into open-ended
configuration knobs.

### 1.4 Explicit non-features

The following features were tried, did not yield consistent gains, and are
**removed entirely** — not even available as toggles. Re-introducing any of them
requires a deliberate revisit, not a flag flip:

| Removed | One-line rationale |
| --- | --- |
| Grouped-query attention (`num_kv_heads < num_heads`) | Modest memory savings for noticeable quality cost at our scales. |
| Attention residuals (block-level residual on attention output, Kimi-style) | No consistent quality win; added bookkeeping. |
| Cross-layer value residuals (Proust-style) | Did not reproduce gains observed in the original. |
| Output gating (static or query-dependent) | Hurt or matched baseline; added parameters. |
| Shared K/V projection | Marginal compute savings, mild quality regression. |
| Multi-token value embeddings | Did not justify the complexity. |
| Query-dependent attention gates | Subsumed by the above. |

### 1.5 Dependency and packaging policy

- **Python** ≥ 3.11.
- **`torch` ≥ 2.11** — pinned because `torch.nn.attention.flex_attention` is the
  fast-path attention kernel, and torch 2.11 is the first release with the
  FlashAttention-4 backend used on Blackwell GPUs. There is no
  separate `flash-attn` package dependency.
- Single `pip install oplm` installs everything. **No optional dependency
  groups** (no `oplm[train]`, no `oplm[eval]`, no `oplm[flash]`).
- Build backend: `hatchling`. Single `pyproject.toml`.

### 1.6 Included / toggle / dropped at a glance

| Feature | Status |
| --- | --- |
| Pre-norm | Default |
| LayerNorm | Default |
| RMSNorm | Toggle |
| RoPE (full) | Default |
| QK-norm | Default |
| Standard multi-head attention (`H == H_kv`) | Default |
| SwiGLU FFN | Default |
| BERT-style MLP MLM head | Default |
| Untied input/output embeddings | Default |
| Tied input/output embeddings | Toggle |
| Residual scaling (`1/sqrt(L)` on each sublayer) | Default |
| Init scaling on attn/FFN output projections (`1/sqrt(2L)`) | Default |
| `flex_attention` fast path | Default |
| Manual fallback when `output_attentions=True` | Default |
| Canon depthwise conv | Toggle |
| Partial RoPE / NoPE split | Toggle |
| Hybrid norm | Toggle |
| Sandwich norm | Toggle |
| Post-SDPA norm | Toggle |
| Grouped-query attention | Removed |
| Attention residuals | Removed |
| Value residuals (Proust) | Removed |
| Output gating | Removed |
| Shared K/V | Removed |
| Value embeddings | Removed |

---

## 2. Public API

OPLM exposes two complementary public surfaces: the **standard HuggingFace
interface** and a **thin ESM-C-style convenience layer** on top of it. Both
return the same underlying objects; the ESM-C layer is sugar.

### 2.1 HuggingFace surface

| HF class | Resolves to |
| --- | --- |
| `AutoConfig` | `OplmConfig` |
| `AutoTokenizer` | `OplmTokenizerFast` |
| `AutoModel` | `OplmModel` |
| `AutoModelForMaskedLM` | `OplmForMaskedLM` |
| `AutoModelForSequenceClassification` | `OplmForSequenceClassification` |
| `AutoModelForTokenClassification` | `OplmForTokenClassification` |

Standard usage:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("brineylab/oplm-base")
tokenizer = AutoTokenizer.from_pretrained("brineylab/oplm-base")

inputs = tokenizer(["MEEPQSDPSVEPPLSQ"], return_tensors="pt", padding=True)
out = model(**inputs, output_hidden_states=True, output_attentions=False)
# out.logits           : (B, T, vocab_size)
# out.hidden_states    : tuple of (B, T, D), one per layer + embedding
```

### 2.2 ESM-C-style convenience layer

All four `Oplm*ForX` classes expose three convenience methods. The first two
mirror ESM-C so switching from ESM-C to OPLM is mechanical at the call site;
the third (`tokenize`) is OPLM-specific and is the recommended way to feed the
model when you do not want to manage tokenizer state yourself.

```python
from oplm import OplmForMaskedLM, LogitsConfig

model = OplmForMaskedLM.from_pretrained("brineylab/oplm-base")
# from_pretrained auto-attaches the saved tokenizer if files are present; you
# can also assign your own: model.tokenizer = AutoTokenizer.from_pretrained(...)

# Full BatchEncoding — preferred. Contains input_ids AND attention_mask.
batch = model.tokenize(["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"])
out = model(**batch)

# ESM-C-compatible: padded input_ids only. WARNING — using `ids` without also
# passing the corresponding attention_mask will silently treat pad as real
# input. Prefer model.tokenize(...) above; see §2.3.
ids = model.encode(["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"])

# Run forward and return a structured result. Internally calls tokenize(), so
# the attention mask is always carried through correctly.
out = model.logits(
    ["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"],
    LogitsConfig(
        sequence=True,
        return_embeddings=True,
        return_hidden_states=False,
        return_attentions=False,
    ),
)
# out.sequence_logits  : (B, T, vocab_size) | None
# out.embeddings       : (B, T, D) | None    (last hidden state)
# out.hidden_states    : tuple[Tensor]  | None
# out.attentions       : tuple[Tensor]  | None  (forces fallback path)
```

If `tokenize` / `encode` / `logits` is called before a tokenizer has been
attached, the model raises with a message pointing to `from_pretrained` or
manual assignment of `model.tokenizer`. There is no lazy
`AutoTokenizer.from_pretrained(self.config._name_or_path)` fallback — that
silently breaks scratch models, offline use, and test cases where the saved
tokenizer files do not exist alongside the config.

`LogitsConfig` (defined in `src/oplm/model/outputs.py`):

```python
@dataclass
class LogitsConfig:
    sequence: bool = True
    return_embeddings: bool = False
    return_hidden_states: bool = False
    return_attentions: bool = False
```

`LogitsOutput`:

```python
@dataclass
class LogitsOutput:
    sequence_logits: Tensor | None
    embeddings: Tensor | None
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None
```

### 2.3 Method-to-HF mapping

| Method | Returns | Delegates to |
| --- | --- | --- |
| `model.tokenize(seqs, **kwargs)` | `BatchEncoding` with `input_ids` + `attention_mask` (and any extras the tokenizer adds) | `self.tokenizer(seqs, return_tensors="pt", padding=True, **kwargs).to(device)` |
| `model.encode(seqs, **kwargs)` | `Tensor` of `input_ids` only — for ESM-C parity | `self.tokenize(seqs, **kwargs).input_ids` |
| `model.logits(seqs, cfg)` | `LogitsOutput` | `self.tokenize(seqs)` → `self.forward(input_ids, attention_mask, output_hidden_states=..., output_attentions=...)` → repackage |

The convenience layer adds **one** piece of model-attached state — a
`self.tokenizer` reference (defaults to `None`). Everything else is a
tokenize-then-forward wrapper. Users who want full control should call
`forward()` directly.

`encode()` is kept for ESM-C call-site parity but is a known footgun in
isolation: it returns the padded `input_ids` tensor with no attention mask. The
recommended call pattern is `model.tokenize(...)` (full BatchEncoding) or
`model.logits(...)` (built-in mask plumbing).

### 2.4 Token-classification example

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "brineylab/oplm-base", num_labels=3
)
tokenizer = AutoTokenizer.from_pretrained("brineylab/oplm-base")
inputs = tokenizer(["MEEPQSDPSV"], return_tensors="pt")
out = model(**inputs)
# out.logits : (B, T, num_labels)
```

---

## 3. Tokenizer

OPLM uses the **ESM-C 33-token vocabulary**, in the same token order and with
the same special-token IDs (matches `esm.utils.constants.esm3` from
`evolutionaryscale/esm`). A batch tokenized for ESM-C is bit-for-bit identical
to one tokenized for OPLM, which makes switching between the two models
mechanical at the data layer. Note the id-31 slot that was `<null_1>` in
ESM-2 is `|` (chain break) in ESM-C; OPLM follows ESM-C.

### 3.1 Vocabulary

| ID | Token | Kind |
| --- | --- | --- |
| 0 | `<cls>` | special (BOS) |
| 1 | `<pad>` | special |
| 2 | `<eos>` | special |
| 3 | `<unk>` | special |
| 4 | `L` | amino acid |
| 5 | `A` | amino acid |
| 6 | `G` | amino acid |
| 7 | `V` | amino acid |
| 8 | `S` | amino acid |
| 9 | `E` | amino acid |
| 10 | `R` | amino acid |
| 11 | `T` | amino acid |
| 12 | `I` | amino acid |
| 13 | `D` | amino acid |
| 14 | `P` | amino acid |
| 15 | `K` | amino acid |
| 16 | `Q` | amino acid |
| 17 | `N` | amino acid |
| 18 | `F` | amino acid |
| 19 | `Y` | amino acid |
| 20 | `M` | amino acid |
| 21 | `H` | amino acid |
| 22 | `W` | amino acid |
| 23 | `C` | amino acid |
| 24 | `X` | ambiguous AA |
| 25 | `B` | ambiguous AA (D/N) |
| 26 | `U` | selenocysteine |
| 27 | `Z` | ambiguous AA (E/Q) |
| 28 | `O` | pyrrolysine |
| 29 | `.` | gap marker |
| 30 | `-` | alignment gap |
| 31 | `\|` | chain break |
| 32 | `<mask>` | MLM mask |

### 3.2 Tokenization rules

- Per-character splitting of the input string against the AA alphabet (no BPE,
  no SentencePiece).
- Each input is wrapped: `[<cls>, t1, t2, ..., tN, <eos>]`.
- Unknown single-character tokens map to `<unk>` (id 3). `<unk>` should be rare
  on real protein input; appearing frequently is a data-cleaning bug.
- Padding to a batch-wide max length uses `<pad>` (id 1).
- Truncation, when requested, drops AA tokens but preserves `<cls>` and `<eos>`.
- No casing transformations — protein sequence input is expected uppercase.

### 3.3 HuggingFace integration

`OplmTokenizerFast` subclasses `transformers.PreTrainedTokenizerFast`. The
backing fast tokenizer is constructed from a small `tokenizers.Tokenizer` JSON
that encodes the per-character WordLevel vocabulary above plus the
`TemplateProcessing` rule that adds `<cls> ... <eos>` wrapping.

`save_pretrained` writes:

- `tokenizer.json` — the fast tokenizer
- `tokenizer_config.json` — `tokenizer_class: "OplmTokenizerFast"`
- `special_tokens_map.json` — explicit special-token assignments

`AutoTokenizer.from_pretrained("brineylab/oplm-base")` resolves
`tokenizer_class` and returns an `OplmTokenizerFast`. The class is registered
with `AutoTokenizer` in `oplm/__init__.py` (see §13).

### 3.4 Example

Input: `"MEEPQ"`.

Tokens: `[<cls>, M, E, E, P, Q, <eos>]`.

IDs: `[0, 20, 9, 9, 14, 16, 2]`.

This sequence of IDs is **byte-identical** to what `EsmTokenizer` produces for
the same input.

---

## 4. Model class hierarchy

```
OplmConfig(PretrainedConfig)
    └─ model_type = "oplm"

OplmPreTrainedModel(PreTrainedModel)
    ├─ config_class = OplmConfig
    ├─ base_model_prefix = "oplm"
    ├─ _no_split_modules = ["OplmBlock"]
    ├─ _supports_sdpa = True
    └─ (handles init, weight tying)

    ├─ OplmModel(OplmPreTrainedModel)
    │       └─ backbone — embeddings + N × OplmBlock + final norm
    │          returns BaseModelOutput
    │
    ├─ OplmForMaskedLM(OplmPreTrainedModel)
    │       └─ OplmModel + MLM head (tied with input embeddings)
    │          returns MaskedLMOutput
    │
    ├─ OplmForSequenceClassification(OplmPreTrainedModel)
    │       └─ OplmModel + pooler + classification head
    │          returns SequenceClassifierOutput
    │
    └─ OplmForTokenClassification(OplmPreTrainedModel)
            └─ OplmModel + per-token classification head
               returns TokenClassifierOutput
```

| Class | Responsibility |
| --- | --- |
| `OplmConfig` | Carries every model hyperparameter. Subclass of `PretrainedConfig`. Validates field combinations in `__post_init__` / `__init__`. |
| `OplmPreTrainedModel` | Abstract base. Wires `from_pretrained` / `save_pretrained`, weight init, embedding tying, gradient checkpointing toggle. Never instantiated directly. |
| `OplmModel` | The encoder backbone. Token embedding → stack of `OplmBlock`s → final norm. No task head. |
| `OplmForMaskedLM` | Wraps `OplmModel`, adds the MLM projection (tied to embedding by default), computes MLM cross-entropy loss. |
| `OplmForSequenceClassification` | Wraps `OplmModel`, applies a pooler (mean over non-pad by default) and a linear classifier. |
| `OplmForTokenClassification` | Wraps `OplmModel`, applies a per-token linear classifier. |

All public `Oplm*` classes live in `src/oplm/model/modeling_oplm.py`. The
ESM-C-style `encode` and `logits` methods are defined on a small mixin
(`EsmcCompatMixin`) that the four task classes inherit from.

The HF auto-class registration (see §13) maps:

| HF Auto class | OPLM class |
| --- | --- |
| `AutoConfig` | `OplmConfig` |
| `AutoModel` | `OplmModel` |
| `AutoModelForMaskedLM` | `OplmForMaskedLM` |
| `AutoModelForSequenceClassification` | `OplmForSequenceClassification` |
| `AutoModelForTokenClassification` | `OplmForTokenClassification` |
| `AutoTokenizer` | `OplmTokenizerFast` |

---

## 5. Transformer block

The block (`OplmBlock`) is the repeating unit of the encoder stack. The default
configuration is pre-norm with LayerNorm, residual scaling enabled, no Canon
convs, and full RoPE. Throughout this section "Norm" refers generically to the
configured `norm_type` (LayerNorm by default; RMSNorm under
`norm_type = "rmsnorm"`). See §9 for the math of each.

### 5.1 Default block

```
x ─────────────────────────────────────────────────────────────────┐
  │                                                                │
  ├─► Norm ──► Attn (QK-norm + RoPE) ──► × (1/sqrt(L)) ──► (+) ────┤
  │                                                                │
  │                                                                ▼
  │                                                                h
  │                                                                │
  ├─► Norm ──► SwiGLU FFN ─────────────► × (1/sqrt(L)) ──► (+) ────┤
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
                                                                   │
                                                                   ▼
                                                                   y
```

Algebraically, with `α = 1 / sqrt(L)` when `residual_scaling = "sqrt_num_layers"`
(default) and `α = 1` when `residual_scaling = "none"`:

```
h = x + α · Attn(Norm(x))
y = h + α · FFN(Norm(h))
```

The final block's output is followed by one more Norm (the "final norm") before
being fed to the task head.

**Why scale.** With residual scaling on, the contribution of each sublayer to
the residual stream is `O(1/sqrt(L))`, so the total contribution from `2L`
sublayers stays `O(sqrt(L))` rather than `O(L)`. This is the load-bearing
stability tool for deep models (target depths in OPLM go up to ~80 layers). The
literal divisor in the spec is `sqrt(L)` per the project owner's preference; an
equally defensible alternative is `sqrt(2L)` (since there are `2L` sublayers).
The two differ only by a constant `sqrt(2)` and can be revisited at scale-up.

### 5.2 Canon insertion points

When `canon_enabled = True`, depthwise 1D convolutions are inserted at one or
more of the following positions (labels match the Canon paper, arXiv
2512.17351):

```
       (A)                         (B)                                 (C)
x ──► [Conv] ──► Norm ──► [Conv] ──► Attn(QK-norm+RoPE) ──► [Conv] ──► (+) ──► h
                                                                                  │
                                                                                  │
                                              (D)                                 │
                                             [Conv]                               │
                                               │                                  │
h ──► Norm ──► SwiGLU FFN ────────────────────────────────────────────────► (+) ──┘
                                                                                  │
                                                                                  ▼
                                                                                  y
```

- **A** — pre-block, on the residual-stream input.
- **B** — between the attention pre-norm and the attention itself.
- **C** — on the attention output before the residual add.
- **D** — between the FFN pre-norm and the FFN.

`canon_positions` is a list-valued config field, e.g. `["A", "C", "D"]`.
Position **B** is supported but discouraged (the paper reports it as the
weakest); we accept it for ablation completeness.

A block with Canon at A+C+D looks like:

```
x ──► [Conv_A] ──► Norm ──► Attn ──► [Conv_C] ──► (+) ──► h
                                                           │
h ──► Norm ──► [Conv_D] ──► SwiGLU FFN ───────────► (+) ──► y
```

See §11 for the conv operator spec.

### 5.3 Normalization placement strategies

`norm_strategy` controls how Norm is placed around each sublayer. Residual
scaling (`α = 1/sqrt(L)` by default) composes orthogonally with placement.

| `norm_strategy` | Attention sublayer | FFN sublayer |
| --- | --- | --- |
| `pre` (default) | `h = x + α · Attn(Norm(x))` | `y = h + α · FFN(Norm(h))` |
| `sandwich` | `h = x + α · Norm₂(Attn(Norm₁(x)))` | `y = h + α · Norm₄(FFN(Norm₃(h)))` |
| `hybrid` (arXiv 2503.04598) | `h = x + α · Attn_QKVNorm(x)` — **no outer pre-norm**; Q, K, V are each normed inside the attention module (see §6.2) | `y = Norm(h) + α · FFN(Norm(h))` — a single `Norm(h)` is reused as both the FFN input and the FFN-side residual stream |
| `post_sdpa` | `h = x + α · Norm(Attn(Norm(x)))` (Norm only on attn output, FFN unchanged) | `y = h + α · FFN(Norm(h))` |

The hybrid row reproduces the paper's "QKV-Post" main method literally:
`Y_l = MHA_QKV(X_l) + X_l` followed by `X_{l+1} = FFN(Norm(Y_l)) + Norm(Y_l)`.
Two things are different from the other strategies:

1. The block-level attention pre-norm is **suppressed** — raw `x` is fed to the
   attention module rather than `Norm(x)`.
2. Inside the attention module, V is normed in addition to Q and K
   ("QKV-norm" — see §6.2). This V-norm is automatic under `hybrid`; it is not
   a separate config knob.

For the other strategies (`pre`, `sandwich`, `post_sdpa`), QK-norm (a separate
Norm applied to Q and K only — V is *not* normed) is **always on** by default
and orthogonal to `norm_strategy`. It can be turned off with `qk_norm = false`
for ablation.

The final post-stack norm is applied regardless of `norm_strategy`.

### 5.4 Toggle composition

- `norm_type` is **global** (LayerNorm or RMSNorm for the whole model).
- `norm_strategy` is **global** (uniform across all layers).
- `residual_scaling` is **global**.
- `canon_kernel_sizes` is **per-layer** (a list of length `num_hidden_layers`,
  or a scalar broadcast across layers).
- `canon_positions` is **global** (same set of positions for every layer).
- `qk_norm` is **global**.
- Illegal: `norm_strategy = "post_sdpa"` together with `norm_strategy =
  "sandwich"` — they are mutually exclusive enum values, not composable.
- Canon at position B and `norm_strategy = "sandwich"` are independent; both can
  be set, but the result becomes hard to reason about. The config validator
  warns; it does not error.

### 5.5 Per-knob scope summary

| Knob | Scope | Notes |
| --- | --- | --- |
| `norm_type` | global | `layernorm` (default) or `rmsnorm`. |
| `norm_strategy` | global | Single enum value applied uniformly. |
| `qk_norm` | global | Per-head Norm inside attention. |
| `residual_scaling` | global | `sqrt_num_layers` (default) or `none`. |
| `rope_dim` / `nope_dim` | global | Same split across all heads of all layers. |
| `canon_enabled` | global | Master switch. |
| `canon_positions` | global | List of insertion points. |
| `canon_kernel_sizes` | per-layer | Scalar broadcasts; list must match `num_hidden_layers`. |
| `ffn_activation` | global | `swiglu` default; future variants added here. |
| `tie_word_embeddings` | global | Off by default. |
| `mlm_head_activation` | global | Default `gelu`. |
| `classifier_pool` | global (per-head model) | Only meaningful for sequence-classification head. |

---

## 6. Attention module (`OplmAttention`)

This is the most subtle piece of the spec. Two compute paths share one set of
parameters and one set of pre-attention transformations; only the kernel
differs.

### 6.1 Math

Standard multi-head attention with `num_attention_heads == num_kv_heads`
(grouped-query attention is **not** supported). Shapes:

| Step | Shape |
| --- | --- |
| Input `x` | `(B, T, D)` |
| After Q/K/V projections | `(B, T, D)` each |
| Reshape to heads | `(B, H, T, d_head)` each, where `d_head = D / H` |
| QK-norm (per head) | `(B, H, T, d_head)` |
| RoPE applied to `Q` and `K` | `(B, H, T, d_head)` |
| Attention output | `(B, H, T, d_head)` |
| Reshape back | `(B, T, D)` |
| Output projection | `(B, T, D)` |

### 6.2 QK-norm / QKV-norm

Q and K are each passed through a **separate Norm** (one for Q, one for K) that
operates over the `d_head` channel dimension. The gain (and bias, under
LayerNorm) is shape `(d_head,)`, shared across heads — i.e., the same channel
gain is broadcast across the `H` heads. By default, V is **not** normed.

```
Q_norm = Norm_q(Q)     # shape unchanged
K_norm = Norm_k(K)
```

The norm type (LayerNorm vs. RMSNorm) follows the model-wide `norm_type`.
QK-norm is computed in fp32 regardless of autocast.

When `norm_strategy == "hybrid"`, V is **also** normed by an independent
`v_norm` of shape `(d_head,)` (same `norm_type`, same `norm_eps`), giving the
"QKV-norm" formulation from arXiv 2503.04598:

```
Q_norm = Norm_q(Q)
K_norm = Norm_k(K)
V_norm = Norm_v(V)     # only under norm_strategy == "hybrid"
```

The block-level attention pre-norm is also suppressed under `hybrid` — raw `x`
is fed into the attention module so that the only norms in the attention path
are these per-projection norms. Under all other `norm_strategy` values,
`v_norm` is `nn.Identity()`.

If a user sets `qk_norm = False` together with `norm_strategy = "hybrid"`, the
attention has no normalization at all — this is an exotic combination, but it
is allowed; the validator emits a warning, not an error.

### 6.3 RoPE application

RoPE is applied to `Q_norm` and `K_norm` (after QK-norm, before scoring). See
§7 for the math. For partial RoPE, only the first `rope_dim` channels of each
head receive the rotation; the remaining `nope_dim` channels pass through
unchanged.

### 6.4 Signature

```python
def forward(
    self,
    x: Tensor,                              # (B, T, D)
    attention_mask: Tensor | None,          # (B, T) {0, 1}; 1 = real, 0 = pad
    output_attentions: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """
    Returns:
        out:           (B, T, D)
        attn_weights:  (B, H, T, T) fp32 if output_attentions else None
    """
```

### 6.5 Path selection

A single helper `_use_fast_path(...)` selects the kernel. The fast path is used
only when **all** of the following hold:

| Condition | Why |
| --- | --- |
| `output_attentions` is `False` | `flex_attention` does not return attention weights. |
| `config.use_flex_attention` is `True` | Debug override. |
| `config.attention_dropout == 0.0` | `flex_attention` has no `dropout_p` argument; honouring the configured dropout exactly requires the fallback. |
| `q.device.type == "cuda"` | `flex_attention` requires a CUDA device. |
| (`flex_attention` import is available — torch ≥ 2.11) | Soft fallback if the kernel symbol is missing. |

If any condition fails, the fallback `_manual_attention` runs instead. The
fallback honours `attention_dropout`, supports CPU/MPS, and can return weights.

```python
q, k, v = self._project_qkv(x)         # (B, H, T, d_head) each
q, k = self._qk_norm(q, k)
v = self._v_norm(v)                    # nn.Identity unless hybrid
q, k = self._apply_rope(q, k)

if self._use_fast_path(output_attentions, q.device):
    out = self._flex_attention(q, k, v, attention_mask)
    attn = None
else:
    out, attn = self._manual_attention(q, k, v, attention_mask)

out = self._output_projection(out)
return out, attn
```

QKV projection, QK/V-norm, RoPE, output projection — all shared between paths.
Only the score → softmax → value combination differs.

### 6.6 Fast path: `torch.nn.attention.flex_attention`

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# attention_mask: (B, T) -> mask_mod closure
def mask_mod(b, h, q_idx, kv_idx):
    return attention_mask[b, kv_idx] == 1

block_mask = create_block_mask(
    mask_mod,
    B=B,
    H=num_attention_heads,
    Q_LEN=T,
    KV_LEN=T,
    device=q.device,
)
out = flex_attention(q, k, v, block_mask=block_mask)  # (B, H, T, d_head)
```

Notes:

- Inputs `q, k, v` are bf16 under autocast; flex_attention internally promotes
  the softmax to fp32.
- On Blackwell with torch ≥ 2.11, flex_attention dispatches to the
  FlashAttention-4 kernel.
- The encoder is **bidirectional**; no causal mask. The only mask is the
  per-token padding mask.
- `flex_attention` returns no attention weights.
- `H` is passed as an explicit integer rather than `None`. Recent PyTorch
  builds tolerate `H=None` for head-broadcasting masks, but the documented
  signature is `H: int`; the explicit form is version-stable.
- `flex_attention` has no `dropout_p` argument. When
  `config.attention_dropout > 0`, the path selector (§6.5) forces fallback so
  the configured dropout is applied exactly.

### 6.7 Fallback path: pure manual matmul

```python
scale = 1.0 / math.sqrt(d_head)
scores = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)

if attention_mask is not None:
    # 0 in mask -> -inf in scores at that KV column
    mask = (attention_mask == 0)[:, None, None, :]   # (B, 1, 1, T)
    scores = scores.masked_fill(mask, float("-inf"))

attn = F.softmax(scores, dim=-1, dtype=torch.float32)  # (B, H, T, T) fp32
attn_dropped = F.dropout(attn, p=self.attention_dropout, training=self.training)
out = (attn_dropped.to(v.dtype) @ v)                  # (B, H, T, d_head)
```

The returned `attn` is the full softmax distribution at fp32. Use it for
precision@L (correlation of `attn` with experimental contact maps), attention
rollout, head-pruning experiments, etc.

This path is slower than `flex_attention`. It is meant for inspection and small
batches, not for training. Setting `output_attentions=True` during training is
permitted but will materially slow the forward pass and bloat memory by
`O(B·H·T²)`.

### 6.8 Output projection and dropout

A single linear `W_o: (D, D)`, no bias, projects the concatenated heads back to
the residual stream. Optional `hidden_dropout` is applied after the projection
(default `0.0`).

### 6.9 Parameter count for one attention block

Let `D = hidden_size`, `H = num_attention_heads`, `d = head_dim = D / H`.

```
params = 4 · D · D                  (Q, K, V, O projections, no bias)
       + n_norm · 2 · d             (QK-norm: one Norm each for Q and K,
                                     gain shape (d,) shared across heads,
                                     plus bias under LayerNorm)
```

where `n_norm = 2` under LayerNorm (gain + bias) and `n_norm = 1` under
RMSNorm (gain only). RoPE caches are buffers, not parameters.

---

## 7. Positional encoding

### 7.1 RoPE

Rotary positional embedding (Su et al., 2021). For each (query/key) head and
each token position `p`, pair the `d_head` channels into `d_head / 2` 2-D pairs
and rotate each pair by an angle `p · θ_i`, where `θ_i = base^(-2i / d_head)`
and `base = rope_theta` (default `10000.0`).

Implementation:

```
cos_p = cos(p · θ)       # (T, d_head/2)
sin_p = sin(p · θ)       # (T, d_head/2)

def rotate(x, cos, sin):
    # x: (..., d_head); pair channels (x_even, x_odd)
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    return interleave(out_even, out_odd)        # back to (..., d_head)
```

Applied to Q and K (post-QK-norm), per head, per position. Not applied to V.

`cos`/`sin` are precomputed buffers up to `max_position_embeddings`. If a longer
sequence arrives at inference, the buffers are extended on the fly (no
parameter update).

RoPE rotations are computed in **fp32** then cast back to the input dtype.

### 7.2 Partial RoPE / NoPE split

When `nope_dim > 0`, the head channels are split:

```
head_dim = rope_dim + nope_dim
```

RoPE rotates only the first `rope_dim` channels of each head. The trailing
`nope_dim` channels are position-invariant.

```
q_split = q[..., :rope_dim], q[..., rope_dim:]
q_rope  = rotate(q_split[0], cos, sin)
q_out   = cat([q_rope, q_split[1]], dim=-1)
```

`rope_dim + nope_dim == head_dim` is enforced at config time. The default is
`nope_dim = 0` (full RoPE).

Background: arXiv 2502.14837v1 reports that some channels prefer to be
position-invariant. Partial RoPE makes that explicit instead of leaving the
model to discover it.

### 7.3 Long-context behavior

OPLM does not currently implement YaRN, NTK-aware scaling, or other RoPE
extrapolation schemes. Training-time context length is set by
`max_position_embeddings`. Extrapolation beyond that is "best effort" — buffers
extend automatically but the model has not been optimized for it. Future work.

---

## 8. Feed-forward block

### 8.1 SwiGLU

Default `ffn_activation = "swiglu"`:

```
gate = W_g(x)                    # (B, T, F)
up   = W_u(x)                    # (B, T, F)
y    = W_d(silu(gate) * up)      # (B, T, D)
```

Three linears, all without bias by default (`ffn_bias = False`). `silu(x) = x ·
sigmoid(x)`.

### 8.2 Hidden dimension

```
F = ffn_dim_multiplier · D
```

Default multiplier `≈ 8/3`, then rounded **up** to the nearest multiple of 256
to keep matmul shapes hardware-friendly:

```
F = round_up_to(8/3 · D, 256)
```

`intermediate_size` (the HF field name) may also be set explicitly; the
multiplier is then ignored.

### 8.3 Extension hooks

The `ffn_activation` field is the single entry point for FFN variants. Future
activations (GeGLU, ReLU², etc.) plug in by adding a new enum value and a
matching small class. The transformer block does not need modification.

### 8.4 Parameter count for one FFN block

```
params = 3 · D · F     (W_g, W_u, W_d, no bias)
```

---

## 9. Normalization

The `norm_type` config field selects the normalization operator used everywhere
in the model (block pre-norm, any post-norms required by `norm_strategy`,
QK-norm, post-embedding norm, final stack norm, optional pre-head norm). The
same operator type is used at every site; mixing LayerNorm and RMSNorm within a
single model is not supported.

### 9.1 LayerNorm (default)

```
LayerNorm(x) = γ · (x − mean(x)) / sqrt(var(x) + eps) + β
```

- `γ` and `β` are learnable vectors of shape `(D,)` (or `(d_head,)` for
  QK-norm), initialized to 1 and 0 respectively.
- `eps = norm_eps`, default `1e-6`.
- mean and variance are taken over the last dimension (the feature dimension).
- Computed in **fp32** regardless of autocast: the input is up-cast to fp32
  for the mean/var, normalized, multiplied by `γ` plus `β`, and cast back to
  the input dtype.

### 9.2 RMSNorm (`norm_type = "rmsnorm"`)

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + eps)
```

- `γ` is a learnable vector of shape `(D,)` (or `(d_head,)`), initialized to 1.
- `eps = norm_eps`, default `1e-6`.
- No bias term.
- fp32 internal compute, same as LayerNorm.

LayerNorm has both centering and scaling; RMSNorm drops the centering step. In
return, RMSNorm has half the parameters at each site and is slightly cheaper
per token. Empirically the two are close at modest scale; we default to
LayerNorm because (a) it matches ESM-C and (b) the centering step is mildly
helpful for some training stabilities.

### 9.3 All norm sites in the model

| Site | Shape of γ (and β under LayerNorm) |
| --- | --- |
| Token-embedding output (optional `post_embed_norm`) | `(D,)` |
| Attention pre-norm — omitted under `hybrid` | `(D,)` |
| Attention QK-norm (Q) | `(d_head,)` |
| Attention QK-norm (K) | `(d_head,)` |
| Attention V-norm (only under `hybrid`) | `(d_head,)` |
| FFN pre-norm | `(D,)` |
| Optional sandwich / post-SDPA post-norms | `(D,)` |
| Final stack norm | `(D,)` |
| MLM head intermediate norm | `(D,)` |
| Optional pre-head norm (classification / token heads) | `(D,)` |

Each `norm_type = "layernorm"` site doubles the parameter count of its
RMSNorm equivalent (γ and β instead of just γ). The QK-norm sites use one
parameter set per attention layer per Q-or-K — i.e., 2L of them total — each
gain shape `(d_head,)` and shared across the `H` heads.

---

## 10. Embeddings and heads

### 10.1 Token embedding

```python
self.embed_tokens = nn.Embedding(vocab_size=33, embedding_dim=D)
```

- No positional embedding (RoPE handles position).
- `post_embed_norm` (Norm after the embedding lookup) is `False` by default.
- Init: truncated normal `std = initializer_range` (default `0.02`); `<pad>`
  row not zeroed (consistent with HF convention).

### 10.2 MLM head

OPLM uses the BERT / RoBERTa-style two-layer MLP MLM head:

```python
class OplmMLMHead(nn.Module):
    def __init__(self, config):
        self.dense   = nn.Linear(D, D)                     # bias=True
        self.act     = activation_fn(config.mlm_head_activation)  # default GELU
        self.norm    = Norm(D, eps=config.norm_eps)        # LayerNorm by default
        self.decoder = nn.Linear(D, vocab_size, bias=True)  # vocab projection

    def forward(self, x):
        x = self.dense(x)
        x = self.act(x)
        x = self.norm(x)
        return self.decoder(x)
```

Order: Dense → activation → Norm → decoder projection. The intermediate Norm
follows the model-wide `norm_type` (LayerNorm by default; RMSNorm under
`norm_type = "rmsnorm"`).

`tie_word_embeddings` is **off by default**. The decoder projection is its own
`Linear(D, V)`. When set to `True`, `decoder.weight` is tied to
`embed_tokens.weight` (the decoder bias remains an independent parameter).
Untied is the default because OPLM's vocabulary is tiny (33 tokens × D ≈ 25K
parameters at D=768), so the parameter savings from tying are trivial, and
keeping the projection independent gives the head a small extra degree of
freedom that historically helps slightly on MLM.

Loss:

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
    ignore_index=-100,
)
```

### 10.3 Sequence-classification head

```python
class OplmClassificationHead(nn.Module):
    def __init__(self, config):
        self.pool = config.classifier_pool      # "mean" (default) | "cls"
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.dense = nn.Linear(D, num_labels, bias=True)
```

Pooling:

| `classifier_pool` | Operation |
| --- | --- |
| `mean` (default) | Mean over the last hidden state at positions where `attention_mask == 1`. Pad and special tokens are excluded if their mask is zero; users can pass a mask that includes or excludes `<cls>`/`<eos>` as they see fit. |
| `cls` | Take the hidden state at position 0 (`<cls>`). |

An optional `pre_head_norm` (using the model's configured `norm_type`) is
available; off by default.

### 10.4 Token-classification head

```python
self.dropout = nn.Dropout(config.classifier_dropout)
self.classifier = nn.Linear(D, num_labels, bias=True)
```

Applied per-token to the last hidden state. No pooling.

### 10.5 Head summary

| Head | Input | Output | Used by |
| --- | --- | --- | --- |
| MLM | last hidden `(B, T, D)` | logits `(B, T, V)` | `OplmForMaskedLM` |
| Sequence classifier | last hidden `(B, T, D)` + mask | logits `(B, num_labels)` | `OplmForSequenceClassification` |
| Token classifier | last hidden `(B, T, D)` | logits `(B, T, num_labels)` | `OplmForTokenClassification` |

---

## 11. Canon-style depthwise conv layers

### 11.1 Operator

A 1D **depthwise** convolution along the sequence dimension:

```python
conv = nn.Conv1d(
    in_channels=D, out_channels=D,
    kernel_size=k, groups=D,
    padding=k // 2, bias=False,
)
```

- `groups = D` makes it depthwise (per-channel).
- `padding = k // 2` with odd `k` gives same-length output and is symmetric.
  For even `k`, the operator pads `k // 2` on the left and `k // 2 - 1` on the
  right (the implementation must document this explicitly to avoid drift).
- An optional pointwise activation (configurable: `none`/`silu`/`gelu`) follows
  the conv. Default `none`.

The expected input layout is `(B, T, D)` (HF convention); the conv internally
transposes to `(B, D, T)`, runs, then transposes back.

### 11.2 Bidirectionality

The encoder is bidirectional. The conv kernel sees both past and future
tokens; there is no causal masking on the conv. This differs from Canon's
decoder-flavored usage where causal convs are required.

### 11.3 Insertion positions

| Position | Insert at | Notes |
| --- | --- | --- |
| **A** | Input of the block, before the attention pre-norm | Modifies the residual-stream tap. |
| **B** | Between attention pre-norm and Q/K/V projections | Discouraged — weakest in Canon paper. |
| **C** | On the attention output, before the residual add | After the attention output projection. |
| **D** | Between the FFN pre-norm and the FFN input | Most common positive result. |

`canon_positions` is a list. Common combos: `["A", "C", "D"]`, `["D"]`,
`["A", "D"]`. The order of the list does not matter (positions are not
sequential — each names an independent injection site).

### 11.4 Kernel-size schedule

`canon_kernel_sizes` accepts:

- A scalar `int` — broadcast across all `num_hidden_layers`.
- A list of `int` of length exactly `num_hidden_layers` — per-layer kernel
  sizes.
- A `dict` with `{schedule: "linear", min: int, max: int}` or `{schedule:
  "constant", value: int}` — generated at config-load time.

Validation: every kernel size must be `≥ 2`. Even kernels are allowed (the
asymmetric-pad convention above applies).

### 11.5 Padding-token leakage

A depthwise conv with non-zero kernel weight on pad positions will smear pad
content into real-token channels. Before each conv, the input is zeroed at
positions where `attention_mask == 0`:

```python
x = x * attention_mask.unsqueeze(-1)
```

This is the model's responsibility; callers do not need to pre-zero inputs.

### 11.6 Parameter and FLOP cost

For a single conv of kernel size `k`:

```
params = k · D
flops  ≈ 2 · k · D · T   per token batch
```

Total Canon cost across the stack: `sum over layers, positions of k · D`.

### 11.7 Block-with-canon diagram

```
x ─► [Conv_A(k_l)] ─► Norm ─► Attn ─► [Conv_C(k_l)] ─► (+) ─► h
                                                               │
h ─► Norm ─► [Conv_D(k_l)] ─► SwiGLU FFN ──────────────► (+) ─► y
```

`k_l` is layer `l`'s kernel size from the schedule.

---

## 12. Configuration schema (`OplmConfig`)

`OplmConfig` subclasses `transformers.PretrainedConfig`. The YAML `model:` block
maps 1:1 to `OplmConfig` constructor kwargs.

### 12.1 Field table

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `model_type` | `str` | `"oplm"` | Auto-class lookup key. Do not override. |
| `vocab_size` | `int` | `33` | Token vocabulary size. |
| `hidden_size` | `int` | `768` | Residual-stream dimension `D`. |
| `num_hidden_layers` | `int` | `12` | Number of `OplmBlock`s. |
| `num_attention_heads` | `int` | `12` | `H`. Must divide `hidden_size`. |
| `head_dim` | `int` | derived | `hidden_size / num_attention_heads` if not set. |
| `intermediate_size` | `int` | derived | `round_up_to(8/3 · hidden_size, 256)` if not set. |
| `max_position_embeddings` | `int` | `1024` | RoPE buffer size; sequences longer than this trigger buffer extension. |
| `rope_theta` | `float` | `10000.0` | RoPE base. |
| `rope_dim` | `int` | `head_dim` | Channels per head that receive RoPE. |
| `nope_dim` | `int` | `0` | Channels per head left position-invariant. `rope_dim + nope_dim == head_dim`. |
| `norm_type` | `str` | `"layernorm"` | One of `layernorm`, `rmsnorm`. Applies to every norm site in the model. |
| `norm_eps` | `float` | `1e-6` | Epsilon for whichever norm is selected. |
| `norm_strategy` | `str` | `"pre"` | One of `pre`, `sandwich`, `hybrid`, `post_sdpa`. |
| `qk_norm` | `bool` | `True` | Per-head Norm on Q and K. |
| `post_embed_norm` | `bool` | `False` | Norm immediately after token embedding. |
| `residual_scaling` | `str` | `"sqrt_num_layers"` | One of `sqrt_num_layers`, `none`. Scales each sublayer output by `1/sqrt(L)` before adding to the residual stream. |
| `init_scale_output_projections` | `bool` | `True` | When `True`, init `std` of attention `W_o` and FFN `W_d` is divided by `sqrt(2L)` (GPT-2 style). Redundant with `residual_scaling = "sqrt_num_layers"`; disable as an ablation. |
| `ffn_activation` | `str` | `"swiglu"` | FFN variant. |
| `ffn_bias` | `bool` | `False` | Bias on FFN linears. |
| `attention_dropout` | `float` | `0.0` | Dropout on attention weights. Any non-zero value forces the fallback path (`flex_attention` has no `dropout_p` argument; see §6.5). |
| `hidden_dropout` | `float` | `0.0` | Dropout after attention/FFN output projections. |
| `tie_word_embeddings` | `bool` | `False` | Tie MLM head decoder weight to `embed_tokens.weight`. |
| `mlm_head_activation` | `str` | `"gelu"` | Activation in the BERT-style MLM head MLP. One of `gelu`, `silu`, `relu`. |
| `canon_enabled` | `bool` | `False` | Master switch for Canon conv sublayers. |
| `canon_positions` | `list[str]` | `[]` | Subset of `["A", "B", "C", "D"]`. |
| `canon_kernel_sizes` | `int \| list[int] \| dict` | `4` | Kernel size or schedule. |
| `canon_activation` | `str` | `"none"` | Pointwise activation after each conv: `none`/`silu`/`gelu`. |
| `initializer_range` | `float` | `0.02` | Truncated-normal std for non-residual params. |
| `classifier_pool` | `str` | `"mean"` | Sequence-classification pooler: `mean` or `cls`. |
| `classifier_dropout` | `float` | `0.0` | Dropout in classification heads. |
| `num_labels` | `int` | `2` | Number of output classes for classification heads. |
| `pre_head_norm` | `bool` | `False` | Norm (using `norm_type`) immediately before any task head. |
| `use_flex_attention` | `bool` | `True` | Debug override. Setting to `False` forces the fallback path always — useful for debugging but slow. |
| `gradient_checkpointing` | `bool` | `False` | Activation checkpointing on `OplmBlock`. |
| `pad_token_id` | `int` | `1` | Matches ESM vocabulary. |
| `bos_token_id` | `int` | `0` | `<cls>`. |
| `eos_token_id` | `int` | `2` | |
| `unk_token_id` | `int` | `3` | |
| `mask_token_id` | `int` | `32` | |
| `auto_map` | `dict[str, str]` | filled by `save_pretrained` | See §13. |

### 12.2 Validation rules

Enforced in `OplmConfig.__init__` (with explicit `ValueError` on violation):

- `hidden_size % num_attention_heads == 0`.
- `head_dim * num_attention_heads == hidden_size` (if `head_dim` is set
  explicitly).
- `rope_dim + nope_dim == head_dim`.
- `rope_dim >= 0`, `nope_dim >= 0`, and `rope_dim` even (each RoPE rotation
  consumes a channel pair).
- `norm_type in {"layernorm", "rmsnorm"}`.
- `norm_strategy in {"pre", "sandwich", "hybrid", "post_sdpa"}`.
- `residual_scaling in {"sqrt_num_layers", "none"}`.
- `ffn_activation in {"swiglu", "geglu"}` (`geglu` reserved for future use).
- `mlm_head_activation in {"gelu", "silu", "relu"}`.
- If `canon_enabled`: `canon_positions` non-empty and ⊆ `{"A", "B", "C", "D"}`;
  resolved `canon_kernel_sizes` is a list of length `num_hidden_layers` with
  each element `≥ 2`.
- `classifier_pool in {"mean", "cls"}`.
- `vocab_size == 33` (warning, not error — only relevant if users want a custom
  vocab, which currently is unsupported).

### 12.3 Forward-compat / deprecation policy

A deprecated field is kept readable for one minor version, with a
`DeprecationWarning` emitted at config load. After that minor version, the
field is removed. The current spec has no deprecated fields.

---

## 13. HuggingFace integration mechanics

OPLM integrates with the HF ecosystem through two complementary mechanisms:

1. **In-process registration** — for users who `pip install oplm` and `import
   oplm` before calling `AutoModel*.from_pretrained(...)`. No
   `trust_remote_code` required.
2. **`auto_map`** — for users who load a checkpoint without first importing
   OPLM (e.g., from a fresh transformers install). The checkpoint's
   `config.json` points HF to `modeling_oplm.py` in the model repo, and
   `trust_remote_code=True` is required.

Both mechanisms are configured by the model repo's maintainer (us); end users
choose which one to rely on.

### 13.1 Required subclass attributes

```python
class OplmPreTrainedModel(PreTrainedModel):
    config_class = OplmConfig
    base_model_prefix = "oplm"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OplmBlock"]
    _supports_sdpa = True
    _supports_flash_attn_2 = False  # we use flex_attention, not flash-attn
```

`_no_split_modules` tells FSDP to wrap `OplmBlock`s atomically.

### 13.2 In-process registration

`src/oplm/__init__.py`:

```python
from transformers import (
    AutoConfig, AutoModel, AutoModelForMaskedLM,
    AutoModelForSequenceClassification, AutoModelForTokenClassification,
    AutoTokenizer,
)
from .model.configuration_oplm import OplmConfig
from .model.modeling_oplm import (
    OplmModel,
    OplmForMaskedLM,
    OplmForSequenceClassification,
    OplmForTokenClassification,
)
from .model.tokenization_oplm import OplmTokenizerFast

AutoConfig.register("oplm", OplmConfig)
AutoModel.register(OplmConfig, OplmModel)
AutoModelForMaskedLM.register(OplmConfig, OplmForMaskedLM)
AutoModelForSequenceClassification.register(OplmConfig, OplmForSequenceClassification)
AutoModelForTokenClassification.register(OplmConfig, OplmForTokenClassification)
AutoTokenizer.register(OplmConfig, fast_tokenizer_class=OplmTokenizerFast)

# Also tell HF to copy these classes' source files on push_to_hub /
# save_pretrained and to populate `auto_map` automatically. This is the
# documented hook — setting `auto_map` by hand does not copy code (see §13.3).
OplmConfig.register_for_auto_class("AutoConfig")
OplmModel.register_for_auto_class("AutoModel")
OplmForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
OplmForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
OplmForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
OplmTokenizerFast.register_for_auto_class("AutoTokenizer")
```

After `import oplm`, calls like `AutoModelForMaskedLM.from_pretrained(
"brineylab/oplm-base")` resolve without `trust_remote_code`.

### 13.3 `register_for_auto_class()` and `auto_map`

`auto_map` (a dict written into `config.json` / `tokenizer_config.json`) is
what `trust_remote_code=True` loaders read to find the Python class to
instantiate. But populating `auto_map` by hand is **not** sufficient to make
`push_to_hub` upload the modeling code — HF only copies custom code for classes
that have called `register_for_auto_class(...)`. The two mechanisms work
together:

| Mechanism | What it does | Who calls it |
| --- | --- | --- |
| `OplmConfig.register_for_auto_class("AutoConfig")` etc. | Marks the class so `save_pretrained` / `push_to_hub` (a) write the right `auto_map` entries into the JSON, and (b) copy the source file into the model repo. | The package, at import time. |
| `auto_map` in `config.json` | Tells a downstream `trust_remote_code=True` loader which class to instantiate. | Written automatically by `save_pretrained` once the class is registered. |

OPLM calls `register_for_auto_class` at import time, in
`src/oplm/__init__.py`, after the in-process `AutoConfig.register(...)` /
`AutoModel.register(...)` calls:

```python
OplmConfig.register_for_auto_class("AutoConfig")
OplmModel.register_for_auto_class("AutoModel")
OplmForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
OplmForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
OplmForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
OplmTokenizerFast.register_for_auto_class("AutoTokenizer")
```

After this, `model.save_pretrained(dir)` writes both the weights and the
modeling / configuration / tokenization `.py` files into `dir`, and
`push_to_hub` uploads all of them — without any further manual `auto_map`
construction.

Verifying the round-trip is mandatory: the `test_save_load.py` /
`test_auto_classes.py` integration tests (§14 of TODOS) must include a case
that calls `save_pretrained(tmpdir)`, then opens `tmpdir` in a *subprocess*
with `transformers` only (no `import oplm`) and runs
`AutoModelForMaskedLM.from_pretrained(tmpdir, trust_remote_code=True)`. Without
that, regressions in the file-copy step go undetected until someone uses the
checkpoint from a clean env.

The resulting `config.json` looks like this (HF writes it automatically):

```json
{
  "model_type": "oplm",
  "auto_map": {
    "AutoConfig": "configuration_oplm.OplmConfig",
    "AutoModel": "modeling_oplm.OplmModel",
    "AutoModelForMaskedLM": "modeling_oplm.OplmForMaskedLM",
    "AutoModelForSequenceClassification": "modeling_oplm.OplmForSequenceClassification",
    "AutoModelForTokenClassification": "modeling_oplm.OplmForTokenClassification"
  },
  "...other fields...": "..."
}
```

End users then run:

```python
model = AutoModelForMaskedLM.from_pretrained(
    "brineylab/oplm-base",
    trust_remote_code=True,
)
```

with `transformers` alone — no `pip install oplm` necessary.

### 13.4 Weight loading conventions

- State-dict keys follow the natural module nesting (`oplm.embed_tokens.weight`,
  `oplm.layers.0.self_attn.q_proj.weight`, etc.).
- Tied embeddings: only `embed_tokens.weight` is stored on disk; `lm_head` is
  re-tied at load time inside `tie_weights()`.
- Sharded loading: standard HF `safetensors` index format. No custom shard
  layout.
- FSDP/DDP compatibility: the model uses no global mutable state, no
  non-determined dispatching, and no in-place ops on shared buffers.

### 13.5 `save_pretrained` / `push_to_hub` notes

Files written by `save_pretrained`:

- `config.json` — `OplmConfig` serialization with `auto_map`.
- `model.safetensors` (or sharded variants) — weights.
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` — when
  the tokenizer is also saved at the same path.
- On `push_to_hub`: additionally the modeling/configuration Python files are
  uploaded so `trust_remote_code` works.

### 13.6 Tokenizer integration

`AutoTokenizer.register(OplmConfig, fast_tokenizer_class=OplmTokenizerFast)`
covers the in-process case. For remote loading, `tokenizer_config.json`'s
`tokenizer_class` field is `"OplmTokenizerFast"` and the `auto_map` includes a
`"AutoTokenizer"` entry. Standard HF lookup handles the rest.

---

## 14. Forward-pass contract

### 14.1 Inputs

All four `Oplm*ForX` classes accept the same kwargs (those they don't need are
ignored):

| Argument | Type / shape | Default | Description |
| --- | --- | --- | --- |
| `input_ids` | `(B, T) int64` | required | Token IDs from `OplmTokenizerFast`. |
| `attention_mask` | `(B, T) int64 {0,1}` | `None` (treated as all-ones) | `1` = real token, `0` = pad. |
| `labels` | `(B, T) int64 \| (B,) int64 \| None` | `None` | MLM labels (`-100` ignored), per-token labels, or per-sequence labels depending on the head. |
| `output_attentions` | `bool` | `False` | When `True`, forces fallback attention globally. |
| `output_hidden_states` | `bool` | `False` | Return all intermediate hidden states. |
| `return_dict` | `bool` | `True` | Return a `ModelOutput`; if `False`, return a tuple. |

### 14.2 Outputs

| Class | Return type | Fields |
| --- | --- | --- |
| `OplmModel` | `BaseModelOutput` | `last_hidden_state (B, T, D)`, `hidden_states` (optional), `attentions` (optional) |
| `OplmForMaskedLM` | `MaskedLMOutput` | `loss (scalar)` (if `labels`), `logits (B, T, V)`, `hidden_states`, `attentions` |
| `OplmForSequenceClassification` | `SequenceClassifierOutput` | `loss`, `logits (B, num_labels)`, `hidden_states`, `attentions` |
| `OplmForTokenClassification` | `TokenClassifierOutput` | `loss`, `logits (B, T, num_labels)`, `hidden_states`, `attentions` |

Dtype contract:

- Logits: **fp32** (cast at the head output).
- Hidden states: bf16 under autocast, fp32 otherwise.
- Attentions: **fp32** when returned.
- Loss: **fp32**.

### 14.3 Loss computation

MLM:

```python
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

Sequence classification:

```python
loss = F.cross_entropy(logits, labels)
```

Token classification:

```python
loss = F.cross_entropy(logits.view(-1, num_labels), labels.view(-1), ignore_index=-100)
```

### 14.4 `output_attentions=True` semantics

When `output_attentions=True`, **every** attention layer switches to the
fallback path for this forward call. Memory cost rises by approximately
`B · L · H · T²` floats at fp32. For `T = 1024, L = 12, H = 12, B = 8`, that
is ~6 GB of additional activation memory. Do not enable this casually during
training.

### 14.5 `output_hidden_states=True` semantics

Returns a tuple of `(L + 1)` tensors: the post-embedding state, then the
output of each `OplmBlock` (post-residual). All shape `(B, T, D)`.

---

## 15. Initialization

### 15.1 Default scheme

`OplmPreTrainedModel._init_weights(module)` walks every submodule and applies:

| Parameter kind | Init |
| --- | --- |
| `nn.Linear.weight` (general) | Truncated normal, mean `0`, std `initializer_range` (`0.02`). |
| `nn.Linear.bias` | Zero. |
| `nn.Embedding.weight` | Truncated normal, mean `0`, std `initializer_range`. |
| Norm gain (`γ`) — both LayerNorm and RMSNorm | One. |
| Norm bias (`β`, LayerNorm only) | Zero. |
| QK-norm gain `γ` | One. (Bias zero under LayerNorm.) |
| Attention output projection `W_o.weight` | Truncated normal, mean `0`, std `initializer_range / sqrt(2 · num_hidden_layers)` when `init_scale_output_projections = True` (default); std `initializer_range` otherwise. |
| FFN down-projection `W_d.weight` | Same rule as `W_o` above. |
| `nn.Conv1d.weight` (Canon) | Truncated normal, mean `0`, std `initializer_range`. |
| MLM head `dense.weight`, `decoder.weight` (when untied) | Truncated normal, mean `0`, std `initializer_range`. (Decoder is not residual-stream-writing, so no `1/sqrt(2L)` factor.) |
| MLM head `dense.bias`, `decoder.bias` | Zero. |
| Classifier `weight` | Truncated normal, mean `0`, std `initializer_range`. |
| Classifier `bias` | Zero. |

The `1 / sqrt(2L)` scaling on residual-stream-writing projections follows the
GPT-2 / NanoGPT convention. With `residual_scaling = "sqrt_num_layers"` active
(default), it is partially redundant — the runtime division by `sqrt(L)`
already controls residual-stream variance growth. The init scaling stays on by
default as a belt-and-braces choice for the 80-layer target depth; users may
ablate it via `init_scale_output_projections = False`.

### 15.2 Tying

After init, `tie_weights()` is called. When `tie_word_embeddings == True`,
`lm_head.decoder.weight` is set to `embed_tokens.weight` (the same
`nn.Parameter`, not a copy). The decoder bias remains an independent learnable
parameter regardless of tying. The default is `False`.

### 15.3 Meta-device init

`from_pretrained` may be invoked with `device_map="meta"` for introspection
(parameter counting, structure inspection). The init scheme is idempotent on
meta tensors: the materialization happens later via `load_state_dict`.

---

## 16. Numerical precision and autocast

### 16.1 Training-time defaults

- Mixed precision: **bf16** autocast on CUDA. fp32 master weights are managed
  by `Accelerator` (out of scope for this doc).
- Norms (LayerNorm / RMSNorm, including QK-norm): always compute in **fp32**
  internally, output cast back to the autocast dtype.
- Attention softmax: **fp32** internal (both paths).
- RoPE rotations: computed in fp32, then cast.
- Logits: returned in **fp32**.

### 16.2 Inference defaults

- Default load dtype: **bf16** on CUDA, **fp32** on CPU. Override via
  `torch_dtype=` on `from_pretrained`.
- `model.eval()` plus `torch.inference_mode()` is the recommended idiom.
- `output_attentions=True` returns fp32 attention regardless of compute dtype.

### 16.3 Why fp32 inside norms

bf16's reduced mantissa precision causes meaningful drift in
sum-of-squares (and mean) reductions. Running both LayerNorm and RMSNorm in
fp32 internally costs negligible time and removes a known source of numerical
instability seen in long-context training.

---

## 17. Determinism and reproducibility

- The **fallback path** is bitwise reproducible given seeded inputs and
  identical hardware. Use it for any precision-critical evaluation (precision@L
  on contact maps, attention rollout, ablation deltas).
- The **fast path** (`flex_attention`) compiles a kernel that may differ across
  torch versions, CUDA versions, and GPU architectures. Numerical drift is
  typically `< 1e-3` relative, but exact bit equality is not guaranteed.
- For paper-grade reported numbers, prefer fallback-path inference and pin
  every relevant version (torch, CUDA, driver, GPU).
- Seeding (`torch.manual_seed`, `np.random.seed`, etc.) is the trainer's
  responsibility, not the model's.

---

## 18. Parameter-count formula

Let `D = hidden_size`, `H = num_attention_heads`, `d = head_dim`,
`F = intermediate_size`, `L = num_hidden_layers`, `V = vocab_size`.

### 18.1 Components

Let `n_norm = 2` under LayerNorm (gain + bias) and `n_norm = 1` under RMSNorm
(gain only). Residual scaling is parameter-free.

```
embedding:                       V · D
final norm:                      n_norm · D

MLM head:                        D · D + D            (dense)
                               + n_norm · D           (intermediate norm)
                               + (D · V + V if untied; else V)
                                                      (decoder weight + bias)

per attention block:
    Q/K/V/O projections:         4 · D · D
    QK-norm gains/biases:        2 · n_norm · d       (one Norm each for Q, K)
    pre-norm:                    n_norm · D           (omitted under hybrid)
    [if sandwich:                + n_norm · D]
    [if post_sdpa:               + n_norm · D]
    [if hybrid:                  + n_norm · d         (V-norm)
                                 − n_norm · D         (no outer pre-norm)]

per FFN block:
    W_g, W_u, W_d:               3 · D · F
    pre-norm:                    n_norm · D           (under hybrid, same gain is
                                                      reused as the post-add Norm;
                                                      no separate norm parameters)
    [if sandwich:                + n_norm · D]

per Canon conv (one per enabled position per layer):
    k_l · D
```

### 18.2 Worked example: hypothetical "base" preset

Parameters:

- `D = 768, L = 12, H = 12, d = 64, F = 2048, V = 33`.
- `norm_type = "layernorm"` (so `n_norm = 2`).
- `tie_word_embeddings = False` (default).
- `canon_enabled = False`.
- `norm_strategy = "pre"`.

```
embedding:                33 · 768                                =     25,344
final LayerNorm:          2 · 768                                 =      1,536

MLM head:
    dense weight:         768 · 768                               =    589,824
    dense bias:           768                                     =        768
    intermediate LN:      2 · 768                                 =      1,536
    decoder weight:       768 · 33                                =     25,344
    decoder bias:         33                                      =         33
    subtotal:                                                          617,505

per block (× 12):
    attn projections:     4 · 768 · 768 = 2,359,296
    QK-norm (LN, 2 norms × (γ + β) of 64):  2 · 2 · 64 =      256
    attn pre-norm (LN):                     2 · 768   =    1,536
    FFN:                  3 · 768 · 2048 = 4,718,592
    FFN pre-norm (LN):                      2 · 768   =    1,536
    subtotal:                                              7,081,216

stack total:              12 · 7,081,216                          = 84,974,592

grand total:              25,344 + 1,536 + 617,505 + 84,974,592   ≈ 85.6 M
```

The actual repo preset numbers live in `src/oplm/configs/model/presets/`.

---

## 19. File-level module breakdown

```
src/oplm/
├── __init__.py                  Auto-registration block (see §13.2).
└── model/
    ├── __init__.py              Re-exports of public classes.
    ├── configuration_oplm.py    OplmConfig + validation.
    ├── modeling_oplm.py         All public Oplm* classes (single file).
    │                            Imports the helper modules below.
    ├── transformer.py           OplmBlock: assembles attention + FFN + norm
    │                            strategy + canon insertions.
    ├── attention.py             OplmAttention with dual-path forward.
    ├── rope.py                  RoPE + partial RoPE.
    ├── norm.py                  LayerNorm + RMSNorm classes + placement helpers.
    ├── ffn.py                   SwiGLU FFN (and future activation variants).
    ├── conv.py                  Bidirectional depthwise conv for Canon.
    ├── embedding.py             Token embeddings; pooling helpers.
    ├── masking.py               Pad-mask helpers, flex_attention mask_mod
    │                            factories, conv-input zeroing.
    ├── outputs.py               LogitsConfig and LogitsOutput dataclasses.
    └── tokenization_oplm.py     OplmTokenizerFast.
```

Why one big `modeling_oplm.py`: the HF `auto_map` mechanism + `trust_remote_code`
loading is simpler when all public classes live in a single file. The internal
helpers (`attention.py`, `rope.py`, `ffn.py`, `conv.py`, `norm.py`,
`embedding.py`, `masking.py`) are imported by `modeling_oplm.py` and are
co-uploaded by `push_to_hub`; users never have to reference them directly.

`tests/` mirrors this layout: per-component tests for each helper, plus end-to-
end tests against `OplmForMaskedLM`.

---

## 20. Distributed-training constraints

The trainer is out of scope here, but the model must satisfy these constraints
to be FSDP/DDP-friendly:

- **No global mutable state.** No module-level caches that change during
  forward. The RoPE cos/sin buffers are registered buffers (state-dict tracked,
  not parameters) and are extended via a method that respects whether the
  module is wrapped.
- **`_no_split_modules = ["OplmBlock"]`** so FSDP wraps blocks atomically.
- **Weight tying** survives FSDP wrapping: `lm_head.weight = embed_tokens.weight`
  is re-applied in `tie_weights()` which FSDP-wrapped models call after
  state-dict load.
- **`flex_attention`** under `torch.compile` interacts cleanly with FSDP as of
  torch ≥ 2.11; we do not call `torch.compile` from inside the model — that is
  the trainer's responsibility.
- **No in-place ops** on shared buffers (cos/sin caches, attention masks).
- **Activation checkpointing** (`config.gradient_checkpointing = True`) wraps
  each `OplmBlock`'s forward in `torch.utils.checkpoint.checkpoint`.

See the (future) `docs/TRAINER.md` for accelerator config, FSDP sharding
policies, optimizer construction, and schedule design.

---

## 21. Out of scope / future work

Not addressed in this document:

- Trainer, optimizer, learning-rate schedules, FLOPs accounting,
  gradient-clipping, checkpoint policy → `docs/TRAINER.md`.
- Eval harness and downstream benchmarks (ProteinGym, TAPE, ProteinGlue,
  EVEREST, structure-prediction probes) → [`EVAL_HARNESS.md`](EVAL_HARNESS.md).
- Data pipeline, masking strategy, dataset formats, multi-dataset weighting →
  `docs/DATA_TOOLING.md`.
- CLI surface and `accelerate launch` invocation patterns → `docs/CLI.md`.
- Exact preset numerical recipes (small/medium/large depth × width × heads)
  → `src/oplm/configs/model/presets/*.yaml`.
- Future model-architecture ablations: alternative activations beyond
  SwiGLU/GeGLU; ALiBi / NoPE-only; mixture-of-experts; long-context schemes
  (YaRN, NTK-aware, position interpolation); structured-state-space hybrids;
  quantized inference (int8/int4).

---

## Appendix A: Glossary

- **MLM** — Masked language modeling. The pretraining objective: predict
  randomly masked tokens from their context.
- **RoPE** — Rotary position embedding. Position is encoded by rotating Q/K
  channel pairs by position-dependent angles.
- **NoPE** — "No positional encoding." Channels in a partial-RoPE scheme that
  receive no rotation.
- **QK-norm** — Norm applied (per head, gain shared across heads) to Q and K
  before scoring. Stabilizes attention logits, especially at scale.
- **SwiGLU** — Gated FFN: `down(silu(gate(x)) * up(x))`. Standard in modern
  PLMs.
- **RMSNorm** — Root-mean-square norm. Cheaper than LayerNorm; no centering.
- **Depthwise conv** — Convolution where each input channel is convolved by its
  own kernel (no cross-channel mixing).
- **FSDP** — Fully Sharded Data Parallel. PyTorch's data-parallel scheme that
  shards parameters and gradients across ranks.
- **`flex_attention`** — `torch.nn.attention.flex_attention`, a programmable
  attention kernel that dispatches to FlashAttention-family backends.
- **FA4** — FlashAttention-4. Latest in the FlashAttention family; required on
  Blackwell GPUs.
- **FlashAttention** — Fused attention kernel that avoids materializing the
  full `T × T` attention matrix.

## Appendix B: References

1. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position
   Embedding.* arXiv:2104.09864.
2. Touvron, H. et al. *LLaMA: Open and Efficient Foundation Language Models.*
   arXiv:2302.13971. (Precedent for SwiGLU + RMSNorm.)
3. Zhang, B. & Sennrich, R. *Root Mean Square Layer Normalization.*
   arXiv:1910.07467.
4. Partial RoPE / NoPE-split — arXiv:2502.14837v1.
5. HybridNorm — arXiv:2503.04598.
6. Canon depthwise conv layers — arXiv:2512.17351.
7. ESM-C — EvolutionaryScale, `evolutionaryscale/esm`. (Tokenizer, public API.)
8. Hao, W. et al. *FlashAttention-4*; PyTorch release notes for torch 2.11,
   `torch.nn.attention.flex_attention` documentation.
9. HuggingFace Transformers: `PreTrainedModel`, `PretrainedConfig`,
   `AutoModel*` documentation.
