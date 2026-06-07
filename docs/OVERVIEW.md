# OPLM Technical Design Overview

> The single, comprehensive technical design document for OPLM (Open Protein
> Language Model). It distills the model architecture, the data pipeline, the
> evaluation harness, and the trainer into one source of truth. Where a section
> describes a "target" design, that is the design the rewritten `hf-compat`
> codebase converges on; divergences from older code are noted inline.
>
> Companion docs: [`CONFIG.md`](CONFIG.md) (field-by-field config reference),
> [`TRAIN.md`](TRAIN.md) (the practical training guide), and
> [`README.md`](../README.md) (inference quickstart). This document owns the
> *design*; those own *usage*.

---

## 0. System at a glance

OPLM is a HuggingFace-native, encoder-only (BERT/RoBERTa-style) protein language
model trained with masked-language modeling, plus the tooling to train and
evaluate it. Four subsystems compose cleanly along stable contracts:

```
┌───────────────┐   build_train_dataloader   ┌──────────────┐
│  data/        │ ─────────────────────────► │ training/    │
│  (tensors)    │   batch: input_ids,        │  Trainer     │
│               │   attention_mask, labels   │              │
└──────┬────────┘                            └──────┬───────┘
       │ builders/loaders                           │ EvalContext (frozen, rank-identical)
       │ (also consumed by eval)                    ▼
       │                                     ┌──────────────┐
       │   load_structures / variant /       │ eval/        │
       └───────────────────────────────────► │  Evaluator   │
                                             │  + tasks     │
                                             └──────┬───────┘
                                                    │ forward()/logits()
                                                    ▼
                                             ┌──────────────┐
                                             │ model/       │
                                             │  OplmModel   │
                                             └──────────────┘
```

**Module map** (under `src/oplm/`):

| Package | Owns |
| --- | --- |
| `model/` | The transformer, tokenizer, `OplmConfig` (HF `PretrainedConfig`), public API. |
| `data/` | On-disk formats → model-ready tensors, organized by modality; the single canonical tokenizer accessor. |
| `eval/` | Task registry, per-task scheduling, metric math; consumes `oplm.data`. |
| `training/` | The `Trainer` loop, optimizer/scheduler construction, FLOPs accounting, checkpointing, callbacks. |
| `config.py` | Root run config (`OplmConfig` = `{model, train, data}`), `load_config`, dataset-entry/schedule dataclasses. |
| `cli.py`, `train.py`, `inference.py` | Entry points and inference helpers. |

**Cross-cutting design principles** (recur in every subsystem):

1. **The trainer owns the clock; the harness owns the policy.** The trainer
   advances state and announces "step N / token M"; the eval harness decides what
   is due. Per-task cadence never leaks into the loop.
2. **One config schema per concern.** The model owns its hyperparameters via the
   HF `OplmConfig` (`PretrainedConfig`); the trainer/data own theirs via
   `TrainConfig`/`DataConfig`. The run config composes the three; it never
   re-describes the model.
3. **One tokenizer, one vocabulary.** `OplmTokenizerFast` is the source of truth
   for token IDs everywhere — training, eval, inference. Masking/scoring constants
   are *derived* from it, never hardcoded.
4. **Share by default; fork only across modalities.** Pretraining and sequence
   eval use the same parquet, tokenizer, and collation core; the differences
   (shuffle, seed, deterministic masking) are policy *parameters*, not separate
   code paths.
5. **Policy, not subclass.** Eval-specific behavior is constructor arguments on
   shared components — no `DeterministicMLMCollator`, no parallel "eval dataset."
6. **Typed config over loose dicts.** YAML enters as dicts at boundaries and is
   cast once into frozen, validated dataclasses.
7. **Accelerate is the only distribution layer.** No manual `torch.distributed`
   wiring in the trainer.

**Packaging.** Python ≥ 3.11; `torch ≥ 2.11` (pinned for the FlashAttention-4
backend on Blackwell — reached via `scaled_dot_product_attention` — and
`torch.optim.Muon`). Single `pip install oplm` installs everything — **no
optional dependency groups**. Build backend: `hatchling`, single `pyproject.toml`.

---

# Part I — Model Architecture

## 1. Philosophy

OPLM is a **vanilla pre-norm encoder transformer with RoPE and QK-norm**. The
default config reproduces a textbook modern PLM: LayerNorm everywhere (RMSNorm a
toggle), SwiGLU FFN, untied input/output embeddings, standard multi-head
attention (`num_heads == num_kv_heads`), residual streams scaled by `1/sqrt(L)`
per sublayer for depth stability, and a BERT-style MLP MLM head.

A small, curated set of **research toggles** layers on top, each independently
switchable from config and each a literature-backed ablation hypothesis (not an
open-ended knob):

- Canon-style bidirectional depthwise conv sublayers (arXiv 2512.17351)
- Partial RoPE / NoPE split (arXiv 2502.14837v1)
- Hybrid norm (arXiv 2503.04598)
- Sandwich norm
- Post-SDPA norm

With every toggle off (the default), the model is the textbook baseline.

**Explicitly removed** (tried, no consistent gain, not even available as toggles;
re-introduction requires a deliberate revisit): grouped-query attention
(`num_kv_heads < num_heads`), attention residuals (Kimi-style), cross-layer value
residuals (Proust-style), output gating (static/query-dependent), shared K/V
projection, multi-token value embeddings, query-dependent attention gates.

| Feature | Status | | Feature | Status |
| --- | --- | --- | --- | --- |
| Pre-norm, LayerNorm | Default | | RMSNorm | Toggle |
| RoPE (full), QK-norm | Default | | Partial RoPE / NoPE | Toggle |
| MHA (`H == H_kv`) | Default | | Hybrid / Sandwich / Post-SDPA norm | Toggle |
| SwiGLU FFN | Default | | Canon depthwise conv | Toggle |
| BERT-style MLM head | Default | | Tied embeddings | Toggle |
| Untied embeddings | Default | | GQA / value residuals / output gating | Removed |
| Residual scaling `1/sqrt(L)` | Default | | shared K/V / value embeddings | Removed |
| Init scaling `1/sqrt(2L)` on output projections | Default | | | |
| SDPA attention (+ manual softmax for weights) | Default | | | |

## 2. Public API

Two surfaces over the same objects: the **standard HuggingFace interface** and a
thin **ESM-C-style convenience layer** (sugar on top).

**HuggingFace auto-classes** (registered in `oplm/__init__.py`, no
`trust_remote_code` needed once `import oplm` has run):

| HF class | Resolves to |
| --- | --- |
| `AutoConfig` | `OplmConfig` |
| `AutoTokenizer` | `OplmTokenizerFast` |
| `AutoModel` | `OplmModel` |
| `AutoModelForMaskedLM` | `OplmForMaskedLM` |
| `AutoModelForSequenceClassification` | `OplmForSequenceClassification` |
| `AutoModelForTokenClassification` | `OplmForTokenClassification` |

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
model = AutoModelForMaskedLM.from_pretrained("brineylab/oplm-base")
tok   = AutoTokenizer.from_pretrained("brineylab/oplm-base")
inputs = tok(["MEEPQSDPSVEPPLSQ"], return_tensors="pt", padding=True)
out = model(**inputs, output_hidden_states=True)   # out.logits (B,T,V); out.hidden_states tuple
```

**ESM-C-style convenience layer** — three methods on every `Oplm*ForX` class
(defined on the `EsmcCompatMixin`). `tokenize`/`logits` carry the attention mask
correctly; `encode` exists only for ESM-C call-site parity:

| Method | Returns | Delegates to |
| --- | --- | --- |
| `model.tokenize(seqs, **kw)` | `BatchEncoding` (`input_ids` + `attention_mask`) | `tokenizer(seqs, return_tensors="pt", padding=True, **kw).to(device)` |
| `model.encode(seqs, **kw)` | `Tensor` of `input_ids` only | `tokenize(...).input_ids` |
| `model.logits(seqs, cfg)` | `LogitsOutput` | `tokenize` → `forward(...)` → repackage |

```python
from oplm import OplmForMaskedLM, LogitsConfig
model = OplmForMaskedLM.from_pretrained("brineylab/oplm-base")  # auto-attaches saved tokenizer
out = model.logits(["MEEPQSDPSVEPPLSQ", "GAGTRWPVQ"],
                   LogitsConfig(sequence=True, return_embeddings=True))
# out.sequence_logits (B,T,V)|None ; out.embeddings (B,T,D)|None (last hidden state)
# out.hidden_states tuple|None ; out.attentions tuple|None (forces fallback path)
```

`LogitsConfig` / `LogitsOutput` live in `src/oplm/model/outputs.py`:

```python
@dataclass
class LogitsConfig:
    sequence: bool = True
    return_embeddings: bool = False
    return_hidden_states: bool = False
    return_attentions: bool = False

@dataclass
class LogitsOutput:
    sequence_logits: Tensor | None
    embeddings: Tensor | None
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None
```

The convenience layer adds exactly **one** piece of model-attached state:
`self.tokenizer` (defaults to `None`). If `tokenize`/`encode`/`logits` is called
before a tokenizer is attached, the model raises (pointing at `from_pretrained`
or manual `model.tokenizer = ...`). There is **no** lazy
`AutoTokenizer.from_pretrained(config._name_or_path)` fallback — it would silently
break scratch models, offline use, and tests. `encode()` is a known footgun in
isolation (padded `input_ids` with no mask); prefer `tokenize`/`logits`.

## 3. Tokenizer & vocabulary

OPLM uses the **ESM-C 33-token vocabulary** in ESM-C order and with ESM-C
special-token IDs (matches `esm.utils.constants.esm3`). A batch tokenized for
ESM-C is **bit-for-bit identical** to one tokenized for OPLM, so switching between
the two is mechanical at the data layer. (The id-31 slot is `|` chain-break, as in
ESM-C, not ESM-2's `<null_1>`.)

| ID | Tok | | ID | Tok | | ID | Tok | | ID | Tok |
| -- | --- | -- | -- | --- | -- | -- | --- | -- | -- | --- |
| 0 | `<cls>` (BOS) | | 9 | E | | 18 | F | | 27 | Z (E/Q) |
| 1 | `<pad>` | | 10 | R | | 19 | Y | | 28 | O (pyrrolysine) |
| 2 | `<eos>` | | 11 | T | | 20 | M | | 29 | `.` (gap) |
| 3 | `<unk>` | | 12 | I | | 21 | H | | 30 | `-` (align gap) |
| 4 | L | | 13 | D | | 22 | W | | 31 | `\|` (chain break) |
| 5 | A | | 14 | P | | 23 | C | | 32 | `<mask>` |
| 6 | G | | 15 | K | | 24 | X (ambig) | | | |
| 7 | V | | 16 | Q | | 25 | B (D/N) | | | |
| 8 | S | | 17 | N | | 26 | U (Sec) | | | |

The **20 standard amino acids occupy the contiguous block IDs 4–23**
(`L,A,G,V,S,E,R,T,I,D,P,K,Q,N,F,Y,M,H,W,C`). Verification anchor:
`tok("MEEPQ").input_ids == [0, 20, 9, 9, 14, 16, 2]` (byte-identical to
`EsmTokenizer`).

**Rules:** per-character splitting against the AA alphabet (no BPE/SentencePiece);
each input wrapped `[<cls>, …, <eos>]`; unknown chars → `<unk>` (rare on clean
data; frequent `<unk>` is a data bug); padding with `<pad>`; truncation drops AA
tokens but keeps `<cls>`/`<eos>`; no casing transforms (input expected uppercase).

**HF integration.** `OplmTokenizerFast` subclasses `PreTrainedTokenizerFast`;
its Rust backend is a WordLevel tokenizer with `TemplateProcessing` for the
`<cls> … <eos>` wrap. `save_pretrained` writes `tokenizer.json`,
`tokenizer_config.json` (`tokenizer_class: "OplmTokenizerFast"`), and
`special_tokens_map.json`. Batch encode is fast enough for the training hot loop;
there is no second "fast" vocabulary to keep in sync.

## 4. Class hierarchy

```
OplmConfig(PretrainedConfig)            # model_type = "oplm"

OplmPreTrainedModel(PreTrainedModel)    # config_class, base_model_prefix="oplm",
  ├─ OplmModel                          #   main_input_name="input_ids",
  │     embeddings → N×OplmBlock → final norm   →  BaseModelOutput
  ├─ OplmForMaskedLM                    #   _no_split_modules=["OplmBlock"],
  │     OplmModel + MLM head (lm_head)          →  MaskedLMOutput
  ├─ OplmForSequenceClassification      #   _supports_sdpa=True,
  │     OplmModel + pooler + classifier        →  SequenceClassifierOutput
  └─ OplmForTokenClassification         #   _supports_flash_attn_2=False
        OplmModel + per-token classifier       →  TokenClassifierOutput
```

`OplmPreTrainedModel` wires `from_pretrained`/`save_pretrained`, weight init,
embedding tying, and the gradient-checkpointing toggle; it is never instantiated
directly. `supports_gradient_checkpointing = True`; `_no_split_modules` tells FSDP
to wrap `OplmBlock`s atomically. The four task classes inherit the
`EsmcCompatMixin` (`encode`/`logits`). All `Oplm*` classes live in
`src/oplm/model/modeling_oplm.py`; submodules in `attention.py`, `ffn.py`,
`norm.py`, `rope.py`, `conv.py`, `embedding.py`, `transformer.py`,
`configuration_oplm.py`, `tokenization_oplm.py`, `outputs.py`, `masking.py`.

## 5. Transformer block (`OplmBlock`)

The repeating unit. Default: pre-norm + LayerNorm, residual scaling on, no Canon,
full RoPE. "Norm" means the configured `norm_type` (LayerNorm default; RMSNorm
under `norm_type="rmsnorm"`).

Default block, with `α = 1/sqrt(L)` when `residual_scaling="sqrt_num_layers"`
(default) and `α = 1` when `"none"`:

```
h = x + α · Attn(Norm(x))
y = h + α · FFN(Norm(h))
```

The last block's output passes one **final norm** before the head.

**Why scale.** Each sublayer contributes `O(1/sqrt(L))` to the residual stream, so
the total from `2L` sublayers stays `O(sqrt(L))` rather than `O(L)` — the
load-bearing stability tool for deep models (target depths up to ~80 layers). The
spec uses `sqrt(L)` (owner preference); `sqrt(2L)` is an equally defensible
alternative differing only by `sqrt(2)`, revisitable at scale-up.

### 5.1 Normalization placement (`norm_strategy`, global)

Residual scaling composes orthogonally with placement.

| `norm_strategy` | Attention sublayer | FFN sublayer |
| --- | --- | --- |
| `pre` (default) | `h = x + α·Attn(Norm(x))` | `y = h + α·FFN(Norm(h))` |
| `sandwich` | `h = x + α·Norm₂(Attn(Norm₁(x)))` | `y = h + α·Norm₄(FFN(Norm₃(h)))` |
| `hybrid` (2503.04598) | `h = x + α·Attn_QKVNorm(x)` — **no outer pre-norm**; Q,K,V normed inside attention | `y = Norm(h) + α·FFN(Norm(h))` — one `Norm(h)` reused as FFN input *and* the FFN-side residual stream |
| `post_sdpa` | `h = x + α·Norm(Attn(Norm(x)))` | `y = h + α·FFN(Norm(h))` |

`hybrid` reproduces the paper's "QKV-Post" literally: the block-level attention
pre-norm is suppressed (raw `x` into attention) and V is normed in addition to Q
and K (see §6). For `pre`/`sandwich`/`post_sdpa`, QK-norm (Q and K only, V not
normed) is always on by default and orthogonal to `norm_strategy`; disable with
`qk_norm=false` for ablation. The final post-stack norm applies regardless.

### 5.2 Canon insertion points

When `canon_enabled=True`, depthwise 1D convs are inserted at any of four labeled
positions (Canon paper, arXiv 2512.17351):

- **A** — pre-block, on the residual-stream input.
- **B** — between attention pre-norm and Q/K/V projection (*discouraged*, weakest
  in the paper; accepted for ablation completeness).
- **C** — on the attention output before the residual add.
- **D** — between the FFN pre-norm and the FFN (most common positive result).

```
x ─► [Conv_A] ─► Norm ─► Attn ─► [Conv_C] ─► (+) ─► h
                                                     │
h ─► Norm ─► [Conv_D] ─► SwiGLU FFN ──────────► (+) ─► y
```

`canon_positions` is a global list (order irrelevant; each is an independent
site), e.g. `["A","C","D"]`. See §11.

### 5.3 Per-knob scope

| Knob | Scope | Notes |
| --- | --- | --- |
| `norm_type` | global | `layernorm` (default) / `rmsnorm` |
| `norm_strategy` | global | one enum, uniform |
| `qk_norm` | global | per-head Norm on Q,K |
| `residual_scaling` | global | `sqrt_num_layers` / `none` |
| `rope_dim`/`nope_dim` | global | same split across all heads/layers |
| `canon_enabled`/`canon_positions` | global | master switch / insertion sites |
| `canon_kernel_sizes` | per-layer | scalar broadcasts; list must match `num_hidden_layers` |
| `ffn_activation`, `tie_word_embeddings`, `mlm_head_activation`, `classifier_pool` | global | |

`post_sdpa` and `sandwich` are mutually exclusive enum values. Exotic
combinations (e.g. Canon-B + sandwich; `qk_norm=false` + `hybrid`) are **warned,
not errored**.

## 6. Attention (`OplmAttention`)

Standard MHA with `num_attention_heads == num_kv_heads` (no GQA). Two compute
paths share one set of parameters and pre-attention transforms; only the
score→softmax→value kernel differs.

Shapes: `x (B,T,D)` → Q/K/V projections `(B,T,D)` → reshape to heads `(B,H,T,d)`
with `d = D/H` → QK-norm `(B,H,T,d)` → RoPE on Q,K → attention out `(B,H,T,d)` →
reshape `(B,T,D)` → output projection `(B,T,D)`.

**QK-norm / QKV-norm.** Q and K each pass through a separate Norm over the
`d_head` channel dim; gain (and bias under LayerNorm) is shape `(d_head,)`,
**shared across heads**. Computed in **fp32** regardless of autocast. V is not
normed — *except* under `norm_strategy="hybrid"`, where an independent `v_norm`
`(d_head,)` is added (the "QKV-norm" formulation) and the block-level attention
pre-norm is suppressed. Under all other strategies `v_norm` is `nn.Identity()`.

**RoPE** applies to `Q_norm`, `K_norm` (after QK-norm, before scoring); not to V.
Partial RoPE rotates only the first `rope_dim` channels (see §7).

**Forward signature:**

```python
def forward(self, x, attention_mask, output_attentions=False):
    # x (B,T,D); attention_mask (B,T) {0,1} 1=real,0=pad
    # returns (out (B,T,D), attn_weights (B,H,T,T) fp32 | None)
```

**Path selection:** `output_attentions=False` (the default, used for training)
takes the SDPA path; `output_attentions=True` takes the manual softmax path,
since SDPA does not expose attention weights. QKV projection, QK/V-norm, RoPE,
and the output projection are shared:

```python
q, k, v = self._project_qkv(x)
q, k = self._qk_norm(q, k)
v = self._v_norm(v)               # Identity unless hybrid
q, k = self._apply_rope(q, k)
if output_attentions:
    out, attn = self._manual_attention(q, k, v, attention_mask)
else:
    out, attn = self._sdpa_attention(q, k, v, attention_mask), None
out = self._output_projection(out)
```

**SDPA path** calls `F.scaled_dot_product_attention(q, k, v, attn_mask=...)` with a
`(B, 1, 1, T)` boolean *key*-padding mask (`True` = attend). The encoder is
**bidirectional** — no causal mask, only the padding mask. Masking only keys keeps
every query row non-empty, so no row is all-masked and the softmax cannot NaN.
`dropout_p` is the configured `attention_dropout` during training (0 in eval), and
SDPA's default scale (`1/sqrt(d_head)`) matches the manual path. On CUDA this
dispatches to a fused FlashAttention / memory-efficient kernel (FlashAttention-4 on
Blackwell + torch ≥ 2.11); on CPU it uses the math backend. It returns no weights.

**Manual path** (`output_attentions=True`, manual matmul): `scores = (q @ kᵀ)/sqrt(d)`,
mask pad KV columns to `-inf`, `softmax(..., dtype=fp32)`, dropout, `@ v`. The
returned `attn` is the full fp32 softmax (for precision@L, attention rollout,
head-pruning). It is slower and `O(B·H·T²)` in memory — for inspection, not training.

**Output projection:** single `W_o: (D,D)` no bias; optional `hidden_dropout`
after (default 0.0).

**Params per attention block:** `4·D·D` (Q,K,V,O, no bias) `+ n_norm·2·d` (QK-norm
gains/biases, `n_norm=2` LayerNorm / `1` RMSNorm). RoPE caches are buffers.

## 7. Positional encoding (RoPE)

Rotary embedding (Su et al. 2021): pair the `d_head` channels into `d_head/2`
2-D pairs and rotate pair `i` at position `p` by `p·θ_i`, `θ_i = base^(-2i/d_head)`,
`base = rope_theta` (default 10000.0). Applied to Q,K post-QK-norm, per head, per
position; **fp32** then cast back. `cos`/`sin` are precomputed buffers up to
`max_position_embeddings` and extended on the fly for longer inference sequences
(no parameter update).

**Partial RoPE / NoPE** (arXiv 2502.14837v1): when `nope_dim > 0`,
`head_dim = rope_dim + nope_dim`; RoPE rotates only the first `rope_dim` channels,
the trailing `nope_dim` are position-invariant. `rope_dim + nope_dim == head_dim`
and `rope_dim` even are enforced at config time; default `nope_dim = 0` (full
RoPE). No YaRN/NTK extrapolation — beyond `max_position_embeddings` is best-effort.

## 8. Feed-forward (`OplmFFN`)

Default `ffn_activation="swiglu"`: `y = W_d(silu(W_g(x)) * W_u(x))` — three
linears, no bias by default (`ffn_bias=False`). Hidden dim `F = round_up(8/3·D,
256)` (≈ 8/3 multiplier, rounded up to a multiple of 256 for matmul-friendly
shapes); set `intermediate_size` explicitly to override. `geglu` is reserved as a
future gated variant (also 3 projections). Params per FFN: `3·D·F`. The
`ffn_activation` enum is the single extension point — new activations add an enum
value + a small class; the block is untouched.

## 9. Normalization

`norm_type` selects one operator used at **every** norm site (mixing is
unsupported). Both compute internally in **fp32** regardless of autocast.

- **LayerNorm** (default): `γ·(x−mean)/sqrt(var+eps) + β`; learnable `γ,β` shape
  `(D,)` or `(d_head,)`, init 1 and 0; `eps = norm_eps` (default 1e-6). Default
  because it matches ESM-C and the centering step mildly helps stability.
- **RMSNorm**: `γ·x/sqrt(mean(x²)+eps)`; gain only, no bias; half the parameters
  per site, slightly cheaper.

**All norm sites** (γ, plus β under LayerNorm): optional post-embedding norm
(`D`), attention pre-norm (`D`, omitted under `hybrid`), QK-norm Q (`d_head`),
QK-norm K (`d_head`), V-norm (`d_head`, only under `hybrid`), FFN pre-norm (`D`),
optional sandwich/post-SDPA post-norms (`D`), final stack norm (`D`), MLM-head
intermediate norm (`D`), optional pre-head norm (`D`). QK-norm gains are 2L total
(one per layer per Q-or-K), each `(d_head,)` shared across heads.

## 10. Embeddings & heads

**Token embedding:** `nn.Embedding(vocab_size=33, D)`; no positional embedding
(RoPE handles position); optional `post_embed_norm` off by default; init
truncated-normal `std = initializer_range` (0.02), `<pad>` row not zeroed (HF
convention).

**MLM head** (BERT/RoBERTa two-layer MLP, `self.lm_head`):

```python
x = self.dense(x)        # Linear(D, D), bias=True
x = self.act(x)          # mlm_head_activation, default GELU
x = self.norm(x)         # Norm(D), follows norm_type
return self.decoder(x)   # Linear(D, vocab_size), bias=True
```

`tie_word_embeddings` is **off by default**: the decoder is its own `Linear(D,V)`.
When `True`, `decoder.weight` ties to `embed_tokens.weight` (the decoder bias stays
independent). Untied is default because the 33-token vocab makes tying savings
trivial (~25K params at D=768) while an independent projection historically helps
MLM slightly. Loss: `cross_entropy(logits.view(-1,V), labels.view(-1),
ignore_index=-100)`, computed inside `OplmForMaskedLM.forward`.

**Sequence-classification head:** pool (`mean` over `attention_mask==1` positions,
default; or `cls` = position 0) → dropout (`classifier_dropout`) → `Linear(D,
num_labels)`. Optional `pre_head_norm` off by default.

**Token-classification head:** dropout → `Linear(D, num_labels)` per token, no
pooling.

| Head | Input | Output | Class |
| --- | --- | --- | --- |
| MLM | last hidden `(B,T,D)` | `(B,T,V)` | `OplmForMaskedLM` |
| Seq classifier | last hidden + mask | `(B,num_labels)` | `OplmForSequenceClassification` |
| Token classifier | last hidden `(B,T,D)` | `(B,T,num_labels)` | `OplmForTokenClassification` |

Pooling helpers for downstream embedding extraction live in `model/embedding.py`.

## 11. Canon depthwise conv layers

Operator: `nn.Conv1d(D, D, kernel_size=k, groups=D, padding=k//2, bias=False)` —
depthwise (per-channel). Odd `k` → symmetric same-length output; even `k` → pad
`k//2` left, `k//2 - 1` right (documented to avoid drift). Optional pointwise
activation (`canon_activation`: `none`/`silu`/`gelu`, default `none`). Input
layout `(B,T,D)`; the conv transposes to `(B,D,T)` internally and back.

**Bidirectional** — the kernel sees past and future; no causal masking (unlike
Canon's decoder usage). **Padding-leakage guard:** before each conv the input is
zeroed where `attention_mask==0` (`x = x * attention_mask.unsqueeze(-1)`) so pad
content cannot smear into real tokens — the model's responsibility, not the
caller's.

**Kernel schedule** (`canon_kernel_sizes`): a scalar (broadcast), a list of length
`num_hidden_layers`, or a dict `{schedule:"linear", min, max}` /
`{schedule:"constant", value}` resolved at config-load. Every kernel ≥ 2; even
kernels allowed. Cost per conv: `params = k·D`, `flops ≈ 2·k·D·T`.

## 12. Configuration schema (`OplmConfig`)

`OplmConfig` subclasses `PretrainedConfig`; the YAML `model:` block maps 1:1 to
its kwargs. Derived fields (`head_dim`, `intermediate_size`, `rope_dim`,
`nope_dim`) resolve from `None` via `_resolve_derived_fields`; `_validate` raises
`ValueError` on bad combinations.

| Field | Default | Notes |
| --- | --- | --- |
| `model_type` | `"oplm"` | auto-class key; do not override |
| `vocab_size` | 33 | ESM-C vocab; **warns** (not errors) if changed |
| `hidden_size` (D) | 768 | |
| `num_hidden_layers` (L) | 12 | |
| `num_attention_heads` (H) | 12 | must divide `hidden_size`; **no GQA** |
| `head_dim` | derived | `D/H` if unset |
| `intermediate_size` (F) | derived | `round_up(8/3·D, 256)` if unset |
| `max_position_embeddings` | 1024 | RoPE buffer size |
| `rope_theta` | 10000.0 | RoPE base |
| `rope_dim` / `nope_dim` | `head_dim` / 0 | `rope_dim+nope_dim==head_dim`, `rope_dim` even |
| `norm_type` | `layernorm` | `layernorm` / `rmsnorm` |
| `norm_eps` | 1e-6 | |
| `norm_strategy` | `pre` | `pre`/`sandwich`/`hybrid`/`post_sdpa` |
| `qk_norm` | `True` | per-head Norm on Q,K |
| `post_embed_norm` | `False` | |
| `residual_scaling` | `sqrt_num_layers` | / `none` |
| `init_scale_output_projections` | `True` | divide init std of `W_o`,`W_d` by `sqrt(2L)` (GPT-2 style) |
| `ffn_activation` | `swiglu` | / `geglu` (reserved) |
| `ffn_bias` | `False` | |
| `attention_dropout` | 0.0 | any >0 forces fallback path |
| `hidden_dropout` | 0.0 | after attn/FFN output projections |
| `tie_word_embeddings` | `False` | |
| `mlm_head_activation` | `gelu` | `gelu`/`silu`/`relu` |
| `canon_enabled` | `False` | |
| `canon_positions` | `[]` | subset of `{A,B,C,D}` |
| `canon_kernel_sizes` | 4 | int / list / dict schedule |
| `canon_activation` | `none` | `none`/`silu`/`gelu` |
| `initializer_range` | 0.02 | truncated-normal std |
| `classifier_pool` / `classifier_dropout` / `num_labels` / `pre_head_norm` | `mean` / 0.0 / 2 / `False` | task-head fields |
| `gradient_checkpointing` | `False` | activation checkpointing on `OplmBlock` |
| `gradient_checkpointing_mode` | `full` | `full` (recompute whole block) \| `selective` (SAC: keep matmul/SDPA, recompute cheap ops) |
| `pad`/`bos`/`eos`/`unk`/`mask_token_id` | 1 / 0 / 2 / 3 / 32 | ESM vocab |
| `auto_map` | filled by `save_pretrained` | see §13 |

**Validation rules** (raise `ValueError`): `hidden_size % num_attention_heads ==
0`; `head_dim·H == hidden_size` (if set); `rope_dim+nope_dim==head_dim`,
`rope_dim,nope_dim ≥ 0`, `rope_dim` even; `norm_type`/`norm_strategy`/
`residual_scaling`/`ffn_activation`/`mlm_head_activation`/`classifier_pool` in
their enums; if `canon_enabled` then `canon_positions` non-empty ⊆ `{A,B,C,D}` and
resolved `canon_kernel_sizes` is a length-`num_hidden_layers` list of values ≥ 2.
`vocab_size != 33` warns only. Deprecation policy: a deprecated field is readable
for one minor version with a `DeprecationWarning`, then removed (currently none).

## 13. HuggingFace integration, initialization & runtime contract

**Weight init.** Linear/embedding weights are truncated-normal `std =
initializer_range` (0.02). When `init_scale_output_projections=True` (default),
the init std of the attention output projection `W_o` and the FFN down projection
`W_d` is divided by `sqrt(2L)` (GPT-2 residual-init style) — this is partly
redundant with `residual_scaling="sqrt_num_layers"` and can be disabled as an
ablation. Norm gains init to 1 (LayerNorm bias to 0). The `<pad>` embedding row is
not zeroed (HF convention).

**Two integration mechanisms**, both configured by the model-repo maintainer; end
users pick one:

1. **In-process registration** — `import oplm` registers the auto-classes
   (`AutoConfig.register("oplm", OplmConfig)`, the four `AutoModel*.register(...)`
   calls, and `AutoTokenizer.register(OplmConfig,
   fast_tokenizer_class=OplmTokenizerFast)`), so `from_pretrained` works with **no
   `trust_remote_code`**.
2. **`auto_map`** — for loading a checkpoint *without* importing OPLM first: the
   checkpoint's `config.json` points HF at the in-repo `modeling_oplm.py` /
   `configuration_oplm.py` / `tokenization_oplm.py` (copied on `push_to_hub` /
   `save_pretrained`), and `trust_remote_code=True` is required.

**Required subclass attributes** on `OplmPreTrainedModel`: `config_class =
OplmConfig`, `base_model_prefix = "oplm"`, `main_input_name = "input_ids"`,
`supports_gradient_checkpointing = True`, `_no_split_modules = ["OplmBlock"]`,
`_supports_sdpa = True`, `_supports_flash_attn_2 = False` (attention runs through
`scaled_dot_product_attention`).

**Gradient checkpointing.** `config.gradient_checkpointing=True` arms
`OplmStack`/`OplmBlock` at init; `model.gradient_checkpointing_enable()` is the
HF-idiomatic call that propagates to every block. `config.gradient_checkpointing_mode`
(`full` | `selective`) selects the flavor: `full` recomputes the whole block on
backward (max memory savings), while `selective` uses PyTorch Selective Activation
Checkpointing (`create_selective_checkpoint_contexts` + a `CheckpointPolicy`) to
keep matmul/SDPA outputs resident and recompute only cheap ops — less memory
savings for substantially less recompute. The mode can also be passed at enable
time: `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"mode": "selective"})`.
Both modes are `torch.compile`-compatible; the SAC policy is a module-level
function so Dynamo can trace it through the checkpoint higher-order op. Under
multi-GPU DDP + compile, the trainer disables DDPOptimizer for `selective` so its
graph-splitting doesn't fragment the SAC op into a full recompute (see
[TRAIN.md](TRAIN.md)).

**Save/load round-trip.** `save_pretrained` writes `config.json` +
`model.safetensors` (honoring tied weights); attaching and saving the tokenizer
makes `from_pretrained(<dir>)` round-trip the model *and* its tokenizer. State-dict
keys follow natural module nesting (`oplm.embed_tokens.weight`,
`oplm.layers.0.self_attn.q_proj.weight`, …). With tying on, only
`embed_tokens.weight` is stored and `lm_head` is re-tied at load in
`tie_weights()`. Sharded loading uses the standard HF `safetensors` index (no
custom layout). The auto-registration (`register_for_auto_class(...)` for each
class at import in `src/oplm/__init__.py`) is what makes `push_to_hub` /
`save_pretrained` (a) write the `auto_map` entries into `config.json` /
`tokenizer_config.json` and (b) copy the modeling/configuration/tokenization `.py`
files into the repo (HF copies only a module's *depth-1* relative imports, so
`modeling_oplm.py` imports every helper directly — `_REMOTE_CODE_DEPS` references
the otherwise-unused names). The clean-env round-trip
(`save_pretrained` → load in a subprocess with `transformers` only +
`trust_remote_code=True`) is a mandatory integration test.

### 13.7 Forward-pass contract

All four `Oplm*ForX` classes accept the same kwargs (unneeded ones ignored):

| Argument | Type / shape | Default | Notes |
| --- | --- | --- | --- |
| `input_ids` | `(B,T)` int64 | required | from `OplmTokenizerFast` |
| `attention_mask` | `(B,T)` int64 {0,1} | `None` → all-ones | 1=real, 0=pad |
| `labels` | `(B,T)` / `(B,)` int64 / `None` | `None` | MLM (`-100` ignored), per-token, or per-sequence depending on head |
| `output_attentions` | bool | `False` | `True` forces the fallback path globally |
| `output_hidden_states` | bool | `False` | return all intermediate states |
| `return_dict` | bool | `True` | `ModelOutput` vs tuple |

| Class | Return | Logits |
| --- | --- | --- |
| `OplmModel` | `BaseModelOutput` | `last_hidden_state (B,T,D)` |
| `OplmForMaskedLM` | `MaskedLMOutput` | `(B,T,V)` |
| `OplmForSequenceClassification` | `SequenceClassifierOutput` | `(B,num_labels)` |
| `OplmForTokenClassification` | `TokenClassifierOutput` | `(B,T,num_labels)` |

Each carries `loss` (when `labels` given), `hidden_states`, `attentions`. Loss:
MLM/token = `cross_entropy(logits.view(-1,C), labels.view(-1), ignore_index=-100)`;
sequence = `cross_entropy(logits, labels)`. `output_hidden_states=True` returns a
tuple of `L+1` tensors (post-embedding, then each block's post-residual output),
all `(B,T,D)`. `output_attentions=True` switches **every** layer to the fallback
path and adds ≈`B·L·H·T²` fp32 floats of activation memory (~6 GB at
`T=1024,L=12,H=12,B=8`) — do not enable casually in training.

### 13.8 Numerical precision, determinism & distributed constraints

**Dtype contract.** Training: bf16 autocast on CUDA (Accelerate manages fp32
master weights); norms (incl. QK-norm), attention softmax, and RoPE compute in
**fp32** internally then cast back; logits and loss are **fp32**. Inference:
default load bf16 on CUDA / fp32 on CPU (override with `torch_dtype=`); use
`model.eval()` + `torch.inference_mode()`. fp32-in-norms removes bf16
sum-of-squares drift seen in long-context training.

**Determinism.** The **fallback path** is bitwise reproducible given seeded inputs
and identical hardware — use it for precision-critical eval (contact P@L, rollout,
ablation deltas). The **fast path** compiles a kernel that can differ across
torch/CUDA/arch versions (drift typically `<1e-3` relative, no bit equality);
pin every version for paper-grade numbers. Seeding is the trainer's job.

**Model-side FSDP/DDP constraints:** no global mutable state (RoPE cos/sin are
registered buffers, extended via a wrap-aware method); `_no_split_modules =
["OplmBlock"]`; weight tying re-applied in `tie_weights()` post-load; no in-place
ops on shared buffers; `gradient_checkpointing=True` wraps each block's forward in
`torch.utils.checkpoint` (selective mode adds an SAC `context_fn`). The model never
calls `torch.compile` itself — that is the trainer's responsibility.

### 13.9 Parameter count & size presets

Init (`_init_weights`): truncated-normal `std=initializer_range` (0.02) for
Linear/Embedding/Conv1d weights; biases 0; norm gains 1, LayerNorm biases 0.
Residual-stream-writing projections (attention `o_proj`, FFN `down_proj`, tagged
`_is_residual_writer`) get `std = initializer_range / sqrt(2L)` when
`init_scale_output_projections=True`. The decoder is not a residual writer (no
`1/sqrt(2L)`).

Parameter count (`n_norm = 2` LayerNorm / `1` RMSNorm; residual scaling is
parameter-free): `embedding V·D` + `final norm n_norm·D` + `MLM head (D·D+D) +
n_norm·D + (D·V+V untied | V tied)` + per layer `[attn 4·D·D + QK-norm 2·n_norm·d
+ pre-norm n_norm·D (+ extra norms per strategy)]` + `[FFN 3·D·F + pre-norm
n_norm·D]` + per Canon conv `k_l·D`. Worked: D=768, L=12, H=12, F=2048, V=33,
LayerNorm, untied, pre-norm → ≈ **85.6 M**.

Published presets (`--preset`; full recipes in
`src/oplm/configs/model/presets/*.yaml`; all share the 33-token tokenizer and a
1024-position context):

| Preset / Hub id | Params | Layers | Hidden | Heads | Head dim |
| --- | --: | --: | --: | --: | --: |
| `oplm-small` | 5.2M | 6 | 256 | 4 | 64 |
| `oplm-medium` | 85.6M | 12 | 768 | 12 | 64 |
| `oplm-base` | 309.5M | 24 | 1024 | 16 | 64 |
| `oplm-large` | 2.5B | 32 | 2560 | 32 | 80 |
| `oplm-xlarge` | 12.7B | 40 | 5120 | 40 | 128 |

`large`/`xlarge` additionally enable `gradient_checkpointing`. `oplm info --preset
<name>` prints the resolved architecture and exact parameter count.

---

# Part II — Data Tooling (`src/oplm/data/`)

## 14. Layered architecture

Organized **by modality, not by phase**. Four layers, each consuming only the one
below; the eval harness imports `oplm.data.*` and never reaches around a layer.
There is **no** `src/oplm/eval/data/` directory.

```
Layer 4  BUILDERS (policy: train vs eval)
  build_train_dataloader · build_sequence_eval_dataloader · load_structures · load_variant_assays
Layer 3  COLLATION
  pad/tokenize primitive  ──►  MLM-mask layer (stochastic | deterministic)
Layer 2  DATASETS (per modality)
  ShardedProteinDataset · InterleavedDataset · StructureData · VariantAssay
Layer 1  TOKENIZER (single source of truth)
  OplmTokenizerFast + derived id constants
```

Module tree:

```
src/oplm/data/
  __init__.py     # re-exports builders + tokenizer accessor
  tokenizer.py    # canonical-tokenizer accessor + derived id constants (no vocab of its own)
  config.py       # parse_train_configs / parse_eval_configs (entry dataclasses live in oplm/config.py)
  sequence/  dataset.py (Sharded/Interleaved) · collate.py (pad primitive + MLM mask) · loaders.py (builders)
  structure/ loader.py  # PDB/CIF → StructureData (lazy biopython)
  variant/   loader.py  # ProteinGym/EVEREST CSV → VariantAssay
  downstream/loader.py  # labeled-sequence tasks (TAPE, ProteinGLUE)
```

What the eval harness imports: sequence MLM → `build_sequence_eval_dataloader`;
variant → `variant.loader` + tokenizer + pad primitive; structure →
`structure.loader.load_structures` + tokenizer; downstream → `downstream.loader` +
tokenizer + pad primitive. The harness owns task logic/scoring/metrics; it never
owns loading.

## 15. Tokenizer as single source of truth

The data layer uses **one** tokenizer — `OplmTokenizerFast` (§3) — so token IDs
produced for data are exactly the IDs the model's embedding table expects.
`data/tokenizer.py` is a thin accessor (`get_tokenizer()`) exposing **derived id
constants** so they can never drift from the vocabulary:

| Constant | Definition | Value |
| --- | --- | --- |
| `special_ids` | `tokenizer.all_special_ids` | {0,1,2,3,32} |
| `non_maskable_ids` | specials never used as a target | {0,1,2,3,32} |
| `mask_token_id` | `tokenizer.mask_token_id` | 32 |
| `pad_token_id` | `tokenizer.pad_token_id` | 1 |
| `canonical_aa_ids` | the 20 standard AAs (contiguous) | `range(4, 24)` |

`canonical_aa_ids` is the random-replacement sampling pool (§16) and the
masked-marginal scoring pool for eval — exposed via one helper
(`canonical_amino_acid_ids(tokenizer)`) reused by collator and metrics alike.

**Aligning per-residue vectors.** Features attached per *residue* (notably
per-position masking weights, §16.1) must follow the same truncation, `<cls>`/
`<eos>` insertion, and padding as `input_ids`. `align_per_residue(values, *,
fill_special, fill_pad, ...)` lives in the tokenizer layer (next to the
truncation/templating rules it mirrors) and returns a `(B,T)` tensor aligned to
the tokens.

> **Historical bug (resolved).** The legacy `ProteinTokenizer` defined a 32-token
> vocab offset by one (`L`=5 vs canonical 4; `<mask>`=4 vs 32) while the model
> embeds the canonical 33-token vocab — so the old data path produced IDs that did
> not align with the embedding rows. This was a correctness bug; the second
> vocabulary is **removed**.

## 16. Sequence modality

The pretraining path and substrate for sequence eval.

**On-disk format.** A single parquet (`.parquet`/`.parq`/`.pq`) or a directory of
shards. Required columns: `sequence_id` (str), `sequence` (str, raw one-letter
AAs, no special tokens). Optional `masking_weights` (`list[float]`, one per
residue) read only when `data.weighted_masking` is on. Only needed columns are
read (`pq.read_table(columns=[...])`).

**`ShardedProteinDataset`** — an `IterableDataset` yielding `{"sequence_id",
"sequence"}` (plus `"masking_weights"` when enabled). Shard discovery at
construction reads per-shard row counts from parquet metadata (no data loaded);
`total_length` is their sum. Per-epoch deterministic shuffle via `set_epoch(epoch)`
(epoch seed mixed from base `seed` + epoch index using a golden-ratio constant and
a large prime); same `(seed, epoch)` ⇒ identical order across runs and ranks. Two
independent shuffle granularities: shard order (`shuffle_shards`) and within-shard
row order (`shuffle_rows`, seeded per shard). **Explicit rank-aware striping** over
the joint `(world_rank, worker_id)` index — disjoint, no duplication, no gaps,
independent of launcher behavior.

**`InterleavedDataset`** — mixes datasets by sampling fraction (normalized to 1.0;
validated ≥0, sum >0). Each step picks a source by fraction then pulls its next
item; exhausted sources re-initialize so unequal sizes keep mixing at the target
ratio for the whole epoch. `set_epoch` propagates; RNG seeded per `(epoch,
worker/rank)`.

**Collation — two composable pieces:**

- **Pad/tokenize primitive** `tokenize_and_pad(batch)`: truncates each raw
  sequence to `max_length - 2`, tokenizes with the canonical tokenizer, pads to the
  batch max with `pad_token_id`, returns `{input_ids, attention_mask}` — **no
  masking, no labels**. Used by variant/structure/downstream consumers. Also
  exposes `align_per_residue` (§15).
- **`MLMCollator`** — calls the primitive, applies masking, adds `labels`.
  Constructor: `mask_prob=0.15`, `mask_token_prob=0.8`, `random_token_prob=0.1`,
  `weighted_masking=False`, `deterministic=False`, `seed=0`, `max_length=1024`.

### 16.1 MLM masking scheme

**Dynamic (RoBERTa-style):** masks are regenerated in the collator every time an
example is drawn (same sequence masked differently across epochs). Only the
**replacement split** (80/10/10) is borrowed from BERT. Constants derive from the
tokenizer (§15).

1. **Eligibility** — a position is maskable iff `attention_mask==1` and its id is
   not in `non_maskable_ids`.
2. **Selection** — a **fixed count** `k = round(mask_prob · n_eligible)` positions
   chosen by **weighted sampling without replacement** via Gumbel-top-k. Uniform
   masking is the special case (all weights equal). Gumbel noise resampled per draw
   → positions vary across epochs; only `k` is fixed.
3. **Targets** — `labels` holds the original id at selected positions, `-100`
   elsewhere.
4. **Replacement** — `mask_token_prob` (0.8) → `<mask>` (32); `random_token_prob`
   (0.1) → uniform id from `canonical_aa_ids` (4–23 only; ambiguous/gap/structure
   tokens excluded); remainder (0.1) → keep original.

**Gumbel-top-k** (Efraimidis–Spirakis): per eligible position `i` with weight
`w_i ≥ 0`, `key_i = log(w_i) + g_i`, `g_i = -log(-log(u_i))`, `u_i ~ U(0,1)`; mask
the `k` largest keys. First-order inclusion ∝ `w_i`; sampling without replacement;
fixed count regardless of weight scale. `w_i=0` is never selected; weights are
relative (scale-invariant); if positive-weight positions < `k`, all of them (only
them) are masked.

**Per-position weighted masking** is gated by `data.weighted_masking` (default
`False`) — *not* by mere column presence. When off, `masking_weights` is ignored
even if present and masking is uniform. Raw per-residue weights flow through
`align_per_residue` (0.0 at `<cls>`/`<eos>`/pad). Fallbacks: a row with `None`
weights → uniform 1.0; an entirely-absent column → warn once, fall back to
uniform; a length-mismatched weight array → **error** (data-integrity bug surfaced
loudly).

**Determinism for eval (policy, not a fork).** With `deterministic=True` the
collator derives a per-batch RNG state from `seed + batch_index` before masking and
restores the ambient RNG after, so the same batch always gets the same mask —
making MLM eval comparable across steps. Training uses `deterministic=False`; the
sequence-eval builder sets `deterministic=True` and disables shuffling. No separate
collator class.

### 16.2 Batch contract

Every sequence batch (train and eval) has exactly:

| Key | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `input_ids` | `(B,T)` | long | canonical IDs, `<cls>…<eos>`, padded |
| `attention_mask` | `(B,T)` | long | 1=real, 0=pad |
| `labels` | `(B,T)` | long | original id at masked positions, else `-100` |

`B = train.batch_size`; `T` = batch-max length, capped at
`model.max_position_embeddings`. `masking_weights` (when used) are consumed
*inside* the collator and never emitted. The pad primitive emits only the first two
keys.

## 17. Structure modality

3-D structures → backbone coordinates, consumed by the contact-prediction eval
task; sequences reuse the canonical tokenizer with **no masking**.

Input: a directory of `*.pdb`/`*.cif`/`*.ent`/`*.mmcif`, parsed with BioPython.

```python
@dataclass
class StructureData:
    name: str               # PDB id / filename stem
    sequence: str           # one-letter AAs
    coords: Tensor          # (L,3,3) backbone N,CA,C; NaN for missing atoms
    chain_id: str | None

load_structures(directory, max_structures=None) -> list[StructureData]
```

Parsing policy: first model, first chain (model 0 X-ray / first NMR conformer);
modified residues mapped to canonical parents (`MSE→M`, `SEP→S`, `PTR→Y`); other
heteroatoms skipped; missing backbone atoms → `NaN` rows so `(L,3,3)` stays aligned
with `sequence`; unparseable files skipped with a warning; results sorted by
filename for determinism. The contact task tokenizes `sequence` via the pad
primitive (mask all ones, no padding for a single sequence), runs the model, and
compares Jacobian/attention-derived contacts to the geometric map from `coords` —
all metric math lives in the harness.

BioPython is imported **lazily** inside the parser so `import oplm.data` never
hard-requires it.

## 18. Variant modality

Zero-shot variant-effect prediction (ProteinGym, EVEREST). Tokenize + pad, **no
MLM masking** — scoring uses position-specific masking.

Input: a directory of CSVs (one per assay) with `mutant` (e.g. `"A42T"`, or
`:`-joined multi-mutants) and `DMS_score` (float); the wild-type sequence is
supplied per assay (metadata/sidecar/config `extra`).

```python
@dataclass
class VariantAssay:
    name: str
    wildtype: str            # one-letter WT sequence
    mutations: list[str]     # raw mutant strings
    labels: list[float]      # DMS_score per row
```

**Marginal scoring** (not batch-MLM): encode WT once; for each mutated position
mask that single residue (`<mask>`), run the model, read log-probs for WT and
mutant AAs (masked-marginal; a WT-marginal variant reads from the unmasked pass).
Variant score = `Σ [log P(mutant_aa) − log P(wt_aa)]` over the mutation set. The
masking is deterministic and position-specific (not `MLMCollator`); it reuses the
tokenizer and the canonical-AA helper for logit indexing. The loader's job ends at
`VariantAssay`; the harness produces per-assay predictions and computes Spearman /
NDCG / AUROC against `labels`.

## 19. Downstream / embedding modality

Supervised benchmarks (TAPE, ProteinGLUE). Tokenize + pad, **no MLM masking**; the
model is a frozen embedder feeding a lightweight head. Per-residue tasks (SS3/SS8,
contacts) and sequence-level tasks (fluorescence/stability regression;
fold/enzyme/GO classification) stored as parquet/CSV with a `sequence` column plus
label column(s). Sequences go through the pad primitive; the harness extracts
pooled (mean/CLS) or per-residue representations and trains a small head (pooling
helpers in `model/embedding.py`; label handling in the harness).

| Task family | Label tensor | Notes |
| --- | --- | --- |
| per-residue | `(B,T)` long / `(B,T,…)` | aligned to non-special positions; pad with `-100` |
| seq-level regression | `(B,)` float | one scalar per sequence |
| seq-level classification | `(B,)` long | class index per sequence |

## 20. `DataConfig` & builders

**`DataConfig`** (in `src/oplm/config.py`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `train` | str/dict/None | None | training dataset(s); `parse_train_configs` |
| `eval` | dict/None | None | eval datasets by name → `{path, type, …}`; `parse_eval_configs` |
| `mask_prob` | float | 0.15 | MLM selection probability |
| `mask_token_prob` | float | 0.8 | masked → `<mask>` |
| `random_token_prob` | float | 0.1 | masked → random AA |
| `weighted_masking` | bool | False | honor `masking_weights` column (else ignored) |
| `num_workers` / `pin_memory` / `prefetch_factor` | int/bool/int | 4 / True / 4 | DataLoader |
| `shuffle_shards` / `shuffle_rows` | bool | True / True | shuffle granularities |

Consumed from siblings: `model.max_position_embeddings` (collator `max_length`),
`train.batch_size` (DataLoader batch), `train.seed` (RNG), `train.eval_every`
(default eval cadence). Validation: `mask_token_prob, random_token_prob ∈ [0,1]` and
their sum ≤ 1; `weighted_masking` lives in `DataConfig` so the whole masking policy
is one block.

**Dataset entries** (dataclasses in `config.py`): `TrainDatasetEntry{name, path,
fraction}`; `EvalDatasetEntry{name, path, type, schedule: ScheduleSpec, metrics,
extra}`. `parse_train_configs` accepts a path string or `{name:{path,fraction}}`
(fractions normalized; omitted ones split the remainder equally).
`parse_eval_configs` parses `{name:{path,type,…}}`, requires `path`+`type`, and
folds unknown keys into `extra`.

**Builders** (`data/sequence/loaders.py`):

```python
build_train_dataloader(cfg)            # ShardedProteinDataset(s) → optional Interleaved
                                       # → MLMCollator(deterministic=False) → DataLoader
build_sequence_eval_dataloader(path, cfg)  # same machinery, eval policy
```

The train/eval difference is **only** these arguments:

| Parameter | Training | Sequence eval |
| --- | --- | --- |
| `shuffle_shards`/`shuffle_rows` | from `cfg.data` | `False` |
| `deterministic` | `False` | `True` |
| mask probability | `cfg.data.mask_prob` | fixed eval value |
| batch contract | §16.2 | identical |

Modality → builder/task map: sequence → `build_*_dataloader` (`sequence`);
structure → `load_structures` (`structure`); variant → `variant.loader`
(`proteingym`, `everest`); downstream → `downstream.loader` (`tape`,
`proteinglue`).

---

# Part III — Evaluation Harness (`src/oplm/eval/`)

## 21. Architecture & the trainer↔eval contract

The trainer builds one immutable `EvalContext` per optimizer step and hands it to
one `Evaluator`, which asks each task's `Schedule` whether it is due, runs the due
tasks, and returns a flat namespaced metrics dict. Tasks read via `oplm.data` and
score with `eval/metrics/`.

```
src/oplm/eval/
  context.py    # EvalContext (frozen)
  schedule.py   # Schedule protocol + EveryNSteps, EveryNTokens (_crossed helper)
  evaluator.py  # Evaluator.run_due
  registry.py   # @register_eval_task + get_eval_task_class + EVAL_TASK_REGISTRY
  tasks/        # base.py (EvalTask ABC) · sequence · structure · proteingym/tape/proteinglue/everest (stubs)
  metrics/      # mlm.py · contact.py · categorical_jacobian.py
```

**`EvalContext`** — the one value crossing the boundary, frozen, with cumulative
counters, per-step deltas, and lifecycle flags:

```python
@dataclass(frozen=True)
class EvalContext:
    global_step: int    # cumulative optimizer steps
    epoch: int          # cumulative epochs (carried for future epoch cadence)
    tokens_seen: int    # cumulative GLOBAL tokens — rank-reduced (§22)
    steps_delta: int    # steps since previous context (== 1)
    tokens_delta: int   # GLOBAL tokens this optimizer step (rank-reduced)
    epoch_delta: int    # epoch boundaries crossed this step (0/1; future use)
    is_final: bool      # global_step >= total_steps
```

`epoch`/`epoch_delta` are present now so epoch cadence later needs no contract
change; no first-cut schedule reads them. The trainer builds it once per optimizer
step from state it already tracks; `epoch_delta` is snapshotted per optimizer step
(not per micro-batch) so gradient accumulation can't double-count an epoch
boundary.

**The rank-sync invariant (load-bearing).** Eval tasks run collective ops inside
`evaluate` (`accelerator.reduce` to sum loss; `dist.all_gather_object` to gather
per-structure results). A collective completes only when every rank enters it, so a
per-rank `is_due` disagreement would hang at the NCCL timeout. The invariant:

> **Every field of `EvalContext` is identical on every rank**, so each rank
> independently computes the same `is_due` for every task — no communication, no
> disagreement.

This is why `tokens_seen`/`tokens_delta` must be rank-reduced (§22) and why
schedules must be pure functions of the context. The invariant is tested directly.

## 22. Scheduling

**Strategy, not branch.** "When to run" and "what to compute" are orthogonal. A
task owns one `Schedule`; the `Evaluator` never grows a per-cadence conditional.
Schedules are **pure functions of synchronized state** — no internal counters,
hence resume-safe with nothing to persist.

```python
class Schedule(Protocol):
    def is_due(self, ctx: EvalContext) -> bool: ...
```

**The crossing test (why not modulo).** For a counter advancing by `delta`:

```python
def _crossed(curr, delta, n):
    "True iff (curr - delta, curr] contains a multiple of n."
    return curr // n > (curr - delta) // n
```

Modulo works for *steps* (`delta==1`, lands on every integer; `_crossed(step,1,n)`
reduces exactly to `step % n == 0`), but **fails for tokens**, which jump by a
whole batch and almost never land on a multiple. `_crossed` fires on the first step
*past* each multiple — the only realizable behavior. It fires **at most once per
optimizer step** even when `delta > n` (a cadence smaller than a batch degenerates
to "every step").

```python
@dataclass(frozen=True)
class EveryNSteps:
    n: int; at_start: bool = False; at_end: bool = True
    def is_due(self, ctx):
        return ((self.at_start and ctx.global_step - ctx.steps_delta == 0)
                or (self.at_end and ctx.is_final)
                or _crossed(ctx.global_step, ctx.steps_delta, self.n))

@dataclass(frozen=True)
class EveryNTokens:           # same shape, over ctx.tokens_seen / ctx.tokens_delta
    ...
```

`at_start`/`at_end` OR with cadence and **inherently dedupe** (a final step that
also lands on cadence still runs once). `at_start` (default `False`) = "the counter
*before* this step was zero" (the first eval call, not `step==0`) — a pre-training
baseline that also fails fast on broken eval data. `at_end` (default `True`, for
**all** tasks including expensive ones) guarantees metrics on the final checkpoint;
opt a costly task out with `at_end: false`.

**Resume safety.** A checkpoint at `global_step=K` already fired any eval due at
`K`. On resume the first step makes `global_step=K+1`, `steps_delta=1`, and
`_crossed(K+1,1,n)` excludes `K` (the open end of the half-open interval) — no
re-fire, next multiple fires normally. Identical argument for tokens (trainer
restores `tokens_seen` verbatim, computes `tokens_delta` from the post-resume
batch). **Epoch cadence is deferred** precisely because a mid-epoch checkpoint
restarts the dataloader at the epoch start, so it can't give the same exactness;
when it ships, `EveryNEpochs` is just `_crossed(ctx.epoch, ctx.epoch_delta, n)`.

## 23. The `Evaluator`

```python
class Evaluator:
    def __init__(self, cfg): ...
    def run_due(self, ctx, model, accelerator) -> dict[str, float]: ...
    @property
    def has_tasks(self) -> bool: ...
    @property
    def needs_token_count(self) -> bool: ...   # any task on a token schedule?
```

Construction: import `oplm.eval.tasks` (triggers registration); call
`oplm.data.parse_eval_configs(cfg.data.eval, default_schedule)`; for each typed
entry, look up the class via the registry and instantiate `cls(entry, cfg)` (which
builds the task's `Schedule` from its `ScheduleSpec`); warn on any provably
unreachable schedule.

```python
def run_due(self, ctx, model, accelerator):
    due = [t for t in self.tasks if t.schedule.is_due(ctx)]
    if not due:
        return {}                                   # cheap: pure is_due only, no unwrap
    unwrapped = accelerator.unwrap_model(model)     # only when ≥1 due
    unwrapped.eval()
    metrics = {}
    try:
        for task in due:
            for key, value in task.evaluate(unwrapped, accelerator).items():
                metrics[f"eval/{task.name}/{key}"] = value
    finally:
        unwrapped.train()
    return metrics
```

`unwrap_model` moves **behind** the due-check so the common "nothing due" path
costs only a list comprehension over pure `is_due` calls. Tasks return **bare**
metric names (`"loss"`, `"precision_at_L"`); the `Evaluator` is the sole place that
applies the `eval/{name}/{metric}` namespace.

## 24. Trainer integration & distributed token accounting

The trainer builds one `EvalContext` per optimizer step and calls
`evaluator.run_due(ctx, self.model, self.accelerator)` with the **wrapped** model
(the evaluator unwraps behind the due-check). Returned metrics are logged at
`ctx.global_step` and forwarded to `on_eval_end`.

**Distributed token accounting** (required for token cadence). A per-rank estimate
(`local_tokens × num_processes`) diverges across ranks on ragged batches and
violates the rank-sync invariant. Instead, reduce truly, **unconditionally** (not
gated on `needs_token_count`; the cost is one tiny all-reduce):

```python
local_tokens = batch["attention_mask"].sum()
tokens_delta = int(accelerator.reduce(local_tokens, reduction="sum").item())
self.tokens_seen += tokens_delta                   # rank-identical by construction
```

Under gradient accumulation, sum micro-batch tokens locally across the window and
reduce once at the optimizer step. `is_final = global_step >= total_steps`
(computed identically on all ranks); the loop must terminate on `global_step` to
keep `is_final` rank-identical. Eval output is first-class training data (drives
the progress bar, checkpoint selection, later early stopping), so the evaluator is
an explicit collaborator, not a fire-and-forget callback.

## 25. Eval tasks

```python
class EvalTask(ABC):
    default_metrics: ClassVar[list[str]] = []
    def __init__(self, entry, cfg):
        self.name = entry.name; self.path = entry.path
        self.metrics = entry.metrics or self.default_metrics
        self.schedule = build_schedule(entry.schedule)   # ScheduleSpec -> Schedule
        self.cfg = cfg
    @abstractmethod
    def evaluate(self, model, accelerator) -> dict[str, float]: ...
```

Data loaders initialize **lazily** on first `evaluate`. The registry:
`@register_eval_task("sequence")` adds to `EVAL_TASK_REGISTRY` (raises on duplicate
type); `get_eval_task_class` raises with the known list on a miss;
`tasks/__init__.py` imports every task module so registration happens on import.
Adding a benchmark = implement a subclass, register it, point a `data.eval` entry's
`type` at it — no `Evaluator`/trainer change.

**Implemented:**

- **`sequence` (`SequenceEvalTask`)** — MLM metrics on held-out sequences via
  `build_sequence_eval_dataloader(path, cfg)` (deterministic frozen masking, no
  shuffle, comparable across checkpoints). Metrics `loss`, `accuracy`,
  `perplexity`; distributed via `accelerator.reduce` over summed loss/correct/
  masked counts.
- **`structure` (`StructureEvalTask`)** — contact-prediction precision@L via
  `load_structures(path, max_structures)` and `get_tokenizer()`. Metrics
  `precision_at_L` (default), `_2`, `_5` from the categorical Jacobian (§26).
  Distributed: structures sharded `structures[rank::world_size]`, each rank
  processes its shard one structure at a time (Jacobian/intermediates offloaded to
  CPU after each forward), per-structure results gathered with
  `dist.all_gather_object`. Cadence affects only *when* it runs; sharding is
  internal.

**Stubs** (registered, each raises `NotImplementedError` — no silent no-op; each
already maps to an `oplm.data` loader so the remaining work is metric work):

| Type | Loader | Planned metrics |
| --- | --- | --- |
| `proteingym` | `load_variant_assays` | spearman, ndcg (zero-shot variant effect) |
| `everest` | `load_variant_assays` | spearman, auroc (viral variant effect) |
| `tape` | downstream | ss3_accuracy, contact_precision, fluorescence/stability spearman, … |
| `proteinglue` | downstream | fold_accuracy, enzyme_accuracy, go_fmax |

**Per-task typed config.** Shared keys (`path`, `type`, `every`, `metrics`) are
consumed by the parser; every other key folds into `EvalDatasetEntry.extra: dict`.
A task converts `extra` to a frozen validated dataclass **once**, in its
constructor, via `from_extra()` — one place to cast/validate, no scattered
`extra.get(...)` in `evaluate`. Example (`StructureTaskConfig`): `contact_threshold
8.0`, `min_seq_sep 6`, `l_divisor 1`, `use_cbeta True`,
`categorical_jacobian_sample_size`, `_sample_seed 42`, `_mutation_batch_size 20`,
`max_structures`.

## 26. Metrics (`eval/metrics/`)

Eval-specific scoring math, kept in the harness (it is scoring, not loading).

- **MLM (`mlm.py`)**, over masked positions only: `loss` (mean cross-entropy,
  summed locally then reduced), `accuracy` (argmax fraction correct), `perplexity`
  (`exp(loss)`, capped at 1000 to stay finite early).
  `compute_mlm_metrics(model, dataloader, accelerator) -> {loss, accuracy,
  perplexity}`.
- **Contact / precision@L primitives (`contact.py`)**: **contact map** — binary
  `(L,L)`, residues `i,j` in contact iff Cβ–Cβ distance (virtual Cβ from N,CA,C
  when absent) `< contact_threshold` (8.0 Å) and `|i−j| ≥ min_seq_sep` (6); **APC**
  — `F_apc[i,j] = F[i,j] − (rowmean_i·colmean_j)/mean(F)`; **precision@L** — rank
  long-range pairs by score, take top `L/l_divisor`, report fraction that are true
  contacts.
- **Categorical-Jacobian coupling (`categorical_jacobian.py`)** — the single
  contact signal. `J[i,a,j,b] = ∂ logit(x_j=b) / ∂(x_i=a)` estimated by finite
  differences (mutate each position to each canonical AA, measure the logit shift);
  center over all four axes, symmetrize `(J+Jᵀ)/2`, APC-correct, reduce to an
  `(L,L)` coupling → precision@L. `categorical_jacobian_sample_size` optionally
  restricts the expensive `L×20`-forwards-per-structure computation to a
  deterministic subset.
- **Variant ranking** (for the stub tasks): score each variant by masked-marginal /
  pseudo-likelihood; compare to labels with **Spearman**, **NDCG** (top-variant
  emphasis), **AUROC** (binary viral labels).

## 27. `data.eval` config schema

```python
@dataclass(frozen=True)
class ScheduleSpec:          # lives in oplm.config — no dependency on oplm.eval
    unit: str                # "steps" | "tokens"  ("epochs" reserved)
    n: int                   # positive
    at_start: bool = False
    at_end: bool = True
```

`ScheduleSpec` is behavior-free and importable without `oplm.eval` (no layering
cycle): the data/config layer parses raw → typed, the eval layer interprets typed →
behavior.

**`every:` grammar** — a mapping with **exactly one** unit key (`steps` or
`tokens`, positive int) plus optional `at_start`/`at_end`; no unknown keys.

```yaml
train:
  eval_every: { steps: 10_000 }     # global default (same grammar)
data:
  eval:
    heldout:
      path: /data/eval_sequences.parquet
      type: sequence
      every: { tokens: 20_000_000 }     # frequent cheap MLM eval
    structures:
      path: /data/pdb
      type: structure
      every: { steps: 20_000 }          # infrequent expensive contact eval
      contact_threshold: 8.0            # task knobs → extra → StructureTaskConfig.from_extra
      categorical_jacobian_sample_size: 12
```

Resolution: an entry's `every` wins, else `train.eval_every`. The old steps-only
**int** form of `train.eval_every` no longer parses (*"cadence must be a mapping
like {steps: N} or {tokens: N}"*); a per-entry `eval_every` key is rejected
outright (*"…uses the removed `eval_every` key. Use `every: {steps: N}`…"*).

---

# Part IV — Trainer (`src/oplm/training/`)

> The model, data, and eval harness were rewritten on `hf-compat`; the trainer is
> the last piece converging onto the HF `OplmConfig`. This part describes the
> target design.

## 28. Config assembly

**Root run config** `oplm.config.OplmConfig` composes `{model, train, data}`.
`train` (`TrainConfig`) and `data` (`DataConfig`) are OmegaConf-structured
dataclasses; **`model` is an untyped mapping** (`Any`) so OmegaConf can carry
arbitrary HF field keys without a parallel schema:

```python
@dataclass
class OplmConfig:                                   # the RUN config (root)
    model: Any = field(default_factory=dict)        # resolved into oplm.model.OplmConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
```

**`load_config`** merge order: defaults → `--preset` → `--config` YAML → CLI
dotlist overrides. The model subtree resolves to a plain dict and is instantiated
as the HF config (HF owns derivation + validation):

```python
from oplm.model import OplmConfig as OplmModelConfig
model_dict = OmegaConf.to_container(base.model, resolve=True) or {}
model_cfg  = OplmModelConfig(**model_dict)          # HF validation + derivation
cfg = OplmConfig(model=model_cfg, train=…, data=…)
```

Derived fields (`head_dim`, `intermediate_size`, `rope_dim`, `nope_dim`) are simply
**omitted** unless the user sets them and resolve to `None` → derived. `train.eval_every`
is the canonical global-cadence field (a cadence mapping; the int form is caught by
the schedule parser). **Unknown `model.*` keys do not raise** — they flow into
`PretrainedConfig`'s `**kwargs`, are retained as attributes, and ignored by the
model (so old/mistyped model keys are silently tolerated; the doc + `model/base.yaml`
document the current schema).

**Name collision.** The root `oplm.config.OplmConfig` and the model
`oplm.model.OplmConfig` share a class name; import sites use the alias
`from oplm.model import OplmConfig as OplmModelConfig`. (A rename to `RunConfig` is
deferred — it would ripple through data/eval/tests.)

Field names migrated from the old dataclass `ModelConfig` to the HF config:
`hidden_dim→hidden_size`, `num_layers→num_hidden_layers`,
`num_heads→num_attention_heads`, `ffn_dim→intermediate_size`,
`max_seq_len→max_position_embeddings`, `tie_embeddings→tie_word_embeddings`,
`conv_positions` (`"ACD"`) → `canon_positions` (`["A","C","D"]`), the
`pre_norm`/`post_norm`/`sandwich_norm`/`post_sdpa_norm` booleans → the single
`norm_strategy` enum, and `partial_rope` → `rope_dim`/`nope_dim`. **Removed with no
replacement:** `num_kv_heads`/`shared_kv`, `value_residual`(+lambda),
`num_value_embeds`/`value_embed_gate_dim`, `output_gate`/`query_dependent_gate`,
`attn_residual`(+block_size), the `conv_kernel_*` schedule family, and `dtype`.

## 29. Model construction, optimizer, schedules, FLOPs

**Construction:**

```python
from oplm.model import OplmForMaskedLM
model = OplmForMaskedLM(cfg.model)                  # cfg.model IS the HF OplmConfig
if cfg.model.gradient_checkpointing:
    model.gradient_checkpointing_enable()           # propagates to every OplmBlock
```

`forward(input_ids, attention_mask, labels=…)` returns `MaskedLMOutput` with the
loss computed inside the model (cross-entropy, `ignore_index=-100`); the loop reads
`outputs["loss"]` (item access supported).

**Optimizer & parameter grouping** (`optim.py`). `partition_optimizer_params`
splits trainable params into three groups: **no-decay** (`ndim <= 1 or "embed" in
name` — biases, norms, embeddings), **Muon** (2-D weights when `optimizer="muon"`,
**excluding the head via the `lm_head.` prefix**), and **AdamW-decay** (other 2-D
weights). `build_optimizers` returns `[AdamW]` (two groups: decay with
`weight_decay`, no-decay with 0.0) or `[Muon, auxiliary AdamW]`. The head-exclusion
prefix is `lm_head.` (the MLM head is `OplmForMaskedLM.lm_head`); the tied decoder
weight already lands in no-decay via the `"embed"` rule. `torch.optim.Muon` is
available (torch ≥ 2.11).

**LR schedules** (`get_schedule_fn`): a three-phase multiplier — warmup (linear
0→1) → optional stable plateau (WSD only) → decay (linear or cosine to
`min_lr/lr`) — wrapped in a `LambdaLR` per optimizer, stepped manually each
optimizer step. Schedulers: `warmup_linear`, `warmup_cosine`, `wsd_linear`,
`wsd_cosine`. The `Accelerator` is built with
`step_scheduler_with_optimizer=False` (the trainer steps schedulers itself).

**FLOPs** (`estimate_flops_per_token`, takes the HF config). No GQA (Q/K/V each
`hidden_size → hidden_size`); FFN always gated (3 projections). Per layer:
attention projections `2·H·(4H)` + FFN `3·2·H·I`; head `2·H·H + 2·H·V`; training
≈ `3×` forward. Documented caveat: attention-score FLOPs, norms, and embedding
lookups are omitted.

## 30. Training loop

Entry points: `oplm.train.main(cfg=None)` (bootstraps the env — writable
`TRITON_CACHE_DIR`, DeepSpeed off unless `OPLM_ENABLE_DEEPSPEED`; loads config from
`sys.argv[1:]` then `Trainer(cfg).train()`), launched directly (`python -m
oplm.train --config …`) or distributed (`accelerate launch -m oplm.train --config …
model.num_hidden_layers=32`); and `oplm.cli train` (a Typer wrapper building the
same argv from `--config`/`--preset` plus bare `key=value` positionals).

`Trainer.__init__` order: `set_seed`; build `Accelerator` (`mixed_precision`,
`gradient_accumulation_steps`, `log_with="wandb"`, `project_dir`,
`DataLoaderConfiguration(dispatch_batches=False)`,
`step_scheduler_with_optimizer=False`); init wandb trackers early; build
`Evaluator` when `cfg.data.eval is not None`; build the model (+ gradient
checkpointing); `build_optimizers` + `build_train_dataloader`;
`_compute_total_steps` + `build_schedulers`; `accelerator.prepare(model,
*optimizers, dataloader, *schedulers)`; init state (`global_step`, `epoch`,
`tokens_seen`, `_samples_seen`, `_epoch_at_last_opt_step`, `_step_local_tokens`,
`flops_per_token`); resume if `cfg.train.resume_from`.

The loop is one `while self.global_step < self.total_steps`:

```python
with self.accelerator.accumulate(self.model):
    outputs = self.model(input_ids=…, attention_mask=…, labels=…)
    loss = outputs["loss"]
    self.accelerator.backward(loss)
    if cfg.max_grad_norm > 0 and self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
    for optimizer in self.optimizers:           # step + zero in one loop
        optimizer.step(); optimizer.zero_grad()

# cadence work runs only on the optimizer-step (sync_gradients) boundary:
if not self.accelerator.sync_gradients:
    continue
for scheduler in self.schedulers: scheduler.step()
self.global_step += 1
# rank-reduce this step's tokens → tokens_delta / tokens_seen rank-identical (§24)
if self.global_step % cfg.log_every == 0: self._log_step(current_loss)
eval_metrics = self._run_eval(tokens_delta)     # build EvalContext, Evaluator.run_due
if eval_metrics: self._log_metrics(...); self._emit_eval_end(...)
if self.global_step % cfg.save_every == 0: self._save_checkpoint()
```

Invariants/cleanups: cadence work (scheduler step, `global_step++`, logging, eval,
checkpoint) runs once per **optimizer** step, not per micro-batch; epoch rollover on
`StopIteration` increments `epoch`, calls `set_epoch(epoch)` (deterministic
reshuffle), re-creates the iterator; **accumulation-aware loss** for logging (a
detached running sum across micro-steps / `gradient_accumulation_steps`, not just
the last micro-batch); a final checkpoint, progress-bar stop, `on_train_end`, and
`accelerator.end_training()` in a `finally`.

**Cadence.** Duration: `max_steps` (default) or `max_epochs` (steps-per-epoch
derived from resolved dataset length and the global effective batch `batch_size ·
grad_accum · num_processes`; documented fallback when an iterable dataset reports no
length). Logging/checkpointing: step-modulo (`log_every`, `save_every`). Eval: step
or token cadence per dataset via harness schedules, defaulting to
`train.eval_every`. **Steps are the native clock**; epochs/tokens are derived
overlays; **epoch-based eval cadence stays deferred** (`epoch`/`epoch_delta` are
carried for forward compatibility).

## 31. Checkpointing, logging, callbacks

**`save_checkpoint`** writes `checkpoint-{step}/`:

- **Accelerate state** at top level (`accelerator.save_state(...)`) — model,
  optimizer(s), scheduler(s), RNG — the resumable state.
- **`trainer_state.json`** — `global_step`, `epoch`, `samples_seen`, `tokens_seen`.
- **HF export** under `checkpoint-{step}/hf/` via the unwrapped model
  (`unwrap_model(model).save_pretrained(.../hf)` → `config.json` +
  `model.safetensors`, honoring tied weights) plus the tokenizer
  (`get_tokenizer().save_pretrained(.../hf)`), so
  `OplmForMaskedLM.from_pretrained(checkpoint-{step}/hf)` round-trips with its
  tokenizer.

The run config is persisted for provenance via `cfg.model.to_dict()` + OmegaConf
YAML for `train`/`data` (this replaces the broken `OmegaConf.structured(cfg)` that
can't structure a `PretrainedConfig`). `_rotate_checkpoints` keeps at most
`save_total_limit`. Resume reads Accelerate state + `trainer_state.json` from the
top level (the `hf/` export is for downstream loading, not resume), re-seeds the
dataset epoch, and resets per-opt-step delta markers.

**Logging / wandb:** `accelerator.log(metrics, step=global_step)`; training metrics
`train/{loss,epoch,samples,tokens,flops,lr}`; eval metrics `eval/<task>/<metric>`.
`_config_to_flat_dict` flattens `cfg.model.to_dict()` under a `model/` prefix and
`asdict` of `train`/`data` under theirs (the old `dataclasses.asdict(cfg)` failed on
the HF config). **Callbacks** (`callbacks.py`): `TrainerCallback` with
`on_train_start`, `on_log`, `on_eval_end`, `on_checkpoint_saved`, `on_train_end`,
all invoked on the main process only; the rich progress bar is main-process only.

## 32. CLI & inference

- **`cli train`** — prints a one-line summary on HF fields (`num_hidden_layers /
  hidden_size`).
- **`cli info`** — architecture/feature tables on HF fields (`hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `head_dim`, `intermediate_size`,
  `norm_strategy`, `qk_norm`, `canon_*`, `residual_scaling`,
  `gradient_checkpointing`, `tie_word_embeddings`); removed-feature rows dropped.
- **`cli encode`** — uses the ESM-C convenience API. Note `model.encode(seqs)`
  returns padded **`input_ids`** (an ESM-C naming quirk), *not* embeddings; for the
  embedding matrix use `model.logits(seqs,
  LogitsConfig(return_embeddings=True)).embeddings` → `(B, L, hidden_size)`
  (post-final-norm hidden states). Requires a tokenizer attached (via
  `from_pretrained` of an `hf/` export, or `model.tokenizer = get_tokenizer()`).
- **`inference.py`** — `load_model_for_inference` builds `OplmForMaskedLM(cfg.model)`
  from the HF config; prefers `OplmForMaskedLM.from_pretrained(<checkpoint>/hf)`
  when an HF export is present, falling back to the state-dict path for bare weights.

## 33. Bugs the trainer rewrite fixes

1. Model built from the wrong config type (dataclass `ModelConfig` vs HF
   `OplmConfig`).
2. `model.encoder` no longer exists → `gradient_checkpointing_enable()`.
3. Muon head exclusion checked `mlm_head.` instead of the actual `lm_head.` (the
   head's dense weight leaked into Muon).
4. FLOPs read removed fields (`hidden_dim`/`num_kv_heads`/`ffn_dim`).
5. Checkpoint config serialization broke on a `PretrainedConfig`
   (`OmegaConf.structured(cfg)`).
6. wandb flat-dict broke on a `PretrainedConfig` (`asdict(cfg)`).
7. Stale `data.max_length` rejection message pointed at the renamed
   `model.max_seq_len` (→ `model.max_position_embeddings`).
8. `cli encode` doubly broken (`ProteinTokenizer` deleted + `model.encoder` gone).
9. Last-micro-batch loss logged as the step loss → accumulation-aware loss.
10. Double optimizer iteration → single step+zero loop.

---

# Part V — Testing & status

## 34. Testing strategy

Prefer **real data** over synthetic; provide small fixtures (drop real samples into
`tests/data/fixtures/`); mark heavy parsing/model tests `@pytest.mark.slow`; make
large fixtures session-scoped. Tests mirror the source tree.

- **Model** — forward shape/dtype correctness across toggles; SDPA-vs-manual
  parity; QK-norm/RoPE/Canon math; config validation raises on bad combinations;
  `save_pretrained`/`from_pretrained` round-trip.
- **Data** — **tokenizer parity** (data-layer IDs equal `OplmTokenizerFast`'s;
  `mask_token_id==32`, `canonical_aa_ids==range(4,24)`, `special_ids=={0,1,2,3,32}`
  — the test that would have caught the legacy off-by-one); sequence determinism
  (same `(seed,epoch)` ⇒ identical order; different epochs ⇒ different order;
  `deterministic=True` ⇒ identical mask per batch index); **dynamic masking** (same
  example, two epochs, different mask sets); masking correctness (exactly `k`
  positions, no duplicates; 80/10/10 within tolerance; specials/pad never selected;
  `labels==-100` exactly at unmasked; random replacements only from
  `canonical_aa_ids`); **weighted masking** (count stays `k`; inclusion ∝ `w_i`;
  `w_i=0` never masked; scale-invariance; `<k` positive weights ⇒ all masked;
  alignment through truncation/specials/pad; `None`/absent-column/length-mismatch
  fallbacks); padding/truncation; striping coverage (union over `(rank,worker)` =
  dataset, no dupes/gaps); interleaving ratios; structure/variant/downstream
  parsing on one real PDB / tiny CSV.
- **Eval** — pure crossing/`Schedule` semantics (no GPU) over the edge cases;
  resume (no re-fire at `K`, correct next-fire, steps and tokens); rank-sync
  (`is_due` pure ⇒ all ranks agree; multi-rank smoke test that token cadence
  doesn't hang on ragged batches); token accounting (reduced `tokens_seen`
  rank-identical and equal to the true global sum); registry (duplicate raises;
  unknown type raises with the list); per-task evaluate on small real data.
- **Training** (`tests/training/`) — `load_config` builds a valid HF `OplmConfig`
  (preset + CLI overrides apply, derived fields resolve, unknown `model.*` keys
  absorbed); `partition_optimizer_params` puts `lm_head.*` on AdamW not Muon and
  embeddings/norms/biases in no-decay (partition covers all params); FLOPs positive,
  finite, scales with depth/width; checkpoint save → `from_pretrained(checkpoint/hf)`
  round-trips, resume restores `global_step/epoch/tokens_seen`, rotation respects
  `save_total_limit`.
- **End-to-end** — a pilot-scale model trains a few steps (CPU,
  `wandb_enabled=False`, `mixed_precision="no"`) with both a token-cadence and a
  step-cadence eval configured, fires one eval, writes a checkpoint, resumes from it
  — asserting no shape/scheduling errors, finite loss, and a correct post-resume
  `global_step`.
- **Gates:** `ruff check src/`, `ruff format src/`, `ty`/`mypy src/`, and the full
  `pytest` suite (`-m "not slow"` for fast iteration).

## 35. Open / deferred

- **Epoch-based eval cadence** — deferred (mid-epoch resume can't match the
  steps/tokens exactness guarantee); `EvalContext` already carries
  `epoch`/`epoch_delta`, and `EveryNEpochs` slots in as
  `_crossed(ctx.epoch, ctx.epoch_delta, n)` with no interface change.
- **Root/model `OplmConfig` name collision** — retained for now (alias at import
  sites); revisit a rename to `RunConfig` later.
- **Long-context RoPE** — no YaRN/NTK scaling yet; extrapolation beyond
  `max_position_embeddings` is best-effort.
- **`geglu`** FFN variant — reserved enum value, not yet implemented.
- **Variant / downstream eval tasks** (`proteingym`, `everest`, `tape`,
  `proteinglue`) — registered stubs that raise `NotImplementedError`; their data
  loaders exist, so the remaining work is metric implementation.

---

## Appendix — references

1. Su et al., *RoFormer: Rotary Position Embedding*, arXiv:2104.09864.
2. Touvron et al., *LLaMA*, arXiv:2302.13971 (SwiGLU + RMSNorm precedent).
3. Zhang & Sennrich, *RMSNorm*, arXiv:1910.07467.
4. Partial RoPE / NoPE split — arXiv:2502.14837v1.
5. HybridNorm — arXiv:2503.04598.
6. Canon depthwise conv layers — arXiv:2512.17351.
7. ESM-C — EvolutionaryScale, `evolutionaryscale/esm` (tokenizer + public API).
8. FlashAttention-4; PyTorch 2.11 release notes,
   `torch.nn.functional.scaled_dot_product_attention` docs.
9. HuggingFace Transformers: `PreTrainedModel`, `PretrainedConfig`, `AutoModel*`.

These map to the architecture toggles: full-RoPE/partial-RoPE (1, 4), SwiGLU and
RMSNorm (2, 3), hybrid norm (5), Canon convs (6), the tokenizer + ESM-C-style API
(7), SDPA attention (8), and the HF integration surface (9).
