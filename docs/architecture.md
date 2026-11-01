# OPLM model architecture

OPLM is a HuggingFace-native, encoder-only (BERT/RoBERTa-style) protein language model trained
with masked-language modeling. This document opens with the architecture that the shipped defaults
produce, then states the semantics contract for the parts that are not obvious from tensor shapes:
Canon, norm strategies, RoPE, residual scaling, looping and the μP hooks. Every field is listed
with its type and default in [config.md](config.md). This document explains what the fields do.

## Default architecture

`oplm train` builds the model from `src/oplm/configs/model/base.yaml`. That file sets every
`OplmConfig` option to its run default. The model it describes:

| Part | Default | Field in `base.yaml` |
| --- | --- | --- |
| Size | 32 layers, hidden 1024, 16 heads of 64 (the `400M` preset), 1024 positions | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `max_position_embeddings` |
| Vocabulary | 33 tokens, no positional embedding table | `vocab_size` |
| Norm operator | LayerNorm, eps `1e-6`, computed in fp32 | `norm_type`, `norm_eps` |
| Norm placement | sandwich: a pre-norm and a post-norm around each sublayer | `norm_strategy` |
| QK-norm | on, per-head norm over `head_dim` (`channel` mode) | `qk_norm`, `qk_norm_mode` |
| Residual scaling | each sublayer write times `1/sqrt(L)` | `residual_scaling` |
| Residual gate | none | `residual_gate` |
| Attention output gate | sigmoid, post-SDPA | `attn_output_gate` |
| Value residual | learnable, one scalar per layer after the first | `value_residual` |
| Attention | standard multi-head, no GQA, full RoPE (`theta` 10000), bidirectional | `rope_theta`, `rope_dim`, `nope_dim` |
| Feed-forward | SwiGLU, no bias, width `round_up(8/3 * hidden_size, 256)` | `ffn_activation`, `ffn_bias`, `intermediate_size` |
| Canon | on at positions A, B, C, D, kernel size 7, residual | `canon_enabled`, `canon_positions`, `canon_kernel_sizes`, `canon_residual` |
| Embeddings and head | untied; two-layer MLM head with GELU | `tie_word_embeddings`, `mlm_head_activation` |
| Init | truncated normal, std 0.02; output projections divided by `sqrt(2L)` | `initializer_range`, `init_scale_output_projections` |
| μP | on, base width 768, output multiplier 1.0 | `mup_enable`, `mup_base_width`, `mup_output_mult` |
| Dropout | none | `attention_dropout`, `hidden_dropout` |
| Packed-batch attention | dense block-diagonal SDPA mask | `packed_attention` |

One default block, with `a = 1/sqrt(L)`, `A` the attention sublayer with its output gate and value
residual, and Canon sites A to D inserted as described in *Canon semantics*:

```text
h = x + a * Norm2(A(Norm1(x)))
y = h + a * Norm4(FFN(Norm3(h)))
```

The last block's output passes one final norm before the head. The optimizer recipe that goes with
this architecture (Muon plus μP) is in `src/oplm/configs/train/base.yaml` and is described in
[mup-lr.md](mup-lr.md). The ESM-C comparison model is a separate recipe in
`src/oplm/configs/train/esm-c.yaml`. It pins pre-norm, full-width QK-norm, `esmc` residual scaling,
and no gates, value residual or Canon, so default changes never leak into it.

### Run defaults and library fallbacks

The YAML layer carries the run defaults above. Bare `OplmConfig()` construction and
`from_pretrained` use more conservative fallbacks so existing checkpoints load unchanged: LayerNorm
pre-norm, no Canon (kernel size 4 if enabled), no attention output gate, no value residual,
`mup_enable=false`, and 768 hidden, 12 layers, 12 heads. config.md, *μP (maximal update
parametrization)*, lists the fallback next to each default that differs.

### Available, not recommended

These toggles exist as literature-backed ablation switches. They are off by default and are not
recommended for new runs unless you are reproducing an ablation or the ESM-C recipe: RMSNorm,
`hybrid` and `post_sdpa` norm placement, `l2` QK-norm, partial RoPE, the residual gate, `geglu` and
`relu2` FFNs, mask dropout, tied embeddings and the post-embedding norm. Each is described in its
section below.

## Transformer block and norm placement

"Norm" means the configured `norm_type`, used at every norm site (mixing operators is unsupported).
`norm_strategy` is one global enum. Residual scaling composes orthogonally with it.

| `norm_strategy` | Attention sublayer | FFN sublayer |
| --- | --- | --- |
| `pre` | `h = x + a*Attn(Norm(x))` | `y = h + a*FFN(Norm(h))` |
| `sandwich` (default) | `h = x + a*Norm2(Attn(Norm1(x)))` | `y = h + a*Norm4(FFN(Norm3(h)))` |
| `post_sdpa` | `h = x + a*Norm(Attn(Norm(x)))` | `y = h + a*FFN(Norm(h))` |
| `hybrid` | `h = x + a*Attn_QKVNorm(x)`, no outer pre-norm; Q, K and V are normed inside attention | `y = Norm(h) + a*FFN(Norm(h))`, one `Norm(h)` reused as FFN input and as the FFN-side residual stream |

`hybrid` follows the "QKV-Post" formulation of arXiv:2503.04598 literally: the block-level attention
pre-norm is suppressed and V is normed in addition to Q and K. For `pre`, `sandwich` and
`post_sdpa`, QK-norm (Q and K only) is on by default and independent of `norm_strategy`. Turn it off
with `qk_norm=false` for ablation. The final post-stack norm applies in every strategy. Unusual
combinations such as `qk_norm=false` with `hybrid` warn rather than raise.

### Residual scaling

Each sublayer contributes `O(1/sqrt(L))` to the residual stream, so the total from `2L` sublayers
stays `O(sqrt(L))` rather than `O(L)`. This is the main depth-stability tool. `residual_scaling`
selects the factor `a`:

- `sqrt_num_layers` (default): `a = 1/sqrt(L)`.
- `esmc`: `a = 1/sqrt(L/36)`, ESM-C's convention.
- `none`: `a = 1`.

`init_scale_output_projections` (default on) additionally divides the init std of the attention
output projection and FFN down projection by `sqrt(2L)`. That is partly redundant with residual
scaling and can be disabled as an ablation.

### Residual gate

`residual_gate` is an optional learnable multiplicative refinement on top of the fixed `a`, with one
gate per sublayer write: `h = x + a * attn_gate * Attn(...)`. `scalar` adds one parameter per write,
`channel` adds a `(hidden_size,)` vector. Both start at `residual_gate_init` (default 1.0, an
identity refinement). They are 1D, so they land in the no-decay AdamW group, and they refine rather
than replace `a`. The default is `none`: under sandwich norm a `channel` gate duplicates the
post-norm gain, and dropping it in the October 2026 μP refresh cost no loss. Available, not
recommended.

## Attention

Standard multi-head attention with `num_attention_heads` equal to the key/value head count (no GQA).
Shapes: `x (B,T,D)` goes to Q/K/V projections, is reshaped to heads `(B,H,T,d)` with `d = D/H`,
passes QK-norm, optional Canon-B, the value residual and RoPE, and the merged output goes through a
bias-free `(D,D)` output projection. The encoder is bidirectional: there is no causal mask, only a
key-padding mask. Masking only keys keeps every query row non-empty, so the softmax cannot produce
NaN.

**QK-norm.** `qk_norm_mode` selects one of three modes (active only when `qk_norm=true`):

- `channel` (default): Q and K each pass through a separate norm over the `head_dim` channels, gain
  `(head_dim,)` shared across heads, computed in fp32. The kernel score scale is `1/sqrt(head_dim)`.
- `l2`: Q and K are L2-normalized over `head_dim` in fp32, then Q is multiplied by a learned per-head
  scale of shape `(num_attention_heads,)`, initialized to `qk_norm_l2_scale_init` or
  `sqrt(head_dim)`. The kernel scale is 1.0 (the temperature lives in the learned scale), and the 1D
  scale is not weight-decayed. Available, not recommended.
- `full` (ESM-C): one bias-free norm over the full `hidden_size`, all heads jointly, before the head
  split. Kernel scale `1/sqrt(head_dim)`. The ESM-C recipe uses it.

With `qk_norm=false` Q and K are unnormalized and the scale stays `1/sqrt(head_dim)`. V is never
normed except under `hybrid`, which adds an independent `(head_dim,)` V-norm.

**Attention output gate** (`attn_output_gate`, default `sigmoid`). The post-SDPA gate of arXiv:2505.06708
(G1): a bias-free `(D,D)` `gate_proj` computes an elementwise gate from the attention input, and the
merged attention output is multiplied by `act(gate_proj(x))` before the output projection. `act` is
`sigmoid` or `silu`, and `none` adds no parameters. The gate uses the standard init (so it starts near
`act(0)`), is not a residual writer (the `1/sqrt(2L)` shrink stays on the output projection), and as
a 2-D hidden matrix it joins the Muon group.

**Value residual** (`value_residual`, default `learnable`). The ResFormer value residual of
arXiv:2410.17897: layer 0 exposes its post-V-norm values `v1`, and every later layer blends
`v' = lambda * v + (1 - lambda) * v1` right after the V projection, before RoPE and the kernel.
`learnable` gives layers 1 to L-1 one raw scalar lambda each, initialized to
`value_residual_lambda_init` (0.5) and not weight-decayed or given to Muon. `fixed` uses the same
constant as a buffer with no new parameters. `none` adds nothing and keeps the two-element attention
return. When enabled, attention and block return a third element carrying `v1`, and the stack threads
it through the layer loop and both gradient-checkpointing paths. `v1` never leaves the stack.

**Kernel paths.** Two paths share all parameters and pre-attention transforms. With
`output_attentions=false` (training and normal inference) attention runs through
`F.scaled_dot_product_attention` with a `(B,1,1,T)` boolean key mask, which dispatches to a fused
kernel on CUDA and the math backend on CPU. With `output_attentions=true` a manual path
(`softmax(q @ k^T * scale)` in fp32) returns the full `(B,H,T,T)` weights for contact precision and
rollout. It is slower and `O(B*H*T^2)` in memory, so it is for inspection, not training.

**Packed batches.** `packed_attention` selects the kernel for packed rows (`data.pack_sequences`; see
[data.md](data.md)). `sdpa` passes a dense block-diagonal mask whose cost grows with row width. `flex`
uses FlexAttention with a block mask that skips the 128-token blocks between sequences, needs
`train.compile` for speed on GPU, has no attention dropout, and has no backward on CPU. Eval
(unpacked) batches always use SDPA.

## Positional encoding

Rotary embedding. The `head_dim` channels are paired and rotated by `position * theta_i` with
`theta_i = rope_theta^(-2i/head_dim)`. It is applied to Q and K after QK-norm, in fp32, and never to
V. The `cos`/`sin` tables are buffers up to `max_position_embeddings` and extend on the fly for longer
inputs. There is no YaRN or NTK extrapolation, so lengths beyond `max_position_embeddings` are
best-effort.

Partial RoPE and NoPE (arXiv:2502.14837): when `nope_dim > 0`, `head_dim = rope_dim + nope_dim`, RoPE
rotates only the first `rope_dim` channels, and the rest are position-invariant. The default is full
RoPE (`nope_dim=0`). `rope_dim + nope_dim == head_dim` and even `rope_dim` are enforced at config
time. Partial RoPE is available, not recommended.

## Feed-forward

`ffn_activation=swiglu` (default): `y = W_d(silu(W_g(x)) * W_u(x))`, three bias-free linears of width
`F = round_up(8/3 * D, 256)`. Set `intermediate_size` to override. `geglu` swaps the gate
nonlinearity for GELU with the same three projections. `relu2` is the non-gated squared-ReLU variant
(arXiv:2109.08668), `y = W_d(relu(W_u(x))^2)`, with two projections and a derived width
`round_up(4 * D, 256)` so the total (about `8 * D^2` parameters) matches the gated variants. `geglu`
and `relu2` are available, not recommended. `ffn_activation` is the single extension point for new
activations.

## Normalization

`norm_type` selects one operator for every norm site. Both compute in fp32 regardless of autocast.

- LayerNorm (default): `gamma * (x - mean)/sqrt(var + eps) + beta`, matching ESM-C.
- RMSNorm: `gamma * x/sqrt(mean(x^2) + eps)`, gain only. It has half the parameters per site and is
  slightly cheaper. Available, not recommended.

The norm sites are: the optional post-embedding norm (`post_embed_norm`, off), the attention
pre-norm (omitted under `hybrid`), QK-norm for Q and K, the V-norm (`hybrid` only), the FFN pre-norm,
the sandwich or post-SDPA post-norms, the final stack norm, the MLM-head intermediate norm, and the
optional pre-head norm (`pre_head_norm`, off).

## Embeddings and heads

**Token embedding.** `nn.Embedding(vocab_size, D)` with no positional table (RoPE handles position).
The `<pad>` row is not zeroed (HF convention).

**Mask dropout** (`mask_dropout`, off; available, not recommended). On the `input_ids` path only,
every `<mask>` embedding row is zeroed and each sequence is rescaled by
`(1 - mask_dropout_reference_ratio) / (1 - observed_mask_ratio)`, where the observed ratio is the
fraction of real tokens that are `<mask>`. The reference ratio (0.12) is the expected fraction of
real tokens that are `<mask>` under the masking policy (`mask_prob * mask_token_prob`), not a fraction
of mask tokens to drop. The rescale is deterministic in train and eval and applies before the
post-embedding norm. The `inputs_embeds` path bypasses it, because mask positions cannot be recovered
from embeddings.

**MLM head.** `Linear(D,D)` with bias, `mlm_head_activation` (GELU), a norm following `norm_type`, then
a `Linear(D, vocab_size)` decoder with bias. `tie_word_embeddings` is off: the 33-token vocabulary
makes tying save only about 25K parameters at D=768, while an independent projection historically
helps MLM slightly. When tied, only the decoder weight ties (its bias stays independent). Loss is
`cross_entropy(..., ignore_index=-100)` computed inside `OplmForMaskedLM.forward`.

**Classification heads.** Sequence classification pools (`mean` over real tokens, or `cls` for
position 0), applies `classifier_dropout`, then `Linear(D, num_labels)`. Token classification applies
dropout and the same linear per token. Pooling helpers for embedding extraction live in
`src/oplm/model/embedding.py`.

| Head | Input | Output | Class |
| --- | --- | --- | --- |
| MLM | last hidden `(B,T,D)` | `(B,T,V)` | `OplmForMaskedLM` |
| Sequence classifier | last hidden and mask | `(B,num_labels)` | `OplmForSequenceClassification` |
| Token classifier | last hidden `(B,T,D)` | `(B,T,num_labels)` | `OplmForTokenClassification` |

## Canon semantics

Canon is controlled by `canon_enabled`, `canon_positions`, `canon_kernel_sizes`, `canon_activation`
and `canon_residual`. There is no legacy Canon mode: the paper-exact encoder implementation
(arXiv:2512.17351) replaced the earlier Canon-inspired tensor paths in place, rather than retaining
old behavior behind a compatibility switch.

The encoder keeps the paper's layer insertion semantics and adapts only the local convolution window
for an encoder MLM. Canon is supported under `norm_strategy` in `{pre, sandwich, post_sdpa}`. All
three have an outer attention pre-norm and FFN pre-norm, and their extra post-norms act on sublayer
outputs, downstream of every Canon insertion point, so positions A and C land identically to plain
pre-norm. `norm_strategy=hybrid` is rejected at config validation: it suppresses the outer attention
pre-norm (QK/V-norm moves inside attention, which sees raw `x`), so Canon-A would have no normalized
stream to act on.

Paper-exact encoder Canon uses `canon_residual=true`, `canon_activation=none` for paper-comparable
experiments, a centered bidirectional same-length depthwise convolution, and no causal masking.

### Convolution window

The operator is a depthwise `nn.Conv1d(D, D, kernel_size=k, groups=D, bias=False)` over sequence
positions; the module owns that weight but computes the cross-correlation itself in the `(B,T,D)`
layout. Pad-token rows are zeroed before each convolution, so pad content cannot leak into real
tokens. That guard is the model's responsibility, not the caller's.

On CUDA the convolution runs as a Triton kernel registered as the custom op `oplm::canon_conv`
(`src/oplm/model/canon_kernel.py`), with its own backward, so `torch.compile` treats each Canon call
as one opaque op. The op does the pad zeroing, the per-sequence masks of packed rows, fp32
accumulation and, when `canon_activation` is `none`, the residual add; the weight gradient is
reduced deterministically. Everywhere else (CPU, or no Triton) the same function runs as
`canon_conv_reference`, `k` shifted multiply-adds, which is also the kernel's test oracle. There is
no config switch between the two. A training run that uses the kernel first checks it against the
reference on its first real batch (output and both gradients, in fp32) and stops on a mismatch.

The library default kernel size is 4 (the paper's choice) and the run default in `base.yaml` is 7.
For even kernels, exact token-centered symmetry is impossible in a same-length convolution. OPLM pads
`k/2` on the left and `k/2 - 1` on the right:

```text
k = 4: pad left by 2, right by 1
output i sees input positions [i-2, i-1, i, i+1]
```

Use odd kernels for experiments that need exact symmetry. `canon_kernel_sizes` accepts a scalar
(broadcast), a per-layer list, or a schedule dict resolved at config load, and every kernel is at least
2. Cost per convolution is `k * D` parameters and about `2 * k * D * T` FLOPs.

### Insertion sites

For `norm_strategy=pre`, one block with paper-exact encoder Canon is (sandwich and post_sdpa are
identical at the insertion points and only add post-norms on the sublayer outputs):

```text
attn_base = AttnNorm(x)
attn_in   = attn_base + CanonA(attn_base)
attn_out  = Attention(attn_in)
h         = x + alpha * attn_out

mlp_base  = FfnNorm(h)
mlp_in    = mlp_base + CanonC(mlp_base)
ffn_out   = FFN_with_CanonD(mlp_in)
y         = h + alpha * ffn_out
```

- **A** acts on the attention pre-norm stream passed to the Q/K/V projections.
- **B** acts inside attention after the Q/K/V projections and after QK/V-norm, before the value
  residual, RoPE and the kernel. It is one depthwise convolution per stream (`conv_b_q`, `conv_b_k`,
  `conv_b_v`) on the attention module. It matches Primer's multi-DConv-head attention
  (arXiv:2109.08668). Canon-B stays after QK/V-norm to preserve the stability assumptions of the
  attention stack:
  `QKV projection -> QK/V norm -> residual Canon-B -> value residual -> RoPE`.
- **C** acts on the FFN pre-norm stream passed into the FFN.
- **D** acts inside the FFN at intermediate width, before the activation.

`canon_positions` is a global list (order irrelevant, each site independent), for example
`[A, C, D]`.

### Residual form

Paper-exact encoder Canon is residual at every enabled position, `z = z + Canon(z, attention_mask)`,
for A, B, C and D. Replacement-style convolution belongs only to future, explicitly named non-paper
research knobs.

### Canon-D tensor space

Canon-D acts at `intermediate_size` channels, before the activation, on the branch that feeds the
nonlinearity:

```text
SwiGLU: hidden = silu(gate + CanonD(gate)) * up
GEGLU:  hidden = gelu(gate + CanonD(gate)) * up
relu2:  hidden = relu(up + CanonD(up)) ** 2
```

It must not be applied to the hidden-size FFN input before the FFN projections.

## Shared-layer looping

`num_hidden_layers` is physical depth L. `num_loops` (default 1) is the total number of invocations of
each block in `[loop_start, loop_end)`, a zero-based, half-open range that defaults to the whole
stack. Blocks outside the range run once. For eight physical layers and the range `[2, 6)`, two loops
execute:

- `stack`: `1 2 -> 3 4 5 6 -> 3 4 5 6 -> 7 8` (one-based labels).
- `interleave`: `1 2 -> 3 3 4 4 5 5 6 6 -> 7 8`.

Each physical block is registered once, and all its attention, FFN, norm, gate and Canon parameters
are reused. Canon kernels keep their physical indices. Embeddings run once before the schedule, and
final normalization and task prediction run once after it. Residue positions and padding masks stay
the same. Training uses the ordinary final-output loss and differentiates through every occurrence.
Dropout samples normally per invocation, with checkpoint recomputation preserving that invocation's
RNG state.

The effective depth is `D = L + (num_loops - 1) * (loop_end - loop_start)` after resolving a null end
to L. Requested hidden states contain D+1 tensors (embedding plus each pre-final-norm block output),
and attentions contain D tensors. The backbone's execution order (`layer_execution_order`, on `backbone` of an `OplmModel`
and `oplm.backbone` of the task models) maps occurrences to physical blocks. One loop reproduces the ordinary traversal.

Residual scaling and projection initialization continue to scale with physical L, not D, so
converting a checkpoint does not alter its weights or scaling buffers. When value residuals are on,
the first execution of physical block zero provides the global reference for all later blocks. Every
invocation of block zero bypasses mixing, and revisiting it never replaces or detaches the reference.
Changing loop settings requires constructing or loading a model with the new config.

Training compute estimates use effective depth D for the block projections and count the MLM head
once. Unique parameter counts use physical blocks and are unchanged by looping. Activation memory
grows with executed depth unless reduced through activation checkpointing, because parameter sharing
alone does not reduce it.

## μP hooks

With `mup_enable=true` the model applies the width multiplier `m = hidden_size / mup_base_width`
(`mup_width_mult()` on the config) in two places. Init: hidden weight matrices have their std divided
by `sqrt(fan_in multiplier)` (`mup_fanin_mult`). Readout: output logits are scaled by
`mup_output_mult / m`. At `hidden_size == mup_base_width` both are identities, so init and forward
are bit-identical to a non-μP model. The per-group learning rates that complete μP live in the
trainer. The recipe, the optimizer interplay and the workflow for transferring `train.lr` across
presets are in [mup-lr.md](mup-lr.md#recipe). `mup_base_width`, `mup_output_mult` and the train-side multipliers
are one tuned set, so do not change one without re-validating the others.

## Initialization and runtime contract

**Initialization.** Linear, embedding and Conv1d weights are truncated normal with std
`initializer_range` (0.02), biases are 0, and norm gains are 1 (LayerNorm biases 0). Residual-writing
projections (attention output, FFN down) get `initializer_range / sqrt(2L)` when
`init_scale_output_projections` is on. The MLM decoder is not a residual writer.

**HuggingFace integration.** Two mechanisms, both set up by the model-repo maintainer:

1. In-process registration. `import oplm` registers the auto-classes (`AutoConfig`, the four
   `AutoModel*` classes and `AutoTokenizer` with `OplmTokenizerFast`), so `from_pretrained` works with
   no `trust_remote_code`.
2. `auto_map`. For loading a checkpoint without importing OPLM first, `config.json` points HF at the
   in-repo `modeling_oplm.py`, `configuration_oplm.py` and `tokenization_oplm.py` (copied by
   `save_pretrained` and `push_to_hub`), and `trust_remote_code=True` is required. HF copies only a
   module's depth-1 relative imports, so `modeling_oplm.py` imports every helper directly. The
   clean-environment round trip (`save_pretrained`, then load in a subprocess with only
   `transformers`) is a mandatory integration test.

`OplmPreTrainedModel` sets `config_class`, `base_model_prefix="oplm"`, `main_input_name="input_ids"`,
`supports_gradient_checkpointing=True`, `_no_split_modules=["OplmBlock"]`, `_supports_sdpa=True` and
`_supports_flash_attn_2=False` (attention runs through `scaled_dot_product_attention`).

**Save and load.** `save_pretrained` writes `config.json` and the safetensors weights, honoring tied
weights. State-dict keys follow module nesting (`oplm.embed_tokens.weight`,
`oplm.layers.0.self_attn.q_proj.weight`, and so on). With tying on, only `embed_tokens.weight` is
stored and the head is re-tied at load. Sharded loading uses the standard HF safetensors index.

**Gradient checkpointing.** `gradient_checkpointing=true` arms the stack and blocks at init, and
`model.gradient_checkpointing_enable()` propagates to every block. `gradient_checkpointing_mode` is
`full` (recompute the whole block on backward, maximum memory savings) or `selective` (PyTorch
selective activation checkpointing keeps matmul, SDPA and FlexAttention outputs resident and
recomputes only cheap ops). The mode can also be passed at enable time with
`gradient_checkpointing_kwargs={"mode": "selective"}`. Both modes are `torch.compile`-compatible, and
the policy is a module-level function so Dynamo can trace it. On multi-GPU with `train.compile`, the
trainer disables DDPOptimizer for `selective` so graph splitting does not fragment it into a full
recompute.

**Model-side distributed constraints.** No global mutable state (RoPE tables are buffers), no in-place
ops on shared buffers, weight tying re-applied in `tie_weights()` after load. The model never calls
`torch.compile` itself, which is the trainer's job.

## Forward-pass contract

All four `Oplm*` model classes accept the same keyword arguments, ignoring the ones they do not use:

| Argument | Type / shape | Default | Notes |
| --- | --- | --- | --- |
| `input_ids` | `(B,T)` int64 | required | from `OplmTokenizerFast` |
| `attention_mask` | `(B,T)` int64 in {0,1} | `None`, meaning all ones | 1 is real, 0 is pad |
| `labels` | `(B,T)` or `(B,)` int64, or `None` | `None` | MLM (`-100` ignored), per-token, or per-sequence, depending on head |
| `output_attentions` | bool | `False` | `True` forces the manual path in every layer |
| `output_hidden_states` | bool | `False` | return all intermediate states |
| `return_dict` | bool | `True` | `ModelOutput` or tuple |

| Class | Return | Logits |
| --- | --- | --- |
| `OplmModel` | `BaseModelOutput` | `last_hidden_state (B,T,D)` |
| `OplmForMaskedLM` | `MaskedLMOutput` | `(B,T,V)` |
| `OplmForSequenceClassification` | `SequenceClassifierOutput` | `(B,num_labels)` |
| `OplmForTokenClassification` | `TokenClassifierOutput` | `(B,T,num_labels)` |

Each carries `loss` (when `labels` is given), `hidden_states` and `attentions`. MLM and token loss is
`cross_entropy(logits.view(-1,C), labels.view(-1), ignore_index=-100)`. Sequence loss is
`cross_entropy(logits, labels)`. `output_hidden_states=true` returns `L+1` tensors (post-embedding,
then each block's post-residual output), all `(B,T,D)`. `output_attentions=true` adds about
`B*L*H*T^2` fp32 floats of activation memory, so do not enable it casually in training.

## Numerical precision and determinism

**Dtypes.** Training uses bf16 autocast on CUDA with fp32 master weights managed by Accelerate. Norms
(including QK-norm), attention softmax and RoPE compute in fp32 internally and cast back. Logits and
loss are fp32. Inference loads bf16 on CUDA and fp32 on CPU (override with `torch_dtype=`). Use
`model.eval()` with `torch.inference_mode()`. fp32 norms remove the bf16 sum-of-squares drift seen in
long-context training.

**Determinism.** The manual attention path is bitwise reproducible given seeded inputs and identical
hardware, so use it for precision-critical evaluation (contact precision, rollout, ablation deltas).
The fused SDPA path can differ across torch, CUDA and architecture versions (typically under 1e-3
relative, no bit equality), so pin every version for paper-grade numbers. Seeding is the trainer's job.

## Parameter count

With `n_norm = 2` for LayerNorm and `1` for RMSNorm (residual scaling is parameter-free), the total
is: embedding `V*D`, final norm `n_norm*D`, MLM head `(D*D + D) + n_norm*D + (D*V + V untied, or V
tied)`, per layer `[attention 4*D*D + QK-norm 2*n_norm*d + pre-norm n_norm*D (+ extra norms per
strategy)] + [FFN 3*D*F + pre-norm n_norm*D]`, and `k_l*D` per Canon convolution (plus the optional
gates and value-residual scalars). Worked example: D=768, L=24, H=12, F=2048, V=33, LayerNorm, untied,
pre-norm, no Canon is about 170M parameters, the `170M` preset. `oplm info --preset <name>` prints the
resolved architecture and the exact count for any preset. The preset dimensions are in config.md,
*Size presets*.

## Removed features

Tried, no consistent gain, and not available even as toggles. Re-introducing one requires a deliberate
revisit: grouped-query attention (`num_kv_heads < num_heads`), attention residuals (Kimi-style),
shared K/V projection, multi-token value embeddings, and query-dependent attention gates.

## Module map

| Package under `src/oplm/` | Owns |
| --- | --- |
| `model/` | The transformer, tokenizer, `OplmConfig` (HF `PretrainedConfig`), public API. |
| `data/` | On-disk formats to model-ready tensors, by modality. See [data.md](data.md). |
| `eval/` | Task registry, per-task scheduling, metric math. |
| `training/` | The `Trainer` loop, optimizer and scheduler construction, FLOPs accounting, checkpointing. |
| `config.py` | The run config (`model`, `train`, `data`, optional `fold`) and `load_config`. See [config.md](config.md). |
| `cli.py`, `train.py`, `inference.py` | Entry points and inference helpers. |
