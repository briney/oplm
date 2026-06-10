# Technical Analysis: Silent Architecture Errors in OPLM

Date: 2026-06-10

## Executive Summary

This audit was triggered by a silent Canon-B implementation error: the code ran
without exceptions, and tests passed, but the implemented operation was not the
operation intended for the experiment. That class of bug is especially dangerous in
OPLM because the project is evaluating architectural changes where "does this
feature help?" only has meaning if the feature under test is implemented with the
intended semantics.

The most important conclusion is that Canon-B has been corrected at the highest-risk
level: it is now applied after Q/K/V projection rather than before projection. The
remaining Canon implementation, however, should not yet be treated as paper-exact.
The current code implements a useful Canon-inspired bidirectional convolution family,
but several operations do not match the Canon layer placements and residual forms
needed for a paper-exact ablation.

The desired target for OPLM is:

- paper-exact Canon operation placement inside the transformer layer;
- paper-exact residual Canon behavior for the reported residual variants;
- encoder-appropriate convolution windows, meaning centered bidirectional same-length
  convolutions rather than causal/backward-looking convolutions;
- tests that assert operation order and numerical equivalence against tiny reference
  implementations, not only that modules exist and tensors have the right shape.

The second major risk is configuration permissiveness. `load_config()` currently
lets unknown `model.*` keys pass through into HuggingFace `PretrainedConfig`
metadata. A typo such as `model.cannon_enabled=true` is accepted and retained while
the actual `model.canon_enabled` remains `False`. This can silently disable an
ablation and should be fixed before relying on experiment results.

## Scope and Method

The audit reviewed the following areas:

- model architecture: `src/oplm/model/attention.py`,
  `src/oplm/model/transformer.py`, `src/oplm/model/ffn.py`,
  `src/oplm/model/conv.py`, `src/oplm/model/modeling_oplm.py`;
- model and run configuration: `src/oplm/model/configuration_oplm.py`,
  `src/oplm/config.py`, config YAML files, docs, and config tests;
- data collation and masking: `src/oplm/data/sequence/collate.py`;
- training/eval surfaces likely to encode implementation assumptions;
- test coverage around Canon, config, masking, and structure evaluation.

External reference:

- Canon paper: Zeyuan Allen-Zhu, "Physics of Language Models: Part 4.1,
  Architecture Design and the Magic of Canon Layers", arXiv:2512.17351.
  https://arxiv.org/abs/2512.17351

This document intentionally distinguishes between:

- **paper-exact transformer operations**: the location, residual form, and tensor
  stream on which a Canon layer acts;
- **encoder adaptation**: replacing a causal convolution window with a centered
  bidirectional convolution window because OPLM trains a bidirectional MLM encoder.

The latter is an intended adaptation. The former should match the paper.

## Current Canon Implementation

Current Canon-related code is concentrated in:

- `src/oplm/model/conv.py`: `CanonConv`, a depthwise 1D convolution over `(B, T, D)`;
- `src/oplm/model/attention.py`: Canon-B modules on Q/K/V;
- `src/oplm/model/transformer.py`: Canon A/C/D wiring in `OplmBlock`;
- `src/oplm/model/configuration_oplm.py`: Canon config validation and kernel-size
  resolution.

Current behavior:

- `CanonConv` is depthwise, bidirectional, same-length, and zeroes pad positions
  before convolution.
- `CanonConv` supports arbitrary kernel sizes, per-layer schedules, and optional
  activations.
- Canon-B lives inside attention and is applied to projected Q/K/V after QK/V
  normalization and before value residual and RoPE.
- Canon-A is applied as `x = x + conv_a(x)` before attention normalization.
- Canon-C is applied to `attn_out` before the attention residual add.
- Canon-D is applied to the hidden-size FFN input before the entire FFN.
- Canon-B, C, and D mostly replace the stream they are applied to rather than using
  an explicit residual Canon form. Canon-A is residual.

This is internally coherent code, but it is not paper-exact Canon in the sense needed
for architecture ablations.

## Finding 1: Canon A/C/D Placement Is Not Paper-Exact

### Why this matters

If an experiment enables `canon_positions=["A", "C", "D"]`, the label implies that
the same operations described by the Canon paper are being tested. Today, that is not
true for A/C/D.

This can produce the same class of silent error as the original Canon-B bug:

- all tensor shapes are valid;
- gradients flow;
- tests can pass;
- model quality may change;
- but the ablation is not testing the intended architecture.

### Current behavior

In `OplmBlock._forward_impl()`:

```python
if hasattr(self, "conv_a"):
    x = x + self.conv_a(x, attention_mask)

a_in = x if self.norm_strategy == "hybrid" else self.attn_norm(x)
attn_result = self.attention(a_in, attention_mask, output_attentions, ...)

if hasattr(self, "conv_c"):
    attn_out = self.conv_c(attn_out, attention_mask)

h = x + self.alpha * self._gate_attn(attn_out)

h_norm = self.ffn_norm(h)
f_in = h_norm
if hasattr(self, "conv_d"):
    f_in = self.conv_d(f_in, attention_mask)
ffn_out = self.ffn(f_in)
```

This means:

- A acts on the raw residual stream before attention norm.
- C acts on the attention output.
- D acts on hidden-size normalized FFN input before the entire FFN.

### Paper-exact target, adapted for an encoder

The target should be:

- **Canon-A**: acts at the paper's attention-side insertion point, after the
  attention pre-norm and before attention. In OPLM terms, this should be applied to
  the tensor that will be passed into Q/K/V projection, not to raw `x` before norm.
- **Canon-B**: acts inside attention after Q/K/V projection. The current
  post-projection placement satisfies the key requirement, but residual behavior and
  ordering relative to OPLM-specific QK/V norm should be explicitly specified and
  tested.
- **Canon-C**: acts at the paper's MLP-side insertion point, after the FFN pre-norm
  and before the FFN. It should not act on the attention output.
- **Canon-D**: acts inside the MLP before the activation. For SwiGLU/GEGLU, this
  implies an intermediate-size convolution on the activation/gate branch before the
  nonlinearity, not a hidden-size convolution on the FFN input before all FFN
  projections.

The convolution should remain bidirectional and centered for OPLM. The target is
not causal Canon. The paper's local-mixing operation should be adapted from a
decoder window to an encoder window while preserving the operation's layer placement.

### Recommended fix

Refactor `OplmBlock` around explicit Canon insertion sites:

```python
# Attention side.
attn_base = self.attn_norm(x)
attn_in = attn_base
if self.conv_a is not None:
    attn_in = attn_base + self.conv_a(attn_base, attention_mask)

attn_out = self.attention(attn_in, attention_mask, ...)
h = x + self.alpha * self._gate_attn(attn_out)

# MLP side.
mlp_base = self.ffn_norm(h)
mlp_in = mlp_base
if self.conv_c is not None:
    mlp_in = mlp_base + self.conv_c(mlp_base, attention_mask)

ffn_out = self.ffn(mlp_in, attention_mask=attention_mask)
y = h + self.alpha * self._gate_ffn(ffn_out)
```

This sketch deliberately omits non-pre-norm strategies. Paper-exact Canon should
initially be supported under `norm_strategy="pre"` unless the project defines and
tests an exact mapping for `sandwich`, `hybrid`, and `post_sdpa`. Allowing every
combination to run is not the same as knowing what each combination means.

## Finding 2: Canon-D Is Currently in the Wrong Tensor Space

### Current behavior

Canon-D uses `CanonConv(config.hidden_size, ...)` in `OplmBlock`, and is applied to
`h_norm`, a `(B, T, D)` tensor before the FFN.

### Why this is likely wrong

The paper-exact D position is inside the MLP before activation. In a standard MLP,
that is after the input projection to the FFN intermediate width and before the
activation. In OPLM's FFN variants:

- SwiGLU has a gate branch and an up/value branch;
- GEGLU has analogous gated structure;
- squared-ReLU has a single activation branch;
- the activation branch is intermediate width, not hidden width.

Therefore a paper-exact Canon-D implementation should use `intermediate_size`
channels and live inside the FFN module.

### Recommended fix

Move Canon-D from `OplmBlock` into the FFN implementation.

For SwiGLU:

```python
gate = self.gate_proj(x)  # (B, T, F)
if self.conv_d is not None:
    gate = gate + self.conv_d(gate, attention_mask)
up = self.up_proj(x)      # (B, T, F)
return self.down_proj(F.silu(gate) * up)
```

For GEGLU:

```python
gate = self.gate_proj(x)
if self.conv_d is not None:
    gate = gate + self.conv_d(gate, attention_mask)
up = self.up_proj(x)
return self.down_proj(F.gelu(gate) * up)
```

For squared-ReLU:

```python
hidden = self.up_proj(x)
if self.conv_d is not None:
    hidden = hidden + self.conv_d(hidden, attention_mask)
return self.down_proj(torch.square(F.relu(hidden)))
```

This requires either:

- passing `attention_mask` into FFN `forward()` methods; or
- returning a callable from the block that applies D between FFN projection and
  activation, which is less clean.

The FFN constructor should receive enough config context to instantiate D at
`intermediate_size`, or `make_ffn()` should accept an optional `canon_d_factory`.

## Finding 3: Canon Residual Behavior Is Inconsistent

### Current behavior

- A is residual: `x + conv_a(x)`.
- B replaces Q/K/V with convolved Q/K/V.
- C replaces `attn_out` with convolved `attn_out`.
- D replaces `f_in` with convolved `f_in`.

### Why this matters

The Canon paper highlights residual Canon variants as stable and important. If OPLM
is targeting those variants, every enabled Canon layer should have the corresponding
identity path unless an experiment explicitly requests a non-residual variant.

Replacement vs residual Canon is not a cosmetic difference. A replacement
convolution can destroy the identity path through a sublayer input, changes
initialization sensitivity, and makes the feature harder to compare with paper
results.

### Recommended fix

Use residual Canon by default for paper-exact mode:

```python
z = z + canon(z, attention_mask)
```

Apply this principle to:

- A on attention pre-norm input;
- B on projected Q/K/V streams;
- C on FFN pre-norm input;
- D on FFN intermediate activation branch.

If non-residual Canon remains useful as a research knob, expose it under an explicit
name such as:

```yaml
model.canon_residual: true
model.canon_variant: paper_exact_encoder
```

Do not let the old replacement behavior silently share the same config label as
paper-exact residual Canon.

## Finding 4: The Bidirectional Encoder Convolution Needs a Pinned Definition

### Current behavior

`CanonConv` is already bidirectional and same-length. It is not causal. It zeroes pad
positions before convolution, which is the right direction for encoder MLM training.

For even kernels, current padding is asymmetric:

```python
F.pad(x, (kernel_size // 2, kernel_size // 2 - 1))
```

For `kernel_size=4`, this gives a slightly left-biased window. That is acceptable
only if it is deliberate and tested.

### Target behavior

OPLM should not use causal Canon windows. The target is a centered bidirectional
window around each token.

There is one unavoidable detail: exact centering is only mathematically clean for
odd kernel sizes. For even kernel sizes such as the paper's `k=4`, a same-length
discrete convolution must choose one of two half-token alignments.

Recommended policy:

1. Keep the convolution bidirectional and same-length.
2. Use an explicit `CanonConv` alignment policy and test it with impulse responses.
3. Prefer odd kernel sizes for experiments where exact symmetry around the current
   token is more important than preserving the paper's kernel size.
4. If preserving `k=4` is more important, document the chosen half-token convention
   as the encoder adaptation of the paper's causal window.

For a paper-exact encoder mode, the config should probably reject arbitrary
activation functions and undocumented kernel schedules unless those are separate
non-paper research knobs.

## Finding 5: Canon-B Is Corrected at the Main Placement Boundary, but Still Needs
Semantic Tests

### Current behavior

`OplmAttention.forward()` now does:

```python
q, k, v = self._project_qkv(x)
q, k = self._qk_norm(q, k)
v = self._apply_v_norm(v)
if self.canon_b_enabled:
    q, k, v = self._apply_canon_b(q, k, v, attention_mask)
if value_residual is not None:
    v = lam * v + (1.0 - lam) * value_residual
q, k = self._apply_rope(q, k)
```

This fixes the critical bug: Canon-B is not applied before Q/K/V projection.

### Remaining ambiguity

The paper-exact statement "after Q/K/V projection" is clear. OPLM also has
architecture features that may not exist in the same form in the paper baseline:

- QK norm;
- optional V norm under `norm_strategy="hybrid"`;
- value residual;
- partial RoPE/NoPE;
- attention output gating.

Current code applies Canon-B after QK/V norm and before value residual and RoPE.
That may be a reasonable OPLM policy, but it must be explicitly specified as the
paper-exact encoder target or adjusted.

Recommended target:

- Canon-B must be after Q/K/V projection.
- Canon-B should use residual form on each stream in paper-exact mode:

```python
q = q + conv_b_q(q, attention_mask)
k = k + conv_b_k(k, attention_mask)
v = v + conv_b_v(v, attention_mask)
```

- The order relative to QK norm should be codified in docs and tests. If the goal is
  closest paper operation, prefer:

```text
QKV projection -> residual Canon-B -> QK/V normalization, RoPE, attention
```

If the goal is to preserve existing OPLM stability assumptions, retain:

```text
QKV projection -> QK/V normalization -> residual Canon-B -> RoPE, attention
```

Either is defensible. Leaving it implicit is not.

## Finding 6: Config Loading Silently Absorbs Misspelled Architecture Keys

### Current behavior

`load_config()` converts the merged `model` subtree to a dict and calls:

```python
cfg.model = OplmModelConfig(**model_dict)
```

`OplmModelConfig` subclasses HuggingFace `PretrainedConfig`, which accepts unknown
`**kwargs`. The code comments explicitly note that unknown, old, or mistyped
`model.*` keys are silently retained.

Observed behavior:

```python
cfg = load_config(["model.cannon_enabled=true"])
assert cfg.model.cannon_enabled is True
assert cfg.model.canon_enabled is False
```

This is a silent ablation failure.

### Recommended fix

Make `load_config()` strict for run configs.

Implementation outline:

1. Build the allowed model-key set from `inspect.signature(OplmModelConfig.__init__)`.
2. Remove `self`, `kwargs`, and any internal-only parameters.
3. Add a small explicit allowlist for HuggingFace metadata keys that are expected
   when loading saved configs, if needed.
4. In `load_config()`, reject unknown keys before constructing `OplmModelConfig`.
5. Keep `OplmConfig.from_pretrained()` permissive if checkpoint compatibility
   requires it.

Example:

```python
unknown = set(model_dict) - allowed_model_keys
if unknown:
    raise ValueError(
        "Unknown model config keys: "
        f"{sorted(unknown)}. Check for typos or update the allowlist."
    )
```

Add tests:

- `model.cannon_enabled=true` raises;
- `model.hidden_dimm=1024` raises;
- known HuggingFace metadata, if allowed, does not raise;
- all packaged YAML presets pass strict validation.

## Finding 7: Tests Are Implementation-Reflective Instead of Semantic Oracles

### Current coverage strengths

The current tests cover many useful invariants:

- Canon modules are created only when requested;
- bad Canon positions and kernel sizes raise;
- Canon-B has gradients;
- identity kernels can match no-Canon behavior;
- nontrivial kernels change outputs;
- SDPA/manual paths agree;
- pad content does not leak into real tokens through Canon-B;
- broad toggle combinations instantiate and run.

These are good engineering tests, but they are not enough for architectural
correctness.

### Missing coverage

The tests do not prove:

- Canon-A is applied to the correct normalized tensor;
- Canon-C is applied to the FFN pre-norm input rather than attention output;
- Canon-D is inside the FFN activation branch and uses intermediate width;
- Canon-B is residual or replacement as intended;
- the convolution window alignment is the intended centered encoder adaptation;
- config names map exactly to the intended ablation.

### Recommended tests

Add a new test module, for example:

```text
tests/model/test_canon_semantics.py
```

Recommended test categories:

1. **Operation-order tests with hooks**
   - Instrument `attn_norm`, `attention`, `ffn_norm`, `ffn`, and Canon modules.
   - Assert the call order for A/B/C/D.

2. **Tiny numerical reference tests**
   - Use tiny dimensions, deterministic weights, dropout off, identity norms where
     possible.
   - Compare block output to a hand-coded reference for each Canon position.

3. **Canon-D tensor-space tests**
   - Assert `conv_d.conv.in_channels == config.intermediate_size`.
   - Assert D sees `(B, T, intermediate_size)`, not `(B, T, hidden_size)`.

4. **Residual-vs-replacement tests**
   - Set convolution weights to a simple non-identity kernel.
   - Compare paper-exact mode against `z + conv(z)`, not `conv(z)`.

5. **Centered-conv impulse tests**
   - Put a single nonzero impulse at position `i`.
   - Use all-one depthwise kernels.
   - Assert exactly the expected neighboring output positions are affected.
   - Include pad positions to verify zeroed pads do not contribute.

6. **Config typo tests**
   - Assert typo keys raise at run-config load time.

7. **Golden config tests**
   - Load every packaged preset and representative Canon configs.
   - Assert resolved configs match expected paper-exact defaults.

## Finding 8: Documentation Is Not a Reliable Source of Truth Yet

### Current state

The docs contain useful architecture detail, but they also contain stale or missing
references:

- `docs/CONFIG.md` references `MODEL_ARCHITECTURE.md`, `DATA_TOOLING.md`,
  `EVAL_HARNESS.md`, and `TESTING_E2E.md`; those files are not present.
- Some docs still describe old preset names (`small`, `medium`, `base`, `large`,
  `xlarge`) while code exposes numeric presets (`50M`, `170M`, `400M`, `800M`,
  `1B`, `3B`, `6B`, `12B`).
- `docs/CONFIG.md` says defaults are effective defaults from base YAML, but lists
  constructor defaults for some model dimensions.
- The docs currently describe the implemented Canon-inspired placements, not the
  desired paper-exact encoder target.

### Recommended fix

After the Canon implementation is corrected:

1. Create or restore `docs/MODEL_ARCHITECTURE.md`.
2. Make it the authoritative architecture spec for model semantics.
3. Move Canon diagrams and exact formulas there.
4. Update `docs/CONFIG.md` to distinguish:
   - paper-exact encoder Canon;
   - experimental non-paper variants, if retained.
5. Update README preset and status tables.
6. Add a docs test or simple link check so missing docs are caught.

## Secondary Observations

### MLM masking uses fixed-k banker rounding

`MLMCollator` uses:

```python
k_per_row = torch.round(self._mask_prob * n_eligible.float()).long()
```

PyTorch rounding uses banker rounding for `.5` cases. This is documented in tests
as fixed-k behavior, so it is not necessarily a bug. It should still be treated as
a deliberate policy choice because it differs from independent Bernoulli masking
and from half-up rounding.

Recommended action:

- confirm fixed-k banker rounding is intended;
- document it in the data spec;
- if not intended, switch to an explicit rounding policy and update tests.

### Structure evaluation has range checks, not oracle checks

The slow structure-eval test confirms that categorical-Jacobian P@L runs and returns
a finite value in `[0, 1]`. That is valuable, but it would not catch many semantic
mistakes in scoring, chain selection, contact-map construction, or indexing.

Recommended action:

- add one deterministic oracle test using a tiny fake model and tiny structure where
  the expected pair scores and precision are known exactly;
- keep the real-structure slow test as an integration test.

## Recommended Fix Plan

### Phase 1: Freeze and specify Canon semantics

Before changing code, write the intended Canon spec in one place.

Minimum decisions:

- Canon mode name: `paper_exact_encoder`.
- Convolution: centered bidirectional same-length, not causal.
- Kernel default: decide whether to preserve `k=4` with documented half-token
  alignment or use an odd kernel for exact symmetry.
- Residual behavior: enabled by default for paper-exact Canon.
- Supported norm strategies: likely `norm_strategy="pre"` first.
- B ordering relative to QK/V norm: decide and test.
- D placement for SwiGLU/GEGLU/relu2.

### Phase 2: Implement paper-exact encoder Canon

Implementation tasks:

1. Refactor `CanonConv` or add `CenteredDepthwiseConv1d`.
   - Accept arbitrary channel count, not just hidden size.
   - Keep pad zeroing.
   - Pin and test centered alignment.
   - Remove activation from paper-exact mode.

2. Refactor `OplmBlock`.
   - A: after attention pre-norm, residual into attention input.
   - C: after FFN pre-norm, residual into FFN input.
   - Remove C from attention-output path.
   - Remove D from block-level hidden-size FFN input.

3. Refactor `OplmAttention`.
   - Keep B after Q/K/V projection.
   - Switch B to residual form in paper-exact mode.
   - Pin order relative to QK/V norm and RoPE.

4. Refactor FFN modules.
   - Add optional D inside each FFN implementation.
   - D should operate at `intermediate_size`.
   - Pass `attention_mask` through FFN forward.

5. Update config.
   - Add explicit Canon variant or mode fields if old behavior must remain.
   - Reject incompatible paper-exact settings.
   - Strictly validate unknown run-config keys.

### Phase 3: Add semantic tests

Implement the test categories listed in Finding 7 before trusting new ablations.

Minimum acceptance criteria:

- each Canon position has a numerical reference test;
- D is proven to operate in intermediate space;
- B is proven to operate after projection;
- residual behavior is proven for all positions;
- centered convolution alignment is proven with impulse tests;
- typo config keys raise.

### Phase 4: Update docs and run verification

Update docs after code semantics are fixed:

- `docs/MODEL_ARCHITECTURE.md`;
- `docs/CONFIG.md`;
- `docs/OVERVIEW.md`;
- README preset and architecture sections.

Run:

```bash
python -m pytest tests/model/test_attention.py tests/model/test_transformer.py tests/model/test_conv.py tests/model/test_config.py tests/training/test_config.py -q
python -m pytest -m "not slow"
ty check src/
ruff check src/ tests/
```

Use `python -m pytest` in the current environment unless PATH is fixed, because the
standalone `pytest` executable currently resolves outside the active Conda
environment.

## Priority Ranking

1. **Critical**: Make Canon paper-exact encoder semantics explicit and implement A/C/D
   placement plus residual behavior.
2. **Critical**: Move Canon-D inside FFN activation branch at intermediate width.
3. **Critical**: Add semantic Canon oracle tests.
4. **High**: Reject unknown `model.*` keys in run config loading.
5. **High**: Update architecture/config docs so they are trustworthy.
6. **Medium**: Add structure-eval oracle tests.
7. **Medium**: Confirm and document MLM fixed-k rounding policy.

## Bottom Line

The codebase is in a better state after the Canon-B QKV-placement fix, but it still
contains silent-error risks that can invalidate architectural conclusions. The most
important work is not broad refactoring; it is making the mapping from paper concept
to code mechanically precise and testable.

For Canon specifically, OPLM should implement paper-exact transformer-layer
operations with an encoder-appropriate centered convolution. The convolution should
not be causal, but A/B/C/D must act on the same tensor streams, at the same points,
and with the same residual semantics intended by the Canon architecture.
