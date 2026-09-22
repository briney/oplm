# Looped OPLM — Design

**Date:** 2026-09-21
**Status:** Draft for written-spec review; implementation has not started.
**Scope:** Fixed-count, fully shared transformer recurrence and weights-only
initialization of a separate training stage.

## 1. Intent and agreed requirements

Enable experiments that increase OPLM's computation depth without increasing its
unique transformer parameters. Support:

- Ordinary, non-looped training with existing behavior preserved.
- Looped training from scratch.
- Loading a non-looped pretrained checkpoint for looped inference.
- Training a non-looped model, then initializing a separate looped training stage
  from its weights; for example, 1M ordinary steps followed by 500k looped steps.
- Repeating the selected stack in order or repeating each selected layer before
  advancing to the next layer.
- Repeating either all transformer layers or a contiguous subset.

The user confirmed that a separate training stage is sufficient and that it
should initialize weights only, with a fresh optimizer and learning-rate schedule.
Optimizer-state transfer and automatic transitions within a run are out of scope.

The remaining detailed policies below are proposed defaults for review. No claim
is made that direct conversion improves protein representations, or that 500k
additional steps is the right adaptation budget.

## 2. Configuration and execution semantics

Add these fields to `OplmConfig` and document them in the model base YAML:

```yaml
model:
  num_hidden_layers: 8
  num_loops: 2            # total executions of each selected layer; default 1
  loop_strategy: stack   # stack | interleave; default stack
  loop_start: 2           # zero-based, inclusive; default 0
  loop_end: 6             # zero-based, exclusive; default null (all remaining layers)
```

`num_hidden_layers` continues to mean the number of unique physical blocks.
`num_loops`, `loop_start`, and a non-null `loop_end` must be integers, not booleans
or silently truncated fractional values. Require `num_loops >= 1` and
`0 <= loop_start < resolved_loop_end <= num_hidden_layers`, even for one loop.
Reject unknown strategies. Resolve a null end to `num_hidden_layers`.

For the example above, using one-based labels for illustration:

| Strategy | Execution order |
|---|---|
| `stack` | `1 2 → 3 4 5 6 → 3 4 5 6 → 7 8` |
| `interleave` | `1 2 → 3 3 4 4 5 5 6 6 → 7 8` |

The prefix and suffix execute once. Selecting the entire range produces
`1…8, 1…8` or `1,1,2,2,…,8,8`. A single loop always produces ordinary execution.

For physical depth L, selected width W, and loop count R, effective depth is
`D = L + (R - 1) * W`. The current 170M preset has 24 physical layers, so two
full-stack loops execute 48 blocks without changing its parameter count.

Use one pure resolver for the execution index tuple and effective depth, shared
by model execution and FLOP accounting. Put it in a focused model module such as
`src/oplm/model/looping.py`; preserve HuggingFace remote-code dependency bundling
when adding relative imports. Derived execution order is not a second serialized
source of configuration.

The execution order is fixed for each constructed model and exposed for
inspection as `OplmStack.layer_execution_order`. Loop fields are validated again
at model construction, covering configs edited after deserialization. Selecting
a different inference loop count/order is supported by constructing/loading a
model with a different config. Per-forward overrides and in-place changes after
compilation are outside the first version.

## 3. Model architecture and sharing

Keep the existing `OplmStack.layers` ModuleList with exactly L blocks. Execute
`layers[index]` according to the resolved order. Do not create block copies,
register aliases under additional layer names, or repeatedly call the full model.

All parameters of a physical block are shared across occurrences, including
attention, FFN, normalization, residual gates, value-mixing parameters, and Canon
convolutions. Gradients from all occurrences accumulate into those parameters.
Existing embedding/head tying remains controlled by its existing configuration.

The computation is:

1. Embed tokens once, including existing embedding normalization/mask dropout.
2. Execute the prefix, repeated region, and suffix on the evolving hidden state.
3. Apply final normalization once and the task head once.

Residue positions and padding masks remain unchanged across occurrences. Canon
kernel selection continues to use physical layer indices. Dropout, when enabled,
runs normally on each invocation; sharing weights does not mean sharing dropout
masks. Activation-checkpoint recomputation must preserve the corresponding RNG
behavior as it does today.

Keep the existing final-output MLM objective. Backpropagate through every
execution; do not detach hidden states between loops. No auxiliary per-loop loss,
loop embeddings, additional normalization at loop boundaries, learned exit gate,
or new input injection is introduced.

## 4. Residual behavior

### Depth scaling

Preserve current physical-depth scaling for both scratch training and conversion.
`sqrt_num_layers` still uses L, as does optional output-projection initialization.
Changing R does not rescale pretrained weights or persistent `alpha` buffers.
This makes recurrence the sole architectural change in the initial comparison.

Effective-depth scaling and damped recurrence are future experimental policies,
not implicit consequences of setting `num_loops`. Tied repeated updates are
correlated, so ordinary independent-layer scaling arguments do not establish an
optimal recurrence scaling law.

### ResFormer value residuals

Keep the values produced by the first execution of physical block zero as the
global reference for the whole forward pass. Every invocation of physical block
zero bypasses value mixing, and every later physical block uses that reference.
Revisiting block zero does not replace the reference or detach it.

This requires an explicit guard in attention: block zero has no
`value_residual_lambda`, while the current code accesses that attribute whenever
a reference is supplied. Merely repeating the existing stack loop would fail on
the second invocation of block zero when value residuals are enabled.

The same rule applies to full-stack and subset looping, both strategies, and
fixed or learnable value mixing. Refreshing the reference per pass is deferred.

## 5. Outputs and accounting

Preserve output types and existing normalization conventions. When requested,
hidden states contain the embedding state followed by each executed block's
pre-final-norm output (`D + 1` entries); attentions contain D entries. Consumers
can map occurrences to physical blocks through `layer_execution_order`.

Final embeddings, pooling, and task heads consume the same designated output as
before, now following the complete execution sequence. Keep ordinary one-loop
behavior unchanged across the public model classes.

Report physical depth, effective depth, loop range/strategy, and unique parameter
count in run configuration/logging. FLOP estimation multiplies per-block work by
D and counts the head once. Keep the estimator's existing documented omissions;
looping does not require an unrelated FLOP-estimator rewrite.

Weights and optimizer-state storage remain approximately unchanged; compute and
activation memory grow with executed depth. Activation checkpointing remains
available per block invocation. Verify repeated-module execution under existing
DDP, FSDP2/HSDP, compilation, and checkpointing paths rather than assuming sharing
guarantees their correctness or performance.

## 6. Checkpoint loading and a separate training stage

### Model loading

Keep existing state-dict keys and tensor shapes. Save all four loop fields in
`config.json`; old checkpoints lacking them default to one full-stack pass.

The intended library workflow is:

```python
config = OplmConfig.from_pretrained(checkpoint)
config.num_loops = 2
config.loop_strategy = "stack"
config.loop_start = 2
config.loop_end = 6
model = OplmForMaskedLM.from_pretrained(checkpoint, config=config)
```

Save/load must preserve execution behavior without creating duplicate layer
weights or losing existing embedding/head ties. Loading one loop from a looped
checkpoint is also mechanically supported, without promising its accuracy.

### Training initialization

Add `train.init_from: str | None = None`, naming a local HuggingFace OPLM model
directory, including the existing `checkpoint-<step>/hf` export. A training
checkpoint root may resolve to its `hf` child. Hub identifiers and additional
remote download/revision controls are outside this first version.

Example stage-two overlay, composed with the same architecture configuration
used for stage one:

```yaml
model:
  num_loops: 2
  loop_strategy: stack
  loop_start: 0
  loop_end: null
train:
  init_from: outputs/nonlooped/checkpoint-1000000/hf
  output_dir: outputs/looped
  max_steps: 500000
```

The resolved target model config remains authoritative. Validate source/target
model compatibility before loading: physical dimensions and forward-affecting
non-loop settings must agree, including normalization, residual scaling, RoPE,
Canon, value residuals, dropout, embedding/head tying, and muP settings. Use a
normalized semantic comparison, excluding serialization metadata, output-reporting
flags, gradient-checkpointing settings, and initialization-only values that do
not change loaded tensors. Report incompatible fields clearly. Reject missing,
unexpected, or mismatched model tensors except existing legitimate tied-weight
serialization handling; do not silently initialize missing trainable weights.

Load source weights before sharding, optimizer construction, and compilation.
Then initialize fresh optimizer state, schedulers, RNG from the new stage's seed,
training counters, data iteration, and tracking identity. `max_steps` counts
additional steps in this stage. The data stream starts anew under its configured
seed; parent data-cursor transfer is not part of weights-only initialization.

Record the initialization source in resolved configuration and run provenance.
Use a distinct output directory for the new stage. This also prevents automatic
resume from selecting the parent stage's checkpoint.

### Resume precedence and compatibility

Resolve explicit/automatic resume first. If a stage resume target exists, restore
that checkpoint and skip `init_from` entirely, including accessing its path. A
stage must remain resumable if its original initialization checkpoint is removed.
If no automatic-resume target exists, initialize from `init_from` when provided,
otherwise use scratch initialization. An invalid explicit resume must still fail;
it must not fall back to the initialization source.

Retain existing full-state resume semantics and LR-schedule compatibility checks.
Additionally compare normalized loop settings against saved stage settings before
restoring state: old absent fields take their defaults, and null/full-range ends
are equivalent. Reject strategy, count, or range drift even when tensor shapes
match. Changing loop execution is a new weights-only stage, not exact resume.

## 7. Implementation boundaries and acceptance criteria

Expected changes are confined to model config/execution/value-residual handling,
training config/bootstrap/checkpoint validation/FLOP accounting, associated tests,
and model/training/configuration documentation. No new model family, optimizer,
Slurm phase system, or training-stage orchestration framework is required.

Acceptance criteria:

1. Old configs and checkpoints retain their existing one-loop outputs, state-dict
   keys, parameter counts, and output tuple lengths.
2. Both strategies and full/subset ranges produce the specified order, including
   a one-layer range and ranges including physical layer zero. Invalid ranges,
   counts, and strategies fail clearly.
3. Forward outputs and shared-parameter gradients match a manually unrolled
   reference, including value-residual and Canon variants. Optimizers contain
   each physical parameter once.
4. HF round trips preserve weights, loop settings, logits, and existing tied
   embeddings/head behavior; remote-code loading includes new dependencies.
5. Weights-only stage initialization preserves source tensors and starts fresh
   training state. Incompatible source configs and incomplete weights fail.
6. Interrupted stage resumption takes precedence over initialization, survives a
   missing original source, restores existing full state, and rejects loop drift.
7. Effective-depth output lengths and FLOP estimates are correct; one-loop
   estimates remain unchanged.
8. Exercise eager and compiled execution, full/selective activation checkpointing,
   DDP and HSDP repeated-module forward/backward and save/resume on appropriate
   hardware. Any unavailable checks must be reported, not claimed as passing.
9. Run focused meaningful tests, the applicable existing suite, `ruff check src/`,
   and `ty check src/` for implementation. This documentation-only stage requires
   spec review and whitespace/link checks rather than model tests.

## 8. Experiments and deferred extensions

Begin with short adaptation/LR pilots from one ordinary checkpoint, comparing
ordinary continuation with stack and interleaved looping. Include conversion-only
evaluation, held-out MLM loss, downstream protein metrics, activation/gradient
stability, throughput, and peak memory. Compare equal-token and equal-compute
budgets; two full loops approximately double backbone work per step. muP width
transfer does not establish learning-rate transfer across recurrence depths.

Deferred extensions: variable-depth curricula, auxiliary intermediate losses,
adaptive exits, truncated backpropagation, pass-specific adapters/norms, residual
damping or effective-depth scaling, alternate value-reference policies, arbitrary
noncontiguous schedules, runtime loop overrides, and optimizer/data-cursor transfer
between stages. Stack looping has natural complete-pass exits; interleaving does
not generally yield complete shallower-model outputs at intermediate boundaries.

## 9. Literature informing the design

- [Scaling Latent Reasoning via Looped Language Models (Ouro)](https://arxiv.org/html/2510.25741v5):
  recurrent shared stacks with intermediate objectives and adaptive exit gates.
  The proposed OPLM first version implements fixed recurrence, not the full recipe.
- [Relaxed Recursive Transformers](https://arxiv.org/html/2410.20672v3):
  recursive parameter sharing and the CYCLE/SEQUENCE distinction; per-depth LoRA
  relaxation is outside this design.
- [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/html/2511.07384v1):
  evidence for continued training and recurrence curricula, using a different
  architecture and text tasks rather than protein MLM.
- [Training-Free Looped Transformers](https://arxiv.org/html/2605.23872v1):
  reports degradation from naive repetition and investigates damped recurrence.
  Direct checkpoint conversion is an experimental capability, not an accuracy
  guarantee.
