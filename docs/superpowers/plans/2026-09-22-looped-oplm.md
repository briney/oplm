# Looped OPLM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable full-stack/subset looping with shared weights and a
weights-only initialization path for a separate training stage.

**Architecture:** Keep one registered module per physical transformer block and
execute a validated tuple of physical indices. Preserve checkpoint tensor names,
physical-depth scaling, final-only prediction, and existing resume machinery.
Separate strict pretrained initialization from exact training-state recovery.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace Transformers, Accelerate,
OmegaConf, safetensors, pytest, Ruff, and ty; no new dependencies or version changes.

**Spec:** [Approved looping design](../specs/2026-09-21-looped-oplm-design.md).

**Planning baseline:** `50f73ec` on `main`. This document is a plan, not a record
of implementation or passing tests. Start execution in an isolated worktree using
the using-git-worktrees skill. Read `AGENTS.md` and the approved spec first.

## Global Constraints

- `num_hidden_layers` continues to mean the number of unique physical blocks.
- Require `num_loops >= 1` and `0 <= loop_start < resolved_loop_end <= num_hidden_layers`, even for one loop.
- The prefix and suffix execute once.
- Keep existing state-dict keys and tensor shapes.
- Changing R does not rescale pretrained weights or persistent `alpha` buffers.
- Keep the values produced by the first execution of physical block zero as the global reference for the whole forward pass.
- Keep the existing final-output MLM objective.
- Backpropagate through every execution; do not detach hidden states between loops.
- Optimizer-state transfer and automatic transitions within a run are out of scope.
- If a stage resume target exists, restore that checkpoint and skip `init_from` entirely, including accessing its path.
- Hub identifiers and additional remote download/revision controls are outside this first version.
- Use `pathlib.Path`, annotated function signatures, future annotations, Google-style public docstrings, Ruff, and `ty check src/` as required by `AGENTS.md`.
- Preserve existing DDP/HSDP restrictions, optimizer ordering, data partitioning, checkpoint commit protocol, and one-loop defaults.
- Do not add runtime loop setters, arbitrary schedules, adaptive exits, intermediate losses, or new scaling knobs.

## Review Focus

These are the five failure cases needing explicit tests beyond the basic execution examples:

1. A config edited after construction contains a boolean/fractional loop count or an invalid range: model construction must reject it (Task 1).
2. A checkpoint is missing a residual buffer or one of the two physically tied embedding/head names: missing independent tensors must fail; valid tied serialization must load (Task 3).
3. Stage 2 is resumed after its parent export was removed: no source-path access, optimizer reset, or fallback to initialization is allowed (Task 4).
4. An old checkpoint omits loop fields, or uses an explicit full-range end: equivalent defaults must resume; actual loop drift must fail before DCP restoration (Task 5).
5. Repeated modules run with nonzero dropout, activation recomputation, and distributed reduction: outputs/gradients must match controls; finite loss alone is insufficient (Tasks 2 and 7).

## File Map and Ownership

| File | Responsibility |
|---|---|
| New `src/oplm/model/looping.py` | Pure validation and index-order resolution; no torch or model imports |
| `src/oplm/model/configuration_oplm.py` | Four serialized loop fields and initial validation |
| `src/oplm/model/transformer.py` | Immutable execution order, shared invocations, existing output conventions |
| `src/oplm/model/attention.py` | Block-zero value-residual bypass |
| `src/oplm/model/modeling_oplm.py` | Remote-code dependency bundling and loading-diagnostics return handling |
| New `src/oplm/training/initialization.py` | Local source resolution, semantic config comparison, strict weight loading |
| `src/oplm/config.py`, packaged model/train base YAMLs | Public config and documented defaults |
| `src/oplm/training/trainer.py` | Resume-first initialization decision and model/run metadata |
| `src/oplm/training/checkpoint.py` | Loop compatibility checks before candidate selection/restoration |
| `src/oplm/training/flops.py` | Effective-depth backbone accounting, single head |
| New `tests/model/test_looping.py` | Resolver, execution, gradients, config mutation, value-reference invariants |
| New `tests/training/test_initialization.py` | Local model artifacts and strict loading behavior |
| New `tests/training/test_e2e_looping.py` | Real-data stage initialization and recovery |
| New `tests/training/test_loop_resume.py` | Resume metadata and mismatch handling |
| New `tests/training/test_e2e_looping_distributed.py`, `_looping_worker.py` | Two-rank parity and actual Trainer save/resume pilots |
| Existing config/save-load/remote-code/gradient-checkpoint/FLOP/compile tests | Focused extensions preserving established fixtures |
| `docs/MODEL_ARCHITECTURE.md`, `docs/CONFIG.md`, `docs/TRAIN.md`, `docs/TESTING_E2E.md` | User semantics and validation coverage |

Execute Tasks 1–7 in order. Tasks 3 and 5 both feed Task 4's complete acceptance
story; the stage-resume regression added in Task 4 initially tests same-config
resumption, with intentional mismatch cases implemented in Task 5. Each task
contains a red/green cycle and a focused commit. Do not run large training jobs.

---

### Task 1: Add the validated loop configuration and order resolver

**Files:**

- Create: `src/oplm/model/looping.py`, `tests/model/test_looping.py`.
- Modify: `src/oplm/model/configuration_oplm.py`, `src/oplm/model/transformer.py` (construction only), `src/oplm/model/modeling_oplm.py` (dependency import only).
- Modify: `src/oplm/configs/model/base.yaml`, `tests/model/test_config.py`, `tests/training/test_config.py`, `docs/CONFIG.md`.

**Interfaces:**

- Consumes: existing `OplmConfig.__init__`, `_validate`, and `load_config(argv)`.
- Produces: `resolve_layer_execution_order(num_hidden_layers: int, *, num_loops: int = 1, loop_strategy: str = "stack", loop_start: int = 0, loop_end: int | None = None) -> tuple[int, ...]`.
- Produces: four same-named config fields and `OplmStack.layer_execution_order: tuple[int, ...]`; no new serialized derived field.

- [ ] **Step 1: Add explicit schedule and validation tests.**

Start `tests/model/test_looping.py` with future annotations and these tests:

```python
import pytest

from oplm.model import OplmConfig
from oplm.model.looping import resolve_layer_execution_order
from oplm.model.transformer import OplmStack


@pytest.mark.parametrize(
    ("strategy", "start", "end", "expected"),
    [
        ("stack", 0, None, (0, 1, 2, 3, 0, 1, 2, 3)),
        ("interleave", 0, None, (0, 0, 1, 1, 2, 2, 3, 3)),
        ("stack", 1, 3, (0, 1, 2, 1, 2, 3)),
        ("interleave", 1, 3, (0, 1, 1, 2, 2, 3)),
        ("stack", 0, 1, (0, 0, 1, 2, 3)),
        ("interleave", 3, 4, (0, 1, 2, 3, 3)),
    ],
)
def test_execution_orders(
    strategy: str, start: int, end: int | None, expected: tuple[int, ...]
) -> None:
    assert resolve_layer_execution_order(
        4, num_loops=2, loop_strategy=strategy, loop_start=start, loop_end=end
    ) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_loops": True}, {"num_loops": 1.5}, {"num_loops": 0},
        {"loop_start": -1}, {"loop_start": True}, {"loop_start": 0.5},
        {"loop_end": False}, {"loop_end": 2.5}, {"loop_end": 0},
        {"loop_end": 5}, {"loop_start": 3, "loop_end": 2},
        {"loop_strategy": "unknown"},
    ],
)
def test_invalid_configuration_and_postconstruction_mutation(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="loop"):
        OplmConfig(num_hidden_layers=4, **overrides)
    cfg = OplmConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=4)
    for name, value in overrides.items():
        setattr(cfg, name, value)
    with pytest.raises(ValueError, match="loop"):
        OplmStack(cfg)
```

Also parametrize valid ranges/strategies with R=1 and assert `(0, 1, 2, 3)`; verify
config JSON and YAML/CLI round trips keep `loop_end: null` and all four fields.
Use actual `load_config` overrides, including fractional/bool inputs, to ensure
OmegaConf does not hide invalid model inputs through coercion.

- [ ] **Step 2: Run the new tests and confirm the expected missing-module failure.**

Run: `pytest tests/model/test_looping.py -q`.
Expected: collection fails because `oplm.model.looping` does not exist. Confirm
the failure is not an environment/dependency issue.

- [ ] **Step 3: Implement resolution and construction-time validation.**

Use this algorithm in `looping.py`, adding its public Google-style docstring:

```python
from __future__ import annotations


def resolve_layer_execution_order(
    num_hidden_layers: int,
    *,
    num_loops: int = 1,
    loop_strategy: str = "stack",
    loop_start: int = 0,
    loop_end: int | None = None,
) -> tuple[int, ...]:
    values = {
        "num_hidden_layers": num_hidden_layers,
        "num_loops": num_loops,
        "loop_start": loop_start,
    }
    if loop_end is not None:
        values["loop_end"] = loop_end
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer; got {value!r}.")
    end = num_hidden_layers if loop_end is None else loop_end
    if num_loops < 1:
        raise ValueError("num_loops must be >= 1.")
    if not 0 <= loop_start < end <= num_hidden_layers:
        raise ValueError("Require 0 <= loop_start < loop_end <= num_hidden_layers.")
    if loop_strategy not in ("stack", "interleave"):
        raise ValueError("loop_strategy must be 'stack' or 'interleave'.")
    region = tuple(range(loop_start, end))
    repeated = (
        region * num_loops
        if loop_strategy == "stack"
        else tuple(index for index in region for _ in range(num_loops))
    )
    return tuple(range(loop_start)) + repeated + tuple(range(end, num_hidden_layers))
```

Add the four keyword arguments to `OplmConfig.__init__` without `int()` casts;
assign them before `_validate`, which invokes this resolver. In `OplmStack.__init__`
invoke the resolver again and store its tuple. Existing unit-test configs use
`SimpleNamespace`: use `getattr(config, field, default)` for the new fields rather
than forcing every unrelated test helper to grow four arguments.

Directly import the resolver in both configuration and modeling modules and add
it to modeling's `_REMOTE_CODE_DEPS`; this keeps new transitive helper code present
in HF custom-code exports. No training import may enter `looping.py`.

Document the four defaults and zero-based, half-open range in the base model YAML
and `docs/CONFIG.md`, including that R counts total executions, not extra repeats.

- [ ] **Step 4: Run resolver and existing configuration regressions.**

Run: `pytest tests/model/test_looping.py tests/model/test_config.py tests/training/test_config.py -q`.
Expected: pass, including packaged-base-YAML field coverage. Correct any test
failure before proceeding to shared execution.

- [ ] **Step 5: Commit this deliverable.**

Commit only the Task 1 files with message `feat(model): configure shared-layer loop execution`.

### Task 2: Execute shared blocks and preserve residual semantics

**Files:**

- Modify: `src/oplm/model/transformer.py`, `src/oplm/model/attention.py`.
- Modify: `tests/model/test_looping.py`, `tests/model/test_gradient_checkpointing.py`, `docs/MODEL_ARCHITECTURE.md`.

**Interfaces:**

- Consumes: `OplmStack.layer_execution_order` from Task 1.
- Produces: existing stack return types, with D+1 hidden states and D attentions;
  no task-head signature changes.

- [ ] **Step 1: Add a numerical reference test independent of the resolver.**

Use two cloned stacks and a fixed, externally specified unrolling:

```python
import copy

import torch


@pytest.mark.parametrize(
    ("strategy", "start", "end", "order"),
    [
        ("stack", 0, None, (0, 1, 2, 0, 1, 2)),
        ("interleave", 0, None, (0, 0, 1, 1, 2, 2)),
        ("stack", 1, 3, (0, 1, 2, 1, 2)),
        ("interleave", 1, 3, (0, 1, 1, 2, 2)),
    ],
)
@pytest.mark.parametrize("value_mode", ["none", "fixed", "learnable"])
def test_forward_and_gradients_match_unrolled_reference(
    strategy: str, start: int, end: int | None, order: tuple[int, ...], value_mode: str
) -> None:
    cfg = OplmConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4,
        num_hidden_layers=3, num_loops=2, loop_strategy=strategy,
        loop_start=start, loop_end=end, value_residual=value_mode,
    )
    actual = OplmStack(cfg).train()
    reference = copy.deepcopy(actual)
    ids = torch.tensor([[0, 20, 9, 9, 14, 16, 2, 1]])  # MEEPQ plus pad
    mask = ids.ne(1).long()
    expected = reference.embed_tokens(ids, mask)
    anchor = None
    for index in order:
        result = reference.layers[index](
            expected, mask, output_attentions=True,
            value_residual=None if index == 0 else anchor,
        )
        expected = result[0]
        if value_mode != "none" and anchor is None:
            anchor = result[2]
    expected = reference.final_norm(expected)
    got, states, attentions = actual(
        ids, mask, output_hidden_states=True, output_attentions=True
    )
    assert states is not None and len(states) == len(order) + 1
    assert attentions is not None and len(attentions) == len(order)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(got)
    (got * probe).sum().backward()
    (expected * probe).sum().backward()
    for (name, param), (ref_name, ref_param) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == ref_name
        assert (param.grad is None) == (ref_param.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5)
```

Both paths use manual attention so kernel choice is not a confounder.
Extend this test with Canon enabled at A/B/C/D
and nonuniform physical kernels `[3, 5, 3]`; also cover scalar/channel residual
gates and each norm strategy in a small, targeted matrix rather than a full
Cartesian product.

Add two controls: explicit R=1 vs the original single physical traversal must
match exactly on CPU with dropout off; identical seeds must create identical
state dicts for R=1 and R=2, proving initialization and persistent alpha do not
depend on effective depth. Compare parameter names/counts and optimizer group
parameter identities for AdamW and Muon+auxiliary AdamW.

- [ ] **Step 2: Confirm the reference tests fail against one-pass execution.**

Run: `pytest tests/model/test_looping.py -q`.
Expected: looped output lengths and numerical reference comparisons fail.

- [ ] **Step 3: Change execution and the block-zero mixing guard.**

Replace only the traversal in `OplmStack.forward`; retain embedding, mask
preparation, final norm, and hidden/attention collection:

```python
v1: torch.Tensor | None = None
for layer_index in self.layer_execution_order:
    block = self.layers[layer_index]
    result = block(x, attention_mask, output_attentions, value_residual=v1)
    if self.value_residual_enabled:
        x, attn, v = result
        if v1 is None:
            v1 = v
    else:
        x, attn = result
    if hidden_states is not None:
        hidden_states = hidden_states + (x,)
    if attentions is not None:
        attentions = attentions + (attn,)
```

Change the attention guard to `if self.layer_idx > 0 and value_residual is not None:`.
Do not add lambda parameters to block zero, change physical layer indices, refresh
`v1`, or detach it. Update the stack and attention docstrings for occurrence counts
and block-zero behavior. Leave alpha and `_init_weights` formulas unchanged.

- [ ] **Step 4: Prove checkpoint recomputation preserves looped gradients.**

Extend existing gradient-checkpointing parity tests with R=2, both strategies,
`value_residual="learnable"`, and full/selective checkpointing. For a dropout
regression, clone an eager model, enable checkpointing on the clone, reset the
torch RNG to the same seed before each forward, use nonzero hidden and attention
dropout, and compare loss/parameter gradients. Exercise a real MLM loss, not the
sum of normalized hidden states. Assert that the first value reference receives
gradients and is not replaced by the second block-zero output.

Run: `pytest tests/model/test_looping.py tests/model/test_transformer.py tests/model/test_attention.py tests/model/test_gradient_checkpointing.py tests/model/test_esmc_api.py tests/model/test_canon_semantics.py -q`.
Expected: pass. Preserve existing input-embedding and padding behavior.

- [ ] **Step 5: Document and commit shared execution.**

Add the execution examples, physical/effective-depth distinction, output tuple
lengths, fixed first-value reference, and physical-depth scaling to
`docs/MODEL_ARCHITECTURE.md`. Commit Task 2 files as
`feat(model): execute tied transformer blocks in configurable loops`.

### Task 3: Preserve HF serialization and implement strict weights-only loading

**Files:**

- Create: `src/oplm/training/initialization.py`, `tests/training/test_initialization.py`.
- Modify: `src/oplm/model/modeling_oplm.py`, `tests/model/test_save_load.py`, `tests/model/test_push_to_hub.py`.

**Interfaces:**

- Consumes: `OplmConfig`, `OplmForMaskedLM.from_pretrained`, existing HF exports.
- Produces: `resolve_initialization_source(source: str) -> Path`.
- Produces: `validate_initialization_config(source: OplmConfig, target: OplmConfig) -> None` (these are model configs).
- Produces: `load_initial_model(source: Path, target: OplmConfig) -> OplmForMaskedLM`.
- Preserves: normal `from_pretrained` model return; supports the standard
  `output_loading_info=True` tuple without breaking tokenizer attachment.

- [ ] **Step 1: Add loading-info and corrupt-export regression tests.**

Use a tiny config (32 hidden, 4 heads, 3 layers), real tokenized `MEEPQ`, and
temporary local exports. Core missing-tensor test:

```python
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from oplm.model import OplmConfig, OplmForMaskedLM
from oplm.training.initialization import load_initial_model


@pytest.mark.parametrize(
    "missing_key",
    ["oplm.backbone.layers.0.alpha", "oplm.backbone.layers.1.ffn.down_proj.weight"],
)
def test_missing_independent_tensor_fails(tmp_path: Path, missing_key: str) -> None:
    cfg = OplmConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=3)
    OplmForMaskedLM(cfg).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    del tensors[missing_key]
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match=missing_key.replace(".", r"\.")):
        load_initial_model(tmp_path, cfg)
```

Parameterize a successful strict-load test over tied/untied embeddings and both
strategies; compare every state-dict tensor with the source, including `alpha`.
Test a normal tied export (only one embedding/head weight stored), then delete
the surviving tensor and require failure. Corrupt an unexpected key and a tensor
shape separately. Test sharded exports via `save_pretrained(max_shard_size="10KB")`.

In `test_save_load.py`, add a test that explicitly requests loading info and
checks `(model, info)`, tokenizer attachment, and empty diagnostic lists. Keep
the existing normal-return test. Parameterize public task-class round trips with
R=2 and a subset; test old `config.json` with loop keys removed defaults to R=1.

- [ ] **Step 2: Run the new tests and inspect the expected failures.**

Run: `pytest tests/training/test_initialization.py tests/model/test_save_load.py -q`.
Expected: the helper import is absent and the loading-info test exposes the
wrapper's attempt to set `.tokenizer` on a tuple. Fix no unrelated failures.

- [ ] **Step 3: Preserve the framework return form when attaching tokenizers.**

Adapt the existing wrapper around this control flow:

```python
result = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
model = result[0] if isinstance(result, tuple) else result
try:
    from transformers import AutoTokenizer

    model.tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=kwargs.get("trust_remote_code", False),
        local_files_only=kwargs.get("local_files_only", False),
    )
except (OSError, ValueError):
    model.tokenizer = None
return result
```

Retain the existing public signature compatibility; annotate it consistently
with the repository's supported Transformers versions. Missing local tokenizer
files remain allowed. Strict initialization must not trigger network access.

- [ ] **Step 4: Resolve local sources and compare semantic model configuration.**

Implement path resolution with `Path(source).expanduser()`; require a directory
and `config.json` either directly or under its `hf` child. Prefer the directory's
own export if both exist. Return a resolved absolute Path; a missing local path
raises `FileNotFoundError`, not a Hub lookup.

Normalize both configs through their model constructors/derived fields before
comparison, without mutating the target object. Build a signature from the named
model-constructor fields, so newly added model fields are compared by default.
Use an explicit documented exclusion set:

```python
_IGNORED_INITIALIZATION_FIELDS = frozenset({
    "num_loops", "loop_strategy", "loop_start", "loop_end",
    "gradient_checkpointing", "gradient_checkpointing_mode",
    "initializer_range", "init_scale_output_projections",
    "residual_gate_init", "qk_norm_l2_scale_init", "value_residual_lambda_init",
    "classifier_pool", "classifier_dropout", "num_labels", "pre_head_norm",
})
```

The last line is task-head-only configuration unused by `OplmForMaskedLM`.
Initialization-only values are safe to ignore only because all corresponding
weights/buffers must load. Serialization/auto-class metadata and output-reporting
flags are not named model-constructor fields and do not enter the signature.

Canonicalize active optional settings: remove Canon positions/kernels/activation/
residual mode if Canon is disabled; sort active positions and compare resolved
kernel tuples if enabled. Ignore muP base-width/output-multiplier when muP is off,
QK normalization mode when QK normalization is off, and mask-dropout reference
ratio when mask dropout is off. Do not ignore dropout probabilities, norm
strategy, RoPE geometry, residual scaling, tying, token IDs, or physical shape.
Use source and target values in each mismatch diagnostic.

Parameterize config tests with a changed attention-head count that preserves
weight shapes, norm strategy, RoPE theta, residual scaling, active Canon kernels,
and active muP output multiplier; all must fail. Different loop fields,
checkpointing settings, inactive optional settings, and initialization-only
values must load. These catch errors that tensor-shape checking cannot detect.

- [ ] **Step 5: Load weights and reject incomplete artifacts.**

The normal path is:

```python
source_config = OplmConfig.from_pretrained(source, local_files_only=True)
validate_initialization_config(source_config, target)
model, info = OplmForMaskedLM.from_pretrained(
    source, config=target, local_files_only=True, output_loading_info=True
)
problems = {
    name: info.get(name, [])
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if info.get(name)
}
if problems:
    raise ValueError(f"Incomplete pretrained initialization from {source}: {problems}")
return model
```

Confirm with the installed framework that diagnostics cover missing persistent
buffers and valid tied serialization. Never blanket-filter embedding/head names:
if both aliases are absent, the missing-weights test must fail. If a supported
framework release suppresses missing-buffer or alias diagnostics, add an artifact
key-inventory check confined to this loader: read safetensors headers (and actual
shard headers for an indexed export), compare to the loaded model's expected
state keys, and allow an omitted tied alias only when its storage-equivalent key
exists. For PyTorch weight files, use the framework's safe weights-only loading
path. Do not silently initialize missing state or use `ignore_mismatched_sizes`.
Catch framework shape-load errors only to add source context and preserve causes.

- [ ] **Step 6: Exercise remote-code exports and round trips.**

Extend `test_push_to_hub.py`'s existing offline subprocess test: save a looped
model, assert `looping.py` is copied, load without importing `oplm`, and compare
saved expected logits plus the resolved order in the child process. This test
uses `trust_remote_code=True` on a local directory and does not upload anything.

Run: `pytest tests/training/test_initialization.py tests/model/test_save_load.py tests/model/test_auto_classes.py tests/model/test_push_to_hub.py -q`.
Expected: pass, with genuine tied-weight preservation and old-checkpoint coverage.

- [ ] **Step 7: Commit the loading deliverable.**

Commit Task 3 files as `feat(training): load pretrained weights with strict compatibility checks`.

### Task 4: Initialize a new training stage without restoring training state

**Files:**

- Modify: `src/oplm/config.py`, `src/oplm/configs/train/base.yaml`, `src/oplm/training/trainer.py`.
- Create: `tests/training/test_e2e_looping.py`.
- Modify: `tests/training/conftest.py`, `tests/training/test_config.py`, `docs/TRAIN.md`, `docs/CONFIG.md`.

**Interfaces:**

- Consumes: Task 3 source resolver and loader; existing `_resolve_resume_target`.
- Produces: `TrainConfig.init_from: str | None = None` and resume-first bootstrap.
- Extend `tiny_train_cfg` with `init_from`, `num_loops`, `loop_strategy`,
  `loop_start`, `loop_end`, and `value_residual`, passing them to their owning configs.

- [ ] **Step 1: Add a real weights-only stage transition test.**

Use `training_parquet` and the existing CPU/Accelerate-reset harness:

```python
from pathlib import Path

import pytest
import torch

from oplm.model import OplmForMaskedLM
from oplm.training.trainer import Trainer
from tests.training.conftest import configure_accelerator_device, tiny_train_cfg

pytestmark = pytest.mark.slow


def test_new_stage_starts_from_weights_only(
    tmp_path: Path, training_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    parent_cfg = tiny_train_cfg(tmp_path / "parent", training_parquet, max_steps=2)
    Trainer(parent_cfg).train()
    source = tmp_path / "parent" / "checkpoint-2" / "hf"
    expected = OplmForMaskedLM.from_pretrained(source).state_dict()
    cfg = tiny_train_cfg(
        tmp_path / "child", training_parquet, max_steps=3,
        init_from=str(source), num_loops=2, loop_strategy="stack",
        lr=2e-4, warmup_steps=1,
    )
    trainer = Trainer(cfg)
    actual = trainer.accelerator.unwrap_model(trainer.model).state_dict()
    assert trainer.global_step == 0
    assert trainer.tokens_seen == 0
    for key, value in expected.items():
        torch.testing.assert_close(actual[key].cpu(), value.cpu(), rtol=0, atol=0)
    for optimizer in trainer.optimizers:
        assert not optimizer.state_dict()["state"]
    trainer.train()
    assert trainer.global_step == 3
    assert (tmp_path / "child" / "checkpoint-3" / "hf" / "config.json").is_file()
```

Extend with callbacks asserting the first LR follows the new scheduler, new-stage
counters begin at zero, and the data cursor is fresh. Parameterize stage-one and
stage-two optimizer choices across AdamW and Muon: no parent optimizer type or
momentum state should constrain weights-only loading.

- [ ] **Step 2: Add resume precedence and source-path cases.**

Run a child stage through step 3 with a checkpoint at step 2. Delete/rename the
parent export; build a new trainer with the original `init_from` string and
explicit `resume_from` pointing to child checkpoint 2. Assert step/counters,
optimizer state, and scheduler position restore before training, then compare
the next step with the uninterrupted child control using existing data-exact
test conventions. Repeat with `auto_resume=True` selecting a child checkpoint.

Patch `resolve_initialization_source` to raise if called during these resumed
constructions; this pins the no-access contract. Also cover an empty auto-resume
directory loading the parent, scratch startup when both sources are unset, and
an invalid explicit resume raising even when `init_from` is valid. Use a mocked
tracker to assert the new stage does not reuse the parent's W&B ID.

- [ ] **Step 3: Run the new tests and verify the missing-config failure.**

Run: `pytest tests/training/test_e2e_looping.py tests/training/test_config.py -q`.
Expected: the new helper/config arguments are absent until implementation.

- [ ] **Step 4: Wire initialization into the existing bootstrap sequence.**

Add the dataclass field and documented YAML default. After the existing agreed
resume target has been resolved, and before tracker/config writes, resolve a
local initialization path only for a fresh stage:

```python
initialization_source = None
if resume_target is None and cfg.train.init_from is not None:
    initialization_source = resolve_initialization_source(cfg.train.init_from)
```

At the existing model-construction point, preserve the saved gradient-checkpoint
flag and choose exactly one model path:

```python
gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
if initialization_source is None:
    model = OplmForMaskedLM(cfg.model)
else:
    model = load_initial_model(initialization_source, cfg.model)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()
```

Restore training mode explicitly for an initialized model: HF `from_pretrained`
returns an evaluation-mode model. Confirm the ordinary Trainer training loop
also invokes `.train()` before stepping. Preserve the configured full/selective
checkpoint mode when enabling checkpointing.

Then retain the current order: shard for HSDP, build optimizers/schedulers,
Accelerate preparation, compile, and finally existing full-state resume if a
target exists. Preserve the new stage's `set_seed` behavior; never restore parent
RNG, scaler, counters, tracker ID, or data cursor. Source provenance is already
carried by serialized `train.init_from`; log the resolved path on fresh startup.

Document distinct stage output directories. Before writing config/tracker files,
reject a new-stage output directory that is the initialization export itself or
the parent run directory when a standard `checkpoint-N/hf` source identifies it;
compare resolved paths to catch symlinks. Do not inspect `init_from` to enforce
this during a valid resume. Add an alias-path regression test to avoid overwriting
the parent's config while preparing the child stage.

- [ ] **Step 5: Run the stage lifecycle and existing bootstrap tests.**

Run: `pytest tests/training/test_e2e_looping.py tests/training/test_config.py tests/training/test_trainer.py tests/training/test_e2e_checkpoint.py tests/training/test_e2e_data_exact.py -q`.
Expected: fresh-stage and same-stage recovery contracts pass. These are tiny runs
on real fixture sequences, not production adaptation experiments.

- [ ] **Step 6: Document and commit the stage workflow.**

Include this command pair in `docs/TRAIN.md`, explaining that `parent.yaml`
contains the desired data/model/training configuration and that overrides retain
its physical architecture:

```bash
oplm train --config parent.yaml train.output_dir=outputs/parent train.max_steps=1000000
oplm train --config parent.yaml train.init_from=outputs/parent/checkpoint-1000000/hf train.output_dir=outputs/looped train.max_steps=500000 model.num_loops=2 model.loop_strategy=stack
```

Document stage-local LR/warmup/max-steps, a fresh data stream, local-only sources,
and resume precedence. Commit Task 4 files as
`feat(training): initialize separate stages from pretrained weights`.

### Task 5: Reject loop drift before restoring a training checkpoint

**Files:**

- Modify: `src/oplm/training/checkpoint.py`.
- Create: `tests/training/test_loop_resume.py`.
- Modify: `tests/training/test_e2e_looping.py`, `docs/TRAIN.md`.

**Interfaces:**

- Consumes: Task 1 resolver and saved `config.yaml` or `hf/config.json`.
- Produces: `validate_loop_resume_compat(checkpoint_dir: Path, cfg: OplmConfig) -> None`
  in `checkpoint.py`; `cfg` is the run config, matching schedule validation.
- Invoke it from both `validate_checkpoint_for_resume` and `load_checkpoint`.

- [ ] **Step 1: Add metadata-level compatibility tests.**

```python
from pathlib import Path

import pytest

from oplm.config import OplmConfig, serialize_config
from oplm.model import OplmConfig as OplmModelConfig
from oplm.training.checkpoint import validate_loop_resume_compat


@pytest.mark.parametrize(
    ("field", "changed"),
    [("num_loops", 3), ("loop_strategy", "interleave"),
     ("loop_start", 1), ("loop_end", 2)],
)
def test_resume_rejects_loop_drift(tmp_path: Path, field: str, changed: object) -> None:
    saved = OplmConfig(model=OplmModelConfig(num_hidden_layers=4, num_loops=2))
    (tmp_path / "config.yaml").write_text(serialize_config(saved))
    live = OplmConfig(model=OplmModelConfig(num_hidden_layers=4, num_loops=2))
    setattr(live.model, field, changed)
    with pytest.raises(ValueError, match=field):
        validate_loop_resume_compat(tmp_path, live)
```

Add null-versus-L end equivalence; saved loop fields absent versus explicit
ordinary defaults; a looped live model against legacy absent fields; strategy
drift even when R=1; and a one-layer region where both strategies have the same
execution order. The last two still fail because the saved settings changed.

- [ ] **Step 2: Verify checks happen before state mutation.**

Take a real child checkpoint from Task 4's harness, change a loop setting in the
live config, patch `torch.distributed.checkpoint.load` to fail if reached, and
assert compatibility rejects first. Test auto-resume when all committed candidates
have mismatched loops: it must raise and must not fall back to weights-only init.
Retain existing fallback behavior for genuinely corrupt checkpoints.

Run: `pytest tests/training/test_loop_resume.py -q`.
Expected: missing validator before implementation.

- [ ] **Step 3: Implement semantic loop comparison in both resume entry points.**

Read the saved `model` mapping directly from OmegaConf `config.yaml`, avoiding a
merge with packaged production defaults. If absent, use `hf/config.json`. Validate
each saved/live model through the resolver and compare this signature:

```python
def _loop_signature(model_config: OplmModelConfig) -> dict[str, object]:
    resolve_layer_execution_order(
        model_config.num_hidden_layers,
        num_loops=model_config.num_loops,
        loop_strategy=model_config.loop_strategy,
        loop_start=model_config.loop_start,
        loop_end=model_config.loop_end,
    )
    return {
        "num_hidden_layers": model_config.num_hidden_layers,
        "num_loops": model_config.num_loops,
        "loop_strategy": model_config.loop_strategy,
        "loop_start": model_config.loop_start,
        "loop_end": (
            model_config.num_hidden_layers
            if model_config.loop_end is None else model_config.loop_end
        ),
    }
```

Load missing loop fields with constructor defaults. If neither config artifact
exists, retain legacy warning/compatibility only for an ordinary default target;
reject nondefault loop settings because saved behavior cannot be verified. This
does not weaken existing schedule checks or allow malformed present metadata.

Compare named signature fields, list saved/live values in the exception, and
explain that intentional execution changes use `train.init_from` in a new stage.
Call the validator during main-rank candidate checks before the existing broadcast,
and before `dcp.load` on explicit restoration. Keep existing error propagation and
collective ordering. Do not repurpose schedule validation to validate model shapes.

- [ ] **Step 4: Run resume tests and commit.**

Run: `pytest tests/training/test_loop_resume.py tests/training/test_e2e_looping.py tests/training/test_checkpoint.py tests/training/test_e2e_dcp.py -q`.
Expected: pass, including legacy defaults and null/end equivalence. Commit as
`fix(training): validate loop semantics before checkpoint resume`.

### Task 6: Report effective computation and preserve parameter accounting

**Files:**

- Modify: `src/oplm/training/flops.py`, `src/oplm/training/trainer.py`.
- Modify: `tests/training/test_flops.py`, `tests/training/test_trainer.py`, `docs/TRAIN.md`, `docs/MODEL_ARCHITECTURE.md`.

**Interfaces:**

- Consumes: Task 1 resolver and physical config.
- Preserves: `estimate_flops_per_token(config: OplmModelConfig) -> int`.
- Produces: logged physical/effective depth and unique parameter count;
  loop fields and initialization source already appear in resolved config.

- [ ] **Step 1: Add an independent arithmetic test for partial-loop FLOPs.**

```python
def test_partial_loop_counts_repeated_blocks_and_one_head() -> None:
    cfg = _config(num_hidden_layers=6, intermediate_size=1024)
    baseline = estimate_flops_per_token(cfg)
    cfg.num_loops = 3
    cfg.loop_start = 1
    cfg.loop_end = 4
    # Three selected blocks, each with two extra executions.
    per_block_forward = 2 * 256 * (4 * 256) + 3 * 2 * 256 * 1024
    assert estimate_flops_per_token(cfg) - baseline == 3 * 6 * per_block_forward
```

Parametrize strategies and verify identical cost, exact one-loop regression, and
head work counted once. Extend trainer log-capture tests to require physical depth,
effective depth, unique parameter count, and resolved range/strategy at startup.

- [ ] **Step 2: Confirm the FLOP regression fails.**

Run: `pytest tests/training/test_flops.py -q`.
Expected: the partial-loop difference is zero before implementation.

- [ ] **Step 3: Compute effective depth using the shared resolver.**

Replace only the backbone multiplier:

```python
effective_depth = len(resolve_layer_execution_order(
    config.num_hidden_layers,
    num_loops=config.num_loops,
    loop_strategy=config.loop_strategy,
    loop_start=config.loop_start,
    loop_end=config.loop_end,
))
backbone_flops = effective_depth * per_layer
```

Keep the current estimator's other formulas and documented omissions. Before
sharding, derive unique parameter count with `sum(p.numel() for p in model.parameters())`
so DTensor local shards cannot undercount it. Emit it through the existing main-rank
logger with `num_hidden_layers`, `len(model.oplm.backbone.layer_execution_order)`,
and the resolved range/strategy. Do not count an aliased parameter twice or multiply
parameter count by R. Existing W&B config already includes the four loop fields;
derived metadata may be logged at startup without adding new model config fields.

- [ ] **Step 4: Validate reporting and commit.**

Run: `pytest tests/training/test_flops.py tests/training/test_trainer.py -q`.
Document equal-token versus equal-compute comparisons, activation memory growth,
and the unchanged unique-parameter budget. Commit as
`feat(training): account for effective loop depth in compute metrics`.

### Task 7: Validate compiled/distributed training and finish documentation

**Files:**

- Create: `tests/training/test_e2e_looping_distributed.py`, `tests/training/_looping_worker.py`.
- Modify: `tests/training/test_e2e_compile.py`, `tests/training/test_e2e_gradckpt.py`, `docs/TESTING_E2E.md`.
- Modify, only if these tests demonstrate an issue: `src/oplm/training/parallel.py`, `src/oplm/training/trainer.py`, `src/oplm/model/transformer.py`.

**Interfaces:**

- Consumes: finished model/initialization/resume paths and `tiny_train_cfg`.
- Produces: real subprocess validation artifacts under pytest temporary paths;
  no production CLI flags or framework abstractions.
- New worker entry: `main(config_path: str, result_dir: str) -> None`; read an
  ordinary resolved YAML with existing `load_config`, run the actual Trainer,
  and write per-rank JSON metrics/step metadata after its checkpoint commits.

- [ ] **Step 1: Extend compiled and activation-checkpoint parity coverage.**

Use the existing AOT-eager test scaffold, add R=2 and both strategies, and call
backward in every case. Compare a cloned eager model to a compiled model with
the same initialized tensors and batch. Cover checkpointing off/full/selective
with learnable value residuals. Use this assertion pattern after both backwards:

```python
torch.testing.assert_close(compiled_out.loss, eager_out.loss, rtol=1e-4, atol=1e-5)
for name, reference_param in eager.named_parameters():
    candidate_param = compiled._orig_mod.get_parameter(name)
    assert (candidate_param.grad is None) == (reference_param.grad is None)
    if reference_param.grad is not None:
        torch.testing.assert_close(
            candidate_param.grad, reference_param.grad, rtol=1e-4, atol=1e-5
        )
```

Retain `reset_dynamo` and `restore_optimize_ddp` fixtures to prevent global-setting
leaks. Extend GPU Inductor real-Trainer coverage using the existing skip markers;
AOT-eager CPU checks are not a substitute for production compiler verification.

- [ ] **Step 2: Add two-rank eager gradient/update parity checks.**

Create a bounded `torch.distributed.run --standalone --nproc_per_node=2` subprocess
test following `_hsdp_worker.py` and `test_e2e_hsdp.py` environment/timeout patterns.
Use CPU/gloo first; the existing project already supports native FSDP2 on CPU.
Test DDP and HSDP, stack/full-range and interleave/subset, learnable value residuals,
and full/selective activation checkpointing. Do not disable existing HSDP guards.

For a parity case, each rank receives the same small real sequence batch and
same initial state, and runs a real MLM loss with dropout off. Compare distributed
gradients/one optimizer update against an unwrapped single-process reference on
that batch. Identical per-rank batches make the averaged distributed gradient
equal to the reference without reduction-denominator ambiguity. For FSDP2,
gather reference comparisons collectively on all ranks before teardown.

Also run one gradient-accumulation case with two microbatches and verify every
optimizer parameter appears once. Nonzero-dropout parity is already covered in
Task 2; do not compare unrelated rank RNG streams bit-for-bit.

- [ ] **Step 3: Exercise weights-only initialization and real distributed resume.**

Save a tiny ordinary parent export. Launch the worker with a looped child YAML
and `init_from`; record initial step zero, finite per-step loss, and a committed
child checkpoint. Remove the parent export, launch again with child auto-resume,
and require restoration of the child step and progression to the new target.
Repeat DDP and HSDP; check the HF export's tensors/config against the DCP-restored
model and verify a sharded child checkpoint loads into a single-process model.

Use this real-Trainer worker body, with the existing distributed-launch environment
setup kept in the parent pytest process:

```python
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main(config_path: str, result_dir: str) -> None:
    from oplm.config import load_config
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback

    cfg = load_config(["--config", config_path])
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, callbacks=[callback])
    raw_model = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    rank = trainer.accelerator.process_index
    payload = {
        "resumed_from_step": trainer.global_step,
        "num_unique_layers": len(raw_model.oplm.backbone.layers),
        "effective_depth": len(raw_model.oplm.backbone.layer_execution_order),
    }
    trainer.train()
    payload["global_step"] = trainer.global_step
    # Trainer callbacks are main-rank-only. Other ranks confirm state/counters;
    # the parent asserts that rank zero recorded every expected loss.
    losses = [metrics["train/loss"] for _, metrics in callback.train_logs]
    payload["loss_count"] = len(losses)
    payload["all_losses_finite"] = all(math.isfinite(loss) for loss in losses)
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(payload))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

The parent test must assert per-rank results, not only subprocess success:

```python
for result in results:
    assert result["resumed_from_step"] == expected_start
    assert result["global_step"] == expected_end
    assert result["num_unique_layers"] == 3
    assert result["effective_depth"] == expected_depth
    assert result["all_losses_finite"]
assert results[0]["loss_count"] == expected_end - expected_start
assert (run_dir / f"checkpoint-{expected_end}" / "hf" / "config.json").is_file()
```

Set expected depth from the explicit test case, not the production resolver.
Use fixed subprocess timeouts and attach stdout/stderr to failures. Match existing
cross-world-size cursor opt-outs when testing 2→1 resume; do not weaken cursor
guards globally. Add a two-rank mismatch case requiring all ranks to exit with
the named loop-compatibility error before checkpoint loading.

- [ ] **Step 4: Run the targeted integration matrix and diagnose failures.**

Run: `pytest tests/training/test_e2e_looping_distributed.py tests/training/test_e2e_looping.py tests/training/test_e2e_compile.py tests/training/test_e2e_gradckpt.py -q`.
Record the expected red test for each discovered integration defect before fixing
it. If tests pass with existing wrappers, no production change is needed. Never
silently drop repeated-block gradients, detach recurrence, or disable checkpointing
to make the tests pass.

On suitable GPU hardware, run DDP/HSDP with Inductor, bf16, both loop orders, and
full/selective checkpointing in the same bounded worker harness. Keep this a
small smoke/parity matrix rather than a full performance benchmark. Report
unavailable hardware rows explicitly; do not equate skipped tests with validation.

- [ ] **Step 5: Run final regression checks once and record evidence.**

```bash
pytest tests/model tests/training -m "not slow" -q
pytest tests/training/test_e2e_looping.py tests/training/test_e2e_looping_distributed.py tests/training/test_e2e_compile.py tests/training/test_e2e_gradckpt.py tests/training/test_e2e_checkpoint.py tests/training/test_e2e_dcp.py tests/model/test_push_to_hub.py -q
pytest -m "not slow"
ruff check src/
ruff format --check src/
ty check src/
git diff --check
```

Avoid redundant repeat runs: the repository-wide non-slow run subsumes the first
command if it has already been chosen as the initial regression run. Run the full
`pytest` suite when the required fixtures/hardware are available; list any skips
or environment failures separately. Fix new issues and rerun only the affected
checks plus required final gates. Do not change dependency pins to bypass errors.

- [ ] **Step 6: Finish user documentation and commit.**

Update `docs/TESTING_E2E.md` with the new stage-transition, loop-resume, and
distributed parity coverage. Check that `docs/CONFIG.md` describes all five new
fields, architecture docs explain D versus L, and training docs show local HF
loading, separate stages, fresh data/optimizer/schedule, and source-free resume.
Keep the staged user workflow free of implementation-only details.

Commit as `test: validate looped training across checkpoint and parallel modes`.
The completion report must state the implemented behavior, exact executed checks,
unavailable hardware validation, and any measured limitations. Integration/merge
follows the user's chosen execution workflow; plan approval alone is not a push
or merge instruction for implementation code.

## Spec Coverage and Self-Review

| Spec requirement | Owning tasks |
|---|---|
| Ordinary/scratch/pretrained/full/subset/stack/interleave | 1–4 |
| Strict integer/range validation and edited configs | 1 |
| Unique modules, full backpropagation, physical-depth scaling | 2 |
| First block-zero values, Canon physical indexing | 2 |
| Execution-indexed hidden/attention outputs and all public task classes | 2–3 |
| Old/new HF saves, ties, tokenizer, remote-code dependencies | 1, 3 |
| Local strict source loading, semantic config checks | 3 |
| Fresh stage optimizer/LR/RNG/data/counters/tracker, provenance | 4 |
| Resume precedence, removed source, loop drift rejection | 4–5 |
| Effective-depth FLOPs, unique parameter accounting | 6 |
| Activation checkpointing, compilation, DDP/HSDP, save/resume | 2, 7 |
| Documentation and required Ruff/ty/regression gates | 1–7 |

Self-review checks before handing this plan to an implementer: source paths exist
or are explicitly marked new; interfaces above match their later consumers;
all five Review Focus cases have an owning test; no task changes the approved
scientific scope. Hardware validation remains an execution requirement, not a
claim made by this planning document.
