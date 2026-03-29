# OPLM Technical Analysis

Date: 2026-03-29

Scope: full repository review with emphasis on test coverage and realism, CLI/API consistency, configuration/override documentation, and maintainability for a research-heavy workflow.

## Executive Summary

The codebase is in better shape than many early research repos. It is relatively cohesive, mostly modular, and does not look weighed down by obvious legacy code. The model, data, training, and evaluation layers are separated cleanly enough that future ablations are plausible without a rewrite.

The highest-value problems are not architectural bloat. They are interface-contract problems:

- The public/user-facing paths are less reliable than the internal training path.
- The test suite is strong at unit coverage, but weaker at validating the actual user workflows.
- Documentation has drifted enough that some examples are now wrong.
- A few subtle bugs have landed exactly because the public workflow contract is not enforced in one place.

The most important concrete findings are:

1. `oplm encode` is effectively broken in normal use.
2. Sequence evaluation is not actually deterministic across repeated eval runs on the same task instance.
3. The manual attention path handles boolean masks incorrectly.
4. The configuration surface is powerful but under-documented and partially misleading.
5. The non-slow suite is more expensive than it needs to be for CI because several tests run full evaluation over a 100k-row fixture on every Python version.

## 1. Test Harness Analysis

### What is already good

- Unit coverage is broad. The repo has focused tests for tokenizer, dataset/loader/collator, individual model components, transformer assembly, config parsing, optimizer/scheduler/checkpoint logic, evaluator orchestration, and metric functions.
- There are real-data integration tests, not just synthetic tensor tests. Training E2E uses `tests/fixtures/training/test_sequences.parquet`, and structure eval uses real PDB fixtures.
- The training loop is exercised on CPU with pilot-scale configs, which is the right direction for CI-compatible research tests.
- Eval scheduling is covered, including training-with-eval integration in [tests/test_e2e.py](/home/briney/git/oplm/tests/test_e2e.py).

### Where the current suite falls short

#### 1.1 Public workflow coverage is missing

There is no CLI-focused test module in `tests/`, and none of the existing tests black-box the actual user commands. The suite exercises `Trainer`, `Evaluator`, and model classes directly, but not `python -m oplm train`, `python -m oplm encode`, or `python -m oplm info`.

This matters because the most serious user-facing issues in this repo are CLI/API mismatches that the current tests never touch.

#### 1.2 "E2E" tests are partly integration tests of internals, not true user workflows

Several tests observe success by monkeypatching private or semi-private internals:

- [tests/test_e2e.py:71](/home/briney/git/oplm/tests/test_e2e.py#L71) overrides `trainer._log_step`
- [tests/test_training.py:386](/home/briney/git/oplm/tests/test_training.py#L386) does the same
- [tests/test_e2e.py:282](/home/briney/git/oplm/tests/test_e2e.py#L282) and nearby patch `accelerator.log`

These are useful integration tests, but they are not a substitute for black-box tests against the public entrypoints.

#### 1.3 The suite missed a real eval correctness bug

`DeterministicMLMCollator` is stateful and increments `_batch_idx` on every call in [src/oplm/eval/data/sequence_loader.py:48](/home/briney/git/oplm/src/oplm/eval/data/sequence_loader.py#L48). `SequenceEvalTask` caches and reuses the same DataLoader in [src/oplm/eval/tasks/sequence.py:58](/home/briney/git/oplm/src/oplm/eval/tasks/sequence.py#L58), but never resets that collator.

Result: repeated evaluations on the same task instance use different mask patterns, despite the docstring claiming "the same positions are masked on every eval run" in [src/oplm/eval/data/sequence_loader.py:78](/home/briney/git/oplm/src/oplm/eval/data/sequence_loader.py#L78).

I reproduced this directly with the real fixture and a fixed model:

```text
first eval:  {'loss': 3.546579..., 'accuracy': 0.026906..., 'perplexity': 34.694429...}
second eval: {'loss': 3.546185..., 'accuracy': 0.027108..., 'perplexity': 34.680784...}
```

The current tests validate:

- reset works on the collator itself in [tests/eval/test_sequence_loader.py:65](/home/briney/git/oplm/tests/eval/test_sequence_loader.py#L65)
- two freshly built dataloaders match in [tests/eval/test_sequence_loader.py:143](/home/briney/git/oplm/tests/eval/test_sequence_loader.py#L143)
- the cached dataloader is reused in [tests/eval/test_sequence_task.py:164](/home/briney/git/oplm/tests/eval/test_sequence_task.py#L164)

But there is no regression test for repeated `SequenceEvalTask.evaluate()` calls returning stable metrics.

#### 1.4 Structure-task integration does not exercise the intended logreg path

`StructureEvalTask` defaults to `logreg_n_train=20` in [src/oplm/eval/tasks/structure.py:63](/home/briney/git/oplm/src/oplm/eval/tasks/structure.py#L63). The test fixture directory contains only three structures. That means the structure integration tests run the fallback path, not the main logistic-regression evaluation protocol.

This is acceptable for a small CI smoke test, but it means the advertised "real" structure evaluation is only unit-tested in pieces, not end-to-end.

#### 1.5 CI cost is higher than necessary

The training fixture parquet contains 100,000 rows. Several non-slow tests evaluate across the full dataset rather than a tiny eval fixture. Combined with CI running `pytest --cov=oplm` across Python 3.11, 3.12, and 3.13 in [ci.yaml:10](/home/briney/git/oplm/.github/workflows/ci.yaml#L10), this makes the suite heavier than it needs to be for default PR validation.

That does not mean the tests are bad. It means the fixture strategy should be tiered:

- a tiny default fixture for fast CI
- the current larger fixture for nightly or slow-marker coverage

#### 1.6 Important gaps remain untested

The most notable missing cases are:

- CLI command behavior
- checkpoint resume via `train.resume_from`
- a CPU-only distributed smoke test using `accelerate` with 2 processes
- model loading from training checkpoints for inference
- repeated-eval determinism
- bool-mask parity between the SDPA path and the `need_weights=True` manual path

### Test harness recommendation

The repo already has enough structure to support a very good research-grade test strategy. The next step should be to reshape the suite around explicit layers:

- fast unit tests
- fast workflow-smoke tests
- slower realism tests

Specifically:

1. Add `tests/test_cli.py` covering `train`, `info`, and `encode` with Typer/CliRunner or subprocess-level smoke tests.
2. Add a tiny eval parquet fixture specifically for sequence-eval tests.
3. Add one repeated-eval determinism regression test.
4. Add one `resume_from` checkpoint smoke test.
5. Add one 2-process CPU distributed smoke test behind `@pytest.mark.slow`.
6. Keep the current larger fixture, but move full-dataset eval coverage behind `slow`.

## 2. CLI / UX / API Analysis

### 2.1 The CLI is not consistently built on a stable public API

`train` delegates to `oplm.train.main()` in [src/oplm/cli.py:56](/home/briney/git/oplm/src/oplm/cli.py#L56), which is fine.

`info` and `encode` do not call a shared library API. They duplicate model-building and model-loading behavior inside the CLI module:

- model construction and parameter counting in [src/oplm/cli.py:111](/home/briney/git/oplm/src/oplm/cli.py#L111)
- inference model loading in [src/oplm/cli.py:82](/home/briney/git/oplm/src/oplm/cli.py#L82)

That duplication is exactly why the CLI and "actual supported path" have drifted apart.

### 2.2 `oplm encode` has two independent correctness problems

#### Problem A: checkpoint format mismatch

Training checkpoints are saved with `accelerator.save_state()` into checkpoint directories in [src/oplm/training/checkpoint.py:45](/home/briney/git/oplm/src/oplm/training/checkpoint.py#L45). In practice that produces files like `model.safetensors`, `optimizer.bin`, `scheduler.bin`, and `trainer_state.json`.

`oplm encode`, however, expects a plain state dict file and does:

```python
checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint)
```

from [src/oplm/cli.py:83](/home/briney/git/oplm/src/oplm/cli.py#L83).

That is incompatible with the checkpoint format emitted by training. I verified that `torch.load(..., weights_only=True)` on the training-produced `model.safetensors` fails.

#### Problem B: mask-shape mismatch

`encode` passes a raw 2D integer attention mask into the encoder in [src/oplm/cli.py:93](/home/briney/git/oplm/src/oplm/cli.py#L93):

```python
hidden = model.encoder(batch["input_ids"], attention_mask=batch["attention_mask"])[0]
```

But the training path explicitly expands masks to 4D boolean format in [src/oplm/training/trainer.py:163](/home/briney/git/oplm/src/oplm/training/trainer.py#L163), and the attention module documents the mask as 4D in [src/oplm/model/attention.py:221](/home/briney/git/oplm/src/oplm/model/attention.py#L221).

Direct reproduction with the current encoder API raises:

```text
RuntimeError: Expected attn_mask dtype to be bool or float ... but got long int
```

So `encode` is not just untested; it is currently broken unless its caller manually normalizes mask shape/dtype first, which defeats the point of a CLI.

### 2.3 README examples mirror the same broken API usage

The embedding and MLM examples in the README pass `batch["attention_mask"]` directly into the model in:

- [README.md:128](/home/briney/git/oplm/README.md#L128)
- [README.md:141](/home/briney/git/oplm/README.md#L141)

Those examples do not match the actually working training/eval path.

This is the clearest signal that the public model API contract is underspecified.

### 2.4 Override UX is inconsistent

`train` and `info` use positional dotlist overrides via `OverridesArg` in [src/oplm/cli.py:23](/home/briney/git/oplm/src/oplm/cli.py#L23), while `encode` uses repeated `--override` options in [src/oplm/cli.py:70](/home/briney/git/oplm/src/oplm/cli.py#L70).

That inconsistency is small, but it is exactly the sort of thing that slows down research use when users expect one override mental model everywhere.

### 2.5 The training progress UI misses eval loss

`Evaluator` returns dataset-prefixed keys like `eval/<dataset>/loss` in [src/oplm/eval/evaluator.py:72](/home/briney/git/oplm/src/oplm/eval/evaluator.py#L72), but the trainer only checks for `"eval/loss"` in [src/oplm/training/trainer.py:205](/home/briney/git/oplm/src/oplm/training/trainer.py#L205).

Result: the progress bar's eval slot stays `N/A` even after eval runs. This is not a major architectural issue, but it is a visible UX paper cut.

### CLI/API recommendation

The repo should define one public interface contract and make every caller use it.

Recommended shape:

1. Public model API accepts `attention_mask` as standard `(B, T)` bool/int or `None`.
2. Mask normalization to 4D happens inside the model or a shared helper, never at call sites.
3. Add a shared inference loader that can read:
   - a training checkpoint directory
   - a plain state dict file
   - its associated config automatically when available
4. Make `train`, `info`, and `encode` thin wrappers over a shared library API.
5. Standardize overrides across all subcommands.

If this is done, the CLI and API stop being two loosely related implementations.

## 3. Configuration Surface and Documentation

### 3.1 The config system is powerful, but not self-documenting

The dataclass + OmegaConf setup in [src/oplm/config.py](/home/briney/git/oplm/src/oplm/config.py) is a solid foundation. It supports defaults, presets, YAML, and dotlist overrides cleanly.

The problem is discoverability:

- there is no authoritative reference for all config fields
- task-specific eval keys are freeform
- some fields are derived, some ignored, and some are effectively dead

For a research repo where ablations are central, this is the main documentation debt.

### 3.2 The repo does not yet have the config reference the project needs

There is no top-level `configs/` directory today. Only package defaults exist under `src/oplm/configs/`.

Given the way this repo is meant to be used, the most sensible addition is exactly what you suggested:

- create a repo-root `configs/README.md`
- link it from the main README
- make it the canonical reference for every config section and override path

### 3.3 Current docs have drifted enough to be misleading

#### Drift A: README eval example uses the wrong structure

The README documents:

```yaml
structures:
  path: ...
  type: structure
  eval_every: 10_000
  extra:
    contact_threshold: 8.0
```

in [README.md:271](/home/briney/git/oplm/README.md#L271).

But `parse_eval_configs()` does not look for an `extra:` sub-dict. It collects unknown keys directly from the dataset entry in [src/oplm/config.py:438](/home/briney/git/oplm/src/oplm/config.py#L438). That means the documented example silently produces `extra={"extra": {...}}`, which `StructureEvalTask` does not expect.

The correct documented shape should place task-specific keys directly at the dataset-entry top level.

#### Drift B: README preset sizing is no longer authoritative

The README still advertises the `small` preset as `~25M` in [README.md:206](/home/briney/git/oplm/README.md#L206). Running `python -m oplm info --preset small` in this workspace reports `4.8M`.

This kind of drift is especially costly in a research repo because users stop trusting the docs once one or two numbers are wrong.

#### Drift C: `docs/ARCHITECTURE.md` documents an older design

[docs/ARCHITECTURE.md:83](/home/briney/git/oplm/docs/ARCHITECTURE.md#L83) says a project-root `configs/` directory "will hold" experiment configs, and [docs/ARCHITECTURE.md:149](/home/briney/git/oplm/docs/ARCHITECTURE.md#L149) still describes stub `TrainConfig` and `DataConfig` classes that no longer match the real code.

This file is now partly historical/planning documentation, not a reliable source of truth.

### 3.4 Some config fields are misleading today

Two notable examples:

- `TrainConfig.config_path` exists in [src/oplm/config.py:172](/home/briney/git/oplm/src/oplm/config.py#L172), but `load_config()` never writes it back into the resolved config in [src/oplm/config.py:459](/home/briney/git/oplm/src/oplm/config.py#L459).
- `ModelConfig.dtype` exists in [src/oplm/config.py:75](/home/briney/git/oplm/src/oplm/config.py#L75), but is not consumed anywhere in the runtime path.

For an ablation-heavy codebase, dead or half-dead config knobs are especially risky because they create false confidence that a run was configured in a certain way.

### Config documentation recommendation

Create `configs/README.md` with these sections:

1. Merge order and override rules.
2. Full field tables for `model`, `train`, and `data`.
3. For each field:
   - override path
   - type
   - default
   - valid values
   - whether it is derived, required, or currently unused
4. Eval-task-specific sections:
   - `sequence`
   - `structure`
   - `proteingym`
   - `tape`
   - `proteinglue`
   - `everest`
5. Copy-paste examples:
   - minimal CPU smoke train
   - train with eval
   - multi-dataset train mix
   - checkpoint resume
   - inference / embedding extraction

This file should be generated or at least mechanically checked against the dataclasses to minimize drift.

## 4. Design and Maintainability Findings

### 4.1 Overall design quality is decent

The strongest maintainability positives are:

- model components are cleanly decomposed
- feature toggles are mostly isolated behind config
- eval tasks use a registry rather than a monolithic switch
- the training loop is simple enough to reason about
- there is no obvious large block of dead legacy code

For a new research repo, this is a strong starting point.

### 4.2 Manual attention mask handling is inconsistent with the real runtime path

The public training/eval path uses boolean keep-masks in [src/oplm/training/trainer.py:163](/home/briney/git/oplm/src/oplm/training/trainer.py#L163). The `need_weights=True` manual attention path claims to accept an "additive" mask and literally adds `attention_mask` to the logits in [src/oplm/model/attention.py:236](/home/briney/git/oplm/src/oplm/model/attention.py#L236).

That is fine for float `0/-inf` masks, but incorrect for boolean masks.

I reproduced a large discrepancy between the SDPA path and the manual path when using the boolean mask format the rest of the codebase actually uses:

- max absolute output difference: about `0.31`
- masked positions still received substantial attention mass

This is a real correctness issue, not just a type-docstring mismatch.

### 4.3 Resume bookkeeping is incomplete

In `_resume_from_checkpoint()`, `_samples_seen` is recomputed from `_fractional_epoch()` in [src/oplm/training/trainer.py:320](/home/briney/git/oplm/src/oplm/training/trainer.py#L320), but `_fractional_epoch()` itself is derived from `_samples_seen` in [src/oplm/training/trainer.py:271](/home/briney/git/oplm/src/oplm/training/trainer.py#L271).

So on resume, `_samples_seen` effectively resets to zero rather than being reconstructed meaningfully.

This is not catastrophic, but it means epoch-fraction logging after resume is not trustworthy.

### 4.4 Flexible config shapes are good for research, but currently under-validated

`DataConfig.train` and `DataConfig.eval` are deliberately flexible, and task-specific eval keys are passed through as a freeform dict. That is reasonable for fast-moving research code, but without typed task configs or generated docs, it becomes easy to write YAML that parses but does not do what the user thinks.

The README `extra:` bug is the clearest example of that failure mode.

### 4.5 The trainer lacks an explicit callback/event surface

The fact that tests repeatedly monkeypatch `_log_step` and `accelerator.log` to observe progress is a sign that the trainer is missing a proper observation surface.

This is not urgent, but a small callback API would help:

- testing
- richer CLI output
- custom experiment logging
- future notebook integration

### 4.6 Benchmark eval support is still mostly scaffolding

The benchmark task classes for ProteinGym, TAPE, ProteinGlue, and EVEREST are registered and documented, but still intentionally raise `NotImplementedError`.

That is fine for a new repo, but it should be described as "eval framework + implemented sequence/structure tasks + benchmark stubs", not as broad benchmark support.

## 5. Prioritized Recommendations

### Priority 0: Fix public workflow correctness

1. Define one supported attention-mask contract for public callers and enforce it in the model.
2. Add a shared checkpoint loader that supports training-produced checkpoint directories.
3. Rewrite `oplm encode` to use that shared loader and shared mask normalization.
4. Add CLI smoke tests so these regressions cannot reappear.

### Priority 1: Make eval results stable and CI-friendly

1. Reset the deterministic eval collator between eval runs or rebuild the eval loader each call.
2. Add a regression test for repeated `SequenceEvalTask.evaluate()` equality.
3. Split the 100k-row fixture usage into fast and slow layers.
4. Add at least one CPU distributed smoke test.

### Priority 2: Make configuration discoverable and trustworthy

1. Add repo-root `configs/README.md`.
2. Link it from `README.md`.
3. Fix the README examples, especially structure-eval config and inference examples.
4. Remove, implement, or clearly mark unused fields like `dtype` and `config_path`.
5. Consider typed per-task eval config objects instead of raw `extra` dicts.

### Priority 3: Smooth the research UX

1. Standardize override syntax across all CLI commands.
2. Fix progress-bar eval reporting.
3. Add a small callback/event interface to the trainer.
4. Keep docs aligned with the code by treating `docs/ARCHITECTURE.md` as generated/validated or slimming it down.

## Bottom Line

OPLM is a promising early-stage research codebase with a solid modular core. The main risk is not internal architectural rigidity. The main risk is that the external contract is currently spread across trainer code, CLI code, README examples, and test assumptions, and those pieces have started to drift.

If the next round of work focuses on:

- one true public API for masks/checkpoints/configs
- black-box workflow tests
- a real config reference under `configs/`

then the repo should become much easier to trust, extend, and use for systematic ablation work.
