# OPLM Architecture

This document is intentionally a high-level map, not a second config/API reference.
The live sources of truth are:

- [`README.md`](../README.md) for user-facing workflows
- [`configs/README.md`](../configs/README.md) for the canonical config surface
- [`src/oplm/config.py`](../src/oplm/config.py) for the actual structured config schema
- [`src/oplm/cli.py`](../src/oplm/cli.py) and [`src/oplm/inference.py`](../src/oplm/inference.py) for public entrypoints
- [`src/oplm/training/trainer.py`](../src/oplm/training/trainer.py) for the training loop and callback hooks

## Design Goals

- Keep the encoder stack modular enough for ablations without a rewrite.
- Keep public workflows thin wrappers around shared library code.
- Keep long-form docs delegated to code-backed references so they do not drift.

## Module Map

| Path | Responsibility |
| --- | --- |
| [`src/oplm/config.py`](../src/oplm/config.py) | Structured OmegaConf config schema, preset loading, YAML merge, and override resolution. |
| [`src/oplm/cli.py`](../src/oplm/cli.py) | Typer CLI for `train`, `info`, and `encode`. |
| [`src/oplm/train.py`](../src/oplm/train.py) | Distributed-friendly entry point for `accelerate launch -m oplm.train`. |
| [`src/oplm/inference.py`](../src/oplm/inference.py) | Shared config/model loading for checkpoint-backed inference. |
| [`src/oplm/model/`](../src/oplm/model/) | Encoder, attention, FFN, residual, embedding, and masking internals. |
| [`src/oplm/data/`](../src/oplm/data/) | Tokenizer plus training/eval data loading. |
| [`src/oplm/eval/`](../src/oplm/eval/) | Eval registry, orchestrator, and implemented task types. |
| [`src/oplm/training/`](../src/oplm/training/) | Trainer, checkpointing, optimizer/scheduler construction, FLOP accounting, and callback hooks. |

## Runtime Flows

### Training

1. `oplm train` resolves config from `--config`, `--preset`, and repeated `--override` flags.
2. [`src/oplm/train.py`](../src/oplm/train.py) hands the resolved config to [`Trainer`](../src/oplm/training/trainer.py).
3. `Trainer` builds the accelerator, dataloader, model, optimizer, scheduler, and optional evaluator.
4. Each optimizer step logs training metrics, runs scheduled eval tasks, and saves checkpoints on cadence.
5. [`TrainerCallback`](../src/oplm/training/callbacks.py) exposes stable main-process hooks for log, eval, checkpoint, and lifecycle events.

### Evaluation

- [`src/oplm/eval/evaluator.py`](../src/oplm/eval/evaluator.py) parses `data.eval`, constructs task instances through the registry, and returns metrics as `eval/<dataset>/<metric>`.
- Sequence and structure evaluation are implemented today.
- ProteinGym, TAPE, ProteinGlue, and EVEREST remain registered stubs rather than full benchmark integrations.

### Inference

- `oplm encode` and Python callers both use [`src/oplm/inference.py`](../src/oplm/inference.py).
- Inference loading supports training-produced checkpoint directories and plain weight files.
- Public callers pass ordinary `(B, T)` attention masks; mask normalization happens in shared model code rather than CLI call sites.

## Public Entry Points

| Entry point | Purpose | Notes |
| --- | --- | --- |
| `oplm train` | Single-process or launcher-managed training via the Typer CLI. | Uses repeated `--override key=value` flags. |
| `oplm info` | Build the model on the meta device and report parameter counts plus enabled features. | Safe for large presets because it avoids real allocation. |
| `oplm encode` | Load inference weights and write encoder embeddings to disk. | Accepts checkpoint directories directly. |
| `accelerate launch -m oplm.train ...` | Distributed training entry point. | Keeps raw dotlist passthrough because `accelerate` forwards trailing args directly. |

## Extension Points

- Add model ablations in [`src/oplm/model/`](../src/oplm/model/) and wire them through [`ModelConfig`](../src/oplm/config.py).
- Add new eval tasks through [`src/oplm/eval/registry.py`](../src/oplm/eval/registry.py).
- Observe training without monkeypatching internals by subclassing [`TrainerCallback`](../src/oplm/training/callbacks.py).

## Documentation Policy

`docs/ARCHITECTURE.md` stays intentionally slim. Detailed field tables, CLI examples,
and workflow snippets belong in the README, config reference, or code docstrings so
tests can validate them mechanically.
