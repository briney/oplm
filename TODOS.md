# TODOS: Add `torch.compile` Support

**Branch:** `add-compilation`  
**Scope:** Self-contained. No FSDP2, no FP8, no Accelerate removal. The existing
Accelerate-based Trainer is left intact; compilation slots in as a new opt-in layer.

## Background

`torch.compile` is not called anywhere in the training stack. This costs meaningful
throughput, particularly on Blackwell where:
- SDPA dispatches to FlashAttention-4 more reliably under graph-mode compilation
- The fp32 upcasts in `OplmLayerNorm`, `OplmRMSNorm`, and `apply_rope` create
  cross-op fusion opportunities the compiler can exploit
- Fused kernel launch paths reduce overhead for the per-block QKV projections and
  FFN activations

**Key technical constraints:**
- `dynamic=True` is required — protein sequences are padded to the batch maximum,
  so `seq_len` varies per batch. Without dynamic shapes every unique `seq_len`
  triggers a recompile, which is catastrophic for throughput.
- `torch.compile` must be called **after** `gradient_checkpointing_enable()` and
  **before** `accelerator.prepare()`. After GC enable: so the compiled graph
  captures the recompute wrapper. Before prepare: so DDP wraps the
  `OptimizedModule` rather than the raw model (both are valid, but pre-compile
  is standard and produces a cleaner graph).
- `accelerator.unwrap_model(compiled_model)` returns a
  `torch._dynamo.OptimizedModule`, not the underlying `PreTrainedModel`. The HF
  export in `checkpoint.py` currently calls `.save_pretrained()` directly on the
  unwrapped model; this must be patched to peel off `OptimizedModule` via
  `._orig_mod` first.

**Forward compatibility note:** When the `native-fsdp2-and-fp8` branch is merged,
the compile call will move from its current position (before `accelerator.prepare`)
to after `fully_shard()`. The `compile` and `compile_mode` config fields introduced
here carry forward unchanged; only the insertion point in `Trainer.__init__` changes.

---

## Phase 1: Config Schema

### 1.1 — `src/oplm/config.py`

Add two fields to `TrainConfig` immediately after the existing `mixed_precision`
field (line 84):

```python
# torch.compile
# Compiles the model with torch.compile before DDP wrapping. Requires an
# initial compilation step on the first forward pass (may take several minutes
# for large models). Uses dynamic=True internally so variable sequence lengths
# do not trigger recompilation. Disabled by default.
compile: bool = False
# Compilation mode passed to torch.compile(mode=...).
# "default"          — balanced; safe for all hardware.
# "reduce-overhead"  — uses CUDA graphs to reduce kernel-launch overhead;
#                      best for small batch sizes.
# "max-autotune"     — tries more optimization strategies; longest compile
#                      time, best peak throughput on Blackwell.
compile_mode: str = "default"
```

Add the validation constant alongside the existing `_VALID_MIXED_PRECISION`:

```python
_VALID_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
```

Add validation in `__post_init__`, after the existing `mixed_precision` check:

```python
if self.compile_mode not in _VALID_COMPILE_MODES:
    raise ValueError(
        f"compile_mode must be one of {_VALID_COMPILE_MODES}, "
        f"got {self.compile_mode!r}"
    )
```

### 1.2 — `src/oplm/configs/train/base.yaml`

Add the two new fields inside the `train:` block, alongside `mixed_precision`:

```yaml
compile: false
compile_mode: default
```

---

## Phase 2: Trainer — Insert the Compile Step

**File:** `src/oplm/training/trainer.py`

**Exact insertion point:** between lines 101 and 103 of the current file —
after `model.gradient_checkpointing_enable()` and before
`optimizers = build_optimizers(model, cfg.train)`.

Current block (lines 98–104):
```python
gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
model = OplmForMaskedLM(cfg.model)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Optimizer and dataloader
optimizers = build_optimizers(model, cfg.train)
```

New block:
```python
gradient_checkpointing = getattr(cfg.model, "gradient_checkpointing", False)
model = OplmForMaskedLM(cfg.model)
if gradient_checkpointing:
    model.gradient_checkpointing_enable()

if cfg.train.compile:
    _status("[dim]Compiling model (torch.compile)...[/dim]")
    model = torch.compile(model, dynamic=True, mode=cfg.train.compile_mode)

# Optimizer and dataloader
optimizers = build_optimizers(model, cfg.train)
```

**Why this position:**
- After `gradient_checkpointing_enable()`: the compiled graph must include the
  `torch.utils.checkpoint.checkpoint` wrapper injected by GC enable; compiling
  first and then enabling GC would require recompilation.
- Before `build_optimizers`: `OptimizedModule.parameters()` delegates to the
  underlying model's parameter tensors, so the optimizer sees the same `Parameter`
  objects regardless. Optimizer construction is unaffected.
- Before `accelerator.prepare()`: DDP then wraps the `OptimizedModule`. This is
  the standard ordering and produces a cleaner compiled graph than compiling a
  DDP-wrapped model.

**No other changes to `train()` are required.** The training loop interacts with
`self.model` via standard `nn.Module` calls (`.train()`, `__call__`, `.parameters()`),
all of which `OptimizedModule` delegates correctly.

---

## Phase 3: Checkpoint — Handle `OptimizedModule` Unwrapping

**File:** `src/oplm/training/checkpoint.py`

**Problem:** When `torch.compile` is active, the sequence is:
```
torch.compile(model)  →  OptimizedModule
accelerator.prepare(OptimizedModule)  →  DDP(OptimizedModule)
accelerator.unwrap_model(DDP(OptimizedModule))  →  OptimizedModule
OptimizedModule.save_pretrained(...)  →  AttributeError (not a PreTrainedModel)
```

**Fix:** In `save_checkpoint`, patch the two lines that call `accelerator.unwrap_model`
(lines 82–83):

Current:
```python
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained(hf_dir)
```

Replace with:
```python
unwrapped = accelerator.unwrap_model(model)
# torch.compile wraps the model in OptimizedModule; peel it off to reach
# the underlying PreTrainedModel for save_pretrained.
if hasattr(unwrapped, "_orig_mod"):
    unwrapped = unwrapped._orig_mod
unwrapped.save_pretrained(hf_dir)
```

`_orig_mod` is the stable internal attribute set by `torch._dynamo.OptimizedModule`
to hold the original module. It is present whenever `torch.compile` was called;
absent otherwise. This change is a no-op for the non-compile path.

The `get_tokenizer().save_pretrained(hf_dir)` call on line 84 is unaffected.

---

## Phase 4: Tests

### 4.1 — Config tests: `tests/training/test_config.py`

Add the following cases (no hardware required):

```python
def test_compile_defaults() -> None:
    cfg = TrainConfig(wandb_enabled=False)
    assert cfg.compile is False
    assert cfg.compile_mode == "default"

@pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune"])
def test_compile_mode_valid(mode: str) -> None:
    cfg = TrainConfig(wandb_enabled=False, compile_mode=mode)
    assert cfg.compile_mode == mode

def test_compile_mode_invalid() -> None:
    with pytest.raises(ValueError, match="compile_mode"):
        TrainConfig(wandb_enabled=False, compile_mode="turbo")
```

### 4.2 — Update `tests/training/conftest.py`

Add `compile` and `compile_mode` kwargs to `tiny_train_cfg` so E2E tests can
opt in:

```python
def tiny_train_cfg(
    output_dir: Path,
    train_data: Path,
    *,
    ...
    mixed_precision: str = "no",
    compile: bool = False,           # new
    compile_mode: str = "default",   # new
    wandb_enabled: bool = False,
    ...
) -> OplmConfig:
    return OplmConfig(
        ...
        train=TrainConfig(
            ...
            mixed_precision=mixed_precision,
            compile=compile,              # new
            compile_mode=compile_mode,    # new
            wandb_enabled=wandb_enabled,
            ...
        ),
        ...
    )
```

Also add a `reset_dynamo` fixture to `tests/training/conftest.py` so compiled
code from one test does not affect the next. Scope it to function level but
**do not** make it autouse — opt in explicitly per test that uses compile:

```python
@pytest.fixture
def reset_dynamo() -> Generator[None, None, None]:
    """Clear torch.compile cache before and after a test."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
```

### 4.3 — Checkpoint test: `tests/training/test_checkpoint.py`

Add one test for the `_orig_mod` unwrapping path. Mark it slow (real IO) and
skip on CPU (torch.compile is not the typical target there):

```python
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hf_export_round_trips_with_compile(
    tmp_path: Path, reset_dynamo: None
) -> None:
    """HF export succeeds and weights match when the model was torch.compiled."""
    cfg = _cfg()
    accelerator = Accelerator(mixed_precision="no")
    model = OplmForMaskedLM(cfg.model)
    model = torch.compile(model, dynamic=True)
    model = accelerator.prepare(model)

    # Reference weight from the underlying model
    original_bias = (
        accelerator.unwrap_model(model)._orig_mod.lm_head.decoder.bias.detach().clone()
    )

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        cfg=cfg,
        output_dir=str(tmp_path),
        global_step=10,
        epoch=1,
        samples_seen=40,
        tokens_seen=400,
    )

    hf_dir = tmp_path / "checkpoint-10" / "hf"
    assert (hf_dir / "model.safetensors").exists()

    reloaded = OplmForMaskedLM.from_pretrained(str(hf_dir))
    assert torch.allclose(reloaded.lm_head.decoder.bias.detach().cpu(), original_bias.cpu())
```

### 4.4 — E2E compile test: `tests/training/test_e2e_compile.py`

New file. Marks: `@pytest.mark.slow`, CUDA skip.

```python
"""End-to-end training test with torch.compile enabled."""
from __future__ import annotations

import pytest
import torch

from tests.training.conftest import FullRecordingCallback, tiny_train_cfg

_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.mark.slow
@_REQUIRES_CUDA
def test_compile_default_mode_trains(
    tmp_path, train_data, reset_dynamo
) -> None:
    """compile=True, mode='default': loss is finite across all logged steps."""
    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=train_data,
        max_steps=5,
        log_every=1,
        compile=True,
        compile_mode="default",
    )
    cb = FullRecordingCallback()
    from oplm.training import Trainer
    Trainer(cfg, callbacks=[cb]).train()

    assert len(cb.train_logs) == 5
    for _, metrics in cb.train_logs:
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))


@pytest.mark.slow
@_REQUIRES_CUDA
def test_compile_checkpoint_hf_export(tmp_path, train_data, reset_dynamo) -> None:
    """compile=True: the HF export under checkpoint-N/hf/ loads via from_pretrained."""
    from oplm.model import OplmForMaskedLM

    cfg = tiny_train_cfg(
        output_dir=tmp_path,
        train_data=train_data,
        max_steps=3,
        save_every=3,
        compile=True,
    )
    from oplm.training import Trainer
    Trainer(cfg).train()

    hf_dir = tmp_path / "checkpoint-3" / "hf"
    assert hf_dir.exists(), "HF export directory missing"
    model = OplmForMaskedLM.from_pretrained(str(hf_dir))
    assert model is not None
```

`train_data` is an existing fixture (defined in `tests/data/conftest.py` or a
shared conftest) that supplies a path to a real parquet shard. Use the same
fixture already exercised by other E2E training tests.

### 4.5 — Verify existing tests still pass

After all changes:

```bash
pytest -m "not slow"            # fast path — no CUDA required
pytest -m "slow and not compile" # existing slow tests, no compile regressions
pytest -m "slow" -k "compile"   # new compile tests (CUDA required)
```

---

## Phase 5: Docs and Cleanup

### 5.1 — `AGENTS.md`

Add `compile` and `compile_mode` to the "Training" config reference:

```
train.compile       bool   false     Enable torch.compile (opt-in; adds first-step latency)
train.compile_mode  str    default   Compile mode: default | reduce-overhead | max-autotune
```

### 5.2 — `docs/TRAIN.md`

Add a "Compilation" section:

```markdown
## torch.compile

Pass `train.compile=true` to enable `torch.compile(model, dynamic=True)`.

First-step latency: compilation runs on the first forward pass and may take
several minutes for large models. Subsequent steps run the compiled graph.
Triton autotune artifacts are cached to `~/.cache/oplm/triton/autotune` so
repeated runs skip recompilation.

Recommended for all multi-GPU production runs. Use `compile_mode=max-autotune`
on Blackwell for best throughput at the cost of a longer initial compile.

```bash
# opt in via CLI override
torchrun --nproc_per_node=8 -m oplm.train --config my_run.yaml train.compile=true
```
```

### 5.3 — Lint and type check

```bash
ruff check src/
ruff format src/
ty check src/
```

Fix any issues before opening a PR.

---

## Verification Checklist

- [ ] `pytest -m "not slow"` passes with no failures
- [ ] `pytest tests/training/test_config.py` passes (new config tests)
- [ ] `pytest -m slow tests/training/test_checkpoint.py` passes (existing + new compile case)
- [ ] `pytest -m slow tests/training/test_e2e_compile.py` passes (CUDA box required)
- [ ] Smoke run — 5 steps, compile on, loss finite:
  ```bash
  ACCELERATE_USE_CPU=false accelerate launch --num_processes=1 -m oplm.train \
    --preset small train.max_steps=5 train.compile=true train.wandb_enabled=false \
    train.output_dir=/tmp/oplm-compile-smoke
  ```
- [ ] HF export from smoke run loads via `OplmForMaskedLM.from_pretrained`
- [ ] Ensure `TODOS.md` on `native-fsdp2-and-fp8` branch is annotated to reflect that
  Phase 4 (Trainer rewrite) must move the compile call to after `fully_shard()`,
  not before `accelerator.prepare()`, and that `compile`/`compile_mode` config
  fields already exist and do not need to be re-added.
