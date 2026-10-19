# FlexAttention for Packed Batches Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `model.packed_attention="flex"`, which runs packed batches through FlexAttention with a block mask built once per forward, so attention skips the blocks between sequences and a 2048-wide row costs about what a 512-wide row costs at equal tokens.

**Architecture:** `OplmStack.forward` builds one `BlockMask` from `segment_ids` when the config asks for flex, and threads it through every `OplmBlock` (including activation-checkpointed calls) to `OplmAttention`, which calls `flex_attention` instead of dense-mask SDPA. The default (`"sdpa"`) and every other path are unchanged. Two GPU scripts under `docs/perf/` check exactness against the bf16 floor and time equal-token batches by width and kernel.

**Tech Stack:** PyTorch 2.11 (`torch.nn.attention.flex_attention`), Hugging Face `PretrainedConfig`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-token-batching-flex-attention-design.md` (§4 is this plan; §7.1 step 2 is the GPU gate that follows it).

## Global Constraints

- Run everything through the repo venv: `.venv/bin/python -m pytest`, `.venv/bin/ruff`, `VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/`.
- `ruff format` only the files you touched; `main` has unrelated formatting drift. Do not commit churn in other files.
- `model.packed_attention` values: `"sdpa"` (default) and `"flex"`. Existing runs and checkpoints must be unchanged with the default.
- `"flex"` with `attention_dropout > 0` is a config error (FlexAttention has no dropout).
- The block mask is built **once per forward**, not once per layer.
- Unchanged paths: `output_attentions=True` (manual), batches without `segment_ids` (eval), `"sdpa"`.
- FlexAttention has **no backward on CPU** in torch 2.11 (verified: `NotImplementedError: FlexAttention does not support backward on CPU`). CPU tests check forward only; gradient parity is checked on GPU by `packing_check.py` (Task 3).
- Comments and docstrings match the surrounding code's density; commit messages use the repo's `type(scope): summary` style and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **bf16 autocast with mixed q/k/v dtypes.** Under autocast, channel QK-norm (LayerNorm) returns fp32 q/k while v is bf16; SDPA's autocast casts all three, FlexAttention does not. Expect flex to cast q/k/v to the autocast dtype and match SDPA to bf16 tolerance. Pinned by `test_flex_matches_sdpa_under_bf16_autocast` (Task 2).
2. **`output_attentions=True` with flex configured.** Expect the manual path (weights returned) and no block mask built. Pinned by `test_flex_config_keeps_manual_path_for_attention_weights` (Task 2).
3. **Padded batch (no `segment_ids`) with flex configured.** Expect output identical to `"sdpa"` and no block mask built. Pinned by `test_flex_config_without_segment_ids_is_unchanged` (Task 2).
4. **Activation checkpointing (full and selective) with flex.** The block mask must survive the `torch_checkpoint.checkpoint` argument list. Pinned by parametrizing `test_flex_packed_logits_match_sdpa` over checkpointing modes (Task 2); backward on GPU by Task 3.
5. **Configs saved before this change** (`hf/config.json` without `packed_attention`) load as `"sdpa"`. Pinned by `test_packed_attention_defaults_to_sdpa_for_old_configs` (Task 1).

---

### Task 1: `model.packed_attention` config field

**Files:**
- Modify: `src/oplm/model/configuration_oplm.py` (valid-values tuples near line 38, `__init__` signature near line 89, attribute assignment near line 154, `_validate` near line 336)
- Modify: `src/oplm/configs/model/base.yaml` (Dropout block near line 50)
- Modify: `docs/CONFIG.md` ("Runtime and special tokens" table near line 201)
- Test: `tests/model/test_config.py`

**Interfaces:**
- Produces: `OplmConfig.packed_attention: str`, one of `("sdpa", "flex")`, default `"sdpa"`. Task 2 reads it as `getattr(config, "packed_attention", "sdpa")`.

- [ ] **Step 1: Write the failing tests** (append to `tests/model/test_config.py`)

```python
def test_packed_attention_defaults_to_sdpa() -> None:
    assert OplmConfig().packed_attention == "sdpa"


def test_packed_attention_defaults_to_sdpa_for_old_configs() -> None:
    saved = OplmConfig().to_dict()
    del saved["packed_attention"]  # an hf/config.json written before the field existed
    assert OplmConfig.from_dict(saved).packed_attention == "sdpa"


def test_packed_attention_flex_round_trips() -> None:
    cfg = OplmConfig(packed_attention="flex")
    assert OplmConfig.from_dict(cfg.to_dict()).packed_attention == "flex"


def test_packed_attention_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="packed_attention must be one of"):
        OplmConfig(packed_attention="flash")


def test_packed_attention_flex_rejects_attention_dropout() -> None:
    with pytest.raises(ValueError, match="no attention dropout"):
        OplmConfig(packed_attention="flex", attention_dropout=0.1)
```

If `tests/model/test_config.py` imports the config under another name, use that name; check the file's top imports first.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_config.py -k packed_attention -v`
Expected: FAIL (`TypeError`/`AttributeError`: no `packed_attention`).

- [ ] **Step 3: Implement**

In `configuration_oplm.py`, next to the other `_VALID_*` tuples:

```python
_VALID_PACKED_ATTENTIONS = ("sdpa", "flex")
```

In `__init__`'s signature, after `hidden_dropout: float = 0.0,`:

```python
        packed_attention: str = "sdpa",
```

After `self.hidden_dropout = float(hidden_dropout)`:

```python
        self.packed_attention = str(packed_attention)
```

In `_validate`, after the `gradient_checkpointing_mode` check:

```python
        if self.packed_attention not in _VALID_PACKED_ATTENTIONS:
            raise ValueError(
                f"packed_attention must be one of {_VALID_PACKED_ATTENTIONS}; "
                f"got {self.packed_attention!r}."
            )
        if self.packed_attention == "flex" and self.attention_dropout > 0:
            raise ValueError(
                "packed_attention='flex' has no attention dropout; set attention_dropout=0.0 "
                f"(got {self.attention_dropout}) or use packed_attention='sdpa'."
            )
```

In `src/oplm/configs/model/base.yaml`, after `hidden_dropout: 0.0`:

```yaml

  # Attention kernel for packed batches (data.pack_sequences): sdpa (dense block-diagonal mask,
  # cost grows with row width) | flex (FlexAttention block mask, skips blocks between sequences)
  packed_attention: sdpa
```

In `docs/CONFIG.md`, add a row to the "Runtime and special tokens" table, after `model.gradient_checkpointing_mode`:

```markdown
| `model.packed_attention` | `str` | `sdpa` | Attention kernel for packed batches (`data.pack_sequences`). `sdpa` passes a dense block-diagonal mask, so cost grows with the row width. `flex` uses FlexAttention with a block mask built once per forward, skipping the 128×128 blocks between sequences; it needs `train.compile` for speed on GPU, has no attention dropout, and has no backward on CPU. Eval (unpacked) batches always use SDPA. |
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/model/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/oplm/model/configuration_oplm.py tests/model/test_config.py
.venv/bin/ruff format src/oplm/model/configuration_oplm.py tests/model/test_config.py
git add src/oplm/model/configuration_oplm.py src/oplm/configs/model/base.yaml docs/CONFIG.md tests/model/test_config.py
git commit -m "feat(model): model.packed_attention selects the packed-batch attention kernel

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: FlexAttention path for packed batches

**Files:**
- Modify: `src/oplm/model/masking.py` (add `segment_block_mask`, update `__all__`)
- Modify: `src/oplm/model/transformer.py` (`OplmBlock._forward_impl` ~line 225, `OplmBlock.forward` ~line 292, `OplmStack.forward` ~line 410; import `segment_block_mask`)
- Modify: `src/oplm/model/attention.py` (`forward` ~line 361, new `_flex_attention` next to `_sdpa_attention`)
- Test: `tests/model/test_packing.py`

**Interfaces:**
- Consumes: `OplmConfig.packed_attention` (Task 1).
- Produces:
  - `oplm.model.masking.segment_block_mask(segment_ids: torch.Tensor) -> BlockMask`;
  - `OplmBlock.forward(..., segment_ids=None, block_mask: BlockMask | None = None)`;
  - `OplmAttention.forward(..., segment_ids=None, block_mask: BlockMask | None = None)`.
  - Task 3's scripts toggle `model.config.packed_attention` at runtime; `OplmStack` must read the attribute at forward time (it holds `self.config`, the same object as `OplmForMaskedLM.config`).

- [ ] **Step 1: Write the failing tests** (append to `tests/model/test_packing.py`)

First give `_model` a `packed_attention` parameter (default `"sdpa"`), passed into `OplmConfig(...)`:

```python
def _model(checkpointing: str | None, packed_attention: str = "sdpa") -> OplmForMaskedLM:
    torch.manual_seed(0)
    cfg = OplmConfig(
        # ... existing fields unchanged ...
        packed_attention=packed_attention,
    )
```

Then the tests:

```python
import oplm.model.transformer as transformer_mod


@pytest.mark.parametrize("checkpointing", [None, "full", "selective"])
def test_flex_packed_logits_match_sdpa(checkpointing: str | None) -> None:
    """Same weights, same packed batch: flex and the dense-mask SDPA path agree (forward only:
    FlexAttention has no CPU backward). Train mode so checkpointing wraps each block."""
    _, packed = _batches()
    with torch.no_grad():
        ref = _model(checkpointing, "sdpa")(**packed).logits
        got = _model(checkpointing, "flex")(**packed).logits
    mask = packed["attention_mask"].bool()
    torch.testing.assert_close(got[mask], ref[mask], rtol=1e-4, atol=1e-5)


def test_flex_packed_logits_match_padded_rows() -> None:
    padded, packed = _batches()
    model = _model(None, "flex").eval()
    with torch.no_grad():
        ref = model(**padded).logits
        out = model(**packed).logits
    for row, got in enumerate(_unpack(out)):
        torch.testing.assert_close(got, ref[row, : LENGTHS[row]], rtol=1e-4, atol=1e-5)


def _count_block_masks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = transformer_mod.segment_block_mask

    def counting(segment_ids: torch.Tensor):  # noqa: ANN202 — returns BlockMask
        calls[0] += 1
        return real(segment_ids)

    monkeypatch.setattr(transformer_mod, "segment_block_mask", counting)
    return calls


def test_flex_block_mask_is_built_once_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_block_masks(monkeypatch)
    _, packed = _batches()
    with torch.no_grad():
        _model(None, "flex").eval()(**packed)  # 3 layers
    assert calls[0] == 1


def test_flex_config_without_segment_ids_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_block_masks(monkeypatch)
    padded, _ = _batches()
    with torch.no_grad():
        ref = _model(None, "sdpa").eval()(**padded).logits
        got = _model(None, "flex").eval()(**padded).logits
    assert calls[0] == 0
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_flex_config_keeps_manual_path_for_attention_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _count_block_masks(monkeypatch)
    _, packed = _batches()
    with torch.no_grad():
        out = _model(None, "flex").eval()(**packed, output_attentions=True)
    assert calls[0] == 0
    assert out.attentions is not None and all(a is not None for a in out.attentions)


def test_flex_matches_sdpa_under_bf16_autocast() -> None:
    """Channel QK-norm returns fp32 q/k under autocast while v is bf16; flex must cast like SDPA."""
    _, packed = _batches()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref = _model(None, "sdpa").eval()(**packed).logits
        got = _model(None, "flex").eval()(**packed).logits
    mask = packed["attention_mask"].bool()
    assert torch.isfinite(got[mask]).all()
    torch.testing.assert_close(got[mask], ref[mask], rtol=5e-2, atol=5e-2)
```

Check that `OplmForMaskedLM`'s output exposes `attentions` when `output_attentions=True` (it is a `MaskedLMOutput`); if the attribute name differs, follow the existing `test_packed_logits_match_padded_rows[manual]` usage.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_packing.py -k flex -v`
Expected: FAIL (`AttributeError: module 'oplm.model.transformer' has no attribute 'segment_block_mask'`).

- [ ] **Step 3: Add `segment_block_mask`** to `src/oplm/model/masking.py`

```python
if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

__all__ = [
    "prepare_attention_mask",
    "segment_attention_mask",
    "segment_block_mask",
    "zero_pad_positions",
]


def segment_block_mask(segment_ids: torch.Tensor) -> BlockMask:
    """FlexAttention block mask for a packed batch: each token attends only within its segment.

    The same pattern as :func:`segment_attention_mask`, as a block-sparse mask that lets
    FlexAttention skip every 128x128 block between different sequences, so attention cost follows
    the sequences' own lengths rather than the row width. Pads (segment 0) attend only to each
    other, so no query row is empty.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def same_segment(b, h, q_idx, kv_idx):  # noqa: ANN001, ANN202 — FlexAttention mask_mod
        return segment_ids[b, q_idx] == segment_ids[b, kv_idx]

    rows, width = segment_ids.shape
    return create_block_mask(
        same_segment, B=rows, H=None, Q_LEN=width, KV_LEN=width, device=segment_ids.device
    )
```

Add `from typing import TYPE_CHECKING` to the imports. The FlexAttention import stays inside the function: `masking.py` ships with the Hugging Face remote-code export (`_REMOTE_CODE_DEPS` in `configuration_oplm.py`), where an older torch must still import the module.

- [ ] **Step 4: Add the flex kernel to `OplmAttention`** (`src/oplm/model/attention.py`)

Add next to `_sdpa_attention`:

```python
    def _flex_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask: BlockMask
    ) -> torch.Tensor:
        """FlexAttention over a packed batch's block mask (`segment_block_mask`).

        Unlike SDPA, FlexAttention does not cast its inputs under autocast, and channel QK-norm
        returns fp32 q/k while v is in the autocast dtype; cast all three as SDPA would. No
        dropout (rejected by config validation).
        """
        from torch.nn.attention.flex_attention import flex_attention

        device_type = q.device.type
        if torch.is_autocast_enabled(device_type):
            dtype = torch.get_autocast_dtype(device_type)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        return flex_attention(q, k, v, block_mask=block_mask, scale=self.attn_scale)
```

Add `block_mask: BlockMask | None = None` to `forward`'s signature after `segment_ids`, document it in the Args (`FlexAttention block mask of a packed batch, built once per forward by OplmStack when config.packed_attention == "flex"; replaces the dense segment mask in the SDPA path`), and change the kernel dispatch:

```python
        if output_attentions:
            out, attn = self._manual_attention(q, k, v, attention_mask, segment_ids)
        elif block_mask is not None:
            out = self._flex_attention(q, k, v, block_mask)
            attn = None
        else:
            out = self._sdpa_attention(q, k, v, attention_mask, segment_ids)
            attn = None
```

Add to the `TYPE_CHECKING` block: `from torch.nn.attention.flex_attention import BlockMask`.

- [ ] **Step 5: Thread `block_mask` through `OplmBlock` and build it in `OplmStack`** (`src/oplm/model/transformer.py`)

Import: `from .masking import prepare_attention_mask, segment_block_mask`, and under `TYPE_CHECKING`: `from torch.nn.attention.flex_attention import BlockMask`.

`OplmBlock._forward_impl`: add `block_mask: BlockMask | None = None` after `segment_ids`, and pass it on:

```python
        attn_result = self.attention(
            attn_in,
            attention_mask,
            output_attentions,
            value_residual=value_residual,
            segment_ids=segment_ids,
            block_mask=block_mask,
        )
```

`OplmBlock.forward`: add `block_mask: BlockMask | None = None` after `segment_ids` (document it: `FlexAttention block mask shared by every block of this forward, or None`), and append `block_mask` after `segment_ids` in **all three** calls: both `torch_checkpoint.checkpoint(self._forward_impl, x, attention_mask, output_attentions, value_residual, segment_ids, block_mask, use_reentrant=False, ...)` calls, and the final `return self._forward_impl(x, attention_mask, output_attentions, value_residual, segment_ids, block_mask)`.

`OplmStack.forward`: before the `for layer_index in self.layer_execution_order:` loop:

```python
        # One FlexAttention block mask per forward, shared by every block (and every loop
        # revisit); read from config at call time so tools can switch kernels on a live model.
        block_mask = None
        if (
            segment_ids is not None
            and not output_attentions
            and getattr(self.config, "packed_attention", "sdpa") == "flex"
        ):
            block_mask = segment_block_mask(segment_ids)
```

and pass it in the block call:

```python
            result = block(
                x,
                attention_mask,
                output_attentions,
                value_residual=v1,
                segment_ids=segment_ids,
                block_mask=block_mask,
            )
```

The test patches `transformer_mod.segment_block_mask`, so call it by that module-level name (as written above), not through `masking.`.

- [ ] **Step 6: Run the flex tests, then the model suite**

Run: `.venv/bin/python -m pytest tests/model/test_packing.py -v`
Expected: all PASS (eager-mode FlexAttention warnings on CPU are expected).

Run: `.venv/bin/python -m pytest tests/model -m "not slow" -q`
Expected: all PASS (no change to existing behavior).

- [ ] **Step 7: Type-check and commit**

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/ty check src/oplm/model/
.venv/bin/ruff check src/oplm/model tests/model/test_packing.py
.venv/bin/ruff format src/oplm/model/masking.py src/oplm/model/attention.py src/oplm/model/transformer.py tests/model/test_packing.py
git add src/oplm/model/masking.py src/oplm/model/attention.py src/oplm/model/transformer.py tests/model/test_packing.py
git commit -m "feat(model): FlexAttention path for packed batches, one block mask per forward

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: GPU checks and docs

**Files:**
- Modify: `docs/perf/packing_check.py` (docstring, `main` loop, verdict loop)
- Modify: `docs/perf/packing-check.sbatch` (default `GIT_REF` to `main`)
- Create: `docs/perf/attention_width_bench.py`
- Create: `docs/perf/attention-width-bench.sbatch`
- Modify: `docs/TRAIN.md` ("Sequence packing" section, ~line 602)

**Interfaces:**
- Consumes: `model.config.packed_attention` toggled at runtime (Task 2 reads it per forward); `docs/perf/packing_memory.py`'s `token_lengths(cfg, shards_per_source)` and `packed_batch(sources, rng, rows, width, vocab)` (existing, unchanged).
- Produces: GPU evidence for spec goal 3 and the per-preset memory cap for `train.batch_tokens`. No unit tests: these are one-GPU scripts, like the existing ones in `docs/perf/`.

**Ruling (recorded here; the spec names "170M and 1.6B"):** there is no 1.6B preset (`AVAILABLE_PRESETS` = 50M, 170M, 400M, 800M, 1B, 3B, 6B, 12B). The bench runs 170M and 1B by default; any preset can be passed.

- [ ] **Step 1: Add the `flex` comparison to `packing_check.py`**

Docstring: add a line to the mode list:

```text
  flex       compiled packed rows through FlexAttention (model.packed_attention="flex") vs
             padded rows through cuDNN
```

In `main`, after `pack_compiled = run(compiled, params, pack)`:

```python
        # Same compiled module, other kernel: OplmStack reads packed_attention per forward
        # (Dynamo recompiles once for the changed attribute).
        model.config.packed_attention = "flex"
        try:
            pack_flex = run(compiled, params, pack)
        finally:
            model.config.packed_attention = "sdpa"
```

Add `("flex", pad_bf16, pack_flex),` to the comparison tuple after `("compiled", pack_cudnn, pack_compiled),`, and change the verdict loop to `for name in ("cudnn", "efficient", "compiled", "flex"):`.

In `docs/perf/packing-check.sbatch`, change the default ref: `GIT_REF="${OPLM_GIT_REF:-main}"`.

- [ ] **Step 2: Create `docs/perf/attention_width_bench.py`**

```python
"""Packed-attention cost by row width and kernel: dense-mask SDPA vs FlexAttention.

With a dense block-diagonal mask, packed attention computes every score of a row, so its cost
grows with the row width. FlexAttention with a block mask skips the 128x128 blocks between
sequences. This times a compiled training forward + backward (bf16 autocast) on packed batches
with the same number of token slots but different widths (default 84 x 512 and 21 x 2048),
with each kernel, and reports synced step time and peak memory. Sequence lengths are drawn from
the run's data.train sources by their fractions, as docs/perf/packing_memory.py does (imported
from the same directory). The model is a fresh preset (weights do not affect timing).

    python docs/perf/attention_width_bench.py --config <run.yaml> --preset 170M
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packing_memory import packed_batch, token_lengths  # noqa: E402 — sibling script

from oplm.config import load_config  # noqa: E402
from oplm.data.tokenizer import canonical_amino_acid_ids, get_tokenizer  # noqa: E402
from oplm.model import OplmForMaskedLM  # noqa: E402
from oplm.training.stream_replay import compile_like_trainer  # noqa: E402


def parse_shapes(text: str) -> list[tuple[int, int]]:
    """``"84x512,21x2048"`` -> ``[(84, 512), (21, 2048)]`` as (rows, width)."""
    return [tuple(int(v) for v in item.split("x")) for item in text.split(",")]  # type: ignore[misc]


def train_step(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> None:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()


def time_steps(model: torch.nn.Module, batch, warmup: int, steps: int) -> tuple[float, float]:
    """Mean synced ms per step and peak GiB over ``steps`` steps after ``warmup``."""
    for _ in range(warmup):
        train_step(model, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, batch)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / steps * 1000, torch.cuda.max_memory_allocated() / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="run YAML (for data.train)")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--shapes", default="84x512,21x2048")
    parser.add_argument("--backends", default="sdpa,flex")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--shards-per-source", type=int, default=4)
    parser.add_argument("--checkpointing", action="store_true", help="activation checkpointing")
    args = parser.parse_args()
    print(torch.__version__, torch.cuda.get_device_name())

    shapes = parse_shapes(args.shapes)
    cfg = load_config(["--config", str(args.config), "--preset", args.preset])
    cfg.model.max_position_embeddings = max(width for _, width in shapes)
    sources = token_lengths(cfg, args.shards_per_source)
    vocab = canonical_amino_acid_ids(get_tokenizer())
    rng = np.random.default_rng(0)
    batches = {shape: packed_batch(sources, rng, *shape, vocab) for shape in shapes}

    model = OplmForMaskedLM(cfg.model).cuda().train()
    if args.checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.preset}: {params:.0f}M params, checkpointing={args.checkpointing}")
    print(f"{'kernel':>6s} {'rows x width':>13s} {'real tokens':>12s} {'ms':>8s} {'GiB':>7s}")

    results: dict[tuple[str, tuple[int, int]], float] = {}
    for backend in args.backends.split(","):
        model.config.packed_attention = backend
        for shape, batch in batches.items():
            torch._dynamo.reset()
            compiled = compile_like_trainer(model, cfg, ddp=False)
            label = f"{shape[0]} x {shape[1]}"
            real = int(batch["attention_mask"].sum())
            try:
                ms, gib = time_steps(compiled, batch, args.warmup, args.steps)
            except torch.OutOfMemoryError:
                print(f"{backend:>6s} {label:>13s} {real:12d} {'OOM':>8s}")
                torch.cuda.empty_cache()
                continue
            results[backend, shape] = ms
            print(f"{backend:>6s} {label:>13s} {real:12d} {ms:8.1f} {gib:7.1f}")

    narrow, wide = min(shapes, key=lambda s: s[1]), max(shapes, key=lambda s: s[1])
    for backend in args.backends.split(","):
        if (backend, narrow) in results and (backend, wide) in results:
            ratio = results[backend, wide] / results[backend, narrow]
            print(f"{backend}: {wide[0]} x {wide[1]} costs {ratio:.2f}x {narrow[0]} x {narrow[1]}")


if __name__ == "__main__":
    main()
```

If `load_config` rejects `--preset` together with `--config`, check its argument parsing (`src/oplm/config.py`, `load_config`) and adapt; the docstring there says both are supported.

- [ ] **Step 3: Create `docs/perf/attention-width-bench.sbatch`**, modelled on `docs/perf/packing-check.sbatch` (same header layout, container, env file, `srun` block):

```bash
#!/bin/bash
# Time packed attention by row width and kernel (dense-mask SDPA vs FlexAttention) on one GPU
# (docs/perf/attention_width_bench.py), for each preset in OPLM_PRESETS. About twenty minutes.
#
# Submit from a login node:
#   OPLM_CONFIG=/path/to/run.yaml sbatch docs/perf/attention-width-bench.sbatch
# Overrides (environment variables):
#   OPLM_CONFIG=<yaml>      REQUIRED: a training YAML whose data.train supplies sequence lengths
#   OPLM_PRESETS="170M 1B"  presets to time (default below)
#   OPLM_GIT_REF=<commit>   oplm version to install (default below)
#   OPLM_RESULTS=/mnt/home/<you>/oplm-perf/attention-width/<jobid>
#
#SBATCH --job-name=oplm-attention-width
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

: "${OPLM_CONFIG:?set OPLM_CONFIG to a training YAML with data.train}"
ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
GIT_REF="${OPLM_GIT_REF:-main}"
PRESETS="${OPLM_PRESETS:-170M 1B}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/oplm-perf/attention-width/${SLURM_JOB_ID}}"
export GIT_REF RESULTS PRESETS OPLM_CONFIG

source "$ENV_FILE"  # creates JOB_WORK_DIR
mkdir -p "$RESULTS" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -euo pipefail
python -m pip install --quiet "oplm[train] @ git+https://github.com/briney/oplm@$GIT_REF"
for script in attention_width_bench.py packing_memory.py; do
  curl -sSfL "https://raw.githubusercontent.com/briney/oplm/$GIT_REF/docs/perf/$script" \
    -o "$RESULTS/$script"
done
python -m pip freeze | grep -i -E '^(oplm|torch|triton)' > "$RESULTS/pip-freeze.txt"
nvidia-smi > "$RESULTS/nvidia-smi.txt" 2>&1 || true
for preset in $PRESETS; do
  python "$RESULTS/attention_width_bench.py" --config "$OPLM_CONFIG" --preset "$preset" \
    2>&1 | tee "$RESULTS/bench-$preset.txt"
done
EOF

srun --nodes=1 --ntasks-per-node=1 --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== done; results in $RESULTS ==="
```

- [ ] **Step 4: Smoke-check the bench's CPU-safe parts**

Run: `.venv/bin/python -c "import ast, pathlib; ast.parse(pathlib.Path('docs/perf/attention_width_bench.py').read_text()); print('ok')"`
Expected: `ok`. Also run `bash -n docs/perf/attention-width-bench.sbatch` (expected: no output) and `.venv/bin/ruff check docs/perf/attention_width_bench.py docs/perf/packing_check.py`.

- [ ] **Step 5: Update `docs/TRAIN.md`'s "Sequence packing" section**

Replace the first "Costs and limits" bullet (the dense-mask one) with:

```markdown
- With the default `model.packed_attention: sdpa`, attention gets a dense block-diagonal
  `(rows, 1, T, T)` mask instead of a key-padding mask, and computes every score of a row: its
  cost grows with the row width, not with the sequences' lengths. (FlashAttention is out
  either way: it takes no mask at all.)
- `model.packed_attention: flex` runs packed batches through FlexAttention with a block mask
  built once per forward from `segment_ids`, skipping every 128×128 block between sequences, so
  attention cost follows each sequence's own length and wide rows (2048) cost about what narrow
  ones do. It is fast only under `train.compile`, has no attention dropout, and has no backward
  on CPU. `docs/perf/packing_check.py` checks it against padded rows at the bf16 floor and
  `docs/perf/attention_width_bench.py` times it against `sdpa` by row width.
```

- [ ] **Step 6: Commit**

```bash
.venv/bin/ruff format docs/perf/attention_width_bench.py docs/perf/packing_check.py
git add docs/perf/packing_check.py docs/perf/packing-check.sbatch docs/perf/attention_width_bench.py docs/perf/attention-width-bench.sbatch docs/TRAIN.md
git commit -m "perf(docs): flex mode in the packing check and a packed-attention width bench

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: Full fast suite**

Run: `.venv/bin/python -m pytest -m "not slow" -q`
Expected: all PASS.

---

## After this plan (not part of it)

Spec §7.1 step 2: the user runs `packing-check.sbatch` (expects `flex: PASS`) and `attention-width-bench.sbatch` on the cluster. If flex's 21 × 2048 is not within about 5% of its 84 × 512 at 170M, stop and report before starting the token-batching plan.
