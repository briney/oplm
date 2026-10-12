# Structure Head Milestone 2b — Losses and Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the folding head trainable end to end on the published wwPDB shards: a
training branch of `OplmForFolding.forward` that returns the weighted loss of spec §6.1,
the four stage-1 losses checked against Boltz-2, a `FoldTask` for the existing Trainer, the
`fold:` run-config block, `oplm fold train`, the frozen ESM-C adapter, and the single-B200
cost measurement that sets milestone 2c's batch and step budget.

**Architecture:** `forward` keeps the parity-verified inference path untouched and gains a
`targets` argument; the shared prefix (inputs embedder, shim, recurrence, readout, distogram
logits) is factored so DDP synchronizes gradients through one `forward`. The training branch
masks LM inputs in-model, draws the loop count per rank, evaluates the EDM denoiser once per
noise sample (microbatched under activation checkpointing so the sample count never multiplies
live memory), rolls out 20 sampler steps without gradient for the confidence head, and reduces
the per-sample losses so the microbatch size does not change the objective. `losses.py` holds
the fp32 loss math with explicit masks and finite degenerate cases; the Boltz-2 functions it
is checked against are vendored as test oracles. `FoldTask` builds the head from the `fold:`
block, attaches the frozen LM on the process device, builds `StructureDataset` loaders through
the existing `InterleavedDataset` seam, and returns `StepResult`s; the Trainer keeps
acceleration, EMA, checkpointing and resume.

**Tech Stack:** Python 3.11 (repo venv), torch 2.11, transformers 4.x/5.x within the repo
pins, accelerate (DDP), OmegaConf, typer; `esm==3.4.1.post1` behind a new optional extra
`esmc` (Python ≥ 3.12, GPU jobs only); Boltz-2 (`boltz==2.2.1`, MIT) loss code vendored
under `tests/fold/oracles/`.

**Spec:** [M2 pipeline-proof addendum](../specs/2026-10-11-fold-m2-pipeline-proof-design.md)
§5 (M2b), §3 (ESM-C), §2 (decisions), over the
[structure prediction head design](../specs/2026-10-07-structure-prediction-head-design.md)
§5.4 (precision), §6.1–6.4 (step, losses, recipe, task seam), §6.5–6.6 (EMA, frozen LM), §9
(losses and trainer rows), §10 (gates).

**Planning baseline:** `c8862ab` on `main` (milestone 2a merged as PR #34; dataset
`brineylab/oplmfold-training-data` tag `wwpdb-2026-10-12`). Work on branch `feat/fold-m2b`
in the worktree `.claude/worktrees/fold-m2b`, which already has its own `.venv`
(`uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,train]"`).
Tasks 1–11 need no GPU; Task 12 is the user-run GPU measurement on the cluster.

## Global Constraints

- `AGENTS.md` rules: Python 3.11+, `from __future__ import annotations` in every file, type hints on every signature, Google-style docstrings on public APIs, ruff line length 100, `pathlib.Path` only, no logic in `__init__.py` beyond imports, tests mirror the source layout, `@pytest.mark.slow` for anything over a few seconds.
- Commands: `.venv/bin/python -m pytest ...`, `.venv/bin/ruff check src/`, `.venv/bin/ruff format <changed files only>`, `VIRTUAL_ENV=.venv .venv/bin/ty check src/`. All three clean before every commit. Framework-boundary diagnostics only via a specific `# ty: ignore[<rule>]` with a reason.
- Dependencies: exactly one new optional extra, `esmc = ["esm==3.4.1.post1"]` (the package needs Python ≥ 3.12, `torch>=2.11,<2.12`, `transformers>=4.57.6,<5`; all inside this repo's ranges). Nothing under `src/` imports `esm` at module scope; the adapter imports it inside `load_esmc`. No `einops`/`einx` anywhere (the vendored oracles rewrite those calls with `torch.einsum`).
- Dependency direction (spec §3): `oplm.fold` imports core packages; core (`oplm.config`, `oplm.training`, `oplm.data`) imports nothing from `oplm.fold` except `src/oplm/cli.py`'s command registration. The `fold:` dataclass lives in `oplm.config` and knows nothing about `FoldConfig`; key validation happens on the fold side. `oplm.fold.cli` keeps torch out of module scope.
- Spec §5.4 precision: the frozen LM runs in bf16 under `no_grad` in eval mode on CUDA (fp32 on CPU); the trunk runs under bf16 autocast on CUDA; coordinates, alignment, the denoiser, the sampler and every loss and logit run in fp32 under `fp32_autocast_off`.
- Spec §6.1 step order, implemented inside `forward` when `targets` is given: (1) LM rows masked in-model (`lm_input_mask_fraction` 0.1, mask id 32, never BOS/EOS/pad, only in train mode); (2) loop count from a clamped Poisson (mean 3, clamp 1–6) drawn per rank, gradient through the last `recurrence_grad_loops` (2) loops, LM-pair dropout refreshed each loop (M1); (3) distogram logits from the symmetrized final pair; (4) per noise sample: augment and center the reference, draw `σ = σ_d · exp(N(−1.2, 1.5))` with `σ_d = 16`, one denoiser evaluation; microbatched over samples with shared conditioning; (5) a no-grad rollout of `confidence_rollout_steps` (20) sampler steps, confidence head on detached coordinates and detached trunk outputs, its loss trains only that head; (6) return the weighted loss, detached per-term metrics, valid token count and chain count. Diffusion replicas do not multiply token or sample accounting.
- Spec §6.2 and addendum §2: four terms with starting weights diffusion MSE 4.0, smooth lDDT 1.0, distogram 0.03, confidence 1e-4 (the confidence composite is pLDDT + PDE + resolved + `pae_weight` · PAE with `pae_weight` 1.0); the bond term and interface weighting are not implemented; protein atom weight 1; references and alignments are detached; invalid frames, empty masks and degenerate alignments give finite values. Bin edges, reductions and sub-term weights are recorded in code and config, not prose.
- Spec §6.4 / docs/TRAIN.md §17: `FoldTask` implements `build_model`, `build_dataloader`, `step -> StepResult(loss, tokens, samples, metrics)`, `flops_per_token -> None`; `StepResult.metrics` keys avoid the Trainer's reserved names; the default `MLMTask` behaviour and the MLM test suite are unchanged.
- Addendum §2 "no new entry field": every `data.train` entry of a fold run is a structure shard directory; `FoldTask` interprets them through `parse_train_configs` and mixes several with `InterleavedDataset`.
- Addendum §3 ESM-C facts, settled from the package source: `esm.models.esmc.model.EsmcModel.forward(input_ids, attention_mask, output_hidden_states, return_dict)` returns `.hidden_states` stacked on dim 0 with `num_hidden_layers + 1` entries: the input of every block (index 0 is the embedding output) and the final post-LayerNorm output. ESM-C 600M (`biohub/ESMC-600M`): 36 blocks, width 1152, 37 states, pad 1, BOS 0, EOS 2, mask 32. Identifier `esmc:esmc_600m`; `esmc_300m` and `esmc_6b` accepted (6B is non-commercial and not the default). The weights are gated behind the Cambrian license on the Hub.
- Addendum §5 config: the `fold:` block carries the LM identifier, revision and `lm_max_chain_length` (1024), `FoldConfig` overrides, crop size (384) and contiguous-crop probability (0.5), diffusion sample count (8) and microbatch, confidence rollout steps (20) and the loss weights. `oplm fold train` parses a run config exactly as `oplm train` does. EMA uses `train.ema_decay` (0.999 in the run configs). Optimizer and schedule reuse the existing options with μP disabled; the starting point is AdamW at 1e-3 with 1000 warmup steps; the run configs are authoritative.
- Carried from M2a (ledger): `FoldTask` passes `cfg.train.seed` to `StructureDataset` and relies on the Trainer's `set_epoch` forwarding; losses mask crops with zero resolved atoms; the frozen-LM gather uses the batched global `lm_rows` the collator offsets.
- Spec §9: loss parity against the vendored Boltz-2 oracles, rigid-transform recovery, masks and degeneracies on CPU with the real fixtures under `tests/fixtures/fold/`; trainer checks (gradients, overfit, resume with EMA, DDP smoke) in the fast or `slow` CPU suites; ESM-C tests skip without the extra.

## Review Focus

1. **A crop with no resolved atoms at all** (a contiguous window inside a long unresolved stretch of an otherwise eligible chain). Every loss term must be finite and zero-gradient-safe, not NaN (Task 4, `test_losses_are_finite_with_no_resolved_atoms`; Task 5, `test_training_forward_with_no_resolved_atoms_is_finite`).
2. **A token whose frame atoms are only partly resolved, or collinear** (CA missing; N, CA, C on a line). The PAE pair mask must drop that frame row without touching other rows (Task 4, `test_pae_mask_drops_unresolved_and_collinear_frames`).
3. **Fewer than three resolved atoms to align** (SVD of a rank-deficient covariance). The MSE term must be finite and the alignment must not raise (Task 4, `test_rigid_align_with_fewer_than_three_points_is_finite`).
4. **A diffusion microbatch that does not divide the sample count** (8 samples, microbatch 3). Chunks of 3, 3, 2 must produce exactly the same loss as one chunk of 8 (Task 5, `test_microbatch_does_not_change_the_loss`).
5. **LM input masking leaking into BOS, EOS or padding, or into eval mode.** Masking touches only inner residue positions in train mode and is a no-op in eval, so `fold()` and the parity oracle are unchanged (Task 5, `test_lm_input_masking_touches_only_inner_positions`, `test_eval_forward_ignores_train_settings`).

## File Map and Ownership

| File | Responsibility |
|---|---|
| `src/oplm/fold/data/featurize.py` | + `FoldFeatures.to(non_blocking)`, `.pin_memory()`; same on `FoldTargets` |
| `src/oplm/fold/data/collate.py` | + `FoldBatch.to`, `FoldBatch.pin_memory` |
| `src/oplm/data/sequence/loaders.py` | `DeviceDataLoader` moves any batch with a `.to` method |
| `src/oplm/fold/lm_shim.py` | batched per-token gather; `lm_state_count` honours a declared `num_hidden_states`; + `mask_lm_inputs` |
| New `src/oplm/fold/esmc.py` | `FrozenEsmC`, `load_esmc`, `parse_esmc_name`, `ESMC_REPOS` |
| `src/oplm/fold/modeling_fold.py` | `load_frozen_lm` resolves `esmc:`; `TrainSettings`, `FoldTrainOutput`, `draw_num_loops`; `forward(targets=...)` training branch |
| `src/oplm/fold/configuration_fold.py` | + `distogram_min_dist`, `distogram_max_dist` |
| `src/oplm/fold/diffusion.py` | `StructureHead.expand`; `prepare` built on it |
| New `src/oplm/fold/losses.py` | `LossWeights`, `edm_weight`, `pair_distances`, `masked_mean`, `diffusion_mse`, `smooth_lddt`, `distogram_loss`, `token_frames`, `express_in_frames`, `plddt_loss`, `pae_loss`, `pde_loss`, `resolved_loss` |
| `src/oplm/config.py` | `FoldRunConfig`, `OplmConfig.fold`, `serialize_config` emits it |
| `src/oplm/training/task.py` | `TrainTask.step(batch: Any)` |
| `src/oplm/training/mup.py` | `mup_lr_multiplier` tolerates configs without `mup_enable` |
| `src/oplm/training/optim.py` | same tolerance in the Muon guard |
| New `src/oplm/fold/training.py` | `require_fold`, `build_fold_config`, `load_initial_head`, `FoldTask`, `PeakMemoryCallback` |
| New `src/oplm/fold/train.py` | `python -m oplm.fold.train` entry (accelerate launch) |
| `src/oplm/train.py` | `main(cfg, task=...)` |
| `src/oplm/fold/cli.py` | + `oplm fold train`, `oplm fold measure` |
| `src/oplm/fold/__init__.py` | + lazy `FoldTask` export |
| `src/oplm/fold/predict.py` | narrows `forward`'s return to `FoldOutput` |
| `pyproject.toml` | `esmc` extra |
| New `tests/fold/oracles/` | vendored Boltz-2 loss functions + `README.md`; `THIRD_PARTY_NOTICES.md` gains the MIT notice |
| `tests/fold/conftest.py` | the real-fixture fixtures move here from `tests/fold/data/conftest.py` (constants stay there) |
| `tests/fold/helpers.py` | + `tiny_fold_model_overrides`, `tiny_fold_run_cfg`, `dummy_targets` |
| New tests | `tests/data/test_device_dataloader.py`, `tests/fold/test_esmc.py`, `tests/fold/test_oracles.py`, `tests/fold/test_losses.py`, `tests/fold/test_modeling_train.py`, `tests/fold/test_training.py`, `tests/fold/_train_worker.py`, `tests/fold/test_training_distributed.py`; extended `tests/fold/data/test_collate.py`, `tests/fold/test_lm_shim.py`, `tests/fold/test_diffusion.py`, `tests/fold/test_cli.py`, `tests/training/test_config.py` |
| New `configs/fold/proof-w128.yaml`, `configs/fold/proof-w256.yaml` | the milestone-2 run configs (batch/steps finalized by M2c) |
| New `docs/fold/b200-m2b-measure.sbatch`, `docs/fold/m2/README.md` | the GPU measurement job and its evidence |
| `docs/FOLD.md`, `docs/TRAIN.md`, `docs/CONFIG.md`, `docs/TESTING_E2E.md` | contract updates |

Execute Tasks 1–11 in order; Task 12 is the user-run GPU acceptance.

---

### Task 1: Batched LM states and device placement of `FoldBatch`

**Files:**
- Modify: `src/oplm/fold/data/featurize.py` (`FoldFeatures.to`, `FoldTargets.to`, new `pin_memory` methods)
- Modify: `src/oplm/fold/data/collate.py` (`FoldBatch`)
- Modify: `src/oplm/data/sequence/loaders.py` (`DeviceDataLoader.__iter__`)
- Modify: `src/oplm/fold/lm_shim.py` (`lm_state_count`, `frozen_lm_hidden_states`)
- Test: `tests/fold/data/test_collate.py` (append), `tests/data/test_device_dataloader.py` (new), `tests/fold/test_lm_shim.py` (append)

**Interfaces:**
- Consumes: `FoldFeatures`, `FoldTargets`, `FoldBatch`, `collate_examples` (M2a); `DeviceDataLoader` (Trainer wraps the task's loader with it at `trainer.py:721`).
- Produces: `FoldBatch.to(device, *, non_blocking=False) -> FoldBatch`, `FoldBatch.pin_memory() -> FoldBatch`; `frozen_lm_hidden_states(lm, features, *, dtype=None) -> Tensor` of shape `(B, L, K, D)` for any batch size, accepting `out.hidden_states` as a sequence of `(C, T, D)` tensors or one stacked `(K, C, T, D)` tensor; `lm_state_count(lm)` returns `int(lm.num_hidden_states)` when the module declares it. Task 2's adapter declares it; Task 5's forward consumes batched states; Task 7's loader relies on the device wrapper.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fold/data/test_collate.py`:

```python
def test_fold_batch_to_and_pin_memory(parsed: Callable[[str], MolecularComplex]) -> None:
    ex = featurize_complex(parsed("1crn"))
    batch = collate_examples([ex, ex], crop_size=64)
    moved = batch.to("cpu")
    assert moved is not batch and moved.ids == batch.ids
    assert moved.features.chain_ids == batch.features.chain_ids
    torch.testing.assert_close(moved.features.ref_pos, batch.features.ref_pos)
    torch.testing.assert_close(moved.targets.coords, batch.targets.coords)
    if torch.cuda.is_available():
        pinned = batch.pin_memory()
        assert pinned.features.ref_pos.is_pinned() and pinned.targets.coords.is_pinned()
```

Create `tests/data/test_device_dataloader.py`:

```python
"""DeviceDataLoader moves dict batches field by field and object batches through their ``.to``."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from oplm.data import DeviceDataLoader


class _Batch:
    def __init__(self, x: torch.Tensor) -> None:
        self.x = x
        self.moved_to: torch.device | None = None

    def to(self, device: torch.device, *, non_blocking: bool = False) -> _Batch:
        out = _Batch(self.x.to(device, non_blocking=non_blocking))
        out.moved_to = device
        return out


def test_dict_batches_move_tensors_only() -> None:
    loader = DataLoader([{"a": torch.zeros(2), "ids": ["x"]}], batch_size=None)
    (batch,) = list(DeviceDataLoader(loader, torch.device("cpu")))
    assert batch["a"].tolist() == [0.0, 0.0] and batch["ids"] == ["x"]


def test_object_batches_use_their_to_method() -> None:
    loader = DataLoader([_Batch(torch.ones(1)), _Batch(torch.zeros(1))], batch_size=None)
    out = list(DeviceDataLoader(loader, torch.device("cpu")))
    assert [b.moved_to for b in out] == [torch.device("cpu")] * 2
    assert [b.x.item() for b in out] == [1.0, 0.0]
```

Append to `tests/fold/test_lm_shim.py` (add `from torch import nn`, `from oplm.fold.data.collate import collate_examples`, `from oplm.fold.data.featurize import FoldExample, FoldTargets` to the imports):

```python
def _targets(features) -> FoldTargets:
    L, A = features.num_tokens, features.num_atoms
    return FoldTargets(
        coords=torch.zeros(1, A, 3),
        resolved=torch.zeros(1, A, dtype=torch.bool),
        bfactor=torch.zeros(1, A),
        frame_atom_idx=torch.zeros(1, L, 3, dtype=torch.long),
    )


def test_frozen_lm_states_gather_a_batch_like_single_examples() -> None:
    lm = tiny_lm()
    a = featurize([ChainSpec("MKVLA", "A")])
    b = featurize([ChainSpec("GGS", "B"), ChainSpec("WW", "C")])
    batch = collate_examples(
        [FoldExample("a", a, _targets(a)), FoldExample("b", b, _targets(b))], crop_size=8
    )
    states = frozen_lm_hidden_states(lm, batch.features)
    assert states.shape == (2, 8, 3, 32)
    torch.testing.assert_close(states[0, :5], frozen_lm_hidden_states(lm, a)[0])
    torch.testing.assert_close(states[1, :5], frozen_lm_hidden_states(lm, b)[0])
    assert states[:, 5:].abs().sum() == 0  # padded tokens are zero


def test_frozen_lm_states_accept_a_stacked_hidden_state_tensor() -> None:
    class _Stacked(nn.Module):
        num_hidden_states = 3

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
            x = torch.ones(*input_ids.shape, 4) * input_ids[..., None].float()
            return type("Out", (), {"hidden_states": torch.stack([x, 2 * x, 3 * x])})()

    f = featurize([ChainSpec("MKV", "A")])
    states = frozen_lm_hidden_states(_Stacked(), f)
    assert states.shape == (1, 3, 3, 4) and lm_state_count(_Stacked()) == 3
    torch.testing.assert_close(states[0, 0, 1], 2 * f.lm_input_ids[0, 1].float() * torch.ones(4))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/data/test_collate.py tests/data/test_device_dataloader.py tests/fold/test_lm_shim.py -q`
Expected: FAIL (`FoldBatch` has no `to`; the object batch has no `items`; the batched gather raises or returns the wrong shape).

- [ ] **Step 3: Implement**

`src/oplm/fold/data/featurize.py`: add a module helper and extend both dataclasses.

```python
def _map_tensors(obj: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
    """Apply ``fn`` to every tensor field of a dataclass instance; other fields pass through."""
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        out[f.name] = fn(value) if isinstance(value, torch.Tensor) else value
    return out
```

(`Any` from `typing`, `Callable` under `TYPE_CHECKING` from `collections.abc`.) Replace `FoldFeatures.to` and `FoldTargets.to` and add `pin_memory` to both:

```python
    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> FoldFeatures:
        """Return a copy with every tensor on ``device``."""
        return FoldFeatures(**_map_tensors(self, lambda t: t.to(device, non_blocking=non_blocking)))

    def pin_memory(self) -> FoldFeatures:
        """Return a copy with every tensor in pinned host memory (``DataLoader(pin_memory=True)``)."""
        return FoldFeatures(**_map_tensors(self, lambda t: t.pin_memory()))
```

`src/oplm/fold/data/collate.py`, on `FoldBatch`:

```python
    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> FoldBatch:
        """Return a copy with every tensor on ``device`` (``DeviceDataLoader`` calls this)."""
        return FoldBatch(
            ids=list(self.ids),
            features=self.features.to(device, non_blocking=non_blocking),
            targets=self.targets.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> FoldBatch:
        """Return a copy in pinned host memory; ``DataLoader(pin_memory=True)`` calls this."""
        return FoldBatch(
            ids=list(self.ids), features=self.features.pin_memory(), targets=self.targets.pin_memory()
        )
```

`src/oplm/data/sequence/loaders.py`, `DeviceDataLoader.__iter__` (add `from collections.abc import Mapping` at module scope; it is used at runtime):

```python
    def __iter__(self):
        for batch in self._dataloader:
            if isinstance(batch, Mapping):
                yield {
                    k: v.to(self._device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:  # a batch object that owns its placement (oplm.fold.data.collate.FoldBatch)
                yield batch.to(self._device, non_blocking=True)
```

Extend the class docstring: "Dict batches move tensor values; any other batch must implement `.to(device, non_blocking=...)` and is moved as a whole."

`src/oplm/fold/lm_shim.py`:

```python
def lm_state_count(lm: nn.Module) -> int:
    """Number of hidden states ``lm`` returns.

    A module that declares ``num_hidden_states`` (the ESM-C adapter) is believed; an
    ``OplmModel`` returns its executed blocks plus the embedding output.
    """
    declared = getattr(lm, "num_hidden_states", None)
    if declared is not None:
        return int(declared)
    return len(lm.backbone.layer_execution_order) + 1  # ty: ignore[unresolved-attribute, invalid-argument-type]  # nn.Module attrs are Tensor | Module


def frozen_lm_hidden_states(
    lm: nn.Module, features: FoldFeatures, *, dtype: torch.dtype | None = None
) -> Tensor:
    """Run the frozen LM on the per-chain rows and gather every state per token: ``(B, L, K, D)``.

    ``features.lm_rows`` names each token's row in the batch-wide LM stack (the collator offsets
    it per example) and ``lm_positions`` its position inside that row. ``out.hidden_states`` may
    be a sequence of ``(C, T, D)`` tensors (OPLM) or one ``(K, C, T, D)`` tensor (ESM-C). The LM
    must already be in eval mode on the features' device. Padded tokens are zero. Runs under
    ``torch.no_grad`` (not inference mode: the OPLM RoPE cache re-assigns buffers lazily).
    """
    with torch.no_grad():
        out = lm(
            input_ids=features.lm_input_ids,
            attention_mask=features.lm_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    states = torch.stack(tuple(out.hidden_states), dim=2)  # (C, T, K, D)
    if dtype is not None:
        states = states.to(dtype)
    gathered = states[features.lm_rows, features.lm_positions]  # (B, L, K, D)
    return gathered * features.token_mask[..., None, None].to(gathered.dtype)
```

- [ ] **Step 4: Run the tests and the touched suites**

Run: `.venv/bin/python -m pytest tests/fold/data tests/fold/test_lm_shim.py tests/fold/test_modeling.py tests/fold/test_predict.py tests/data -q`
Expected: PASS (the batch-size-1 callers see identical values).

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/data/featurize.py src/oplm/fold/data/collate.py src/oplm/data/sequence/loaders.py src/oplm/fold/lm_shim.py tests/data/test_device_dataloader.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/data/featurize.py src/oplm/fold/data/collate.py src/oplm/data/sequence/loaders.py src/oplm/fold/lm_shim.py tests/fold/data/test_collate.py tests/data/test_device_dataloader.py tests/fold/test_lm_shim.py
git commit -m "feat(fold): batched frozen-LM gather and FoldBatch device placement"
```

---

### Task 2: Frozen ESM-C adapter and the `esmc:` identifier

**Files:**
- Create: `src/oplm/fold/esmc.py`
- Modify: `src/oplm/fold/modeling_fold.py` (`load_frozen_lm`), `pyproject.toml` (`esmc` extra)
- Test: `tests/fold/test_esmc.py` (new), `tests/fold/test_modeling.py` (append one test)

**Interfaces:**
- Consumes: `frozen_lm_hidden_states`, `lm_state_count` (Task 1); `OplmForFolding.attach_lm` reads `lm.config.hidden_size`.
- Produces: `ESMC_PREFIX = "esmc:"`, `ESMC_REPOS`, `parse_esmc_name(name_or_path) -> str | None`, `FrozenEsmC(model, *, name, revision=None)` with `.config`, `.num_hidden_states`, `.name`, `.revision`, `forward(input_ids, attention_mask=None, output_hidden_states=True, return_dict=True)`; `load_esmc(name, *, revision=None, device=None, dtype=None) -> FrozenEsmC`; `load_frozen_lm` returns `nn.Module` and resolves the prefix. Task 7's `build_model` calls `load_frozen_lm(cfg.fold.lm, ...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fold/test_esmc.py`:

```python
"""The frozen ESM-C adapter: contract on a fake encoder, the ``esmc:`` prefix, the real model."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.esmc import ESMC_PREFIX, ESMC_REPOS, FrozenEsmC, load_esmc, parse_esmc_name
from oplm.fold.lm_shim import frozen_lm_hidden_states, lm_state_count
from oplm.fold.modeling_fold import OplmForFolding, load_frozen_lm
from tests.fold.helpers import tiny_fold_config

_HAS_ESM = importlib.util.find_spec("esm") is not None


class _FakeEsmc(nn.Module):
    """Mimics ``EsmcModel``: ``config`` fields and ``(n_layers + 1, B, T, D)`` hidden states."""

    def __init__(self, layers: int = 2, width: int = 32) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=width, num_hidden_layers=layers, pad_token_id=1, mask_token_id=32
        )
        self.embed = nn.Embedding(64, width)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
        x = self.embed(input_ids)
        states = torch.stack([x * (k + 1) for k in range(self.config.num_hidden_layers + 1)])
        return SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])


def test_parse_esmc_name_and_repo_table() -> None:
    assert parse_esmc_name("esmc:esmc_600m") == "esmc_600m"
    assert parse_esmc_name("/some/path") is None and parse_esmc_name("biohub/x#esmc") is None
    assert ESMC_REPOS == {
        "esmc_300m": "biohub/ESMC-300M",
        "esmc_600m": "biohub/ESMC-600M",
        "esmc_6b": "biohub/ESMC-6B",
    }
    assert ESMC_PREFIX == "esmc:"


def test_frozen_wrapper_presents_the_hidden_state_contract() -> None:
    lm = FrozenEsmC(_FakeEsmc(), name="esmc_fake", revision="r0")
    assert lm.num_hidden_states == 3 and lm_state_count(lm) == 3
    assert lm.config.hidden_size == 32 and lm.name == "esmc_fake" and lm.revision == "r0"
    assert all(not p.requires_grad for p in lm.parameters())
    lm.train()
    assert not lm.training and not lm.model.training
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")])
    states = frozen_lm_hidden_states(lm, f)
    assert states.shape == (1, 5, 3, 32)
    head = OplmForFolding(tiny_fold_config(lm_hidden_size=32, lm_num_hidden_states=3)).eval()
    head.attach_lm(lm, name_or_path="esmc:esmc_fake")
    with torch.no_grad():
        out = head(f, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(out.coords).all()
    assert not any("embed" in k for k in head.state_dict())


def test_wrapper_rejects_a_state_count_mismatch() -> None:
    class _Short(_FakeEsmc):
        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
            out = super().forward(input_ids, attention_mask)
            out.hidden_states = out.hidden_states[:-1]
            return out

    lm = FrozenEsmC(_Short(), name="x")
    with pytest.raises(RuntimeError, match="hidden states"):
        lm(input_ids=torch.zeros(1, 4, dtype=torch.long))


@pytest.mark.skipif(_HAS_ESM, reason="esm is installed; the ImportError path is unreachable")
def test_load_frozen_lm_esmc_prefix_names_the_extra_without_the_package() -> None:
    with pytest.raises(ImportError, match=r"oplm\[esmc\]"):
        load_frozen_lm("esmc:esmc_600m")


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_ESM, reason="esm package not installed (oplm[esmc], Python >= 3.12)")
def test_real_esmc_300m_states_match_its_config() -> None:
    lm = load_esmc("esmc_300m", device="cpu")
    f = featurize([ChainSpec("MKTAYIAKQRQISFVKSHFSRQ", "A")])
    states = frozen_lm_hidden_states(lm, f)
    assert states.shape == (1, 22, lm.config.num_hidden_layers + 1, lm.config.hidden_size)
    assert torch.isfinite(states).all()
    assert FoldConfig(lm_hidden_size=lm.config.hidden_size, lm_num_hidden_states=lm.num_hidden_states)
```

Append to `tests/fold/test_modeling.py`:

```python
def test_from_pretrained_resolves_an_esmc_identifier_through_the_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import oplm.fold.esmc as esmc

    class _Fake(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("C", (), {"hidden_size": 32, "num_hidden_layers": 2})()
            self.embed = torch.nn.Embedding(64, 32)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True):
            x = self.embed(input_ids)
            return type("O", (), {"hidden_states": torch.stack([x, x, x])})()

    monkeypatch.setattr(
        esmc, "load_esmc", lambda name, **kw: esmc.FrozenEsmC(_Fake(), name=name, revision=kw.get("revision"))
    )
    model = OplmForFolding(_tiny_cfg(lm_name_or_path="esmc:esmc_fake", lm_revision="v1"))
    model.save_pretrained(tmp_path / "fold")
    reloaded = OplmForFolding.from_pretrained(tmp_path / "fold")
    assert reloaded.lm is not None and reloaded.lm.name == "esmc_fake" and reloaded.lm.revision == "v1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/test_esmc.py tests/fold/test_modeling.py -q`
Expected: FAIL with `ModuleNotFoundError: oplm.fold.esmc`.

- [ ] **Step 3: Implement**

Create `src/oplm/fold/esmc.py`:

```python
"""Frozen ESM-C behind the hidden-state contract (addendum §3).

``load_esmc`` wraps the ``esm`` package's HF-style ``EsmcModel`` (``esm.models.esmc.model``,
``esm==3.4.1.post1``) in :class:`FrozenEsmC`, which presents the call
:func:`oplm.fold.lm_shim.frozen_lm_hidden_states` makes (``input_ids``, ``attention_mask``,
``output_hidden_states``, ``return_dict``) and returns the package's output, whose
``hidden_states`` is one ``(num_hidden_layers + 1, C, T, D)`` tensor: the input of every block
(index 0 is the embedding output) and the final post-LayerNorm output. ESM-C 600M therefore has
37 states of width 1152. The package needs Python >= 3.12 (extra ``oplm[esmc]``); it is imported
inside :func:`load_esmc` only, and CPU tests use a fake encoder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["ESMC_PREFIX", "ESMC_REPOS", "FrozenEsmC", "load_esmc", "parse_esmc_name"]

ESMC_PREFIX = "esmc:"
ESMC_REPOS = {
    "esmc_300m": "biohub/ESMC-300M",
    "esmc_600m": "biohub/ESMC-600M",
    "esmc_6b": "biohub/ESMC-6B",  # non-commercial license; a config choice, never the default
}


def parse_esmc_name(name_or_path: str) -> str | None:
    """The model name after ``esmc:``, or ``None`` when ``name_or_path`` has no such prefix."""
    text = str(name_or_path)
    return text[len(ESMC_PREFIX) :] if text.startswith(ESMC_PREFIX) else None


class FrozenEsmC(nn.Module):
    """A frozen, eval-only ESM-C encoder with the shape facts the shim needs.

    ``config`` is the wrapped model's config (``hidden_size``, ``num_hidden_layers``,
    ``pad_token_id``, ``mask_token_id``); ``num_hidden_states`` is ``num_hidden_layers + 1``.
    ``train()`` never leaves eval mode, so the head's ``train()`` cannot switch it.
    """

    def __init__(self, model: nn.Module, *, name: str, revision: str | None = None) -> None:
        super().__init__()
        self.model = model
        self.name = name
        self.revision = revision
        self.config: Any = getattr(model, "config")  # noqa: B009  # nn.Module attrs type as Tensor | Module
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def num_hidden_states(self) -> int:
        return int(self.config.num_hidden_layers) + 1

    def train(self, mode: bool = True) -> FrozenEsmC:
        super().train(False)
        return self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        """Run the encoder; the result's ``hidden_states`` is ``(num_hidden_states, C, T, D)``.

        Raises:
            RuntimeError: The encoder returned no hidden states or a different count.
        """
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        states = getattr(out, "hidden_states", None)
        if states is None or states.shape[0] != self.num_hidden_states:
            got = None if states is None else int(states.shape[0])
            raise RuntimeError(
                f"ESM-C returned {got} hidden states; expected {self.num_hidden_states} "
                "(block inputs plus the final output)"
            )
        return out


def load_esmc(
    name: str,
    *,
    revision: str | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FrozenEsmC:
    """Load ``esmc_300m`` / ``esmc_600m`` / ``esmc_6b`` (or any ESM-C repo id or local dir).

    The package selects its kernels from ``device`` at load time, so the device is passed
    explicitly (CPU when ``None``). The Hub weights are gated behind the Cambrian license; a
    logged-in ``huggingface_hub`` token is required once per machine.

    Raises:
        ImportError: The ``esm`` package is missing (install ``oplm[esmc]`` on Python >= 3.12).
    """
    try:
        from esm.models.esmc.model import EsmcModel  # ty: ignore[unresolved-import]  # optional extra
    except ImportError as exc:
        raise ImportError(
            f"{ESMC_PREFIX}{name} needs the `esm` package: install `oplm[esmc]` (Python >= 3.12)"
        ) from exc
    target = torch.device("cpu") if device is None else torch.device(device)
    model = EsmcModel.from_pretrained(
        ESMC_REPOS.get(name, name), device=target, dtype=dtype, revision=revision
    )
    return FrozenEsmC(model, name=name, revision=revision)
```

`src/oplm/fold/modeling_fold.py`: import `parse_esmc_name` from `oplm.fold.esmc` at module scope and change `load_frozen_lm`'s return annotation to `nn.Module` with this branch before the `#esmc` check:

```python
    esmc_name = parse_esmc_name(str(name_or_path))
    if esmc_name is not None:
        from oplm.fold.esmc import load_esmc

        return load_esmc(esmc_name, revision=revision, device=device, dtype=dtype)
```

Update its docstring: "`esmc:<name>` loads the frozen ESM-C adapter (addendum §3); OPLM paths and Hub ids load the bare encoder; the `<repo>#esmc` form still raises."

`pyproject.toml`, after the `fold` extra:

```toml
esmc = [
    # Frozen ESM-C for the milestone-2 pipeline proof (oplm.fold.esmc, docs/FOLD.md §9). The
    # package needs Python >= 3.12 and transformers < 5, both inside this repo's ranges; the
    # repo's own dev venv is 3.11, so GPU jobs build a 3.12 venv (docs/fold/b200-m2b-measure.sbatch).
    "esm==3.4.1.post1",
]
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_esmc.py tests/fold/test_modeling.py -q`
Expected: PASS; the real-model test skips (no `esm` in the 3.11 venv).

- [ ] **Step 5: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/esmc.py src/oplm/fold/modeling_fold.py tests/fold/test_esmc.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/esmc.py src/oplm/fold/modeling_fold.py pyproject.toml tests/fold/test_esmc.py tests/fold/test_modeling.py
git commit -m "feat(fold): frozen ESM-C adapter behind the esmc: identifier"
```

---

### Task 3: Vendored Boltz-2 loss oracles

**Files:**
- Create: `tests/fold/oracles/__init__.py`, `tests/fold/oracles/README.md`, `tests/fold/oracles/boltz2_diffusion.py`, `tests/fold/oracles/boltz2_distogram.py`, `tests/fold/oracles/boltz2_confidence.py`
- Modify: `THIRD_PARTY_NOTICES.md`
- Test: `tests/fold/test_oracles.py` (new)

**Interfaces:**
- Produces (verbatim Boltz-2 signatures): `boltz2_diffusion.weighted_rigid_align(true_coords, pred_coords, weights, mask)`, `boltz2_diffusion.smooth_lddt_loss(pred_coords, true_coords, is_nucleotide, coords_mask, nucleic_acid_cutoff=30.0, other_cutoff=15.0, multiplicity=1)`, `boltz2_distogram.distogram_loss(output, feats, aggregate_distogram=True)`, `boltz2_confidence.{resolved_loss, get_target_lddt, plddt_loss, lddt_dist, express_coordinate_in_frame, get_target_pae, pae_loss, get_target_pde, pde_loss, compute_collinear_mask, compute_frame_pred}` and `CHAIN_TYPE_IDS`. Task 4's parity tests call them.

- [ ] **Step 1: Fetch the pinned source**

The oracle is the `boltz==2.2.1` wheel from PyPI (sha256 `b8c62bbdede1922931d9203118f62c858f11aa699bf91fd4c05a5ed6a6d8b4fc`; upstream <https://github.com/jwohlwend/boltz>, MIT). Download it without dependencies into a scratch directory and unpack it (it is not installed and never imported from `src/`):

```bash
mkdir -p /tmp/boltz-src && .venv/bin/python -m pip download -q --no-deps --only-binary=:all: "boltz==2.2.1" -d /tmp/boltz-src
.venv/bin/python -m zipfile -e /tmp/boltz-src/boltz-2.2.1-py3-none-any.whl /tmp/boltz-src
sha256sum /tmp/boltz-src/boltz-2.2.1-py3-none-any.whl   # must print the hash above
```

Source files: `boltz/model/loss/diffusionv2.py`, `boltz/model/loss/distogramv2.py`, `boltz/model/loss/confidencev2.py`, `boltz/model/layers/confidence_utils.py`, `boltz/data/const.py` (for `chain_types`).

- [ ] **Step 2: Write the failing smoke test**

Create `tests/fold/test_oracles.py`:

```python
"""The vendored Boltz-2 oracles import and reproduce their closed-form values on trivial inputs."""

from __future__ import annotations

import torch

from tests.fold.oracles import boltz2_confidence, boltz2_diffusion, boltz2_distogram


def test_weighted_rigid_align_is_the_identity_on_matching_clouds() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 7, 3)
    aligned = boltz2_diffusion.weighted_rigid_align(x, x, torch.ones(2, 7), torch.ones(2, 7))
    torch.testing.assert_close(aligned, x, atol=1e-5, rtol=0)


def test_smooth_lddt_of_identical_coordinates_is_the_sigmoid_constant() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 9, 3) * 3
    loss = boltz2_diffusion.smooth_lddt_loss(x, x, torch.zeros(1, 9), torch.ones(1, 9))
    expected = 1 - sum(torch.sigmoid(torch.tensor(t)) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=0)


def test_lddt_dist_of_identical_matrices_is_one() -> None:
    d = torch.rand(1, 5, 5) * 10
    mask = torch.ones(1, 5, 5) - torch.eye(5)[None]
    score, _ = boltz2_confidence.lddt_dist(d, d, mask, cutoff=15.0, per_atom=True)
    torch.testing.assert_close(score, torch.ones(1, 5), atol=1e-6, rtol=0)


def test_distogram_oracle_runs_on_a_one_hot_target() -> None:
    logits = torch.zeros(1, 3, 3, 1, 4)
    target = torch.nn.functional.one_hot(torch.zeros(1, 3, 3, 1, dtype=torch.long), 4).float()
    feats = {"disto_target": target, "token_disto_mask": torch.ones(1, 3)}
    loss, per_example = boltz2_distogram.distogram_loss({"pdistogram": logits}, feats)
    torch.testing.assert_close(loss, torch.log(torch.tensor(4.0)), atol=1e-5, rtol=0)
    assert per_example.shape == (1,)
    assert boltz2_confidence.CHAIN_TYPE_IDS["PROTEIN"] == 0
```

Run: `.venv/bin/python -m pytest tests/fold/test_oracles.py -q`
Expected: FAIL with `ModuleNotFoundError: tests.fold.oracles`.

- [ ] **Step 3: Vendor the functions**

`tests/fold/oracles/__init__.py` holds only a docstring: `"""Frozen, attributed Boltz-2 loss functions used as test oracles (README.md)."""`.

Copy the functions listed in **Interfaces** into the three modules, each starting with:

```python
# ruff: noqa
# fmt: off
# Vendored from boltz==2.2.1 (https://github.com/jwohlwend/boltz, MIT License, Copyright (c) 2024
# Jeremy Wohlwend, Gabriele Corso, Saro Passaro): <upstream file path>. Modifications are listed
# in tests/fold/oracles/README.md. Test oracle only; never imported from src/.
```

Allowed modifications, and no others:
- `boltz2_diffusion.py` (from `boltz/model/loss/diffusionv2.py`): drop `import einx` and `from einops import einsum, rearrange`; replace `einsum(weights * pred_coords_centered, true_coords_centered, "... n i, ... n j -> ... i j")` with `torch.einsum("...ni,...nj->...ij", weights * pred_coords_centered, true_coords_centered)`, `einsum(U, F, V, "... i j, ... j k, ... l k -> ... i l")` with `torch.einsum("...ij,...jk,...lk->...il", U, F, V)`, and `einsum(true_coords_centered, rot_matrix, "... n i, ... j i -> ... n j")` with `torch.einsum("...ni,...ji->...nj", true_coords_centered, rot_matrix)`. Keep the `print` warnings.
- `boltz2_distogram.py` (from `boltz/model/loss/distogramv2.py`): verbatim.
- `boltz2_confidence.py` (from `boltz/model/loss/confidencev2.py` and `boltz/model/layers/confidence_utils.py`): define `CHAIN_TYPE_IDS = {name: i for i, name in enumerate(["PROTEIN", "DNA", "RNA", "NONPOLYMER"])}` (copy the order from `boltz/data/const.py`'s `chain_types`) and replace every `const.chain_type_ids[...]` with `CHAIN_TYPE_IDS[...]`; replace the deprecated `torch.cuda.amp.autocast(enabled=False)` with `torch.autocast("cuda", enabled=False)`; drop the `confidence_loss` wrapper and `tm_function`/`compute_ptms` (unused). Include `compute_collinear_mask` and `compute_frame_pred` from `confidence_utils.py` (not the duplicates in `confidence.py`).

`tests/fold/oracles/README.md`:

```markdown
# Boltz-2 loss oracles

Frozen copies of the Boltz-2 loss functions `src/oplm/fold/losses.py` is checked against
(spec §6.2, §9). Source: the `boltz==2.2.1` wheel from PyPI, sha256
`b8c62bbdede1922931d9203118f62c858f11aa699bf91fd4c05a5ed6a6d8b4fc`, upstream
<https://github.com/jwohlwend/boltz> (MIT License, Copyright (c) 2024 Jeremy Wohlwend, Gabriele
Corso, Saro Passaro; full text in `THIRD_PARTY_NOTICES.md`). These modules are test code only.

| module | upstream file | functions |
|---|---|---|
| `boltz2_diffusion.py` | `boltz/model/loss/diffusionv2.py` | `weighted_rigid_align`, `smooth_lddt_loss` |
| `boltz2_distogram.py` | `boltz/model/loss/distogramv2.py` | `distogram_loss` |
| `boltz2_confidence.py` | `boltz/model/loss/confidencev2.py`, `boltz/model/layers/confidence_utils.py` | `resolved_loss`, `get_target_lddt`, `plddt_loss`, `lddt_dist`, `express_coordinate_in_frame`, `get_target_pae`, `pae_loss`, `get_target_pde`, `pde_loss`, `compute_collinear_mask`, `compute_frame_pred` |

Modifications: `einx`/`einops` calls rewritten as `torch.einsum` (no new dependency);
`boltz.data.const.chain_type_ids` inlined as `CHAIN_TYPE_IDS`; `torch.cuda.amp.autocast` replaced by
`torch.autocast("cuda", ...)`; unused wrappers (`confidence_loss`, `tm_function`, `compute_ptms`)
omitted. Everything else is verbatim, including the eps constants and reductions the parity tests
in `tests/fold/test_losses.py` match.

Conventions the oracles encode (and OPLM reproduces): the reference is aligned onto the
prediction; smooth lDDT scores pairs under 15 Å (30 Å for nucleotides) excluding `i == j`;
pLDDT targets are per-atom lDDT against representative atoms within 15 Å; PAE expresses token `j`'s
frame center in token `i`'s (N, CA, C) frame; PDE is the representative-atom distance error;
`resolved` class 0 means resolved.
```

`THIRD_PARTY_NOTICES.md`: add, before the Apache text, a section

```markdown
## Boltz-2 (test oracles only)

`tests/fold/oracles/` carries frozen copies of loss functions from Boltz-2
(<https://github.com/jwohlwend/boltz>, `boltz==2.2.1`), used only to check
`src/oplm/fold/losses.py`; see `tests/fold/oracles/README.md` for the file list and
modifications. Nothing under `src/` is derived from Boltz. The MIT license text follows.

MIT License

Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

<the remainder of the MIT text from the wheel's LICENSE file, verbatim>
```

- [ ] **Step 4: Run the smoke tests**

Run: `.venv/bin/python -m pytest tests/fold/test_oracles.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format tests/fold/test_oracles.py
git add tests/fold/oracles tests/fold/test_oracles.py THIRD_PARTY_NOTICES.md
git commit -m "test(fold): vendor the Boltz-2 loss functions as attributed oracles"
```

---
### Task 4: Losses (`oplm.fold.losses`)

**Files:**
- Create: `src/oplm/fold/losses.py`
- Modify: `src/oplm/fold/configuration_fold.py` (+ `distogram_min_dist`, `distogram_max_dist`), `tests/fold/conftest.py` and `tests/fold/data/conftest.py` (move the real-fixture fixtures up one level), `tests/fold/helpers.py` (+ `dummy_targets`)
- Test: `tests/fold/test_losses.py` (new)

**Interfaces:**
- Consumes: `weighted_rigid_align(x, x_gt, weights)` and `fp32_autocast_off` (M1; `x` is superposed onto `x_gt`); `FoldFeatures`/`FoldTargets` axes (`distogram_atom_idx (B, L)`, `frame_atom_idx (B, L, 3)` = N, CA, C; `resolved (B, A)`); the oracles of Task 3.
- Produces, every function fp32 under `fp32_autocast_off`, masks `bool`, one value per leading row:
  - `LossWeights(diffusion_mse=4.0, smooth_lddt=1.0, distogram=0.03, confidence=1e-4, pae=1.0)` (frozen dataclass).
  - `edm_weight(sigma, sigma_data) -> Tensor`, `pair_distances(x, eps=1e-8) -> Tensor`, `masked_mean(values, mask, dims) -> Tensor`.
  - `diffusion_mse(pred (N, A, 3), target (N, A, 3), weights (N, A) float) -> (N,)`.
  - `smooth_lddt(pred (N, A, 3), target (N, A, 3), mask (N, A), *, cutoff=15.0) -> (N,)`.
  - `distogram_loss(logits (B, L, L, bins), coords (B, A, 3), resolved (B, A), token_mask (B, L), rep_atom_idx (B, L), *, min_dist, max_dist) -> (B,)`.
  - `token_frames(coords (N, A, 3), frame_atom_idx (N, L, 3), atom_valid (N, A), token_mask (N, L)) -> (frames (N, L, 3, 3), centers (N, L, 3), valid (N, L))`, `express_in_frames(frames, centers) -> (N, L, L, 3)`.
  - `plddt_loss(logits (N, A, bins), pred (N, A, 3), target (B, A, 3), resolved (B, A), atom_mask (B, A), token_mask (B, L), rep_atom_idx (B, L), *, cutoff=15.0) -> (N,)`, `pae_loss(logits (N, L, L, bins), pred, target, resolved, token_mask, frame_atom_idx (B, L, 3), *, max_dist) -> (N,)`, `pde_loss(logits (N, L, L, bins), pred, target, resolved, token_mask, rep_atom_idx, *, max_dist) -> (N,)`, `resolved_loss(logits (N, A, 2), resolved (B, A), atom_mask (B, A)) -> (N,)`; `N = B · k` rollouts per example, base-batch tensors are repeated `k` times inside.
  - `FoldConfig.distogram_min_dist = 2.0`, `distogram_max_dist = 22.0` (validated `0 < min < max`).
  Task 5's training branch calls all of them.

- [ ] **Step 1: Move the real-fixture fixtures up to `tests/fold/conftest.py`**

Move `ccd_table`, `conformers`, `parsed`, `_preprocess_fixtures`, `fixture_dataset` and `fixture_dataset_early_cutoff` from `tests/fold/data/conftest.py` into `tests/fold/conftest.py` (keeping the existing `fixtures_dir` fixture there), importing `FOLD_FIXTURES`, `MMCIF_DIR` from `tests.fold.data.conftest`. `tests/fold/data/conftest.py` keeps only the three constants and its docstring (`test_mmcif.py` and `test_dataset.py` import them from there). Add to `tests/fold/helpers.py`:

```python
def dummy_targets(features: FoldFeatures) -> FoldTargets:
    """All-unresolved targets with the right axes, for tests that need a FoldExample."""
    from oplm.fold.data.featurize import FoldTargets

    L, A = features.num_tokens, features.num_atoms
    return FoldTargets(
        coords=torch.zeros(1, A, 3),
        resolved=torch.zeros(1, A, dtype=torch.bool),
        bfactor=torch.zeros(1, A),
        frame_atom_idx=torch.zeros(1, L, 3, dtype=torch.long),
    )
```

Run: `.venv/bin/python -m pytest tests/fold/data -q` — Expected: PASS (fixtures resolve through the parent conftest).

- [ ] **Step 2: Write the failing tests**

Create `tests/fold/test_losses.py`:

```python
"""Loss terms against the vendored Boltz-2 oracles, rigid-transform recovery, masks, degeneracies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.data.collate import collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.fold.diffusion import random_rotations
from oplm.fold.losses import (
    LossWeights,
    diffusion_mse,
    distogram_loss,
    edm_weight,
    express_in_frames,
    pae_loss,
    pair_distances,
    pde_loss,
    plddt_loss,
    resolved_loss,
    smooth_lddt,
    token_frames,
)
from tests.fold.oracles import boltz2_confidence, boltz2_diffusion, boltz2_distogram

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.collate import FoldBatch
    from oplm.fold.data.complex import MolecularComplex

BINS = 8
MAX_DIST = 32.0


@pytest.fixture(scope="module")
def batch(parsed: Callable[[str], MolecularComplex]) -> FoldBatch:
    """1crn and 1ejg (46 residues each, 1ejg has unresolved atoms) collated at crop 64."""
    return collate_examples(
        [featurize_complex(parsed("1crn")), featurize_complex(parsed("1ejg"))], crop_size=64
    )


def _pred(batch: FoldBatch, noise: float = 0.5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return batch.targets.coords + noise * torch.randn(batch.targets.coords.shape, generator=g)


def _boltz_feats(batch: FoldBatch) -> dict[str, torch.Tensor]:
    f, t = batch.features, batch.targets
    B, L, A = f.token_mask.shape[0], f.num_tokens, f.num_atoms
    atom_to_token = F.one_hot(f.atom_to_token, L).float() * f.atom_mask[..., None].float()
    token_to_rep = torch.zeros(B, L, A)
    token_to_rep[torch.arange(B)[:, None], torch.arange(L)[None], f.distogram_atom_idx] = 1.0
    token_to_rep = token_to_rep * f.token_mask[..., None].float()
    rep_resolved = torch.gather(t.resolved, 1, f.distogram_atom_idx) & f.token_mask
    frame_ok = f.token_mask.clone()
    for k in range(3):
        frame_ok &= torch.gather(t.resolved, 1, t.frame_atom_idx[..., k])
    return {
        "atom_to_token": atom_to_token,
        "token_to_rep_atom": token_to_rep,
        "r_set_to_rep_atom": token_to_rep,
        "token_pad_mask": f.token_mask.float(),
        "atom_pad_mask": f.atom_mask.float(),
        "mol_type": torch.zeros(B, L, dtype=torch.long),
        "asym_id": f.asym_id,
        "frames_idx": t.frame_atom_idx,
        "frame_resolved_mask": frame_ok.float(),
        "token_disto_mask": rep_resolved.float(),
        "atom_resolved_mask": t.resolved.float(),
    }


def test_edm_weight_matches_the_boltz_formula() -> None:
    sigma = torch.tensor([1.0, 16.0, 100.0])
    torch.testing.assert_close(edm_weight(sigma, 16.0), (sigma**2 + 256.0) / (sigma * 16.0) ** 2)


def test_diffusion_mse_matches_boltz(batch: FoldBatch) -> None:
    pred, true, res = _pred(batch), batch.targets.coords, batch.targets.resolved.float()
    ours = diffusion_mse(pred, true, res)
    aligned = boltz2_diffusion.weighted_rigid_align(true, pred, torch.ones_like(res), res)
    err = ((pred - aligned) ** 2).sum(-1)
    expected = (err * res).sum(-1) / ((3 * res).sum(-1) + 1e-5)
    torch.testing.assert_close(ours, expected, atol=1e-4, rtol=1e-4)


def test_diffusion_mse_recovers_a_rigid_transform(batch: FoldBatch) -> None:
    true = batch.targets.coords
    rot = random_rotations(2, device=true.device, dtype=true.dtype, generator=torch.Generator().manual_seed(1))
    moved = torch.einsum("bad,bds->bas", true, rot) + torch.tensor([[[3.0, -2.0, 7.0]]])
    loss = diffusion_mse(moved, true, batch.targets.resolved.float())
    assert loss.shape == (2,) and (loss < 1e-6).all()


def test_rigid_align_with_fewer_than_three_points_is_finite(batch: FoldBatch) -> None:
    """Review Focus 3."""
    pred, true = _pred(batch), batch.targets.coords
    for n_points in (0, 1, 2):
        w = torch.zeros_like(batch.targets.resolved.float())
        w[:, :n_points] = 1.0
        loss = diffusion_mse(pred.clone().requires_grad_(True), true, w)
        assert torch.isfinite(loss).all()
        loss.sum().backward()


def test_smooth_lddt_matches_boltz(batch: FoldBatch) -> None:
    pred, true = _pred(batch), batch.targets.coords
    ours = smooth_lddt(pred, true, batch.targets.resolved)
    expected = boltz2_diffusion.smooth_lddt_loss(
        pred, true, torch.zeros(2, pred.shape[1]), coords_mask=batch.targets.resolved.float()
    )
    torch.testing.assert_close(ours.mean(), expected, atol=1e-4, rtol=1e-4)


def test_smooth_lddt_of_a_rigid_transform_is_the_sigmoid_constant(batch: FoldBatch) -> None:
    true = batch.targets.coords
    rot = random_rotations(2, device=true.device, dtype=true.dtype, generator=torch.Generator().manual_seed(2))
    moved = torch.einsum("bad,bds->bas", true, rot) + 1.0
    expected = 1 - sum(torch.sigmoid(torch.tensor(t)) for t in (0.5, 1.0, 2.0, 4.0)) / 4
    torch.testing.assert_close(smooth_lddt(moved, true, batch.targets.resolved), expected.expand(2), atol=1e-4, rtol=0)


def test_distogram_matches_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    torch.manual_seed(0)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    ours = distogram_loss(logits, t.coords, t.resolved, f.token_mask, f.distogram_atom_idx, min_dist=2.0, max_dist=22.0)
    reps = torch.gather(t.coords, 1, f.distogram_atom_idx[..., None].expand(-1, -1, 3))
    bins = torch.bucketize(torch.cdist(reps, reps), torch.linspace(2.0, 22.0, BINS - 1))
    feats = _boltz_feats(batch)
    feats["disto_target"] = F.one_hot(bins, BINS).float()[:, :, :, None, :]
    expected, _ = boltz2_distogram.distogram_loss({"pdistogram": logits[:, :, :, None, :]}, feats)
    torch.testing.assert_close(ours.mean(), expected, atol=1e-5, rtol=1e-5)


def test_distogram_perfect_logits_give_near_zero_loss(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    reps = torch.gather(t.coords, 1, f.distogram_atom_idx[..., None].expand(-1, -1, 3))
    bins = torch.bucketize(pair_distances(reps), torch.linspace(2.0, 22.0, BINS - 1))
    logits = 50.0 * F.one_hot(bins, BINS).float()
    loss = distogram_loss(logits, t.coords, t.resolved, f.token_mask, f.distogram_atom_idx, min_dist=2.0, max_dist=22.0)
    assert (loss < 1e-4).all()


def test_token_frames_and_pae_match_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    pred = _pred(batch, noise=1.0)
    torch.manual_seed(1)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    ours = pae_loss(logits, pred, t.coords, t.resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST)
    expected, _ = boltz2_confidence.pae_loss(logits, pred, _boltz_feats(batch), t.coords, t.resolved.float(), max_dist=MAX_DIST)
    torch.testing.assert_close(ours.mean(), expected, atol=1e-5, rtol=1e-5)
    frames, centers, ok = token_frames(t.coords, t.frame_atom_idx, t.resolved, f.token_mask)
    assert frames.shape == (2, f.num_tokens, 3, 3) and ok.dtype == torch.bool
    torch.testing.assert_close(torch.einsum("blij,blkj->blik", frames, frames)[ok], torch.eye(3).expand(int(ok.sum()), 3, 3), atol=1e-4, rtol=0)
    assert express_in_frames(frames, centers).shape == (2, f.num_tokens, f.num_tokens, 3)


def test_pae_mask_drops_unresolved_and_collinear_frames(batch: FoldBatch) -> None:
    """Review Focus 2: a frame missing CA, and a collinear frame, drop out of the pair mask."""
    f, t = batch.features, batch.targets
    coords, resolved = t.coords.clone(), t.resolved.clone()
    resolved[0, t.frame_atom_idx[0, 3, 1]] = False  # token 3 of 1crn loses its CA
    n_idx, ca_idx, c_idx = (t.frame_atom_idx[0, 5, k] for k in range(3))
    coords[0, c_idx] = coords[0, ca_idx] + (coords[0, ca_idx] - coords[0, n_idx])  # token 5: N, CA, C collinear
    _, _, ok = token_frames(coords, t.frame_atom_idx, resolved, f.token_mask)
    assert not ok[0, 3] and not ok[0, 5] and ok[0, 4]
    pred = _pred(batch, noise=1.0)
    torch.manual_seed(2)
    logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    base = pae_loss(logits, pred, coords, resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST)
    perturbed = logits.clone()
    perturbed[0, 3] += 10.0
    perturbed[0, 5] += 10.0
    torch.testing.assert_close(pae_loss(perturbed, pred, coords, resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST), base)


def test_pde_and_plddt_and_resolved_match_boltz(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    pred = _pred(batch, noise=1.0)
    feats = _boltz_feats(batch)
    torch.manual_seed(3)
    pde_logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS)
    plddt_logits = torch.randn(2, f.num_atoms, 10)
    resolved_logits = torch.randn(2, f.num_atoms, 2)
    ours_pde = pde_loss(pde_logits, pred, t.coords, t.resolved, f.token_mask, f.distogram_atom_idx, max_dist=MAX_DIST)
    exp_pde, _ = boltz2_confidence.pde_loss(pde_logits, pred, feats, t.coords, t.resolved.float(), max_dist=MAX_DIST)
    torch.testing.assert_close(ours_pde.mean(), exp_pde, atol=1e-5, rtol=1e-5)
    ours_plddt = plddt_loss(plddt_logits, pred, t.coords, t.resolved, f.atom_mask, f.token_mask, f.distogram_atom_idx)
    exp_plddt, _ = boltz2_confidence.plddt_loss(plddt_logits, pred, feats, t.coords, t.resolved.float(), token_level_confidence=False)
    torch.testing.assert_close(ours_plddt.mean(), exp_plddt, atol=1e-5, rtol=1e-5)
    ours_res = resolved_loss(resolved_logits, t.resolved, f.atom_mask)
    exp_res = boltz2_confidence.resolved_loss(resolved_logits, feats, t.resolved.float())
    torch.testing.assert_close(ours_res.mean(), exp_res, atol=1e-5, rtol=1e-5)


def test_confidence_targets_of_a_rigid_transform_are_bin_zero(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    rot = random_rotations(2, device=t.coords.device, dtype=t.coords.dtype, generator=torch.Generator().manual_seed(4))
    moved = torch.einsum("bad,bds->bas", t.coords, rot) - 5.0
    zero_bin = torch.zeros(2, f.num_tokens, f.num_tokens, BINS)
    zero_bin[..., 0] = 50.0
    assert (pae_loss(zero_bin, moved, t.coords, t.resolved, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST) < 1e-4).all()
    assert (pde_loss(zero_bin, moved, t.coords, t.resolved, f.token_mask, f.distogram_atom_idx, max_dist=MAX_DIST) < 1e-4).all()
    top_bin = torch.zeros(2, f.num_atoms, 10)
    top_bin[..., -1] = 50.0  # lDDT 1.0 -> the last bin
    assert (plddt_loss(top_bin, moved, t.coords, t.resolved, f.atom_mask, f.token_mask, f.distogram_atom_idx) < 1e-4).all()


def test_rollout_multiplicity_repeats_the_base_batch(batch: FoldBatch) -> None:
    f, t = batch.features, batch.targets
    # Two rollouts per example; rollout j of example b sits at row b·2 + j.
    pred = torch.stack([_pred(batch, seed=0), _pred(batch, seed=1)], dim=1).reshape(4, f.num_atoms, 3)
    logits = torch.randn(4, f.num_atoms, 2)
    losses = resolved_loss(logits, t.resolved, f.atom_mask)
    single = torch.cat([resolved_loss(logits[2 * b : 2 * b + 2], t.resolved[b : b + 1], f.atom_mask[b : b + 1]) for b in range(2)])
    torch.testing.assert_close(losses, single)


def test_losses_are_finite_with_no_resolved_atoms(batch: FoldBatch) -> None:
    """Review Focus 1."""
    f, t = batch.features, batch.targets
    none = torch.zeros_like(t.resolved)
    pred = _pred(batch).requires_grad_(True)
    torch.manual_seed(5)
    pair_logits = torch.randn(2, f.num_tokens, f.num_tokens, BINS, requires_grad=True)
    atom_logits = torch.randn(2, f.num_atoms, 10, requires_grad=True)
    res_logits = torch.randn(2, f.num_atoms, 2, requires_grad=True)
    terms = [
        diffusion_mse(pred, t.coords, none.float()),
        smooth_lddt(pred, t.coords, none),
        distogram_loss(pair_logits, t.coords, none, f.token_mask, f.distogram_atom_idx, min_dist=2.0, max_dist=22.0),
        pae_loss(pair_logits, pred, t.coords, none, f.token_mask, t.frame_atom_idx, max_dist=MAX_DIST),
        pde_loss(pair_logits, pred, t.coords, none, f.token_mask, f.distogram_atom_idx, max_dist=MAX_DIST),
        plddt_loss(atom_logits, pred, t.coords, none, f.atom_mask, f.token_mask, f.distogram_atom_idx),
        resolved_loss(res_logits, none, torch.zeros_like(f.atom_mask)),
    ]
    total = sum(term.sum() for term in terms)
    assert torch.isfinite(total) and float(total) == 0.0
    total.backward()
    for leaf in (pred, pair_logits, atom_logits, res_logits):
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()


def test_loss_weights_defaults() -> None:
    w = LossWeights()
    assert (w.diffusion_mse, w.smooth_lddt, w.distogram, w.confidence, w.pae) == (4.0, 1.0, 0.03, 1e-4, 1.0)
    assert math.isclose(LossWeights(pae=0.0).pae, 0.0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/test_losses.py -q`
Expected: FAIL with `ModuleNotFoundError: oplm.fold.losses`.

- [ ] **Step 4: Implement `losses.py` and the config fields**

`src/oplm/fold/configuration_fold.py`: add keyword arguments `distogram_min_dist: float = 2.0`, `distogram_max_dist: float = 22.0` (stored as floats next to `distogram_bins`) and the validation `if not 0 < self.distogram_min_dist < self.distogram_max_dist: raise ValueError("distogram_min_dist must be positive and below distogram_max_dist.")`. Document them in the class docstring: "`distogram_{min,max}_dist` are the training target's bin range (AF3/Boltz: 64 bins over 2–22 Å); they do not change the released head".

Create `src/oplm/fold/losses.py`:

```python
"""Training losses for the structure head (spec §6.2; addendum §5), checked against Boltz-2.

Every function runs in fp32 with any outer autocast disabled and returns one value per leading
row so the model can weight and reduce; masks are explicit and an empty mask gives a finite zero.
The conventions, recorded here because the oracle code (tests/fold/oracles/) encodes them
implicitly:

- Diffusion MSE: the reference is aligned onto the prediction (AF3 algorithm 28), detached, then
  ``Σ w |x − x_aligned|² / (3 Σ w)``; the EDM weight is ``(σ² + σ_d²) / (σ σ_d)²``.
- Smooth lDDT: pairs with reference distance below ``cutoff`` (15 Å, protein), both atoms masked
  in, ``i ≠ j``; ``1 − mean(¼ Σ_{t ∈ {0.5, 1, 2, 4}} sigmoid(t − |Δd|))``.
- Distogram: representative-atom distances bucketized with ``bins − 1`` boundaries evenly spaced
  on ``[min_dist, max_dist]``; masked mean cross-entropy over pairs whose representative atoms are
  both resolved, ``i ≠ j``.
- pLDDT target: per-atom lDDT (thresholds 0.5, 1, 2, 4 Å) against the representative atoms within
  ``cutoff`` in the reference, the atom's own representative excluded; ``floor(lddt · bins)``
  clamped to the last bin; atoms without a scored neighbour are masked out.
- PAE target: token ``j``'s CA expressed in token ``i``'s (N, CA, C) frame, reference minus
  prediction, Euclidean norm; ``floor(d · bins / max_dist)`` clamped. A frame row is valid when its
  three atoms are resolved and non-collinear in the reference (|cos| < 0.9063, bond norms > 0.01 Å)
  and non-collinear in the prediction; the column needs a resolved CA.
- PDE target: ``|d_ref − d_pred|`` between representative atoms, binned like PAE, both resolved.
- Resolved: two-class cross-entropy per non-padding atom; class 0 means resolved.

Rollout tensors (``pred``, logits) have ``N = B · k`` rows for ``k`` rollouts of each of the
``B`` base examples, in the order ``b · k + j``; base-batch tensors are repeated inside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from oplm.fold.diffusion import weighted_rigid_align
from oplm.fold.trunk import fp32_autocast_off

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "LossWeights",
    "diffusion_mse",
    "distogram_loss",
    "edm_weight",
    "express_in_frames",
    "masked_mean",
    "pae_loss",
    "pair_distances",
    "pde_loss",
    "plddt_loss",
    "resolved_loss",
    "smooth_lddt",
    "token_frames",
]

_LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class LossWeights:
    """Spec §6.2 starting weights; ``pae`` weights the PAE sub-term inside the confidence composite."""

    diffusion_mse: float = 4.0
    smooth_lddt: float = 1.0
    distogram: float = 0.03
    confidence: float = 1e-4
    pae: float = 1.0


def edm_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    """``(σ² + σ_d²) / (σ σ_d)²`` per sample."""
    return (sigma.float() ** 2 + sigma_data**2) / (sigma.float() * sigma_data) ** 2


def pair_distances(x: Tensor, eps: float = 1e-8) -> Tensor:
    """``(..., n, 3) -> (..., n, n)`` Euclidean distances with a finite gradient at zero distance."""
    return torch.sqrt(((x[..., :, None, :] - x[..., None, :, :]) ** 2).sum(-1) + eps)


def masked_mean(values: Tensor, mask: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Mean of ``values`` over ``dims`` where ``mask``; exactly zero where the mask is empty."""
    m = mask.to(values.dtype)
    return (values * m).sum(dims) / m.sum(dims).clamp(min=1.0)


def _repeat(k: int, *tensors: Tensor) -> tuple[Tensor, ...]:
    return tuple(t.repeat_interleave(k, dim=0) for t in tensors)


def _gather_atoms(values: Tensor, index: Tensor) -> Tensor:
    """``values (N, A, ...)`` at ``index (N, L)`` -> ``(N, L, ...)``."""
    if values.ndim == 2:
        return torch.gather(values, 1, index)
    return torch.gather(values, 1, index[..., None].expand(-1, -1, values.shape[-1]))


def diffusion_mse(pred: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    """Weighted MSE / 3 per sample after aligning ``target`` onto ``pred`` (detached): ``(N,)``."""
    with fp32_autocast_off(pred.device):
        w = weights.float()
        aligned = weighted_rigid_align(target.float(), pred.detach().float(), w).detach()
        err = ((pred.float() - aligned) ** 2).sum(-1)
        return (err * w).sum(-1) / (3.0 * w.sum(-1)).clamp(min=1e-5)


def smooth_lddt(pred: Tensor, target: Tensor, mask: Tensor, *, cutoff: float = 15.0) -> Tensor:
    """``1 − smooth lDDT`` per sample ``(N,)``; 0 for a sample with fewer than two masked atoms.

    Each sample is compacted to its masked atoms before the ``(n, n)`` distance matrices are
    formed, so a padded 9216-atom crop costs its real atoms squared, not the budget squared.
    """
    with fp32_autocast_off(pred.device):
        out = []
        for n in range(pred.shape[0]):
            idx = mask[n].nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                out.append(pred.new_zeros(()))
                continue
            d_true = pair_distances(target[n, idx].float())
            d_pred = pair_distances(pred[n, idx].float())
            pair = (d_true < cutoff) & ~torch.eye(idx.numel(), dtype=torch.bool, device=pred.device)
            diff = (d_true - d_pred).abs()
            eps = sum(torch.sigmoid(t - diff) for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
            out.append(1.0 - (eps * pair).sum() / (pair.sum() + 1e-5))
        return torch.stack(out)


def distogram_loss(
    logits: Tensor,
    coords: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    min_dist: float,
    max_dist: float,
) -> Tensor:
    """Masked mean cross-entropy over representative-atom distance bins per example ``(B,)``."""
    with fp32_autocast_off(logits.device):
        B, L, bins = logits.shape[0], logits.shape[1], logits.shape[-1]
        reps = _gather_atoms(coords.float(), rep_atom_idx)
        rep_ok = _gather_atoms(resolved, rep_atom_idx) & token_mask
        boundaries = torch.linspace(min_dist, max_dist, bins - 1, device=logits.device)
        target = torch.bucketize(pair_distances(reps), boundaries)
        mask = rep_ok[:, :, None] & rep_ok[:, None, :]
        mask &= ~torch.eye(L, dtype=torch.bool, device=logits.device)
        ce = F.cross_entropy(logits.float().reshape(-1, bins), target.reshape(-1), reduction="none")
        return masked_mean(ce.reshape(B, L, L), mask, (-1, -2))


def token_frames(
    coords: Tensor, frame_atom_idx: Tensor, atom_valid: Tensor, token_mask: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token frames ``(N, L, 3, 3)`` (rows e1, e2, e3), centers ``(N, L, 3)`` and validity.

    Boltz-2 basis: ``w1 = (N − CA)/|·|``, ``w2 = (C − CA)/|·|``, ``e1 ∝ w1 + w2``, ``e2 ∝ w2 − w1``,
    ``e3 = e1 × e2``. Valid where the token is real, its three atoms are ``atom_valid``,
    ``|cos(N−CA, C−CA)| < 0.9063`` and both bonds are longer than 0.01 Å.
    """
    with fp32_autocast_off(coords.device):
        a, b, c = (_gather_atoms(coords.float(), frame_atom_idx[..., k]) for k in range(3))
        a_ok, b_ok, c_ok = (_gather_atoms(atom_valid, frame_atom_idx[..., k]) for k in range(3))
        v1, v2 = a - b, c - b
        n1, n2 = v1.norm(dim=-1, keepdim=True), v2.norm(dim=-1, keepdim=True)
        w1, w2 = v1 / (n1 + 1e-5), v2 / (n2 + 1e-5)
        e1 = (w1 + w2) / ((w1 + w2).norm(dim=-1, keepdim=True) + 1e-5)
        e2 = (w2 - w1) / ((w2 - w1).norm(dim=-1, keepdim=True) + 1e-5)
        e3 = torch.linalg.cross(e1, e2)
        cos = ((v1 / (n1 + 1e-6)) * (v2 / (n2 + 1e-6))).sum(-1).abs()
        ok = token_mask & a_ok & b_ok & c_ok & (cos < 0.9063) & (n1[..., 0] > 1e-2) & (n2[..., 0] > 1e-2)
        return torch.stack([e1, e2, e3], dim=-2), b, ok


def express_in_frames(frames: Tensor, centers: Tensor) -> Tensor:
    """Token ``j``'s center in token ``i``'s frame: ``(N, L, L, 3)`` indexed ``[n, i, j]``."""
    d = centers[:, None, :, :] - centers[:, :, None, :]
    return torch.einsum("bijd,bikd->bijk", d, frames)


def _binned_cross_entropy(logits: Tensor, values: Tensor, scale: float) -> Tensor:
    """Cross-entropy of ``floor(values · scale)`` (clamped to the last bin) against ``logits``."""
    bins = logits.shape[-1]
    target = torch.floor(values * scale).long().clamp(min=0, max=bins - 1)
    ce = F.cross_entropy(logits.float().reshape(-1, bins), target.reshape(-1), reduction="none")
    return ce.reshape(values.shape)


def plddt_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    atom_mask: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    cutoff: float = 15.0,
) -> Tensor:
    """Per-rollout pLDDT cross-entropy ``(N,)`` (per-atom lDDT target, ``bins`` from the logits)."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, a_mask, tok, rep_idx = _repeat(k, target, resolved, atom_mask, token_mask, rep_atom_idx)
        A = pred.shape[1]
        t_reps, p_reps = _gather_atoms(t_coords.float(), rep_idx), _gather_atoms(pred.float(), rep_idx)
        rep_ok = _gather_atoms(t_res, rep_idx) & tok
        d_true = torch.sqrt(((t_coords.float()[:, :, None] - t_reps[:, None]) ** 2).sum(-1) + 1e-8)
        d_pred = torch.sqrt(((pred.float()[:, :, None] - p_reps[:, None]) ** 2).sum(-1) + 1e-8)
        own = rep_idx[:, None, :] == torch.arange(A, device=pred.device)[None, :, None]
        scored = t_res[:, :, None] & rep_ok[:, None, :] & ~own & (d_true < cutoff)
        l1 = (d_true - d_pred).abs()
        score = sum((l1 < t).float() for t in _LDDT_THRESHOLDS) / len(_LDDT_THRESHOLDS)
        n_scored = scored.float().sum(-1)
        lddt = (1e-10 + (score * scored).sum(-1)) / (1e-10 + n_scored)
        ce = _binned_cross_entropy(logits, lddt, float(logits.shape[-1]))
        return masked_mean(ce, t_res & a_mask & (n_scored > 0), (-1,))


def pae_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    frame_atom_idx: Tensor,
    *,
    max_dist: float,
) -> Tensor:
    """Per-rollout PAE cross-entropy ``(N,)`` over valid (frame ``i``, resolved CA ``j``) pairs."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, tok, fidx = _repeat(k, target, resolved, token_mask, frame_atom_idx)
        true_frames, true_centers, true_ok = token_frames(t_coords, fidx, t_res, tok)
        pred_frames, pred_centers, pred_ok = token_frames(pred, fidx, torch.ones_like(t_res), tok)
        delta = express_in_frames(true_frames, true_centers) - express_in_frames(pred_frames, pred_centers)
        dist = torch.sqrt((delta**2).sum(-1) + 1e-8)
        ca_ok = _gather_atoms(t_res, fidx[..., 1]) & tok
        mask = (true_ok & pred_ok)[:, :, None] & ca_ok[:, None, :]
        return masked_mean(_binned_cross_entropy(logits, dist, logits.shape[-1] / max_dist), mask, (-1, -2))


def pde_loss(
    logits: Tensor,
    pred: Tensor,
    target: Tensor,
    resolved: Tensor,
    token_mask: Tensor,
    rep_atom_idx: Tensor,
    *,
    max_dist: float,
) -> Tensor:
    """Per-rollout PDE cross-entropy ``(N,)`` over pairs of resolved representative atoms."""
    with fp32_autocast_off(logits.device):
        k = pred.shape[0] // target.shape[0]
        t_coords, t_res, tok, rep_idx = _repeat(k, target, resolved, token_mask, rep_atom_idx)
        rep_ok = _gather_atoms(t_res, rep_idx) & tok
        err = (pair_distances(_gather_atoms(t_coords.float(), rep_idx)) - pair_distances(_gather_atoms(pred.float(), rep_idx))).abs()
        mask = rep_ok[:, :, None] & rep_ok[:, None, :]
        return masked_mean(_binned_cross_entropy(logits, err, logits.shape[-1] / max_dist), mask, (-1, -2))


def resolved_loss(logits: Tensor, resolved: Tensor, atom_mask: Tensor) -> Tensor:
    """Per-rollout two-class cross-entropy ``(N,)``; class 0 is "resolved"."""
    with fp32_autocast_off(logits.device):
        k = logits.shape[0] // resolved.shape[0]
        res, am = _repeat(k, resolved, atom_mask)
        ce = F.cross_entropy(logits.float().reshape(-1, 2), (~res).long().reshape(-1), reduction="none")
        return masked_mean(ce.reshape(res.shape), am, (-1,))
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_losses.py tests/fold/test_configuration.py -q`
Expected: PASS. If a parity test is off by more than the tolerance, compare the masks first (the oracle's `token_disto_mask`, `frame_resolved_mask`, and `compute_frame_pred`'s collinear mask) before touching the math.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/losses.py src/oplm/fold/configuration_fold.py tests/fold/test_losses.py tests/fold/conftest.py tests/fold/data/conftest.py tests/fold/helpers.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/losses.py src/oplm/fold/configuration_fold.py tests/fold/test_losses.py tests/fold/conftest.py tests/fold/data/conftest.py tests/fold/helpers.py
git commit -m "feat(fold): stage-1 losses with Boltz-2 parity and finite degenerate cases"
```

---

### Task 5: Training branch of `OplmForFolding.forward`

**Files:**
- Modify: `src/oplm/fold/modeling_fold.py`, `src/oplm/fold/diffusion.py` (`StructureHead.expand`), `src/oplm/fold/lm_shim.py` (`mask_lm_inputs`), `src/oplm/fold/predict.py` (narrow the return type)
- Test: `tests/fold/test_modeling_train.py` (new), `tests/fold/test_diffusion.py` (append), `tests/fold/test_lm_shim.py` (append)

**Interfaces:**
- Consumes: Task 4's losses; `StructureHead.prepare/denoise/sample`, `center_random_augmentation`, `ConfidenceHead` (M1); `FoldTargets`.
- Produces: `TrainSettings(diffusion_samples=8, diffusion_microbatch=None, confidence_rollout_steps=20, weights=LossWeights())` (frozen dataclass); `FoldTrainOutput(loss, metrics, tokens, chains)`; `draw_num_loops(mean, min_loops, max_loops) -> int`; `mask_lm_inputs(features, fraction, mask_id) -> FoldFeatures`; `StructureHead.expand(inp, num_samples) -> DenoiserInputs`; `OplmForFolding.forward(features, *, targets=None, train=None, ...) -> FoldOutput | FoldTrainOutput`. Task 7's `FoldTask.step` calls `model(batch.features, targets=batch.targets, train=settings)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fold/test_lm_shim.py` (import `mask_lm_inputs`, `VOCAB` from `oplm.model.tokenization_oplm`):

```python
def test_lm_input_masking_touches_only_inner_positions() -> None:
    """Review Focus 5: BOS, EOS and padding are never masked; fraction 0 is the identity."""
    f = featurize([ChainSpec("MKVLAG", "A"), ChainSpec("WW", "B")])
    torch.manual_seed(0)
    masked = mask_lm_inputs(f, 1.0, VOCAB["<mask>"])
    ids, mask = masked.lm_input_ids, f.lm_attention_mask.bool()
    lengths = mask.sum(1)
    for row in range(ids.shape[0]):
        assert ids[row, 0] == VOCAB["<cls>"] and ids[row, lengths[row] - 1] == VOCAB["<eos>"]
        assert (ids[row, 1 : lengths[row] - 1] == VOCAB["<mask>"]).all()
        assert torch.equal(ids[row, lengths[row] :], f.lm_input_ids[row, lengths[row] :])
    assert torch.equal(mask_lm_inputs(f, 0.0, VOCAB["<mask>"]).lm_input_ids, f.lm_input_ids)
    assert masked.token_mask is f.token_mask  # everything but the ids is shared
```

Append to `tests/fold/test_diffusion.py` (`InputsEmbedder`, `featurize`, `ChainSpec`, `StructureHead` and `tiny_fold_config` are already imported there):

```python
def test_expand_matches_prepare_with_num_samples() -> None:
    cfg = tiny_fold_config()
    torch.manual_seed(0)
    f = featurize([ChainSpec("MKVLAG", "A")])
    emb = InputsEmbedder(cfg).eval()(f)
    head = StructureHead(cfg).eval()
    z_trunk = torch.randn(1, f.num_tokens, f.num_tokens, cfg.pair_width)
    with torch.no_grad():
        base = head.prepare(
            s_inputs=emb.s_inputs, z_trunk=z_trunk, relpos=emb.relpos, atom_features=emb.atom_features,
            rope=emb.rope, atom_mask=f.atom_mask, atom_to_token=f.atom_to_token,
            token_mask=f.token_mask, num_samples=1,
        )
        three = head.expand(base, 3)
        direct = head.prepare(
            s_inputs=emb.s_inputs, z_trunk=z_trunk, relpos=emb.relpos, atom_features=emb.atom_features,
            rope=emb.rope, atom_mask=f.atom_mask, atom_to_token=f.atom_to_token,
            token_mask=f.token_mask, num_samples=3,
        )
    assert three.s_inputs.shape[0] == 3 and three.c.shape[0] == 3 and three.z.shape[0] == 1
    torch.testing.assert_close(three.atom_mask, base.atom_mask.repeat_interleave(3, 0))
    torch.testing.assert_close(three.c, direct.c)
    torch.testing.assert_close(three.z, direct.z)
    assert head.expand(base, 1) is base
```

Create `tests/fold/test_modeling_train.py`:

```python
"""The training branch of OplmForFolding.forward: losses, gradients, microbatching, masking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.fold.data.collate import collate_examples
from oplm.fold.data.featurize import featurize_complex
from oplm.fold.losses import LossWeights
from oplm.fold.modeling_fold import (
    FoldOutput,
    FoldTrainOutput,
    OplmForFolding,
    TrainSettings,
    draw_num_loops,
)
from tests.fold.helpers import tiny_fold_config, tiny_lm

if TYPE_CHECKING:
    from collections.abc import Callable

    from oplm.fold.data.collate import FoldBatch
    from oplm.fold.data.complex import MolecularComplex

_TINY: dict[str, Any] = dict(
    plddt_bins=10,
    pae_bins=8,
    pde_bins=8,
    confidence_dist_bins=5,
    distogram_bins=8,
    inference_num_steps=2,
    inference_num_loops=2,
    lm_hidden_size=32,
    lm_num_hidden_states=3,
)
_SETTINGS = TrainSettings(diffusion_samples=3, diffusion_microbatch=2, confidence_rollout_steps=2)


@pytest.fixture(scope="module")
def batch(parsed: Callable[[str], MolecularComplex]) -> FoldBatch:
    return collate_examples(
        [featurize_complex(parsed("1crn")), featurize_complex(parsed("1ejg"))], crop_size=64
    )


def _model(**over: Any) -> OplmForFolding:
    torch.manual_seed(0)
    model = OplmForFolding(tiny_fold_config(**{**_TINY, **over}))
    model.attach_lm(tiny_lm())
    return model.train()


def test_training_forward_returns_loss_metrics_and_counts(batch: FoldBatch) -> None:
    model = _model()
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    assert isinstance(out, FoldTrainOutput)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.requires_grad
    assert out.tokens == int(batch.features.token_mask.sum()) and out.chains == 2
    assert set(out.metrics) == {
        "diffusion_mse", "smooth_lddt", "distogram", "confidence", "plddt", "pde", "pae",
        "resolved", "loops", "sigma_mean",
    }
    assert 1 <= out.metrics["loops"] <= 6 and all(v == v for v in out.metrics.values())


def test_gradients_reach_every_head_parameter_and_none_of_the_lm(batch: FoldBatch) -> None:
    model = _model()
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    out.loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.lm.parameters())
    nonzero = {n for n, p in model.named_parameters() if p.grad.abs().sum() > 0}
    for prefix in ("language_model.", "folding_trunk.", "distogram_head.", "structure_head.", "confidence_head.pae_head"):
        assert any(n.startswith(prefix) for n in nonzero), prefix


def test_microbatch_does_not_change_the_loss(batch: FoldBatch) -> None:
    """Review Focus 4: chunks of 3, 3, 2 give the same loss as one chunk of 8."""
    model = _model()
    whole = TrainSettings(diffusion_samples=8, diffusion_microbatch=None, confidence_rollout_steps=2)
    chunked = TrainSettings(diffusion_samples=8, diffusion_microbatch=3, confidence_rollout_steps=2)
    torch.manual_seed(7)
    a = model(batch.features, targets=batch.targets, train=whole, num_loops=1)
    torch.manual_seed(7)
    b = model(batch.features, targets=batch.targets, train=chunked, num_loops=1)
    torch.testing.assert_close(a.loss, b.loss, atol=1e-5, rtol=1e-5)
    a.loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    b.loss.backward()
    for g, p in zip(grads, model.parameters(), strict=True):
        torch.testing.assert_close(g, p.grad, atol=1e-5, rtol=1e-4)


def test_weights_scale_their_terms(batch: FoldBatch) -> None:
    model = _model()
    torch.manual_seed(1)
    base = model(batch.features, targets=batch.targets, train=_SETTINGS, num_loops=1)
    torch.manual_seed(1)
    only_disto = model(
        batch.features,
        targets=batch.targets,
        train=TrainSettings(3, 2, 2, LossWeights(diffusion_mse=0, smooth_lddt=0, distogram=1.0, confidence=0)),
        num_loops=1,
    )
    torch.testing.assert_close(only_disto.loss, torch.tensor(base.metrics["distogram"]), atol=1e-5, rtol=1e-5)


def test_training_forward_with_no_resolved_atoms_is_finite(batch: FoldBatch) -> None:
    """Review Focus 1."""
    model = _model()
    targets = batch.targets
    targets = type(targets)(coords=targets.coords, resolved=torch.zeros_like(targets.resolved), bfactor=targets.bfactor, frame_atom_idx=targets.frame_atom_idx)
    out = model(batch.features, targets=targets, train=_SETTINGS)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_eval_forward_ignores_train_settings(batch: FoldBatch) -> None:
    """Review Focus 5: no targets -> the inference path, whatever ``train`` says."""
    model = _model().eval()
    with torch.no_grad():
        out = model(batch.features, train=_SETTINGS, generator=torch.Generator().manual_seed(0))
        again = model(batch.features, generator=torch.Generator().manual_seed(0))
    assert isinstance(out, FoldOutput)
    torch.testing.assert_close(out.coords, again.coords)


def test_num_loops_is_poisson_clamped() -> None:
    torch.manual_seed(0)
    draws = [draw_num_loops(3.0, 1, 6) for _ in range(500)]
    assert min(draws) >= 1 and max(draws) <= 6 and 2.0 < sum(draws) / len(draws) < 4.0
    assert draw_num_loops(0.0, 1, 6) == 1


def test_train_mode_draws_loops_and_eval_uses_the_config(batch: FoldBatch) -> None:
    model = _model(recurrence_min_loops=2, recurrence_max_loops=2)
    out = model(batch.features, targets=batch.targets, train=_SETTINGS)
    assert out.metrics["loops"] == 2.0
    model.eval()
    with torch.no_grad():
        inference = model(batch.features, return_intermediates=True)
    assert len(inference.intermediates["states"]) == model.config.inference_num_loops
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/fold/test_modeling_train.py tests/fold/test_lm_shim.py tests/fold/test_diffusion.py -q`
Expected: FAIL (`ImportError` on `TrainSettings`, `mask_lm_inputs`; no `expand`).

- [ ] **Step 3: Implement `mask_lm_inputs` and `StructureHead.expand`**

`src/oplm/fold/lm_shim.py` (add `from dataclasses import replace` and `VOCAB` is not needed; the mask id comes from the caller):

```python
def mask_lm_inputs(features: FoldFeatures, fraction: float, mask_id: int) -> FoldFeatures:
    """Replace a ``fraction`` of inner residue ids on every LM row with ``mask_id`` (spec §6.1).

    BOS (position 0), EOS (the last attended position) and padding are never masked. Draws from
    the global torch RNG, which the Trainer seeds and checkpoints per rank. ``fraction <= 0``
    returns ``features`` unchanged.
    """
    if fraction <= 0:
        return features
    ids, attended = features.lm_input_ids, features.lm_attention_mask.bool()
    inner = attended.clone()
    inner[:, 0] = False
    last = attended.sum(dim=1) - 1
    inner[torch.arange(ids.shape[0], device=ids.device), last] = False
    draw = torch.rand(ids.shape, device=ids.device) < fraction
    masked = torch.where(inner & draw, torch.full_like(ids, mask_id), ids)
    return replace(features, lm_input_ids=masked)
```

`src/oplm/fold/diffusion.py`: split `prepare` so the expansion is reusable.

```python
    def prepare(self, *, s_inputs, z_trunk, relpos, atom_features, rope, atom_mask, atom_to_token, token_mask, num_samples) -> DenoiserInputs:
        """Condition the pair, embed the atoms, then expand the per-sample tensors ``num_samples`` times."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples!r}")
        with fp32_autocast_off(s_inputs.device):
            z = self.conditioning.pair(z_trunk, relpos)
            c = self.atom_encoder.embed(atom_features)
            flex = resolve_attention_backend(z, self.config.attention_backend) == "flex"
            base = DenoiserInputs(
                s_inputs=s_inputs, z=z, c=c, rope=rope, atom_mask=atom_mask, atom_to_token=atom_to_token,
                token_mask=token_mask,
                token_block_mask=pair_bias_block_mask(token_mask) if flex else None,
                atom_block_mask=sliding_window_block_mask(atom_mask, self.config.atom_window // 2) if flex else None,
            )
        return self.expand(base, num_samples)

    def expand(self, inp: DenoiserInputs, num_samples: int) -> DenoiserInputs:
        """Repeat every per-sample tensor of a one-sample ``DenoiserInputs`` ``num_samples`` times.

        ``z`` stays at the base batch (the blocks broadcast it); the block masks are rebuilt for
        the expanded masks. Returns ``inp`` itself for ``num_samples == 1``.
        """
        if num_samples == 1:
            return inp
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples!r}")

        def rep(t: Tensor) -> Tensor:
            return t.repeat_interleave(num_samples, dim=0)

        flex = resolve_attention_backend(inp.z, self.config.attention_backend) == "flex"
        atom_mask_s, token_mask_s = rep(inp.atom_mask), rep(inp.token_mask)
        return DenoiserInputs(
            s_inputs=rep(inp.s_inputs), z=inp.z, c=rep(inp.c), rope=(rep(inp.rope[0]), rep(inp.rope[1])),
            atom_mask=atom_mask_s, atom_to_token=rep(inp.atom_to_token), token_mask=token_mask_s,
            token_block_mask=pair_bias_block_mask(token_mask_s) if flex else None,
            atom_block_mask=sliding_window_block_mask(atom_mask_s, self.config.atom_window // 2) if flex else None,
        )
```

Run `tests/fold/test_diffusion.py tests/fold/test_modeling.py` to confirm the inference values are unchanged before moving on.

- [ ] **Step 4: Implement the training branch**

In `src/oplm/fold/modeling_fold.py` add the imports (`from dataclasses import dataclass`, `from torch.utils.checkpoint import checkpoint`, `from oplm.fold.diffusion import StructureHead, center_random_augmentation`, `from oplm.fold.lm_shim import ..., mask_lm_inputs`, `from oplm.fold.losses import LossWeights, diffusion_mse, distogram_loss, edm_weight, pae_loss, pde_loss, plddt_loss, resolved_loss, smooth_lddt`, `from oplm.fold.trunk import ..., fp32_autocast_off`, and `FoldTargets` under `TYPE_CHECKING`), extend `__all__` with `"FoldTrainOutput", "TrainSettings", "draw_num_loops"`, and add:

```python
@dataclass(frozen=True)
class TrainSettings:
    """Per-run training knobs the forward pass needs (the ``fold:`` block; addendum §5)."""

    diffusion_samples: int = 8
    diffusion_microbatch: int | None = None  # samples per denoiser call; None = all at once
    confidence_rollout_steps: int = 20
    weights: LossWeights = LossWeights()

    def __post_init__(self) -> None:
        if self.diffusion_samples < 1 or self.confidence_rollout_steps < 1:
            raise ValueError("diffusion_samples and confidence_rollout_steps must be >= 1")
        if self.diffusion_microbatch is not None and self.diffusion_microbatch < 1:
            raise ValueError("diffusion_microbatch must be >= 1 or None")


@dataclass
class FoldTrainOutput:
    """What the training branch returns: the loss to backpropagate and the Trainer's accounting."""

    loss: Tensor
    metrics: dict[str, float]
    tokens: int
    chains: int


def draw_num_loops(mean: float, min_loops: int, max_loops: int) -> int:
    """One clamped Poisson draw from the global RNG (spec §5.1: mean 3, clamped to 1–6)."""
    draw = int(torch.poisson(torch.tensor(float(mean))).item())
    return max(min_loops, min(max_loops, draw))
```

Replace `forward` with the factored version (the inference tail is the existing code verbatim):

```python
    def forward(
        self,
        features: FoldFeatures,
        *,
        targets: FoldTargets | None = None,
        train: TrainSettings | None = None,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_samples: int | None = None,
        num_steps: int | None = None,
        z0: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
    ) -> FoldOutput | FoldTrainOutput:
        """Predict a structure, or with ``targets`` run the training branch (spec §6.1).

        Args:
            features: Output of :func:`oplm.fold.featurize` or a collated batch.
            targets: Supervision aligned with ``features``; selects the training branch, which
                returns :class:`FoldTrainOutput`. ``train`` supplies its settings (defaults
                otherwise) and is ignored without ``targets``.
            train: Sample count, microbatch, rollout steps and loss weights.
            lm_hidden_states: ``(B, L, K, D)`` precomputed LM states (skips the attached LM).
            num_loops: Recurrence iterations; default ``config.inference_num_loops`` in eval mode
                and a clamped Poisson draw in train mode.
            num_samples: Diffusion samples (inference); default ``config.inference_num_samples``.
            num_steps: Sampler steps before the sigma cap; default ``config.inference_num_steps``.
            z0: Initial pair state; drawn from ``generator`` when ``None``.
            generator: RNG for the initial state and the samplers (deterministic when set);
                must live on the model's device.
            return_intermediates: Also return the stage tensors used by the parity tests.

        Raises:
            RuntimeError: No LM is attached and ``lm_hidden_states`` is ``None``.
        """
        cfg = self.config
        f = features.to(self.device)
        if self.training:
            f = mask_lm_inputs(f, cfg.lm_input_mask_fraction, cfg.lm_mask_token_id)
        if lm_hidden_states is None:
            if self._lm is None:
                raise RuntimeError(
                    "no language model attached: call attach_lm(), load with lm_name_or_path, "
                    "or pass lm_hidden_states"
                )
            lm_device = next(self._lm.parameters()).device
            lm_dtype = torch.bfloat16 if lm_device.type == "cuda" else None
            lm_hidden_states = frozen_lm_hidden_states(self._lm, f.to(lm_device), dtype=lm_dtype)
        lm_hidden_states = lm_hidden_states.to(self.device)
        if num_loops is not None:
            loops = num_loops
        elif self.training:
            loops = draw_num_loops(
                cfg.recurrence_poisson_mean, cfg.recurrence_min_loops, cfg.recurrence_max_loops
            )
        else:
            loops = cfg.inference_num_loops
        on_cuda = self.device.type == "cuda"
        if on_cuda:
            ensure_flex_recompile_limit()
        flex = on_cuda and resolve_attention_backend(f.ref_pos, cfg.attention_backend) == "flex"
        atom_block_mask = (
            sliding_window_block_mask(f.atom_mask, cfg.atom_window // 2) if flex else None
        )
        pair_mask = f.token_mask[:, :, None].float() * f.token_mask[:, None, :].float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_cuda):
            emb = self.input_embedder(f, block_mask=atom_block_mask)
            lm_z = self.language_model(lm_hidden_states)
            if z0 is None:
                z0 = self.parcae.init_state(emb.z_init, generator=generator)
            drop = cfg.pair_dropout if self.training else 0.0

            def inject(_t: int) -> Tensor:
                lm_t = F.dropout(lm_z, drop, training=True) if drop > 0 else lm_z
                return emb.z_init + self.lm_encoder(lm_t.to(emb.z_init.dtype), pair_mask)

            z, states = self.parcae.run(
                self.folding_trunk,
                inject,
                z0=z0.to(emb.z_init.dtype),
                pair_mask=pair_mask,
                num_loops=loops,
                grad_loops=cfg.recurrence_grad_loops if self.training else None,
                return_states=return_intermediates,
            )
            z = self.parcae.readout(z, pair_mask)
        z = z.float()
        distogram_logits = symmetrized_distogram(self.distogram_head, z)
        if targets is not None:
            return self._train_branch(
                f, targets, train or TrainSettings(), emb, z, distogram_logits, loops, generator
            )
        samples = cfg.inference_num_samples if num_samples is None else num_samples
        inp = self.structure_head.prepare(...)        # unchanged from milestone 1
        ...                                            # sampler, confidence head, intermediates, FoldOutput
```

and the branch:

```python
    def _train_branch(
        self,
        f: FoldFeatures,
        targets: FoldTargets,
        s: TrainSettings,
        emb: InputsEmbedding,
        z: Tensor,
        distogram_logits: Tensor,
        loops: int,
        generator: torch.Generator | None,
    ) -> FoldTrainOutput:
        """Spec §6.1 steps 3–6 on the shared prefix (see the module docstring for the design)."""
        cfg, w = self.config, s.weights
        device = z.device
        B = z.shape[0]
        t = targets.to(device)
        with fp32_autocast_off(device):
            resolved = t.resolved & f.atom_mask
            disto = distogram_loss(
                distogram_logits, t.coords, resolved, f.token_mask, f.distogram_atom_idx,
                min_dist=cfg.distogram_min_dist, max_dist=cfg.distogram_max_dist,
            )
            base = self.structure_head.prepare(
                s_inputs=emb.s_inputs, z_trunk=z, relpos=emb.relpos, atom_features=emb.atom_features,
                rope=emb.rope, atom_mask=f.atom_mask, atom_to_token=f.atom_to_token,
                token_mask=f.token_mask, num_samples=1,
            )
            # Diffusion: S samples per example, rows b·S + j; one denoiser call per microbatch
            # of samples, checkpointed so only one chunk's activations are live at a time.
            S = s.diffusion_samples
            micro = S if s.diffusion_microbatch is None else min(s.diffusion_microbatch, S)
            x_gt = t.coords.float().repeat_interleave(S, 0)
            res_s = resolved.repeat_interleave(S, 0)
            w_s = res_s.float()
            x_aug = center_random_augmentation(x_gt, w_s)
            sigma = cfg.sigma_data * torch.exp(
                cfg.train_noise_log_mean + cfg.train_noise_log_std * torch.randn(B * S, device=device)
            )
            x_noisy = x_aug + sigma[:, None, None] * torch.randn_like(x_aug)
            x_den = torch.zeros_like(x_noisy)
            use_ckpt = self.training and torch.is_grad_enabled() and micro < S
            for start in range(0, S, micro):
                m = min(micro, S - start)
                inp = self.structure_head.expand(base, m)
                index = (
                    torch.arange(B, device=device)[:, None] * S
                    + torch.arange(start, start + m, device=device)[None]
                ).reshape(-1)

                def run(x: Tensor, tt: Tensor, inp: DenoiserInputs = inp) -> Tensor:
                    return self.structure_head.denoise(x, tt, inp)

                pred = (
                    checkpoint(run, x_noisy[index], sigma[index], use_reentrant=False)
                    if use_ckpt
                    else run(x_noisy[index], sigma[index])
                )
                x_den = x_den.index_put((index,), pred)
            mse = diffusion_mse(x_den, x_aug, w_s)
            lddt = smooth_lddt(x_den, x_aug, res_s)
            diffusion_term = (edm_weight(sigma, cfg.sigma_data) * mse).mean()
            lddt_term = lddt.mean()
            # Confidence: a no-grad rollout, then the head on detached trunk outputs.
            with torch.no_grad():
                rollout = self.structure_head.sample(
                    base, num_steps=s.confidence_rollout_steps, generator=generator
                )
            conf = self.confidence_head(
                s_inputs=emb.s_inputs.detach(), z=z.detach(), relpos=emb.relpos.detach(),
                bonds=emb.bonds.detach(), coords=rollout, distogram_atom_idx=f.distogram_atom_idx,
                token_mask=f.token_mask, atom_to_token=f.atom_to_token, atom_mask=f.atom_mask,
                asym_id=f.asym_id,
            )
            plddt = plddt_loss(conf.plddt_logits, rollout, t.coords, resolved, f.atom_mask, f.token_mask, f.distogram_atom_idx)
            pde = pde_loss(conf.pde_logits, rollout, t.coords, resolved, f.token_mask, f.distogram_atom_idx, max_dist=cfg.pae_max_dist)
            pae = pae_loss(conf.pae_logits, rollout, t.coords, resolved, f.token_mask, t.frame_atom_idx, max_dist=cfg.pae_max_dist)
            res = resolved_loss(conf.resolved_logits, resolved, f.atom_mask)
            confidence = (plddt + pde + res + w.pae * pae).mean()
            loss = (
                w.diffusion_mse * diffusion_term
                + w.smooth_lddt * lddt_term
                + w.distogram * disto.mean()
                + w.confidence * confidence
            )
        metrics = {
            "diffusion_mse": float(diffusion_term), "smooth_lddt": float(lddt_term),
            "distogram": float(disto.mean()), "confidence": float(confidence),
            "plddt": float(plddt.mean()), "pde": float(pde.mean()), "pae": float(pae.mean()),
            "resolved": float(res.mean()), "loops": float(loops), "sigma_mean": float(sigma.mean()),
        }
        return FoldTrainOutput(loss=loss, metrics=metrics, tokens=int(f.token_mask.sum()), chains=B)
```

Import `DenoiserInputs` and `InputsEmbedding` under `TYPE_CHECKING` (from `oplm.fold.diffusion` and `oplm.fold.atoms`). Update the module docstring with the training-branch design: the shared prefix, in-model masking, per-rank Poisson loops, microbatched checkpointed denoising, the no-grad rollout, and that the branch runs whenever `targets` is given (train-mode behaviour such as dropout, masking and the Poisson draw still follows `self.training`).

`src/oplm/fold/predict.py`, in `fold()` after the model call: `assert isinstance(output, FoldOutput), "fold() never passes targets"` with `FoldOutput` imported at runtime from `oplm.fold.modeling_fold`.

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold -q -m "not slow"`
Expected: PASS (M1 inference tests unchanged; the new training tests pass). If `OPLM_FOLD_FIXTURES` is available locally (`~/fold-fixtures/25479`), also run `OPLM_FOLD_FIXTURES=~/fold-fixtures/25479 .venv/bin/python -m pytest tests/fold/test_parity.py -q` and expect 37 passed.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/modeling_fold.py src/oplm/fold/diffusion.py src/oplm/fold/lm_shim.py src/oplm/fold/predict.py tests/fold/test_modeling_train.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/modeling_fold.py src/oplm/fold/diffusion.py src/oplm/fold/lm_shim.py src/oplm/fold/predict.py tests/fold/test_modeling_train.py tests/fold/test_diffusion.py tests/fold/test_lm_shim.py
git commit -m "feat(fold): training branch of OplmForFolding.forward (spec §6.1)"
```

---
### Task 6: The `fold:` run-config block

**Files:**
- Modify: `src/oplm/config.py` (`FoldRunConfig`, `OplmConfig.fold`, `load_config`, `serialize_config`)
- Test: `tests/training/test_config.py` (append)

**Interfaces:**
- Consumes: `OplmConfig`, `load_config`, `serialize_config` (existing).
- Produces: `FoldRunConfig(lm="esmc:esmc_600m", lm_revision=None, lm_max_chain_length=1024, model={}, crop_size=384, contiguous_prob=0.5, atoms_per_token_budget=24, diffusion_samples=8, diffusion_microbatch=None, confidence_rollout_steps=20, loss_diffusion_mse=4.0, loss_smooth_lddt=1.0, loss_distogram=0.03, loss_confidence=1e-4, loss_pae=1.0)`; `OplmConfig.fold: FoldRunConfig | None = None`; `load_config` returns it populated from YAML/CLI (`fold.crop_size=256`, `fold.model.pair_width=128`); `serialize_config` round-trips it. Task 7 reads `cfg.fold`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/training/test_config.py`:

```python
def test_fold_block_defaults_to_none_and_parses_from_overrides(tmp_path: Path) -> None:
    from oplm.config import FoldRunConfig, load_config, serialize_config

    assert load_config([]).fold is None
    cfg = load_config(["fold.crop_size=256", "fold.model.pair_width=128", "fold.diffusion_microbatch=4"])
    assert isinstance(cfg.fold, FoldRunConfig)
    assert cfg.fold.crop_size == 256 and cfg.fold.diffusion_microbatch == 4
    assert cfg.fold.model == {"pair_width": 128}
    assert cfg.fold.lm == "esmc:esmc_600m" and cfg.fold.loss_confidence == 1e-4
    path = tmp_path / "run.yaml"
    path.write_text(serialize_config(cfg))
    assert "fold:" in path.read_text()
    reloaded = load_config(["--config", str(path)])
    assert reloaded.fold == cfg.fold


def test_fold_block_validates_its_fields() -> None:
    from oplm.config import FoldRunConfig

    with pytest.raises(ValueError, match="crop_size"):
        FoldRunConfig(crop_size=100)
    with pytest.raises(ValueError, match="contiguous_prob"):
        FoldRunConfig(contiguous_prob=1.5)
    with pytest.raises(ValueError, match="diffusion_microbatch"):
        FoldRunConfig(diffusion_microbatch=0)
    with pytest.raises(ValueError, match="loss_"):
        FoldRunConfig(loss_distogram=-1.0)
    with pytest.raises(ValueError, match="lm"):
        FoldRunConfig(lm="")
```

Run: `.venv/bin/python -m pytest tests/training/test_config.py -q -k fold_block`
Expected: FAIL with `ImportError: FoldRunConfig`.

- [ ] **Step 2: Implement**

`src/oplm/config.py`, after `DataConfig`:

```python
@dataclass
class FoldRunConfig:
    """The ``fold:`` block of a structure-head run (docs/FOLD.md §9; addendum §5).

    ``model`` holds ``FoldConfig`` overrides (``pair_width``, ``trunk_blocks``, ...); the
    ``lm_*`` shape fields are inferred from the loaded LM unless given. Loss weights are spec
    §6.2's starting values; ``loss_pae`` weights the PAE sub-term of the confidence composite.
    """

    lm: str = "esmc:esmc_600m"
    lm_revision: str | None = None
    lm_max_chain_length: int = 1024
    model: Any = field(default_factory=dict)
    crop_size: int = 384
    contiguous_prob: float = 0.5
    atoms_per_token_budget: int = 24
    diffusion_samples: int = 8
    diffusion_microbatch: int | None = None
    confidence_rollout_steps: int = 20
    loss_diffusion_mse: float = 4.0
    loss_smooth_lddt: float = 1.0
    loss_distogram: float = 0.03
    loss_confidence: float = 1e-4
    loss_pae: float = 1.0

    def __post_init__(self) -> None:
        """Validate the block (crop multiples, probabilities, counts, weights)."""
        if not self.lm or not str(self.lm).strip():
            raise ValueError("fold.lm must name a language model (esmc:<name>, a path or a Hub id)")
        if self.lm_max_chain_length < 1:
            raise ValueError(f"fold.lm_max_chain_length must be >= 1, got {self.lm_max_chain_length}")
        if self.crop_size < 128 or self.crop_size % 128:
            raise ValueError(f"fold.crop_size must be a positive multiple of 128, got {self.crop_size}")
        if not 0.0 <= self.contiguous_prob <= 1.0:
            raise ValueError(f"fold.contiguous_prob must be in [0, 1], got {self.contiguous_prob}")
        if self.atoms_per_token_budget < 1:
            raise ValueError(f"fold.atoms_per_token_budget must be >= 1, got {self.atoms_per_token_budget}")
        if self.diffusion_samples < 1:
            raise ValueError(f"fold.diffusion_samples must be >= 1, got {self.diffusion_samples}")
        if self.diffusion_microbatch is not None and self.diffusion_microbatch < 1:
            raise ValueError(f"fold.diffusion_microbatch must be >= 1 or null, got {self.diffusion_microbatch}")
        if self.confidence_rollout_steps < 1:
            raise ValueError(f"fold.confidence_rollout_steps must be >= 1, got {self.confidence_rollout_steps}")
        for name in ("loss_diffusion_mse", "loss_smooth_lddt", "loss_distogram", "loss_confidence", "loss_pae"):
            if getattr(self, name) < 0:
                raise ValueError(f"fold.{name} must be >= 0, got {getattr(self, name)}")
```

`OplmConfig` gains `fold: FoldRunConfig | None = None` after `data`. In `load_config`, after `cfg: OplmConfig = OmegaConf.to_object(base)`, add:

```python
    # ``fold`` is an optional structured node: ``to_object`` yields FoldRunConfig or None.
    # A plain dict can only appear if OmegaConf typed the node loosely; normalize it.
    if isinstance(cfg.fold, dict):
        cfg.fold = FoldRunConfig(**cfg.fold)
```

`serialize_config`: add `"fold": asdict(cfg.fold)` to `config_dict` when `cfg.fold is not None`, and note in its docstring that the `fold` block round-trips like `train` and `data`. Add `FoldRunConfig` to the module's public names (it has no `__all__`; the import in Task 7 is `from oplm.config import FoldRunConfig`).

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest tests/training/test_config.py tests/test_cli.py -q`
Expected: PASS, including the existing MLM round-trip tests (a `None` fold block is omitted from the YAML).

- [ ] **Step 4: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/config.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/config.py tests/training/test_config.py
git commit -m "feat(config): fold run-config block"
```

---

### Task 7: `FoldTask`

**Files:**
- Create: `src/oplm/fold/training.py`
- Modify: `src/oplm/training/task.py` (`step(batch: Any)`), `src/oplm/training/mup.py` and `src/oplm/training/optim.py` (`getattr(config, "mup_enable", False)`), `src/oplm/fold/__init__.py` (lazy `FoldTask`), `tests/fold/helpers.py` (+ `tiny_fold_model_overrides`, `tiny_fold_run_cfg`)
- Test: `tests/fold/test_training.py` (new)

**Interfaces:**
- Consumes: `FoldRunConfig` (Task 6), `TrainSettings`/`FoldTrainOutput` (Task 5), `load_frozen_lm` (Task 2), `StructureDataset`, `collate_rows`, `InterleavedDataset`, `parse_train_configs`, `resolve_initialization_source`, `StepResult`, `TrainerCallback`.
- Produces: `require_fold(cfg) -> FoldRunConfig` (raises `ValueError` without a block), `build_fold_config(run, *, lm=None) -> FoldConfig`, `load_initial_head(source, target) -> OplmForFolding`, `FoldTask()` with `build_model`, `build_dataloader`, `step`, `flops_per_token`, `settings` (the `TrainSettings` built in `build_model`) and `last_batch_ids`; `PeakMemoryCallback` (records `train/step_time_s` and CUDA peak memory per log; Task 10 uses it). `Trainer(cfg, task=FoldTask())` trains a fold run.

- [ ] **Step 1: Core seam changes**

`src/oplm/training/task.py`: change `TrainTask.step`'s and `MLMTask.step`'s `batch` annotation to `Any` (docstring: "the dataloader's batch: a dict for MLM, a `FoldBatch` for folding"). `src/oplm/training/mup.py::mup_lr_multiplier`: `if not getattr(config, "mup_enable", False): return 1.0` (docstring: "a config without μP knobs, such as `FoldConfig`, takes the non-μP path"). `src/oplm/training/optim.py::build_optimizers`: the Muon guard reads `getattr(model.config, "mup_enable", False)` instead of the cast. Run `.venv/bin/python -m pytest tests/training/test_optim.py tests/training/test_mup.py -q` and expect PASS.

- [ ] **Step 2: Test helpers**

Append to `tests/fold/helpers.py`:

```python
TINY_FOLD_MODEL: dict[str, Any] = dict(
    pair_width=32,
    token_width=64,
    atom_width=32,
    atom_encoder_blocks=1,
    atom_encoder_heads=2,
    uid_rope_pairs=2,
    trunk_blocks=1,
    lm_encoder_blocks=1,
    coda_blocks=1,
    diffusion_blocks=1,
    diffusion_heads=4,
    diffusion_atom_blocks=1,
    diffusion_atom_heads=2,
    fourier_dim=16,
    confidence_blocks=1,
    inference_num_steps=2,
    inference_num_loops=2,
    attention_backend="dense",
    trimul_backend="reference",
    plddt_bins=10,
    pae_bins=8,
    pde_bins=8,
    confidence_dist_bins=5,
    distogram_bins=8,
)


def tiny_fold_run_cfg(
    output_dir: Path,
    dataset: Path,
    lm_dir: Path,
    *,
    max_steps: int = 4,
    batch_size: int = 2,
    save_every: int = 10_000,
    ema_decay: float | None = None,
    resume_from: str | None = None,
    init_from: str | None = None,
    seed: int = 0,
    gradient_accumulation_steps: int = 1,
    lr: float = 1e-3,
) -> OplmConfig:
    """A CPU fold run on the preprocessed fixtures: tiny head, the saved tiny LM, crop 128."""
    from oplm.config import DataConfig, FoldRunConfig, OplmConfig, TrainConfig

    return OplmConfig(
        train=TrainConfig(
            max_steps=max_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optimizer="adamw",
            lr=lr,
            warmup_steps=0,
            max_grad_norm=1.0,
            log_every=1,
            save_every=save_every,
            save_final=False,
            ema_decay=ema_decay,
            resume_from=resume_from,
            init_from=init_from,
            seed=seed,
            output_dir=str(output_dir),
            mixed_precision="no",
            wandb_enabled=False,
        ),
        data=DataConfig(train=str(dataset), num_workers=0, pin_memory=False),
        fold=FoldRunConfig(
            lm=str(lm_dir),
            model=dict(TINY_FOLD_MODEL),
            crop_size=128,
            diffusion_samples=2,
            diffusion_microbatch=1,
            confidence_rollout_steps=2,
        ),
    )
```

(`Path` under `TYPE_CHECKING`; `OplmConfig` here is the run config from `oplm.config`, which conflicts with the `OplmConfig` already imported from `oplm.model` in this file: import it inside the function as shown, and keep the model one as is.) `tiny_fold_config` stays for the model-level tests.

- [ ] **Step 3: Write the failing tests**

Create `tests/fold/test_training.py`:

```python
"""FoldTask through the real Trainer on CPU: config building, gradients, overfit, resume, EMA."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import torch

from oplm.config import FoldRunConfig
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.modeling_fold import OplmForFolding
from oplm.fold.training import FoldTask, build_fold_config, load_initial_head, require_fold
from oplm.training.task import StepResult
from tests.fold.helpers import TINY_FOLD_MODEL, tiny_fold_run_cfg, tiny_lm
from tests.training.conftest import FullRecordingCallback, configure_accelerator_device

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def lm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("lm") / "tiny"
    tiny_lm().save_pretrained(path)
    return path


class _Recording(FoldTask):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[list[str]] = []

    def step(self, model, batch) -> StepResult:
        result = super().step(model, batch)
        self.ids.append(list(batch.ids))
        return result


def test_build_fold_config_infers_lm_shape_and_rejects_bad_keys() -> None:
    run = FoldRunConfig(lm="x", model=dict(TINY_FOLD_MODEL))
    cfg = build_fold_config(run, lm=tiny_lm())
    assert isinstance(cfg, FoldConfig)
    assert cfg.lm_hidden_size == 32 and cfg.lm_num_hidden_states == 3 and cfg.pair_width == 32
    assert cfg.lm_name_or_path == "x" and cfg.lm_max_chain_length == 1024
    with pytest.raises(ValueError, match="unknown FoldConfig key"):
        build_fold_config(FoldRunConfig(model={"pair_widht": 32}), lm=tiny_lm())
    with pytest.raises(ValueError, match="lm_name_or_path"):
        build_fold_config(FoldRunConfig(model={"lm_name_or_path": "y"}), lm=tiny_lm())
    explicit = build_fold_config(FoldRunConfig(model={"lm_hidden_size": 16}))
    assert explicit.lm_hidden_size == 16


def test_require_fold_names_the_missing_block(tmp_path: Path, fixture_dataset: Path, lm_dir: Path) -> None:
    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir)
    assert require_fold(cfg) is cfg.fold
    cfg.fold = None
    with pytest.raises(ValueError, match="fold:"):
        require_fold(cfg)


def test_step_returns_accounting_and_gradients_stay_in_the_head(
    tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir)
    task = FoldTask()
    model = task.build_model(cfg, None)
    assert model.training and model.lm is not None and not model.lm.training
    assert model.folding_trunk.gradient_checkpointing
    loader = task.build_dataloader(cfg)
    batch = next(iter(loader))
    result = task.step(model, batch)
    assert isinstance(result, StepResult) and result.samples == 2
    assert result.tokens == int(batch.features.token_mask.sum())
    assert {"diffusion_mse", "distogram", "confidence", "loops"} <= set(result.metrics)
    assert task.last_batch_ids == batch.ids and task.flops_per_token(cfg) is None
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in model.lm.parameters())


def test_dataloader_mixes_sources_with_the_run_seed(tmp_path: Path, fixture_dataset: Path, lm_dir: Path) -> None:
    from oplm.data.sequence.dataset import InterleavedDataset
    from oplm.fold.data.dataset import StructureDataset

    cfg = tiny_fold_run_cfg(tmp_path, fixture_dataset, lm_dir, seed=11)
    cfg.data.train = {"a": {"path": str(fixture_dataset), "fraction": 0.5}, "b": str(fixture_dataset)}
    loader = FoldTask().build_dataloader(cfg)
    dataset = loader.dataset
    assert isinstance(dataset, InterleavedDataset)
    sources = dataset._datasets
    assert all(isinstance(s, StructureDataset) and s._seed == 11 for s in sources)
    assert len(dataset) == sum(len(s) for s in sources)


@pytest.mark.slow
def test_trainer_overfits_one_chain(tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir, max_steps=40, batch_size=1, lr=2e-3)
    cfg.data.train = str(fixture_dataset)
    callback = FullRecordingCallback()
    trainer = Trainer(cfg, task=FoldTask(), callbacks=[callback])
    trainer.train()
    losses = [m["train/loss"] for _, m in callback.train_logs]
    assert len(losses) == 40 and all(torch.isfinite(torch.tensor(losses)).tolist())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5
    assert "train/distogram" in callback.train_logs[0][1]


@pytest.mark.slow
def test_resume_restores_step_cursor_and_ema(tmp_path: Path, fixture_dataset: Path, lm_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from oplm.training.trainer import Trainer

    configure_accelerator_device("cpu", monkeypatch)
    full = _Recording()
    Trainer(tiny_fold_run_cfg(tmp_path / "full", fixture_dataset, lm_dir, max_steps=6, ema_decay=0.5), task=full).train()
    first = _Recording()
    Trainer(tiny_fold_run_cfg(tmp_path / "split", fixture_dataset, lm_dir, max_steps=3, save_every=3, ema_decay=0.5), task=first).train()
    ckpt = tmp_path / "split" / "checkpoint-3"
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["ema"]["n_averaged"] == 3 and (ckpt / "hf_ema" / "config.json").is_file()
    assert '"model_type": "oplm_fold"' in (ckpt / "hf" / "config.json").read_text()
    second = _Recording()
    resumed = Trainer(tiny_fold_run_cfg(tmp_path / "split", fixture_dataset, lm_dir, max_steps=6, save_every=3, ema_decay=0.5, resume_from=str(ckpt)), task=second)
    assert resumed.global_step == 3 and int(resumed._ema.n_averaged) == 3
    resumed.train()
    assert first.ids + second.ids == full.ids
    assert int(resumed._ema.n_averaged) == 6


def test_init_from_loads_a_head_and_checks_its_architecture(tmp_path: Path) -> None:
    source = OplmForFolding(build_fold_config(FoldRunConfig(model=dict(TINY_FOLD_MODEL)), lm=tiny_lm()))
    source.save_pretrained(tmp_path / "hf")
    target = build_fold_config(FoldRunConfig(lm="other", model=dict(TINY_FOLD_MODEL)), lm=tiny_lm())
    loaded = load_initial_head(tmp_path / "hf", target)
    assert loaded.config.lm_name_or_path == "other"
    for k, v in source.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k
    wider = build_fold_config(FoldRunConfig(model={**TINY_FOLD_MODEL, "pair_width": 64}), lm=tiny_lm())
    with pytest.raises(ValueError, match="pair_width"):
        load_initial_head(tmp_path / "hf", wider)
```

Run: `.venv/bin/python -m pytest tests/fold/test_training.py -q`
Expected: FAIL with `ModuleNotFoundError: oplm.fold.training`.

- [ ] **Step 4: Implement `training.py`**

```python
"""FoldTask: the structure head's model, dataloader and step for the Trainer (spec §6.4).

The Trainer keeps acceleration, optimization, accumulation, EMA, checkpointing, resume and
eval cadence; this module supplies the three objective-specific pieces. ``build_model`` builds
the head from the ``fold:`` block (shape fields inferred from the loaded LM), attaches the frozen
LM on the process device (bf16 on CUDA) and enables block-level checkpointing; ``build_dataloader``
makes one :class:`StructureDataset` per ``data.train`` entry (every entry of a fold run is a
structure shard directory) mixed through :class:`InterleavedDataset`, seeded with the run seed so
the Trainer's ``set_epoch``/resume-skip forwarding restores the exact example stream; ``step``
runs the model's training branch and returns a :class:`StepResult`.
"""

from __future__ import annotations

import inspect
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader

from oplm.config import FoldRunConfig
from oplm.data.config import parse_train_configs
from oplm.data.sequence.dataset import InterleavedDataset
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.collate import FoldBatch, collate_rows
from oplm.fold.data.dataset import StructureDataset
from oplm.fold.lm_shim import lm_state_count
from oplm.fold.losses import LossWeights
from oplm.fold.modeling_fold import FoldTrainOutput, OplmForFolding, TrainSettings, load_frozen_lm
from oplm.training.callbacks import TrainerCallback
from oplm.training.initialization import resolve_initialization_source
from oplm.training.task import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from oplm.config import OplmConfig
    from oplm.training.trainer import Trainer

__all__ = ["FoldTask", "PeakMemoryCallback", "build_fold_config", "load_initial_head", "require_fold"]

_LM_RUN_FIELDS = ("lm_name_or_path", "lm_revision", "lm_max_chain_length")
_LM_SHAPE_FIELDS = ("lm_hidden_size", "lm_num_hidden_states")


def require_fold(cfg: OplmConfig) -> FoldRunConfig:
    """Return ``cfg.fold`` or raise a ``ValueError`` naming the missing block."""
    if cfg.fold is None:
        raise ValueError("this is not a fold run: the config has no `fold:` block (docs/FOLD.md §9)")
    return cfg.fold


def build_fold_config(run: FoldRunConfig, *, lm: nn.Module | None = None) -> FoldConfig:
    """``FoldConfig`` from the run block: ``run.model`` overrides, LM fields, inferred LM shape.

    Raises:
        ValueError: ``run.model`` names a key ``FoldConfig`` does not accept, or one of the
            LM fields that come from the run block itself.
    """
    overrides = dict(run.model or {})
    allowed = {
        name
        for name, p in inspect.signature(FoldConfig.__init__).parameters.items()
        if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown FoldConfig key(s) in fold.model: {unknown}")
    clash = sorted(set(overrides) & set(_LM_RUN_FIELDS))
    if clash:
        raise ValueError(f"fold.model may not set {clash}; use fold.lm / fold.lm_revision / fold.lm_max_chain_length")
    if lm is not None:
        overrides.setdefault("lm_hidden_size", int(lm.config.hidden_size))  # ty: ignore[unresolved-attribute, invalid-argument-type]  # nn.Module attrs
        overrides.setdefault("lm_num_hidden_states", lm_state_count(lm))
    return FoldConfig(
        **overrides,
        lm_name_or_path=run.lm,
        lm_revision=run.lm_revision,
        lm_max_chain_length=run.lm_max_chain_length,
    )


def load_initial_head(source: Path | str, target: FoldConfig) -> OplmForFolding:
    """Load every head tensor from a saved ``hf/`` or ``hf_ema/`` export (``train.init_from``).

    The saved architecture must equal ``target`` field for field, except the LM identity fields
    (``lm_name_or_path``, ``lm_revision``, ``lm_max_chain_length``), which the run owns and which
    are written onto the loaded config.

    Raises:
        ValueError: An architecture field differs, or tensors are missing or unexpected.
    """
    path = resolve_initialization_source(str(source))
    saved = FoldConfig.from_pretrained(path, local_files_only=True)
    architecture = [
        name
        for name, p in inspect.signature(FoldConfig.__init__).parameters.items()
        if p.kind == p.KEYWORD_ONLY and name not in _LM_RUN_FIELDS
    ]
    differences = [
        f"{k} (saved={getattr(saved, k)!r}, run={getattr(target, k)!r})"
        for k in architecture
        if getattr(saved, k) != getattr(target, k)
    ]
    if differences:
        raise ValueError("init_from head does not match the run's fold.model: " + "; ".join(differences))
    model, info = OplmForFolding.from_pretrained(path, local_files_only=True, output_loading_info=True)
    problems = {k: info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(k)}
    if problems:
        raise ValueError(f"incomplete head initialization from {path}: {problems}")
    for name in _LM_RUN_FIELDS:
        setattr(model.config, name, getattr(target, name))
    return model
```

```python
class FoldTask:
    """The structure head's :class:`~oplm.training.task.TrainTask` (docs/TRAIN.md §17)."""

    def __init__(self) -> None:
        self.settings = TrainSettings()
        self.last_batch_ids: list[str] = []

    def build_model(self, cfg: OplmConfig, initialization_source: Path | None) -> nn.Module:
        """Build the head, attach the frozen LM on the process device, enable checkpointing."""
        from accelerate import PartialState

        run = require_fold(cfg)
        device = PartialState().device
        dtype = torch.bfloat16 if device.type == "cuda" else None
        lm = load_frozen_lm(run.lm, revision=run.lm_revision, device=device, dtype=dtype)
        fold_cfg = build_fold_config(run, lm=lm)
        model = OplmForFolding(fold_cfg) if initialization_source is None else load_initial_head(initialization_source, fold_cfg)
        model.attach_lm(lm, name_or_path=run.lm, revision=run.lm_revision)
        model.train()
        model.gradient_checkpointing_enable()
        self.settings = TrainSettings(
            diffusion_samples=run.diffusion_samples,
            diffusion_microbatch=run.diffusion_microbatch,
            confidence_rollout_steps=run.confidence_rollout_steps,
            weights=LossWeights(
                diffusion_mse=run.loss_diffusion_mse,
                smooth_lddt=run.loss_smooth_lddt,
                distogram=run.loss_distogram,
                confidence=run.loss_confidence,
                pae=run.loss_pae,
            ),
        )
        return model

    def build_dataloader(self, cfg: OplmConfig) -> DataLoader[Any]:
        """One ``StructureDataset`` per ``data.train`` entry, mixed by fraction, seeded by the run."""
        run = require_fold(cfg)
        entries = parse_train_configs(cfg.data.train)
        if not entries:
            raise ValueError("cfg.data.train resolved to no datasets; nothing to train on")
        sources = [
            StructureDataset(
                entry.path,
                split="train",
                crop_size=run.crop_size,
                contiguous_prob=run.contiguous_prob,
                max_chain_length=run.lm_max_chain_length,
                seed=cfg.train.seed,
                shuffle_shards=cfg.data.shuffle_shards,
                shuffle_rows=cfg.data.shuffle_rows,
            )
            for entry in entries
        ]
        dataset: Any = (
            InterleavedDataset(sources, [e.fraction for e in entries], seed=cfg.train.seed)
            if len(sources) > 1
            else sources[0]
        )
        atom_pad_multiple = int((run.model or {}).get("atom_pad_multiple", FoldConfig().atom_pad_multiple))
        collate = partial(
            collate_rows,
            crop_size=run.crop_size,
            atoms_per_token_budget=run.atoms_per_token_budget,
            atom_pad_multiple=atom_pad_multiple,
        )
        num_workers = cfg.data.num_workers
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=cfg.data.pin_memory,
            prefetch_factor=cfg.data.prefetch_factor if num_workers > 0 else None,
        )

    def step(self, model: nn.Module, batch: Any) -> StepResult:
        """Run the training branch on one ``FoldBatch`` and return its loss and counts."""
        assert isinstance(batch, FoldBatch), type(batch)
        out = model(batch.features, targets=batch.targets, train=self.settings)
        assert isinstance(out, FoldTrainOutput)
        self.last_batch_ids = list(batch.ids)
        return StepResult(loss=out.loss, tokens=out.tokens, samples=out.chains, metrics=out.metrics)

    def flops_per_token(self, cfg: OplmConfig) -> int | None:
        """No honest estimate: cost depends on crop, loops and samples (spec §6.4)."""
        return None


class PeakMemoryCallback(TrainerCallback):
    """Record per-log step time and CUDA peak memory; ``oplm fold measure`` writes the result."""

    def __init__(self) -> None:
        self.records: list[dict[str, float]] = []
        self._started = time.monotonic()

    def on_train_start(self, trainer: Trainer) -> None:
        self._started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_log(self, trainer: Trainer, metrics: dict[str, float], step: int) -> None:
        if "train/loss" not in metrics:
            return
        record = {"step": float(step), "elapsed_s": time.monotonic() - self._started}
        record.update({k.removeprefix("train/"): float(v) for k, v in metrics.items() if k.startswith("train/")})
        if torch.cuda.is_available():
            record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        self.records.append(record)
```

`src/oplm/fold/__init__.py`: add `"FoldTask": ("oplm.fold.training", "FoldTask")` to `_LAZY` and `__all__` (and the `TYPE_CHECKING` import).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_training.py tests/training/test_e2e_ema.py tests/training/test_e2e_data_exact.py -q`
Expected: PASS (the two MLM E2E files prove the seam change left the default task alone). If `Trainer.__init__` fails on `cfg.model` for a fold run (it reads `cfg.model.gradient_checkpointing` and `num_loops` through `getattr`), fix the read site with `getattr(..., default)`, never by adding a model block to fold configs.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/training.py src/oplm/training/task.py src/oplm/training/mup.py src/oplm/training/optim.py src/oplm/fold/__init__.py tests/fold/helpers.py tests/fold/test_training.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/training.py src/oplm/training/task.py src/oplm/training/mup.py src/oplm/training/optim.py src/oplm/fold/__init__.py tests/fold/helpers.py tests/fold/test_training.py
git commit -m "feat(fold): FoldTask for the Trainer (model, loaders, step, peak-memory callback)"
```

---

### Task 8: `oplm fold train` and the fold entry module

**Files:**
- Create: `src/oplm/fold/train.py`
- Modify: `src/oplm/train.py` (`main(cfg, *, task=None)`), `src/oplm/fold/cli.py` (+ `train` command)
- Test: `tests/fold/test_cli.py` (append), `tests/test_train_entrypoint.py` (append one test)

**Interfaces:**
- Consumes: `FoldTask`, `require_fold` (Task 7), `load_config`.
- Produces: `oplm.train.main(cfg=None, *, task=None)`; `oplm.fold.train.main(cfg=None)` (`python -m oplm.fold.train --config run.yaml [overrides]`, `accelerate launch -m oplm.fold.train ...`); `oplm fold train [--config PATH] [--name NAME] [overrides...]`. Task 10's `measure` and M2c's Slurm rendering launch `oplm.fold.train`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/fold/test_cli.py` (import `tiny_fold_run_cfg` and `serialize_config`; `fixture_dataset` comes from `tests/fold/conftest.py`):

```python
def test_fold_train_runs_a_tiny_cpu_run_from_a_config(
    tmp_path: Path, fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    cfg = tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir, max_steps=2, save_every=2)
    config = tmp_path / "run.yaml"
    config.write_text(serialize_config(cfg))
    result = runner.invoke(app, ["fold", "train", "--config", str(config), "train.max_steps=2"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "checkpoint-2" / "hf" / "config.json").is_file()
    assert "pair_width" in plain(result.output)


def test_fold_train_refuses_a_config_without_a_fold_block(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fold", "train", "train.max_steps=1"])
    assert result.exit_code != 0 and "fold:" in plain(result.output)
```

And to `tests/test_train_entrypoint.py`:

```python
def test_fold_entry_module_requires_a_fold_block() -> None:
    from oplm.fold.train import main

    from oplm.config import load_config

    with pytest.raises(ValueError, match="fold:"):
        main(load_config([]))
```

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q -k fold_train`
Expected: FAIL (`No such command 'train'`).

- [ ] **Step 2: Implement**

`src/oplm/train.py`: `def main(cfg: OplmConfig | None = None, *, task: TrainTask | None = None) -> None` and `trainer = Trainer(cfg, task=task)` (`TrainTask` under `TYPE_CHECKING` from `oplm.training.task`). Docstring: "`task` selects the objective; `None` is MLM. Fold runs enter through `oplm.fold.train`."

Create `src/oplm/fold/train.py`:

```python
"""Fold training entry point.

Run directly:     python -m oplm.fold.train --config configs/fold/proof-w128.yaml train.max_steps=10
Run distributed:  accelerate launch -m oplm.fold.train --config configs/fold/proof-w128.yaml

Parses the run config exactly as ``oplm.train`` does and hands ``FoldTask`` to the Trainer.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oplm.config import OplmConfig


def main(cfg: OplmConfig | None = None) -> None:
    """Train the structure head from ``cfg`` (parsed from ``sys.argv`` when ``None``).

    Raises:
        ValueError: The config has no ``fold:`` block.
    """
    if cfg is None:
        from oplm.config import load_config

        cfg = load_config(sys.argv[1:])
    from oplm.fold.training import FoldTask, require_fold
    from oplm.train import main as train_main

    require_fold(cfg)
    train_main(cfg, task=FoldTask())


if __name__ == "__main__":
    main()
```

`src/oplm/fold/cli.py`:

```python
@app.command("train")
def train(
    overrides: Annotated[list[str] | None, typer.Argument(help="Dotlist overrides, e.g. train.max_steps=10")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Run config YAML with a fold: block")] = None,
    name: Annotated[str | None, typer.Option("--name", help="W&B run name and ./<name> output dir")] = None,
) -> None:
    """Train the structure head (single process; use `accelerate launch -m oplm.fold.train` for DDP)."""
    from oplm.config import load_config

    argv: list[str] = []
    if config is not None:
        argv += ["--config", str(config)]
    if name is not None:
        argv += ["--name", name]
    argv += list(overrides or [])
    cfg = load_config(argv)
    if cfg.fold is None:
        raise typer.BadParameter("the config has no fold: block (docs/FOLD.md §9)")
    model = cfg.fold.model or {}
    console.print(
        f"[bold]Head:[/bold] pair_width {model.get('pair_width', 256)} / trunk_blocks "
        f"{model.get('trunk_blocks', 24)}  [bold]LM:[/bold] {cfg.fold.lm}  "
        f"[bold]crop:[/bold] {cfg.fold.crop_size}"
    )
    console.print(f"[bold]Output:[/bold] {cfg.train.output_dir}")
    from oplm.fold.train import main as train_main

    train_main(cfg)
```

Update the module docstring's command list.

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py tests/test_cli.py tests/test_train_entrypoint.py -q`
Expected: PASS.

- [ ] **Step 4: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/train.py src/oplm/fold/train.py src/oplm/fold/cli.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/train.py src/oplm/fold/train.py src/oplm/fold/cli.py tests/fold/test_cli.py tests/test_train_entrypoint.py
git commit -m "feat(fold): oplm fold train and the oplm.fold.train entry module"
```

---
### Task 9: Two-process DDP smoke with resume

**Files:**
- Create: `tests/fold/_train_worker.py`, `tests/fold/test_training_distributed.py`

**Interfaces:**
- Consumes: `tiny_fold_run_cfg` (Task 7), `serialize_config`, `tests.training.test_e2e_hsdp._child_env`, `FoldTask`.
- Produces: a `slow` test that launches two gloo ranks on CPU, trains, checkpoints, resumes, and checks the per-rank example stream continues exactly (spec §9 "DDP smoke/resume check").

- [ ] **Step 1: Write the worker**

`tests/fold/_train_worker.py`:

```python
"""Two-rank fold training worker: trains from a resolved YAML and records the example stream."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch


def main(config_path: str, result_dir: str) -> None:
    from oplm.config import load_config
    from oplm.fold.training import FoldTask
    from oplm.training.trainer import Trainer
    from tests.training.conftest import FullRecordingCallback

    cfg = load_config(["--config", config_path])
    callback = FullRecordingCallback()

    class _Recording(FoldTask):
        ids: list[list[str]] = []

        def step(self, model, batch):
            result = super().step(model, batch)
            self.ids.append(list(batch.ids))
            return result

    task = _Recording()
    trainer = Trainer(cfg, task=task, callbacks=[callback])
    rank = trainer.accelerator.process_index
    resumed_from = trainer.global_step
    trainer.train()
    losses = [m["train/loss"] for _, m in callback.train_logs]
    payload = {
        "rank": rank,
        "world_size": trainer.accelerator.num_processes,
        "resumed_from_step": resumed_from,
        "global_step": trainer.global_step,
        "ids": task.ids,
        "all_losses_finite": all(math.isfinite(x) for x in losses),
        "ema_n_averaged": None if trainer._ema is None else int(trainer._ema.n_averaged),
        "is_ddp": type(trainer.model).__name__ == "DistributedDataParallel",
    }
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(payload))
    torch.distributed.barrier()
    trainer.accelerator.end_training()


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    except (ValueError, RuntimeError) as exc:
        Path(sys.argv[2], f"rank{os.environ.get('RANK', '0')}.error").write_text(str(exc))
        raise
```

- [ ] **Step 2: Write the test**

`tests/fold/test_training_distributed.py`:

```python
"""Two gloo ranks on CPU: DDP-wrapped fold head, checkpoint at step 2, exact resume to step 4."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from oplm.config import serialize_config
from tests.fold.helpers import tiny_fold_run_cfg, tiny_lm
from tests.training.test_e2e_hsdp import _child_env

pytestmark = pytest.mark.slow


def _launch(cfg: Any, result: Path) -> list[dict[str, Any]]:
    result.mkdir(parents=True)
    config = result / "run.yaml"
    config.write_text(serialize_config(cfg))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--rdzv_backend=c10d",
        "--rdzv_endpoint=localhost:0",
        str(Path(__file__).with_name("_train_worker.py")),
        str(config),
        str(result),
    ]
    completed = subprocess.run(command, env=_child_env(), capture_output=True, text=True, timeout=600)
    (result / "stdout.log").write_text(completed.stdout)
    (result / "stderr.log").write_text(completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return [json.loads((result / f"rank{rank}.json").read_text()) for rank in (0, 1)]


def test_two_rank_training_checkpoints_and_resumes_the_example_stream(
    tmp_path: Path, fixture_dataset: Path
) -> None:
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    full = _launch(
        tiny_fold_run_cfg(tmp_path / "full", fixture_dataset, lm_dir, max_steps=4, batch_size=1, ema_decay=0.5),
        tmp_path / "full-result",
    )
    first = _launch(
        tiny_fold_run_cfg(tmp_path / "split", fixture_dataset, lm_dir, max_steps=2, batch_size=1, save_every=2, ema_decay=0.5),
        tmp_path / "first-result",
    )
    ckpt = tmp_path / "split" / "checkpoint-2"
    assert (ckpt / "ema.pt").is_file()
    second = _launch(
        tiny_fold_run_cfg(tmp_path / "split", fixture_dataset, lm_dir, max_steps=4, batch_size=1, save_every=2, ema_decay=0.5, resume_from=str(ckpt)),
        tmp_path / "second-result",
    )
    for rank in (0, 1):
        assert full[rank]["is_ddp"] and full[rank]["world_size"] == 2
        assert full[rank]["all_losses_finite"] and second[rank]["all_losses_finite"]
        assert second[rank]["resumed_from_step"] == 2 and second[rank]["global_step"] == 4
        assert first[rank]["ids"] + second[rank]["ids"] == full[rank]["ids"]
        assert second[rank]["ema_n_averaged"] == 4
    assert full[0]["ids"] != full[1]["ids"]  # the ranks see different stripes
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/python -m pytest tests/fold/test_training_distributed.py -q`
Expected: PASS within a few minutes on CPU (three launches of two ranks over the six-entry fixture dataset). If the ranks disagree on `ids` only in order within a step, that is a bug in the dataset striping, not in this test; do not loosen the assertion.

- [ ] **Step 4: Commit**

```bash
.venv/bin/ruff format tests/fold/_train_worker.py tests/fold/test_training_distributed.py
git add tests/fold/_train_worker.py tests/fold/test_training_distributed.py
git commit -m "test(fold): two-rank DDP training smoke with exact resume"
```

---

### Task 10: `oplm fold measure`, the run configs and the B200 job

**Files:**
- Modify: `src/oplm/fold/cli.py` (+ `measure`)
- Create: `configs/fold/proof-w128.yaml`, `configs/fold/proof-w256.yaml`, `docs/fold/b200-m2b-measure.sbatch`
- Test: `tests/fold/test_cli.py` (append)

**Interfaces:**
- Consumes: `PeakMemoryCallback`, `FoldTask` (Task 7), `load_config`, `Trainer`.
- Produces: `oplm fold measure --config PATH --steps N --out PATH [overrides...]` writing a JSON report (`environment`, `settings`, `steps`, `peak_allocated_gib`, `step_time_s`); the two proof configs (M2c sets `train.batch_size`, `gradient_accumulation_steps` and `max_steps` from the measurement); the job that runs the measurement on one B200 (Task 12).

- [ ] **Step 1: Write the failing test**

Append to `tests/fold/test_cli.py`:

```python
def test_fold_measure_writes_a_report(tmp_path: Path, fixture_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    lm_dir = tmp_path / "lm"
    tiny_lm().save_pretrained(lm_dir)
    config = tmp_path / "run.yaml"
    config.write_text(serialize_config(tiny_fold_run_cfg(tmp_path / "run", fixture_dataset, lm_dir)))
    out = tmp_path / "measure.json"
    result = runner.invoke(app, ["fold", "measure", "--config", str(config), "--steps", "2", "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["settings"]["steps"] == 2 and len(report["steps"]) == 2
    assert all(torch.isfinite(torch.tensor(s["loss"])) for s in report["steps"])
    assert report["environment"]["torch"] and "git" in report["environment"]
    assert report["settings"]["pair_width"] == 32 and report["settings"]["crop_size"] == 128
    assert report["peak_allocated_gib"] is None  # CPU run
```

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q -k measure` — Expected: FAIL (`No such command 'measure'`).

- [ ] **Step 2: Implement the command**

In `src/oplm/fold/cli.py`:

```python
@app.command("measure")
def measure(
    config: Annotated[Path, typer.Option("--config", help="Run config YAML with a fold: block")],
    out: Annotated[Path, typer.Option("--out", help="JSON report path")],
    steps: Annotated[int, typer.Option(help="Optimizer steps to run")] = 20,
    overrides: Annotated[list[str] | None, typer.Argument(help="Dotlist overrides")] = None,
) -> None:
    """Run a few real optimizer steps and record step time and peak memory (docs/FOLD.md §9)."""
    import torch

    from oplm.config import load_config
    from oplm.fold.training import FoldTask, PeakMemoryCallback, require_fold
    from oplm.training.trainer import Trainer

    cfg = load_config(["--config", str(config), *(overrides or [])])
    run = require_fold(cfg)
    cfg.train.max_steps = steps
    cfg.train.log_every = 1
    cfg.train.save_every = 10**9
    cfg.train.save_final = False
    cfg.train.save_every_minutes = None
    cfg.train.wandb_enabled = False
    cfg.train.throughput_warmup_steps = min(3, steps - 1) if steps > 1 else 0
    callback = PeakMemoryCallback()
    task = FoldTask()
    trainer = Trainer(cfg, task=task, callbacks=[callback])
    trainer.train()
    device = trainer.accelerator.device
    model = cfg.fold.model or {}
    records = callback.records
    settled = [r for r in records if r["step"] > cfg.train.throughput_warmup_steps] or records
    times = [r["step_time_s"] for r in settled if "step_time_s" in r]
    report = {
        "environment": _environment(device),
        "settings": {
            "steps": steps,
            "pair_width": int(model.get("pair_width", 256)),
            "trunk_blocks": int(model.get("trunk_blocks", 24)),
            "lm": run.lm,
            "crop_size": run.crop_size,
            "batch_size": cfg.train.batch_size,
            "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
            "diffusion_samples": run.diffusion_samples,
            "diffusion_microbatch": run.diffusion_microbatch,
            "confidence_rollout_steps": run.confidence_rollout_steps,
            "mixed_precision": cfg.train.mixed_precision,
            "num_processes": trainer.accelerator.num_processes,
        },
        "steps": records,
        "peak_allocated_gib": max((r["peak_allocated_gib"] for r in records), default=None) if torch.cuda.is_available() else None,
        "peak_reserved_gib": max((r["peak_reserved_gib"] for r in records), default=None) if torch.cuda.is_available() else None,
        "step_time_s": (sum(times) / len(times)) if times else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    console.print(
        f"wrote {out}  steps {steps}  step_time_s {report['step_time_s']}  "
        f"peak_allocated_gib {report['peak_allocated_gib']}"
    )
```

`_environment(device)` already exists in this module (host, device name, torch/CUDA/cuEquivariance versions, the short git revision under `git`). The report's `steps` entries carry every `train/*` metric of that step (loss, the per-term fold metrics, `step_time_s`, `tokens_per_sec`).

- [ ] **Step 3: Run configs**

`configs/fold/proof-w128.yaml`:

```yaml
# Milestone-2 pipeline proof, pair width 128 (addendum §6; docs/FOLD.md §9). Train from the
# published dataset: hf download brineylab/oplmfold-training-data --repo-type dataset \
#   --revision wwpdb-2026-10-12 --include "wwpdb/2026-10-12/*" --local-dir $OPLM_DATA/hf
# and point data.train at $OPLM_DATA/hf/wwpdb/2026-10-12 (override on the command line).
# train.batch_size, gradient_accumulation_steps and max_steps are set by milestone 2c from the
# milestone-2b measurement (docs/fold/m2/); the values below are the measurement defaults.
fold:
  lm: esmc:esmc_600m          # 36 blocks, width 1152 -> 37 hidden states (inferred at load)
  lm_max_chain_length: 1024
  model:
    pair_width: 128
    trunk_blocks: 24
    lm_encoder_blocks: 4
    coda_blocks: 2
    confidence_blocks: 4
    inference_num_loops: 10
  crop_size: 384
  contiguous_prob: 0.5
  diffusion_samples: 8
  diffusion_microbatch: 4
  confidence_rollout_steps: 20
  loss_diffusion_mse: 4.0
  loss_smooth_lddt: 1.0
  loss_distogram: 0.03
  loss_confidence: 1.0e-4
  loss_pae: 1.0
train:
  optimizer: adamw            # μP/Muon are the MLM defaults; the head trains with plain AdamW
  lr: 1.0e-3
  min_lr: 0.0
  warmup_steps: 1000
  scheduler: warmup_linear
  weight_decay: 0.01
  max_grad_norm: 1.0
  batch_size: 1               # per device
  gradient_accumulation_steps: 16   # 8 devices x 1 x 16 = 128 (spec §6.3 initial global batch)
  max_steps: 50000
  ema_decay: 0.999
  mixed_precision: bf16
  seed: 0
  save_every: 2000
  save_total_limit: 3
  keep_every_n_steps: 10000
  log_every: 10
  wandb_project: oplm-fold
  output_dir: outputs/fold-proof-w128
data:
  train: /mnt/data/briney/oplm-data/hf/wwpdb/2026-10-12
  num_workers: 4
  pin_memory: true
  prefetch_factor: 4
```

`configs/fold/proof-w256.yaml`: identical except `pair_width: 256` and `output_dir: outputs/fold-proof-w256`. Both must pass `load_config(["--config", path])` (no `model:` block is needed; the MLM model defaults are unused). Add a test in `tests/fold/test_cli.py`:

```python
@pytest.mark.parametrize("width", [128, 256])
def test_proof_configs_load(width: int) -> None:
    from oplm.config import load_config

    cfg = load_config(["--config", f"configs/fold/proof-w{width}.yaml"])
    assert cfg.fold is not None and cfg.fold.model["pair_width"] == width
    assert cfg.train.optimizer == "adamw" and cfg.train.ema_decay == 0.999 and cfg.fold.crop_size == 384
```

- [ ] **Step 4: The measurement job**

`docs/fold/b200-m2b-measure.sbatch`, modelled on `docs/fold/b200-fixtures.sbatch` (same container, mounts, `ENV_FILE`, `run` step helper, `OPLM_REPO`/`OPLM_GIT_REF`/`OPLM_RESULTS` overrides). Header comment: milestone-2b acceptance (addendum §5), one B200, Python 3.12 venv with `esm==3.4.1.post1` plus this repo's `[dev,train,fold]` extras, the gated ESM-C weights (the env file must export `HF_TOKEN` of an account that accepted the Cambrian license), the dataset download, then `oplm fold measure` at both widths. Steps and commands (inside the `run` helper):

```bash
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"
DATA="${OPLM_DATA_ROOT:-/mnt/data/${SLURM_JOB_USER}/oplm-data/hf}"
run venv bash -c "python -m pip install --quiet uv && ([ -x '$ESM_VENV/bin/python' ] || python -m uv venv --seed --python 3.12 '$ESM_VENV')"
run install bash -c "python -m uv pip install --python '$ESM_VENV/bin/python' --quiet 'esm==3.4.1.post1' && python -m uv pip install --python '$ESM_VENV/bin/python' --quiet -e '$REPO[dev,train,fold]' && python -m uv pip freeze --python '$ESM_VENV/bin/python' | grep -i -E '^(esm|torch|transformers|accelerate|cuequivariance|huggingface)' > '$RESULTS/pip-freeze.txt'"
run env bash -c "source '$ESM_VENV/bin/activate' && python -c 'import esm, torch, transformers; print(esm.__version__, torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))'"
run download bash -c "source '$ESM_VENV/bin/activate' && hf download brineylab/oplmfold-training-data --repo-type dataset --revision wwpdb-2026-10-12 --include 'wwpdb/2026-10-12/*' --local-dir '$DATA'"
run esmc bash -c "source '$ESM_VENV/bin/activate' && python -m pytest tests/fold/test_esmc.py -q -rs"
run measure-w128 bash -c "source '$ESM_VENV/bin/activate' && oplm fold measure --config configs/fold/proof-w128.yaml --steps 20 --out '$RESULTS/measure-w128.json' data.train='$DATA/wwpdb/2026-10-12' train.output_dir='$JOB_WORK_DIR/measure-w128'"
run measure-w256 bash -c "source '$ESM_VENV/bin/activate' && oplm fold measure --config configs/fold/proof-w256.yaml --steps 20 --out '$RESULTS/measure-w256.json' data.train='$DATA/wwpdb/2026-10-12' train.output_dir='$JOB_WORK_DIR/measure-w256'"
run gpu-tests bash -c "source '$ESM_VENV/bin/activate' && python -m pytest tests/fold -m slow -q -rs --junitxml='$RESULTS/fold-gpu-tests.xml'"
```

with `--gres=gpu:1`, `--cpus-per-task=32`, `--mem=256G`, `--time=06:00:00`, `--job-name=fold-m2b-measure`, results under `/mnt/home/${SLURM_JOB_USER}/fold-m2b/${SLURM_JOB_ID}`, and a closing echo naming the files to copy into `docs/fold/m2/`. Keep `set -euo pipefail` at the outer level and the inner `run` semantics (a failing step is recorded in `status.txt`, later steps still run).

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest tests/fold/test_cli.py -q`
Expected: PASS. Also `bash -n docs/fold/b200-m2b-measure.sbatch` for syntax.

- [ ] **Step 6: Lint, type-check, commit**

```bash
.venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/cli.py tests/fold/test_cli.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/
git add src/oplm/fold/cli.py configs/fold docs/fold/b200-m2b-measure.sbatch tests/fold/test_cli.py
git commit -m "feat(fold): oplm fold measure, the proof run configs and the B200 measurement job"
```

---

### Task 11: Documentation

**Files:**
- Modify: `docs/FOLD.md` (§1 layout, §4, new §9), `docs/TRAIN.md` §17, `docs/CONFIG.md` (new "Fold fields" section), `docs/TESTING_E2E.md` (tests/fold and fixtures rows)
- Create: `docs/fold/m2/README.md` (placeholder for Task 12's evidence)
- Test: `tests/test_docs_links.py` (existing; must stay green)

- [ ] **Step 1: FOLD.md**

§1: add `esmc.py`, `losses.py`, `training.py`, `train.py` to the tree, `cli.py`'s command list (`train | measure`), and `tests/fold/oracles/` under the tests line. §4: replace "`FoldTask` lands in milestone 2" with a pointer to §9. Update the status line at the top: "milestone 2b (losses and training) implemented; GPU measurement pending" (Task 12 rewrites it). New section:

```markdown
## 9. Milestone 2b: losses and training (`oplm.fold.losses`, `oplm.fold.training`)

**One training step** (spec §6.1) runs inside `OplmForFolding.forward(features, targets=...)` so DDP
synchronizes through one forward: in train mode 10 % of inner residue ids on the LM rows are masked
(id 32; never BOS/EOS/pad), the loop count is a clamped Poisson draw per rank (mean 3, 1–6; eval uses
`inference_num_loops`), the trunk runs as at inference (gradient through the last 2 loops, per-loop
LM-pair dropout), distogram logits come from the symmetrized final pair, and then: `diffusion_samples`
copies of the reference are centred, rotated and translated, noised at `σ = 16 · exp(N(−1.2, 1.5))`,
denoised once each in microbatches of `diffusion_microbatch` samples (each microbatch under
`torch.utils.checkpoint`, so the sample count never multiplies live memory; the per-sample losses are
averaged, so the microbatch size does not change the objective); a `confidence_rollout_steps`-step
sampler rollout runs without gradient and the confidence head scores it on detached trunk outputs.
`FoldTrainOutput` carries the weighted loss, detached metrics, the valid token count and the chain count.

**Losses** (`losses.py`, all fp32, one value per sample, finite on empty masks) and their conventions:
| term | definition | weight |
|---|---|---|
| diffusion MSE | reference aligned onto the prediction (detached), `Σ w |Δ|² / (3 Σ w)`, times `(σ² + σ_d²)/(σ σ_d)²` | 4.0 |
| smooth lDDT | pairs under 15 Å in the reference, both resolved, `i ≠ j`; `1 − mean ¼ Σ_t sigmoid(t − |Δd|)` | 1.0 |
| distogram | representative-atom distances in 64 bins over 2–22 Å (`distogram_{min,max}_dist`), masked mean CE | 0.03 |
| confidence | pLDDT (per-atom lDDT vs representative atoms within 15 Å, 50 bins) + PDE (64 bins over 0–32 Å) + resolved (class 0 = resolved) + `loss_pae` × PAE (token CA in the N–CA–C frame of the other token, 64 bins over 0–32 Å; frames need resolved, non-collinear atoms) | 1e-4 |
Oracles: Boltz-2 (`boltz==2.2.1`) functions vendored under `tests/fold/oracles/`; parity, rigid-transform
recovery, mask and degeneracy tests in `tests/fold/test_losses.py`.

**Frozen ESM-C** (`esmc.py`, addendum §3): `fold.lm: esmc:esmc_600m` loads `biohub/ESMC-600M` through
the `esm` package (`oplm[esmc]`, Python ≥ 3.12; gated weights) wrapped as `FrozenEsmC`, which exposes
`num_hidden_states = num_hidden_layers + 1` (the package returns every block input, the embedding output
first, plus the final post-norm output: 37 states of width 1152 for 600M). `lm_hidden_size` and
`lm_num_hidden_states` are inferred from the loaded LM when the run does not set them.

**Run config and CLI.** A run is an ordinary `OplmConfig` plus a `fold:` block (docs/CONFIG.md "Fold
fields"); `data.train` entries are structure shard directories. `oplm fold train --config run.yaml
[overrides]` for one process, `accelerate launch -m oplm.fold.train --config run.yaml` for DDP;
`Trainer(cfg, task=FoldTask())` underneath. The head is built from `fold.model`, the LM attached on the
process device (bf16 on CUDA), block-level checkpointing enabled, `train.init_from` loads a saved
`hf/` or `hf_ema/` head whose architecture must match. EMA, checkpoints (`hf/` + `hf_ema/`), resume
(step, data cursor, EMA, RNG) and DDP come from the Trainer unchanged; `tests/fold/test_training.py`
and `test_training_distributed.py` exercise them on CPU.

**Measurement.** `oplm fold measure --config configs/fold/proof-w128.yaml --steps 20 --out m.json`
runs real optimizer steps and records step time and CUDA peak memory; `docs/fold/b200-m2b-measure.sbatch`
runs it at both widths on one B200. Results and the derived per-device batch, accumulation and step
budget for milestone 2c are recorded in [`docs/fold/m2/`](fold/m2/).

**Deviations, all deliberate.** Four loss terms (no bond term, no interface weighting; addendum §2);
single-chain examples; block-level checkpointing only; the ESM-C head is not transferable to OPLM.
```

- [ ] **Step 2: TRAIN.md, CONFIG.md, TESTING_E2E.md, docs/fold/m2/README.md**

TRAIN.md §17: after the table, "`FoldTask` (`oplm.fold.training`) is the second implementation: `oplm fold train` / `python -m oplm.fold.train` build `Trainer(cfg, task=FoldTask())`; batches are `FoldBatch` objects moved through their own `.to`, and `train/<term>` metrics come from `StepResult.metrics` (docs/FOLD.md §9)."

CONFIG.md: a "## Fold fields (`fold.*`)" section listing every `FoldRunConfig` field with its default and one line of meaning, noting that the block is absent (`null`) for MLM runs, that `fold.model.*` keys are `FoldConfig` constructor names validated at model build, and that `train.optimizer=adamw` must be set explicitly because the packaged defaults are μP + Muon.

TESTING_E2E.md: extend the `tests/fold/` row with "losses against the vendored Boltz-2 oracles (`tests/fold/oracles/`), the training branch, `FoldTask` through the real Trainer (gradients, overfit, resume with EMA; `slow`), and a two-rank gloo DDP smoke (`slow`)"; extend the `tests/fixtures/` row with the oracles' location.

`docs/fold/m2/README.md`:

```markdown
# Milestone 2b acceptance: single-B200 cost measurement

Filled by Task 12 of `docs/superpowers/plans/2026-10-12-fold-m2b-training.md`: the
`oplm fold measure` reports at widths 128 and 256 (`measure-w128.json`, `measure-w256.json`),
the GPU test log, the environment (`pip-freeze.txt`), and the per-device batch, accumulation
and step budget derived for milestone 2c.
```

- [ ] **Step 3: Verify and commit**

Run: `.venv/bin/python -m pytest tests/test_docs_links.py -q` — Expected: PASS.

```bash
git add docs/FOLD.md docs/TRAIN.md docs/CONFIG.md docs/TESTING_E2E.md docs/fold/m2/README.md
git commit -m "docs(fold): milestone-2b training contract, config block and measurement"
```

Then run the whole fast suite, ruff and ty one last time before the final review:
`.venv/bin/python -m pytest -q -m "not slow" && .venv/bin/ruff check src/ && VIRTUAL_ENV=.venv .venv/bin/ty check src/`.

---

### Task 12: GPU measurement on one B200 (user-run acceptance)

**Files:**
- Create: `docs/fold/m2/measure-w128.json`, `docs/fold/m2/measure-w256.json`, `docs/fold/m2/fold-gpu-tests.xml`, `docs/fold/m2/pip-freeze.txt`, `docs/fold/m2/status.txt`; fill `docs/fold/m2/README.md`
- Modify: `docs/FOLD.md` §9 status line; `configs/fold/proof-w*.yaml` (batch, accumulation, steps)

This task runs on the cluster after Tasks 1–11 are pushed. Prerequisites: a Hub account that accepted the ESM-C license, `HF_TOKEN` exported from the job's env file, and the published dataset tag `wwpdb-2026-10-12`.

- [ ] **Step 1: Submit**

```bash
OPLM_GIT_REF=feat/fold-m2b sbatch docs/fold/b200-m2b-measure.sbatch
```

- [ ] **Step 2: Read the results**

Copy `measure-w128.json`, `measure-w256.json`, `fold-gpu-tests.xml`, `pip-freeze.txt`, `status.txt` from the job's results directory into `docs/fold/m2/`. Check: every step finite, `peak_allocated_gib` well under the B200's usable memory at both widths, the `esmc` and `gpu-tests` steps exited 0 (the ESM-C 300M test downloads the gated weights; a 403 means the token's account has not accepted the license).

- [ ] **Step 3: Derive the milestone-2c budget**

From `step_time_s` and `peak_allocated_gib` at `batch_size=1`, `diffusion_microbatch=4`: choose the largest per-device batch that keeps peak allocated below 75 % of the device, set `gradient_accumulation_steps` so `8 × batch × accumulation = 128`, and the step budget as the smaller of spec §6.3's 50k and what fits the allocation (steps × step time × accumulation). Write the numbers and the reasoning into `docs/fold/m2/README.md` (machine, job id, environment, both widths' step time and peak memory, the chosen batch/accumulation/steps) and into both proof configs.

- [ ] **Step 4: Record and commit**

Update the FOLD.md status line to "milestone 2b accepted <date> (measured on one B200; docs/fold/m2/)".

```bash
git add docs/fold/m2 docs/FOLD.md configs/fold
git commit -m "docs(fold): record the milestone-2b B200 measurement and the proof-run budget"
```

Then finish the branch with the usual options (PR against `main`).
