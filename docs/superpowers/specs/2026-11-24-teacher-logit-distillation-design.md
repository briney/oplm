# Teacher-logit distillation for OPLM annealing — design

Date: 2026-11-24. Status: approved 2026-11-25.

## Intent

We want to infuse co-evolutionary and structural knowledge from precomputed MSAs into a purely
single-sequence, sequence-packed OPLM encoder during the WSD annealing (decay) phase, without
changing the model architecture or requiring MSAs at inference.

The completed MSA Pairformer spike (flyloft campaign `b0426a4b`, published toolshed tool
`msa-pairformer@1.0.2`) established on `pdb40_256` that query-masked MSA Pairformer conditional
`(L, 20)` logits beat 170M OPLM contact `P@L` across every source-`Neff` tercile (`0.507 / 0.699 /
0.742` vs `0.289 / 0.395 / 0.504`; `0.659` macro at depth 256 uncapped, `K=7`) and only drop below
OPLM on the shallowest alignments (`selected Neff / sqrt(L) < 2` or `raw selected Neff < 20`).

This design specifies the **OPLM training and data-loading side** only: consuming sharded sequence
Parquet that optionally carries precomputed per-residue `(L, 20)` teacher logits and `Neff`
scalars, blending KL distillation with hard MLM cross-entropy on distilled masked tokens, and
falling back cleanly to plain hard MLM on non-MSA sources (`deepclust70`, OAS antibodies) and
`Neff`-gated sequences.

## Goals

1. **Teacher-agnostic Parquet contract:** `ShardedProteinDataset` optionally reads a per-sequence
   `teacher_logits` binary float16 column plus optional `teacher_neff` / `teacher_neff_sqrt_l`
   scalars, with runtime `Neff` gating so one packed UniRef50 dataset supports A/B tests of
   `no gate` vs `Neff / sqrt(L) >= 2` vs `Neff >= 20` via config alone.
2. **Packing-, compile-, and Canon-safe collation:** `MLMCollator` aligns and packs teacher logits
   alongside `labels` while preserving identical RNG/masking draws, sequence packing
   (`pack_sequences=true`, `TokenBatcher`), static-shape `torch.compile(dynamic=False)`, and
   `Trainer._canon_self_check`.
3. **Exact per-masked-token blended objective:** `DistillMLMTask(MLMTask)` computes hard CE on
   plain masked tokens and `(1 - alpha) * CE + alpha * T^2 * KL` on distilled masked canonical
   tokens, with every masked token weighted `1.0` so mixed-source token shares stay balanced and
   `alpha = 0.0` matches vanilla `MLMTask` loss.
4. **Zero-impact default:** When `distill` is omitted (`None`), existing configs, datasets,
   collation, `MLMTask`, evals, and checkpoints behave identically with zero overhead.
5. **Clean decay-branch ergonomics:** Works out of the box with `oplm train`, `oplm slurm generate`,
   `train.branch_from`, DDP, HSDP, Muon/μP, and mixed `sample_members: true` sources (e.g., OAS
   antibodies and `uniref70`).

## Non-goals

- **Offline `msa-pairformer@1.0.2` → sharded Parquet packing inside `oplm`.** Extraction and
  sharding run in a flyloft campaign/dataset task on `coreweave-barcelona`, targeting the Parquet
  schema defined here.
- **GREMLIN / Potts counterfactual perturbation distillation (A2)**, including a per-row
  `use_hard_ce` toggle. The `distill:` block leaves room for it later, but it is not built now.
- **Auxiliary heads** (pairwise contact heads, PSSM heads, attention distillation) or any changes to
  `OplmModel` / `OplmForMaskedLM`.
- **Distillation in eval dataloaders.** Sequence and structure evals always ignore `cfg.distill`.
- **Running the UniRef50 extraction or 170M proxy decay bakeoff** (separate execution work after
  this PR lands).

## Anti-goals

- **Silent amino-acid channel scrambling.** `msa-pairformer@1.0.2` emits 20 channels in
  `ARNDCEQGHILKMFPSTWYV` order, while `OplmTokenizerFast` stores canonical AAs at token IDs `4..23`
  in ESM-C order `LAGVSERTIDPKQNFYMHWC`. Never assume channel `i` matches vocab ID `4 + i`.
- **Silent bypass of a configured `Neff` gate.** If a distilled shard has `teacher_logits` and the
  config sets `min_neff` or `min_neff_sqrt_l`, lacking the corresponding column must raise, never
  silently pass every row through.
- **Breaking mixed `sample_members: true` anneal mixes.** Antibody OAS (`antiref*`) and `uniref70`
  sources use `sample_members: true` in the same decay mix as `uniref50_msa`. Enabling `distill:`
  must not reject `sample_members: true` on sources that carry no `teacher_logits`.
- **Changing mask draws between control and distilled arms.** With the same seed and sequence
  stream, enabling `distill:` must produce the exact same masked positions and 80/10/10
  replacements as a hard-MLM control.

## Constraints

- Python 3.11+, `from __future__ import annotations`, full type hints, `ruff check src/`,
  `ty check src/`, and `tests/test_docs.py` clean.
- No changes to `src/oplm/model/` (the model forward already returns fp32 logits when `labels=None`).
- Must work with `data.pack_sequences: true`, `train.batch_tokens` (`TokenBatcher`),
  `model.packed_attention: flex`, `train.compile: true` with `compile_dynamic: false`, and
  `Trainer._canon_self_check` (which slices every tensor whose `shape[0] == rows` to `[:4]`).
- Must preserve `train_data_fingerprint`, cursor resume, `train.branch_from`, and `serialize_config`
  round-tripping through `load_config`.

## Approaches

Two viable designs were considered (plus two ruled-out data-path variants):

1. **Flat `train.distill_*` fields + in-place extension of `MLMTask`.**
   Adds five flat fields to `TrainConfig` (`distill_alpha: 0.0`, `distill_temperature`, etc.) and an
   `if distill_alpha > 0:` branch inside `MLMTask.step`. Smallest diff (~150 LOC, zero new classes
   or top-level YAML sections), but crowds `TrainConfig` (~45 fields already) and mixes distillation
   logic into vanilla `MLMTask`.
2. **Optional top-level `distill:` config block (`DistillConfig | None = None`) + `DistillMLMTask(MLMTask)` subclass (selected).**
   Mirrors `fold: FoldRunConfig | None = None` on `OplmConfig` and plugs into the existing
   `TrainTask` seam (`Trainer.__init__` selects `DistillMLMTask` when `task is None and cfg.distill
   is not None`). Adds ~30 lines over Approach 1, keeps `TrainConfig` and vanilla `MLMTask`
   untouched, makes decay-branch configs self-contained (`distill: { alpha: 0.5, ... }`), and leaves
   a clean seam if A2 or other distillation knobs are added later.

**Ruled out:**
- *Sidecar lookup of per-chain `<id>/logits.parquet` at dataloader time:* random file I/O bottleneck
  at millions of sequences and breaks `train_data_fingerprint` / shard-streaming semantics.
- *Sparse flat `(N_distill, 20)` batch tensors:* breaks `Trainer._canon_self_check` (which slices
  `shape[0] == rows` tensors to `[:4]`) and introduces variable-length tensors into fixed-shape
  token batches, while dense `(R, T, 20)` float16 is only 2.6 MB at `batch_tokens=65536`.

## Design

### 1. Parquet data contract

Training shards follow the existing `ShardedProteinDataset` format (`sequence_id: str`,
`sequence: str`), with three optional columns read only when `cfg.distill is not None`:

| Column | Arrow type | Semantics |
| --- | --- | --- |
| `teacher_logits` | `binary` or `large_binary` | Flat little-endian float16 (`<f2`) bytes of shape `(len(sequence), 20)` (`len(sequence) * 40` bytes), channels ordered by `distill.teacher_alphabet`. `null` on rows with no teacher targets. Any position whose 20-vector contains a non-finite value (`NaN` / `±inf`, e.g., uncomputed positions in `K=2` extraction) is treated as having no teacher target and falls back to hard CE. |
| `teacher_neff` | `float32` or `float64` | Optional scalar selected-MSA `Neff`. Read only when `distill.min_neff` is set. |
| `teacher_neff_sqrt_l` | `float32` or `float64` | Optional scalar selected-MSA `Neff / sqrt(L)`. Read only when `distill.min_neff_sqrt_l` is set. |

**Validation and gating rules (per source / shard):**

- Sources whose shards lack `teacher_logits` (e.g., `deepclust70`, OAS antibodies, `uniref70`) are
  allowed in the same `InterleavedDataset` mix; every row yields `teacher_logits = None` silently.
- If a source has `sample_members: true` **and** any of its shards contains `teacher_logits`, raise
  `ValueError` at dataset init (teacher logits belong to the representative sequence, not sampled
  cluster members). A `sample_members: true` source without `teacher_logits` (such as OAS
  antibodies) works normally.
- If a shard contains `teacher_logits` and `distill.min_neff` (or `distill.min_neff_sqrt_l`) is
  set, that shard **must** contain `teacher_neff` (or `teacher_neff_sqrt_l`); otherwise raise
  `ValueError` at dataset init.
- On a row from a shard with `teacher_logits`: if a configured `min_neff` or `min_neff_sqrt_l`
  threshold is not met—including when the scalar value is `None` or non-finite (`NaN`)—surface
  `teacher_logits = None` so the row trains with plain hard MLM CE. When both thresholds are set,
  both must pass.
- On a row with non-`None` `teacher_logits`, byte length must equal `len(sequence) * 40` (checked
  in the collator before truncation); any mismatch raises `ValueError`.
- Sequences longer than `max_length - 2` truncate `teacher_logits` to the first `max_length - 2`
  residues, matching `input_ids` and `align_per_residue`.

### 2. Configuration (`src/oplm/config.py`)

Add `DistillConfig` and attach `distill: DistillConfig | None = None` to `OplmConfig`:

```python
@dataclass
class DistillConfig:
    alpha: float = 0.5
    temperature: float = 1.5
    teacher_alphabet: str = "ARNDCEQGHILKMFPSTWYV"
    min_neff: float | None = None
    min_neff_sqrt_l: float | None = None
```

Validation in `DistillConfig.__post_init__`:
- `alpha`: finite, `0.0 <= alpha <= 1.0`. (`0.0` is permitted for exact parity checks against
  vanilla `MLMTask`.)
- `temperature`: finite, `> 0.0`.
- `teacher_alphabet`: string of length 20 whose character set equals the 20 canonical amino acids
  (`set("LAGVSERTIDPKQNFYMHWC")`, no duplicates).
- `min_neff`, `min_neff_sqrt_l`: when not `None`, finite and `>= 0.0`.

Integration in `config.py`:
- `load_config`: normalize `dict` → `DistillConfig(**...)` (mirroring `cfg.fold`), and reject
  `cfg.fold is not None and cfg.distill is not None` with `ValueError` (also checked via a shared
  validator called from `Trainer.__init__` so directly constructed configs cannot bypass it).
- `serialize_config`: emit `"distill": asdict(cfg.distill)` when `cfg.distill is not None` so
  saved run and checkpoint `config.yaml` files round-trip cleanly through `load_config`.
- Packaged base YAML files (`model/base.yaml`, `train/base.yaml`, `data/base.yaml`) stay unchanged:
  `distill` defaults to `None`.

### 3. Dataset and dataloader (`dataset.py`, `loaders.py`)

- **`ShardedProteinDataset.__init__`** gains optional keyword-only parameters:
  - `load_teacher_logits: bool = False`
  - `min_teacher_neff: float | None = None`
  - `min_teacher_neff_sqrt_l: float | None = None`
  At init, inspect each shard's Arrow schema once (already done for `masking_weights` and
  `sample_members`), recording `_shard_has_teacher_logits: list[bool]` and enforcing the
  `sample_members` and `Neff`-column guards above.
- **`ShardedProteinDataset._read_rows`**: when `load_teacher_logits` is true and the shard has
  `teacher_logits`, append `"teacher_logits"` (and any needed `Neff` column) to `pq.read_table`'s
  `columns`. Apply the `Neff` gate before attaching `row["teacher_logits"]` (bytes or `None`). On
  shards without `teacher_logits`, attach `row["teacher_logits"] = None` (or omit—collator treats
  absent and `None` identically).
- **`build_train_dataloader`** passes `load_teacher_logits = cfg.distill is not None`,
  `min_teacher_neff`, and `min_teacher_neff_sqrt_l` to each `ShardedProteinDataset`, and
  `load_teacher_logits = cfg.distill is not None` to `MLMCollator`.
- **`build_sequence_eval_dataloader`** is unchanged (never loads teacher logits).

### 4. Collation and packing (`collate.py`)

- **`MLMCollator.__init__`** gains `load_teacher_logits: bool = False`.
- When `load_teacher_logits=True`:
  1. Run `tokenize_and_pad` and `_mask_batch` **first and identically** to vanilla MLM—no extra RNG
     draws before or during masking.
  2. Allocate `teacher_logits = torch.zeros((B, T, 20), dtype=torch.float16)` and
     `teacher_mask = torch.zeros((B, T), dtype=torch.bool)`.
  3. Build `canonical_label_mask = torch.isin(labels, self._canonical_ids)` (`(B, T)` bool; note
     `-100` is automatically `False`, as are non-canonical AAs `X/B/U/Z/O`).
  4. For each row `b` where `raw = item.get("teacher_logits")` is not `None`:
     - Validate `isinstance(raw, (bytes, bytearray, memoryview))` and `len(raw) == raw_len * 40`
       (else raise `ValueError` naming the row and lengths).
     - Find masked canonical positions `pos = canonical_label_mask[b].nonzero(as_tuple=False).squeeze(-1)`.
       (All such positions satisfy `1 <= pos <= clipped_len` because `<cls>` and `<eos>` are
       non-maskable.)
     - If `pos.numel() > 0`: view `raw` via `np.frombuffer(raw, dtype="<f2").reshape(raw_len, 20)`,
       slice `picked = arr[pos.numpy() - 1]` (copying only masked positions, not the full `(L, 20)`
       array), check `finite = np.isfinite(picked).all(axis=-1)`, write the finite rows into
       `teacher_logits[b, pos[finite]]`, and set `teacher_mask[b, pos[finite]] = True`.
  5. Emit `"teacher_logits"` and `"teacher_mask"` in the returned batch dict.
  6. In **`_pack`**: when `teacher_logits` and `teacher_mask` are present, pack them into
     `(len(rows), width, 20)` float16 and `(len(rows), width)` bool alongside `labels` using the
     exact same `start:end` slice loop.
- **Shape and compatibility check:**
  - Every new batch tensor has `shape[0] == rows` (packed) or `B` (unpacked).
  - `TokenBatcher` passes the collator output through unchanged.
  - `DeviceDataLoader` moves both new tensors to GPU automatically.
  - `Trainer._canon_self_check` slices both to `[:4]` automatically alongside `input_ids` and
    `labels`.
  - When `load_teacher_logits=False`, neither key is emitted.

### 5. Training task and loss (`task.py`, `trainer.py`)

Add **`DistillMLMTask(MLMTask)`** in `src/oplm/training/task.py`:

- **`__init__(self, distill_cfg: DistillConfig)`**:
  - Stores `distill_cfg`.
  - Resolves `self._teacher_token_ids: Tensor` (`(20,)` long) via
    `[tok.convert_tokens_to_ids(aa) for aa in distill_cfg.teacher_alphabet]` against
    `get_tokenizer()`, caching a per-device copy lazily on first step so there is no H2D transfer
    per micro-batch.
- **Inherits** `build_model`, `build_dataloader`, and `flops_per_token` unchanged from `MLMTask`,
  and satisfies `isinstance(self.task, MLMTask)` in `Trainer.__init__`.
- **`step(self, model, batch) -> StepResult`**:
  1. Forward: `outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
     labels=None, segment_ids=batch.get("segment_ids"))`. `logits = outputs["logits"]` is already
     fp32 `(R, T, V)` (`OplmForMaskedLM.forward` casts to `.float()`).
  2. Masks: `labels = batch["labels"]`, `masked = labels != -100`,
     `teacher_mask = batch["teacher_mask"] & masked`.
  3. Zero-masked guard: if `n_masked == 0`, return `loss = logits.sum() * 0.0` (preserves autograd
     graph for DDP/FSDP2) and empty metrics.
  4. Hard CE over all `V` vocab classes (fp32):
     `ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100,
     reduction="none").view_as(labels)` (zero at non-masked positions).
  5. Soft KL over the 20 canonical channels in `teacher_alphabet` order (fp32), computed only on
     `teacher_mask` positions when `n_distill > 0`:
     - `s20 = logits[teacher_mask].index_select(-1, token_ids)` (`(N_d, 20)`)
     - `t20 = batch["teacher_logits"][teacher_mask].float()` (`(N_d, 20)`)
     - `T = self._distill.temperature`
     - `kl = F.kl_div(F.log_softmax(s20 / T, dim=-1), F.softmax(t20 / T, dim=-1),
       reduction="none").sum(dim=-1) * (T * T)`
     - Notice: `DistillMLMTask.step` executes outside `torch.compile(model)` (only `model` is
       compiled), so boolean indexing `logits[teacher_mask]` never triggers graph recompilation
       under `compile_dynamic=False`.
  6. Per-masked-token blended loss (every masked token has total weight `1.0`):
     - `plain_ce_sum = ce[masked & ~teacher_mask].sum()`
     - `distill_ce_sum = ce[teacher_mask].sum()`
     - `kl_sum = kl.sum()` (or `0.0` if `n_distill == 0`)
     - `loss = (plain_ce_sum + (1.0 - alpha) * distill_ce_sum + alpha * kl_sum) / n_masked`
  7. Metrics (`StepResult.metrics`, logged by `Trainer` under `train/*`):
     - `loss_ce`: `float((plain_ce_sum + distill_ce_sum).detach().item() / n_masked)`
     - `distill_frac`: `float(n_distill / n_masked)`
     - `loss_kl`: `float(kl_sum.detach().item() / n_distill)` — included only when `n_distill > 0`
       so micro-batches without distilled tokens do not drag down the window average with fake
       zeros.
- **`Trainer.__init__`** (`src/oplm/training/trainer.py`):
  - Select `DistillMLMTask(cfg.distill)` when `task is None and cfg.distill is not None`, else
    `MLMTask()`.
  - Validate `cfg.fold is None or cfg.distill is None` alongside `validate_parallelism_compat` and
    `validate_token_batching`.

### 6. Documentation and tests

**Docs to update in the same PR:**
- `docs/data.md`: document the optional `teacher_logits`, `teacher_neff`, `teacher_neff_sqrt_l`
  Parquet columns, channel-order contract, `NaN` partial-validity convention, and `sample_members`
  interaction.
- `docs/config.md`: document the `distill.*` section (`alpha`, `temperature`, `teacher_alphabet`,
  `min_neff`, `min_neff_sqrt_l`).
- `docs/training.md`: brief *Teacher-logit distillation* subsection covering the blended loss,
  logged `train/loss_ce`, `train/loss_kl`, `train/distill_frac` metrics, and decay-branch usage via
  `train.branch_from`.
- `tests/test_docs.py`: add `"distill": DistillConfig` to `test_documented_config_fields_exist`.

**Tests (fast suite, mirroring `src/` layout):**
- Config (`tests/training/test_config.py`): `DistillConfig` validation (`alpha`, `temperature`,
  invalid/duplicate `teacher_alphabet`, negative `min_neff*`), `fold` + `distill` mutual exclusion,
  `load_config` + `serialize_config` round-trip.
- Dataset (`tests/data/sequence/test_dataset.py`): mixed `InterleavedDataset` with one distilled
  source + one plain source + one `sample_members: true` plain source; `sample_members: true` on a
  source *with* `teacher_logits` raises; `min_neff` / `min_neff_sqrt_l` gating (including `None` /
  `NaN` scalar values and missing column error).
- Collation (`tests/data/sequence/test_collate.py`): identical mask draws with vs without
  `load_teacher_logits`; truncation alignment; byte-length mismatch raises; non-canonical masked
  tokens excluded from `teacher_mask`; `NaN` partial-validity positions (`K=2` simulation) excluded
  from `teacher_mask`; packed vs unpacked parity (`_pack` + `TokenBatcher`).
- Task & trainer (`tests/training/test_task.py` / `tests/training/test_trainer.py`):
  - `alpha = 0.0` loss and gradients match vanilla `MLMTask` identically (including mixed distilled
    + plain batches).
  - Exact channel-order test: feed a teacher logit spike on `"A"` (index 0 in `ARNDCEQGHILKMFPSTWYV`,
    token ID 5 in `OplmTokenizerFast`, **not** ID 4 which is `"L"`) and verify KL drives student
    logit 5, not logit 4.
  - Packed vs unpacked `DistillMLMTask.step` loss parity.
  - `n_distill == 0` micro-batch (no `loss_kl` metric, finite CE loss, valid grad) and `n_masked == 0`
    guard.
  - Small end-to-end `Trainer` smoke test (with `pack_sequences=true`, mixed distilled + plain
    sources, and `branch_from` from a plain MLM checkpoint into a `distill:` decay branch).

## Left to the builder

- Exact helper decomposition inside `collate.py` and `task.py` and exact error message wording.
- Whether `validate_distill_compat` lives as a small helper next to `validate_token_batching` in
  `src/oplm/config.py` or inline in `load_config` / `Trainer.__init__`.
- Exact phrasing of the `docs/{data,config,training}.md` sections (keeping `tests/test_docs.py`
  passing).
