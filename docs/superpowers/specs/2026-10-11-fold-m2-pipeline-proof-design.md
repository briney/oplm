# Structure Head Milestone 2 — Pipeline Proof Addendum

**Date:** 2026-10-11
**Status:** Design agreed in discussion on 2026-10-11. Addendum to the
[structure prediction head design](2026-10-07-structure-prediction-head-design.md)
(the "spec"), which stays the design authority. Three implementation plans
follow this document: M2a data, M2b losses and training, M2c eval and the proof
runs. Milestones 0 and 1 are merged and accepted (`docs/FOLD.md` §6–§7).
**Scope:** Spec §10 milestone 2, "pipeline proof": a 24-block trunk trained on a
frozen language model at crop 384, on a PDB subset, with held-out monomer
TM-score improving across checkpoints, resume exercised on Slurm, and the pair
width chosen from accuracy and the milestone-0 cost table.

## 1. What this addendum does

The spec assigned M2 the sections §4.2–4.4, §4.7, §6.1–6.3, §7.1–7.3 and §8.
This document records the decisions the spec left to M2 (§10's decision gates),
the deviations the proof takes from the spec and where each deferred item lands,
and the split into three plans. It does not restate the spec's designs; each
plan cites the spec section it implements and this document for the deltas.

The proof is shaped by one rule: build the representation, storage and
featurization in the spec's full generality, so that later milestones are data
and loss changes, and build downstream of that only what a single-chain,
crop-384 monomer proof exercises.

## 2. Decisions against the spec

| Topic | Spec | M2 decision |
|---|---|---|
| Frozen LM (§1, §4.5, §6.6) | "an existing OPLM checkpoint" | No OPLM checkpoint exists yet (scaling runs in progress). The proof uses **ESM-C 600M** behind the hidden-state contract (§3 below). The spec's sentence is amended to "any frozen LM behind the hidden-state contract", which §1 already promises for later LM swaps. |
| Head initialization | not stated | **From scratch.** The shim's per-layer norms and projections are LM-specific, so no warm start from ESMFold2-Fast is possible against a different LM; a warm start would also pin width 256 and defeat the width selection. |
| Training sources (§7.1, §10) | "PDB/CCD/distillation subset" | **wwPDB only**, the whole archive, protein chains only (§4.4 filters). Distillation sources (OpenFold3 monomers, AFDB dimers) move to the start of milestone 3 as their own data plan. |
| Training examples (§1 "multimers from the start", §6.2) | multi-chain crops with symmetry handling | **Single chains.** One protein chain per example, contiguous and spatial crops inside that chain. The on-disk representation stores full complexes, so multi-chain training in M3 is a loader and loss change only. |
| Loss terms (§6.2) | five terms | **Four**: diffusion MSE, smooth lDDT, distogram, confidence. The bond term has no data before milestone 5 and is not implemented. Interface weighting is deferred with multi-chain training. |
| Memory plan (§5.5) | nested loop- and block-level checkpointing; 2048-token acceptance | **Block-level checkpointing only** (wired in M1). Loop-level checkpointing and the 2048-token acceptance belong to milestone 3, where they are measured. M2b records the measured peak at crop 384 for both widths. |
| Run shape (§6.3) | DDP, global batch 128 | **One node, 8× B200, DDP**, rendered by `oplm slurm generate`. Two runs, widths 128 and 256. Per-device batch, accumulation and step budget are set from the M2b measurement and recorded in the run configs. |
| Evaluation (§8) | CAMEO, CASP14, CASP15; FoldBench; baselines | **Monomer only**: CAMEO post-cutoff and CASP14, both drawn from our own PDB snapshot through target-list manifests. CASP15, FoldBench, SAbDab and the same-harness ESMFold2 baselines stay in milestone 3. |
| Config (§3 "structure datasets appear under `data.train` with new entry types") | new entry types | **No new entry field.** Every `data.train` entry of a fold run is a structure shard directory; `FoldTask` interprets them. A type field arrives only if a run ever mixes sequence and structure sources. |
| Publishing (§11) | release deliverable | **Pulled forward to M2a.** The processed PDB shards, index, manifest and dataset card are published to Hugging Face before the proof runs, and the runs train from the published revision. |
| Clustering and exclusions (§7.2) | MMseqs2 40 % clusters; homology exclusions | **Uniform sampling** (allowed for pipeline proof) and **exact-sequence exclusion** of held-out targets from the train split. Cluster weights and homology exclusions land in milestone 3 as a dataset revision. |

Consequences recorded for later milestones:

- A head trained on ESM-C states is **not transferable to OPLM**. M2's accepted
  artifacts are the pipeline, the published dataset, the measured costs and the
  width decision. The proof reruns on the first OPLM checkpoint as the start of
  milestone 3.
- Monomer training without the atom-symmetry correction of §6.2 carries bounded
  noise on the few side-chain atoms with symmetric names (for example ASP OD1/OD2,
  PHE ring atoms). It is a known deviation, not a bug to fix in M2.
- The width decision is made on ESM-C states. The decision gate in spec §10 is
  closed by M2c on that basis and revisited only if the OPLM rerun contradicts it.

## 3. Frozen LM for the proof: ESM-C

The head never sees a language model directly; it consumes a stack of per-layer
hidden states of shape batch × tokens × states × width (`lm_shim.py`), with the
state count and width in `FoldConfig.lm_num_hidden_states` and
`FoldConfig.lm_hidden_size`. The featurizer's LM rows already use ESM-C's token
ids (the OPLM tokenizer is ESM-C compatible: BOS 0, PAD 1, EOS 2, X 24, mask 32).

- **Adapter.** `src/oplm/fold/esmc.py` wraps the `esm` package's `ESMC` in a frozen
  module that presents the same call `frozen_lm_hidden_states` makes today
  (`input_ids`, `attention_mask`, `output_hidden_states`, `return_dict`) and returns
  an object with `.hidden_states`. `lm_state_count` dispatches on the wrapped type.
  The adapter emits the per-block states the package returns; whether the embedding
  output is prepended to match the OPLM convention is settled in the M2b plan from
  the package source, and the resulting count is what `lm_num_hidden_states`
  records. The LM runs in bf16 on CUDA under `no_grad` and eval, as §5.4 requires.
- **Identifier.** `FoldConfig.lm_name_or_path = "esmc:esmc_600m"` (the package's
  model name after the prefix). `load_frozen_lm` resolves the `esmc:` prefix through
  the adapter; OPLM paths and the M1 `<repo>#esmc` form behave as before.
- **Model.** ESM-C 600M: 36 blocks, width 1152, Cambrian Open License Agreement. The
  adapter accepts the 300M and 6B names too; 6B is non-commercial and roughly
  triples the frozen-LM cost per step, so it is a config flag, not the default.
  The EvolutionaryScale weights are gated behind accepting the license on
  Hugging Face; that acceptance is a one-time prerequisite for any job that loads
  them.
- **Dependency.** New optional extra `esmc = ["esm==3.4.1.post1"]`. The package
  requires Python ≥ 3.12 and transformers < 5, both inside this repo's supported
  ranges; the M1 fixture venv (`docs/fold/b200-fixtures.sbatch`) already runs this
  combination. CPU tests use the existing fake LM from `tests/fold/helpers.py` and
  the ESM-C tests skip without the extra.
- **Chain length.** The LM sees the full chain even when the crop is shorter
  (§4.5): the LM row is the whole chain sequence and crop tokens gather their
  positions by residue index. `FoldConfig.lm_max_chain_length` bounds the chains
  the proof trains and evaluates on; the proof default is 1024. Chains above it are
  excluded from training sampling by the index filter (§4 below) and reported as
  excluded in eval coverage, never truncated (§5.5).

## 4. M2a — data

Implements spec §4.2–4.4, §4.7, §7.1–7.3 for the PDB source, and §11 publishing.
New modules under `src/oplm/fold/data/`: `mmcif.py`, `crop.py`, `dataset.py`,
`collate.py`; extensions to `ccd.py` and `featurize.py`; `oplm fold preprocess`.

**Acquisition.** The wwPDB mmCIF `divided` tree by rsync from a wwPDB mirror and
the CCD `components.cif.gz` by one download, both documented as commands rather
than wrapped in code. rsync is resumable; the preprocessing manifest records the
snapshot date, mirror, per-file checksums and the CCD version and checksum, which
is the verification §7.1 asks for. Raw archives stay local: the PDB is already
public and the manifest pins the snapshot.

**Parsing (`mmcif.py`).** One entry becomes a `MolecularComplex` with the flat
arrays of §4.2 (atoms, tokens, empty bonds, metadata), read by gemmi from the
gzipped file. §4.4 filters apply at parse time with the rejection reason recorded:
protein polymer chains from the asymmetric unit, dropped waters, ions, ligands and
nucleic-acid chains flagged in metadata; the token sequence from the entity polymer
sequence so unresolved residues exist as tokens with masked coordinates; first
model; highest-occupancy altloc; hydrogens dropped; resolution ≤ 9 Å; chains with
fewer than 4 resolved residues dropped. Bioassembly expansion stays deferred.

**CCD (`ccd.py`).** A parser for `components.cif.gz` yields the parent mapping
(`_chem_comp.mon_nstd_parent_comp_id`) and atom tables keyed by code, cached once
on disk. Non-standard residues with a parent collapse to the parent token and
atom set; residues without a parent become `UNK`. The packaged reference
conformers for the twenty amino acids and `UNK` stay as they are, since M1 parity
depends on them.

**Storage.** `oplm fold preprocess` parses in parallel and writes Parquet shards
of about 1 GB, **one row per entry**, with ragged list columns for atoms
(coordinates, element, atom name, charge, resolved, B-factor, atom-to-token),
tokens (type, molecule type, asym, entity, symmetry copy, residue index, center and
representative atom, atom range), bonds (empty) and per-chain sequences, plus the
metadata of §4.2. An index Parquet has **one row per chain**: entry id, chain id,
entity id, sequence, length, resolved count, resolution, release date, method,
dropped-entity flags and split. A manifest JSON records the schema, CCD version,
input checksums, preprocessing configuration, source licenses (wwPDB and CCD:
CC0 1.0) and split rules. A generated dataset card completes the published repo.

**Splits, in the index before the first publish.**

- `train`: release date before 2021-09-30, excluding entries whose chain sequence
  exactly matches a held-out target.
- `valid`: a fixed, seeded subset of post-cutoff monomer chains (entries with one
  protein chain, resolution ≤ 2.5 Å, length 50–400) for in-training eval, 64 chains.
- `cameo` and `casp14`: held out, resolved from the public CAMEO target lists for
  the first year after the cutoff (2021-10-01 through 2022-09-30) and the CASP14
  target list, against entries present in the snapshot. The plan fixes both
  downloads; the manifests record targets missing from the snapshot as excluded.

**Runtime.** `StructureDataset` subclasses `ShardedProteinDataset` and overrides
only the row reader, so rank and worker striping, epoch shuffling and the
data-exact resume contract (`set_resume_skip`, `_arm_sample_skip`) are inherited
and `InterleavedDataset` works unchanged. It strides over the index-filtered
eligible rows of each shard (split, at least one protein chain within
`lm_max_chain_length`), so every selected row yields exactly one example and the
resume arithmetic holds. For each row it samples one eligible chain uniformly,
crops and featurizes, with a generator seeded from the run seed, epoch, shard, row
and occurrence, so a resumed run restores the same example. `crop.py` provides
contiguous and spatial crops at a fixed size per stage, both multiples of 128.
`featurize()` gains a path from a cropped complex that also emits targets
(coordinates, resolved mask, B-factor, representative-atom index) and applies fresh
per-token conformer rotations when a generator is passed (§4.3); inference keeps the
raw conformers. `collate.py` pads tokens to the crop and atoms to the
`atoms_per_token_budget` layout of §4.7 (default 24 per token), rejecting an
over-budget example before collation rather than truncating it. Token padding,
atom padding, reference-geometry validity and resolved masks stay distinct.

**Publishing.** One dataset repo for the PDB source under the project's
Hugging Face namespace, name fixed in the M2a plan, uploaded with
`huggingface_hub`'s resumable large-folder upload (already a dependency) and
revision-tagged. Training reads local files only: a cluster or CoreWeave node runs
one documented download command into a data root and the `data.train` entry's
`path` points there. Later sources are separate repos.

**Tests.** Real fixtures under `tests/fixtures/fold/` with provenance notes: a
monomer, a heterodimer, a modified-residue entry with an unresolved loop and a
multi-model NMR entry, plus the CCD subset they need. The §9 data checks: parent
mapping, unresolved positions, altloc and model selection, dropped entities, index
consistency, crops, atom budgets, and standalone plus interleaved resume with
several worker counts under the supported resume-topology rules.

**Acceptance.** The whole archive preprocessed on the development machine
(32 cores, 1.6 TB free; the gzipped archive is about 60 GB and is never
decompressed to disk) with accepted and rejected counts by reason; the repo
published; a clean-environment download that loads a batch through
`StructureDataset` and the collator.

## 5. M2b — losses and training

Implements spec §6.1–6.3 for stage 1, §6.4's `FoldTask`, the `fold:` config
block and `oplm fold train`, plus the ESM-C adapter of §3.

**Training entry point.** DDP synchronizes gradients only through the wrapped
module's `forward`, so the training step cannot call sub-modules directly.
`OplmForFolding.forward` gains an optional `targets` argument. The shared prefix
(inputs embedder, shim, recurrence, readout, distogram logits) is factored out.
Without targets the parity-verified inference path runs unchanged. With targets
the training branch runs and returns the weighted loss, detached per-term metrics
and the counts the Trainer needs.

**Training branch (§6.1).**

1. LM input masking in-model on the chain rows before the frozen LM
   (`lm_input_mask_fraction` 0.1, mask id 32), so eval mode disables it.
2. Loop count from the clamped Poisson (mean 3, 1–6 loops), drawn per rank; every
   loop count uses the same parameters, so DDP needs no synchronization. Gradient
   through the last two loops and per-loop LM-pair dropout exist from M1.
3. Distogram logits from the symmetrized final pair.
4. Diffusion: per noise sample, augment and center the reference, draw the
   log-normal noise level (−1.2, 1.5, `sigma_data` 16), evaluate the denoiser once;
   microbatched over samples with shared conditioning and normalized so the
   microbatch size does not change the objective.
5. Confidence: a no-grad rollout of 20 sampler steps, then the confidence head on
   detached coordinates and detached trunk outputs; its loss trains only that head.
6. Return the weighted loss, per-term metrics, valid token count and chain count.

**Losses (`losses.py`).** Diffusion MSE after weighted rigid alignment with the
EDM weight, reusing M1's `weighted_rigid_align`; smooth lDDT; distogram
cross-entropy over the released 64 bins; confidence cross-entropies for per-atom
lDDT (50 bins), PAE and PDE (64 bins over 0–32 Å) and resolved, with protein token
frames from N, CA and C built in the same module and invalid where any frame atom
is unresolved. Starting weights from §6.2: 4.0, 1.0, 0.03, 1e-4. Everything runs
in fp32 under the existing autocast-off helper, with explicit finite behaviour for
empty masks, missing frames and degenerate alignments. The oracle is the pinned
Boltz-2 loss code, vendored under `tests/fold/oracles/` with its revision and
attribution.

**`FoldTask` (`training.py`).** `build_model` constructs the head from the `fold:`
block, attaches the LM on the process device through `load_frozen_lm`, honours
`train.init_from`, and enables block-level checkpointing. `build_dataloader` builds
a `StructureDataset` per `data.train` entry (every entry of a fold run is a
structure shard directory; no new entry type field) through `InterleavedDataset`
and the collator. `step` calls the model and returns `StepResult` with valid tokens,
chains and per-term metrics; `flops_per_token` returns `None`.

**Config and CLI.** `OplmConfig` gains a `fold:` block: the `FoldConfig` fields,
the LM identifier, revision and `lm_max_chain_length`, crop size and the
contiguous-crop probability (proof default 0.5), diffusion sample count (proof
default 8) and microbatch, confidence rollout steps (20) and the four loss weights.
`oplm fold train` parses a run config exactly as `oplm train` does and constructs
`Trainer(cfg, task=FoldTask())`. EMA uses the existing `train.ema_decay` (0.999).
Optimizer and schedule reuse the existing options with μP disabled; the plan's
starting point is AdamW at 1e-3 with 1000 warmup steps and the existing clipping
and weight-decay defaults, and the run configs are authoritative.

**Tests.** Tiny CPU config with the fake LM: gradients reach every head parameter
and none of the LM; a short overfit on one fixture chain reduces the loss; resume
restores step, data cursor and EMA; a two-process gloo DDP smoke with resume; loss
terms against the Boltz-2 oracles; rigid-transform recovery; mask and degeneracy
cases; the default MLM suite unchanged. ESM-C adapter tests skip without the extra.

**Acceptance.** CPU suite, ruff and ty clean. One slow GPU job on a single B200:
real steps at crop 384 with ESM-C 600M at both widths, recording peak allocated
memory and step time to `docs/fold/m2/`. Those numbers set M2c's per-device batch,
accumulation and step budget.

## 6. M2c — eval and the proof runs

Implements spec §8 for monomers, the lazy eval registration of §6.4, and the
milestone-2 acceptance.

**Metrics (`fold/eval/metrics.py`).** TM-score with the known residue
correspondence and TM-align's iterative superposition (not a single Kabsch fit),
and lDDT at CA and all-atom level with the standard inclusion radius and
thresholds, both with explicit masks and reference-length normalization. Oracles:
identity and rigid-transform cases, the vendored Boltz-2 lDDT, and the `tmtools`
wrapper of TM-align as an optional dev dependency, skipped when absent.

**Tasks (`fold/eval/tasks.py`).** `fold_monomer` registers in the existing eval
registry on import of the fold eval module, which `oplm fold train` and
`oplm fold eval` import, so MLM tooling never needs fold extras. In training it
runs on the `valid` subset through a `data.eval` entry of that type, sharded across
ranks and gathered as the structure task does, predicting each target with
`fold()` at recorded loops, steps, sample count and seed, and reporting TM-score,
CA and all-atom lDDT, mean pLDDT and pTM. The `EvalTask.evaluate` model annotation
generalizes to the common module interface. Offline, `oplm fold eval` runs the
full `cameo` and `casp14` manifests from an ordinary or EMA checkpoint directory
and writes per-target tables, aggregates, coverage and failures, and the sampling
metadata. Full-chain TM-score is reported; CASP's domain-based scoring is not
reproduced.

**Job rendering.** `oplm slurm generate` hard-codes the training module; it
launches the fold entry module when the config carries a `fold:` block. Requeue,
drain, auto-resume and the derived per-device batch are reused unchanged.

**The runs.** Two run configs under `configs/`, widths 128 and 256, identical
otherwise: crop 384, ESM-C 600M, 8× B200 DDP, EMA on, checkpoints with permanent
keeps, in-training eval on `valid` at a fixed cadence, training from the published
dataset revision. Per-device batch, accumulation and step budget come from the M2b
measurement and are recorded in the configs. A resume is forced at least once per
run through the drain path and verified from the logs. Every kept checkpoint is
evaluated offline on both held-out manifests.

**Acceptance and the decision.** Held-out TM-score improves across checkpoints
for both runs; resume was exercised; the width decision is recorded from held-out
accuracy at matched steps together with the M0 cost table (`docs/FOLD.md` §6).
Results, the decision, the configs and the reproduction commands land in
`docs/fold/m2/` and a new FOLD.md section.

## 7. Deferred items and where they land

| Item | Lands in |
|---|---|
| Distillation sources: OpenFold3 monomers, AFDB dimers (§7.1, §7.4) | M3, first data plan |
| Multi-chain training: interface crops, chain permutation, atom-symmetry correction, interface weighting, per-source crop probabilities (§6.2, §7.3) | M3 |
| Interface pLDDT (`complex_iplddt`, M1 deferral) | M3 |
| Bond loss (§6.2) | M5 |
| Loop-level checkpointing and the 2048-token acceptance (§5.5) | M3 |
| MMseqs2 clustering, cluster-weighted sampling, homology exclusions (§7.2) | M3, as a dataset revision |
| CASP15, FoldBench, SAbDab, same-harness ESMFold2 baselines (§8) | M3 |
| Rerun of the proof on the first OPLM checkpoint | M3 start |
| Stage 2–4 recipe knobs (§6.3) | M3 |
| M1 minor deferrals in the SDD ledger not touched by M2 | triaged per plan; none block M2 |

M1 deferrals absorbed by M2: budgeted atom padding and per-token conformer
rotations (M2a); the bf16 RoPE-table note is moot with ESM-C and is closed.

## 8. Plan order and gates

M2a, then M2b, then M2c, each with the subagent-driven loop used for M1 (per-task
review caught real bugs in five of eleven M1 tasks). M2b can start its CPU work
while the M2a archive preprocessing runs, since it tests against the fixtures, but
its GPU measurement needs a published shard set. M2c's runs need M2b's
measurement. The milestone is accepted when §6's acceptance holds and the decision
gates of spec §10 that M2 owns (pair width; LR and global batch for the proof) are
recorded.
