# Structure Prediction Head (`oplm.fold`)

Architecture contract for the folding head, in the role `MODEL_ARCHITECTURE.md`
plays for the language model. The agreed design is
[`docs/superpowers/specs/2026-10-07-structure-prediction-head-design.md`](superpowers/specs/2026-10-07-structure-prediction-head-design.md);
this file records what is implemented. **Status: milestone 2b (losses and training) accepted 2026-10-13 (measured on one B200; docs/fold/m2/); milestone 2a (data) accepted 2026-10-12 (dataset published, tag `wwpdb-2026-10-12`); milestone 1 (inference port) accepted 2026-10-11.**
Kernels, the model, prediction, the training data pipeline, the losses and the training step exist
(§7 holds the M1 acceptance evidence, §8 the data contract, §9 the losses and training; the
single-B200 cost measurement of §9 is still to be run).

## 1. Package layout (implemented so far)

```
src/oplm/fold/
├── configuration_fold.py   # FoldConfig (released ESMFold2-Fast defaults)
├── modeling_fold.py        # OplmFoldPreTrainedModel, OplmForFolding, frozen-LM ownership
├── lm_shim.py              # per-layer LN + proj, softmax layer mix, LM -> pair
├── esmc.py                 # frozen ESM-C adapter (FrozenEsmC, load_esmc; extra oplm[esmc])
├── pair.py                 # block-local pair features
├── trimul.py               # staged triangle multiplication + cuEquivariance dispatch
├── trunk.py                # pair update block, pair stack, contractive recurrence
├── atoms.py                # atom encoder/decoder, 3D RoPE, inputs embedder
├── attention.py            # pair-biased / sliding-window attention: FlexAttention + dense oracle
├── diffusion.py            # DiT, EDM schedule, sampler
├── confidence.py           # confidence + distogram heads, pTM/ipTM
├── losses.py               # diffusion MSE, smooth lDDT, distogram and confidence losses (fp32)
├── training.py             # FoldTask, build_fold_config, PeakMemoryCallback
├── train.py                # python -m oplm.fold.train entry point
├── predict.py              # fold(), mmCIF writer
├── fixtures.py             # ESMFold2 golden-fixture generator (runs in the esm venv)
├── cli.py                  # oplm fold bench-kernels | make-fixtures | measure | predict | preprocess | train
└── data/
    ├── ccd.py              # residue vocabulary, reference conformers, ChemCompTable (CCD parents)
    ├── complex.py          # MolecularComplex (flat atoms/tokens/bonds) + Parquet row
    ├── mmcif.py            # gemmi mmCIF -> MolecularComplex with the §4.4 filters
    ├── crop.py             # contiguous / spatial single-chain crops
    ├── featurize.py        # ChainSpec or MolecularComplex -> FoldFeatures (+ FoldTargets)
    ├── collate.py          # FoldBatch under the atom budget
    └── dataset.py          # oplm fold preprocess + StructureDataset (resume contract)
tests/fold/                 # mirrors the above; tests/fixtures/fold/ holds real entries
tests/fold/oracles/         # frozen Boltz-2 loss functions the losses are checked against (§9)
```

Dependency direction: `oplm.fold` imports core packages; core imports nothing
from `oplm.fold` except the CLI registration in `oplm/cli.py`. Ported modules
name their Biohub source in the module docstring; `THIRD_PARTY_NOTICES.md`
carries the Apache 2.0 text.

## 2. Triangle multiplication (`oplm.fold.trimul`)

Parameters match ESMFold2's `TriangleMultiplicativeBlock` exactly:
`norm_start`, `norm_mix` (LayerNorm, eps 1e-5), `proj_bundle` (`D -> 4D`, rows
`[0:2D]` signal / `[2D:4D]` gate logits), `proj_emit`, `proj_gate` (no biases).
The forward is three explicit stages:

1. `trimul_pre` (pointwise): fp32-internal LayerNorm, signal and gate projections,
   sigmoid gate, pair mask → `left`, `right`, `zn`.
2. `trimul_contract` (the only cross-row/column op): outgoing
   `out[i,j] = Σ_k left[i,k]·right[j,k]`, incoming `out[i,j] = Σ_k left[k,i]·right[k,j]`,
   fp32 accumulation, chunked over output rows (`chunk_size`, default 64). Chunking
   bounds forward/no-grad memory only; under autograd the reference saves both fp32
   operands, which the `bench-kernels` reference-path peak numbers will show. The
   pair block must be square (`I == J`).
3. `trimul_post` (pointwise): fp32-internal LayerNorm, `proj_emit`, output gate.

Stages 1 and 3 are block-local by construction (tested on random sub-blocks), which
is the discipline later Fold-CP sharding relies on (spec §5.6).

Dispatch (`resolve_trimul_path`) is explicit, per call, from device/dtype/width/grad:

| Path | When | Backward |
| --- | --- | --- |
| `reference` | CPU, no cuEquivariance, width not a multiple of 32, or `backend="reference"` | autograd through the staged reference |
| `fused_autograd` | CUDA + cuEquivariance, `backend` in `auto`/`fused`, or any no-grad call | the library's own backward |
| `fused_forward_reference_backward` | `backend="fused_forward_reference_backward"` | fused forward saved as a single recomputation boundary; the reference graph is rebuilt inside `backward` and freed after it |

`TriangleMultiplication.backend` is the knob; `"auto"` trusts the library's
autograd, and the B200 run in §6 confirmed that trust for both widths with
cuEquivariance 0.12: `auto` is the setting for widths 128 and 256, and the mixed
path stays as the measured fallback (same forward, reference-cost backward).
Record the choice in each stage config. Residual add and row-shared dropout
belong to the owning pair block (milestone 1), as upstream.

## 3. Attention primitives (`oplm.fold.attention`)

- `pair_biased_attention(q, k, v, bias, key_mask)`: logits `q·k/√d + bias[b,h,i,j]`,
  keys masked by `key_mask`; a batch row with no valid key returns zeros.
- `sliding_window_attention(q, k, v, valid, half_window)`: window measured in rank
  among *valid* atoms (`rank = cumsum(valid) - 1`), self always visible, invalid
  outputs zeroed — ESMFold2's reference-space semantics, identical to its
  FlashAttention varlen path.

`backend="auto"` is FlexAttention (compiled, block masks) on CUDA and the dense
formulation on CPU; FlexAttention has no CPU backward. Q/K/V dtypes are never
changed by these functions. Gradient parity (incl. the bias) is a slow GPU test.

**M1 entry criterion.** `_compiled_flex` uses `torch.compile(..., dynamic=False)` and both
functions rebuild their `BlockMask` per call. Every new `(B, N, dtype)` recompiles, and past
dynamo's recompile limit flex silently falls back to eager, which materializes the full
score matrix. Before `AttentionPairBias`/atom attention call these in a training loop, M1
must bucket shapes (crops are multiples of 128), accept a precomputed `BlockMask`, and raise
`torch._dynamo.config.recompile_limit` as needed.

## 4. Training seam and EMA

The Trainer builds the model/dataloader and runs a step through a `TrainTask`
(`docs/TRAIN.md` §17); the structure head's `FoldTask` is described in §9. `train.ema_decay`
keeps an EMA of the trainable weights, updated once per optimizer step, saved as `ema.pt` and
exported as `hf_ema/` with every checkpoint (`docs/TRAIN.md` §16). The frozen LM is an
unregistered attribute of the fold model that `__deepcopy__` shares instead of copying, so the
EMA never holds a second LM (milestone 1).

## 5. Kernel benchmark

```bash
oplm fold bench-kernels --out bench-kernels.json            # B200 defaults: widths 128,256; lengths 384..2048
oplm fold bench-kernels --paths reference --compile-reference --out bench-reference-compiled.json
oplm fold bench-kernels --device cpu --dtype fp32 --widths 32 --lengths 16 --paths reference
```

On the SUNK cluster, [`docs/fold/b200-task7.sbatch`](fold/b200-task7.sbatch) runs the whole
milestone-0 acceptance checklist (GPU parity tests, both benchmarks, the cuda-bf16 MLM/EMA
regression files) on one B200 inside the standard Pyxis container, installing the branch
editable from a clone or a mounted checkout; its header lists the environment overrides.

Per (width, length, direction, path) the report records forward, forward+backward
and checkpointed forward+backward ms/iter, peak allocated/reserved GiB, the path
that actually ran (`"unavailable"` when the request cannot resolve on the machine),
whether the reference was `torch.compile`d, and the max abs error against an fp32
reference (lengths ≤ 512). The environment block pins host, GPU,
torch/CUDA/cuEquivariance versions and git revision. Results for the target B200
are recorded in §6 once measured.

## 6. Measured results

Milestone-0 acceptance run on one NVIDIA B200 (`slurm-b200-193-055`, 2026-10-09,
branch at `918a094`, via [`docs/fold/b200-task7.sbatch`](fold/b200-task7.sbatch)):
torch 2.11.0+cu130, CUDA 13.0, cuequivariance / cuequivariance-torch /
cuequivariance-ops-torch-cu13 0.12.0, triton 3.6.0, transformers 5.3.0,
accelerate 1.13.0 ([`docs/fold/pip-freeze.txt`](fold/pip-freeze.txt)). Bench
settings: bf16, 5 timed iterations after 2 warm-ups, reference chunk size 64,
batch 1. Full reports: [`docs/fold/bench-kernels-b200.json`](fold/bench-kernels-b200.json)
and [`docs/fold/bench-kernels-b200-compiled-reference.json`](fold/bench-kernels-b200-compiled-reference.json).

**Every case ran** (`status: ok`) at both widths, both directions, and all three
paths, including the library's own backward (`fused_autograd`) at width 256, so
the width-aware fallback the plan held in reserve is not needed.

**Parity and regression tests** ([`docs/fold/status.txt`](fold/status.txt)):

- `tests/fold -m slow`: 28 passed ([`gpu-tests.log`](fold/gpu-tests.log),
  [`fold-gpu-tests.xml`](fold/fold-gpu-tests.xml)): both fused trimul paths
  against the fp32 reference at widths 128 and 256, lengths 128/512/1024, both
  directions (24 cases), and compiled FlexAttention against the dense oracle for
  pair-biased and sliding-window attention in fp32 and bf16 at 256 and 512 tokens,
  including gradients into Q, K, V, and the pair bias (4 cases).
- `tests/training/test_e2e_logging.py`, `test_e2e_precision.py`, `test_e2e_ema.py`
  on cuda-bf16: 7 passed, 1 failed ([`mlm-regression.log`](fold/mlm-regression.log)).
  The failure was in the test, not the trainer: `test_ema_counts_optimizer_steps_and_survives_resume`
  compared the resumed EMA tensors (on `cuda:0`) with a sidecar loaded to CPU, and
  `torch.equal` refuses mixed devices. The restore had already passed its
  `n_averaged == 4` check; the comparison now moves the tensor to CPU first. The
  fixed file was re-run on the same container (job 25270, 2026-10-09): 3 passed.

Outgoing direction (incoming is within 5% everywhere); ms per iteration, peak
*allocated* GiB during forward+backward (peak *reserved* in the JSON is cumulative
across cases and is not a per-case number):

| Width | Tokens | Path | Forward | Fwd+bwd | Checkpointed | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 1024 | `reference` | 53.4 | 123.1 | 176.5 | 5.6 |
| 128 | 1024 | `reference` (compiled) | 48.4 | 106.5 | 154.8 | 7.7 |
| 128 | 1024 | `fused_forward_reference_backward` | 1.1 | 124.2 | 125.3 | 6.1 |
| 128 | 1024 | `fused_autograd` | 1.1 | 7.7 | 8.7 | 3.0 |
| 128 | 2048 | `reference` | 424.9 | 944.7 | 1374.7 | 22.1 |
| 128 | 2048 | `reference` (compiled) | 407.8 | 852.5 | 1259.2 | 63.7 |
| 128 | 2048 | `fused_forward_reference_backward` | 4.9 | 949.7 | 954.4 | 24.1 |
| 128 | 2048 | `fused_autograd` | 4.9 | 32.4 | 37.3 | 12.1 |
| 256 | 1024 | `reference` | 107.2 | 251.2 | 358.1 | 11.1 |
| 256 | 1024 | `reference` (compiled) | 99.8 | 223.2 | 323.1 | 15.3 |
| 256 | 1024 | `fused_forward_reference_backward` | 4.4 | 255.5 | 259.8 | 12.1 |
| 256 | 1024 | `fused_autograd` | 4.2 | 19.6 | 23.9 | 6.0 |
| 256 | 2048 | `reference` | 864.8 | 1961.1 | 2821.0 | 44.2 |
| 256 | 2048 | `reference` (compiled) | 842.4 | 1820.6 | 2661.5 | 127.3 |
| 256 | 2048 | `fused_forward_reference_backward` | 20.5 | 1980.6 | 2000.0 | 48.2 |
| 256 | 2048 | `fused_autograd` | 20.9 | 75.7 | 99.8 | 24.1 |

Numerical error, max |bf16 path − fp32 reference| at 384 tokens (the bench measures
it up to 512; outgoing / incoming):

| Width | bf16 `reference` | fused forward (both fused paths) |
| --- | --- | --- |
| 128 | 1.60e-2 / 1.55e-2 | 1.03e-2 / 1.00e-2 |
| 256 | 1.73e-2 / 1.77e-2 | 1.10e-2 / 1.06e-2 |

The fused kernel is closer to the fp32 reference than the bf16 reference is, so
the parity tolerance in `tests/fold/test_trimul.py` (4× the bf16-reference error
plus a relative floor) holds with margin.

Decisions recorded from this run:

- **`backend="auto"` for both widths.** Library autograd is 25× faster than the
  reference for forward+backward at 2048/256 (76 ms vs 1961 ms) at 55% of its
  allocated peak (24 GiB vs 44 GiB), and 125× faster at 2048/128.
- **The mixed path is a fallback only.** Its forward matches `fused_autograd`;
  its backward costs the full reference recompute plus roughly 10% more memory
  than eager reference.
- **The compiled reference is not used.** `torch.compile` buys about 10% on time
  but allocates 1.4–2.9× more than the eager reference (127 GiB vs 44 GiB at
  2048/256), which is the wrong trade for the memory-bound regime.
- **Reference cost at 2048 tokens.** 44 GiB allocated and 2.8 s per checkpointed
  step for one width-256 block: the pure-PyTorch path is an oracle, not a training
  path, at this length.

## 7. Milestone 1: ESMFold2 inference port (`oplm.fold.modeling_fold`)

**What exists.** `FoldConfig` (defaults = the released `biohub/ESMFold2-Fast` config; `docs/fold/m1/`
records the parity run), `featurize()` (protein chains -> `FoldFeatures`; one LM row per chain
with BOS/EOS; atoms padded to 32; tokens optionally padded to a crop multiple; duplicate chain
ids, including `ID_2` copy suffixes, raise `ValueError`),
`OplmForFolding` (checkpoint-identical module names; `base_model_prefix` is `oplm_fold`; frozen
LM held outside the module tree via `attach_lm` / `lm_name_or_path`;
`forward(features) -> FoldOutput`), `fold()` + `write_mmcif()` and `oplm fold predict`.
`oplm fold make-fixtures` + `docs/fold/b200-fixtures.sbatch` record the parity oracle;
`tests/fold/test_parity.py` runs when `OPLM_FOLD_FIXTURES` points at it.

**Deviations from upstream and the spec, all deliberate.**
- Per-loop LM-pair dropout (`pair_dropout`) and LM input masking are training-only; upstream
  forces dropout on at inference. Inference is deterministic given `generator`.
- `inference_num_loops` counts iterations executed (upstream `num_loops + 1`); the spec default
  is 10, the released config maps to 21.
- The initial pair state and the sampler take a `torch.Generator`; upstream draws from the
  global RNG. Independently sampled structures are therefore not an oracle (spec §9).
- Padded query rows of the diffusion token transformer are zeroed (upstream leaves them finite
  garbage); interface pLDDT (`complex_iplddt`) is deferred to milestone 2.
- Loading a fold checkpoint needs `oplm` installed; `trust_remote_code` bundling is not
  provided (the featurizer, tokenizer vocabulary and LM live in this package). A config whose
  `lm_name_or_path` has the `<repo>#esmc` form (a head trained against the ESMC bundled in that
  repo) loads through `from_pretrained` with no LM attached (`load_frozen_lm` rejects that form
  with a `ValueError`); such heads run with precomputed `lm_hidden_states`.
- `fold()` ranks samples by ipTM (complex) / pTM (monomer); upstream's `fold()` does not rank.
  It requires an eval-mode model (train mode would apply `pair_dropout`).
- Coordinates, the sampler and the confidence math disable any outer autocast locally (spec
  §5.4); the trunk portion of `forward` runs under bf16 autocast on CUDA only.
- The frozen LM is cast to bf16 only when loaded through `from_pretrained`/the CLI on CUDA; an
  explicitly attached LM keeps the caller's dtype.
- Atoms pad to a multiple of 32 (upstream's layout), not spec §4.7's
  `atoms_per_token_budget × crop`; the budgeted layout arrives with milestone-2 batching.
- Reference conformers are used raw at inference (no per-token rotation, as upstream); spec
  §4.3's rotations are a training augmentation for milestone 2.

**Parity run** ([`docs/fold/m1/`](fold/m1/), SUNK job 25479, 2026-10-11). CPU fp32 oracle from
`esm==3.4.1.post1` (Python 3.12.13, torch 2.11.0+cu130 on CPU, transformers 4.57.6) against
`biohub/ESMFold2-Fast@45fe865` with the CCD from `biohub/ESMFold2@69869f7` (ESMFold2-Fast ships no
`ccd.pkl`); five cases (trp-cage, villin, GB1, a heterodimer, a homodimer with an X residue), two
recurrence loops and three sampler steps each. Every stage passed at the draft tolerances; the one
failure in the recorded log, `test_packaged_reference_conformers_match_upstream_dump`, was the
expected one (resolved below). The per-stage errors were then measured on a workstation (same
fixtures, fp32 CPU, dense attention, reference trimul) with the assertions replaced by recording.
Each `atol` is the smallest 1-2-5 round number ≥ 10× the atol the assertion needs after its `rtol`
term, floored at 1e-6. `x_inputs` and `z_init` exceeded the draft 1e-4 on the workstation but not
on the cluster: float accumulation order, growing with sequence length.
[`parity-rerun.log`](fold/m1/parity-rerun.log) / [`parity-rerun-status.txt`](fold/m1/parity-rerun-status.txt)
is the cluster rerun at the final tolerances (SUNK job 25489, same venv and fixtures, `parity`
step only): all 37 pass, the GPU test skips without CUDA.

| stage | atol | rtol | max abs err | atol needed after rtol | worst case |
|---|---|---|---|---|---|
| ref_charge, ref_pos, relpos, bonds | 0 / 1e-6 / 1e-5 / 1e-5 | 0 | 0 | 0 | exact on all five |
| lm_z (shim) | 1e-4 | 1e-4 | 1.7e-5 | 5.4e-6 | gb1 |
| x_inputs | 1e-2 | 1e-4 | 1.1e-3 | 9.9e-4 | gb1 |
| z_init | 5e-2 | 1e-4 | 2.9e-3 | 2.6e-3 | gb1 |
| lm_encoder | 1e-4 | 1e-3 | 6.1e-5 | 5.7e-6 | gb1 |
| trunk states 1-3 | 5e-3 | 2e-3 | 2.7e-3 | 2.6e-4 | gb1, trp_cage |
| readout | 2e-3 | 2e-3 | 1.7e-3 | 1.1e-4 | gb1 |
| coda | 5e-3 | 3e-3 | 1.3e-3 | 2.7e-4 | gb1 |
| distogram logits | 2e-4 | 3e-3 | 1.2e-4 | 1.5e-5 | gb1 |
| x_denoised (Å) | 1e-1 | 0 | 7.1e-3 | 7.1e-3 | trp_cage |
| pae / pde / plddt logits | 5e-5 | 3e-3 | 1.8e-5 | 4.8e-6 | gb1, villin |
| plddt | 5e-6 | 0 | 4.2e-7 | 4.2e-7 | homodimer_x |
| ptm, iptm, pair_chains_iptm | 1e-6 | 0 | 4.5e-8 | 4.5e-8 | gb1, heterodimer |

**Reference conformers.** The packaged `reference_conformers.json` is now the generator's dump of
upstream `get_idealized_atom_pos` (the CCD "Computed" conformers in `biohub/ESMFold2@69869f7`'s
`ccd.pkl`), which is what upstream's featurizer uses for protein tokens, UNK included (it gets the
CCD UNK backbone, not zeros). The milestone-1 transcription of `PROTEIN_REF_POS` was a different
conformer set (up to 6.4 Å per atom after centring, MET worst); atoms, elements and charges were
already identical. Predictions made before the swap used the wrong atom reference frame.

**End-to-end prediction** (released head, recorded ESMC states for trp-cage, 21 loops, released
sampler schedule, seed 0, CPU, 4.6 s): pTM 0.215, mean pLDDT 0.794. Upstream's recorded 3-loop /
3-step run on the same input: pTM 0.052, mean pLDDT 0.276 (not comparable: different loop and
step counts and RNG). The `write_mmcif` output read back with gemmi: chain A, 20 residues, 153
atoms. [`parity-local.log`](fold/m1/parity-local.log) is the same parity suite on the workstation
at the final tolerances (all 37 pass; the GPU test skips without CUDA).

**GPU run** ([`gpu-tests.log`](fold/m1/gpu-tests.log), [`gpu-status.txt`](fold/m1/gpu-status.txt),
SUNK job 25487, one B200, 2026-10-11): `pytest tests/fold -m slow` against the same fixtures, all 29
pass: the milestone-0 FlexAttention and fused-trimul parity tests and
`test_gpu_pipeline_tracks_the_cpu_oracle` (bf16 autocast, FlexAttention and the fused trimul end to
end on trp-cage: finite coordinates and PAE, final pair within the 5e-2 relative Frobenius bound of
the CPU oracle). That run predates the test recording its observed value; it now lands in the job's
JUnit XML as the `pair_rel_fro_err` property.

## 8. Milestone 2a: training data (`oplm.fold.data`)

Implements spec §4.2–4.4, §4.7, §7.1–7.3 and §11 for the PDB source under the
[M2 addendum](superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md) §4: whole
archive, protein chains only, single-chain training examples, published to Hugging Face.

### Representation and parsing

`MolecularComplex` (`data/complex.py`) is the flat layout of spec §4.2: atom arrays
(coordinates, resolved, element, name, charge, B-factor, atom→token), token arrays (residue
type, molecule type, asym/entity/symmetry ids, residue index, center and representative atom,
atom range), an always-present bond list (empty in v1), per-chain `ChainInfo` (label and
auth chain ids, entity, copy, full sequence, mean pLDDT for predicted data) and
`ComplexMetadata` (id, source, method, resolution, release date, dropped-entity counts). The
atoms of a token are contiguous and in the packaged template order, so training coordinates
line up with the `ref_*` features by construction (`featurize_complex` verifies the names).

`parse_mmcif` (`data/mmcif.py`, gemmi) applies §4.4 at parse time and records why an entry
was rejected (`parse_error`, `resolution`, `no_protein_chains`, `no_resolved_chain`) or what
was dropped (`water`, `ligand`, `nucleic_acid`, `other_polymer`, `short_chain`,
`empty_sequence`, `unknown`): L-peptide polymer entities of the asymmetric unit only; the token
sequence is the entity polymer sequence (unmodeled residues are tokens with `atom_resolved`
false); first model; highest-occupancy altloc per atom name; hydrogens dropped; microheterogeneity
keeps the sequence's first alternative; resolution ≤ 9 Å (NMR has none); chains with fewer than 4
resolved residues dropped. Non-standard residues collapse to their CCD parent
(`ChemCompTable.protein_parent`, from `_chem_comp.mon_nstd_parent_comp_id`) only when the
component type contains "PEPTIDE" and is not D-handed (`D-` prefix); D-peptides, non-polymer
components, multi-parent and parent-less components become `UNK`. Atoms match the parent template
by name, with AlphaFold's `MSE` `SE → SD` convention, and parent atoms the modified residue lacks
are unresolved (norleucine loses `CD1`/`CD2`). Release date is the first
`_pdbx_audit_revision_history.revision_date`. `oplm fold preprocess` records `internal_error` for
unexpected worker exceptions (with the exception message as `detail`).

### Shards, index and splits (`oplm fold preprocess`)

```bash
oplm fold preprocess --archive <mmCIF root> --ccd components.cif.gz --out <dataset dir> \
    --heldout cameo=targets/cameo.txt --heldout casp14=targets/casp14.txt [--workers N] [--ids ids.txt]
```

| Output | Contents |
| --- | --- |
| `shards/shard-NNNNN.parquet` | one row per entry (`COMPLEX_SCHEMA`), ~1 GB each, row groups of 16 rows for random access |
| `index.parquet` | one row per chain: id, chain ids, chain index, entity/sym, sequence, length, resolved count, chain count, resolution, release date, method, shard, row, `split` |
| `targets/<name>.json` | the held-out target lists as `[{"id", "chain"}]` |
| `ccd_table.json`, `checksums.txt`, `manifest.json`, `README.md` | parsed CCD table; sha256 of every archive entry file (accepted or rejected; CCD's sha256 in `manifest.json` under `ccd.sha256`); provenance/config/counts; dataset card |

Splits are entry-level: `cameo`/`casp14` (from the target lists, ids case-insensitive),
`excluded_heldout_sequence` (pre-cutoff entries sharing a chain sequence with a held-out
entry), `train` (release date ≤ 2021-09-30), `valid` (64 seeded post-cutoff monomer chains:
one protein chain, resolution ≤ 2.5 Å, length 50–400), `post_cutoff`, `undated`. Clustering
and homology exclusions are milestone 3 (a dataset revision).

The tool refuses an empty archive and requires `out/shards` and `out/targets` to be empty or
absent (a rerun needs a fresh directory; other files in `out/` are overwritten); `--cutoff` is
normalized to an ISO date, and `checksums.txt` is written with paths relative to the archive root
(so `sha256sum -c checksums.txt` works from the archive root).

### Runtime

`StructureDataset(path, split="train", crop_size=384, contiguous_prob=0.5,
max_chain_length=1024, ...)` subclasses `ShardedProteinDataset`: rank/worker striping, per-epoch
shuffling, `InterleavedDataset` mixing and the data-exact resume contract (`set_resume_skip`,
`_arm_sample_skip`) are inherited. It strides over the index-eligible rows of each shard and
yields exactly one example per row: a generator seeded from `(seed, epoch, shard, row)` picks
one eligible chain uniformly, crops it (`crop_chain`: contiguous window or spatial crop around
a resolved center; unresolved tokens fill a spatial crop last, by index) and featurizes it
with the full chain as the LM row (`lm_positions = residue_index + 1`) and fresh per-token
rotations of the centered reference conformers. Rows are `{"id": "<entry>_<chain>",
"example": FoldExample(features, targets)}`; `collate_rows(rows, crop_size=...)` pads tokens
to the crop and atoms to `24 × crop` rounded up to 32, stacks the LM rows of the batch and
offsets `lm_rows`, and rejects an over-budget example instead of truncating it.
`FoldTargets` carries coordinates, resolved mask, B-factor and the N/CA/C atom indices of
each token for M2b's frame-based losses.

### Acquisition, publishing, download

```bash
# wwPDB mmCIF archive (about 90 GB gzipped, 260k entries; read in place, never decompressed)
rsync -rlpt -v -z --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ $OPLM_DATA/pdb/mmCIF/
# Chemical Component Dictionary
curl -o $OPLM_DATA/pdb/components.cif.gz https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
# preprocess straight into the repository layout, then publish (resumable) and tag
oplm fold preprocess ... --out $OPLM_DATA/hf/wwpdb/<snapshot>
hf upload-large-folder brineylab/oplmfold-training-data $OPLM_DATA/hf --repo-type dataset
hf repo tag create brineylab/oplmfold-training-data wwpdb-<snapshot> --repo-type dataset
# pull one source on a cluster node
hf download brineylab/oplmfold-training-data --repo-type dataset --revision wwpdb-<snapshot> \
    --include "wwpdb/<snapshot>/*" --local-dir $OPLM_DATA/hf
```

Published data lives in the Hugging Face dataset repository
[`brineylab/oplmfold-training-data`](https://huggingface.co/datasets/brineylab/oplmfold-training-data):
one immutable folder per source and version (`wwpdb/2026-10-12/`; later `afdb-distill/`,
`afdb-dimers/`, `antibody-antigen/`), one tag per publish (`<source>-<version>`), and a root
card with the inventory and download recipes. `hf upload-large-folder` has no target-path
option, so the local folder mirrors the repository layout. The manifest's `snapshot_date`,
`checksums.txt` and the CCD checksum pin the inputs; the tag, or its commit, pins the artifact
a run trained on. Acceptance evidence lives in [`docs/fold/m2a/`](fold/m2a/).

**Published:** `wwpdb/2026-10-12` at tag `wwpdb-2026-10-12` (commit `0fad20d4`):
254,424 entries, 1,080,220 chains, 82 shards, 37 GiB, preprocessed from the 2026-10-12 archive
snapshot with `feat/fold-m2a` at e0ea2fd. Held-out splits: `cameo` 744 entries, `casp14` 128.

## 9. Milestone 2b: losses and training (`oplm.fold.losses`, `oplm.fold.training`)

Implements spec §6.1–6.4 for stage 1 under the
[M2 addendum](superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md) §2, §3 and §5.

**One training step** (spec §6.1) runs inside `OplmForFolding.forward(features, targets=...)` so
DDP synchronizes through one forward. In train mode each inner residue id on the LM rows is masked
with probability 0.1 (id 32; never BOS, EOS or padding), the loop count is a clamped Poisson draw per rank
(mean 3, 1–6; eval uses `inference_num_loops`), the trunk runs as at inference (gradient through
the last 2 loops, per-loop LM-pair dropout), and the distogram logits come from the symmetrized
final pair. Then `diffusion_samples` copies of the reference are centred, randomly rotated and
translated, noised at `σ = 16 · exp(N(−1.2, 1.5))` and denoised once each, in microbatches of
`diffusion_microbatch` samples. When `diffusion_microbatch` is below `diffusion_samples`, each
chunk runs under `torch.utils.checkpoint`, so only one chunk's activations are live at a time;
with the default `null` every sample is denoised in one unchecked call and activations scale with
the sample count (the proof configs use 4 of 8). The per-sample losses are averaged, so the
microbatch size does not change the objective. A `confidence_rollout_steps`-step sampler
rollout runs without gradient and the confidence head scores it on detached trunk outputs.
`FoldTrainOutput` carries the weighted loss, the detached per-term metrics, the valid token count
and the example (chain) count. The branch runs whenever `targets` is given; masking, the Poisson
draw and dropout still follow `self.training`.

**Losses** (`losses.py`; all fp32 with any outer autocast off, one value per sample, finite on
empty masks) and their conventions:

| term | definition | weight |
|---|---|---|
| diffusion MSE | reference aligned onto the prediction (detached), `Σ w ‖Δ‖² / (3 Σ w)` over resolved atoms, times the EDM weight `(σ² + σ_d²) / (σ σ_d)²` with `σ_d` = `sigma_data` = 16 | `loss_diffusion_mse` 4.0 |
| smooth lDDT | pairs under 15 Å in the reference, both atoms resolved, `i ≠ j`; `1 − mean ¼ Σ_t sigmoid(t − \|Δd\|)` for `t` in 0.5, 1, 2, 4 Å | `loss_smooth_lddt` 1.0 |
| distogram | representative-atom (CB, CA for glycine) distances in 64 bins over 2–22 Å (`distogram_{min,max}_dist`), masked mean cross-entropy | `loss_distogram` 0.03 |
| confidence | pLDDT (per-atom lDDT against the token CA atoms within 15 Å, 50 bins) + PDE (representative-atom distance error, 64 bins over 0–32 Å) + resolved (class 0 = resolved) + `loss_pae` × PAE (token CA in the N–CA–C frame of the other token, 64 bins over 0–32 Å; frames need resolved, non-collinear atoms) | `loss_confidence` 1e-4 |

The confidence terms train only the confidence head. Oracles: Boltz-2 (`boltz==2.2.1`) loss
functions vendored under [`tests/fold/oracles/`](../tests/fold/oracles/README.md) (test code only;
attribution in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)); parity, rigid-transform
recovery, mask and degeneracy tests in `tests/fold/test_losses.py`.

**Frozen ESM-C** (`esmc.py`, addendum §3): `fold.lm: esmc:esmc_600m` loads `biohub/ESMC-600M`
through the `esm` package (`oplm[esmc]`, Python ≥ 3.12; the `biohub/ESMC-*` weights are public,
MIT-licensed and ungated, so no Hub login is needed) wrapped as `FrozenEsmC`, which exposes `num_hidden_states = num_hidden_layers + 1`: the
package returns every block input (the embedding output first) plus the final post-norm output,
so 600M gives 37 states of width 1152. `lm_hidden_size` and `lm_num_hidden_states` are inferred
from the loaded LM when `fold.model` does not set them.

**Run config and CLI.** A run is an ordinary `OplmConfig` plus a `fold:` block
([`CONFIG.md`](CONFIG.md#fold-fields-fold)); every `data.train` entry is a structure shard
directory. `oplm fold train --config run.yaml [overrides]` runs one process and
`accelerate launch -m oplm.fold.train --config run.yaml` runs DDP; both build
`Trainer(cfg, task=FoldTask())`. `FoldTask.build_model` builds the head from `fold.model`
(`build_fold_config`), attaches the frozen LM on the process device (bf16 on CUDA), enables
block-level checkpointing of the pair stacks and, with `train.init_from`, loads a saved `hf/` or
`hf_ema/` head whose architecture must match the run's `fold.model`. EMA, checkpoints (`hf/` and
`hf_ema/`), resume (step, data cursor, EMA, RNG) and DDP come from the Trainer unchanged;
`tests/fold/test_training.py` and `test_training_distributed.py` exercise them on CPU.
`flops_per_token` is `None` (the cost depends on crop, loops and sample count), so the
FLOP-derived metrics are not logged.

**Measurement.** `oplm fold measure --config configs/fold/proof-w128.yaml --steps 20 --out m.json`
runs real optimizer steps and writes a JSON report (`environment`, `settings`, per-log `steps`,
`peak_allocated_gib`, `peak_reserved_gib`, mean `step_time_s`, `micro_step_time_s`). `step_time_s`
is per optimizer step at the config's `gradient_accumulation_steps` (it already includes every
accumulation micro-step); `micro_step_time_s` is that divided by the accumulation, the time of
one micro-batch. A run that fails still writes the records so far plus an `error` field.
[`docs/fold/b200-m2b-measure.sbatch`](fold/b200-m2b-measure.sbatch) runs it for
`configs/fold/proof-w128.yaml` and `proof-w256.yaml` on one B200, a per-device batch sweep (`sweep`:
batch 2, 4 and 8 at the same 16 samples per optimizer step), plus the fold CPU suite and the
GPU-marked fold tests in the ESM-C environment. The results and the per-device batch, accumulation and step budget derived from them for
milestone 2c are recorded in [`docs/fold/m2/`](fold/m2/).

**Deviations, all deliberate** (addendum §2). Four loss terms (no bond term, no interface
weighting); single-chain examples, so no atom-symmetry correction; block-level checkpointing only;
the ESM-C head is not transferable to OPLM.
