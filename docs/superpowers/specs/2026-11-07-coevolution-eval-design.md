# Coevolution eval: categorical-Jacobian P@L against MSA Potts couplings

Status: approved 2026-11-07. Board card #79.

## Intent

Structure P@L tells us how many true contacts the model's categorical Jacobian recovers. It does
not tell us *where* that signal comes from. A Potts model fit to an MSA recovers contacts from
coevolution alone. Scoring the same Jacobian against both references, on the same sequences,
separates "the model learned coevolution" from "the model learned structure beyond coevolution".

Two uses, in priority order:

1. **In-training eval (primary).** Logged next to eval CE and structure P@L, on the structure sets
   that already run in training, at the same cadence. Its cost must be close to zero: the Jacobian
   is already computed for structure P@L, and the Potts reference is precomputed and shipped as
   data.
2. **Final-model analysis (secondary).** The same metrics on a trained checkpoint answer how much
   of the model's structure signal goes beyond coevolution, broken down by MSA depth.

It is a complement to structure P@L, never a replacement.

## Goals

- Coevolution metrics computed from the Jacobian the `structure` task already computes, with no
  second Jacobian pass.
- MSAs and Potts couplings for `casp15_16`, `cameo_256` and `pdb40_2048` (which contains
  `pdb40_256`), built by a rebuildable command and published with the structure eval sets.
- A metric set that decomposes structure P@L into coevolution-shared and beyond-coevolution parts,
  reported overall and by MSA-depth (Neff) bin.

## Non-goals

- Full `pdb40` (29,429) and full `cameo` (435) coevolution data. Deferred to a separate card; the
  build must not preclude it (it is the same command over more sequences).
- The full `(L, A, L, A)` tensor correlation between Jacobian and Potts couplings (cf. Zhang et
  al. 2024). Shipping the tensors is too large (≈210 MB per L=512 sequence at fp16).
- Any MSA search or Potts fitting during training or eval.
- Standalone checkpoint eval (that is #57). For now the final-model numbers are the in-loop eval
  at the final step (`is_final`). HSDP runs can't eval in-loop
  (`validate_parallelism_compat`), so they get coevolution numbers once #57 lands.
- A per-sequence metric dump. The Neff bins are the depth breakdown; a finer one can be added
  with #80 if wanted.

## Anti-goals

- **A coevolution number read as "higher is better" on its own.** A model that beats Potts on
  structure finds contacts Potts misses, so Jacobian-vs-Potts agreement can fall while the model
  improves. Every coevolution metric is logged next to structure P@L, and the docs must say this.
- **A Potts reference that is mostly noise counted in the headline agreement number.** Shallow
  MSAs give unreliable couplings; the headline agreement metric excludes the low-Neff bin.
- **Numbers not directly comparable with structure P@L.** Same pairs, same exclusions, same `k`.

## Decisions (made by the user)

- Build on the `structure` eval sets so every sequence has all three signals (Jacobian, Potts,
  structure), and reuse the Jacobian already computed for structure P@L.
- Sets now: `casp15_16`, `cameo_256`, `pdb40_2048`. Full `pdb40` and `cameo` deferred.
- MSAs: ColabFold-style MMseqs2 search against the full ColabFold DB (UniRef30 + ColabFold
  environmental DB). Primary source is the lab's local ColabFold MSA server (address to be
  supplied); the public ColabFold MSA server is a rate-limited fallback. Both speak the same API,
  so the builder takes the server URL as a parameter.
- This build uses the public ColabFold server (decided 2026-11-07), accepting its rate limit. Record
  the DB versions it reports in the README. If #80 later uses the local server with different DB
  versions, #80 rebuilds these subsets too, so all published coevolution data shares one DB
  version.
- Potts fitter: GREMLIN-style pseudolikelihood fit, implemented in PyTorch in this repo.
- The contact decomposition and Neff breakdown are built now, logged during training on the
  three sets, not deferred.

## Metrics

Per sequence, over the same scorable pairs as structure P@L (sequence separation ≥
`min_seq_sep`, both residues with coordinates; `build_structure_pair_score_data` in
`src/oplm/eval/metrics/categorical_jacobian.py`), with `k = min(max(1, L // d), n_pairs)` exactly
as `compute_mean_pair_score_precision_at_l` defines it:

- **M** = the top-`k` pairs by Jacobian score.
- **P** = the top-`k` pairs by Potts score.
- **T** = true contacts (the existing labels).

Each metric is a per-sequence count divided by `k`, averaged over sequences, for `d ∈ {1, 2, 5}`
(suffixes `_at_L`, `_at_L_2`, `_at_L_5`, as for structure P@L):

| metric | definition | reads as |
|---|---|---|
| `coevo_precision_at_L` | \|M ∩ P\| / k | agreement of the model's top pairs with Potts |
| `p_at_L_shared` | \|M ∩ T ∩ P\| / k | correct contacts coevolution also finds |
| `p_at_L_beyond` | \|M ∩ T \ P\| / k | correct contacts Potts misses |
| `p_at_L_potts_only` | \|P ∩ T \ M\| / k | coevolution contacts the model hasn't picked up |
| `potts_precision_at_L` | \|P ∩ T\| / k | the Potts reference's own structure P@L (constant over training) |
| `coevo_spearman` | Spearman ρ between Jacobian and Potts scores over all scorable pairs | whole-matrix agreement (no `d`) |

Identity, which tests must check: `p_at_L_shared + p_at_L_beyond == precision_at_L` per sequence
and therefore in the mean.

`potts_precision_at_L` is computed at eval time from the shipped Potts map and the eval's own
labels and valid mask (cheap, and guaranteed consistent with the eval), not read from the
manifest.

**Neff bins.** Every metric is also reported per Neff bin (`low`, `mid`, `high`), with bin edges
fixed at build time and stored in the published manifest, so they never move between runs.
Headlines:

- `coevo_precision_at_*` and `coevo_spearman` headline (unsuffixed) over `mid` + `high` only.
- The decomposition metrics and `potts_precision_at_*` headline over all sequences.
- Per-bin values for every metric, including `coevo_*` in `low`.

Mean over the sequences in each group; a bin with no sequences in a set is omitted, not logged
as 0.

## Data

**Keying.** MSAs and couplings depend only on the sequence, so they are keyed by the sha256 of
the sequence. That dedupes CASP domains that share a target sequence, `pdb40_256` ⊂
`pdb40_2048`, and CAMEO/PDB40 overlaps, and lets the deferred full sets reuse everything already
built. For CASP domains the MSA and Potts fit cover the full target sequence (what the model
sees); only the domain's residues are scored, as in the structure eval.

**Contents**, under a `coevolution/` folder alongside the sets in the published dataset:

- `msas/<sha256>.a3m.gz`: the raw MSA as returned by the server.
- `couplings/<sha256>.npy` (or one file per set; builder's call): the `(L, L)` Potts contact map,
  float16, over the full sequence.
- `manifest.parquet`: one row per (set, structure name) with the sequence sha256, L, MSA depth
  (raw and after filtering), Neff, Neff/√L, Neff bin, and sha256 of each data file.
- `README.md` and `summary.json`: server URL and date, ColabFold DB versions, filtering and fit
  settings, bin edges, per-set counts per bin, and the code commit.

**Potts map.** GREMLIN couplings, gauge-fixed, Frobenius norm over the 20 amino-acid states
(gap excluded), symmetrized, APC-corrected with the existing `apply_apc`
(`src/oplm/eval/metrics/contact.py`).

## Build

A new command in the same family as `oplm fold eval-sets` (name and module placement are the
builder's call) that, given the set folders and an MSA server URL:

1. Collects distinct sequences across the requested sets.
2. Fetches unpaired MSAs from the ColabFold MSA server API (uniref + environmental mode), with
   resume (skip sequences already fetched), backoff on rate limits, and the server URL as a
   parameter.
3. Filters each MSA and fits GREMLIN on GPU.
4. Writes the Potts maps, manifest, bin edges, README and summary.
5. Validates: every structure in each set has a row and data files that match their checksums,
   and each Potts map has shape `(L, L)` for its sequence.

The command must be resumable and deterministic given the same MSAs on the same hardware and
software (fixed seeds), so the published couplings can be rebuilt from the published MSAs.
Bit-identical rebuilds across GPU types are not required: the manifest checksums guard file
integrity, and a rebuild is judged by agreement within a stated tolerance (for example, identical
top-L Potts pairs and Potts P@L within 0.005), checked on a sample and recorded in the README.

## Eval integration

- Extend the `structure` task (`src/oplm/eval/tasks/structure.py`); no new task type. A new
  optional task key (for example `coevolution: <path to coevolution folder>`) turns the metrics
  on; without it the task behaves exactly as today.
- When it is set, the coevolution metrics are added to the task's default metrics, so a config
  only needs the path. An explicit `metrics:` list still filters as today.
- A structure without a coevolution row is scored for structure P@L as today and left out of
  the coevolution metrics, with one warning per eval naming the count.
- Keys follow the evaluator's `eval/<entry>/<metric>` pattern, e.g.
  `eval/cameo/p_at_L_beyond`, with a bin suffix such as `eval/cameo/p_at_L_beyond_neff_low`.
- Per-sequence data gathered across ranks gains the Potts scores and the Neff bin; everything
  else is computed on the gathered list, as for structure P@L.
- Add `coevolution:` to all four structure entries in `configs/scaling.yaml` (`pdb40`, which
  points at `pdb40_256`; `pdb40_2048`; `cameo`; `casp`), so the PDB coevolution signal is logged
  at the 5k-step structure cadence. `pdb40_256` needs no extra data (it is a subset of
  `pdb40_2048`).

## Docs

- `docs/evaluation.md`: the metric definitions, the decomposition identity, the
  "not higher-is-better" warning, and how to enable it.
- `docs/config.md`: the new `structure` task key.
- `docs/data.md`: a *Coevolution data* section with sources, versions, filtering, fit settings,
  bin edges and counts, in the style of *Structure evaluation sets*.

## Testing

- The decomposition identity on synthetic score matrices.
- Each metric on a hand-built case with known M, P, T.
- GREMLIN recovers planted couplings on an MSA sampled from a small known Potts model.
- Structure task with and without the coevolution key on a tiny model: identical structure
  metrics either way; coevolution metrics present only with it.
- A missing coevolution row: structure metrics unaffected, sequence excluded from coevolution.

## Publishing

Publishing to `brineylab/oplm-structure-evals` is outward-facing: report counts and the README
and wait for the user's go-ahead. Before publishing the MSAs, check the license terms of the
ColabFold DB components (UniRef30, the environmental DB's sources) and record them in the README.
The existing `2026-10-12` tag stays untouched.

## Calls made by Claude (not the user)

Checked against the conversation but not explicitly decided by the user:

- Extend the `structure` task behind an optional key rather than add a new task type.
- Key coevolution data by sequence sha256; ship it as a `coevolution/` folder in
  `brineylab/oplm-structure-evals` under a new tag, leaving `2026-10-12` untouched.
- Compute `potts_precision_at_*` at eval time rather than storing it.
- Headline `coevo_*` metrics over the `mid` + `high` Neff bins; everything else over all
  sequences.
- Don't ship the `(L, A, L, A)` tensors.
- Final-model numbers come from the final in-loop eval until #57; no per-sequence dump.
- Rebuilds match within a tolerance, not bit for bit.

## Left to the builder

These are the builder's calls; record what was chosen in the README and `docs/data.md`:

- MSA filtering before the fit (insertion removal, gap-fraction and identity filters, any depth
  cap), within standard GREMLIN practice.
- GREMLIN hyperparameters (regularization, optimizer, iterations), matching the reference
  GREMLIN implementation's defaults unless there is a measured reason not to.
- The Neff definition (sequence reweighting at 80% identity is the convention) and the exact bin
  edges, chosen once from the pooled three sets so each bin is populated in each set.
- Module placement, command name, coupling file layout, and metric-key suffix spelling.
