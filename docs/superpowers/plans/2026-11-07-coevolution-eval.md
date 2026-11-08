# Coevolution Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the `structure` task's categorical Jacobian against precomputed MSA Potts couplings, logging coevolution agreement and a shared/beyond decomposition of structure P@L during training, and build the MSAs and couplings for the in-training structure sets.

**Architecture:** Two halves joined by a data folder. The eval half (pure metrics + a lookup of shipped Potts maps by sequence sha256) plugs into `StructureEvalTask` behind an optional `coevolution:` key, so the Jacobian is computed once. The build half (`oplm fold eval-coevolution`) fetches ColabFold MSAs, fits GREMLIN in PyTorch, and writes `coevolution/` with maps, manifest, bin edges and a README, in the style of `oplm fold eval-sets`.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pyarrow, typer, stdlib `urllib` / `tarfile` / `gzip` for the ColabFold API.

**Spec:** `docs/superpowers/specs/2026-11-07-coevolution-eval-design.md`

## Global Constraints

- No new dependencies: no `requests`, no `scipy` (neither is installed). HTTP via `urllib.request`; Spearman via torch ranks.
- Run Python as `.venv/bin/python`; fast suite `.venv/bin/python -m pytest -m "not slow" -q`; `ruff check src/`, `ruff format src/`, `ty check src/` clean before each commit.
- Google-style docstrings on public APIs; line length 100; slow tests marked `@pytest.mark.slow`.
- Same pairs, same exclusions, same `k = min(max(1, L // d), n_pairs)` as `compute_mean_pair_score_precision_at_l`.
- Without the `coevolution:` key the `structure` task behaves exactly as today.
- Coevolution data is keyed by `hashlib.sha256(sequence.encode()).hexdigest()`.
- Publishing to Hugging Face is outward-facing: stop and wait for the user's go-ahead. The `2026-10-12` tag stays untouched.
- MSAs come from the public ColabFold server (`https://api.colabfold.com`) for this build; the host is a parameter.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Decisions this plan makes (spec leaves them to the builder)

- Default logged metrics are the `_at_L` variants plus `coevo_spearman`; `_at_L_2` / `_at_L_5` variants are computed only when listed in `metrics:`. Keeps W&B at 6 base metrics × 4 groups per set.
- A `metrics:` list names **base** metrics; each requested coevolution base metric emits its headline key and its `_neff_{low,mid,high}` keys.
- ColabFold mode `env` (the ColabFold default: UniRef30 + environmental DB, server-side filtered). The combined MSA is `uniref.a3m` followed by the env a3m minus its repeated query.
- MSA preprocessing (GREMLIN convention): drop a3m insertions (lowercase) ; map non-canonical letters to gap; keep the first `max_seqs = 20_000` rows in MSA order (query first); drop columns with > 25% gaps. Dropped columns are NaN in the shipped map.
- Neff = Σ 1/|{m : identity(n, m) ≥ 0.8}| over the preprocessed MSA, identity = matching states (gap included) / kept columns. Bins are tertiles of Neff/√L over the build's distinct sequences, overridable with fixed edges (for #80).
- Module placement: build code in `src/oplm/fold/evalsets/{msa,gremlin,coevolution}.py`; eval code in `src/oplm/eval/metrics/coevolution.py`. Couplings are one `.npy` per sequence.

## Review Focus

1. **Potts map with NaN positions** (gap-dropped columns): those pairs rank last in P and are left out of the Spearman; never a crash, never NaN in a logged metric. Test in Task 1.
2. **A Neff bin, or `mid`+`high`, with no sequences in a set** (likely for `casp15_16`): the key is omitted, never logged as 0.0 or NaN. Test in Task 1.
3. **Coevolution data that doesn't match the set** (wrong tag; map shape ≠ `(L, L)` for the sequence): raise `ValueError` naming the structure, not silently misalign. Test in Task 2.
4. **The build interrupted, or the server rate-limits / returns `ERROR`**: rerun resumes without refetching or refitting; `ERROR` raises with the sequence's sha256 instead of writing an empty MSA. Tests in Tasks 3 and 5.
5. **A query-only MSA** (no homologs found): the fit still returns a finite map over kept columns, Neff = 1, and the sequence lands in `low`. Test in Task 4.

---

### Task 1: Coevolution metrics

**Files:**
- Modify: `src/oplm/eval/metrics/categorical_jacobian.py` (`StructurePairScoreData`)
- Create: `src/oplm/eval/metrics/coevolution.py`
- Test: `tests/eval/test_coevolution_metrics.py`

**Interfaces:**
- Produces:
  - `StructurePairScoreData` gains `potts_scores: Tensor | None = None` (same shape as `scores`; NaN where unfit) and `neff_bin: str | None = None` (`"low" | "mid" | "high"`).
  - `NEFF_BINS: tuple[str, ...] = ("low", "mid", "high")`
  - `COEVOLUTION_METRIC_DIVISORS: dict[str, int]`: every base name → divisor (`coevo_spearman` → 0), i.e. for `(s, d)` in `(("at_L", 1), ("at_L_2", 2), ("at_L_5", 5))`: `coevo_precision_{s}`, `p_{s}_shared`, `p_{s}_beyond`, `p_{s}_potts_only`, `potts_precision_{s}`; plus `coevo_spearman`.
  - `DEFAULT_COEVOLUTION_METRICS: list[str]`: the five `at_L` names + `coevo_spearman`.
  - `compute_coevolution_metrics(structures: Sequence[StructurePairScoreData], metrics: Collection[str]) -> dict[str, float]`: uses only structures with `potts_scores is not None`; for each requested base name emits the headline key (over `mid`+`high` for names starting `coevo_`, else over all) and `f"{name}_neff_{bin}"` per populated bin. Empty groups are omitted.

- [ ] **Step 1: Write the failing tests**

```python
def _data(scores, potts, labels, seq_len, neff_bin="high"): ...  # flat tensors, pair_indices unused

def test_hand_built_counts():
    # 6 pairs, k = 2. M = {0,1}, P = {1,2}, T = {0,1,2}
    d = _data([.9,.8,.1,.1,.1,.1], [.1,.9,.8,.1,.1,.1], [1,1,1,0,0,0], seq_len=2)
    m = compute_coevolution_metrics([d], COEVOLUTION_METRIC_DIVISORS)
    assert m["coevo_precision_at_L"] == 0.5      # |M∩P| = 1
    assert m["p_at_L_shared"] == 0.5             # pair 1
    assert m["p_at_L_beyond"] == 0.5             # pair 0
    assert m["p_at_L_potts_only"] == 0.5         # pair 2
    assert m["potts_precision_at_L"] == 1.0
    assert m["p_at_L_shared_neff_high"] == 0.5

def test_shared_plus_beyond_equals_structure_precision():
    # 20 random structures, mixed bins, every divisor
    assert shared + beyond == pytest.approx(compute_mean_pair_score_precision_at_l(ds, l_divisor=d))

def test_coevo_headline_excludes_low_bin():
    # one low structure with agreement 0, one high with agreement 1
    assert m["coevo_precision_at_L"] == 1.0 and m["coevo_precision_at_L_neff_low"] == 0.0
    assert m["p_at_L_beyond"] == mean over both

def test_empty_groups_are_omitted():
    # only low-bin structures
    assert "coevo_precision_at_L" not in m and "coevo_precision_at_L_neff_mid" not in m
    assert "p_at_L_beyond" in m

def test_nan_potts_pairs_rank_last_and_skip_spearman():
    # potts NaN on the two highest-Jacobian pairs; P must avoid them; spearman finite
    assert all(math.isfinite(v) for v in m.values())

def test_spearman_perfect_and_inverse():
    assert m_same["coevo_spearman"] == pytest.approx(1.0)
    assert m_rev["coevo_spearman"] == pytest.approx(-1.0)

def test_structures_without_potts_are_ignored_and_only_requested_metrics_emitted():
    m = compute_coevolution_metrics([with_potts, without_potts], ["p_at_L_beyond"])
    assert set(m) == {"p_at_L_beyond", "p_at_L_beyond_neff_high"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_coevolution_metrics.py -q`
Expected: FAIL (`ModuleNotFoundError: oplm.eval.metrics.coevolution`)

- [ ] **Step 3: Add the two fields to `StructurePairScoreData` and implement `src/oplm/eval/metrics/coevolution.py`**

Per structure and divisor: `k` as in `compute_mean_pair_score_precision_at_l`; `M`/`P` boolean masks from `torch.topk` (Potts NaN → `-inf` first); counts / `k`. Spearman over pairs where Potts is finite: Pearson of `argsort(argsort(x))` ranks (`# ponytail: ordinal ranks, ties not averaged; fp16 ties are rare`). Average per group.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval/test_coevolution_metrics.py tests/eval/test_categorical_jacobian.py -q`
Expected: PASS

- [ ] **Step 5: Commit** — `feat(eval): coevolution metrics over Jacobian and Potts pair scores (#79)`

---

### Task 2: Structure-task integration

**Files:**
- Modify: `src/oplm/eval/metrics/coevolution.py` (add the reference reader)
- Modify: `src/oplm/eval/tasks/structure.py`
- Modify: `configs/scaling.yaml` (the four `type: structure` entries)
- Modify: `docs/config.md` (*structure* key table), `docs/evaluation.md` (new *Coevolution metrics* section)
- Test: `tests/eval/test_structure_task.py`, `tests/eval/test_coevolution_metrics.py`

**Interfaces:**
- Consumes: Task 1's fields, `DEFAULT_COEVOLUTION_METRICS`, `COEVOLUTION_METRIC_DIVISORS`, `compute_coevolution_metrics`.
- Produces (the on-disk contract Task 5 writes):
  - `<coevolution>/manifest.parquet` with at least columns `sha256: str`, `length: int`, `neff_bin: str`; `<coevolution>/couplings/<sha256>.npy`, float16 `(L, L)`.
  - `class CoevolutionReference` with `@classmethod load(cls, path: str | Path) -> CoevolutionReference` and `lookup(self, sequence: str) -> tuple[Tensor, str] | None` returning the float32 `(L, L)` map and its bin; raises `ValueError` naming the sha256 if the map's shape isn't `(len(sequence), len(sequence))`.
  - `StructureTaskConfig.coevolution: str | None = None` (task key `coevolution`).

- [ ] **Step 1: Write the failing tests**

```python
def test_structure_config_parses_coevolution_path(): ...  # from_extra({"coevolution": "/x"}).coevolution == "/x"

def test_coevolution_key_extends_default_metrics():
    # entry without metrics: task.metrics == ["precision_at_L", *DEFAULT_COEVOLUTION_METRICS]
    # entry with metrics=["precision_at_L"]: unchanged

def test_reference_lookup_and_missing(tmp_path):   # write a manifest + one .npy
    assert ref.lookup(seq)[1] == "mid" and ref.lookup("ACD") is None

def test_reference_shape_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match=sha):
        ref.lookup(seq)

@pytest.mark.slow
def test_structure_metrics_identical_with_and_without_coevolution(structure_fixtures_dir, tmp_path, make_model):
    # write coevolution data for 1CRN only (random symmetric map, bin "high")
    assert with_coevo["precision_at_L"] == without["precision_at_L"]
    assert "p_at_L_beyond" in with_coevo and "p_at_L_beyond" not in without
    assert with_coevo["p_at_L_shared"] + with_coevo["p_at_L_beyond"] == pytest.approx(
        precision over the 1CRN structure alone)   # 1L2Y, 1VII missing -> excluded
    assert "2 of 3 structures have no coevolution data" in caplog.text
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/eval/test_structure_task.py tests/eval/test_coevolution_metrics.py -q -m "not slow"`
Expected: FAIL on the new tests

- [ ] **Step 3: Implement**

- `CoevolutionReference.load` reads only `sha256`, `length`, `neff_bin` from the manifest (dedupe on sha256); `lookup` loads the `.npy` lazily.
- `StructureEvalTask.__init__`: when `tcfg.coevolution` is set and `entry.metrics` is falsy, `self.metrics = [*self.default_metrics, *DEFAULT_COEVOLUTION_METRICS]`. Reference loaded lazily in `evaluate`, like `_structures`.
- `_process_single_structure`: after `build_structure_pair_score_data`, if a reference is loaded and `lookup` hits, set `potts_scores = potts_map[i_idx, j_idx]` and `neff_bin`.
- `evaluate`: after the gather, add `compute_coevolution_metrics(all_data, requested coevolution names)`; on the main process warn once `"%s: %d of %d structures have no coevolution data; left out of coevolution metrics"` when any gathered item lacks `potts_scores`. Structure metrics are computed as today.
- `configs/scaling.yaml`: add `coevolution: /path/to/oplm-structure-evals/coevolution` to `pdb40`, `pdb40_2048`, `cameo`, `casp`.
- `docs/config.md`: one row for `coevolution` (`str | null`, `null`, path to the coevolution folder; enables the metrics). `docs/evaluation.md`: the metric table and identity from the spec's *Metrics* section, the Neff headline rule, how to enable, and the warning that agreement is **not** higher-is-better and must be read next to `precision_at_L`.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/eval tests/test_docs.py tests/test_scaling_config.py -q` (includes slow eval tests)
Expected: PASS

- [ ] **Step 5: Commit** — `feat(eval): coevolution metrics in the structure task (#79)`

---

### Task 3: ColabFold MSA client and MSA preprocessing

**Files:**
- Create: `src/oplm/fold/evalsets/msa.py`
- Test: `tests/fold/evalsets/test_msa.py`

**Interfaces:**
- Produces:
  - `PUBLIC_COLABFOLD_HOST = "https://api.colabfold.com"`
  - `Fetch = Callable[[str, bytes | None], bytes]` (url, POST body or None for GET → response body).
  - `fetch_msa(sequence: str, host: str, *, fetch: Fetch | None = None, mode: str = "env", poll_seconds: float = 5.0, sleep: Callable[[float], None] = time.sleep) -> str`: the combined a3m text. Default `fetch` uses `urllib.request` with `User-Agent: oplm/<version>`.
  - `parse_a3m(text: str) -> list[str]`: aligned rows (insertions removed), query first.
  - `encode_msa(rows: list[str], max_seqs: int = 20_000) -> Tensor`: `(N, L)` int8 in `0..20` (`CANONICAL_AMINO_ACIDS` order from `oplm.eval.metrics.categorical_jacobian`, gap = 20; other letters → 20).
  - `filter_gap_columns(msa: Tensor, max_gap_fraction: float = 0.25) -> tuple[Tensor, Tensor]`: kept MSA and the kept column indices.
  - `neff(msa: Tensor, identity: float = 0.8, chunk: int = 4096) -> tuple[float, Tensor]`: Neff and per-sequence weights.

API, as ColabFold's client uses it: POST `{host}/ticket/msa` with form `q=>{name}\n{seq}`, `mode=env` → JSON `{"id", "status"}`; poll GET `{host}/ticket/{id}` until `COMPLETE`; GET `{host}/result/download/{id}` → tar.gz holding `uniref.a3m` and `bfd.mgnify30.metaeuk30.smag30.a3m`. Status `RATELIMIT`/`UNKNOWN`/`PENDING`/`RUNNING`/`MAINTENANCE` → sleep and retry (back off ×2 to a 10-minute cap on `RATELIMIT`); `ERROR` → `RuntimeError` naming the sequence.

- [ ] **Step 1: Write the failing tests**

```python
def test_fetch_msa_polls_then_combines(fake_server):
    # fake fetch: submit -> RATELIMIT then PENDING id; status RUNNING then COMPLETE; download tar
    a3m = fetch_msa(SEQ, "http://x", fetch=fake, sleep=lambda s: None)
    assert a3m.count(">") == 1 + n_uniref + n_env      # env query not repeated
    assert sleeps grew after RATELIMIT

def test_fetch_msa_error_raises():
    with pytest.raises(RuntimeError, match="ERROR"): ...

def test_parse_a3m_removes_insertions():
    assert parse_a3m(">q\nAC-D\n>h\nAcC-D\n") == ["AC-D", "AC-D"]

def test_encode_maps_noncanonical_to_gap_and_caps_depth():
    assert encode_msa(["LAX-"]).tolist() == [[0, 1, 20, 20]]
    assert encode_msa(["LA"] * 5, max_seqs=3).shape == (3, 2)

def test_filter_gap_columns():
    kept, cols = filter_gap_columns(msa_with_one_gappy_column)
    assert cols.tolist() == [0, 2]

def test_neff_identical_and_distinct():
    assert neff(four_identical_rows)[0] == pytest.approx(1.0)
    assert neff(four_unrelated_rows)[0] == pytest.approx(4.0)
    assert neff(query_only)[0] == 1.0
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/fold/evalsets/test_msa.py -q` → FAIL (module missing)
- [ ] **Step 3: Implement `src/oplm/fold/evalsets/msa.py`** (identity via one-hot matmul in chunks of `chunk` rows)
- [ ] **Step 4: Run to verify they pass** — same command → PASS
- [ ] **Step 5: Commit** — `feat(fold): ColabFold MSA client and MSA preprocessing (#79)`

---

### Task 4: GREMLIN fit and Potts contact map

**Files:**
- Create: `src/oplm/fold/evalsets/gremlin.py`
- Test: `tests/fold/evalsets/test_gremlin.py`

**Interfaces:**
- Consumes: Task 3's encoded MSA (`(N, L)`, 21 states) and weights.
- Produces:
  - `@dataclass(frozen=True) class GremlinConfig: lam_v: float = 0.01; lam_w_scale: float = 0.01; lr: float; iterations: int` — values copied from the reference GREMLIN implementation (sokrypton/GREMLIN_TF v2); record the source and values in the module docstring.
  - `fit_gremlin(msa: Tensor, weights: Tensor, cfg: GremlinConfig = GremlinConfig(), device: str | torch.device = "cpu") -> Tensor`: couplings `W`, `(L, 21, L, 21)` float32 on CPU.
  - `potts_contact_map(w: Tensor) -> Tensor`: `(L, L)`: Frobenius norm over the 20 amino-acid states (gap excluded), symmetric, zero diagonal, then `apply_apc` from `oplm.eval.metrics.contact`.
  - `full_length_map(contacts: Tensor, kept_columns: Tensor, length: int) -> Tensor`: `(length, length)` with NaN on rows/columns not in `kept_columns`.

Objective (full batch, zero-initialized `W`, `V` initialized from the weighted log-frequencies of the MSA, so no seed is needed):

```
W_sym = (W + W.permute(2, 3, 0, 1)) / 2, with W_sym[i, :, i, :] = 0
logits[n, i, :] = V[i, :] + Σ_j Σ_b onehot[n, j, b] · W_sym[j, b, i, :]
loss = Σ_n weights[n] · Σ_i CE(logits[n, i], msa[n, i])
       + lam_v · ‖V‖² + lam_w_scale · (L − 1) · 20 · ‖W_sym‖²
```

Adam on `(V, W)`; if the reference uses a different regularizer scaling or optimizer, follow the reference and note it.

- [ ] **Step 1: Write the failing tests**

```python
def test_recovers_planted_couplings():
    # L = 12, couplings planted on (1, 8) and (3, 10); sample 2,000 sequences by Gibbs from that Potts model
    contacts = potts_contact_map(fit_gremlin(msa, neff(msa)[1]))
    top2 = set(top pairs with |i - j| >= 3)
    assert top2 == {(1, 8), (3, 10)}

def test_query_only_msa_gives_finite_map():
    assert torch.isfinite(potts_contact_map(fit_gremlin(query_only, torch.ones(1)))).all()

def test_contact_map_symmetric_zero_diagonal():
    ...

def test_full_length_map_nans_dropped_columns():
    m = full_length_map(torch.ones(2, 2), torch.tensor([0, 2]), 3)
    assert torch.isnan(m[1]).all() and m[0, 2] == 1
```

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/fold/evalsets/test_gremlin.py -q` → FAIL
- [ ] **Step 3: Implement `src/oplm/fold/evalsets/gremlin.py`**
- [ ] **Step 4: Run to verify they pass** — same command → PASS (the planted test must run in < 30 s on CPU; otherwise mark it slow)
- [ ] **Step 5: Commit** — `feat(fold): GREMLIN pseudolikelihood fit in PyTorch (#79)`

---

### Task 5: `oplm fold eval-coevolution` build command

**Files:**
- Create: `src/oplm/fold/evalsets/coevolution.py`
- Modify: `src/oplm/fold/cli.py` (new command; module docstring lists it)
- Test: `tests/fold/evalsets/test_coevolution_build.py`, `tests/fold/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3–4; `load_structures` from `oplm.data`; `sha256sum` from `oplm.fold.evalsets.casp`; Task 2's on-disk contract (it must be readable by `CoevolutionReference`).
- Produces:
  - `@dataclass class CoevolutionBuildConfig: sets_root: Path; out: Path; sets: tuple[str, ...] = ("casp15_16", "cameo_256", "pdb40_256", "pdb40_2048"); host: str = PUBLIC_COLABFOLD_HOST; device: str = "cpu"; max_seqs: int = 20_000; max_gap_fraction: float = 0.25; identity: float = 0.8; bin_edges: tuple[float, float] | None = None; fit: bool = True; fetch: Fetch | None = None`
  - `build_coevolution(cfg: CoevolutionBuildConfig) -> dict[str, Any]`: the summary.
  - CLI `oplm fold eval-coevolution --sets-root P --out P [--host URL] [--sets a,b] [--device cuda] [--max-seqs N] [--bin-edges e1,e2] [--no-fit]`.

Layout under `out` (`out` is the `coevolution/` folder):

```
msas/<sha256>.a3m.gz        combined a3m (Task 3)
couplings/<sha256>.npy      float16 (L, L), NaN on dropped columns
stats/<sha256>.json         depth, depth_filtered, kept_columns, neff (resume state)
manifest.parquet            set, name, sha256, length, msa_depth, msa_depth_filtered, neff,
                            neff_per_sqrt_l, neff_bin, msa_sha256, couplings_sha256
summary.json, README.md
```

Flow: distinct sequences across `sets` (sorted by sha256) → fetch each missing `msas/` file → (unless `--no-fit`) fit each missing `couplings/` + `stats/` → bin edges = `cfg.bin_edges` or `numpy.quantile(neff_per_sqrt_l, [1/3, 2/3])` over distinct sequences; `low` if `< e1`, `mid` if `< e2`, else `high` → manifest, summary (host, build date, mode, oplm version, config, bin edges, per-set counts per bin, ColabFold DB versions as documented by the server at build time) → README → validate (every structure in every set has a row; every file matches its manifest checksum; every map is `(L, L)`; warn naming the set when a bin is empty in a set). With `--no-fit` stop after MSAs and print the count.

- [ ] **Step 1: Write the failing tests**

```python
def test_build_end_to_end(tmp_path, fake_fetch):   # sets_root with two sets sharing 1CRN, plus 1L2Y
    s = build_coevolution(cfg)
    assert fake_fetch.calls == 2                         # deduped by sequence
    ref = CoevolutionReference.load(cfg.out)
    potts, bin_ = ref.lookup(CRAMBIN)
    assert potts.shape == (46, 46) and bin_ in NEFF_BINS
    assert s["neff_bin_edges"] and manifest rows == 3

def test_build_resumes_without_refetch_or_refit(tmp_path, fake_fetch): # second run: 0 fetches, couplings mtimes unchanged

def test_no_fit_stops_after_msas(tmp_path, fake_fetch): # msas/ populated, no couplings/, no manifest

def test_fixed_bin_edges_are_used(tmp_path, fake_fetch): # bin_edges=(0.0, 1e9) -> every row "mid"

def test_eval_coevolution_cli_help():
    assert runner.invoke(app, ["fold", "eval-coevolution", "--help"]).exit_code == 0
```

`fake_fetch` serves the Task 3 protocol with a small a3m of mutated copies of the query.

- [ ] **Step 2: Run to verify they fail** — `.venv/bin/python -m pytest tests/fold/evalsets/test_coevolution_build.py tests/fold/test_cli.py -q` → FAIL
- [ ] **Step 3: Implement `src/oplm/fold/evalsets/coevolution.py` and the CLI command** (write each file to a temp name and rename, so an interrupted run never leaves a partial file that resume would skip)
- [ ] **Step 4: Run to verify they pass** — same command, then the fast suite → PASS
- [ ] **Step 5: Commit** — `feat(fold): oplm fold eval-coevolution builds MSAs and Potts maps (#79)`

---

### Task 6: Build the data, document it, open the PR

Needs the user for the cluster and the publish. Nothing here is code.

- [ ] **Step 1: Fetch MSAs** on a host with network access and the `2026-10-12` structure sets (download them from `brineylab/oplm-structure-evals` tag `2026-10-12` if absent):

```bash
oplm fold eval-coevolution --sets-root $OPLM_DATA/hf/structure-evals/2026-10-12 \
    --out $OPLM_DATA/hf/structure-evals/coevolution --no-fit
```

Run it in the background; the public server is rate-limited and this may take many hours. Expected: one `msas/` file per distinct sequence (≈2,300 after dedup).

- [ ] **Step 2: Fit** on a GPU node (same command, `--device cuda`, without `--no-fit`). Expected: validation passes; the summary table prints per-set counts per bin.
- [ ] **Step 3: Sanity check** before writing docs: report `potts_precision_at_L` by bin per set to the user (expect it to rise from `low` to `high`; deep-MSA PDB chains typically well above 0.5). Stop if it doesn't.
- [ ] **Step 3b: Rebuild check** (spec: tolerance, not bit-identical): refit 10 sequences spread across bins from their published MSAs on a different device (CPU if the build used GPU) into a scratch `out`. Expected: identical top-L Potts pairs and per-sequence Potts P@L within 0.005. Record the result in the README.
- [ ] **Step 4: `docs/data.md`**: a *Coevolution data* section after *Structure evaluation sets*: server and date, mode, DB versions, preprocessing, GREMLIN settings, Neff definition, bin edges, counts per set per bin, build commit, and the ColabFold DB license terms. Run `.venv/bin/python -m pytest tests/test_docs.py -q` → PASS. Commit `docs(data): coevolution data for the structure eval sets (#79)`.
- [ ] **Step 5: Open the PR** with `Closes #79`, then report counts and the README to the user and **wait for the go-ahead** before uploading `coevolution/` to `brineylab/oplm-structure-evals` under a new tag.
