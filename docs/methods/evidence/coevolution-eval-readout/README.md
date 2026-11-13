# The coevolution metrics split contact precision into contacts MSA couplings also find and contacts beyond them

**Status:** provisional

## Question

`precision_at_L` says how many predicted contacts are right, not where they come from. Do the
coevolution metrics (comparison with GREMLIN Potts couplings fitted on each sequence's MSA) give
a readout that plain P@L does not, and do they separate models that P@L already separates?

## Arms

Nine decayed 170M checkpoints from the 1/35 proxy, scored standalone on `casp15_16`,
`cameo_256`, `pdb40_256` and `pdb40_2048` with `coevolution:` set:

- **oplm**, **stack2** (OPLM with the block stack looped twice) and **esmc** (the ESM-C recipe);
- each decayed with uniref70 at 25% of tokens from the stable run's step 56k, 140k and 252k.

The training runs are the decay arms of [oplm-vs-esmc-recipe](../oplm-vs-esmc-recipe/README.md)
and the proxy arms of [looping](../looping/README.md). The Potts reference is the published
build (HF `brineylab/oplm-structure-evals`, tag `2026-11-10`); how it was built and checked is in
[data.md](../../../data.md), *Coevolution data*.

## Result

The split is informative: on shallow MSAs the model finds about three times as many contacts
as the Potts reference, on deep MSAs it roughly matches it, and about half of every model's
correct contacts are ones the Potts top-L misses.

`pdb40_2048`, decay from step 252k. `shared + beyond = P@L`.

| arm | P@L | agreement with Potts | shared | beyond | Potts only | Potts P@L |
|---|---|---|---|---|---|---|
| oplm | 0.390 | 0.365 | 0.187 | 0.203 | 0.134 | 0.321 |
| stack2 | 0.394 | 0.367 | 0.189 | 0.205 | 0.132 | 0.321 |
| esmc | 0.303 | 0.281 | 0.142 | 0.160 | 0.179 | 0.321 |

Model P@L against Potts P@L by MSA depth (Neff bin), same checkpoints and set:

| | low | mid | high |
|---|---|---|---|
| oplm | 0.266 | 0.411 | 0.488 |
| esmc | 0.209 | 0.310 | 0.385 |
| Potts | 0.095 | 0.339 | 0.518 |

By branch point, `pdb40_2048` (56k / 140k / 252k):

| arm | P@L | agreement with Potts | beyond |
|---|---|---|---|
| oplm | 0.346 / 0.373 / 0.390 | 0.318 / 0.348 / 0.365 | 0.184 / 0.195 / 0.203 |
| stack2 | 0.351 / 0.380 / 0.394 | 0.324 / 0.354 / 0.367 | 0.185 / 0.198 / 0.205 |
| esmc | 0.269 / 0.294 / 0.303 | 0.242 / 0.269 / 0.281 | 0.147 / 0.157 / 0.160 |

- OPLM's lead over ESM-C shows in both halves (shared +0.045, beyond +0.043 at 252k) and grows
  with training length in both.
- The other three sets show the same pattern; all 36 rows and the per-bin metrics are in
  `results.csv`.

## Runs

- Eval jobs 35750–35758, one GPU each, oplm at commit 28c97ac; scripts and raw output in
  `~/projects/oplm/mup_redo/coevo_eval/` on the cluster (`results/<arm>.jsonl`).
- In `results.csv`, `wandb_run`, `config` and `step` identify the training run and checkpoint
  that was scored; `job_id` and `commit` are the eval job's.
- The eval: issue #79, PRs #82 and #86. Metric definitions:
  [evaluation.md](../../../evaluation.md), *Coevolution metrics*.

## Caveats

- **Proxy scale, one seed per arm.** 170M models on 1/35 of the data.
- **Agreement with Potts is not higher-is-better**; read it next to P@L.
- The low bin holds the 16 sequences with no MSA hits, whose Potts maps come from the query
  alone.
- Stack2 step numbers are on a different scale from the other arms' (see the looping record).
- `casp15_16` has only 10 sequences in the high bin and 23 in the mid bin.
