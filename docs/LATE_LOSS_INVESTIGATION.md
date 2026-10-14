# Late-stage loss rise in the constant-LR Muon + μP runs

Working notes for the October 2026 investigation. The symptom: on the WSD stable phase
(`lr 0.0035` constant after a 5k warmup), `eval/uniref70/loss` on oplm-170M flattens
around 350–450k steps and rises by ~0.01 through 600k, with a visibly noisier curve past
~300k; oplm-400M shows a much weaker version. Train loss does the same, so it is not
overfitting, and no data-epoch boundary (~700k steps) has been crossed.

Everything below runs on the code merged from
[PR #29](https://github.com/briney/oplm/pull/29): the per-window train metrics, the
`weight_diag_every` weight diagnostics, `oplm weight-rms`, `train.branch_from`,
`train.adamw_lr_mult`, and `KEY=VALUE` overrides on `oplm slurm generate`. See
[TRAIN.md §9 "Branching a run"](TRAIN.md#branching-a-run-wsd-decays-and-lr-experiments)
and [TRAIN.md §10](TRAIN.md#10-logging-and-monitoring) for the mechanics.

---

## 1. What the optimizer actually does (code inspection)

Per-group LR after all μP width/depth scaling, from `build_optimizers` on
`configs/scaling.yaml` (`train.lr=0.0035`, `weight_decay=0.01`,
`muon_adjust_lr_fn=original`, depth exponent 0.5 at reference depth 24):

| Group | Optimizer | 170M LR | 400M LR | WD |
| --- | --- | --- | --- | --- |
| q/k/v/o/gate_proj (square) | Muon | 0.0035 | 0.00303 | 0.01 |
| ffn gate/up_proj (aspect ≈ 2.7, factor ≈ 1.64) | Muon | 0.00572 | 0.00503 | 0.01 |
| ffn down_proj (factor clipped at 1) | Muon | 0.0035 | 0.00303 | 0.01 |
| embeddings | AdamW | 0.0035 | 0.0035 | **0** |
| `lm_head.decoder` (readout) | AdamW | 0.0035 | 0.0035 | 0.01 |
| `lm_head.dense` | AdamW | 0.0035 | 0.002625 | 0.01 |
| final norm + head norm gains | AdamW | 0.0035 | 0.0035 | **0** |
| in-block norm gains (incl. qk-norm), biases, channel residual gates, value-residual λ | AdamW | 0.0035 | 0.00303 | **0** |
| Canon depthwise kernels (1.0M params at 170M, 1.8M at 400M) | AdamW | 0.0035 | 0.00303 | 0.01 |

Takeaways:

- **Every AdamW-side parameter inherits the Muon base LR with no μP reduction.** With
  `original`, a Muon update is ~1.3e-4 per element per step; an Adam update at this LR
  is up to 3.5e-3, and ~8e-4 in the noise-dominated late regime (hypothesis 1, confirmed
  structurally).
- **Embeddings, all norm gains, the residual gates, and the value-residual lambdas have
  no weight decay** (`_uses_no_weight_decay`: 1-D or `embed`-named). At Adam LR 0.0035
  that is an unbounded random walk. qk-norm gains set attention temperature and the
  residual gates scale every residual write.
- Muon's decoupled decay is `p *= 1 - lr * wd` with the *unadjusted* group LR (torch
  2.11 and 2.14 agree): a 28.6k-step timescale at 170M, 33k at 400M.
- The data loader reshuffles shard order and in-shard rows per epoch and has no length
  bucketing; every rank/worker walks the same shard order in lockstep, so there is
  short-timescale composition noise but no slow in-epoch drift. One caveat: a mixed
  source sampled above its row share is refilled mid-epoch in the *identical* order, so a
  source-level epoch boundary may already have passed (depends on the three sources'
  row counts).
- `train/lr` only ever logged the Muon LR; the AdamW LR was invisible until
  `train/lr_adamw`.
- The width-independent 0.0035 on embeddings/head is identical at both sizes and so
  cannot by itself explain why 400M is less affected; the in-block groups at ×0.866 and
  the larger parameter count can.

---

## 2. Diagnostics to read

Always on (every train log, per log window): `train/loss_mean`, `train/grad_norm`,
`train/grad_norm_max`, `train/clip_frac`, `train/mean_seq_len`, `train/lr_adamw`.

Opt-in with `train.weight_diag_every=1000`: `diag/weight_rms/<group>` and
`diag/update_ratio/<group>` for `embed`, `head_dense`, `head_decoder`, `attn`, `mlp`,
`norm_gain`, `residual_gate`, `canon`, `bias`, `other`. Look at the no-decay groups first
(`embed`, `norm_gain`, `residual_gate`).

Backfill weight RMS from any checkpoints on disk:

```bash
oplm weight-rms $MAIN/checkpoint-100000 $MAIN/checkpoint-250000 $MAIN/checkpoint-400000 $MAIN/checkpoint-500000
```

### Re-running 170M with branch points every 50k

The original 170M run kept only a 500k checkpoint. For the re-run, keep a permanent
checkpoint every 50k steps and turn the weight diagnostics on:

```bash
oplm slurm generate --config configs/scaling.yaml --preset 170M --out jobs/170M --name 170M \
  train.output_dir=$MAIN train.keep_every_n_steps=50000 train.weight_diag_every=1000
```

(`save_every` stays at 10k for crash recovery; `keep_every_n_steps` exempts every 50k
checkpoint from `save_total_limit` rotation.) The rendered job runs
`pip install oplm[train]`, so the release that contains PR #29 must be on PyPI first, or
point `slurm.install` in the YAML at a git ref.

---

## 3. Branch experiments (Parts C and D of the handoff)

Set `MAIN` to the 170M run directory and `OUT` to a branches directory, then one line per
experiment. Each renders a 4-node job (same world size as the main run, so the data
cursor carries over), logs weight diagnostics every 1k steps, and auto-resumes itself on
requeue. Submit each with `oplm slurm submit jobs/<name>`.

```bash
MAIN=/mnt/home/briney/projects/oplm/scaling/170M; OUT=/mnt/home/briney/projects/oplm/branches
gen() { n=$1; shift; oplm slurm generate --config configs/scaling.yaml --preset 170M --out jobs/$n --name $n \
         train.output_dir=$OUT/$n train.wandb_run_name=$n train.weight_diag_every=1000 "$@"; }

# C1 decay_branch: 250k -> 300k, linear to 0, same data mix (the "headroom" baseline)
gen 170M-decay-250k     train.branch_from=$MAIN/checkpoint-250000 train.stable_steps=245000 train.max_steps=300000
# C2 adam_lr_div: Muon LR unchanged, AdamW-side LR /3 and /10, constant for 100k steps
gen 170M-adamdiv3-250k  train.branch_from=$MAIN/checkpoint-250000 train.stable_steps=345000 train.max_steps=350000 train.adamw_lr_mult=0.33333
gen 170M-adamdiv10-250k train.branch_from=$MAIN/checkpoint-250000 train.stable_steps=345000 train.max_steps=350000 train.adamw_lr_mult=0.1
# C3 all_lr_div3: every LR /3, constant, WD unchanged (lr*wd drops with it, on purpose)
gen 170M-alldiv3-250k   train.branch_from=$MAIN/checkpoint-250000 train.stable_steps=345000 train.max_steps=350000 train.lr=0.00116667
# C4 wd_0.03: only if the Muon-matrix (attn/mlp) weight RMS is still growing at 600k
gen 170M-wd003-250k     train.branch_from=$MAIN/checkpoint-250000 train.stable_steps=345000 train.max_steps=350000 train.weight_decay=0.03
# D  WSD branch-point comparison: 50k-step decays from 400k and 600k (250k is C1)
gen 170M-decay-400k     train.branch_from=$MAIN/checkpoint-400000 train.stable_steps=395000 train.max_steps=450000
gen 170M-decay-600k     train.branch_from=$MAIN/checkpoint-600000 train.stable_steps=595000 train.max_steps=650000
```

Notes:

- A constant-LR branch is `stable_steps = max_steps - warmup_steps`; the single trailing
  decay step is harmless.
- `branch_from` must be a checkpoint directory (not its `hf/` export). The branch's
  `max_steps` must exceed the checkpoint step.
- The D branches keep the stable-phase data mix so the data cursor carries over. To decay
  on the decay mix instead, override `data.train.<name>.path`/`fraction`; the cursor is
  then dropped (with a warning) and the stream restarts at row 0.
- If the main run was launched from a config other than `configs/scaling.yaml`, use that
  file: model geometry and data layout must match the checkpoint.

**Decision rule (D).** Compare the final decayed `eval/uniref70/loss` of the 250k, 400k,
and 600k branches. If decayed(600k) < decayed(400k), the stable phase is still productive
and the raw uptick is noise-floor drift: continue. If decayed(600k) ≥ decayed(400k), the
drift is doing real damage: lower the LR per the C results (most likely the AdamW side)
before continuing the stable phase. This is also the LR check at the true horizon that the
20k-step sweep could not provide.

---

## 4. Replication results (October 2026)

The 170M re-run with diagnostics (W&B `thebrineylab/oplm-scaling/426y3e7m`, 600k steps,
`weight_diag_every=1000`, permanent checkpoints every 50k under
`/mnt/home/briney/projects/oplm/scaling/170M/`) reproduces the original 170M run
(`dncy6l8u`) almost exactly: their binned train losses agree to about 0.003. The original 400M
run (`hereiygl`) shows the same bumps at the same steps. The data stream is deterministic and
identical across all three runs.

### The late rise is a step at the data-epoch boundary

- The run's mix is `uniref70` at 25 % (1,294 shards, 129.4M rows) and `deepclust70` at 75 %
  (10,708 shards, 5.35B rows). The mixed epoch is their 5.48B rows at 8192 sequences per step:
  669,312 steps per rank.
- In both original runs, train loss steps up about 0.045 within the 2.5k-step window that holds
  that boundary:
  - 170M: 1.92 → 1.97;
  - 400M: 1.85 → 1.90.

  It stays up through 780k. The 170M − 400M gap holds at about 0.07 across it.
- The fixed eval sets rise only about 0.005 right after the boundary (uniref70 2.021 → 2.025,
  omg70 2.205 → 2.210), then drift up slowly. Before the boundary, eval/uniref70 is flat from
  about 445k to 600k (2.022–2.025) rather than rising.
- Most of the jump is therefore the second-epoch stream being harder, which repeated data alone
  would not do.

### The data stream drifts and repeats

- Detrended train loss correlates −0.62 to −0.68 with the logging window's mean sequence length,
  at every window size from 100 to 20,000 steps.
- Every rank and worker reads the same shard at the same moment, so every batch's `deepclust70`
  rows come from one shard for about 81 steps, and its `uniref70` rows from one shard for about
  49. Loss follows each shard's composition.
- `uniref70` is sampled at about 10.6 times its row share. The interleaver refills it in
  identical order every 63,177 steps, about 10.6 passes per epoch, while only 77 % of
  `deepclust70` is read per epoch.
- This likely explains why eval/uniref70 plateaus around 450k while the `deepclust70` evals keep
  improving.
- Each worker's stream ends on a partial batch, so an epoch closes with four steps of about 7
  rows per rank.

### The optimizer side builds up slowly

- The no-decay AdamW groups grow linearly with no sign of saturating:

  | Group | RMS at 0–50k | RMS at 600k |
  | --- | --- | --- |
  | `norm_gain` | 1.46 | 7.84 |
  | `residual_gate` | 1.68 | 9.98 |
  | `embed` | 0.15 | 0.92 |
  | `bias` | 0.21 | 1.50 |

  The Muon-trained `attn` and `mlp` RMS have been flat since 50k.
- Over the same span the gradient roughens:
  - median grad norm 0.078 → 0.127;
  - 99th percentile of the per-window max 0.20 → 0.54;
  - windows with a clipped step: 0–1 per 50k until 450k, then 5, 4 and 10;
  - largest spike: 3.1, at 550k.
- Eval loss has not suffered yet. This is the structural issue of §1, and it will reach longer
  runs.

### Replaying the stream on a frozen checkpoint

`oplm stream-replay <checkpoint> --world-size <N> --step S [--step S ...]` rebuilds the exact
batches one rank trained on from each step. It reads the run's resolved `config.yaml` from the
checkpoint and uses the trainer's own epoch and resume-skip arithmetic. It scores the
checkpoint's frozen model on them and reports, per window:

- the rows and mean sequence length;
- the source mix;
- the masked-token loss, overall and per `data.train` source.

`docs/late-loss/stream-replay.sbatch` scores `checkpoint-600000` on 30 windows of each of three
spans:

- a mid-epoch control, 450–479k;
- the last 29k steps of epoch 0;
- the first 29k steps of epoch 1.

Read the result like this:

- **The frozen model's loss is about 0.045 higher after the boundary:** the epoch-1 data is
  harder. The per-source columns and sequence lengths then say which source and why.
- **Pre and post look alike:** the jump came from the trained model's trajectory around the
  boundary.

The comparison is biased toward an easier post-boundary side: the 600k checkpoint never saw the
pre-boundary rows, but saw about 77 % of the post-boundary `deepclust70` shards once. A jump
despite that bias is strong evidence.
