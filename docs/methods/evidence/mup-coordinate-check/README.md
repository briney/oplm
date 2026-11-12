# The μP implementation passes the coordinate check

**Status:** settled

## Question

Is μP implemented correctly, so that a base learning rate tuned at one width carries over to the
other model sizes? The test is one-sided: with μP on, no module's activation RMS may grow with
width after a few optimizer steps; with μP off it must.

## Arms

`oplm sweep coord-check`, 3 optimizer steps, activation RMS per hooked module, for two recipes:

- **oplm:** the production recipe (Muon, sandwich norm, Canon), at commit 7150334.
- **esmc:** the ESM-C comparison recipe (AdamW), at commit 90e8d9a.

Each recipe has three runs:

- **width, μP on:** widths 384, 768, 1536 at depth 24, base width 768.
- **width, μP off:** the same, as the control.
- **preset ray, μP on:** widths 512, 768, 1024, 1280 with depth co-scaled at 32:1, as the model
  presets do.

## Result

Both recipes pass: with μP on the worst module grows 1.5× or less across a fourfold width range
(the tool's threshold is 2×), and the μP-off control fans out by 10× to 20×.

| arm | widths | worst module | growth |
|---|---|---|---|
| oplm, width, μP on | 384 → 1536 | `lm_head.decoder` | 1.51× |
| oplm, width, μP off | 384 → 1536 | `lm_head.decoder` | 9.74× |
| oplm, preset ray, μP on | 512 → 1280 | `lm_head.dense` | 1.23× |
| esmc, width, μP on | 384 → 1536 | `embed_tokens` | 1.00× |
| esmc, width, μP off | 384 → 1536 | `ffn.down_proj` | 20.62× |
| esmc, preset ray, μP on | 512 → 1280 | `lm_head.decoder` | 1.42× |

In the oplm width run every backbone module is within 0.95× to 1.03×; only the readout moves.
`growth.csv` has every module for every run, and `raw/` has the tool's per-layer, per-step output
(columns `width`, `module`, `step`, `rms`).

## Runs

- oplm: Slurm job 36026 (2026-11-12, 1 GPU), cluster output in
  `~/projects/oplm/mup_redo/coord-oplm-7150334/`, log `~/logs/coord-oplm_36026.out`.
- esmc: Slurm job 33343 (2026-10-31), cluster output in `~/projects/oplm/mup_redo/sweeps-esmc/coord-*`,
  log `~/logs/coord-esmc_33343.out`.
- `configs/` holds the sweep config each run was given, not a resolved config: the tool builds its
  own models at each width. These runs do not log to W&B.
- How to run it and how to read it: [mup-lr.md](../../../mup-lr.md), *Run the coordinate check* and
  *Coordinate check*.

## Caveats

- **The oplm run was made for this record,** on current `main`, not before the learning-rate
  funnel (commit 6e8b277). No output from an earlier Muon coordinate check was kept.
- **Not every module is hooked.** Attention pre-softmax logits and the Canon convolutions do not
  appear in the table.
- **The oplm readout grows 1.51×,** inside the tool's threshold but the largest μP-on growth of
  either recipe.
- **Three steps at initialization scale.** The check verifies the parameterization, not that the
  learning-rate optimum transfers; that is tested by the 800M cells of
  [base-lr-funnel](../base-lr-funnel/README.md).
- The preset ray co-scales depth with width, so it is an empirical check of the combined behavior,
  not a substitute for the fixed-depth width run.
