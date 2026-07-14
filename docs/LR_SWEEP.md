# Production μP Learning-Rate Sweep

This document defines the learning-rate calibration protocol for the OPLM
production scaling series. It supersedes the short example sweep in
[MUP.md](MUP.md) when selecting a production learning rate.

The protocol has four goals:

1. tune the peak, stable-phase μP base learning rate on an affordable proxy;
2. verify transfer across OPLM's combined width-and-depth scaling ray;
3. measure, rather than assume, the correction from a roughly 1M-token proxy
   batch to the roughly 4M-token production batch; and
4. spend 100k-step and production-batch runs only on finalists.

Here, "tokens per step" means **total padded and non-padded input tokens**. Both
the 1M and 4M targets use this definition. OPLM does not pack unrelated proteins,
so using one quarter as many examples at the same per-device microbatch produces
a token ratio very close to 1:4. Log the measured ratio rather than assuming it
is exact.

## 1. Summary recommendation

Do not run every LR candidate for 100k steps at the 4M-token production batch.
Use the following funnel:

| Stage | Batch per optimizer step | Models | Typical duration |
| --- | ---: | --- | ---: |
| Parameterization gate | Small fixed batch | Several widths | 10–20 steps |
| Coarse LR screen | ~1M total tokens | 170M | 10k steps |
| Local refinement | ~1M total tokens | 170M | 20–30k steps |
| Width/depth confirmation | ~1M total tokens | 400M, 800M, ~1.6B | 10–30k steps |
| Batch bridge | ~4M total tokens | 170M | 10k steps |
| Joint batch/scale check | ~4M total tokens | 800M | 10k steps |
| Scaling experiment | ~4M total tokens | Selected series | 100k steps, winner only |

μP transfers hyperparameters across width when the other training conditions are
fixed. It does not make batch size irrelevant. The 1M-token sweep therefore
selects a 1M-batch LR; a small empirical bridge determines the production-batch
correction. Do not replace the bridge with an assumed linear or square-root LR
rule.

The 100k runs are scaling experiments, not LR-grid cells.

## 2. Canonical production scaling ray

Use a constant model aspect ratio and attention head dimension:

```text
hidden_size / num_hidden_layers = 32
head_dim = 64
num_attention_heads = hidden_size / 64
```

The intended series is:

| Role | Approximate size | Hidden | Layers | Heads | Head dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| Diagnostic only | 50M | 512 | 16 | 8 | 64 |
| μP/production anchor | 170M | 768 | 24 | 12 | 64 |
| Production | 400M | 1024 | 32 | 16 | 64 |
| Production | 800M | 1280 | 40 | **20** | 64 |
| Scaling confirmation | ~1.6B (`1B` preset) | 1600 | 50 | 25 | 64 |
| Optional | ~3.3B | 2048 | 64 | 32 | 64 |
| Production candidate | ~6B | 2560 | 80 | 40 | 64 |
| Production candidate | ~12.5B | 3200 | 100 | 50 | 64 |
| Production candidate | ~21B | 3840 | 120 | 60 | 64 |
| Production candidate | ~26B | 4096 | 128 | 64 | 64 |

The current `800M` preset has 16 heads, giving an unintended head dimension of
80. **Do not use that preset for this sweep until it is corrected to 20 heads.**
Verify every resolved model with `oplm info` before launch.

The existing `170M` preset is the closest current preset to the desired ~150M
production anchor. With all production features enabled, its resolved parameter
count is approximately 186M; use resolved counts rather than preset labels in
scaling-law analysis.

Use the 170M model as the primary tuning proxy and set:

```yaml
model:
  mup_enable: true
  mup_base_width: 768
```

This makes the smallest intended production model the μP reference. Changing
the reference from the old 512-wide model invalidates the historical `0.01`
selection; `0.01` is a candidate in the new sweep, not an accepted result. The
50M model remains useful for inexpensive diagnostics and scaling plots, but it
does not select the production LR.

## 3. Production recipe that must remain fixed

All LR cells must resolve from the standard run configuration, not from a bare
`OplmConfig()` or `TrainConfig()`. Keep these model settings fixed:

```yaml
model:
  norm_type: layernorm
  norm_strategy: sandwich
  qk_norm: true
  qk_norm_mode: channel
  residual_scaling: sqrt_num_layers
  attn_output_gate: sigmoid
  value_residual: learnable
  value_residual_lambda_init: 0.5
  init_scale_output_projections: true
  ffn_activation: swiglu
  attention_dropout: 0.0
  hidden_dropout: 0.0
  tie_word_embeddings: false
  canon_enabled: true
  canon_residual: true
  canon_positions: [A, B, C, D]
  canon_kernel_sizes: 7
  canon_activation: none
  mup_enable: true
  mup_base_width: 768
```

Keep these optimizer settings fixed during the primary LR sweep:

```yaml
train:
  optimizer: muon
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.98
  adam_eps: 1.0e-8
  muon_adjust_lr_fn: original
  muon_momentum: 0.95
  muon_nesterov: true
  muon_ns_steps: 5
  max_grad_norm: 1.0
  mixed_precision: bf16
```

`muon_adjust_lr_fn: original` is mandatory for μP. Do not use
`match_rms_adamw`, whose LR multiplier grows with width.

Also hold the following fixed:

- training and validation datasets and their ordering;
- dataset mixture;
- maximum-length and sequence-length distribution;
- masking probabilities and masking seed;
- loss reduction and ignored-label handling;
- per-device microbatch size, when possible;
- precision, compilation, checkpointing mode, and gradient clipping;
- Muon/AdamW parameter ownership and all relative parameter-group LR
  multipliers; and
- validation set and evaluation cadence.

The sweep is invalid if a cheap proxy silently disables sandwich norm, Canon,
attention gating, value residuals, weight decay, or other production features.

## 4. Batch accounting

The configuration expresses batch size in examples:

```text
global_examples = per_device_batch * gradient_accumulation_steps * world_size
```

Measure both of these quantities per optimizer step:

```text
total_tokens  = sum(input_ids.numel())
nonpad_tokens = sum(attention_mask.sum())
```

The sweep's 1M and 4M labels refer to `total_tokens`. The trainer currently logs
`train/tokens` from the attention mask, which is the non-padding count; the sweep
artifacts must additionally record total padded tokens. Also record the number
of supervised masked targets, `sum(labels != -100)`, because that is the closest
simple measure of the number of loss terms contributing to the gradient.

With approximately 512 total tokens per example, useful starting points are:

| Target | Global examples | Example one-process construction |
| --- | ---: | --- |
| ~1M total tokens | 2,048 | `batch_size=32`, `gradient_accumulation_steps=64` |
| ~4M total tokens | 8,192 | `batch_size=32`, `gradient_accumulation_steps=256` |

For distributed runs, divide accumulation by `world_size`. Recalculate from the
observed padded sequence length before launching the full grid. Keep the local
microbatch size the same between 1M and 4M runs so padding behavior stays as
similar as possible.

For the cleanest bridge, construct each 4M batch as the union of four consecutive
1M batches, including the same masking realizations. Evaluate batch comparisons
at both:

- equal optimizer steps, which exposes update-level stability; and
- equal total tokens, where a 1M run has four times as many optimizer updates.

## 5. Tooling prerequisites

Do not start the costly sweep until all of these are true:

1. The 800M preset resolves to 20 attention heads and `head_dim=64`.
2. Coordinate-check and sweep models load the resolved production configuration.
3. The sweep supports a stable peak LR: linear warmup followed by a constant
   plateau with no decay.
4. Sweep artifacts record the complete resolved model/train/data configuration,
   Git commit, seed, world size, global examples, total tokens, non-padding
   tokens, masked targets, and every optimizer group's effective LR.
5. Selection uses held-out loss at fixed checkpoints, not only final EMA
   training loss.
6. Failed and partial grid cells cannot produce a successful transfer verdict.
7. The runner can assign different seeds and resume or extend finalists without
   changing their data stream.

The current `scripts/mup_pilot_run.py` and `scripts/mup_sweep.py` do **not** meet
items 2 and 3: the pilot constructs a bare conservative model, and its default
`warmup_linear` scheduler decays after warmup. Until the tooling is updated, use
the standard training entry point for each cell so the production YAML defaults
are loaded.

For a 10k-step 170M example cell with 5k warmup steps and 5k stable steps:

```bash
accelerate launch -m oplm.train --preset 170M \
  data.train=/data/train/ \
  model.mup_base_width=768 \
  train.lr=0.01 \
  train.scheduler=wsd_linear \
  train.max_steps=10000 \
  train.warmup_steps=5000 \
  train.stable_steps=5000 \
  train.batch_size=32 \
  train.gradient_accumulation_steps=64 \
  train.output_dir=outputs/lr-sweep/170M/lr-0.01/seed-42 \
  train.wandb_run_name=lr-sweep-170M-lr-0.01-seed-42
```

Adjust accumulation for distributed world size and the measured sequence-length
distribution. Setting `stable_steps = max_steps - warmup_steps` leaves no decay
phase in OPLM's WSD scheduler.

## 6. Search space

### 6.1 Base learning rate

Start with a logarithmic grid centered around the historical `0.01`:

```text
0.0025, 0.0040, 0.0063, 0.0100, 0.0160, 0.0250, 0.0400
```

Adjacent values differ by approximately 1.6×. This is intentionally broad. If
the best result is the smallest or largest value, expand the grid before drawing
a conclusion.

After the coarse screen, refine around the winner with approximately 1.2–1.3×
spacing. For example, if `0.01` wins:

```text
0.0063, 0.0080, 0.0100, 0.0125, 0.0160
```

### 6.2 μP output multiplier

Start the coarse LR screen with:

```yaml
model:
  mup_output_mult: 1.0
```

Then evaluate this output-multiplier grid around the best LR and its immediate
neighbors:

```text
mup_output_mult = 0.5, 1.0, 2.0
```

For example, test a local 3×3 grid of three adjacent LRs by the three output
multipliers. Do not run the full seven-LR grid at all three multipliers unless
the local optimum lands at a boundary.

### 6.3 Weight decay

Keep `weight_decay=0.01` during LR and output-multiplier selection. Then test the
winning pair at:

```text
weight_decay = 0.003, 0.01, 0.03
```

Weight decay acts over optimizer steps and interacts with LR. Do not select it
from a zero-weight-decay pilot or silently rescale it between batch sizes. Use the
production-batch bridge to validate the final combination.

### 6.4 Depth correction

Standard μP primarily addresses width. This series also increases depth. Keep
one user-facing base LR, but test an internal repeated-block correction:

```text
effective_block_lr(L) = base_lr * (24 / L) ** alpha
alpha = 0.0, 0.25, 0.5
```

`alpha=0` is unchanged LR; `alpha=0.5` is the full `1/sqrt(depth)` hypothesis.
Apply the multiplier to parameters owned by repeated Transformer blocks,
including block attention/FFN, Canon, block norms, and block gates/residual
parameters. Do not apply it to token embeddings, the final stack norm, or the
MLM head/readout. Preserve all existing width-aware μP multipliers underneath
this depth multiplier.

The resulting multipliers are:

| Layers | Model example | `alpha=0` | `alpha=0.25` | `alpha=0.5` |
| ---: | --- | ---: | ---: | ---: |
| 24 | 170M | 1.000 | 1.000 | 1.000 |
| 32 | 400M | 1.000 | 0.931 | 0.866 |
| 40 | 800M | 1.000 | 0.880 | 0.775 |
| 50 | ~1.6B | 1.000 | 0.832 | 0.693 |
| 80 | ~6B | 1.000 | 0.740 | 0.548 |
| 100 | ~12.5B | 1.000 | 0.700 | 0.490 |
| 120 | ~21B | 1.000 | 0.669 | 0.447 |
| 128 | ~26B | 1.000 | 0.658 | 0.433 |

Do not choose `alpha` separately for each size. Select one exponent for the
entire production ray. The current configuration already combines
`1/sqrt(L)` residual writes with `1/sqrt(2L)` residual-writer initialization, so
the `alpha=0.5` LR correction is a hypothesis, not an automatic default.

If no exponent transfers cleanly, run a targeted architecture ablation comparing
`init_scale_output_projections=true` and `false` at 170M and 800M. Do not mix that
ablation into the primary LR grid.

### 6.5 Batch bridge multiplier

If the 1M-token sweep selects LR `eta_1m`, test at 4M tokens:

```text
eta_4m / eta_1m = 0.7, 1.0, 1.4, 2.0
```

Expand only if the winner is at an edge. The value 2.0 covers the usual
square-root hypothesis for a fourfold batch increase, but it is only a bracket,
not a prescription. Do not automatically test or adopt a 4× LR.

Keep production AdamW betas, epsilon, Muon momentum, and weight decay fixed in
the initial bridge. Analytically rescaling Adam's moments would introduce a
second, unvalidated transfer rule and would not cover the Muon parameter groups.

### 6.6 Seeds and intervals

Use:

```text
screening seed: 42
finalist seeds: 42, 43, 44
training log interval: 10 optimizer steps
validation interval: every 500–1,000 steps for 10k runs
                     every 2,000–5,000 steps for 20–100k runs
checkpoint interval: 5,000–10,000 steps during pilots
```

Use the same seed across LR cells during screening so data order and masks are
paired. Add seeds only after eliminating clearly inferior candidates.

## 7. Staged execution protocol

### Stage 0 — configuration and parameterization gates

Before consuming training-scale compute:

1. Run `oplm info` for every intended size and save the output.
2. Confirm the 32:1 width/depth ratio and 64-dimensional heads.
3. Run a fixed-depth μP coordinate check with production features enabled.
4. Run the non-μP control; it should visibly fan out where the μP run does not.
5. Run a short preset-ray coordinate check over at least 50M, 170M, 400M, and
   800M.
6. Inspect effective LR and parameter ownership for every optimizer group.

The coordinate check may use a tiny batch and short sequences. It tests
parameterization correctness, not the production LR or batch-size regime.

Reject the gate for non-finite values, systematic activation/update growth with
width, missing parameters, duplicated parameters, or unexpected optimizer
ownership.

### Stage 1 — 170M smoke tests

Run the low, middle, and high coarse-grid LRs for approximately 1,000 steps at
the 1M batch:

```text
0.0025, 0.0100, 0.0400
```

This stage checks launch correctness and catches immediate divergence. It does
not select the LR. In particular, the previously observed 1k–2k plateau and
later loss drop make final-loss selection at this horizon unsafe.

Use a short smoke-test warmup, for example 100 warmup steps followed by 900
stable steps, so every candidate actually reaches its proposed peak LR. This is
only an aggressive stability check; it does not replace validation under the
intended longer warmup.

### Stage 2 — 170M coarse LR screen

Run all seven coarse LRs for 10k steps at the 1M batch, using one seed. A useful
example schedule is 5k linear warmup steps followed by 5k stable steps. If a
shorter warmup is used for economy, document it and require at least 5k steps at
the full LR.

Eliminate clear instability and retain the best two or three candidates. Use
held-out loss at fixed checkpoints, not final training-loss EMA alone.

### Stage 3 — local LR, output-multiplier, and weight-decay refinement

At 170M and the 1M batch:

1. run the five-point local LR grid for 20–30k steps;
2. run the local 3-LR × 3-output-multiplier grid;
3. test the best LR/output pair at the three weight decays; and
4. repeat the final two or three configurations with seeds 42, 43, and 44.

The winning LR must be interior to the tested range. If two settings are
statistically indistinguishable, prefer the lower LR unless the higher LR has a
clear held-out-loss advantage without worse stability metrics.

### Stage 4 — width/depth transfer at the 1M batch

Using only the best two base configurations, run:

```text
400M: 10–20k steps
800M: 20–30k steps
~1.6B: 10–20k steps
alpha: 0.0, 0.25, 0.5
```

The 800M run receives the longest confirmation because that is where the
original instability appeared. Require at least 10k stable-LR steps after
warmup before accepting it.

Select one `alpha` for the ray. A transfer passes when:

- the preferred base LR is identical or within one adjacent grid interval at
  every size;
- no model's optimum lies on the edge of the LR grid;
- larger models do not show the anomalous loss inversion seen previously;
- instability, clipping, activation, and update statistics remain comparable;
  and
- the conclusion is unchanged across finalist seeds.

Exact agreement on one discrete LR value is not required when neighboring
values are statistically tied. Conversely, exact argmin agreement on a coarse
grid is not sufficient evidence of transfer.

### Stage 5 — 170M batch bridge

At 170M, run the four batch-multiplier candidates at the 4M production batch
using the intended production warmup followed by at least 5k stable-LR steps.
With a 5k warmup this is a 10k-step run. Repeat the best one or two candidates
with additional seeds.

Compare 1M and 4M runs by both tokens and optimizer steps. Record:

```text
c_batch = eta_4m / eta_1m
```

For example, the 5k-step point of a 4M run has approximately the same token
exposure as the 20k-step point of a 1M run. Use measured total-token counts to
pair checkpoints exactly.

If the same LR region wins, use `c_batch=1`. If a different region wins, retain
the empirical correction. Do not redo the entire width grid at 4M.

### Stage 6 — joint batch/scale confirmation

Run the top one or two corrected configurations on the 800M model at the 4M
batch for at least 10k steps. This is the critical interaction test: it checks
that the independently estimated depth and batch corrections compose.

If they do not compose, stop. First expand the local LR bracket at 800M/4M; if
the discrepancy persists, batch and model scale interact and the simple
factorized transfer law is not adequate.

Before a 6–24B production launch, also run one short safety confirmation at the
selected large-model geometry. This is a safety check, not another sweep.

### Stage 7 — 100k scaling runs

Only the winning configuration proceeds to the planned 100k-step, 4M-token
scaling experiments. These runs use warmup followed by a stable LR and no
decay. Run the desired sizes, including 50M if useful for the scaling curve, but
do not let the 50M result override the production anchor.

Use the 100k results to verify:

- monotonically improving held-out loss with model scale at comparable compute
  and data exposure;
- stable per-group update/weight ratios;
- absence of the 800M early plateau and LR spikes;
- consistent transfer through ~1.6B; and
- weight-decay behavior over a materially longer plateau.

## 8. Warmup, stable phase, and production cooldown

The sweep selects the **peak stable LR**. It does not select the cooldown.

Pilot and 100k runs use:

```text
linear warmup -> constant stable LR -> stop
```

Production uses:

```text
linear warmup
-> 1–2M stable-LR steps
-> 0.5–1M linear-decay steps
```

Configure production with `scheduler=wsd_linear`, `stable_steps` equal to the
desired plateau, and `max_steps` equal to warmup + plateau + decay. For example:

```yaml
train:
  scheduler: wsd_linear
  lr: 0.01  # illustrative only; replace with the Stage 6 production LR
  min_lr: 0.0
  warmup_steps: 10_000
  stable_steps: 1_500_000
  max_steps: 2_260_000  # 10k warmup + 1.5M stable + 750k decay
```

Tune cooldown duration by branching several decay schedules from a common
stable checkpoint; do not repeat the base-LR sweep. Continue optimizer state
through the small phase-two dataset-composition change. Do not reset momentum or
add a second warmup. When operationally possible, blend the mixture change over
roughly 10k–50k steps rather than switching every source weight at one boundary.

## 9. Required metrics and rejection rules

### Primary selection metric

Use deterministic held-out MLM loss evaluated at fixed total-token checkpoints.
Also report masked-token accuracy and perplexity, but do not combine them into a
single score.

### Stability metrics

Record at least:

- raw and smoothed training loss;
- validation loss versus optimizer steps and total tokens;
- global gradient norm before clipping;
- fraction of optimizer steps clipped;
- non-finite loss, gradient, parameter, or optimizer-state counts;
- weight RMS, update RMS, and update/weight RMS by optimizer group;
- effective LR for every Muon and AdamW parameter group;
- residual-stream RMS by depth;
- attention-logit RMS or scale, attention entropy, and gate statistics;
- Canon output RMS at A/B/C/D;
- total, non-padding, and masked-target tokens per optimizer step; and
- throughput and MFU.

Automatically reject a run for any non-finite value or sustained unbounded
growth. Flag for review when, after warmup:

- more than 10% of steps clip gradients;
- clipping or update/weight ratios drift systematically upward;
- the loss repeatedly exceeds its rolling median by more than five median
  absolute deviations; or
- a larger model is persistently worse than the 50M/170M reference under the
  same comparison basis.

These review thresholds are diagnostics, not a substitute for inspecting the
curves. A candidate with frequent spikes should not win on a marginally lower
final loss.

## 10. Provenance and artifacts

Each run directory must contain:

```text
resolved_config.yaml
model_info.txt
git_commit.txt
environment.txt
metrics.jsonl or metrics.parquet
validation_metrics.jsonl or validation_metrics.parquet
optimizer_groups.json
batch_accounting.json
checkpoints/ (finalists only)
```

`batch_accounting.json` should include the requested and observed batch sizes,
world size, accumulation, examples, total tokens, non-padding tokens, masked
targets, and their distributions over steps.

The sweep-level summary must include:

- every planned cell, including failures and incomplete cells;
- loss-versus-LR curves with uncertainty across seeds;
- best LR region per model size;
- best depth exponent and batch correction;
- stability/rejection reasons;
- compute consumed per stage; and
- the exact final production formula.

A representative final formula is:

```text
eta_nonblock = eta_base_4m
eta_block(L) = eta_base_4m * (24 / L) ** alpha
```

Record the selected `eta_base_4m`, `alpha`, `mup_output_mult`, weight decay, and
the empirical `c_batch` that produced it.

## 11. Where to economize

Economize in this order:

1. shorten clearly inferior grid cells;
2. use one seed for screening;
3. refine locally instead of repeating the broad grid;
4. sweep broadly only at 170M;
5. use 1M-token batches for most cells; and
6. reserve 100k and 4M-token runs for finalists.

Do not economize by changing the architecture, optimizer ownership, data
mixture, sequence-length distribution, masking, loss normalization, or head
dimension. Those changes answer a different question and invalidate transfer.

## 12. Decision checklist

The production LR is approved only when every item below is true:

- [ ] All production models use 32:1 width/depth and 64-dimensional heads.
- [ ] The 800M model has 20 attention heads.
- [ ] The production architecture is used in every LR-training cell.
- [ ] Fixed-depth and preset-ray coordinate checks pass.
- [ ] The coarse and local LR optima are interior to their grids.
- [ ] `mup_output_mult` and weight decay have been checked locally.
- [ ] One depth exponent transfers through at least ~1.6B.
- [ ] The 1M-to-4M batch correction has been measured at 170M.
- [ ] The combined correction is stable at 800M/4M.
- [ ] Finalists have been repeated across seeds.
- [ ] The winning configuration completes the 100k scaling series without the
      previous plateau, spike, or loss-inversion behavior.
- [ ] A short safety run passes at the selected 6–24B geometry.
- [ ] The resolved production config and transfer formula are archived.

## References

- [MUP.md](MUP.md) — OPLM's μP implementation and coordinate checks.
- [TRAIN.md](TRAIN.md) — training, schedules, distributed launch, and resume.
- [CONFIG.md](CONFIG.md) — model and training configuration fields.
- [Tensor Programs V: μTransfer](https://arxiv.org/abs/2203.03466).
- [Depthwise Hyperparameter Transfer in Residual Networks](https://arxiv.org/abs/2309.16620).
- [Tensor Programs VI: Depth-μP](https://arxiv.org/abs/2310.02244).
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162).
- [Critical Batch Size Revisited](https://arxiv.org/abs/2505.23971).
- [Surge Phenomenon in Optimal Learning Rate and Batch Size
  Scaling](https://arxiv.org/abs/2405.14578).
