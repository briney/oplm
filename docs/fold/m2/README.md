# Milestone 2b acceptance: single-B200 cost measurement

Plan: `docs/superpowers/plans/2026-10-12-fold-m2b-training.md`, Task 12. Addendum:
`docs/superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md` §5–6. Script:
[`docs/fold/b200-m2b-measure.sbatch`](../b200-m2b-measure.sbatch).

**State: accepted 2026-10-13.**

- **Run 1** measured both widths at batch 1. Three steps failed, none of them in the code under
  measurement, and all three were fixed.
- **Run 2** passed every step on the corrected environment, and its per-device batch sweep set
  the milestone-2c configs: width 128 at batch 8 × accumulation 2, width 256 at batch 4 ×
  accumulation 4, 50k steps.

The status, environment, test and node files here are run 2's. Run 1's copies are in git
history at `25ce5af`, e.g. `git show 25ce5af:docs/fold/m2/status.txt`.

## Run 1: job 25716, 2026-10-13

- Node `slurm-b200-193-139`, one NVIDIA B200 (183,359 MiB), container
  `deeplearning_v2026-05-26.sqsh`.
- Code: branch `feat/fold-m2b` at `f05d625`.
- Environment (run 1's `pip-freeze.txt`): CPython 3.12.13, torch 2.11.0+cu130 (CUDA 13.0),
  cuequivariance 0.12.0, esm 3.4.1.post1, accelerate 1.15.0, **transformers 5.3.0** and
  huggingface-hub 1.33.0. That transformers version is outside esm's `<5` pin (failure 3 below).
- Data: `brineylab/oplmfold-training-data` at tag `wwpdb-2026-10-12`.

Step exits (run 1's `status.txt`): `venv`, `install`, `env`, `download`, `measure-w128`
and `measure-w256` passed (exit 0). `esmc`, `cpu-tests` and `gpu-tests` failed (exit 1).

### Measurement

Settings, shared by both widths: `configs/fold/proof-w{128,256}.yaml`, 20 optimizer steps,
`batch_size` 1, `gradient_accumulation_steps` 16, crop 384, 24 trunk blocks, frozen
`esmc:esmc_600m`, `diffusion_samples` 8, `diffusion_microbatch` 4, `confidence_rollout_steps`
20, bf16, one process. Step time is the mean over steps 4–20, per optimizer step, and includes all
16 micro-steps.

| width | step time | micro-step | range (steps 4–20) | peak allocated | peak reserved | share of device |
|---|---|---|---|---|---|---|
| 128 ([`measure-w128.json`](measure-w128.json)) | 16.21 s | 1.013 s | 15.3–17.2 s | 15.26 GiB | 21.62 GiB | 8.5 % |
| 256 ([`measure-w256.json`](measure-w256.json)) | 20.62 s | 1.289 s | 19.3–22.0 s | 20.76 GiB | 26.91 GiB | 11.6 % |

Every logged value is finite at both widths, and no `error` field was written.

**Losses.** The learning rate is still in warmup (≤ 2e-5), so these show only that every term
trains:

- **Distogram:** falls 4.21 → 3.29 (w128) and 4.36 → 3.23 (w256).
- **Diffusion MSE:** stays at 0.8–0.9.
- **Confidence:** 12.95 → 12.82. That is the uniform-logit value: ln 50 + ln 64 + ln 64 + ln 2 =
  12.92 over the pLDDT, PDE, PAE and resolved heads. At weight 1e-4 it is expected to move
  slowly.
- **Gradient clipping:** every step was clipped (`clip_frac` 1.0). The gradient norm falls
  5.2 → 1.9 (w128) and 4.4 → 1.2 (w256) against `max_grad_norm` 1.0. Watch this in 2c once the
  learning rate is up.

**What the time is spent on.** Tokens always pad to `crop_size` (`oplm.fold.data.collate`), so
every micro-step costs the full 384-token shape.

- Step time barely tracks the batch's mean chain length (141–260 residues here).
- Width 256 costs only 1.27× width 128. Much of a micro-step is therefore width-independent
  (frozen LM, the diffusion module at 8 samples, the 20-step confidence rollout), or the GPU is
  not saturated at batch 1.
- Whether a larger per-device batch amortises that cost is what run 2 measures.

### Failures and fixes

1. **`esmc` and `gpu-tests`: one test, a tolerance too strict for fp32**
   (run 1's `esmc.log` and `fold-gpu-tests.xml`).
   `test_real_esmc_300m_padded_chain_rows_match_the_unpadded_chain` compared a chain's ESM-C
   states, batched right-padded next to a longer chain, against the chain alone at `atol=1e-4`.
   It failed at 6.7e-4, on layer 29.

   A per-layer probe (esmc_300m, CPU, esm 3.4.1.post1) shows this is reduction-order noise, not
   leakage. With padding masked, the max difference grows with depth from 3.8e-6 to 4.3e-4
   locally. That is at most 8.9e-6 of the layer's RMS, which is about 53 at layer 29, so the
   job's 6.7e-4 is about 1.3e-5 relative. With the padding left unmasked, the difference reaches
   115, at least 0.90 of the layer RMS in every layer after the embedding.

   The test now bounds the worst per-layer `max|diff| / RMS` below 1e-4. It also asserts that the
   unmasked control exceeds 0.1, so the check can still see a leak.
2. **`cpu-tests`: an artefact of a GPU host** (run 1's `cpu-tests.log` and
   `fold-cpu-tests.xml`).
   `test_step_returns_accounting_and_gradients_stay_in_the_head` builds the model without a
   Trainer. `build_model` places the frozen LM on `PartialState().device`, and a fresh
   `PartialState` ignores `ACCELERATE_USE_CPU` and picks the visible GPU. The LM ran in bf16 on
   CUDA while the head stayed fp32 on the CPU: `expected m1 and m2 to have the same dtype`. The
   test now initialises `PartialState(cpu=True)`, as the Trainer's Accelerator would. That was
   the only failure in 309 tests. `build_model` itself needs no change: in real runs the Trainer
   creates the Accelerator first.
3. **Environment: transformers 5.3.0 beside esm's `transformers<5`.** The install step ran
   esm first and the repo second. The second resolve never saw esm, upgraded huggingface-hub to
   1.x (through the latest `datasets`) and transformers with it.

   The measurements and the remaining tests ran correctly on it. Still, the ESM-C loader is
   unsupported off its pin. The install is now a single resolve of `[dev,train,fold,esmc]`
   followed by `uv pip check`. Locally that resolves to transformers 4.57.6 and huggingface-hub
   0.36.2 with cuequivariance 0.12.0, and a reused venv is downgraded back into range.

**Local check of the fixed toolchain** (CPython 3.12, esm 3.4.1.post1, transformers 4.57.6,
CPU):

- **Fold tests:** every one passes, including the real ESM-C tests. The 29 GPU-only tests skip.
- **Rest of the repository:** one MLM test fails,
  `test_explicit_tied_aliases_preserve_tying_or_reject_conflicting_weights[True]`. Under
  transformers 4.x, `from_pretrained` re-ties the embedding and head before
  `oplm.training.initialization` can compare them. A conflicting saved alias is then dropped
  without error. On the repository's transformers 5.3 venv it is rejected.
- **Effect on this job:** none, since the job runs only `tests/fold`. The failure is a gap in the
  declared `transformers>=4.45` range, tracked separately.

## Run 2: job 25799, 2026-10-13

```bash
OPLM_GIT_REF=feat/fold-m2b OPLM_STEPS="venv install env esmc cpu-tests sweep gpu-tests" \
  sbatch docs/fold/b200-m2b-measure.sbatch
```

- Node `slurm-b200-193-135`, one NVIDIA B200 (183,359 MiB, [`nvidia-smi.txt`](nvidia-smi.txt)).
- Code: `25ce5af` ([`git-rev.txt`](git-rev.txt)).
- Environment ([`pip-freeze.txt`](pip-freeze.txt), [`install.log`](install.log)): esm
  3.4.1.post1, **transformers 4.57.6**, huggingface-hub 0.36.2, torch 2.11.0+cu130, cuequivariance
  0.12.0. `uv pip check` reports all 162 packages compatible.

Every step that ran exited 0 ([`status.txt`](status.txt)); `download` and the batch-1 `measure`
steps were skipped by `OPLM_STEPS`.

| step | result |
|---|---|
| `esmc` ([`esmc.log`](esmc.log)) | 5 passed, 1 skipped (the ImportError path, unreachable with esm installed) |
| `cpu-tests` ([`fold-cpu-tests.xml`](fold-cpu-tests.xml), [`cpu-tests.log`](cpu-tests.log)) | 309 tests, 0 failures, 40 skipped |
| `gpu-tests` ([`fold-gpu-tests.xml`](fold-gpu-tests.xml), [`gpu-tests.log`](gpu-tests.log)) | 34 tests, 0 failures, 1 skipped (ESMFold2 parity, which needs the M1 fixtures) |

### Batch sweep

Each sweep entry is 20 optimizer steps at 16 samples per device per step (batch × accumulation =
16), with otherwise the same settings as run 1. Every value is finite at every size, and no
`error` field was written. The batch-1 row is run 1's, measured on another node of the same type.

| width | batch × accum | step time | range | peak allocated | peak reserved | report |
|---|---|---|---|---|---|---|
| 128 | 1 × 16 | 16.21 s | 15.3–17.2 s | 15.26 GiB (8.5 %) | 21.62 GiB (12.1 %) | [`measure-w128.json`](measure-w128.json) |
| 128 | 2 × 8 | 12.09 s | 11.2–12.7 s | 26.22 GiB (14.6 %) | 40.16 GiB (22.4 %) | [`sweep-w128-b2.json`](sweep-w128-b2.json) |
| 128 | 4 × 4 | 10.89 s | 10.0–11.8 s | 48.02 GiB (26.8 %) | 67.24 GiB (37.6 %) | [`sweep-w128-b4.json`](sweep-w128-b4.json) |
| 128 | **8 × 2** | **10.22 s** | 8.1–11.4 s | **91.70 GiB (51.2 %)** | 131.53 GiB (73.5 %) | [`sweep-w128-b8.json`](sweep-w128-b8.json) |
| 256 | 1 × 16 | 20.62 s | 19.3–22.0 s | 20.76 GiB (11.6 %) | 26.91 GiB (15.0 %) | [`measure-w256.json`](measure-w256.json) |
| 256 | 2 × 8 | 18.48 s | 16.4–20.0 s | 36.41 GiB (20.3 %) | 50.20 GiB (28.0 %) | [`sweep-w256-b2.json`](sweep-w256-b2.json) |
| 256 | **4 × 4** | **17.31 s** | 15.2–19.5 s | **67.66 GiB (37.8 %)** | 86.65 GiB (48.4 %) | [`sweep-w256-b4.json`](sweep-w256-b4.json) |
| 256 | 8 × 2 | 16.89 s | 11.8–19.7 s | 130.21 GiB (72.7 %) | 169.77 GiB (94.8 %) | [`sweep-w256-b8.json`](sweep-w256-b8.json) |

Percentages are of the 179.1 GiB device.

- **Memory** grows linearly with batch, as padding every sample to the crop predicts. Run 1
  projected 95 GiB (w128 batch 8), 69 GiB (w256 batch 4) and 134 GiB (w256 batch 8); the measured
  values are 91.7, 67.7 and 130.2 GiB.
- **Throughput** gains from batching are modest. Batch 1 already keeps a B200 mostly busy at a
  384-token crop. From batch 1 to batch 8, an optimizer step gets 1.59× faster at width 128 and
  1.22× faster at width 256. Most of that comes by batch 4.

## Milestone-2c budget

**Device budget.** Rule (plan Task 12): the largest per-device batch whose peak allocated memory
stays under 75 % of the device (134.3 GiB). `8 × batch × accumulation = 128`.

- **Width 128: batch 8, accumulation 2.** Batch 16 would need about 180 GiB. Batch 8 peaks at
  51 % allocated and 74 % reserved.
- **Width 256: batch 4, accumulation 4. This is one step below the rule.** Batch 8 passes the
  rule at 72.7 % allocated, but in 20 steps the caching allocator already reserved 94.8 % of the
  device. A 50k-step run will meet rarer shapes than 20 steps do, such as more recycling loops
  or longer LM inputs, so batch 8 would run with almost no margin against an out-of-memory error.
  It is only 2.4 % faster than batch 4 (16.89 s against 17.31 s). Batch 4 peaks at 38 %
  allocated and 48 % reserved. If a later run wants the 2.4 %, change `batch_size: 8` and
  `gradient_accumulation_steps: 2` in `configs/fold/proof-w256.yaml`; nothing else depends on
  the choice.

**Step budget: 50k**, spec §6.3's stage-1 budget. The spec sets no GPU-hour allocation that
would cut it. At the chosen batches, on 8 B200s, with DDP all-reduce and data loading assumed
hidden:

| width | step time | 50k steps | GPU-hours |
|---|---|---|---|
| 128 | 10.22 s | 142 h (5.9 days) | 1,136 |
| 256 | 17.31 s | 240 h (10.0 days) | 1,923 |

Together: 382 h on one node in sequence (15.9 days), or 10 days on two nodes in parallel; about
3,060 GPU-hours. At batch 1 the same budget would have cost 225 h and 286 h, so batching saves 37 %
(w128) and 16 % (w256).

Both settings are in [`configs/fold/proof-w128.yaml`](../../../configs/fold/proof-w128.yaml) and
[`proof-w256.yaml`](../../../configs/fold/proof-w256.yaml).
`tests/fold/test_cli.py::test_proof_configs_load` checks that each config gives a global batch of
128 on 8 devices.
