# Milestone 2b acceptance: single-B200 cost measurement

Plan: `docs/superpowers/plans/2026-10-12-fold-m2b-training.md`, Task 12. Addendum:
`docs/superpowers/specs/2026-10-11-fold-m2-pipeline-proof-design.md` §5–6. Script:
[`docs/fold/b200-m2b-measure.sbatch`](../b200-m2b-measure.sbatch).

**State: run 1 recorded. Both widths measured cleanly. Three steps failed, none of them in the
code under measurement, and all three are fixed on the branch. Run 2 (tests plus a per-device
batch sweep) is pending; the milestone-2c batch and accumulation are set from it.**

## Run 1: job 25716, 2026-10-13

- Node `slurm-b200-193-139`, one NVIDIA B200 (183,359 MiB, [`nvidia-smi.txt`](nvidia-smi.txt)),
  container `deeplearning_v2026-05-26.sqsh`.
- Code: branch `feat/fold-m2b` at `f05d625` ([`git-rev.txt`](git-rev.txt)).
- Environment ([`pip-freeze.txt`](pip-freeze.txt)): CPython 3.12.13, torch 2.11.0+cu130 (CUDA 13.0),
  cuequivariance 0.12.0, esm 3.4.1.post1, accelerate 1.15.0, **transformers 5.3.0** and
  huggingface-hub 1.33.0. That transformers version is outside esm's `<5` pin (failure 3 below).
- Data: `brineylab/oplmfold-training-data` at tag `wwpdb-2026-10-12`.

Step exits ([`status.txt`](status.txt)): `venv`, `install`, `env`, `download`, `measure-w128`
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
   ([`esmc.log`](esmc.log), [`fold-gpu-tests.xml`](fold-gpu-tests.xml)).
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
2. **`cpu-tests`: an artefact of a GPU host** ([`cpu-tests.log`](cpu-tests.log),
   [`fold-cpu-tests.xml`](fold-cpu-tests.xml)).
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

## Run 2: tests and batch sweep (pending)

```bash
OPLM_GIT_REF=feat/fold-m2b OPLM_STEPS="venv install env esmc cpu-tests sweep gpu-tests" \
  sbatch docs/fold/b200-m2b-measure.sbatch
```

The `sweep` group runs `oplm fold measure` at per-device batch 2, 4 and 8, each width, with
accumulation 16/batch. That keeps 16 samples per device per optimizer step, so
8 devices × batch × accumulation stays 128. Copy `status.txt`, `pip-freeze.txt`, the two junit
XMLs and `sweep-w*.json` here. Run 2's files replace run 1's; git history keeps run 1's.

## Milestone-2c budget

**Device budget.** Rule (plan Task 12): the largest per-device batch whose peak allocated memory
stays under 75 % of the device (134.3 GiB of 179.1 GiB). `8 × batch × accumulation = 128`.

**Projection from run 1, to be replaced by run 2's measurements.**

- Fixed state, per device:
  - head parameters, gradients, two AdamW moments and the EMA copy at 20 bytes per parameter:
    2.73 GiB (146.3 M parameters, w128) and 3.50 GiB (188.1 M, w256);
  - the bf16 ESM-C 600M: about 1.1 GiB.
- Per sample: what remains of the batch-1 peak, about 11.4 GiB (w128) and 16.2 GiB (w256).
- Linear projection, which padding to the crop makes reasonable:

| width | batch 2 | batch 4 | batch 8 | batch 16 | projected choice |
|---|---|---|---|---|---|
| 128 | 27 GiB | 49 GiB | 95 GiB (53 %) | 186 GiB (over) | batch 8, accumulation 2 |
| 256 | 37 GiB | 69 GiB | 134 GiB (75 %, at the line) | — | batch 4, accumulation 4; batch 8 only if measured under the line |

**Step budget.** Spec §6.3 sets 50k steps for stage 1. At batch 1 × 16 on 8 devices (DDP
all-reduce not included), 50k steps take:

- w128: 225 h, 9.4 days, about 1,800 GPU-hours;
- w256: 286 h, 11.9 days, about 2,300 GPU-hours.

A larger batch shortens that only by as much as run 2 shows it raising throughput. The spec gives
no GPU-hour allocation. The steps that fit, and so the cut below 50k if any, are set against the
allocation once run 2's step times are in.

`configs/fold/proof-w{128,256}.yaml` keep the measurement defaults (batch 1, accumulation 16,
50k steps) until then. The `measure-w*` steps read those values, and the sweep overrides them on
the command line.
