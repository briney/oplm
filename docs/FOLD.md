# Structure Prediction Head (`oplm.fold`)

Architecture contract for the folding head, in the role `MODEL_ARCHITECTURE.md`
plays for the language model. The agreed design is
[`docs/superpowers/specs/2026-10-07-structure-prediction-head-design.md`](superpowers/specs/2026-10-07-structure-prediction-head-design.md);
this file records what is implemented. **Status: milestone 0 (foundations and
kernels).** Model, data, losses, and training land in later milestones.

## 1. Package layout (implemented so far)

```
src/oplm/fold/
├── trimul.py      # staged triangle multiplication + cuEquivariance dispatch
├── attention.py   # pair-biased / sliding-window attention: FlexAttention + dense oracle
└── cli.py         # oplm fold bench-kernels
tests/fold/        # mirrors the above; GPU parity tests are `slow` and skip without CUDA
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
(`docs/TRAIN.md` §17); `FoldTask` lands in milestone 2. `train.ema_decay` keeps an
EMA of the trainable weights, updated once per optimizer step, saved as `ema.pt` and
exported as `hf_ema/` with every checkpoint (`docs/TRAIN.md` §16). The frozen LM
will be an unregistered attribute of the fold model and must be excluded from the
EMA's deep copy (milestone 1).

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

**What exists.** `FoldConfig` (defaults = the released `biohub/ESMFold2-Fast` config; `docs/
fold/m1/` records the parity run), `featurize()` (protein chains -> `FoldFeatures`; one LM row
per chain with BOS/EOS; atoms padded to 32; tokens optionally padded to a crop multiple),
`OplmForFolding` (checkpoint-identical module names; `base_model_prefix` is `oplm_fold`; frozen
LM held outside the module tree via `attach_lm` / `lm_name_or_path`;
`forward(features) -> FoldOutput`), `fold()` + `write_mmcif()` and `oplm fold predict`.
`oplm fold make-fixtures` + `docs/fold/b200-fixtures.sbatch` record the parity oracle;
`tests/fold/test_parity.py` runs when `OPLM_FOLD_FIXTURES` points at it.

**Deviations from upstream, all deliberate.**
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
  repo) makes `from_pretrained` raise a `ValueError`; such heads run with precomputed
  `lm_hidden_states`.
- `fold()` ranks samples by ipTM (complex) / pTM (monomer); upstream's `fold()` does not rank.
- Coordinates, the sampler and the confidence math disable any outer autocast locally (spec
  §5.4); the trunk portion of `forward` runs under bf16 autocast on CUDA only.

**Parity (filled in by Task 12).** | stage | atol used | max abs err observed | cases |
