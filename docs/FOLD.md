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
autograd. Choose per width from the benchmark below and record the choice in the
stage config. Residual add and row-shared dropout belong to the owning pair
block (milestone 1), as upstream.

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

_Pending: filled by the milestone-0 acceptance run (plan Task 7)._
