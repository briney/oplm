# Fused Canon kernel

## Why

OPLM trains at about half ESM-C's speed: 895 ms vs 418 ms per step for the 170M preset on
4 nodes at 4M tokens/step (profile of 2026-11-01, oplm@90e8d9a; summaries and traces in
`~/data/tmp/prof/{oplm-muon,oplm-adamw,esmc}/profile/`). Matmuls (~100 ms/step) and flex
attention (~186 ms/step) cost the same in both models. The gap is memory-bound elementwise and
reduction kernels: 531 ms/step for OPLM vs 110 ms/step for ESM-C, almost all of it Canon. Muon is
45 ms (5%) and the sandwich norms plus the attention output gate about 20 ms.

Canon (`src/oplm/model/conv.py`, `CanonConv.forward`) is written as `kernel_size` shifted
multiply-adds that `torch.compile` fuses. What makes it slow:

- the padded input is materialized in fp32;
- in packed batches every tap builds and applies its own same-segment mask;
- in the backward pass each tap's weight gradient is a separate full-tensor reduction (about
  2,000 small reduction kernels per step), and the input gradient is likewise a sum of shifted
  masked products.

A 170M layer runs six Canon convs (positions A and C at `hidden_size`, B on each of Q, K and V at
`hidden_size`, D at `intermediate_size`), so this cost scales with every token and every layer.

The goal is the production runs coming next (170M–1B on 4–8 nodes): as close to ESM-C's step
time as is cheap, without touching the model.

## Goals

- A hand-written GPU kernel for the Canon convolution, forward and backward, that computes the
  same function as today's `CanonConv.forward` and gives the same gradients.
- Canon's share of the 170M step down from roughly 400 ms to roughly 50 ms; target step at or
  below about 550 ms in the same 4-node profile (ESM-C: 418 ms).
- Works under `torch.compile` (the trainer compiles the DDP-wrapped model) with no graph breaks.

## Non-goals

- Distributed or batched Muon (5% at this scale; stays on card #53 for the 6–24B scale-up).
- Fusing the sandwich norms, the attention output gate or the value residual (~20 ms together).
- Any change to what the model computes. The μP result for the current architecture was
  expensive and must carry over untouched; so must OPLM's loss lead over ESM-C.

## Anti-goals

- A kernel that changes the loss curve beyond numerical noise. Equivalence is the acceptance
  test, not a nice-to-have.
- A config knob users have to set. The fast path is chosen automatically.
- New checkpoint formats. `CanonConv.conv.weight` keeps owning the `(D, 1, k)` weight, so
  checkpoints, initialization and μP parameter grouping are unchanged.

## What the kernel must compute

Exactly today's semantics (see `CanonConv` and `tests/model/test_canon_semantics.py`):

- Input `x` is `(B, T, D)`, any float dtype; under autocast the output has the autocast dtype,
  otherwise `x`'s dtype. Accumulation is fp32.
- Depthwise, bidirectional cross-correlation along T with kernel `k >= 2`, odd or even. Left
  padding `k // 2`, right padding `k - 1 - k // 2` (even kernels are asymmetric). Kernel size can
  differ per layer (`resolve_canon_kernel_sizes`), so `k` is a per-call parameter.
- Pad positions (`attention_mask == 0`) are zeroed before the conv, so they contribute nothing.
- With `segment_ids` (packed rows: `1..n` per sequence, `0` at pads), a tap that reads a token of
  a different segment, or beyond the row ends, contributes zero. Without `segment_ids`, only the
  row ends and the pad zeroing bound the receptive field.
- `canon_activation` other than `none` is applied after the conv, as today.
- Gradients: `dx` (zero at pad positions, because of the zeroing multiply) and `dW` with the
  `(D, 1, k)` shape of `conv.weight`.

## Decided

- **Triton**, not CUDA C++: Triton 3.6 is already installed with torch 2.11, it compiles on the
  cluster's B200s without a build step, and the kernel is memory-bound, so Triton's ceiling is not
  the constraint.
- **Registered as a custom op with its own autograd** (`torch.library` custom/triton op with a
  registered backward and fake/meta implementation), so `torch.compile` treats it as one opaque
  op and does not graph-break or re-trace it.
- **The op boundary is the whole Canon call, not just the conv.** Inductor cannot fuse across an
  opaque op, so everything around the conv moves inside: the op takes the raw `x`,
  `attention_mask`, optional `segment_ids`, the weight, `k`, the output dtype and a residual
  flag, and does the pad zeroing, the per-tap segment masks, fp32 accumulation, the residual add
  (`x + conv(x)`) and the cast to the output dtype. Its backward applies the pad mask to `dx`
  itself. The residual is folded in only when `canon_activation` is `none` (the default and the
  production setting); with another activation the op returns the conv and the call site applies
  activation and residual as today.
- **Masks built inside the kernel** from `attention_mask` and `segment_ids` loaded once per
  block, instead of `k` materialized boolean tensors. `attention_mask` is an arbitrary `(B, T)`
  mask (bool or integer, pads anywhere, not only trailing); `segment_ids` is `(B, T)` integer.
- **Dtypes.** The op takes the output dtype as an explicit argument (the call site reads it from
  autocast as `CanonConv.forward` does today), so no autocast rule needs registering. `dW` is
  returned in the weight's dtype (fp32 parameters), accumulated in fp32.
- **Packing is the primary path.** Every funnel and production config runs with
  `data.pack_sequences: true` and flex attention, so `segment_ids` is always present in training.
  Benchmarks use packed batches; the unpacked path only has to be correct.
- **Weight gradient in one pass**: each program reduces its tile of `x_shifted * dy` for all `k`
  taps into fp32 partial sums, written to a partials buffer and summed afterwards. This keeps the
  result deterministic run to run (no float atomics), which matters for reproducible training.
- **Dispatch**: CUDA tensors with Triton available take the kernel; everything else (CPU, the
  existing CPU test suite) keeps today's shift-add path, which also serves as the test oracle.
  No config option (my call; see below).
- **The reference implementation stays** in `conv.py` as the fallback and the oracle; it is not
  deleted.

## Acceptance

1. **Numerical equivalence (tests).** Forward output and both gradients match the reference path
   across: odd and even `k` (at least 2, 3, 4, 7), channel counts that are and are not multiples
   of the block size (e.g. 768, 2048, 100), packed rows with several segments, rows with trailing
   pads, unpacked rows, `T` not a multiple of the block, fp32 and bf16 inputs, autocast on and
   off. fp32 tolerance tight (max abs diff ~1e-5 relative to output scale); bf16 within normal bf16
   tolerance. Kernel correctness tests must run without a GPU via Triton's interpreter mode
   (`TRITON_INTERPRET=1`) on small shapes, because the development machine has no GPU; the same
   tests at real shapes run on one cluster GPU.
2. **Compile.** A compiled OplmBlock with Canon A–D runs forward and backward with no graph
   breaks and matches the eager result.
3. **Speed.** Re-run the 4-node profile (`profile-submit.sh`, same configs, new commit): Canon's
   elementwise share drops to ≤ ~60 ms/step and the OPLM step to ≤ ~550 ms. A single-GPU
   microbenchmark (per-call fwd+bwd time vs the reference, at the 170M shapes, packed batches) is
   the intermediate check. Equivalence (1, 2, 4) is the hard gate; 550 ms is a target. Nobody has
   split the ~400 ms into the conv itself versus the surrounding mask/residual/cast ops; if the
   step misses 550 ms, report the measured split rather than widening the scope.
4. **Training equivalence.** Baseline: W&B `oplm-mup-4M-nogate` run `b0x13s29`
   (`refine-170M-lr0.004-om1-a0-s42`; eval every 500 steps, 2.5452 / 2.4169 / 2.3641 / 2.3301 at
   500–2,000). Re-run that cell's `run.yaml` (from the no-gate sweep directory) on the new commit,
   same seed, LR and data, and cancel it after 3,000 steps. It should track the old curve: eval
   loss within ~0.003 at matched steps and no change in grad-norm behaviour. Small drift from
   changed summation order is expected; a systematic offset is a bug.

## Who runs what

The development machine has no GPU and cannot reach the cluster. The builder writes the code,
the interpreter-mode tests and the sbatch scripts (1-GPU tests + microbenchmark; the 4-node
profile via the existing `profile-submit.sh`; the 3,000-step equivalence run). Bryan submits them
from `slurm-login-0` (`~/projects/oplm/mup_redo`, container and env as in the existing sbatch
files) and copies results back to `~/data/tmp/`. W&B results the builder reads directly. So the
plan has hand-off points at the GPU gates, not autonomous ones.

## Left to the builder

- Tiling (block sizes over T and D), number of warps, and whether to autotune.
- Whether the Canon-B call sites pass the three Q/K/V streams to one launch (2,304 channels)
  or three, and whether the kernel accepts strided input so the `(B, H, T, d)` →
  `(B, T, D)` copy in `_apply_canon_b` goes away. Worth it if cheap; not required.
- Whether the residual add (`x + conv(x)`, the default `canon_residual=True`) is folded into
  the kernel when the activation is `none`. Same rule.
- Whether the forward saves anything beyond `x`, the weight and the masks for the backward.
- Test file layout, and the 1-GPU sbatch used for the GPU-shape tests and microbenchmark.

## My calls (not discussed with Bryan)

- No config switch between kernel and reference. Escape hatch is pinning the previous commit.
- Deterministic partial-sum reduction for `dW` instead of atomics.
- Interpreter-mode tests as the local correctness gate, given no local GPU.
