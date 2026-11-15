# Batched + distributed Muon (DDP) — design

Card: briney/oplm#53. Date: 2026-11-15.

## Intent

Groundwork for the 6B production run, which trains under DDP. Stock `torch.optim.Muon`
orthogonalizes one matrix at a time, and under DDP every rank repeats the identical work for every
matrix. At 170M that is ~5–7% of the step. At 6B (80 layers, width 2560, FFN 6912, ~640 Muon
matrices) Newton–Schulz cost grows with width cubed while forward/backward grows with width
squared times tokens, so on one GPU it is estimated (not measured) to be the same order as
forward/backward at 16k tokens per GPU.

## Goals

- Each rank orthogonalizes only the matrices it owns; all ranks apply identical updates.
- Same-shape matrices are orthogonalized in one batched call (the win at 170M, single process
  included).
- Same math as `torch.optim.Muon` within bf16 rounding; same optimizer state layout, so existing
  checkpoints resume unchanged and new checkpoints load under stock Muon.

## Non-goals

- HSDP. `train.parallelism=hsdp` keeps stock `torch.optim.Muon`; distributed Muon on sharded
  parameters is a follow-up card.
- A config switch. The new class is Muon under DDP. A/B runs use an old commit's venv.
- Any change to the Newton–Schulz iteration, coefficients, LR adjustment or weight decay.

## Anti-goals

- Training curves that move by more than bf16 noise.
- Ranks drifting apart, or a resume that silently loses momentum.

## Approaches

All three share the core below and differ in how updates travel between ranks.

- **A. Zero-filled buffer + sum all-reduce (chosen).** Each rank writes its owned updates into a
  bf16 buffer that is zero elsewhere; one all-reduce per shape group fills in the rest. Exact
  (adding zeros changes nothing), no padding, any rank count. ~12 GB/step on the wire at 6B, about
  half of the fp32 gradient sync.
- **B. Padded all-gather** (modded-nanogpt). About half the bytes of A at best, but all-gather
  needs equal sizes per rank: 320 attention matrices over 256 ranks pads to 512 slots, which
  erases most of the saving, for more code. **Kept as the fallback** if the 6B profile shows A's
  all-reduce hurting; it replaces only the communication step.
- **C. Distribute within a node only.** Traffic stays intra-node, but each rank still does 1/8 of
  the work instead of 1/256.

Chosen by the user: A, with B as the fallback.

## Design

`class Muon(torch.optim.Muon)` in `src/oplm/training/optim.py`, next to the existing `AdamW`
subclass, overriding only `step`. `build_optimizers` uses it unless `cfg.parallelism == "hsdp"`,
which keeps `torch.optim.Muon`.

`step`, per param group (params with a grad, as stock):

1. **Momentum, every rank, every matrix:** `buf.lerp_(grad, 1 - momentum)`, exactly as stock.
2. **Ownership:** number the group's matrices in order; matrix `i` belongs to rank
   `i % world_size`. World size and rank come from `torch.distributed` when it is initialized with
   more than one rank; otherwise the single process owns everything and no communication happens.
3. **Per shape bucket:** allocate a zero bf16 buffer `(n, rows, cols)`. For owned matrices, form
   the Nesterov input (`grad.lerp(buf, momentum)`, or `buf`), stack, and run batched
   Newton–Schulz: transpose if tall, divide each matrix by its own Frobenius norm (clamped at
   `eps`), then `ns_steps` iterations with `bmm`/`baddbmm`, all in bf16 as stock. Write results
   into the owned slots.
4. **Communicate:** `all_reduce(SUM)` on the bucket buffer when world size > 1.
5. **Apply, every rank:** `param.mul_(1 - lr * weight_decay)`, then
   `param.add_(update, alpha=-adjusted_lr)` using torch's `_adjust_lr`.

State is still one `momentum_buffer` per parameter, so `state_dict` is interchangeable with stock.

## Tests (`tests/training/test_optim.py`)

- Matches `torch.optim.Muon` over a few steps on tall, wide, square and repeated shapes, within a
  bf16 tolerance.
- `state_dict` round-trips with stock Muon in both directions.
- Two ranks equal one: a 2-rank CPU/gloo run (the `_fsdp2_muon_spike_worker.py` pattern) ends with
  the same parameters on both ranks and matches the single-process result.
- `build_optimizers` returns stock Muon for `parallelism=hsdp`.

## Proof on the cluster (closes the card)

Old commit against new, same config and data:

- 170M, 1 node: profile; the `optimizer` phase drops.
- 170M, 3,000 steps, seed 42: eval loss matches within noise at each eval (the Canon-gate recipe).
- 6B, 4 nodes, short profile: `optimizer` phase and step time. If 6B cannot run on 4 nodes, use 3B
  and say so.

## Docs

- `docs/training.md` (optimizer section) is the home: say what the DDP Muon does and that HSDP
  uses stock Muon.
- `docs/methods/`: an evidence record once the cluster numbers exist.

## Left to the builder

- Whether the ownership counter restarts per bucket or runs across buckets (pick whichever
  balances load better at 6B/256 ranks; it must be identical on every rank).
- Whether a very large bucket needs chunking on a single GPU (not needed when distributed).
- How the profiler reports Muon separately from the auxiliary AdamW, if the single `optimizer`
  phase proves too coarse.
