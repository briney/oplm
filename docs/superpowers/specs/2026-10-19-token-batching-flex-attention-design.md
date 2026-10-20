# Token-Budget Batching and FlexAttention for Packed Batches

**Date:** 2026-10-19
**Status:** Design agreed in discussion on 2026-10-19. Two implementation plans follow, one per
pull request: PR A (FlexAttention, §4) and PR B (token batching and its resume, §5–§6).
**Scope:** Let the production series (400M to 12–24B, possibly 170M) train with one global token
count per optimizer step, the same for every model size, node count, data mix and max length, and
with a max length of 2048 from the start of pretraining.

## 1. Why

Batches are counted in sequences: `train.batch_size` per GPU, times GPUs, times accumulation.
Before packing, that count was really sizing padded memory: 8192 sequences × 512 = 4.2M slots.
With packing (PR #39) the padded size no longer matters, and the sequence count is a poor proxy
for what the model learns from:

- The 170M runs train on about **1.38M real tokens per step** (8192 × 168 mean tokens), not 4.2M.
- The planned phase 2 (2k sequences × 2048 max length) would hold about **0.5M real tokens**,
  cutting the real batch about 2.5× at the moment the learning rate starts decaying.
- Tokens per step move with the mix: at fixed sequences, a 5:95 uniref:deepclust batch averages
  about 138 tokens per sequence and a 65:35 batch about 228.

Two further reasons to fix the count in tokens:

- **Hyperparameter transfer.** muP with depth scaling transfers the learning rate across width and
  depth at a fixed batch. One token batch for the whole series lets settings tuned on a small model
  carry to 6–24B, where sweeps are unaffordable.
- **Large jobs need a large global batch.** At 1.4M tokens over 512 GPUs, each GPU gets about
  2.7k tokens, roughly one 2048-token row, far too little work per step. 4M tokens gives 8k.

The second part of the design is attention cost. Packed batches pass a dense `(rows, 1, T, T)`
block-diagonal mask to SDPA (`OplmAttention._sdpa_attention`, `src/oplm/model/attention.py`).
The kernel computes every T×T score of a row and masks afterwards, so attention cost grows with
the **row width**, not with sequence lengths. Attention FLOPs relative to the rest of a layer are
about W/(6d):

| model | d | 512-wide rows | 2048-wide rows | total FLOPs at 2048 vs 512 |
|---|---|---|---|---|
| 170M | 768 | 0.11 | 0.44 | about 1.3× |
| 6B | 2560 | 0.03 | 0.13 | about 1.10× |
| 12B | 3200 | 0.03 | 0.11 | about 1.08× |

A kernel that skips the blocks between different sequences makes width nearly free, which is what
makes a 2048 max length from the start affordable at the small end of the series.

## 2. Goals and success criteria

1. **Constant tokens per step.** A step's real tokens stay within 1–3% of its configured capacity
   (the packing waste), whatever the mix and max length. Measured by `train/batch_fill` (§5.5).
2. **Fixed batch shapes.** Every micro-batch is exactly `rows × W`: no recompiles from changing
   shapes, and peak memory is set by the configuration, not by the data.
3. **FlexAttention is exact to bf16 noise and makes width nearly free.** `docs/perf/packing_check.py`
   passes its `flex` mode at no more than 2× the bf16 floor. At equal tokens, a 21 × 2048 batch
   costs within about 5% of an 84 × 512 batch (measured, §7.2).
4. **Data-exact resume.** A resumed run serves exactly the rows the uninterrupted run would have
   served next, on every GPU and worker, including when the token budget changed at the resume.

## 3. Non-goals

- **Smarter packing.** Batches close at the first sequence that does not fit (§5.3). Lookahead
  packing would save about one more point of fill but would break the contiguous-run property
  that keeps resume simple.
- **Random cropping of long sequences.** Sequences longer than W − 2 residues are still truncated
  to their first residues, as today. At W = 2048 almost none are.
- **Batch-size ramps.** The cursor design (§6) allows changing the token budget at a resume or
  branch, which is enough to run a manual ramp later. No ramp schedule is built.
- **Sweep and Slurm tooling.** `resolve_batch_plan` and the sweep phases stay sequence-based.
- **Loss normalization.** Each micro-batch's loss stays a mean over its own masked tokens. With
  near-equal tokens per micro-batch, per-token weights differ by at most a few percent.
- **Eval.** Eval batches stay padded and sized by `train.batch_size`. Eval at a 2048 max length
  will change eval numbers relative to existing runs; that is a run-configuration decision, not
  part of this work.
- **Fold head.** `StructureDataset` and the fold trainer path are untouched.
- **`oplm stream-replay` for token-mode runs.** It raises a clear error on such checkpoints (§6.6).

## 4. Part 1: FlexAttention for packed batches (PR A)

### 4.1 Config

`model.packed_attention`, an `OplmConfig` field (`src/oplm/model/configuration_oplm.py`), with
values `"sdpa"` (default) and `"flex"`.

- `"sdpa"` keeps today's dense block-diagonal mask, so existing runs and checkpoints are unchanged.
- `"flex"` is opted into by production configs. The default can change once §7 validates it.
- Validation rejects `"flex"` with `attention_dropout > 0`; FlexAttention has no dropout. Dropout
  is 0 in every current config.
- The field serializes into `hf/config.json` like any other config field. It changes no weights.

### 4.2 Block mask

When a forward pass receives `segment_ids` and `packed_attention == "flex"`, `OplmStack` builds
one `BlockMask` before its loop over blocks:

```python
def same_segment(b, h, q, kv):
    return segment_ids[b, q] == segment_ids[b, kv]

block_mask = create_block_mask(same_segment, B=rows, H=None, Q_LEN=T, KV_LEN=T, device=...)
```

- It is passed to every `OplmBlock`, including blocks run under activation checkpointing, and on
  to `OplmAttention`, which calls `flex_attention(q, k, v, block_mask=block_mask,
  scale=self.attn_scale)`.
- The mask is built **once per forward**, not once per layer.
- Padding positions are segment 0 and attend only to each other. No query row is empty, and
  padding positions carry no labels.
- Canon convolutions keep using `segment_ids`, as today.
- Unchanged: the `output_attentions` (manual) path, padded batches without `segment_ids`
  (eval), and the `"sdpa"` setting.

Under `torch.compile` (how training runs), `create_block_mask` and `flex_attention` are traced
into the compiled graph and `flex_attention` runs as a fused kernel. In eager mode both run
unfused, which is fine for CPU tests but slow on GPU. With today's sequence-count packing, rows
and width vary per batch, so a compiled run may recompile; token mode's fixed `(R, W)` shapes are
the intended pairing.

### 4.3 Tests (CPU)

- FlexAttention output and gradients match the dense-mask SDPA path on small packed batches in
  fp32, including rows with several segments and trailing padding.
- The block mask is built once per forward pass regardless of the number of layers.
- A full `OplmForMaskedLM` packed forward and backward with `"flex"` matches `"sdpa"` in fp32.
- Config validation: unknown values and `"flex"` with attention dropout are rejected.

### 4.4 GPU checks (one B200 each, sbatch alongside, under `docs/perf/`)

- `packing_check.py` gains a `flex` mode: compiled packed FlexAttention against padded cuDNN rows,
  judged against the bf16 floor like the existing modes (PASS at no more than 2×).
- New `attention_width_bench.py`: for a given checkpoint or preset, compiled forward and backward
  on equal-token batches (84 × 512 and 21 × 2048, plus any `--shapes` given), with `sdpa` and
  `flex`. It reports synced step time and peak memory per shape. Run at 170M and 1.6B. This
  decides goal 3 and gives the per-preset memory cap for `train.batch_tokens` (§5.1).

## 5. Part 2: the token batcher (PR B)

### 5.1 Config

Two `TrainConfig` fields (`src/oplm/config.py`), both `None` by default:

- `train.tokens_per_step`: global token capacity per optimizer step (for example 4,194,304).
- `train.batch_tokens`: each GPU's micro-batch capacity, `rows × model.max_position_embeddings`
  (for example 8,192 = 4 × 2048).

Setting both turns on **token mode**. Setting only one is an error.

At startup, before the `Accelerator` is built, the trainer reads the world size from
`accelerate.PartialState().num_processes` and computes

```
gradient_accumulation_steps = tokens_per_step / (batch_tokens × world_size)
```

It writes the result into `cfg.train.gradient_accumulation_steps`, so all downstream code is
unchanged and the checkpoint's saved config records the value used.

Config errors, each naming what to change:

- the accumulation is not a whole number (checked at startup, when the world size is known);
- `batch_tokens` is not a multiple of `model.max_position_embeddings`;
- token mode without `data.pack_sequences=true`;
- token mode with `gradient_accumulation_steps` set to anything other than 1;
- token mode with `train.max_epochs` instead of `train.max_steps` (steps per epoch cannot be known
  in advance).

`train.batch_size` still sets the eval batch size.

### 5.2 Data loader

`build_train_dataloader` (`src/oplm/data/sequence/loaders.py`) wraps the dataset it builds today
(`InterleavedDataset`, or the single `ShardedProteinDataset`) in a `TokenBatcher` when token mode
is on, and builds the `DataLoader` with `batch_size=None`. Workers, prefetching and pinning are as
today, and `DeviceDataLoader` is unchanged.

### 5.3 `TokenBatcher`

A new `IterableDataset` in `src/oplm/data/sequence/batching.py`, running inside each data-loading
worker. It holds the wrapped dataset, an `MLMCollator`, `rows` R and width W.

- **Token length** of a sequence: `min(len(sequence), W - 2) + 2` (residues plus `<cls>`/`<eos>`).
- **Placement:** each sequence goes into the first of the R rows with room for it, in stream order
  (first-fit, no sorting).
- **Closing a batch:** at the first sequence that fits in no row, the buffered sequences become a
  batch and that sequence starts the next buffer. Each batch is therefore a contiguous run of the
  worker's stream.
- **Stream end:** the final partial batch is dropped when `data.drop_last` is true (the default)
  and emitted otherwise.
- **Output** per batch:
  - `input_ids`, `attention_mask`, `labels`, `segment_ids`, each exactly `(R, W)`;
  - `num_sequences` (an `int`);
  - `stream_position`: `(worker_id, rows_emitted)`, two plain `int`s, where `rows_emitted` counts
    the wrapped stream's rows in this worker's emitted batches so far this epoch.
- It forwards `set_epoch` and the resume calls (§6.3) to the wrapped dataset.

One placement helper, `first_fit(lengths, rows, width)`, returns each sequence's row or `None`
when one does not fit. The batcher uses it incrementally to decide where a batch closes. The
collator uses it to lay the batch out (§5.4). Same function, same lengths, same order, so the two
cannot disagree.

Simulated waste of this rule, with lognormal stand-ins for the two sources (deepclust70 mean about
131 tokens, uniref70 about 280 with 17.5% truncated at 512):

| rows × width | tokens per GPU | waste at 5:95 | waste at 25:75 |
|---|---|---|---|
| 4 × 2048 | 8,192 | 2.1% | 3.2% |
| 8 × 2048 | 16,384 | 1.3% | 1.8% |
| 21 × 2048 | 43,008 | 0.8% | 1.0% |
| 16 × 512 | 8,192 | 4.8% | 5.7% |
| 84 × 512 | 43,008 | 2.5% | 2.8% |

### 5.4 Collator

`MLMCollator` gains a fixed-shape packing mode (constructor arguments `pack_rows` and the existing
`max_length` as the width):

- Masking is unchanged: per sequence, a fixed 15% of eligible positions by Gumbel-top-k, with the
  80/10/10 replacement, applied before packing.
- `_pack` lays sequences out with `first_fit` in input order and pads to exactly `(pack_rows, W)`;
  unused rows and tails are padding with segment 0.
- It checks that each tokenized length equals the batcher's token length and raises if not. That
  could only fail if the tokenizer stopped emitting one token per residue.
- The existing packing path (fixed sequence count, first-fit decreasing, variable row count) is
  unchanged.

### 5.5 Trainer metrics

- New `train/batch_fill`: real tokens ÷ capacity over the logging window, token mode only.
- Samples per step are summed across GPUs in the same reduction as tokens (`_reduce_step_flags`),
  because GPUs now differ in sequences per batch. `train/mean_seq_len` and the fractional epoch
  become exact again in both modes.
- `_compute_total_steps` returns `max_steps` in token mode (validated present, §5.1).
- `_global_effective_batch_size`, which feeds the Slurm recovery checks, returns `tokens_per_step`
  in token mode.

### 5.6 Tests

- `first_fit`: placement, the close-at-first-miss rule, sequences of exactly W, and truncation of
  longer ones.
- `TokenBatcher`: every batch has shape exactly `(R, W)`; batches are contiguous, non-overlapping
  runs of the stream; `rows_emitted` is correct; the final partial batch follows `drop_last`.
- Masking parity: a packed token batch carries the same per-sequence masks and labels as the
  unpacked collator run on the same sequences with the same RNG state.
- Each config error in §5.1.
- A CPU training run in token mode: the derived accumulation is applied, `train/batch_fill` is
  logged and plausible, and loss decreases.

## 6. Part 3: resume, per-GPU epochs and the data cursor (PR B)

### 6.1 Tracking position

For each batch the training loop consumes, the trainer records the batch's `stream_position` in
`self._worker_rows[worker_id] = rows_emitted`. Prefetched batches not yet consumed are not
recorded, so the record reflects exactly what has trained. On a GPU's epoch rollover the record
resets to zeros. `MLMTask.step` (`src/oplm/training/task.py`) reads only the keys it needs, so
`stream_position` never reaches the model.

### 6.2 Cursor

`DataCursor` (`src/oplm/data/sequence/dataset.py`) gains two optional fields, `None` outside
token mode:

- `rank_epochs: list[int]`, each GPU's epoch at the save;
- `worker_rows: list[list[int]]`, a GPUs × workers table of `rows_emitted`.

The save path is already entered by every GPU at the same step: save triggers are reduced across
ranks. At that point each GPU contributes its epoch and its `_worker_rows` in one
`all_gather_object`, and rank 0 writes the cursor into `trainer_state.json` next to today's
fields. The table is small: 512 GPUs × 8 workers is 4k integers.

### 6.3 Resume

- Each GPU sets `self.epoch = rank_epochs[rank]`, sets the dataset epoch, and calls the new
  `set_resume_rows(worker_rows[rank])` on the top-level dataset.
  - `TokenBatcher` forwards it.
  - `InterleavedDataset` and `ShardedProteinDataset` store the per-worker list (it pickles into
    workers like today's skip). Each worker reads its own entry by `worker_id` and uses it as its
    sample skip, clamped to the stream length as today. `InterleavedDataset` reuses its existing
    arming path (`_arm_source_skip`), which handles resuming inside a later pass over a refilled
    source.
  - `clear_resume_skip` clears both the batch-count skip and the row-count skip.
- The batcher starts with an empty buffer. Every recorded count falls on a batch boundary, so each
  worker's next batch is the one the uninterrupted run would have produced.
- **Layout check.** Which rows a worker serves depends only on the world size, worker count and
  seed. In token mode the cursor's layout check compares those three fields.
  - Changing `tokens_per_step` or `batch_tokens` at a resume or branch keeps the exact position.
  - A sequence-mode cursor maps exactly onto token mode: worker `w` of every rank has served
    `len(range(w, batches_in_epoch, num_workers)) * per_rank_batch` rows, all ranks in the cursor's
    epoch, so it resumes or branches at its exact position. A token-mode cursor in a sequence-mode
    run raises, naming the existing `train.resume_data_position=false` escape hatch (per-worker
    counts are not batch multiples and ranks may be in different epochs).
- `train.branch_from` restores the cursor through the same path, so it inherits all of this.

### 6.4 Per-GPU epochs

GPUs exhaust their share of an epoch at slightly different steps, since sequences per batch vary.
Each GPU rolls over on its own when its DataLoader raises `StopIteration`, as today. The rollover
contains no collective, so nothing waits on another GPU.

- `self.epoch` is per GPU; rank 0's is the one logged and stored as the state's `epoch`.
- `self._batches_in_epoch` is per GPU and no longer identical across ranks.
- **`_SaveDeferral` is disabled in token mode**, logged once at startup. It assumes
  `batches_in_epoch` is identical on every rank; with per-GPU rollovers, ranks could disagree on
  whether to defer, and a save entered by only some ranks would deadlock. Data stays exact without
  it. After a resume, only the order in which a GPU's workers' batches interleave can shift within
  one worker cycle, as can already happen today.

### 6.5 Tests

- **Dataset level:** with `set_resume_rows`, `ShardedProteinDataset` and `InterleavedDataset`
  serve exactly the rows not yet served, for one source, for mixed sources, and when resuming
  inside a refilled pass with `reshuffle_on_refill`.
- **Batcher level:** with 2 workers, serve N batches, record positions, rebuild with
  `set_resume_rows`, and continue; the sequence ids served match an uninterrupted run.
- **Cursor:** token mode accepts a changed token budget; rejects a changed world size, worker count
  or seed; a sequence-mode cursor resumes into token mode at its exact position while a
  token-mode cursor in a sequence-mode run raises; `_SaveDeferral` is off in token mode.
- **Two-GPU CPU run** (gloo, on the harness in `tests/training/test_e2e_dcp.py`) in token mode,
  with streams sized so the two GPUs roll over at different steps: nothing hangs; checkpointing
  mid-epoch, resuming and continuing serves the same sequence ids per GPU as an uninterrupted run.

### 6.6 Tooling

`oplm stream-replay` raises a clear error on checkpoints whose cursor has `worker_rows`.

## 7. Rollout

### 7.1 Order

1. **PR A**, FlexAttention (§4), with its CPU tests and GPU scripts.
2. **GPU validation** of PR A: `packing_check.py --mode flex` and `attention_width_bench.py` at
   170M and 1.6B. If 2048-wide rows are not within about 5% of 512-wide at equal tokens, stop and
   report before PR B.
3. **PR B**, token batching and resume (§5–§6).
4. **The 400M batch check** from the 2026-10-19 discussion runs on PR B: shuffled data, 2048 max
   length, pass-budgeted mix; 4M tokens per step with ÷3, 4M with ÷10, and 2M with ÷3, compared at
   equal tokens.

### 7.2 Docs

`docs/TRAIN.md` and `docs/CONFIG.md` describe `model.packed_attention`, `train.tokens_per_step`,
`train.batch_tokens`, `train/batch_fill`, per-GPU epochs and the resume rules. The new GPU scripts
carry usage docstrings like the existing ones in `docs/perf/`.

## 8. Risks

| Risk | Mitigation |
|---|---|
| `create_block_mask` breaks or recompiles under `torch.compile` | Caught by the compiled `packing_check` and bench runs (§4.4). Fallback: build the mask outside the compiled region (in `MLMTask.step`) and pass it in. |
| FlexAttention is slower than expected at head dimension 64 on B200 | Measured by the bench before PR B starts (§7.1 step 2). |
| A save path entered by rank 0 alone would deadlock on the cursor gather | The two-GPU test (§6.5) covers periodic, drain and final saves; the plan verifies every save path is rank-synchronized. |
| Equal tokens do not mean equal attention cost: a GPU with several long sequences does more work | Step-time spread across ranks is visible in the profiler; long sequences are a few percent of rows. |
| A worker's last partial batch is dropped each epoch | At most one batch per worker per epoch, negligible. |
| Masked-token counts differ slightly between micro-batches | Accepted (§3); fill is 97–99%, so counts are within a few percent. |
