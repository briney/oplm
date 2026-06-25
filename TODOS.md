# TODO: Pad-to-multiple batches + configurable compile dynamism + throughput logging

Add a "pad to the nearest multiple of N" collation option and make the
`torch.compile` `dynamic` flag configurable, so we can benchmark whether bucketed
sequence lengths + static-shape compilation improve training throughput on
Blackwell — the regime other groups reported gains in (`dynamic=False`, default
mode). To compare runs we also add steady-state throughput metrics (tokens/sec,
step time, achieved TFLOPs, optional MFU), which the trainer does not log today.

## Why this is expected to help (benchmark hypothesis)

Two distinct mechanisms, in two compile regimes:

1. **Compilation binning / static specialization (the big lever, `dynamic=False`).**
   With `dynamic=False`, `torch.compile` specializes per concrete shape. Pad-to-batch-max
   produces near-continuously-varying lengths → a recompile per batch (pathological).
   Padding to a multiple of N collapses lengths into a few buckets, so a small fixed
   set of *static-shape* graphs is compiled once and reused. Static graphs also beat
   dynamic ones independent of recompiles (more constant-folding, unrolling, better tile
   configs). This is almost certainly where the other group's gains came from.
2. **Tensor-core alignment (smaller, present under `dynamic=True` too).** cuBLAS/cuDNN
   fast paths want leading matmul dims to be multiples of 8 (BF16) / 16 (FP8). "Multiple
   of 8" buys alignment / kernel eligibility — NOT tile-count reduction (tiles are
   64/128/256; the hardware computes `ceil(dim/tile)` tiles regardless).

Today's setup is `dynamic=True` (hardcoded) + `bf16` + SDPA, padding to batch-max. Under
`dynamic=True`, pad-to-multiple captures only mechanism (2). To exercise mechanism (1) the
benchmark needs `dynamic=False`, hence the new compile knob. Optimal N differs by regime:
small (8/16) under `dynamic=True` to minimize wasted padded FLOPs; large (64/128/256)
under `dynamic=False` to keep bucket count low.

## Design invariants

- **Every new knob defaults to today's behavior.** `pad_to_multiple_of=None`
  (pad to batch-max), `compile_dynamic=True` (one dynamic graph). Bare configs and
  existing YAML overlays are byte-for-byte unchanged in behavior.
- **Padding is loss-neutral.** Padded positions are already masked everywhere
  (`attention_mask`, CanonConv `zero_pad_positions`, MLM `ignore_index`). The benefit
  of pad-to-multiple shows up as *higher tokens/sec*, never as a loss/metric change.
- **Strict divisibility, no silent truncation.** When `pad_to_multiple_of` is set we
  require `max_position_embeddings % pad_to_multiple_of == 0` (a config-load error),
  rather than capping padded length and risking position-id overflow.
- **Throughput = steady state.** Exclude the first `throughput_warmup_steps` optimizer
  steps (compile/warmup) and the wall time of eval/checkpoint steps.
- **The benchmark matrix needs no code beyond these knobs** — `compile_mode` is already
  configurable. Matrix: (A) `dynamic=true`+`pad=null` baseline · (B) `dynamic=true`+`pad∈{8,16}`
  · (C) `dynamic=false`+`pad∈{64,128,256}`, optionally × `compile_mode∈{default,max-autotune}`.

---

## Phase 1 — Config knobs + validation (`src/oplm/config.py`, YAML)

- [ ] `DataConfig` (after `weighted_masking`, ~line 261): add
      `pad_to_multiple_of: int | None = None` with a comment that `None` = pad to
      batch-longest, `N` = pad to smallest multiple of `N` ≥ batch-longest.
- [ ] `DataConfig.__post_init__`: if `pad_to_multiple_of is not None`, require it be an
      `int` (reject `bool`) and `>= 1`, else `raise ValueError`.
- [ ] `TrainConfig` (near `compile_mode`, ~line 106): add
      `compile_dynamic: bool | None = True` (passed to `torch.compile(dynamic=...)`:
      `True`=single dynamic graph, `False`=static per-shape, `None`=Dynamo auto),
      `throughput_warmup_steps: int = 50`, and `peak_tflops: float | None = None`
      (device peak for MFU; `None` → log achieved TFLOPs only). Update the
      `# torch.compile` comment block (lines 94-99) to note `dynamic` is now a knob.
- [ ] `TrainConfig.__post_init__`: add `throughput_warmup_steps >= 0` check and, if
      `peak_tflops is not None`, a `peak_tflops > 0` check.
- [ ] `load_config`, right after `cfg.model = OplmModelConfig(**model_dict)` (~line 496):
      add cross-config validation — if `cfg.data.pad_to_multiple_of` is set and
      `cfg.model.max_position_embeddings % cfg.data.pad_to_multiple_of != 0`, raise a
      `ValueError` naming both values (the divisibility invariant). This is the only
      place both subtrees are resolved together.
- [ ] `src/oplm/configs/data/base.yaml`: add `pad_to_multiple_of: null` with a one-line
      comment (default = pad to batch-longest).
- [ ] `src/oplm/configs/train/base.yaml`: under the compile block (lines 63-64) add
      `compile_dynamic: true   # true | false | null(auto) — torch.compile dynamic flag`,
      `throughput_warmup_steps: 50`, and `peak_tflops: null   # set device peak (TFLOPs) to log MFU`.
- [ ] `tests/training/test_config.py`: add cases —
      (a) bare `DataConfig()` / `TrainConfig()` keep the new defaults
      (`pad_to_multiple_of is None`, `compile_dynamic is True`,
      `throughput_warmup_steps == 50`, `peak_tflops is None`);
      (b) `DataConfig(pad_to_multiple_of=0)` and `=-1` raise; `pad_to_multiple_of=True` raises;
      (c) `TrainConfig(throughput_warmup_steps=-1)` and `TrainConfig(peak_tflops=0)` raise;
      (d) divisibility validation — `load_config` with `data.pad_to_multiple_of=128`
      AND an explicit `model.max_position_embeddings=1000` override (all presets
      inherit 1024, which IS divisible, so the non-divisible case must be forced via
      override) raises; the same call with `max_position_embeddings=1024` loads cleanly.
- [ ] Run `python -m pytest tests/training/test_config.py -v`; expect PASS.
- [ ] Commit: `feat(config): add pad_to_multiple_of, compile_dynamic, throughput knobs`.

## Phase 2 — Collator pad-to-multiple (`src/oplm/data/sequence/`)

- [ ] `collate.py` `tokenize_and_pad` (line 68): add keyword param
      `pad_to_multiple_of: int | None = None`; pass it into the tokenizer call
      (line 103): `tokenizer(truncated, padding=True, pad_to_multiple_of=pad_to_multiple_of, return_tensors="pt")`.
      HF tokenizers support this natively; the `masking_weights` branch already keys off
      `out["input_ids"].shape[1]` (line 110), so aligned weights follow automatically.
      Update the docstring's "padded to the batch's longest member" sentence.
- [ ] `collate.py` `MLMCollator.__init__` (line 154): add keyword param
      `pad_to_multiple_of: int | None = None`; store `self._pad_to_multiple_of`.
- [ ] `collate.py` `MLMCollator.__call__` (line 189): thread it through —
      `tokenize_and_pad(batch, self._tokenizer, self._max_length, weights=raw_weights, pad_to_multiple_of=self._pad_to_multiple_of)`.
- [ ] `data/sequence/loaders.py`: in BOTH `MLMCollator(...)` constructions
      (train builder ~line 89, eval builder ~line 127) pass
      `pad_to_multiple_of=cfg.data.pad_to_multiple_of`. (Eval must match so it does not
      introduce fresh static shapes / pollute timing.)
- [ ] `tests/data/sequence/test_collate.py` (or the existing collate test): add cases —
      (a) `pad_to_multiple_of=16` → `input_ids.shape[1] % 16 == 0` and
      `>= unpadded longest`; `attention_mask`, `labels`, and (when weighted)
      `masking_weights` all share that `T`;
      (b) `pad_to_multiple_of=None` reproduces today's batch-max width exactly;
      (c) masking correctness is unchanged (same eligible-position contract on the
      non-pad region; pad columns stay `ignore_index`/`0.0`).
- [ ] Run `python -m pytest tests/data -v`; expect PASS.
- [ ] Commit: `feat(data): pad batches to a configurable multiple of N`.

## Phase 3 — Compile wiring + Dynamo recompile budget (`src/oplm/training/trainer.py`)

- [ ] Line 166: replace `dynamic=True` with `dynamic=cfg.train.compile_dynamic`.
- [ ] Immediately before the `torch.compile` call (after line 160, inside the
      `if cfg.train.compile:` block): when `cfg.train.compile_dynamic is not True`,
      raise the Dynamo recompile budget so bucketed static shapes do not silently fall
      back to eager. Use a local `from torch import _dynamo` (NOT `import torch._dynamo`
      — see the comment at lines 155-157) and:
      - if `cfg.data.pad_to_multiple_of`: `buckets = math.ceil(cfg.model.max_position_embeddings / cfg.data.pad_to_multiple_of)`;
        set `_dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, buckets + 8)`
        and log the expected bucket count.
      - else (`pad_to_multiple_of is None` with non-dynamic compile): log a `logger.warning`
        that batch-max padding under non-dynamic compile yields unbounded shapes and
        will thrash recompiles. (`math` is already imported, line 6.)
- [ ] Update the compile-block `_status`/comment to mention the resolved dynamic mode.
- [ ] `tests/training/test_trainer.py` (or nearest trainer test): add a unit test that
      monkeypatches `torch.compile` to capture kwargs, builds a `Trainer` with
      `compile=True` + `compile_dynamic=False` on a tiny config, and asserts the captured
      `dynamic` kwarg is `False` and `cache_size_limit` was raised. (Introspect the call;
      do not run a real compile — keep it fast.)
- [ ] Run `python -m pytest tests/training/test_trainer.py -v`; expect PASS.
- [ ] Commit: `feat(train): make torch.compile dynamic flag configurable`.

## Phase 4 — Steady-state throughput logging (`src/oplm/training/trainer.py`)

- [ ] Add `import time` (top of file, stdlib group, line ~6).
- [ ] `Trainer.__init__` (after line 184, the FLOP block): initialize timing state —
      `self._step_timer_start: float | None = None`,
      `self._tput_window_tokens = 0`, `self._tput_window_seconds = 0.0`,
      `self._tput_window_steps = 0`.
- [ ] Training loop, right after `self._step_local_tokens = 0` (line 288): accumulate the
      step's compute time into the window, excluding warmup —
      ```python
      now = time.perf_counter()
      if self._step_timer_start is not None and self.global_step > cfg.throughput_warmup_steps:
          self._tput_window_seconds += now - self._step_timer_start
          self._tput_window_tokens += tokens_delta
          self._tput_window_steps += 1
      ```
- [ ] At the END of the opt-step body, after the progress-bar update (after line 317):
      `self._step_timer_start = time.perf_counter()`. Restarting here is what excludes
      the eval (line 295) and checkpoint (line 304) wall time from the next step's dt.
- [ ] `_log_step` (line 399): when `self._tput_window_steps > 0` and
      `self._tput_window_seconds > 0`, add to the `metrics` dict —
      `train/tokens_per_sec = self._tput_window_tokens / self._tput_window_seconds`,
      `train/step_time_s = self._tput_window_seconds / self._tput_window_steps`,
      `train/achieved_tflops = self.flops_per_token * self._tput_window_tokens / self._tput_window_seconds / 1e12`,
      and `train/mfu = achieved_tflops / self.cfg.train.peak_tflops` only when
      `self.cfg.train.peak_tflops` is set. Then reset the three window accumulators to 0.
      (Add a one-line comment noting `flops_per_token` omits attention-score FLOPs, so
      `achieved_tflops`/`mfu` undercount padding waste — `tokens_per_sec` is the headline.)
- [ ] `tests/training/test_trainer.py`: add a unit test that drives the throughput
      accumulators directly (set window fields, call `_log_step` via a captured
      `_log_metrics`) asserting `train/tokens_per_sec`, `train/step_time_s`,
      `train/achieved_tflops` are present and correct, `train/mfu` appears only when
      `peak_tflops` is set, and the window resets after logging.
- [ ] Run `python -m pytest tests/training/test_trainer.py -v`; expect PASS.
- [ ] Commit: `feat(train): log steady-state throughput (tokens/sec, TFLOPs, MFU)`.

## Phase 5 — End-to-end smoke + docs

- [ ] Extend the existing tiny-pilot end-to-end training test (the few-steps + one-eval
      smoke) with a parametrization that sets `data.pad_to_multiple_of=16` (compile left
      OFF — real compilation is too slow for fast CI). Assert: training completes the
      steps with no shape mismatch, the eval pass runs, and a `train/tokens_per_sec`
      metric is emitted (capture logged metrics). Set `throughput_warmup_steps=0` in the
      test so throughput is measured within the short run.
- [ ] (Optional, `@pytest.mark.slow`) A variant with `compile=True` +
      `compile_dynamic=False` + `pad_to_multiple_of=16` over a couple of steps to prove
      the static-bucket path compiles and runs end-to-end on GPU.
- [ ] Update the relevant doc (training/perf section under `docs/`, and `AGENTS.md` if it
      enumerates train knobs): document `data.pad_to_multiple_of`,
      `train.compile_dynamic`, `train.throughput_warmup_steps`, `train.peak_tflops`, the
      divisibility requirement, and the A/B/C benchmark matrix.
- [ ] Run the full fast suite: `python -m pytest -m "not slow" -q`; expect PASS.
- [ ] Commit: `test+docs: e2e pad-to-multiple smoke and knob documentation`.
