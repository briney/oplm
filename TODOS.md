# TODOS: Fix selective activation checkpointing under DDP + `torch.compile`

**Branch:** `feature/selective-activation-checkpointing`
**Scope:** Trainer-only change. No model/SAC code changes — the SAC implementation
in `src/oplm/model/transformer.py` is correct (verified on single GPU).

## Problem (diagnosed, reproduced on 8×B200 / torch 2.10)

With `gradient_checkpointing_mode="selective"` + `train.compile=true` on multi-GPU
(DDP), selective checkpointing **silently collapses to `full` recompute**: identical
peak memory and step time to `mode="full"`, with none of its intended ~1.2× /
partial-memory tradeoff. Single-GPU is unaffected and works as designed.

**Measured (bs=64, seq=1024):**

| config | selective step | selective peak |
|---|---|---|
| 1 GPU | 554 ms (1.24×) | 54 GB ✓ |
| 8-GPU DDP, default | 678 ms (== full) | 14.5 GB ✗ |
| 8-GPU DDP, `optimize_ddp=False` | 566 ms (1.20×) | 55.6 GB ✓ |

**Root cause:** `torch._dynamo.config.optimize_ddp="ddp_optimizer"` (default) splits
the compiled graph into ~48 bucket-sized subgraphs (25 MB default) to overlap
allreduce with backward. This fragments each `OplmBlock`'s activation-checkpoint
higher-order op across subgraphs, so AOT autograd's min-cut partitioner can't honor
the SAC `MUST_SAVE` set → every op recomputed → `selective` == `full`. No DDP on a
single GPU, so `optimize_ddp` never engages and SAC works.

---

## Fix (recommended): disable DDPOptimizer graph-splitting for selective + compile

**File:** `src/oplm/training/trainer.py`, immediately before the `torch.compile`
call (currently ~line 127, inside the `if cfg.train.compile:` block, after
`accelerator.prepare`).

Gate it so only the affected configuration pays the cost (`full`/`none` keep
comm/compute overlap):

```python
if cfg.train.compile:
    # Selective activation checkpointing (SAC) is incompatible with the default
    # DDPOptimizer: graph-splitting at gradient-bucket boundaries fragments the
    # per-block activation-checkpoint HOP, so AOT's partitioner drops the SAC
    # MUST_SAVE policy and `selective` silently degrades to full recompute.
    # Disabling graph-splitting keeps SAC intact; the only cost is lost
    # comm/compute overlap (~2-3 ms allreduce on a ~500 ms step for a 300M model
    # on NVLink — negligible vs. the ~20% recompute SAC saves back).
    if getattr(cfg.model, "gradient_checkpointing_mode", "full") == "selective":
        import torch._dynamo
        torch._dynamo.config.optimize_ddp = False
    _status("[dim]Compiling model (torch.compile)...[/dim]")
    self.model = cast(
        "nn.Module",
        torch.compile(self.model, dynamic=True, mode=cfg.train.compile_mode),
    )
```

**Notes:**
- `optimize_ddp` is a process-global dynamo config flag, set once before compile.
- Numerically transparent — only affects graph partitioning for comm overlap, not
  math. Existing SAC transparency tests (`test_gradient_checkpointing.py`,
  `test_e2e_gradckpt.py`) remain valid.
- Only meaningful under DDP+compile; harmless to set on single GPU.
- Gate on `mode == "selective"` only — `full`/`none` should keep the default
  `ddp_optimizer` so they retain comm/compute overlap.

### Alternative (defer): `optimize_ddp="python_reducer"`
Preserves comm/compute overlap *and* SAC (no graph split), but requires wrapping
the backward in `torch._dynamo.compiled_autograd` — non-trivial with
`accelerator.backward()`. Revisit only if the lost overlap proves material at
larger model sizes. For 300M on NVLink it does not.

---

## Tests / validation

- [ ] Guard test: with `compile=true` + `mode="selective"`, assert
      `torch._dynamo.config.optimize_ddp is False` after `Trainer.__init__`
      (CPU-safe; check the flag, no real compile needed).
- [ ] Multi-GPU peak-memory check (slow, CUDA, ≥2 ranks): selective peak must be
      **strictly greater** than full peak (proves SAC saves matmul/SDPA outputs
      rather than collapsing to full). Mirror the single-process ordering test
      `test_sac_peak_memory_between_none_and_full` for the DDP path.
- [ ] Confirm `full`/`none` runs are unaffected (`optimize_ddp` left at default).
- [ ] Re-run the repro: 8-GPU selective → ~1.2× / ~55 GB (not 14.5 GB).

## Docs

- [ ] `docs/TRAIN.md`: note that `selective` + `compile` on multi-GPU disables
      DDPOptimizer automatically (in the trainer) and the negligible overlap
      tradeoff.
- [ ] Cross-reference in the `gradient_checkpointing_mode` config docs
      (`docs/CONFIG.md` / `base.yaml` comment).

## Lint / type

- [ ] `ruff check src/ && ruff format src/ && ty check src/` clean.
