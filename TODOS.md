# TODOS: Native FSDP2 + FP8 Training via torchao

**Branch:** `native-fsdp2-and-fp8`

## Background

The current training stack uses HuggingFace Accelerate as a DDP wrapper with
optional DeepSpeed via env-var gate. On Blackwell GPUs (the exclusive production
target) this leaves two remaining performance gaps (one has already been closed):

1. ~~**No compilation**~~ — `torch.compile` is now **implemented** as opt-in
   (`train.compile=true`, PR #3). SDPA dispatches to FA4 more reliably under
   graph-mode compilation; fused ops reduce kernel-launch overhead.
2. **DDP instead of FSDP2** — at 2.5B–12.7B parameter scales, DDP replicates full
   model state on every GPU. FSDP2 (`fully_shard`) shards weights and optimizer state.
3. **No FP8** — Blackwell's FP8 tensor cores roughly double effective throughput
   vs BF16. `torchao`'s rowwise-scaling recipe preserves accuracy at near-BF16 quality.

**Key constraints:**
- BF16 is the default; FP8 is opt-in via `train.precision=fp8`
- `torch.compile` is opt-in via `train.compile=true`  
- Model code (`src/oplm/model/`) is **not touched** — torchao replaces `nn.Linear`
  layers in-place; the forward API is unchanged
- DeepSpeed is removed entirely
- Accelerate is removed from the training stack (model, trainer, checkpointing);
  it may remain as a dev dep if needed elsewhere

**Initialization order (fixed, non-negotiable):**
```
1. dist.init_process_group("nccl")
2. Build model on CPU
3. [FP8 only] torchao.convert_to_float8_training(model)   ← BEFORE fully_shard
4. fully_shard(model, ...)                                  ← per-block, then root
5. [compile] torch.compile(model, dynamic=True)             ← AFTER fully_shard
6. build_optimizers(model, ...)                             ← AFTER fully_shard
```

---

## Phase 1: Dependency and Config Schema

- [x] **1.1** Create branch `native-fsdp2-and-fp8` from `main`

- [x] **1.2** Update `pyproject.toml`:
  - Add `torchao>=0.7` to `[project.dependencies]` (core dep, not extras)
  - In `[project.optional-dependencies]` under `train`: remove `accelerate>=0.30`
    and `deepspeed>=0.14`; add a comment indicating accelerate is no longer required
  - Torch constraint `>=2.11,<2.12` is already correct — `fully_shard` composable API
    and `torch.optim.Muon` are 2.11 features; leave it untouched

- [x] **1.3** Add new fields to `TrainConfig` in `src/oplm/config.py`:

  > **Note:** `compile: bool = False` and `compile_mode: str = "default"` were already
  > added on `main` (PR #3). Still needed: `precision`, `fsdp_sharding_strategy`, and
  > their validators.

  ```python
  # Precision — "bf16" for standard mixed-precision, "fp8" for torchao FP8 training
  # (FP8 requires sm90+ / Blackwell hardware)
  precision: str = "bf16"

  # FSDP2 sharding strategy:
  #   "full"   — shard weights + grads + optim state across all ranks (default)
  #   "hybrid" — shard within a node, replicate across nodes (for NVLink-rich clusters)
  #   "none"   — no sharding (single-GPU or debugging)
  fsdp_sharding_strategy: str = "full"
  ```

  Add `_VALID_PRECISION = ("bf16", "fp8")` and `_VALID_FSDP_STRATEGIES = ("full", "hybrid", "none")`.
  Validate both in `__post_init__`. If `precision == "fp8"` and CUDA device capability
  is known at config time, emit a warning (not an error — config loading happens before
  CUDA init in some test contexts).

  Keep the existing `mixed_precision` field but mark it deprecated in its docstring
  and make `__post_init__` raise `ValueError` if both `mixed_precision != "bf16"` and
  `precision != "bf16"` are set simultaneously (prevents ambiguity). In the new Trainer,
  only `precision` is consulted.

- [x] **1.4** Update `src/oplm/configs/train/base.yaml` to include the new fields:

  > **Note:** `compile: false` and `compile_mode: default` already present on `main`
  > (PR #3). Still needed: `precision: bf16` and `fsdp_sharding_strategy: full`.

  ```yaml
  train:
    precision: bf16
    fsdp_sharding_strategy: full
  ```

- [x] **1.5** Add `@pytest.mark.blackwell` to `pyproject.toml` markers list:

  ```toml
  [tool.pytest.ini_options]
  markers = [
      "slow: ...",
      "blackwell: requires sm90+ GPU (Blackwell / H100+); skip on other hardware",
  ]
  ```

---

## Phase 2: New `precision.py` Module

Create `src/oplm/training/precision.py`. This module is standalone — no Accelerate
dependency, no Trainer dependency. It encapsulates all torchao FP8 logic.

- [x] **2.1** Implement `is_fp8_supported() -> bool`:

  ```python
  def is_fp8_supported() -> bool:
      """Return True if the current CUDA device supports FP8 (sm90+)."""
      if not torch.cuda.is_available():
          return False
      major, _ = torch.cuda.get_device_capability()
      return major >= 9
  ```

- [x] **2.2** Implement `apply_fp8_training(model: nn.Module) -> None`:

  ```python
  def apply_fp8_training(model: nn.Module) -> None:
      """Convert all nn.Linear layers to Float8Linear with rowwise scaling.

      Must be called BEFORE fully_shard(). Norms, RoPE, Conv1d, and embedding
      tables are not affected — torchao's filter function skips non-Linear modules.
      """
      from torchao.float8 import Float8LinearConfig, convert_to_float8_training
      config = Float8LinearConfig.from_recipe_name("rowwise")
      convert_to_float8_training(
          model,
          config=config,
          module_filter_fn=lambda m, fqn: isinstance(m, nn.Linear),
      )
  ```

- [x] **2.3** Implement `sync_fp8_history(model: nn.Module) -> None`:

  > **API change (torchao 0.17, satisfies the `>=0.7` pin):**
  > `sync_float8_amax_and_scale_history` was a *delayed-scaling* primitive and has
  > been **removed**. The `rowwise` recipe is *dynamic* scaling — scales are derived
  > from current tensor values each forward, so there is no cross-iteration amax
  > history to sync. The modern FSDP2 equivalent is
  > `precompute_float8_dynamic_scale_for_fsdp`, which precomputes each sharded
  > weight's scale in one all-reduce that overlaps the next all-gather.
  >
  > **Downstream impact on Phase 4 (4.13 / 4.16):** the FP8 sync call **moves to
  > AFTER `optimizer.step()`**, not before — the weights must be updated first.
  > It no-ops when there are no FSDP2-sharded `Float8Linear` weights (including the
  > `fsdp_sharding_strategy="none"` debug path, where weights are not `DTensor`),
  > so it stays safe to call unconditionally when `precision == "fp8"`.

  ```python
  def sync_fp8_history(model: nn.Module) -> None:
      """Precompute dynamic FP8 weight scales for FSDP2.

      Call AFTER optimizer.step(). No-op when the model has no FSDP2-sharded
      Float8Linear weights, so safe to call unconditionally if precision=="fp8".
      """
      from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
      precompute_float8_dynamic_scale_for_fsdp(model)
  ```

- [x] **2.4** Export all three from `src/oplm/training/__init__.py`.

---

## Phase 3: Rewrite `checkpoint.py`

Replace Accelerate's `save_state` / `load_state` with PyTorch Distributed
Checkpointing (DCP). The HF export (`save_pretrained`) path is preserved.

- [x] **3.1** Replace `save_checkpoint` signature and body:

  ```python
  def save_checkpoint(
      model: nn.Module,
      optimizer: torch.optim.Optimizer,   # primary optimizer
      cfg: OplmConfig,
      output_dir: str,
      global_step: int,
      epoch: int,
      samples_seen: int,
      tokens_seen: int,
      save_total_limit: int = 3,
  ) -> None:
  ```

  Body:
  ```python
  import torch.distributed as dist
  import torch.distributed.checkpoint as dcp
  from torch.distributed.checkpoint.state_dict import (
      StateDictOptions, get_state_dict, set_state_dict,
  )

  ckpt_path = Path(output_dir) / f"checkpoint-{global_step}"
  model_sd, optim_sd = get_state_dict(model, optimizer)
  dcp.save({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(ckpt_path))

  if dist.get_rank() == 0:
      # trainer_state.json — unchanged
      # config.yaml — unchanged (use OmegaConf.to_yaml)
      
      # HF export: gather full model state dict on rank 0, then save_pretrained
      full_sd, _ = get_state_dict(
          model, optimizer,
          options=StateDictOptions(full_state_dict=True, cpu_offload=True),
      )
      hf_dir = ckpt_path / "hf"
      _save_hf_export(model, full_sd, cfg, hf_dir)

      _rotate_checkpoints(Path(output_dir), save_total_limit)
  dist.barrier()
  ```

  `_save_hf_export` temporarily loads `full_sd` into a CPU-resident unwrapped model
  copy via `model.load_state_dict(full_sd, strict=True)` then calls
  `model.save_pretrained(hf_dir)`. This avoids needing the Accelerate
  `unwrap_model` utility.

- [x] **3.2** Replace `load_checkpoint` signature and body:

  ```python
  def load_checkpoint(
      model: nn.Module,
      optimizer: torch.optim.Optimizer,
      checkpoint_dir: str,
  ) -> dict[str, Any]:
  ```

  Body:
  ```python
  model_sd, optim_sd = get_state_dict(model, optimizer)
  dcp.load({"model": model_sd, "optimizer": optim_sd}, checkpoint_id=str(ckpt_path))
  set_state_dict(model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd)
  
  # Read trainer_state.json from rank 0 and broadcast
  state_json = ...  # rank 0 reads; dist.broadcast_object_list to all ranks
  return state_json
  ```

- [x] **3.3** Remove the `Accelerator` type annotation and import from the file.

- [x] **3.4** `_rotate_checkpoints` helper is unchanged; keep it.

---

## Phase 4: Rewrite `trainer.py`

This is the largest change. The new Trainer replaces every Accelerate touchpoint
with native PyTorch distributed primitives.

### 4a. `__init__` — process group, model, FSDP2, compile, optimizer

- [x] **4.1** Remove all `accelerate` imports. Add:

  ```python
  import os
  import torch.distributed as dist
  from torch.distributed.device_mesh import init_device_mesh
  from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
  ```

- [x] **4.2** Replace Accelerator construction with process-group init:

  ```python
  if not dist.is_initialized():
      dist.init_process_group(backend="nccl")
  self.rank = dist.get_rank()
  self.world_size = dist.get_world_size()
  self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
  torch.cuda.set_device(self.local_rank)
  self.device = torch.device("cuda", self.local_rank)
  self.is_main = self.rank == 0
  ```

- [x] **4.3** Seeding: replace `set_seed(cfg.train.seed)` with
  `torch.manual_seed(cfg.train.seed + self.rank)`. Each rank gets a different
  data seed; model init happens before sharding so all ranks start from the same
  weights.

- [x] **4.4** wandb init (rank 0 only):

  ```python
  if self.is_main and cfg.train.wandb_enabled:
      import wandb
      wandb.init(
          project=cfg.train.wandb_project,
          name=cfg.train.wandb_run_name,
          config=_config_to_flat_dict(cfg),
      )
  ```

- [x] **4.5** Model construction and FP8 conversion (before FSDP2):

  ```python
  model = OplmForMaskedLM(cfg.model)  # CPU init
  if gradient_checkpointing:
      model.gradient_checkpointing_enable()
  if cfg.train.precision == "fp8":
      from oplm.training.precision import apply_fp8_training
      apply_fp8_training(model)
  ```

- [x] **4.6** FSDP2 sharding. Per-block sharding is critical — it enables
  activation offloading and keeps all-gather granularity manageable:

  ```python
  from torch.distributed._composable.fsdp import CPUOffloadPolicy

  mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("dp",))
  
  # BF16 param dtype, FP32 gradient reduction (standard for stability)
  mp_policy = MixedPrecisionPolicy(
      param_dtype=torch.bfloat16,
      reduce_dtype=torch.float32,
  )
  
  fsdp_kwargs = dict(mesh=mesh, mp_policy=mp_policy)
  
  # Shard each transformer block independently
  for block in model.model.encoder.blocks:
      fully_shard(block, **fsdp_kwargs)
  
  # Shard the root model (covers embedding, final norm, MLM head)
  fully_shard(model, **fsdp_kwargs)
  ```

  For `fsdp_sharding_strategy == "none"` (single GPU / debug), skip `fully_shard`
  entirely and just move the model to device: `model = model.to(self.device)`.

  For `fsdp_sharding_strategy == "hybrid"`, use a 2D mesh:
  ```python
  mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                          mesh_dim_names=("replicate", "shard"))
  ```
  and pass `mesh["shard"]` to per-block `fully_shard`, `mesh` to the root shard.
  Document this as a follow-on — for now implementing `"full"` and `"none"` is sufficient.

- [x] **4.7** `torch.compile` (after FSDP2):

  > **Reference:** already implemented in `src/oplm/training/trainer.py:127–133`
  > (Accelerate trainer). Port the same pattern here, after task 4.6.

  ```python
  if cfg.train.compile:
      model = torch.compile(model, dynamic=True, mode=cfg.train.compile_mode)
      # dynamic=True is required — protein sequences have variable lengths per batch.
      # Without it, each unique (batch_size, seq_len) combination triggers a recompile.
  ```

- [x] **4.8** Optimizer construction after FSDP2. The current `build_optimizers(model, cfg.train)`
  call is unchanged in signature; it must be called after `fully_shard` so
  `model.parameters()` yields FSDP2-managed DTensor parameters. AdamW handles
  DTensors natively. Muon's NS orthogonalization all-gathers the full weight
  matrix before computing the update, so it is also correct with FSDP2 sharded params.

- [x] **4.9** Dataloader: `build_train_dataloader(cfg)` is unchanged. Remove
  `accelerator.prepare()` wrapping. The dataloader's `DistributedSampler` (or
  dataset's built-in sharding for iterable datasets) handles rank-aware batching.

- [x] **4.10** Replace `_compute_total_steps` reference to `self.accelerator.num_processes`
  with `self.world_size`.

- [x] **4.11** Replace `_global_effective_batch_size` reference to
  `self.accelerator.num_processes` with `self.world_size`.

### 4b. `train()` — training loop

- [x] **4.12** Rich progress bar: replace `self.accelerator.is_main_process` with
  `self.is_main`.

- [x] **4.13** Replace the `with self.accelerator.accumulate(self.model):` block with
  manual gradient accumulation. Track micro-step index:

  ```python
  micro_step = 0  # init outside loop
  
  # Inside the loop, after batch retrieval:
  is_last_micro_step = (micro_step + 1) % cfg.gradient_accumulation_steps == 0
  micro_step += 1
  
  # Forward + backward
  if not is_last_micro_step:
      # no_sync skips all-reduce; grads accumulate locally
      with self.model.no_sync():
          loss = self.model(...).loss / cfg.gradient_accumulation_steps
          loss.backward()
  else:
      loss = self.model(...).loss / cfg.gradient_accumulation_steps
      loss.backward()
  ```

  > **CORRECTION (see Phase 2.3):** the FP8 sync does **not** go here. With torchao's
  > dynamic `rowwise` recipe, `sync_fp8_history` wraps
  > `precompute_float8_dynamic_scale_for_fsdp`, which must run **after**
  > `optimizer.step()` (weights must be updated first). Wire it into task **4.16**,
  > not here.

  Note: `model.no_sync()` on an FSDP2 module skips the reduce-scatter of gradients
  during backward. This is the correct accumulation pattern for FSDP2.

  Note: divide loss by `gradient_accumulation_steps` so the effective gradient
  magnitude is the same regardless of accumulation depth.

- [x] **4.14** Replace `self.accelerator.backward(loss)` with `loss.backward()`.

- [x] **4.15** Gradient clipping: replace `self.accelerator.clip_grad_norm_()` with:

  ```python
  if cfg.max_grad_norm > 0 and is_last_micro_step:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
  ```

- [x] **4.16** Optimizer and scheduler steps: move inside the `is_last_micro_step`
  block (mirrors the current `accelerator.sync_gradients` guard):

  ```python
  if is_last_micro_step:
      for optimizer in self.optimizers:
          optimizer.step()
          optimizer.zero_grad()
      # FP8 (dynamic rowwise): precompute next-iter weight scales AFTER the step.
      # No-op unless the model has FSDP2-sharded Float8Linear weights. See Phase 2.3.
      if self.cfg.train.precision == "fp8":
          from oplm.training.precision import sync_fp8_history
          sync_fp8_history(self.model)
      for scheduler in self.schedulers:
          scheduler.step()
      self.global_step += 1
      micro_step = 0  # reset for next optimizer step
  ```

- [x] **4.17** Token reduction: replace `self.accelerator.reduce(tokens_tensor, reduction="sum")`
  with:

  ```python
  tokens_tensor = torch.tensor(
      self._step_local_tokens, device=self.device, dtype=torch.long
  )
  dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
  tokens_delta = int(tokens_tensor.item())
  ```

- [x] **4.18** Metric logging: replace `self.accelerator.log(metrics, step=...)` with:

  ```python
  def _log_metrics(self, metrics: dict[str, float]) -> None:
      if self.is_main and self.cfg.train.wandb_enabled:
          import wandb
          wandb.log(metrics, step=self.global_step)
      if not self.is_main:
          return
      for callback in self.callbacks:
          callback.on_log(self, dict(metrics), self.global_step)
  ```

- [x] **4.19** Replace `self.accelerator.end_training()` with:

  ```python
  if self.cfg.train.wandb_enabled and self.is_main:
      import wandb
      wandb.finish()
  dist.destroy_process_group()
  ```

- [x] **4.20** Replace all remaining `self.accelerator.is_main_process` with `self.is_main`,
  `self.accelerator.wait_for_everyone()` with `dist.barrier()`, and
  `self.accelerator.device` with `self.device`.

### 4c. Checkpoint integration

- [x] **4.21** Update `_save_checkpoint` to call the new DCP-based `save_checkpoint`:

  ```python
  def _save_checkpoint(self) -> None:
      from oplm.training.checkpoint import save_checkpoint
      checkpoint_dir = Path(self.cfg.train.output_dir) / f"checkpoint-{self.global_step}"
      save_checkpoint(
          model=self.model,
          optimizer=self.optimizer,   # primary optimizer
          cfg=self.cfg,
          output_dir=self.cfg.train.output_dir,
          global_step=self.global_step,
          epoch=self.epoch,
          samples_seen=self._samples_seen,
          tokens_seen=self.tokens_seen,
          save_total_limit=self.cfg.train.save_total_limit,
      )
      self._emit_checkpoint_saved(checkpoint_dir)
  ```

- [x] **4.22** Update `_resume_from_checkpoint` to call the new `load_checkpoint(model, optimizer, ...)`.

### 4d. Evaluator interface update

- [x] **4.23** The current `_run_eval` passes `self.accelerator` to
  `evaluator.run_due(ctx, model, accelerator)`. Replace with a thin context object:

  ```python
  @dataclass
  class DistContext:
      rank: int
      world_size: int
      device: torch.device
      is_main: bool
  ```

  Update `_run_eval` to pass `DistContext(self.rank, self.world_size, self.device, self.is_main)`.

  Update `src/oplm/eval/evaluator.py`: replace `accelerator` parameter in
  `run_due` with `dist_ctx: DistContext`. Replace all `accelerator.is_main_process`,
  `accelerator.device`, etc. with `dist_ctx.is_main`, `dist_ctx.device`, etc.
  Do not import Accelerate anywhere in `eval/`.

---

## Phase 5: Simplify `train.py`

- [x] **5.1** Remove the DeepSpeed env-var gate entirely:
  - Delete `_DEEPSPEED_OPT_IN_ENV`, `_DEEPSPEED_ENV_VARS`, `_DEEPSPEED_LOGGER_NAME`
  - Delete `_env_flag_is_enabled` and `_set_deepspeed_logger_enabled`
  - Remove the deepspeed branch from `_bootstrap_training_environment`

- [x] **5.2** Keep `_ensure_triton_cache_dir` — still needed for `torch.compile`'s
  Triton backend. Rename `_bootstrap_training_environment` to `_setup_triton_cache`
  to reflect reduced scope.

- [x] **5.3** Update module docstring:

  ```python
  """Training entry point.

  Single GPU:   torchrun --nproc_per_node=1 -m oplm.train --config configs/my_run.yaml
  Multi-GPU:    torchrun --nproc_per_node=8 -m oplm.train --config configs/my_run.yaml
  FP8:          torchrun --nproc_per_node=8 -m oplm.train --config configs/my_run.yaml train.precision=fp8
  With compile: torchrun --nproc_per_node=8 -m oplm.train --config configs/my_run.yaml train.compile=true
  """
  ```

---

## Phase 6: Tests

- [x] **6.1** Add `blackwell` marker skip logic to `tests/conftest.py`:

  ```python
  import torch
  import pytest

  def pytest_runtest_setup(item: pytest.Item) -> None:
      if "blackwell" in item.keywords:
          if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
              pytest.skip("Test requires sm90+ (Blackwell / H100+) GPU")
  ```

- [x] **6.2** Create `tests/training/test_precision.py`:

  - `test_apply_fp8_skips_non_linear`: construct a small `OplmForMaskedLM`, call
    `apply_fp8_training`, verify all `nn.Linear` are replaced with `Float8Linear`
    and all `nn.Conv1d` and norm modules are unchanged. Mark `@pytest.mark.blackwell`.
  - `test_bf16_path_untouched`: verify that with `precision="bf16"`, no conversion
    happens and the model's Linear modules remain standard `nn.Linear`.
  - `test_is_fp8_supported_returns_bool`: basic type check, no GPU required.

- [x] **6.3** Update `tests/test_train_entrypoint.py`:
  - Remove patching of `accelerate.Accelerator`.
  - Patch `torch.distributed.init_process_group` and `torch.distributed.destroy_process_group`
    for unit-test isolation.
  - Patch `torch.distributed.get_rank` → 0, `torch.distributed.get_world_size` → 1.
  - Verify `main()` calls `Trainer(cfg).train()` with a mocked Trainer.

- [x] **6.4** Update `tests/test_e2e_lifecycle.py` and any existing E2E tests that
  construct a `Trainer` directly:
  - Update `Trainer(cfg)` construction expectations (no Accelerator).
  - Run 3–5 training steps with `fsdp_sharding_strategy="none"` (single-GPU, no
    actual sharding) so the test works locally.
  - Use a tiny `small` preset model and a synthetic in-memory dataset.

- [x] **6.5** Create `tests/training/test_e2e_fsdp2.py` (`@pytest.mark.slow`):

  ```python
  @pytest.mark.slow
  def test_single_rank_bf16_fsdp2():
      """Single-rank FSDP2 + BF16: a few steps complete without error."""
      cfg = make_test_config(precision="bf16", fsdp_sharding_strategy="full", compile=False)
      trainer = Trainer(cfg)
      trainer.train()

  @pytest.mark.slow
  def test_single_rank_bf16_fsdp2_compile():
      """Single-rank FSDP2 + BF16 + compile: verify compile doesn't break training."""
      cfg = make_test_config(precision="bf16", fsdp_sharding_strategy="full", compile=True)
      trainer = Trainer(cfg)
      trainer.train()

  @pytest.mark.slow
  @pytest.mark.blackwell
  def test_single_rank_fp8_fsdp2():
      """Single-rank FSDP2 + FP8: verify loss is finite and FP8 sync runs."""
      cfg = make_test_config(precision="fp8", fsdp_sharding_strategy="full", compile=True)
      trainer = Trainer(cfg)
      trainer.train()
  ```

- [x] **6.6** Verify `pytest -m "not slow and not blackwell"` passes cleanly on
  local hardware (no Blackwell GPU required).

---

## Phase 7: Documentation and Cleanup

- [x] **7.1** Update `AGENTS.md`:
  - Replace `accelerate launch -m oplm.train` with `torchrun --nproc_per_node=N`
    in the "Run distributed" example
  - Add `torchao>=0.7` to any dependency notes
  - Remove DeepSpeed references

- [x] **7.2** Update `docs/TRAIN.md` (or create it if not present):
  - New "Precision" section: explain `train.precision=bf16` (default) vs `train.precision=fp8`
    (Blackwell only)
  - New "Compilation" section: explain `train.compile=true` and the first-step
    compile latency tradeoff
  - Update launch command examples to use `torchrun`
  - Document checkpoint format change (DCP vs Accelerate state)
  - Note: existing Accelerate-format checkpoints are not compatible with the new
    DCP loader; provide a one-time migration note

- [x] **7.3** Update `README.md` training section: replace `accelerate launch` with
  `torchrun` example.

- [x] **7.4** Remove `deepspeed` from any CI config, workflow files under `.github/`,
  or environment setup scripts.

- [x] **7.5** Run `ruff check src/` and `ruff format src/` — fix any lint issues
  introduced by the changes.

- [x] **7.6** Run `ty check src/` — fix any new type errors. Use `# ty: ignore[<rule>]`
  sparingly and only with an explanation comment.

---

## Phase 8: End-to-End Verification Checklist

Run these in order before merging:

- [x] **8.1** `pytest -m "not slow and not blackwell"` — all pass
- [x] **8.2** `pytest -m "slow and not blackwell"` — all pass (single-rank FSDP2 BF16)
- [x] **8.3** Single-GPU smoke test (run with `--preset 50M`; see preset-rename note below):
  ```bash
  torchrun --nproc_per_node=1 -m oplm.train --preset small \
    train.max_steps=5 train.precision=bf16 train.save_every=5
  ```
  Verify: checkpoint directory created, `hf/` subdirectory loadable via
  `OplmForMaskedLM.from_pretrained(...)`.
- [x] **8.4** Single-GPU compile smoke test (run with `--preset 50M`):
  ```bash
  torchrun --nproc_per_node=1 -m oplm.train --preset small \
    train.max_steps=5 train.precision=bf16 train.compile=true
  ```
  Verify: no graph-break errors, loss decreases.
- [x] **8.5** Resume from checkpoint (run with `--preset 50M`):
  ```bash
  torchrun --nproc_per_node=1 -m oplm.train --preset small \
    train.max_steps=10 train.save_every=5 train.resume_from=outputs/checkpoint-5
  ```
  Verify: training resumes at step 5, global_step reaches 10.
- [ ] **8.6** *(On Blackwell node)* Multi-GPU FP8:
  ```bash
  torchrun --nproc_per_node=8 -m oplm.train --preset base \
    train.max_steps=20 train.precision=fp8 train.compile=true
  ```
  Verify: loss is finite at every logged step, no NaN in gradients, checkpoint saves.
- [ ] **8.7** *(On Blackwell node)* `pytest -m "blackwell"` — all pass.

---

## Known Risks and Mitigations

**Gradient checkpointing + `torch.compile`**: HF's `gradient_checkpointing_enable()`
uses `torch.utils.checkpoint.checkpoint` which is compile-compatible in 2.11+ but
may produce graph breaks on some configurations. If graph breaks are observed in
8.4, fall back to disabling gradient checkpointing when `compile=True` is set and
document this limitation. A `compile_with_gradient_checkpointing` config flag can
be added if needed.

**Muon + FSDP2**: `torch.optim.Muon` (PyTorch 2.11) was designed with FSDP2
compatibility — it all-gathers each weight matrix before NS orthogonalization and
scatters back. Verify this empirically in 8.6. If instability is observed, fall
back to AdamW for FP8 runs initially.

**DCP checkpoint format incompatibility**: Existing checkpoints created with
Accelerate's `save_state` use a different format (per-shard safetensors files +
a pickle). They cannot be loaded by the new DCP-based `load_checkpoint`. This is
a one-time breaking change. Document it in `docs/TRAIN.md` and provide a migration
note: the `hf/` subdirectory (saved by `save_pretrained`) remains loadable for
inference regardless of format; only the trainer-state resume path is affected.

**FP8 scope on non-Blackwell**: `convert_to_float8_training` will raise at
runtime if the hardware doesn't support FP8 matmuls. The `is_fp8_supported()`
check in `Trainer.__init__` should gate FP8 conversion and raise a clear error
early rather than letting torchao raise deep in the training loop.
