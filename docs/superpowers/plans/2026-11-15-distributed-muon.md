# Batched + Distributed Muon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Under DDP, each rank orthogonalizes only the Muon matrices it owns, in batched calls, and all ranks apply identical updates.

**Architecture:** `class Muon(torch.optim.Muon)` in `src/oplm/training/optim.py` overrides `step` only: local momentum for every matrix, round-robin ownership, batched bf16 Newton–Schulz per shape bucket into a zero-filled buffer, one `all_reduce(SUM)` per bucket, then apply. State layout is stock Muon's.

**Tech Stack:** PyTorch (`torch.optim.Muon`, `torch.distributed`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-distributed-muon-design.md`

## Global Constraints

- Same math as `torch.optim.Muon` within bf16 rounding; `state_dict` interchangeable with stock in both directions.
- `train.parallelism=hsdp` keeps stock `torch.optim.Muon`.
- No config switch, no change to NS coefficients, LR adjustment or weight decay.
- `ruff check src/`, `ruff format src/`, `ty check src/` clean; run tests with `PYTHONPATH=src .venv/bin/python -m pytest`.

## Review Focus

- A matrix with an all-zero gradient: update is zero, never NaN (norm clamped at `eps`).
- More ranks than matrices in a bucket: ranks owning nothing still join the all-reduce.
- A parameter with `grad is None`: skipped, as stock; ownership stays rank-identical.
- Several param groups with different LRs (depth multipliers): each group's LR applies to its own matrices.
- `nesterov=False`: the momentum buffer itself is orthogonalized and must not be modified by it.

---

### Task 1: `Muon` subclass, wired into `build_optimizers`

**Files:**
- Modify: `src/oplm/training/optim.py`
- Test: `tests/training/test_optim.py`

**Interfaces:**
- Produces: `oplm.training.optim.Muon(torch.optim.Muon)` with stock constructor; `_newton_schulz(x: Tensor, coefficients: tuple[float, float, float], steps: int, eps: float) -> Tensor` taking and returning a `(k, rows, cols)` bf16 stack.

- [ ] **Step 1: Failing tests** in `tests/training/test_optim.py`:
  - `test_muon_matches_torch_muon` (parametrized `nesterov` True/False): params of shapes `(8, 8) ×3`, `(8, 24) ×2`, `(24, 8) ×2`, split over two groups with LRs `0.02` and `0.01`, `weight_decay=0.01`; one `(8, 8)` param gets a zero gradient and one gets `grad=None`. Four steps of fresh random grads on both optimizers; every param within `atol=2e-2 * lr` of stock (bf16 reordering), no NaN, the `grad=None` param unchanged.
  - `test_muon_state_dict_interchangeable_with_torch_muon`: after one step, `load_state_dict` works stock→ours and ours→stock, and the next step still matches.
  - `test_hsdp_keeps_torch_muon`: `build_optimizers(model, TrainConfig(optimizer="muon", parallelism="hsdp", ...))[0]` has `type(...) is torch.optim.Muon`; under `ddp` it is `oplm.training.optim.Muon`.
- [ ] **Step 2:** Run them; expect import failure on `Muon`.
- [ ] **Step 3:** Implement per the spec's Design section. Ownership counter runs across buckets within a group (balances when a bucket has fewer matrices than ranks). World size/rank from `torch.distributed` only when initialized.
- [ ] **Step 4:** Tests pass; lint, format and type check clean.
- [ ] **Step 5:** Commit.

### Task 2: Two ranks equal one

**Files:**
- Create: `tests/training/_distributed_muon_worker.py`
- Test: `tests/training/test_distributed_muon.py`

- [ ] **Step 1:** Worker (pattern of `_fsdp2_muon_spike_worker.py`): gloo group; seed 0; params `(8, 8) ×3`, `(8, 24) ×1`; three steps with seeded grads identical on both ranks; write final params to `out_dir/rank<i>.pt`.
- [ ] **Step 2:** `test_two_ranks_match_one_process` (`@pytest.mark.slow`): launch with `torch.distributed.run --nproc_per_node=2`; assert rank 0 and rank 1 params are bit-equal, and equal (bit-exact) to the same loop run in-process without `torch.distributed`.
- [ ] **Step 3:** Run; pass. Commit.

### Task 3: Docs

**Files:**
- Modify: `docs/training.md` (optimizer section, the `muon` bullet), `src/oplm/training/parallel.py` (note 3: point at #128)

- [ ] **Step 1:** State that under DDP Muon is batched and distributed (one sentence on how), and that HSDP uses stock `torch.optim.Muon`.
- [ ] **Step 2:** `PYTHONPATH=src .venv/bin/python -m pytest tests/test_docs.py -q` passes. Commit.

### Task 4: Cluster proof (after the branch is pushed)

- [ ] 170M 1-node profile and 3,000-step seed-42 equivalence, old commit against new.
- [ ] 6B 4-node short profile, old against new (3B if 6B will not run).
- [ ] Evidence record in `docs/methods/`; close #53 with the numbers.
