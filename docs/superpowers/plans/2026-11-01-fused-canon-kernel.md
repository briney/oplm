# Fused Canon Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the compiled shift-add Canon convolution with a Triton forward/backward kernel that computes the same function, cutting the 170M step from ~895 ms toward ~550 ms without changing the model.

**Architecture:** `src/oplm/model/conv.py` keeps today's shift-add code as a functional reference (CPU path and test oracle). A new `src/oplm/model/canon_kernel.py` holds the Triton kernels, their Python launchers, a `torch.library` custom op `oplm::canon_conv` with registered autograd and fake implementation, and the startup self-check. `CanonConv.forward` gains a `residual` argument so the residual add can live inside the op; the four call sites (A, B, C, D) pass it instead of adding themselves. The trainer runs the self-check once on the first micro-batch.

**Tech Stack:** Python 3, PyTorch 2.11 (`torch.library.triton_op`, `torch.library.register_autograd`/`register_fake`, `torch.library.opcheck`), Triton 3.6 (GPU, and `TRITON_INTERPRET=1` on CPU), pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-fused-canon-kernel-design.md`

## Global Constraints

- Same function as today's `CanonConv.forward`: depthwise, bidirectional cross-correlation, left pad `k // 2`, right pad `k - 1 - k // 2`, any `k >= 2`, per-call `k`.
- Pad positions (`attention_mask == 0`, pads anywhere, bool or integer mask) are zeroed before the conv. With `segment_ids` (`(B, T)` int, `1..n` per sequence, `0` at pads) a tap reading another segment or beyond the row ends contributes zero.
- Accumulate in fp32. Conv output dtype = the autocast dtype when autocast is enabled, else `x.dtype` (read at the call site, passed to the op as `out_dtype`).
- Residual (`canon_residual: true`, the default) is `x + act(conv(zero_pad(x)))`: the **raw** `x` is added, pads included. It is folded into the op only when `canon_activation == "none"`; otherwise the op returns the conv and the call site applies activation and residual. Residual output dtype = `torch.promote_types(x.dtype, out_dtype)` (what `x + conv` gives today).
- `dx` has `x`'s dtype; `dW` has the weight's dtype (fp32 parameter) and the `(D, 1, k)` shape of `conv.weight`. `dW` is reduced deterministically (per-program fp32 partials, then a sum), no float atomics.
- Weights stay in `CanonConv.conv.weight`; checkpoints, init and μP parameter groups unchanged.
- No config option. Kernel used iff `x.is_cuda` and Triton is importable; everything else uses the reference.
- The reference implementation is not deleted.
- Kernel correctness tests must run on CPU under `TRITON_INTERPRET=1` (no local GPU); GPU-only tests skip with `pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")` (the pattern in `tests/fold/test_attention.py`).
- The development machine cannot reach the cluster. GPU gates are hand-offs: the implementer writes the sbatch/commands, Bryan runs them and copies results to `~/data/tmp/`.
- Packed batches are the primary path; benchmarks use packed batches.
- Use `.venv/bin/python` / `.venv/bin/pytest`; fast suite is `-m "not slow"`.

## Review Focus

1. Residual at pad positions: with `residual=True` the output at a pad equals the raw `x` there (not zero), and `dx` at a pad equals `dy` there. Pinned in Task 2 and Task 3 tests.
2. Mixed dtypes: fp32 `x` with `out_dtype=bf16` (norm outputs can be fp32 under autocast) gives a conv in bf16 and a residual result in fp32. Pinned in Task 2.
3. Segment boundaries on tile edges: boundaries at `BLOCK_T - 1`, `BLOCK_T`, `BLOCK_T + 1`, and `T` not a multiple of the block. Pinned by the randomized layout test in Tasks 2 and 3.
4. Degenerate rows: an all-pad row inside a packed batch, `T` smaller than `k` (`T = 1, 2`), a single-token segment. Output finite, zeros where the reference has zeros, zero gradients there. Pinned in Tasks 2 and 3.
5. The self-check firing on a real mismatch: a perturbed kernel must raise, naming the module. Pinned in Task 5.

---

### Task 1: Functional reference and `residual` argument

Behaviour-preserving refactor so the op boundary (Task 4) can include the residual add.

**Files:**
- Modify: `src/oplm/model/conv.py` (`CanonConv.forward`; new module-level function)
- Modify: `src/oplm/model/transformer.py` (Canon-A ~line 259, Canon-C ~line 287)
- Modify: `src/oplm/model/attention.py` (`_apply_canon_b`, ~line 297)
- Modify: `src/oplm/model/ffn.py` (`_apply_canon_d`, ~line 49)
- Test: `tests/model/test_conv.py`

**Interfaces:**
- Produces: `canon_conv_reference(x: Tensor, weight: Tensor, attention_mask: Tensor, segment_ids: Tensor | None, out_dtype: torch.dtype, residual: bool) -> Tensor` in `conv.py`. `weight` is `(D, k)` (the squeezed `conv.weight`). Returns `conv` in `out_dtype` when `residual=False`, else `x + conv` (promoted dtype). No activation; the module applies it.
- Produces: `CanonConv.forward(x, attention_mask, segment_ids=None, residual: bool = False) -> Tensor`, returning `x + act(conv)` when `residual` else `act(conv)`.

- [ ] **Step 1: Write the failing tests** in `tests/model/test_conv.py`:
  - `test_residual_matches_explicit_add`: for activation in `("none", "silu")`, `conv(x, m, seg, residual=True)` equals `x + conv(x, m, seg)` exactly (`torch.equal`), including at pad positions, packed and unpacked.
  - `test_reference_function_matches_module`: `canon_conv_reference(x, conv.conv.weight.squeeze(1), m, seg, x.dtype, residual=False)` equals `conv(x, m, seg)` for activation `none`, `k` in `(2, 3, 4, 7)`.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model/test_conv.py -q` — expect the new tests to fail (`residual` / `canon_conv_reference` undefined).
- [ ] **Step 3: Implement.** Move the body of today's `forward` into `canon_conv_reference` unchanged (same fp32 path, same per-tap masks). `forward` reads `out_dtype` from autocast as today, calls the reference with `residual=False`, applies the activation, then adds `x` if `residual`. Call sites: replace `base + conv(base, ...) if self.canon_residual else conv(base, ...)` with `conv(base, ..., residual=self.canon_residual)` at A, B, C and D.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/model -q -m "not slow"` — all pass, including `test_canon_semantics.py` unchanged.
- [ ] **Step 5: Commit** `refactor(model): Canon reference function and residual argument`.

### Task 2: Triton forward kernel

**Files:**
- Create: `src/oplm/model/canon_kernel.py`
- Create: `tests/model/test_canon_kernel.py`

**Interfaces:**
- Consumes: `canon_conv_reference` (Task 1).
- Produces: `canon_forward(x: Tensor, weight: Tensor, attention_mask: Tensor, segment_ids: Tensor | None, out_dtype: torch.dtype, residual: bool) -> Tensor` — same contract as `canon_conv_reference`; `x` `(B, T, D)` contiguous, `weight` `(D, k)`. Launches a Triton kernel; works on CUDA tensors and on CPU tensors under `TRITON_INTERPRET=1`.
- Produces: `TRITON_AVAILABLE: bool` (Triton importable).

- [ ] **Step 1: Write the failing tests** in `tests/model/test_canon_kernel.py`. Module-level skip unless `torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"`; device is CUDA if available else CPU. A helper `_case(B, T, D, k, packed, seed)` builds `x`, `weight`, `attention_mask` and (if packed) random `segment_ids` with random segment lengths, trailing pads, and optionally an all-pad row. Tests compare `canon_forward` to `canon_conv_reference` on the same inputs:
  - `test_forward_matches_reference_fp32`: parametrize `k` in `(2, 3, 4, 7)`, `D` in `(64, 100)`, `T` in `(1, 2, 37, 128)`, packed in `(True, False)`, residual in `(True, False)`; `torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)`.
  - `test_forward_bf16_and_mixed_dtypes`: `x` bf16 with `out_dtype=bf16`, and `x` fp32 with `out_dtype=bf16`; output dtypes equal the reference's (`bf16` conv; fp32 residual for the mixed case); values `assert_close` at bf16 tolerance (`rtol=1.6e-2, atol=1e-2`).
  - `test_forward_segment_boundaries_on_tile_edges`: for `T = 2 * BLOCK_T + 3` (import the kernel's T block constant), boundaries placed at `BLOCK_T - 1`, `BLOCK_T`, `BLOCK_T + 1`; 20 random layouts; fp32 close as above.
  - `test_forward_degenerate_rows`: all-pad row → zeros (conv) / raw `x` (residual); `T < k`; single-token segments; output finite.
  - `test_forward_residual_keeps_raw_pads`: with residual, `out[pad] == x[pad]` exactly.
  Keep shapes small so the interpreter run stays under a minute.
- [ ] **Step 2: Run** `TRITON_INTERPRET=1 .venv/bin/pytest tests/model/test_canon_kernel.py -q` — fail (`canon_forward` undefined).
- [ ] **Step 3: Implement `canon_forward`.** Grid over (`B`, T-blocks, D-blocks). Each program loads its `x` tile plus a `k - 1` halo along T, loads `attention_mask` and `segment_ids` for the tile and halo once, zeroes masked positions, and for each tap adds `w[:, i] * x[t + i - left]` only where the source is in-row and has the output's segment id (no segment check when `segment_ids is None`). fp32 accumulation, residual add of the raw `x`, cast, store. `k` and `left` are `tl.constexpr`. Block sizes are the implementer's choice; expose the T block as a module constant for the tile-edge test.
- [ ] **Step 4: Run** the Step 2 command — pass. Also `.venv/bin/pytest tests/model -q -m "not slow"` (the new file skips without the env var).
- [ ] **Step 5: Commit** `feat(model): Triton Canon forward kernel`.

### Task 3: Triton backward kernels

**Files:**
- Modify: `src/oplm/model/canon_kernel.py`
- Modify: `tests/model/test_canon_kernel.py`

**Interfaces:**
- Consumes: `canon_forward`, the test helper `_case` (Task 2).
- Produces: `canon_backward(dy: Tensor, x: Tensor, weight: Tensor, attention_mask: Tensor, segment_ids: Tensor | None, residual: bool) -> tuple[Tensor, Tensor]` returning `(dx, dw)`; `dx` in `x.dtype`, shape `(B, T, D)`; `dw` in `weight.dtype`, shape `(D, k)`. With `m` the pad mask and `C` the masked conv: `dx = m ⊙ Cᵀ(dy)` (+ `dy` when `residual`), `dw[d, i] = Σ_{b,t} keep_i(b,t) · (m⊙x)[b, t+i-left, d] · dy[b, t, d]`.

- [ ] **Step 1: Write the failing tests.** For each case, get reference gradients by autograd through `canon_conv_reference` (inputs `requires_grad`, same random `dy`), then compare with `canon_backward`:
  - `test_backward_matches_reference_fp32`: same parametrization as the forward fp32 test; `dx` and `dw` `assert_close` at `rtol=1e-5, atol=1e-5` (scale `atol` by `B*T` for `dw` if needed, stated in the test).
  - `test_backward_bf16`: bf16 `x`, fp32 weight; `dx` bf16, `dw` fp32; bf16 tolerance.
  - `test_backward_tile_edges` and `test_backward_degenerate_rows`: the forward cases' layouts; gradients at pads are `0` (no residual) or `dy` (residual); all-pad row contributes nothing to `dw`.
  - `test_backward_deterministic`: two calls on the same inputs give bitwise-equal `dw`.
- [ ] **Step 2: Run** `TRITON_INTERPRET=1 .venv/bin/pytest tests/model/test_canon_kernel.py -q` — new tests fail.
- [ ] **Step 3: Implement `canon_backward`.** `dx`: the transposed conv (taps mirrored), with the keep test evaluated on the (input, output) pair, times the pad mask, plus `dy` for residual. `dw`: one kernel, each program accumulates its tile's `k` per-tap fp32 sums into a `(num_programs_over_B_and_T, D, k)` partials buffer; finish with `partials.sum(0)` cast to the weight dtype. `dx` and `dw` may share one kernel or use two; the implementer chooses.
- [ ] **Step 4: Run** the Step 2 command — pass.
- [ ] **Step 5: Commit** `feat(model): Triton Canon backward kernels`.

### Task 4: Custom op, autograd and dispatch

**Files:**
- Modify: `src/oplm/model/canon_kernel.py`
- Modify: `src/oplm/model/conv.py` (`CanonConv.forward` dispatch, docstring)
- Modify: `docs/architecture.md` (*Canon semantics*: one paragraph on the kernel, its CPU fallback and the self-check)
- Test: `tests/model/test_canon_kernel.py`

**Interfaces:**
- Consumes: `canon_forward`, `canon_backward`, `TRITON_AVAILABLE`.
- Produces: op `torch.ops.oplm.canon_conv(x, weight, attention_mask, segment_ids, out_dtype, residual) -> Tensor` (`weight` `(D, k)`; `segment_ids` optional), registered with `torch.library.triton_op` (or `custom_op` if `triton_op` cannot express it), `register_fake` (output shape `x.shape`, dtype per the Global Constraints) and `register_autograd` (saves `x`, `weight`, masks; backward calls `canon_backward`; no grads for masks).
- Produces: `canon_conv_kernel(x, weight, attention_mask, segment_ids, out_dtype, residual) -> Tensor` — thin Python wrapper calling the op; same signature as `canon_conv_reference`.
- Produces: in `conv.py`, `CanonConv.forward` uses `canon_conv_kernel` when `x.is_cuda and TRITON_AVAILABLE`, else `canon_conv_reference`; `residual` is passed to the op only when the activation is `none` (otherwise op with `residual=False`, then activation, then add). Module-level `_KERNEL_ENABLED = True` (private, tests only) forces the reference when `False`.

- [ ] **Step 1: Write the failing tests** (all CUDA-only):
  - `test_op_opcheck`: `torch.library.opcheck(torch.ops.oplm.canon_conv, args)` passes for a packed bf16 case and an unpacked fp32 case.
  - `test_module_kernel_matches_reference`: a `CanonConv` on CUDA with `_KERNEL_ENABLED` True vs False: outputs and `conv.weight.grad` close (fp32 tolerance in fp32, bf16 tolerance under `torch.autocast("cuda", torch.bfloat16)`), activation `none` and one other.
  - `test_compiled_block_no_graph_breaks`: an `OplmBlock` (pre-norm, Canon A–D, `k=7`, packed `segment_ids`, flex attention as in `tests/model/test_packing.py`) compiled with `torch.compile(block, fullgraph=True)` runs forward + backward and matches the eager block with `_KERNEL_ENABLED=False` at bf16 tolerance.
  - CPU (always runs): `test_cpu_uses_reference` — on CPU, `CanonConv.forward` output equals `canon_conv_reference` exactly.
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model/test_canon_kernel.py -q` — CPU test fails; CUDA tests skip locally.
- [ ] **Step 3: Implement** the op registration, wrapper and dispatch; update the `CanonConv` docstring (the "computes that convolution as shifted multiply-adds" paragraph) and `docs/architecture.md`.
- [ ] **Step 4: Run** `.venv/bin/pytest -q -m "not slow"` — full fast suite passes; `TRITON_INTERPRET=1 .venv/bin/pytest tests/model/test_canon_kernel.py -q` still passes.
- [ ] **Step 5: Commit** `feat(model): oplm::canon_conv custom op with Triton autograd`.

### Task 5: Startup self-check

**Files:**
- Modify: `src/oplm/model/canon_kernel.py`
- Modify: `src/oplm/training/trainer.py` (`train`, before the first `self.task.step` of the run)
- Modify: `docs/training.md` (one paragraph near *Profiling*: what the self-check does and the log line)
- Test: `tests/model/test_canon_kernel.py`

**Interfaces:**
- Consumes: `canon_conv_reference`, `canon_conv_kernel`, `CanonConv`.
- Produces: `class CanonKernelMismatch(RuntimeError)`.
- Produces: `canon_self_check(model: nn.Module, run_forward: Callable[[], object], kernel: Callable[..., Tensor] = canon_conv_kernel) -> int` — registers forward pre-hooks on every `CanonConv` in `model`, calls `run_forward()` under `torch.no_grad()`, and in each hook (under `torch.enable_grad()`) runs `kernel` and `canon_conv_reference` on that call's `x`, `attention_mask`, `segment_ids` cast to fp32, with `out_dtype=torch.float32` and the same `residual` value `CanonConv.forward` passes to the op for that call (read from the hook's kwargs; register with `with_kwargs=True`), comparing the conv before the module's activation, backprops one shared random `dy` through both, compares output, `dx` and `dW` (`rtol=1e-4, atol=1e-4`), and raises `CanonKernelMismatch` naming the module's qualified name, the input shape and the max abs error per tensor. Hooks are removed in a `finally`. Returns the number of calls checked.

- [ ] **Step 1: Write the failing tests** (CPU; `kernel` injected so no GPU is needed):
  - `test_self_check_passes_with_reference_kernel`: a small 2-layer `OplmModel` with Canon A–D, packed batch; `canon_self_check(model, lambda: model(**batch), kernel=canon_conv_reference)` returns `2 * 6`.
  - `test_self_check_raises_on_mismatch`: kernel = reference output `+ 1e-2` on one specific call; raises `CanonKernelMismatch` whose message contains that module's name.
  - `test_self_check_removes_hooks`: after a raise, a second forward does not invoke the hook (no hooks left on any `CanonConv`).
- [ ] **Step 2: Run** `.venv/bin/pytest tests/model/test_canon_kernel.py -q -k self_check` — fail.
- [ ] **Step 3: Implement** `canon_self_check`. Wire it in `Trainer.train`: once per run (also after resume), on the first micro-batch, when the device is CUDA, `TRITON_AVAILABLE`, and the model contains a `CanonConv`, call it on `self._unwrapped_model` with `run_forward` = `self.task.step(self._unwrapped_model, small_batch)` inside the accelerator's autocast, where `small_batch` keeps the first `min(4, B)` rows of every tensor whose leading dim is the batch size (bounds the eager flex-attention memory; packed rows still contain many sequence boundaries). Log `Canon self-check passed: N calls` on rank 0. The exception propagates and stops the run.
- [ ] **Step 4: Run** `.venv/bin/pytest -q -m "not slow"` — pass.
- [ ] **Step 5: Commit** `feat(train): Canon kernel startup self-check`.

### Task 6: GPU validation scripts (hand-off)

**Files:**
- Create: `docs/perf/canon_kernel_bench.py`
- Create: `docs/perf/canon-kernel.sbatch`

**Interfaces:**
- Consumes: `canon_conv_kernel`, `canon_conv_reference`.

- [ ] **Step 1: Write `canon_kernel_bench.py`.** On one CUDA GPU, for the 170M Canon shapes (packed rows from 65,536 tokens: `D = 768` and `D = 2048`, `k = 7`, realistic segment lengths), time forward+backward per call for the compiled reference (`torch.compile` of `canon_conv_reference`) and the kernel, with CUDA-synced medians over 50 iterations after 10 warmups; print a table and write JSON to `--out`.
- [ ] **Step 2: Write `canon-kernel.sbatch`**, modelled on the coord-check sbatch (partition `hpc-mid`, 1 GPU, 1 h, container `/mnt/data/containers/deeplearning_v2026-05-26.sqsh`, logs `/mnt/home/briney/logs`, `source /mnt/home/briney/.env`). It installs `oplm[train]` from the branch commit given as `$OPLM_REF`, runs `pytest tests/model/test_canon_kernel.py tests/model/test_conv.py tests/model/test_canon_semantics.py -q` on the GPU, then the bench, writing to `$OUT` (default `/mnt/home/briney/projects/oplm/mup_redo/canon-kernel`).
- [ ] **Step 3: Verify locally** `bash -n docs/perf/canon-kernel.sbatch` and `.venv/bin/python docs/perf/canon_kernel_bench.py --help`.
- [ ] **Step 4: Commit** `perf(docs): Canon kernel GPU test and benchmark job`.
- [ ] **Step 5: Hand-off — stop and ask Bryan** to push the branch (needs his OK) and run, on `slurm-login-0`: (a) `OPLM_REF=<commit> sbatch canon-kernel.sbatch` — gate: all GPU tests pass, kernel faster than the compiled reference at both widths; (b) `profile-submit.sh` with its pin changed to the commit — gate: OPLM step ≤ ~550 ms, elementwise kernels ≤ ~60 ms/step above ESM-C's 110, self-check log line present; (c) the training-equivalence run: the `refine-170M-lr0.004-om1-a0-s42` cell's `run.yaml` from the no-gate sweep on the new commit, cancelled after 3,000 steps — gate: eval loss within ~0.003 of W&B `b0x13s29` at 500/1,000/1,500/2,000/2,500/3,000 and matching grad-norm behaviour. Results come back to `~/data/tmp/`; record them in the PR description. A miss on (b) alone is reported with the measured conv vs surrounding split, not fixed by widening scope.
