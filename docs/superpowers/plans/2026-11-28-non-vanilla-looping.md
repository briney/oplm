# Non-Vanilla Shared-Layer Looping (Step 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add composable, default-off prelude input injection (#142), per-cycle residual gating (#143 part 1), and lightweight pass symmetry breaking (#144 part 1) to `OplmStack`/`OplmBlock`/`OplmAttention`, wired end-to-end through config, optimizer, μP, FLOPs, warm-start `train.init_from`, resume guards, `trust_remote_code`, and docs.

**Architecture:** `looping.py` exposes an explicit `LoopSchedule` (prelude / core / coda), a self-contained `loop_modules.py` implements the three core-boundary modules (`LoopInputInjector`, `LoopCycleGate`, `LoopStepConditioner`), and `OplmBlock`/`OplmAttention` add key-preserving pass `1..K-1` norm/lambda lists when `loop_untie_norms=True`. Pass 0 runs un-injected, un-conditioned, and un-gated so defaults are bit-identical to `main` and `loop_gate_init=0.0` gives an exact identity warm start.

**Tech Stack:** Python 3.11+, PyTorch 2.10+, HuggingFace `transformers`, `pytest`, `ruff`, `ty`.

**Spec:** [docs/superpowers/specs/2026-11-28-non-vanilla-looping-design.md](../specs/2026-11-28-non-vanilla-looping-design.md)

## Global Constraints

- Python 3.11+, `from __future__ import annotations` in every file, modern union typing, 100-char line length, no `os.path`, no logic in `__init__.py`, no bare `# type: ignore`.
- `.venv/bin/python -m pytest -m "not slow" -q`, `ruff check src/`, `ruff format --check src/`, and `ty check src/` must all pass cleanly.
- Zero regression at defaults (`loop_input_injection="none"`, `loop_gate="none"`, `loop_untie_norms=False`, `loop_step_conditioning="none"`): state-dict keys, parameter counts, init, forward outputs, and gradients must be bit-identical to `main`.
- Any new module under `src/oplm/model/` must be imported directly in both `configuration_oplm.py` and `modeling_oplm.py` and referenced in `_REMOTE_CODE_DEPS` for depth-1 HuggingFace `trust_remote_code=True` bundling.
- Non-default loop extensions are rejected unless `loop_strategy == "stack"` and `num_loops >= 2`.
- All pass-0 norm/lambda attribute names stay unchanged when `loop_untie_norms=True`; extra passes `1..num_loops-1` live in `loop_*` module/parameter lists.

## Review Focus

1. **Zero-norm prelude rows (`<pad>` or zeroed `<mask>`) under `orthogonal` injection:** clamping `||e||_2` in fp32 (`min=1e-6`) must produce finite outputs and finite gradients (no NaN). *(Pinned in Task 2)*
2. **Odd `hidden_size` in sinusoidal `LoopStepConditioner`:** padding one zero channel must avoid shape errors on odd widths while keeping `mup_fanin_mult(D) == 1.0` at base width. *(Pinned in Task 2 & Task 4)*
3. **HF `post_init` / `from_pretrained` clobbering custom inits or trained values:** `_init_weights` must respect `_init_zero` and leave custom `decayed`/`orthogonal`/`LoopCycleGate` parameters intact, and `save_pretrained`/`from_pretrained` (including `trust_remote_code=True`) must round-trip mutated non-init weights bit-for-bit. *(Pinned in Task 3 & Task 6)*
4. **Step-0 zero-grad & weight-decay safety under Muon + `weight_decay_to_init=0.1`:** with `loop_gate_init=0.0`, step-0 forward equals unlooped identically, `LoopStepConditioner.fc1` zero grad does not NaN under Muon Newton-Schulz, `gates`/`log_delta`/`log_state_decay`/`inject_scale`/`loop_value_residual_lambdas` stay undecayed, and `decayed.input_matrix_continuous` lands in AdamW with `weight_decay=0.0`. *(Pinned in Task 3 & Task 4)*
5. **Activation checkpointing (`full` & `selective`) + hidden-state/QK taps with loop extensions:** checkpointed forward and gradients match uncheckpointed reference, `len(hidden_states) == len(layer_execution_order) + 1`, and the last block of pass `k >= 1` records the post-`LoopCycleGate` state. *(Pinned in Task 3)*

---

## File Structure

| File | Responsibility |
|---|---|
| `src/oplm/model/looping.py` | `LoopSchedule` dataclass and `resolve_loop_schedule(...)` alongside unchanged `resolve_layer_execution_order`. |
| `src/oplm/model/loop_modules.py` (new) | `LoopInputInjector` (`concat_proj`, `decayed`, `orthogonal`), `LoopCycleGate` (1-D `nn.ParameterList`), `LoopStepConditioner` (`sinusoidal`, width-`D`, zero-init residual FiLM). |
| `src/oplm/model/configuration_oplm.py` | Five new `OplmConfig` fields, validation rules, `mup_fanin_mult(2 * hidden_size)`, `_REMOTE_CODE_DEPS` import. |
| `src/oplm/configs/model/base.yaml` | Authoritative run defaults for the five new `model.*` fields. |
| `src/oplm/model/attention.py` | Optional `loop_value_residual_lambdas` (`nn.ParameterList`) and `pass_idx: int = 0` on `OplmAttention.forward`. |
| `src/oplm/model/transformer.py` | Optional `loop_*_norms` (`nn.ModuleList`) and `pass_idx: int = 0` on `OplmBlock`; prelude/core/coda loop boundary wiring in `OplmStack`. |
| `src/oplm/model/modeling_oplm.py` | `_init_zero` support in `_init_weights` and `_REMOTE_CODE_DEPS` import of `loop_modules`. |
| `src/oplm/training/optim.py` | `_init_value` support for untied norms and no-decay exclusions; AdamW routing (`weight_decay=0.0`) for `decayed.input_matrix_continuous`. |
| `src/oplm/training/flops.py` | Add `(num_loops - 1)` injection matmul FLOPs for `concat_proj` and `decayed`. |
| `src/oplm/training/initialization.py` | Extension-transition checks, namespace-scoped missing-key allowlist on newly enabled extensions, pass-0 -> pass `1..K-1` norm/lambda copy. |
| `src/oplm/training/checkpoint.py` | Extend `validate_loop_resume_compat` signature with the five new fields + legacy defaults. |
| `docs/architecture.md`, `docs/config.md`, `docs/training.md` | Document new config fields, semantics, warm-start, and resume rules. |

---

### Task 1: Config surface, validation, and `LoopSchedule`

**Files:**
- Modify: `src/oplm/model/looping.py`
- Modify: `src/oplm/model/configuration_oplm.py`
- Modify: `src/oplm/configs/model/base.yaml`
- Test: `tests/model/test_looping.py`

**Interfaces:**
- Produces:
  - `LoopSchedule(prelude: tuple[int, ...], core: tuple[int, ...], coda: tuple[int, ...], num_loops: int, loop_strategy: str, execution_order: tuple[int, ...])` and `resolve_loop_schedule(num_hidden_layers: int, *, num_loops: int = 1, loop_strategy: str = "stack", loop_start: int = 0, loop_end: int | None = None) -> LoopSchedule` in `src/oplm/model/looping.py`.
  - `OplmConfig` keyword-only fields: `loop_input_injection: str = "none"`, `loop_gate: str = "none"`, `loop_gate_init: float = 1.0`, `loop_untie_norms: bool = False`, `loop_step_conditioning: str = "none"`, plus helper property `has_loop_extensions -> bool`.
  - `OplmConfig.mup_fanin_mult(2 * hidden_size)` returns `self.mup_width_mult()`.

- [ ] **Step 1: Write failing tests in `tests/model/test_looping.py`**
  - `test_resolve_loop_schedule_partitions_prelude_core_coda`: verify `resolve_loop_schedule(6, num_loops=3, loop_strategy="stack", loop_start=1, loop_end=5)` yields `prelude=(0,)`, `core=(1, 2, 3, 4)`, `coda=(5,)`, `execution_order=(0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5)`.
  - `test_loop_extension_validation_rejects_invalid_values_and_combos`: parametrize invalid enums (`loop_input_injection="bad"`, `loop_gate="bad"`, `loop_step_conditioning="bad"`), non-finite `loop_gate_init` (`float("nan")`, `float("inf")`), non-bool `loop_untie_norms` (`1`), any extension with `num_loops=1`, and any extension with `loop_strategy="interleave"` (`num_loops=2`) — all raise `ValueError`.
  - `test_loop_extension_config_round_trip`: `load_config` with all 5 CLI overrides (`num_loops=2`, `loop_start=1`, `loop_end=3`, `loop_input_injection=decayed`, `loop_gate=channel`, `loop_gate_init=0.0`, `loop_untie_norms=true`, `loop_step_conditioning=sinusoidal`) round-trips through `save_pretrained`/`from_pretrained` and `serialize_config`/`load_config`.
  - `test_mup_fanin_mult_concat_proj_width`: with `mup_enable=True, mup_base_width=768`, assert `OplmConfig(hidden_size=768, mup_enable=True, mup_base_width=768).mup_fanin_mult(1536) == 1.0` and `OplmConfig(hidden_size=1536, num_attention_heads=24, mup_enable=True, mup_base_width=768).mup_fanin_mult(3072) == 2.0`.

- [ ] **Step 2: Run tests to verify RED**
  Run: `.venv/bin/python -m pytest tests/model/test_looping.py -k "schedule or extension or concat_proj" -v`
  Expected: FAIL (`ImportError` / unexpected keyword argument).

- [ ] **Step 3: Implement `LoopSchedule`, config fields, validation, and `base.yaml` defaults**
  - Add `LoopSchedule` and `resolve_loop_schedule` to `src/oplm/model/looping.py` (delegating validation to `resolve_layer_execution_order`).
  - Add the 5 fields + `_VALID_LOOP_INPUT_INJECTIONS`, `_VALID_LOOP_GATES`, `_VALID_LOOP_STEP_CONDITIONINGS` + validation in `OplmConfig._validate` + `has_loop_extensions` property + `fan_in == 2 * self.hidden_size` branch in `OplmConfig.mup_fanin_mult` (checked before `intermediate_size`, or handle overlap cleanly: `fan_in == 2 * self.hidden_size` returns `self.mup_width_mult()` only when `fan_in != self.intermediate_size`? No—if `intermediate_size == 2 * hidden_size` in a tiny test, `mup_base_intermediate_size()` is derived from `mup_base_width`, whereas `2 * hidden_size` is explicitly for `concat_proj` whose base fan-in is `2 * mup_base_width`! Checking `fan_in == self.intermediate_size` first preserves exact existing `down_proj` behavior, then `elif fan_in == 2 * self.hidden_size: return self.mup_width_mult()`).
  - Add the 5 default keys under `model:` in `src/oplm/configs/model/base.yaml`.

- [ ] **Step 4: Run tests to verify GREEN**
  Run: `.venv/bin/python -m pytest tests/model/test_looping.py tests/model/test_config.py tests/training/test_config.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/oplm/model/looping.py src/oplm/model/configuration_oplm.py src/oplm/configs/model/base.yaml tests/model/test_looping.py
  git commit -m "feat(model): add config surface and LoopSchedule for Step 2 looping extensions"
  ```

---

### Task 2: Loop boundary modules (`src/oplm/model/loop_modules.py`)

**Files:**
- Create: `src/oplm/model/loop_modules.py`
- Test: `tests/model/test_loop_modules.py`

**Interfaces:**
- Consumes: `make_norm` from `src/oplm/model/norm.py`.
- Produces:
  - `LoopInputInjector(mode: str, hidden_size: int, norm_type: str, norm_eps: float)`: `forward(z: Tensor, e: Tensor) -> Tensor`, `reset_parameters() -> None`.
  - `LoopCycleGate(mode: str, num_loops: int, hidden_size: int, init_val: float)`: `self.gates: nn.ParameterList` of `num_loops - 1` 1-D `nn.Parameter`s (`(1,)` or `(D,)`), `forward(z_prev: Tensor, core_out: Tensor, gate_idx: int) -> Tensor`.
  - `LoopStepConditioner(hidden_size: int, norm_type: str, norm_eps: float)`: `self.norm`, `self.fc1: nn.Linear(D, D)`, `self.fc2: nn.Linear(D, 2 * D)` (tagged `_init_zero = True`), `forward(z: Tensor, pass_idx: int, num_loops: int) -> Tensor`.

- [ ] **Step 1: Write failing tests in `tests/model/test_loop_modules.py`**
  - `test_concat_proj_injector_shapes_and_grads`: `(B=2, T=6, D=32)` forward shape `(2, 6, 32)`, dtype preserved (fp32 and bf16), finite grads to `z`, `e`, and `proj.weight`.
  - `test_decayed_injector_matches_formula_and_init`: verify init gives `a ≈ 1/sqrt(5)`, `input_matrix_continuous == I_D`, fp32 dynamics cast to input dtype, finite grads.
  - `test_orthogonal_injector_projects_out_e_and_handles_zero_norm_rows` *(Review Focus #1)*: verify `(z_out - scale * e)` is orthogonal to `e` (inner product `< 1e-5` on non-zero rows), AND a row where `e == 0` produces finite outputs and finite gradients for `z`, `e`, and `inject_scale`.
  - `test_cycle_gate_parameter_list_is_1d_and_zero_init_is_exact_identity`: for `scalar` and `channel` at `num_loops=3, init_val=0.0`, assert `len(gate.gates) == 2`, every param has `ndim == 1`, `torch.equal(gate(z_prev, core_out, 0), z_prev)`, and backward produces nonzero grad on `gate.gates[0]`.
  - `test_step_conditioner_zero_init_identity_and_odd_hidden_size` *(Review Focus #2)*: for `D=32` and odd `D=31`, assert `torch.equal(cond(z, pass_idx=1, num_loops=3), z)` at init (`fc2` zeroed), `cond.fc2._init_zero is True`, and after perturbing `fc2.weight`, `pass_idx=1` and `pass_idx=2` produce distinct outputs.

- [ ] **Step 2: Run tests to verify RED**
  Run: `.venv/bin/python -m pytest tests/model/test_loop_modules.py -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'oplm.model.loop_modules'`).

- [ ] **Step 3: Implement `src/oplm/model/loop_modules.py`**
  - Implement `LoopInputInjector`, `LoopCycleGate`, and `LoopStepConditioner` exactly per Spec §4a–4c:
    - `concat_proj`: `z_norm`, `e_norm`, `proj = nn.Linear(2 * D, D, bias=False)`, inputs scaled by `2.0 ** -0.5`.
    - `decayed`: `input_norm`, `log_delta`, `log_state_decay`, `input_matrix_continuous` (initialized in `reset_parameters()` via `delta0 = 0.5 * math.log(5.0)` and `torch.eye(D)`).
    - `orthogonal`: `inject_scale = nn.Parameter(torch.ones(D))`, fp32 unit-vector projection with `.clamp(min=1e-6)`.
    - `LoopCycleGate`: `nn.ParameterList` of `num_loops - 1` 1-D tensors.
    - `LoopStepConditioner`: sinusoidal encoding of `t = pass_idx / max(num_loops - 1, 1)` at width `D` (if `D` is odd, generate `D // 2` sin/cos pairs and append one zero channel), `fc1 = nn.Linear(D, D)`, `SiLU`, `fc2 = nn.Linear(D, 2 * D)` with zero-init weight/bias and `fc2._init_zero = True`, residual FiLM `z + gamma * self.norm(z) + beta`.

- [ ] **Step 4: Run tests to verify GREEN**
  Run: `.venv/bin/python -m pytest tests/model/test_loop_modules.py -v`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/oplm/model/loop_modules.py tests/model/test_loop_modules.py
  git commit -m "feat(model): add loop boundary modules (injector, cycle gate, step conditioner)"
  ```

---

### Task 3: Wire Step 2 extensions into `OplmStack`, `OplmBlock`, `OplmAttention`, and `OplmPreTrainedModel`

**Files:**
- Modify: `src/oplm/model/attention.py`
- Modify: `src/oplm/model/transformer.py`
- Modify: `src/oplm/model/modeling_oplm.py`
- Modify: `src/oplm/model/configuration_oplm.py`
- Test: `tests/model/test_looping.py`
- Test: `tests/model/test_save_load.py`

**Interfaces:**
- Consumes: `LoopSchedule`, `resolve_loop_schedule` (Task 1); `LoopInputInjector`, `LoopCycleGate`, `LoopStepConditioner` (Task 2).
- Produces:
  - `OplmAttention(config, layer_idx=0, *, num_loop_passes: int = 1)` with optional `self.loop_value_residual_lambdas: nn.ParameterList` (length `num_loop_passes - 1`, only when `loop_untie_norms and num_loop_passes > 1 and value_residual == "learnable" and layer_idx > 0`); `forward(..., pass_idx: int = 0)`.
  - `OplmBlock(config, layer_idx, *, num_loop_passes: int = 1)` with optional `loop_attn_norms`, `loop_ffn_norms`, `loop_attn_post_norms`, `loop_ffn_post_norms` (`nn.ModuleList` of length `num_loop_passes - 1`, preserving pass-0 keys unchanged); `forward(..., pass_idx: int = 0)`.
  - `OplmStack` holding optional `self.loop_injector`, `self.loop_step_cond`, `self.loop_gate_module` and running prelude -> pass 0 -> passes `1..K-1` -> coda.
  - `OplmPreTrainedModel._init_weights` honoring `getattr(module, "_init_zero", False)` on `nn.Linear`.
  - `_REMOTE_CODE_DEPS` in both `configuration_oplm.py` and `modeling_oplm.py` referencing `LoopInputInjector`, `LoopCycleGate`, `LoopStepConditioner`, `resolve_loop_schedule`.

- [ ] **Step 1: Write failing tests in `tests/model/test_looping.py` and `tests/model/test_save_load.py`**
  - In `tests/model/test_looping.py`:
    - `test_default_loop_extensions_are_bit_identical_to_vanilla`: confirm `num_loops=2` with all defaults has no `loop_injector` / `loop_gate_module` / `loop_step_cond` / `loop_*norms` keys and matches existing forward.
    - `test_zero_init_loop_gate_matches_unlooped_forward_exactly` *(Review Focus #4)*: copy weights from an unlooped `OplmForMaskedLM(num_loops=1)` into a Step 2 model with `num_loops=3, loop_start=1, loop_end=3, loop_input_injection=mode, loop_gate="channel", loop_gate_init=0.0, loop_untie_norms=True, loop_step_conditioning="sinusoidal"` (copying pass-0 norms/lambdas into `loop_*`), and assert `torch.equal` on logits for all `mode in ("none", "concat_proj", "decayed", "orthogonal")`.
    - `test_step2_extensions_forward_and_gradients_match_manual_reference`: parametrize `loop_input_injection` (`none`, `concat_proj`, `decayed`, `orthogonal`), `loop_gate` (`none`, `scalar`, `channel`), `loop_untie_norms` (`False`, `True`), `loop_step_conditioning` (`none`, `sinusoidal`) on prelude–core–coda (`num_hidden_layers=4, loop_start=1, loop_end=3, num_loops=3`, `norm_strategy="sandwich"`, `value_residual="learnable"`); assert `len(hidden_states) == len(layer_execution_order) + 1`, final core block hidden state equals post-`LoopCycleGate` state, and every trainable parameter receives a finite gradient (after a non-zero `fc2` or verifying step-0/step-1 flow).
    - `test_step2_extensions_checkpointing_parity` *(Review Focus #5)*: parametrize `mode in ("full", "selective")` with all extensions enabled (`decayed`, `channel`, `loop_untie_norms=True`, `sinusoidal`) + `return_hidden_layers` / `return_qk_layers`; assert loss and all parameter gradients match uncheckpointed reference.
    - `test_untied_norms_preserve_pass0_keys_and_skip_prelude_coda`: verify prelude (`layer 0`) and coda (`layer 3`) have NO `loop_*norms` or `loop_value_residual_lambdas`, while core (`layers 1, 2`) have pass-0 keys unchanged plus `loop_*` of length `num_loops - 1`.
  - In `tests/model/test_save_load.py`:
    - `test_step2_loop_extensions_save_load_preserves_custom_and_mutated_weights` *(Review Focus #3)*: mutate `decayed` (`log_delta`, `input_matrix_continuous`), `LoopCycleGate.gates`, `LoopStepConditioner.fc2`, and `loop_attn_norms` to non-init values, `save_pretrained` -> `from_pretrained`, assert `torch.equal` on every tensor in `state_dict()` and bit-identical eval logits.

- [ ] **Step 2: Run tests to verify RED**
  Run: `.venv/bin/python -m pytest tests/model/test_looping.py tests/model/test_save_load.py -k "step2 or zero_init_loop_gate or untied_norms or default_loop_extensions" -v`
  Expected: FAIL.

- [ ] **Step 3: Implement wiring in `attention.py`, `transformer.py`, `modeling_oplm.py`, `configuration_oplm.py`**
  - `attention.py`: accept `num_loop_passes: int = 1`; instantiate `self.loop_value_residual_lambdas` (`nn.ParameterList`) when `config.loop_untie_norms and num_loop_passes > 1 and self.value_residual == "learnable" and layer_idx > 0`; select `self.value_residual_lambda` at `pass_idx == 0` else `self.loop_value_residual_lambdas[pass_idx - 1]`.
  - `transformer.py`: accept `num_loop_passes: int = 1` on `OplmBlock` (computed per layer in `OplmStack.__init__`: `num_loops` if `loop_start <= idx < end and loop_strategy == "stack"` else `1`); instantiate extra pass norm `nn.ModuleList`s only when `config.loop_untie_norms and num_loop_passes > 1`; thread `pass_idx: int = 0` cleanly through `forward` / `_forward_impl` / `torch_checkpoint.checkpoint`; in `OplmStack.__init__` build `self.loop_schedule`, optional `self.loop_injector`, `self.loop_step_cond`, `self.loop_gate_module`; in `OplmStack.forward` execute prelude -> core pass 0 -> core passes `1..K-1` (inject -> step_cond -> blocks -> cycle_gate, updating the pass's final hidden-state tap to post-gate `z`) -> coda.
  - `modeling_oplm.py`: in `OplmPreTrainedModel._init_weights`, if `isinstance(module, nn.Linear)` and `getattr(module, "_init_zero", False)`, zero `module.weight` and `module.bias` instead of trunc-normal; import `loop_modules` symbols and `resolve_loop_schedule` in `_REMOTE_CODE_DEPS` in both `configuration_oplm.py` and `modeling_oplm.py`.

- [ ] **Step 4: Run tests to verify GREEN**
  Run: `.venv/bin/python -m pytest tests/model/test_looping.py tests/model/test_save_load.py tests/model/test_transformer.py tests/model/test_attention.py tests/model/test_gradient_checkpointing.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/oplm/model/attention.py src/oplm/model/transformer.py src/oplm/model/modeling_oplm.py src/oplm/model/configuration_oplm.py tests/model/test_looping.py tests/model/test_save_load.py
  git commit -m "feat(model): wire loop injection, cycle gating, step conditioning, and untied norms"
  ```

---

### Task 4: Optimizer grouping, μP parity, and FLOPs accounting

**Files:**
- Modify: `src/oplm/training/optim.py`
- Modify: `src/oplm/training/flops.py`
- Test: `tests/training/test_optim.py`
- Test: `tests/training/test_mup.py`
- Test: `tests/training/test_flops.py`

**Interfaces:**
- Consumes: `OplmForMaskedLM` with Step 2 modules (Task 3).
- Produces:
  - Updated `_init_value` and `partition_optimizer_params` in `src/oplm/training/optim.py`:
    - Untied block norm weights (`*.loop_*_norms.*.weight`) decay toward `1.0` under `weight_decay_to_init`; biases toward `0.0`.
    - `loop_gate_module.gates.*`, `loop_injector.log_delta`, `loop_injector.log_state_decay`, `loop_injector.inject_scale`, `loop_value_residual_lambdas.*` have `weight_decay == 0.0`.
    - `loop_injector.input_matrix_continuous` routes to AdamW (not Muon) with `weight_decay == 0.0`.
    - `loop_injector.proj.weight` (`concat_proj`) and `loop_step_cond.{fc1,fc2}.weight` follow standard Muon/AdamW hidden 2-D routing.
  - Updated `estimate_flops_per_token` in `src/oplm/training/flops.py` adding `3 * (num_loops - 1) * injection_fwd_flops` for `concat_proj` (`2 * (2 * h) * h`) and `decayed` (`2 * h * h`).

- [ ] **Step 1: Write failing tests in `tests/training/test_optim.py`, `tests/training/test_mup.py`, `tests/training/test_flops.py`**
  - In `tests/training/test_optim.py`:
    - `test_step2_loop_param_groups_and_no_decay_invariants` *(Review Focus #4)*: build `OplmForMaskedLM` with `decayed`, `channel` gate (`loop_gate_init=0.0`), `loop_untie_norms=True`, `sinusoidal`, `value_residual="learnable"` under both `optimizer="muon"` and `optimizer="adamw"` with `weight_decay_to_init=0.1`; assert exact group placement: untied norm weights decay toward `1.0`; `gates.*`, `log_delta`, `log_state_decay`, `loop_value_residual_lambdas.*`, and `input_matrix_continuous` are in AdamW with `weight_decay == 0.0`; `loop_step_cond.{fc1,fc2}.weight` and `concat_proj` (`proj.weight`) land in Muon when `optimizer="muon"`.
    - `test_muon_step0_and_step1_with_zero_init_loop_gate_and_step_cond` *(Review Focus #4)*: run 2 Muon optimizer steps with `loop_gate_init=0.0` + `loop_step_conditioning="sinusoidal"` + `weight_decay_to_init=0.1`; assert all weights finite, `gates[0]` moves away from `0.0` after step 1, and `fc1.weight` remains finite at step 0 (despite zero grad) and updates on step 2.
  - In `tests/training/test_mup.py`:
    - `test_mup_is_noop_at_base_width_with_step2_loop_extensions` *(Review Focus #2)*: verify bit-identical init, optimizer groups/LRs, and 1 step between `mup_enable=False` and `mup_enable=True, mup_base_width=hidden_size` for `concat_proj`, `decayed`, `orthogonal`, `loop_gate="channel"`, `loop_untie_norms=True`, `loop_step_conditioning="sinusoidal"`.
  - In `tests/training/test_flops.py`:
    - `test_step2_injection_flops`: verify `concat_proj` adds `3 * (K - 1) * 4 * h^2`, `decayed` adds `3 * (K - 1) * 2 * h^2`, and `orthogonal` / `none` add `0`.

- [ ] **Step 2: Run tests to verify RED**
  Run: `.venv/bin/python -m pytest tests/training/test_optim.py tests/training/test_mup.py tests/training/test_flops.py -k "step2 or loop_gate" -v`
  Expected: FAIL.

- [ ] **Step 3: Implement updates in `src/oplm/training/optim.py` and `src/oplm/training/flops.py`**
  - In `optim.py`: extend `_init_value` to recognize untied norm `.weight` suffixes (`.loop_attn_norms.`, `.loop_ffn_norms.`, `.loop_attn_post_norms.`, `.loop_ffn_post_norms.`); in `partition_optimizer_params`, intercept `name.endswith("loop_injector.input_matrix_continuous")` so it stays on AdamW with `weight_decay=0.0` (never Muon, never decayed).
  - In `flops.py`: add `(num_loops - 1)` injection matmul FLOPs when `loop_strategy == "stack"` and `num_loops >= 2`.

- [ ] **Step 4: Run tests to verify GREEN**
  Run: `.venv/bin/python -m pytest tests/training/test_optim.py tests/training/test_mup.py tests/training/test_flops.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/oplm/training/optim.py src/oplm/training/flops.py tests/training/test_optim.py tests/training/test_mup.py tests/training/test_flops.py
  git commit -m "feat(training): wire Step 2 loop extensions into optim, muP, and FLOPs"
  ```

---

### Task 5: Warm-start `train.init_from` and full-state resume guards

**Files:**
- Modify: `src/oplm/training/initialization.py`
- Modify: `src/oplm/training/checkpoint.py`
- Test: `tests/training/test_initialization.py`
- Test: `tests/training/test_loop_resume.py`

**Interfaces:**
- Consumes: `OplmConfig`, `OplmForMaskedLM` (Tasks 1–3).
- Produces:
  - `validate_initialization_config(source, target)` permitting enabling Step 2 extensions on unlooped/vanilla checkpoints while rejecting mode swaps, disabling enabled extensions, or changing `num_loops`/range on checkpoints that already carry per-pass modules (`loop_gate != "none"` or `loop_untie_norms=True`).
  - `load_initial_model(source, target)` permitting `missing_keys` strictly inside newly enabled loop-extension namespaces and copying pass-0 norm weights/biases + `value_residual_lambda` into `loop_*` copies when `loop_untie_norms` transitions `False -> True`.
  - `validate_loop_resume_compat(checkpoint_dir, cfg)` checking all 5 new loop fields alongside the existing 4 (with legacy defaults when absent).

- [ ] **Step 1: Write failing tests in `tests/training/test_initialization.py` and `tests/training/test_loop_resume.py`**
  - In `tests/training/test_initialization.py`:
    - `test_warm_start_unlooped_to_step2_extensions_succeeds_and_copies_norms`: save unlooped `OplmForMaskedLM(norm_strategy="sandwich", value_residual="learnable")`; load via `load_initial_model` into target with `num_loops=3, loop_start=1, loop_end=3, loop_input_injection="decayed", loop_gate="channel", loop_gate_init=0.0, loop_untie_norms=True, loop_step_conditioning="sinusoidal"`; assert all base weights match source, every `loop_*norms[p]` equals its pass-0 norm, every `loop_value_residual_lambdas[p]` equals pass-0 `value_residual_lambda`, and forward logits at step 0 equal source logits identically.
    - `test_warm_start_rejects_corrupt_missing_base_key_even_with_step2_extensions`: delete `oplm.backbone.layers.1.ffn.down_proj.weight` from source safetensors and verify `load_initial_model` into a Step 2 target still raises `ValueError`.
    - `test_warm_start_rejects_invalid_step2_transitions`: parametrize (a) mode swap (`decayed -> concat_proj`), (b) disabling enabled extension (`decayed -> none`, `loop_untie_norms: True -> False`), (c) changing `num_loops` (`2 -> 3`) or `loop_start` (`1 -> 0`) when `source` already has `loop_gate="scalar"` or `loop_untie_norms=True` — all raise `ValueError`.
  - In `tests/training/test_loop_resume.py`:
    - `test_resume_rejects_step2_loop_extension_drift`: parametrize changes to each of the 5 new fields across both `"yaml"` and `"hf"` artifacts — all raise `ValueError` mentioning the drifted field.
    - `test_legacy_checkpoint_without_step2_fields_resumes_vanilla_and_rejects_extensions`: legacy checkpoint lacking the 5 new keys resumes cleanly against vanilla `num_loops=2` and raises against `loop_input_injection="decayed"`.

- [ ] **Step 2: Run tests to verify RED**
  Run: `.venv/bin/python -m pytest tests/training/test_initialization.py tests/training/test_loop_resume.py -k "step2 or warm_start" -v`
  Expected: FAIL.

- [ ] **Step 3: Implement updates in `src/oplm/training/initialization.py` and `src/oplm/training/checkpoint.py`**
  - `initialization.py`: add the 5 fields to `_IGNORED_INITIALIZATION_FIELDS`; add `_validate_loop_extension_transition(source, target)` inside `validate_initialization_config`; add `_allowed_new_loop_keys(source, target, model)` in `load_initial_model` to filter only newly enabled extension keys out of `missing_keys` (failing on any other missing, unexpected, or mismatched key); when `not source.loop_untie_norms and target.loop_untie_norms`, copy pass-0 norm state dicts and `value_residual_lambda` values into `loop_*` copies under `torch.no_grad()`.
  - `checkpoint.py`: add the 5 new fields to `signature(model)` and `ordinary` in `validate_loop_resume_compat`.

- [ ] **Step 4: Run tests to verify GREEN**
  Run: `.venv/bin/python -m pytest tests/training/test_initialization.py tests/training/test_loop_resume.py tests/training/test_e2e_looping.py -q`
  Expected: PASS.

- [ ] **Step 5: Commit**
  ```bash
  git add src/oplm/training/initialization.py src/oplm/training/checkpoint.py tests/training/test_initialization.py tests/training/test_loop_resume.py
  git commit -m "feat(training): support Step 2 warm-start init_from and loop resume guards"
  ```

---

### Task 6: Documentation, `trust_remote_code` round-trip, and full verification

**Files:**
- Modify: `docs/config.md`
- Modify: `docs/architecture.md`
- Modify: `docs/training.md`
- Test: `tests/model/test_push_to_hub.py`
- Test: `tests/test_docs.py`

**Interfaces:**
- Consumes: complete Step 2 implementation (Tasks 1–5).

- [ ] **Step 1: Add clean-environment `trust_remote_code` test for Step 2 extensions in `tests/model/test_push_to_hub.py`** *(Review Focus #3)*
  - Add a test saving an `OplmForMaskedLM` with Step 2 extensions enabled (`loop_input_injection="decayed"`, `loop_gate="channel"`, `loop_untie_norms=True`, `loop_step_conditioning="sinusoidal"`) and loading it via `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` in the clean-environment round-trip helper, verifying logits match.

- [ ] **Step 2: Update docs (`docs/config.md`, `docs/architecture.md`, `docs/training.md`)**
  - `docs/config.md` (*Shared-layer looping*): add the 5 new `model.loop_*` fields with types, defaults, and validation rules.
  - `docs/architecture.md` (*Shared-layer looping*): document prelude–core–coda boundary flow, input injection modes, per-cycle gating, untied pass norms, and sinusoidal step conditioning.
  - `docs/training.md` (*Training a separate looped stage*): note warm-starting an unlooped checkpoint into Step 2 extensions via `train.init_from` (including zero-init `model.loop_gate_init=0.0` and pass-0 norm copy) and full-state resume checks.

- [ ] **Step 3: Run full verification suite**
  Run:
  ```bash
  .venv/bin/python -m pytest -m "not slow" -q && ruff check src/ && ruff format --check src/ && ty check src/
  ```
  Expected: 0 failures, 0 lint errors, 0 formatting diffs, 0 type errors.

- [ ] **Step 4: Commit docs & remote-code test**
  ```bash
  git add docs/config.md docs/architecture.md docs/training.md tests/model/test_push_to_hub.py
  git commit -m "docs(model): document Step 2 looping extensions and verify trust_remote_code"
  ```
