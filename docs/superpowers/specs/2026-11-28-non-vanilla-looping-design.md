# Non-Vanilla Shared-Layer Looping (Step 2: Boundary Primitives & Pass Symmetry Breaking)

**Status:** Draft (awaiting approval)
**Board cards:** #142 (input injection), #143 part 1 (per-cycle residual gating), #144 part 1 (per-pass norm untying & step conditioning)
**Related:** #141 (prelude–core–coda layout), #145 (from-scratch looped pretraining), #98 (elastic-K decay), #99 (effective size)

---

## Intent and Why

OPLM's current shared-layer looping (`OplmStack`, `docs/methods/evidence/looping/README.md`) re-executes a contiguous slice `[loop_start, loop_end)` of physical `OplmBlock`s and passes state across passes strictly through the residual stream `x`. On 170M decay-only warm starts, `stack2` gave modest gains (`-0.010` to `-0.014` `uniref70_gte2` CE, `+0.003` to `+0.010` P@L) and suffered three structural limitations:

1. **Information dilution / drift across passes (#142):** Pass `k >= 1` receives only the previous pass output `z_{k-1}` with no anchor back to the prelude representation `e`. Recent recursive-depth work—specifically `recAMPLIFY` (Oct 2026, recursive protein MLM) and *Towards Looped Models Done Right* (Parts I–II)—shows prelude input injection at every core entry stabilizes trajectories and improves contact/retrieval quality at negligible parameter cost.
2. **Undamped loop updates and warm-start shock (#143 part 1):** Applying a second pass to an unlooped checkpoint immediately spikes held-out CE (`2.04 -> 2.42` in `loop_count_eval.csv`). A per-cycle residual gate `z_k = z_{k-1} + g_k * (core_out - z_{k-1})` bounds pass updates during scratch training and—when initialized to `0.0`—makes warm-starting from an unlooped checkpoint an exact identity at step 0.
3. **100% blind weight tying (#144 part 1):** Pass 1 (local motifs/syntax) and pass 2+ (global contacts) share identical sandwich norms and ResFormer `value_residual_lambda` scalars despite distinct input activation distributions. Untying per-pass norms/lambdas or adding sinusoidal step FiLM conditioning breaks pass symmetry for `<0.1%` extra parameters.

The overarching goal is stronger protein representations and contact precision per unique parameter for the memory-bound `oplm.fold` head (`LanguageModelShim`).

---

## Goals, Non-Goals, and Anti-Goals

### Goals
- Expose explicit prelude / core-pass / coda execution structure from `src/oplm/model/looping.py` without changing `resolve_layer_execution_order`'s return signature or semantics.
- Add three composable, default-off model extensions behind `OplmConfig`:
  1. **Prelude input injection** (`loop_input_injection`: `"none" | "concat_proj" | "decayed" | "orthogonal"`).
  2. **Per-cycle residual gate** (`loop_gate`: `"none" | "scalar" | "channel"`, `loop_gate_init`: float).
  3. **Lightweight pass symmetry breaking**: `loop_untie_norms: bool` (per-pass norms + per-pass `value_residual_lambda` in looped blocks) and `loop_step_conditioning: "none" | "sinusoidal"` (sinusoidal `(pass_idx, num_loops)` FiLM modulation at core entry).
- Support both **from-scratch looped pretraining** and **strict-safe `train.init_from` warm-starting** from ordinary unlooped checkpoints (initializing new loop modules cleanly while still failing on any unexpected or non-loop missing keys).
- Preserve exact compatibility with `torch.compile`, full and selective activation checkpointing, DDP/HSDP, μP width scaling (`mup_enable`), HF `save_pretrained`/`from_pretrained` (including `trust_remote_code=True`), `LanguageModelShim` hidden-state counts, and resume compatibility guards.

### Non-Goals (deferred to follow-on PRs)
- **Step 3 (#143 part 2, #98):** Dynamic per-step `num_loops`, Ouro per-cycle MLM loss, `recAMPLIFY` GMR halting head / probability-weighted mixture readout, and rank-synced elastic-`K` training in `Trainer`.
- **Step 4 (#144 part 2):** Per-pass low-rank adapters (depth-wise LoRA) inside attention/FFN projections.
- Cross-loop KV attention (`MoLT-R`) or token-level routing (`MoR`).
- Supporting loop extensions under `loop_strategy="interleave"` (extensions act at core-pass boundaries, which only exist for `loop_strategy="stack"`).

### Anti-Goals
- **Zero regression at defaults:** When all new fields stay at their defaults (`loop_input_injection="none"`, `loop_gate="none"`, `loop_untie_norms=False`, `loop_step_conditioning="none"`), state-dict keys, parameter counts, initialization, forward outputs, and gradients must be bit-identical to `main`.
- **No silent resume drift:** `validate_loop_resume_compat` must reject any change to the new loop fields on full-state resume.
- **No accidental weight-decay collapse:** `train.weight_decay_to_init` must not pull `loop_gate` (especially at `loop_gate_init=0.0`), `decayed` dynamics (`log_delta`, `log_state_decay`), `orthogonal` scale, or untied `value_residual_lambda` toward zero/init.

---

## Approaches Considered

1. **Approach A (Selected) — Modular loop-boundary primitives + lightweight pass symmetry breaking (`#142` + `#143` gating + `#144` norms/step conditioning):**
   - Places input injection, step conditioning, and cycle gating at the core boundary in `OplmStack` (backed by a small `src/oplm/model/loop_modules.py`), and keeps `OplmBlock` / `OplmAttention` edits minimal (optional extra pass norms + `value_residual_lambda` while keeping pass-0 keys unchanged).
   - *Trade-off:* Clean, minimal, independently ablatable levers; defers dynamic-`K` / GMR supervision and LoRA to follow-on PRs.
2. **Approach B (Deferred) — Full `recAMPLIFY` GMR + dynamic-`K` training in one PR:**
   - Couples model changes with `Trainer` (dynamic per-step `K`, rank-synced RNG, multi-graph compile, GMR mixture loss). Harder to isolate which architectural lever drove any gain or instability.
3. **Approach C (Deferred) — Warm-start-first depth-wise LoRA only:**
   - Adds ~2–5% params and locks to fixed `K` without fixing input anchoring or cycle damping. Better evaluated in Step 4 on top of Approach A.

---

## Detailed Design

### 1. Configuration Surface (`OplmConfig`, `src/oplm/configs/model/base.yaml`)

All new keyword-only fields on `OplmConfig` (with matching defaults in `src/oplm/configs/model/base.yaml`):

| Field | Type | Default | Valid values / semantics |
|---|---|---|---|
| `loop_input_injection` | `str` | `"none"` | `"none"`, `"concat_proj"`, `"decayed"`, `"orthogonal"` |
| `loop_gate` | `str` | `"none"` | `"none"`, `"scalar"`, `"channel"` |
| `loop_gate_init` | `float` | `1.0` | Finite float; `0.0` yields exact identity on passes `k >= 1` for warm starts |
| `loop_untie_norms` | `bool` | `False` | Untie block norms and `value_residual_lambda` per pass in `[loop_start, loop_end)` |
| `loop_step_conditioning` | `str` | `"none"` | `"none"`, `"sinusoidal"` |

**Validation rules (`OplmConfig._validate` & `OplmStack.__init__`):**
- Enum checks for `loop_input_injection`, `loop_gate`, `loop_step_conditioning`; finite float check for `loop_gate_init`; strict `bool` for `loop_untie_norms`.
- Let `any_loop_ext = (loop_input_injection != "none") or (loop_gate != "none") or loop_untie_norms or (loop_step_conditioning != "none")`.
- If `any_loop_ext` and `loop_strategy != "stack"`: raise `ValueError` ("loop extensions require loop_strategy='stack'").
- **Builder call — `num_loops == 1` behavior:** Allow `any_loop_ext` with `num_loops == 1` to raise `ValueError`? **No — allow `num_loops == 1` ONLY when `any_loop_ext` is False; reject `any_loop_ext` when `num_loops < 2` at config validation.** Rationale: avoids creating dead, unused parameters (which would break DDP unused-parameter checks!) when `num_loops == 1`. Eval-time `K=1` sweeps on a trained `num_loops >= 2` checkpoint are performed by passing an optional runtime override or slicing passes in eval—not by mutating `num_loops=1` at construction. *(Call made here: reject `any_loop_ext` when `num_loops < 2` to keep DDP strictly free of unused parameters.)*

### 2. Execution Schedule (`src/oplm/model/looping.py`)

Keep `resolve_layer_execution_order(...) -> tuple[int, ...]` completely unchanged. Add a pure helper:

```python
@dataclass(frozen=True)
class LoopSchedule:
    prelude: tuple[int, ...]     # range(0, loop_start)
    core: tuple[int, ...]        # range(loop_start, end)
    coda: tuple[int, ...]        # range(end, num_hidden_layers)
    num_loops: int
    loop_strategy: str
    execution_order: tuple[int, ...]
```

`resolve_loop_schedule(...) -> LoopSchedule` validates via `resolve_layer_execution_order` and returns the explicit partition.

### 3. Forward Dataflow (`OplmStack.forward` in `src/oplm/model/transformer.py`)

When `not any_loop_ext` (or `loop_strategy != "stack"`), `OplmStack.forward` executes the exact existing loop untouched.

When `loop_strategy == "stack"` and `num_loops >= 2`:
1. **Prelude:** Run physical blocks `idx in schedule.prelude` once (passing `pass_idx=0`). Let `e = x` be the prelude output (if `loop_start == 0`, `e` is the post-embedding state `x`).
2. **Core pass `k = 0`:** Run physical blocks `idx in schedule.core` with `pass_idx=0` directly on `e` -> `z`.
   - Pass 0 intentionally runs un-injected, un-conditioned, and un-gated so that pass 0 matches an unlooped forward pass verbatim (critical for warm-starting via `train.init_from` and for clean state initialization).
3. **Core passes `k = 1, ..., num_loops - 1`:**
   - Save `z_prev = z`.
   - **Input injection:** `z_in = self.loop_injector(z_prev, e)` (identity if `loop_input_injection == "none"`).
   - **Step conditioning:** `z_in = self.loop_step_cond(z_in, k, num_loops)` (identity if `loop_step_conditioning == "none"`).
   - **Core blocks:** Run physical blocks `idx in schedule.core` on `z_in` with `pass_idx=k` -> `core_out`.
   - **Cycle residual gate:** `z = self.loop_gate_module(z_prev, core_out, k - 1)` (returns `core_out` if `loop_gate == "none"`).
   - *Hidden-state tap / `output_hidden_states` semantics:* Record each executed block's output as today; on the last block of pass `k >= 1`, record `z` (the post-cycle-gate state that actually feeds pass `k + 1` or the coda), preserving `len(hidden_states) == len(layer_execution_order) + 1` for `LanguageModelShim` and `StabilityDiagnosticsCallback`.
4. **Coda:** Run physical blocks `idx in schedule.coda` once (with `pass_idx=0`), then `self.final_norm(x)`.

### 4. Loop Extension Modules (`src/oplm/model/loop_modules.py`)

A new self-contained module imported directly in `configuration_oplm.py` and `modeling_oplm.py` (`_REMOTE_CODE_DEPS`) for `trust_remote_code=True` bundling.

#### 4a. Input Injection (`LoopInputInjector`)
- **`"concat_proj"`** (Huginn / `recAMPLIFY` style):
  - `self.z_norm = make_norm(norm_type, D, eps)`, `self.e_norm = make_norm(norm_type, D, eps)`.
  - `self.proj = nn.Linear(2 * D, D, bias=False)` (2-D hidden weight; participates in Muon/AdamW and μP).
  - Forward: `h = torch.cat([self.z_norm(z), self.e_norm(e)], dim=-1) * (2.0 ** -0.5)`, return `self.proj(h)`.
  - *μP fan-in:* Update `OplmConfig.mup_fanin_mult(fan_in)` so `fan_in == 2 * self.hidden_size` returns `self.mup_width_mult()` (since the `2**-0.5` factor already normalizes the doubled input variance and keeps `m_W == 1.0` at `hidden_size == mup_base_width`).
- **`"decayed"`** (`oplm.fold.trunk.Recurrence` style):
  - `self.input_norm = make_norm(norm_type, D, eps)`.
  - `self.log_delta = nn.Parameter(torch.empty(D))`, `self.log_state_decay = nn.Parameter(torch.empty(D))`, `self.input_matrix_continuous = nn.Parameter(torch.empty(D, D))`.
  - Init (guarded with `_skip_generic_init = True` / direct parameter init so `_init_weights` never overwrites it): `delta0 = 0.5 * ln(5)`, `log_delta = inv_softplus(delta0)`, `log_state_decay = 0`, `input_matrix_continuous = I_D`.
  - Forward (dynamics computed in fp32, cast to `z.dtype`): `delta = F.softplus(self.log_delta.float())`, `a = torch.exp(-delta * torch.exp(self.log_state_decay.float()))`, `B = delta[:, None] * self.input_matrix_continuous.float()`, return `a.to(z.dtype) * z + F.linear(self.input_norm(e).to(z.dtype), B.to(z.dtype))`.
- **`"orthogonal"`** (*Towards Looped Models Done Right, Part II*):
  - `self.inject_scale = nn.Parameter(torch.ones(D))` (learnable per-channel injection scale, initialized to `1.0`).
  - Forward (projection computed in fp32 for bf16 numerical stability, cast back to `z.dtype`):
    ```python
    z_f, e_f = z.float(), e.float()
    e_hat = e_f / e_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    z_orth = z_f - (z_f * e_hat).sum(dim=-1, keepdim=True) * e_hat
    out = z_orth + self.inject_scale.float() * e_f
    return out.to(z.dtype)
    ```

#### 4b. Per-Cycle Residual Gate (`LoopCycleGate`)
- Parameters: `self.gates = nn.ParameterList([nn.Parameter(torch.full((1 if mode == "scalar" else D,), float(init_val))) for _ in range(num_loops - 1)])`.
- Storing each pass's gate as a **1-D** `nn.Parameter` inside an `nn.ParameterList` (matching `OplmBlock.attn_gate`/`ffn_gate` and `attention.loop_value_residual_lambdas`) guarantees every gate lands in AdamW no-decay with `lr_mult = 1.0` automatically, avoids Muon orthogonalization / 2-D μP fan-in scaling bugs, and avoids 2-D `(1, 1)` FSDP2 sharding edge cases.
- Forward for pass `k >= 1` (index `idx = k - 1`):
  `g = self.gates[idx].to(z_prev.dtype)`
  `return z_prev + g * (core_out - z_prev)`
- At `loop_gate_init = 0.0`, output is `z_prev` identically on step 0 while gradients flow cleanly into `self.gates` and the core on step 1.

#### 4c. Sinusoidal Step Conditioning (`LoopStepConditioner`, when `loop_step_conditioning == "sinusoidal"`)
- Computes a fixed sinusoidal encoding of normalized progress `t = pass_idx / max(num_loops - 1, 1)` at width **`D` (`hidden_size`)** (computed in fp32, cast to `z.dtype`; if `D` is odd, pad one zero channel) -> `Linear(D, D)` -> `SiLU` -> `Linear(D, 2 * D)` (last linear **zero-initialized** with `_init_zero = True` so at step 0 `gamma = 0, beta = 0`, exact identity).
- Using encoding width `D` ensures both `fc1` and `fc2` have hidden fan-in `D`, so `OplmConfig.mup_fanin_mult(D) == m` (`1.0` at `hidden_size == mup_base_width`) and both layers obey standard μP init and Muon/AdamW routing with zero special cases.
- Applies residual FiLM at core entry: `self.norm = make_norm(norm_type, D, eps)`, `return z + gamma * self.norm(z) + beta`.
- *Why residual FiLM (`z + gamma * norm(z) + beta`) instead of `(1 + gamma) * norm(z) + beta`:* Keeps `z` untouched at step 0 of warm starts even when `z` has grown to `O(sqrt(L))` norm.

#### 4d. Untied Per-Pass Norms & Value-Residual Lambdas (`loop_untie_norms = True`)
- Applies only to physical blocks with `loop_start <= layer_idx < loop_end` (prelude and coda blocks run once and keep zero extra modules).
- **Key-preserving layout:** Pass 0 continues to use the existing attribute names (`self.attn_norm`, `self.attn_post_norm`, `self.ffn_norm`, `self.ffn_post_norm`, and `attention.value_residual_lambda`). Extra passes `1 .. num_loops - 1` live in `nn.ModuleList`s of length `num_loops - 1`:
  - `self.loop_attn_norms` (when `norm_strategy != "hybrid"`)
  - `self.loop_ffn_norms`
  - `self.loop_attn_post_norms` (when `norm_strategy in {"sandwich", "post_sdpa"}`)
  - `self.loop_ffn_post_norms` (when `norm_strategy == "sandwich"`)
  - `attention.loop_value_residual_lambdas`: `nn.ParameterList` of length `num_loops - 1` when `value_residual == "learnable"` and `layer_idx > 0` (initialized to `value_residual_lambda_init`).
  - *(Call made here: keep `q_norm`, `k_norm`, `v_norm` shared across passes; untying only the outer pre/post sandwich norms + `value_residual_lambda` captures the residual-stream scale shift between passes while keeping `OplmAttention` simple.)*
- `OplmBlock.forward(..., pass_idx: int = 0)` selects pass-0 modules when `pass_idx == 0` (or `not loop_untie_norms`) and `loop_*[pass_idx - 1]` when `pass_idx >= 1`. `pass_idx` is a static Python `int` inside `OplmStack.forward`'s unrolled loop, so `torch.compile` and `torch.utils.checkpoint.checkpoint` trace it cleanly with zero dynamic graph breaks.

### 5. Training, Optimizer, Init, Resume, and FLOPs Integration

- **Optimizer grouping (`src/oplm/training/optim.py`):**
  - `_init_value(name, model_config)`:
    - Match both `name.endswith("norm.weight")` and untied norm list entries (`".loop_attn_norms."`, `".loop_ffn_norms."`, `".loop_attn_post_norms."`, `".loop_ffn_post_norms."` ending with `".weight"` -> `1.0`; `".bias"` already returns `0.0`).
    - Explicitly return `None` (never decayed toward init) for `loop_gate_module.gates.*`, `loop_injector.log_delta`, `loop_injector.log_state_decay`, `loop_injector.inject_scale`, and `loop_value_residual_lambdas.*`.
  - 2-D weights:
    - `loop_injector.proj.weight` (`concat_proj`, shape `(D, 2D)`) and `loop_step_cond` linear weights: standard Muon/AdamW hidden 2-D routing.
    - `loop_injector.input_matrix_continuous` (`decayed`, shape `(D, D)`, identity-initialized like `oplm.fold`'s `Recurrence`): keep on **AdamW** (not Muon, whose Newton-Schulz orthogonalization + decay toward 0 fights identity-anchored continuous-time dynamics) with `weight_decay = 0.0` (or small `weight_decay` — keep `0.0` matching `oplm.fold` stability, explicitly documented).
  - `_depth_lr_multiplier`: untied block norms live under `oplm.backbone.layers.*` and inherit the block depth multiplier automatically; stack-level `loop_injector` / `loop_gate_module` / `loop_step_cond` live under `oplm.backbone.*` and take `1.0`.
- **`_init_weights` (`src/oplm/model/modeling_oplm.py`):**
  - Honor `_init_zero = True` on `nn.Linear` (zeros weight & bias, used by `LoopStepConditioner`'s output projection).
  - `decayed` and `orthogonal` parameters are direct `nn.Parameter` attributes (not `nn.Linear`), so `_init_weights` already leaves their custom inits untouched; call `reset_parameters()` only on construction / unloaded tensors so `from_pretrained` never overwrites loaded weights.
- **Warm-start initialization (`src/oplm/training/initialization.py`):**
  - Add the five new loop extension fields to `_IGNORED_INITIALIZATION_FIELDS` so `train.init_from` can warm-start a Step 2 model from an ordinary unlooped (or vanilla looped) checkpoint.
  - Validate extension transitions explicitly in `validate_initialization_config(source, target)`:
    - Allowed: enabling an extension that was `"none"` / `False` in `source`, or keeping the same active extension configuration.
    - Rejected (`ValueError`): changing an already-enabled extension's mode (e.g., `decayed -> concat_proj`), disabling an enabled extension, or changing `num_loops` / `[loop_start, loop_end)` when `source` already owns per-pass parameters (`loop_gate != "none"` or `loop_untie_norms=True`).
  - When `missing_keys` occur during `load_initial_model`, allow missing keys **if and only if** every missing key belongs to an extension newly enabled in `target` (i.e., `"none"` / `False` in `source`): `oplm.backbone.loop_injector.*`, `oplm.backbone.loop_gate_module.*`, `oplm.backbone.loop_step_cond.*`, or `oplm.backbone.layers.*.loop_*norms.*` / `*.loop_value_residual_lambdas.*`. When `loop_untie_norms` transitions `False -> True`, copy each looped block's loaded pass-0 norm weights/biases and `value_residual_lambda` into its newly created `loop_*` copies. Any other missing, unexpected, or mismatched key still raises `ValueError`.
- **Full-state resume (`src/oplm/training/checkpoint.py`):**
  - Extend `validate_loop_resume_compat` signature dict with all five new fields (with ordinary legacy defaults `("none", "none", 1.0, False, "none")` when loading older checkpoints missing those keys). Any drift raises `ValueError`.
- **FLOPs accounting (`src/oplm/training/flops.py`):**
  - When `loop_strategy == "stack"` and `num_loops >= 2`, add `(num_loops - 1)` core-entry injection matmul FLOPs: `2 * (2 * D) * D` for `concat_proj` and `2 * D * D` for `decayed` (times 3 for fwd+bwd).
- **Documentation (`docs/architecture.md`, `docs/config.md`, `docs/training.md`):**
  - Document the five new config fields, their defaults, and warm-start semantics in the same PR so `tests/test_docs.py` stays green.

---

## Decisions Made by Designer (for quick confirmation)

1. Reject non-default loop extensions when `num_loops < 2` or `loop_strategy != "stack"` (prevents unused parameters under DDP).
2. Pass 0 runs un-injected, un-conditioned, and un-gated; all extensions fire on passes `k = 1 .. num_loops - 1` (gives exact identity warm-start at `loop_gate_init = 0.0`).
3. `loop_gate` is static per-pass (`scalar` / `channel`) stored as a 1-D `nn.ParameterList` (automatically AdamW no-decay + `lr × 1` + FSDP2-safe); input-dependent gating + halting stays with GMR in Step 3.
4. `loop_untie_norms` unties outer pre/post block norms + `value_residual_lambda` while keeping `q_norm`/`k_norm`/`v_norm` shared, and preserves pass-0 state-dict keys unchanged.
5. `decayed.input_matrix_continuous` stays on AdamW (not Muon) so orthogonalization/decay toward 0 doesn't destabilize its identity-anchored continuous-time dynamics.
6. `LoopStepConditioner` uses width-`D` sinusoidal encoding (`Linear(D, D) -> SiLU -> Linear(D, 2D)`) with zero-initialized residual FiLM (`z + gamma * norm(z) + beta`) so it is an exact identity at step 0 and exact μP no-op at base width.
7. `train.init_from` supports enabling Step 2 extensions on an unlooped/vanilla checkpoint (copying pass-0 norms/lambdas on `loop_untie_norms: False -> True`) and rejects mutating `num_loops`/range or swapping modes on checkpoints that already carry per-pass Step 2 modules.

---

## Left to the Builder

- Exact internal helper decomposition inside `src/oplm/model/loop_modules.py` and unit-test matrix organization (`tests/model/test_looping.py`, `tests/training/test_loop_resume.py`, `tests/training/test_initialization.py`, `tests/training/test_optim.py`, `tests/model/test_save_load.py`, `tests/model/test_push_to_hub.py`).
- Exact `torch.compile` + gradient-checkpointing (full + selective) verification tests on CPU/mock paths.
