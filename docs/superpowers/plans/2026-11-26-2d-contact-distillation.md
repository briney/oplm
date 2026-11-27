# 2D Contact-Pair Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional 2D Cβ contact-pair distillation from precomputed MSA Pairformer contact maps under `distill:`, supporting both `"attention"` (upper-layer post-RoPE `(Q, K)` maps) and `"pair"` (hidden-state symmetric pair MLP) discarded heads with zero inference-checkpoint changes.

**Architecture:** Precomputed upper-triangle `uint8` teacher Cβ contact probabilities (4-byte self-describing header `[version=1, stored_min_sep, stored_len_u16]`) stream through `ShardedProteinDataset` and `MLMCollator` as length-sorted segment metadata plus one flat `uint8` tensor. `OplmAttention`/`OplmBlock`/`OplmStack`/`OplmForMaskedLM` expose an opt-in, parameter-free `return_qk_layers` tap so `FlexAttention`, `torch.compile(dynamic=False)`, and selective AC stay intact; `DistillMLMTask` owns the discarded 2D contact head, its standalone branch-safe AdamW optimizer (synced across ranks via `all_reduce`), and an optional `distill_aux.pt` checkpoint sidecar mirroring `ema.pt`.

**Tech Stack:** Python 3.11+, PyTorch 2.x (FlexAttention, `torch.compile`, DDP/HSDP, DCP), HuggingFace Transformers, PyArrow, OmegaConf, pytest, ruff, ty.

**Spec:** `docs/superpowers/specs/2026-11-26-2d-contact-distillation-design.md`

## Global Constraints

- Python 3.11+, `from __future__ import annotations` in every file, type hints on all signatures, Google-style docstrings on public APIs, `pathlib.Path` (never `os.path`), 100-char line length.
- `ruff check src/`, `ruff format --check src/`, `ty check src/`, and `.venv/bin/python -m pytest -m "not slow" -q` must all pass cleanly.
- Never materialize `(B, H, 2048, 2048)` dense packed attention matrices; 2D contact computations run strictly per distilled segment in length-sorted mini-chunks.
- Never add trainable parameters to `OplmForMaskedLM` or leak auxiliary contact-head weights into `hf/` exports or DCP model/optimizer state.
- Preserve exact backward compatibility when `distill.contact_weight == 0.0` (default) and when `distill` is `None`.
- Update `docs/config.md`, `docs/data.md`, and `docs/training.md` in the same PR so `tests/test_docs.py` stays green.

## Review Focus

- **Empty contact batch (`M = 0`) under distributed training:** A rank receiving zero distilled contact segments in a micro-batch (or whole accumulation window) must produce zero 2D loss, zero-fill `None` aux grads before `all_reduce`, never deadlock, and never emit `NaN` (`Task 4`, `Task 5`).
- **Truncated long sequences & larger runtime `contact_min_sep`:** Chains with `stored_len > max_length - 2` or `contact_min_sep > stored_min_sep` must slice the exact upper-triangle sub-matrix for `(res_len, res_len)` without offset drift, and chains with `res_len <= contact_min_sep` must be cleanly skipped (`Task 2`).
- **`branch_from` from non-2D checkpoint & `auto_resume` round-trip:** Branching from a checkpoint without `distill_aux.pt` must restore backbone + optimizer state cleanly, initialize `aux_start_step = global_step` (200-step branch-local warmup × live backbone schedule decay), write clean `hf/` exports without aux weights, and restore `distill_aux.pt` + `aux_start_step` exactly on `auto_resume` (`Task 5`).
- **Packed rows with RoPE, `qk_norm_mode="l2"`, and `num_loops > 1`:** `return_qk_layers` indexes executed passes (`0..D-1`), leaves `FlexAttention`/SDPA outputs bit-for-bit unchanged, and `AttentionContactHead` uses per-layer `attn_scale` (`1.0` for L2 QK-norm, `1/sqrt(d_head)` otherwise) with masked `[<cls>, res..., <eos>]` softmax and exact masked APC matching single-sequence `apply_apc` (`Task 3`, `Task 4`).
- **Mixed sources, shared Neff gate, and pure-2D shards without `teacher_logits`:** In `InterleavedDataset` mixes, non-distilled sources missing `teacher_contacts` yield `None`; pure-2D runs (`alpha=0, contact_weight>0`) work on shards lacking `teacher_logits`; failing `min_neff` / `min_neff_sqrt_l` strips *both* `teacher_logits` and `teacher_contacts`; `sample_members` rejects shards with `teacher_contacts` (`Task 2`).

---

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `src/oplm/config.py` | Modify | Add 9 2D fields & validation to `DistillConfig`; validate `contact_layers <= effective_depth` in `validate_distill_compat`. |
| `src/oplm/data/sequence/dataset.py` | Modify | `TEACHER_CONTACTS_COLUMN`, `load_teacher_contacts`, shared Neff gating, `sample_members` guard. |
| `src/oplm/data/sequence/collate.py` | Modify | `encode_teacher_contacts` / `decode_teacher_contacts` (4-byte v1 header), `MLMCollator` length-sorted `teacher_contact_meta` + flat `uint8` `teacher_contact_values` (unpacked & packed). |
| `src/oplm/data/sequence/loaders.py` | Modify | Wire `load_teacher_contacts`, `contact_min_sep`, and pure-2D `load_teacher_logits` policy in `build_train_dataloader`. |
| `src/oplm/model/attention.py` | Modify | Opt-in `return_qk: bool = False` returning post-RoPE `(q, k)` right before FlexAttention/SDPA. |
| `src/oplm/model/transformer.py` | Modify | Thread `return_qk` through `OplmBlock` (incl. full/selective AC) and `return_qk_layers` through `OplmStack`. |
| `src/oplm/model/modeling_oplm.py` | Modify | Add `OplmBaseModelOutput`, `OplmMaskedLMOutput`, and opt-in `return_qk_layers` on `OplmModel` & `OplmForMaskedLM`. |
| `src/oplm/training/contact_distill.py` | Create | `AttentionContactHead`, `PairContactHead`, masked APC, length-sorted chunked soft-BCE loss, and `AuxContactOptimizer` (branch-safe LR + DDP/HSDP grad sync). |
| `src/oplm/training/task.py` | Modify | Wire `DistillMLMTask` 2D forward/loss + `prepare_aux` / `sync_aux_schedule` / `step_aux` / `aux_state_dict` / `load_aux_state_dict`. |
| `src/oplm/training/trainer.py` | Modify | Invoke optional task aux lifecycle hooks on init, resume/branch, optimizer step, and checkpoint save/load. |
| `src/oplm/training/checkpoint.py` | Modify | Write/restore optional `distill_aux.pt` sidecar mirroring `ema.pt`. |
| `src/oplm/training/remote.py` | Modify | Add `"distill_aux.pt"` to `_SHARED_ARTIFACT_NAMES`. |
| `docs/{config,data,training}.md` | Modify | Document new `distill.contact_*` config fields, `teacher_contacts` schema, and 2D training/sidecar behavior. |

---

### Task 1: `DistillConfig` 2D Fields & Cross-Section Depth Validation

**Files:**
- Modify: `src/oplm/config.py:620-706`
- Test: `tests/training/test_config.py`

**Interfaces:**
- Consumes: `oplm.model.looping.resolve_layer_execution_order`
- Produces: `DistillConfig` fields (`contact_weight: float = 0.0`, `contact_head: str = "attention"`, `contact_layers: int = 6`, `contact_min_sep: int = 6`, `contact_pos_weight: float = 1.0`, `contact_pair_dim: int = 128`, `contact_head_lr: float = 1e-3`, `contact_head_warmup_steps: int = 200`, `contact_max_chunk_pairs: int = 2_000_000`) and updated `validate_distill_compat(cfg: OplmConfig) -> None`.

- [ ] **Step 1: Write failing config validation tests in `tests/training/test_config.py`**

Add tests covering:
- `test_distill_config_contact_defaults`: default `DistillConfig()` has `contact_weight == 0.0`, `contact_head == "attention"`, `contact_layers == 6`, `contact_min_sep == 6`, `contact_pos_weight == 1.0`, `contact_pair_dim == 128`, `contact_head_lr == 1e-3`, `contact_head_warmup_steps == 200`, `contact_max_chunk_pairs == 2_000_000`.
- `test_distill_config_contact_validation_errors`: parametrized invalid values (`contact_weight=-0.1`, `contact_weight=float("nan")`, `contact_weight=True`, `contact_head="unknown"`, `contact_layers=0`, `contact_layers=True`, `contact_min_sep=0`, `contact_pos_weight=0.0`, `contact_pair_dim=8`, `contact_head_lr=0.0`, `contact_head_warmup_steps=-1`, `contact_max_chunk_pairs=1000`) all raise `ValueError`.
- `test_validate_distill_compat_contact_layers_depth`: `OplmConfig(model=ModelConfig(num_hidden_layers=4, num_loops=1), distill=DistillConfig(contact_weight=1.0, contact_layers=6))` raises `ValueError`; succeeds when `contact_weight=0.0` (disabled) or `num_loops=2` (`effective_depth=8 >= 6`).

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -k "contact" -v`  
Expected: FAIL (`TypeError: DistillConfig.__init__() got an unexpected keyword argument 'contact_weight'`).

- [ ] **Step 3: Implement `DistillConfig` 2D fields and `validate_distill_compat` depth check in `src/oplm/config.py`**

Add `_VALID_DISTILL_CONTACT_HEADS = ("attention", "pair")` and the 9 fields with strict bool/finite/bounds validation in `DistillConfig.__post_init__`. In `validate_distill_compat(cfg)`, when `cfg.distill is not None and cfg.distill.contact_weight > 0.0`, compute `effective_depth = len(resolve_layer_execution_order(...))` from `cfg.model` (handling both `dict` and resolved `OplmConfig` attributes) and raise `ValueError` if `cfg.distill.contact_layers > effective_depth`.

- [ ] **Step 4: Run config tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/training/test_config.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/config.py tests/training/test_config.py
git commit -m "feat(config): add 2D contact distillation knobs to DistillConfig"
```

---

### Task 2: `teacher_contacts` Binary Codec, Dataset Loading/Gating, and Packed Collation

**Files:**
- Modify: `src/oplm/data/sequence/dataset.py`
- Modify: `src/oplm/data/sequence/collate.py`
- Modify: `src/oplm/data/sequence/loaders.py`
- Test: `tests/data/sequence/test_dataset.py`, `tests/data/sequence/test_collate.py`, `tests/data/sequence/test_loaders.py`, `tests/data/sequence/test_member_sampling.py`

**Interfaces:**
- Consumes: `DistillConfig` contact fields from Task 1.
- Produces:
  - `TEACHER_CONTACTS_COLUMN = "teacher_contacts"` and `ShardedProteinDataset(..., load_teacher_contacts: bool = False)` in `dataset.py`.
  - `CONTACT_FORMAT_VERSION = 1`, `encode_teacher_contacts(probs: np.ndarray, *, min_sep: int = 6) -> bytes`, `decode_teacher_contacts(raw: bytes | bytearray | memoryview, raw_len: int, res_len: int, contact_min_sep: int) -> np.ndarray | None` in `collate.py`.
  - `MLMCollator(..., load_teacher_contacts: bool = False, contact_min_sep: int = 6)` emitting `batch["teacher_contact_meta"]: list[tuple[int, int, int, int]]` (`(row_idx, start_pos, res_len, value_offset)`, sorted by `res_len` ascending) and `batch["teacher_contact_values"]: Tensor` (`(total_pairs,)` `uint8`) when `load_teacher_contacts=True`.

- [ ] **Step 1: Write failing tests for codec, dataset gating/mixing, collation, and loader wiring**

Add tests covering:
- `tests/data/sequence/test_collate.py`:
  - `test_encode_decode_teacher_contacts_roundtrip`: exact round-trip of `(20, 20)` symmetric probs at `min_sep=6`; verifies 4-byte header `<BBH` (`version=1, min_sep=6, stored_len=20`) and `len(raw) == 4 + 14*15//2`.
  - `test_decode_teacher_contacts_truncated_and_larger_min_sep` (*Review Focus*): encodes `(30, 30)` at `stored_min_sep=6`, decodes with `res_len=18, contact_min_sep=6` AND with `res_len=30, contact_min_sep=12` AND with prefix-cropped `stored_len=20 < raw_len=30` (`res_len=20`); verifies bit-for-bit exact match against `triu_indices(res_len, k=contact_min_sep)` on the ground-truth matrix. Also checks `res_len <= contact_min_sep` returns `None`, and malformed version / wrong byte length / `stored_min_sep > contact_min_sep` / `stored_len > raw_len` raise `ValueError`.
  - `test_mlm_collator_contacts_unpacked_and_packed`: mixed batch (some rows with contacts, some `None`, varying lengths) with `pack_sequences=False` and `pack_sequences=True` (`pack_rows=2`); verifies `teacher_contact_meta` is sorted by `res_len` ascending, `(row_idx, start_pos)` points to each segment's exact `<cls>` in `packed["input_ids"]` (`input_ids[row, start] == cls_id`, `input_ids[row, start + res_len + 1] == eos_id`), and `teacher_contact_values[offset : offset + n_pairs]` matches decoded targets. Also verifies all-`None` batch emits `meta == []` and `values.shape == (0,)`.
- `tests/data/sequence/test_dataset.py` & `test_member_sampling.py`:
  - `test_dataset_load_teacher_contacts_and_shared_neff_gate` (*Review Focus*): row failing `min_teacher_neff_sqrt_l` strips BOTH `teacher_logits` and `teacher_contacts` to `None`; shards lacking `teacher_contacts` yield `None`; `sample_members="uniform"` raises `ValueError` on a shard with `teacher_contacts`.
- `tests/data/sequence/test_loaders.py`:
  - `test_build_train_dataloader_pure_2d_without_teacher_logits` (*Review Focus*): `distill=DistillConfig(alpha=0.0, contact_weight=1.0)` loads `teacher_contacts` cleanly from a shard that has NO `teacher_logits` column and skips 1D logit decoding (`"teacher_logits"` not in batch, `"teacher_contact_meta"` and `"teacher_contact_values"` in batch).

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/data/sequence/test_collate.py tests/data/sequence/test_dataset.py tests/data/sequence/test_loaders.py tests/data/sequence/test_member_sampling.py -k "contact" -v`  
Expected: FAIL (`ImportError` / unexpected keyword `load_teacher_contacts`).

- [ ] **Step 3: Implement codec, dataset loading/gating, collator packing, and loader wiring**

- In `src/oplm/data/sequence/dataset.py`: add `TEACHER_CONTACTS_COLUMN = "teacher_contacts"` and `load_teacher_contacts: bool = False`. Extend `_read_shard_columns`, Neff gating (strip both `teacher_logits` and `teacher_contacts` when gated out), and the `sample_members` guard (reject shards carrying either `teacher_logits` or `teacher_contacts`).
- In `src/oplm/data/sequence/collate.py`: implement `encode_teacher_contacts` and `decode_teacher_contacts` using `struct.pack("<BBH", 1, min_sep, stored_len)`. Fast path when `stored_len == res_len and stored_min_sep == contact_min_sep` slices `raw[4:]` zero-copy; general path computes 1D flat indices into the `(stored_len, stored_min_sep)` triangular layout: for row `i` in `0..res_len - contact_min_sep - 1`, row `i` starts at `row_start(i) = i * (stored_len - stored_min_sep) - i * (i - 1) // 2` (verify formula for `i=0, 1, 2`!), taking columns `j = i + contact_min_sep .. res_len - 1` at offsets `row_start(i) + (j - i - stored_min_sep)`.
  In `MLMCollator.__call__` & `_pack`: collect contact payloads per row, sort entries by `res_len` ascending, remap `(row_idx, start_pos)` during `_pack` (preserving `res_len` order after packing!), concatenate into one `uint8` tensor, and emit `teacher_contact_meta` and `teacher_contact_values`.
- In `src/oplm/data/sequence/loaders.py`: wire `load_teacher_contacts=(distill is not None and distill.contact_weight > 0.0)`, `contact_min_sep=distill.contact_min_sep if distill is not None else 6`, and `load_teacher_logits=(distill is not None and (distill.alpha > 0.0 or distill.contact_weight == 0.0))`.

- [ ] **Step 4: Run data test suite**

Run: `.venv/bin/python -m pytest tests/data/sequence/ -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/data/sequence/dataset.py src/oplm/data/sequence/collate.py src/oplm/data/sequence/loaders.py tests/data/sequence/
git commit -m "feat(data): add teacher_contacts v1 codec, Neff gating, and packed collation"
```

---

### Task 3: Opt-In `(Q, K)` Tap on `OplmAttention`, `OplmBlock`, `OplmStack`, `OplmModel`, and `OplmForMaskedLM`

**Files:**
- Modify: `src/oplm/model/attention.py:402-466`
- Modify: `src/oplm/model/transformer.py:238-501`
- Modify: `src/oplm/model/modeling_oplm.py:323-458`
- Create/Modify Test: `tests/model/test_qk_tap.py`

**Interfaces:**
- Consumes: existing `OplmAttention`, `OplmBlock`, `OplmStack`, `OplmModel`, `OplmForMaskedLM`.
- Produces:
  - `OplmBaseModelOutput(BaseModelOutput)` and `OplmMaskedLMOutput(MaskedLMOutput)` carrying `qk_states: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None`.
  - `OplmForMaskedLM.forward(..., return_qk_layers: tuple[int, ...] | None = None)` returning `OplmMaskedLMOutput` when `return_qk_layers is not None` and plain `MaskedLMOutput` otherwise.

- [ ] **Step 1: Write failing tests in `tests/model/test_qk_tap.py`**

Add tests covering:
- `test_return_qk_layers_parity_and_backward_compat`: calling `OplmForMaskedLM(..., return_qk_layers=None)` returns plain `MaskedLMOutput`; calling with `return_qk_layers=(2, 3)` returns `OplmMaskedLMOutput` (`isinstance(..., MaskedLMOutput)`), `len(out.qk_states) == 2`, each `(q, k)` has shape `(B, H, T, d_head)`, and `out.logits` is bit-for-bit identical to `return_qk_layers=None` (both unpacked and packed `segment_ids`).
- `test_return_qk_layers_matches_manual_attention_scores` (*Review Focus*): under `qk_norm_mode="channel"` AND `qk_norm_mode="l2"` AND `canon_enabled=True` (`["A", "B"]`) AND packed `segment_ids`, verifies `softmax((q_seg @ k_seg.T) * attn_scale)` from `out.qk_states` matches `output_attentions=True`'s manual attention weights on that segment (`atol=1e-5`).
- `test_return_qk_layers_gradients_under_checkpointing_and_loops` (*Review Focus*): tests both `gradient_checkpointing_mode="full"` and `"selective"`, `value_residual="learnable"`, and `num_loops=2` (`effective_depth=8`, `return_qk_layers=(6, 7)`); verifies loss on `out.qk_states` backpropagates non-zero finite grads to `q_proj.weight`, `k_proj.weight`, and earlier layers.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/model/test_qk_tap.py -v`  
Expected: FAIL (`TypeError: OplmForMaskedLM.forward() got an unexpected keyword argument 'return_qk_layers'`).

- [ ] **Step 3: Implement `return_qk` / `return_qk_layers` across `attention.py`, `transformer.py`, and `modeling_oplm.py`**

- `OplmAttention.forward(..., return_qk: bool = False)`: when `return_qk=True`, append `(q, k)` (captured right after `self._apply_rope(q, k)`) to the return tuple; when `False`, return exact existing tuple.
- `OplmBlock._forward_impl` & `OplmBlock.forward(..., return_qk: bool = False)`: pass `return_qk` through `torch_checkpoint.checkpoint` as a constant Python bool and propagate `(q, k)` when `return_qk=True`.
- `OplmStack.forward(..., return_qk_layers: tuple[int, ...] | None = None)`: iterate `for exec_idx, layer_index in enumerate(self.layer_execution_order):`, set `want_qk = (qk_layer_set is not None and exec_idx in qk_layer_set)`, collect `qk_states`, return 3-tuple when `return_qk_layers is None` and 4-tuple `(last_hidden, hidden_states, attentions, qk_states)` otherwise.
- `OplmModel.forward` & `OplmForMaskedLM.forward`: accept `return_qk_layers: tuple[int, ...] | None = None`, return `OplmBaseModelOutput` / `OplmMaskedLMOutput` when set (keep `return_dict=False` tuple order unchanged when `None`).

- [ ] **Step 4: Run model test suite**

Run: `.venv/bin/python -m pytest tests/model/test_qk_tap.py tests/model/test_attention.py tests/model/test_transformer.py tests/model/test_modeling.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/model/attention.py src/oplm/model/transformer.py src/oplm/model/modeling_oplm.py tests/model/test_qk_tap.py
git commit -m "feat(model): add opt-in return_qk_layers tap preserving FlexAttention"
```

---

### Task 4: Auxiliary 2D Contact Heads, Soft-BCE Loss, and `DistillMLMTask` Integration

**Files:**
- Create: `src/oplm/training/contact_distill.py`
- Modify: `src/oplm/training/task.py`
- Create Test: `tests/training/test_contact_distill.py`
- Modify Test: `tests/training/test_task.py`

**Interfaces:**
- Consumes: `DistillConfig` (Task 1), `teacher_contact_meta` / `teacher_contact_values` batch keys (Task 2), `return_qk_layers` / `OplmMaskedLMOutput.qk_states` (Task 3), `oplm.training.optim.get_schedule_fn`.
- Produces:
  - `DISTILL_AUX_SIDECAR_NAME = "distill_aux.pt"`
  - `AttentionContactHead`, `PairContactHead`, `build_contact_head`, `compute_contact_distill_loss`, `AuxContactManager` in `src/oplm/training/contact_distill.py`.
  - `DistillMLMTask` methods: `prepare_aux(model, accelerator, cfg, total_steps)`, `sync_aux_schedule(global_step, *, fresh_start: bool)`, `step_aux(accelerator, max_grad_norm, global_step)`, `aux_state_dict() -> dict[str, Any] | None`, `load_aux_state_dict(state: dict[str, Any], *, reset_hparams: bool = False) -> None`.

- [ ] **Step 1: Write failing tests in `tests/training/test_contact_distill.py` and `tests/training/test_task.py`**

Add tests covering:
- `test_masked_apc_matches_eval_apply_apc` (*Review Focus*): verifies batched masked APC on padded `(C, H, L_max, L_max)` matches `oplm.eval.metrics.contact.apply_apc` per segment (`atol=1e-5`) and stays strictly zero outside each segment's `L_c × L_c` square.
- `test_attention_contact_head_chunking_and_symmetry`: tests chunk splitting when `contact_max_chunk_pairs` forces multiple chunks, verifies exact equivalence between chunked and single-segment forward (`atol=1e-5`), verifies symmetric logits, bias init `-3.0`, and per-layer `attn_scale` handling (`qk_norm_mode="l2"` vs `"channel"`).
- `test_pair_contact_head_symmetry_and_gradients`: verifies `PairContactHead` outputs exact symmetric `(C, L_max, L_max)` logits (`(x - x.T).abs().max() < 1e-6`), bias init `-3.0`, and finite gradients to both `layer_weights`/MLP and input `hidden_states`.
- `test_soft_bce_per_sequence_averaging_and_pos_weight`: verifies a long segment (`L=60`) and a short segment (`L=15`) receive equal 50/50 weight in `loss_contact` (not 16:1), and `pos_weight=2.0` matches `bce * (1 + (pos_weight - 1) * p)`.
- `test_aux_schedule_branch_warmup_and_decay`: tests `AuxContactManager` LR trajectory on a WSD branch (`warmup_steps=2000, stable_steps=998_000, max_steps=1_025_000`, `contact_head_lr=1e-3, contact_head_warmup_steps=200`, `fresh_start=True` at `global_step=1_000_000`): ramps linearly from `1e-3 / 200` at step `1_000_000` to near peak at `1_000_199`, then decays linearly to `0.0` at `1_025_000` (never stuck on `stable_steps`!).
- `test_distill_mlm_task_2d_only_and_1d2d_and_empty_contacts` (*Review Focus*): in `tests/training/test_task.py`, tests `DistillMLMTask.step` for (a) `contact_head="attention"`, `alpha=0.0, contact_weight=1.0`, (b) `contact_head="pair"`, `alpha=0.5, contact_weight=1.0`, and (c) `M = 0` (`teacher_contact_meta=[]`, `teacher_contact_values=empty(0)`): verifies `loss_contact`, `contact_frac`, `contact_pos_rate` metrics, finite grads on backbone `q_proj`/`k_proj` (for `"attention"`) and backbone layers (for `"pair"`), and `step_aux()` zero-filling missing grads without error when `M = 0`.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_contact_distill.py tests/training/test_task.py -k "contact or 2d" -v`  
Expected: FAIL (`ModuleNotFoundError: No module named 'oplm.training.contact_distill'`).

- [ ] **Step 3: Implement `src/oplm/training/contact_distill.py` and wire `DistillMLMTask` in `src/oplm/training/task.py`**

- In `src/oplm/training/contact_distill.py`:
  - Implement `AttentionContactHead(num_layers: int, num_heads: int, attn_scales: tuple[float, ...])` with chunked masked softmax over `[<cls>, res..., <eos>]`, residue slice `[1 : L_max + 1, 1 : L_max + 1]`, symmetrization, masked APC (`g = sum(A) / L_c^2`, `A - r*c / (g + 1e-8)`), and `nn.Linear(K * H, 1, bias=True)` (`trunc_normal_(std=0.01)`, bias `-3.0`).
  - Implement `PairContactHead(num_layers: int, hidden_size: int, pair_dim: int)` with `layer_weights`, `LayerNorm`, `Linear(hidden_size, pair_dim, bias=False)`, `[u_i * u_j, |u_i - u_j|]`, 2-layer GELU MLP (`fc2.bias` init `-3.0`).
  - Implement `chunk_contact_segments(meta, max_chunk_pairs)` grouping consecutive length-sorted segments so `len(chunk) * (max_L + 2)^2 <= max_chunk_pairs` (minimum 1 segment per chunk).
  - Implement `compute_contact_distill_loss(...) -> tuple[Tensor, float]` (per-sequence mean soft BCE averaged over `M` segments, returning `(loss_contact, mean_target_pos_rate)`).
  - Implement `AuxContactManager`: owns `head`, `optimizer` (`AdamW`, `betas=(0.9, 0.98)`, `weight_decay=0.01`), `schedule_fn`, `aux_start_step`, `broadcast` on init, `sync_schedule(global_step, fresh_start=...)`, `step(accelerator, max_grad_norm, global_step)` with zero-grad fill + `dist.all_reduce(..., ReduceOp.AVG)`, and `state_dict()` / `load_state_dict(state, reset_hparams=...)`.
- In `src/oplm/training/task.py`:
  - Update `DistillMLMTask` to resolve `_qk_layers` / `_hidden_layer_slice`, create `_aux: AuxContactManager | None` in `prepare_aux` when `contact_weight > 0.0`, call `model(..., return_qk_layers=...)` or `output_hidden_states=True`, skip 1D KL cleanly when `alpha == 0.0 and contact_weight > 0.0` (so pure-2D shards don't need `"teacher_logits"` in `batch`), add `contact_weight * loss_contact`, and emit `loss_contact`, `contact_frac`, `contact_pos_rate`.

- [ ] **Step 4: Run task & contact_distill tests**

Run: `.venv/bin/python -m pytest tests/training/test_contact_distill.py tests/training/test_task.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/training/contact_distill.py src/oplm/training/task.py tests/training/test_contact_distill.py tests/training/test_task.py
git commit -m "feat(training): add AttentionContactHead, PairContactHead, and DistillMLMTask 2D loss"
```

---

### Task 5: Trainer, Checkpoint Sidecar (`distill_aux.pt`), and Remote Sync Wiring

**Files:**
- Modify: `src/oplm/training/trainer.py`
- Modify: `src/oplm/training/checkpoint.py`
- Modify: `src/oplm/training/remote.py`
- Modify Test: `tests/training/test_checkpoint.py`, `tests/training/test_task.py`

**Interfaces:**
- Consumes: `DISTILL_AUX_SIDECAR_NAME` and `DistillMLMTask` aux hooks from Task 4.
- Produces: `save_checkpoint(..., distill_aux=...)` and `load_checkpoint(..., distill_aux=...)` writing/restoring `distill_aux.pt`, `"distill_aux.pt"` in `remote._SHARED_ARTIFACT_NAMES`, and full `Trainer` integration (`prepare_aux`, `sync_aux_schedule`, `step_aux`, `branch_from`, `auto_resume`).

- [ ] **Step 1: Write failing end-to-end Trainer, `branch_from`, `auto_resume`, and sidecar tests**

Add tests covering:
- `tests/training/test_checkpoint.py`:
  - `test_distill_aux_sidecar_roundtrip_and_missing_warning`: `save_checkpoint` writes `distill_aux.pt` and does NOT add any aux keys to `hf/model.safetensors`; `load_checkpoint` restores `distill_aux.pt` cleanly; loading a checkpoint without `distill_aux.pt` warns and returns `aux_restored=False` without raising.
  - `test_remote_shared_artifacts_includes_distill_aux`: asserts `"distill_aux.pt" in remote._SHARED_ARTIFACT_NAMES`.
- `tests/training/test_task.py`:
  - `test_trainer_2d_contact_distill_branch_resume_and_clean_hf_export` (*Review Focus*, parametrized over `contact_head=("attention", "pair")`):
    1. Runs a 2-step plain MLM `Trainer` saving `checkpoint-2` (no `distill_aux.pt`).
    2. Branches a 2D `Trainer` (`branch_from=checkpoint-2`, `max_steps=4`, `stable_steps=2`, `warmup_steps=1`, `contact_weight=1.0`, `contact_head_warmup_steps=2`, packed sequences): verifies `aux_start_step == 2`, runs steps 3–4 cleanly, saves `checkpoint-4` containing `distill_aux.pt`, and verifies `OplmForMaskedLM.from_pretrained(checkpoint-4 / "hf")` loads with zero unexpected/missing keys.
    3. Resumes (`auto_resume=True`, `max_steps=6`) from `checkpoint-4`: verifies `distill_aux.pt` and `aux_start_step == 2` are restored cleanly and runs to step 6 with finite losses.

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/training/test_checkpoint.py tests/training/test_task.py -k "distill_aux or 2d_contact_distill" -v`  
Expected: FAIL.

- [ ] **Step 3: Wire `checkpoint.py`, `remote.py`, and `trainer.py`**

- In `src/oplm/training/remote.py`: add `"distill_aux.pt"` to `_SHARED_ARTIFACT_NAMES`.
- In `src/oplm/training/checkpoint.py`: add `_write_distill_aux_sidecar` (rank 0 into `tmp_dir` before commit rename) and `_restore_distill_aux_sidecar` (every rank, returns `bool` whether restored; warns on mismatch/absence), wired via optional `task: Any = None` (or `distill_aux_task: Any = None`) and `reset_aux_hparams: bool = False` on `save_checkpoint` / `load_checkpoint`.
- In `src/oplm/training/trainer.py`:
  - After `self._unwrapped_model` is resolved (line 765), call `prepare_aux` if present on `self.task`.
  - In `_resume_from_checkpoint` and `_branch_from_checkpoint`, pass `distill_aux_task=self.task` (`reset_aux_hparams=True` on branch) to `load_checkpoint`, then call `sync_aux_schedule(self.global_step, fresh_start=not restored)` (and on a fresh run with neither resume nor branch, call `sync_aux_schedule(0, fresh_start=True)`).
  - In `Trainer.train()` inside `if self.accelerator.sync_gradients:`, call `step_aux(self.accelerator, cfg.max_grad_norm, self.global_step)` right alongside `self.optimizers` / `self.schedulers`.
  - In `_save_checkpoint()`, pass `distill_aux_task=self.task` to `save_checkpoint`.

- [ ] **Step 4: Run checkpoint & training tests**

Run: `.venv/bin/python -m pytest tests/training/test_checkpoint.py tests/training/test_task.py tests/training/test_remote.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oplm/training/trainer.py src/oplm/training/checkpoint.py src/oplm/training/remote.py tests/training/test_checkpoint.py tests/training/test_task.py
git commit -m "feat(training): wire 2D aux contact optimizer and distill_aux.pt checkpoint sidecar"
```

---

### Task 6: Documentation Updates & Full Verification

**Files:**
- Modify: `docs/config.md`
- Modify: `docs/data.md`
- Modify: `docs/training.md`
- Test: `tests/test_docs.py`

**Interfaces:**
- Consumes: all finalized `DistillConfig` fields, `teacher_contacts` format, and `distill_aux.pt` sidecar from Tasks 1–5.
- Produces: accurate documentation passing all `tests/test_docs.py` invariants.

- [ ] **Step 1: Update `docs/config.md`, `docs/data.md`, and `docs/training.md`**

- `docs/config.md`: document all 9 `distill.contact_*` fields, defaults, valid values, and depth compatibility check.
- `docs/data.md`: document the `teacher_contacts` Parquet binary column (`<BBH` 4-byte header `[version=1, stored_min_sep, stored_len_u16]` + row-major upper-triangle `uint8` probabilities), shared `min_neff` / `min_neff_sqrt_l` gating, and `sample_members` incompatibility.
- `docs/training.md`: document 2D contact-pair distillation (`"attention"` vs `"pair"` discarded heads, `return_qk_layers`, `distill_aux.pt` sidecar, `branch_from` warmup/decay behavior, clean `hf/` inference exports, and `train/loss_contact`, `train/contact_frac`, `train/contact_pos_rate` metrics).

- [ ] **Step 2: Run docs tests, full fast pytest suite, ruff, and ty**

Run:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
env -u CONDA_PREFIX ty check --python /home/bryanbriney/git/oplm/.venv/bin/python src/
TRITON_CACHE_DIR=/tmp/triton_cache .venv/bin/python -m pytest -m "not slow" -q
```
Expected: 0 lint/format errors, 0 `ty` diagnostics, all fast tests PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/config.md docs/data.md docs/training.md
git commit -m "docs: document 2D contact-pair distillation config, schema, and training"
```

---

## Phase 2 (Cluster Validation on `coreweave-barcelona` — after PR implementation)

- Extraction jobs `11970` (`stage1a`) → `11971` (`stage1b`) → `11972` (`ops-2d-check`, 5-MSA exact parity + 50-MSA `unmasked` vs `fold-0` diagnostic) → `11973_[0-39]` (`ops-2d-s2`) → `ops-2d-fin` write `/mnt/data/briney/oplm-data/openproteinset-1m-1d2d/train/shard-00000..00039.parquet`.
- Run single-GPU `torch.compile(dynamic=False)` + `FlexAttention` + `SAC` + `branch_from` + `hf/` smoke test on both `"attention"` and `"pair"`.
- Queue matched 170M 25k-step `ops50` pilot runs (`min_neff_sqrt_l: 2.0`, `save_total_limit: 1`, branching from `checkpoint-1000000`):
  - Arm 1: `distill-ops50-2d-attn-w1-g2` (`alpha=0.0, contact_weight=1.0, contact_head="attention"`)
  - Arm 2: `distill-ops50-2d-pair-w1-g2` (`alpha=0.0, contact_weight=1.0, contact_head="pair"`)
  - Arm 3 (conditional): joint 1D+2D (`alpha=0.5, T=1.0` + winning 2D head).
