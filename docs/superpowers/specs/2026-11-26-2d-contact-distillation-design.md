# 2D Contact-Pair Distillation from MSA Pairformer into OPLM

**Date:** 2026-11-26  
**Board card:** [#138](https://github.com/briney/oplm/issues/138)  
**Status:** Ready for approval (builder-checked)

---

## 1. Intent & Why

In the 10-arm 170M OpenProteinSet-1M annealing pilot (25k-step WSD decay from `checkpoint-1000000`, `thebrineylab/oplm-distill-pilot`), 1D `<mask>`-only teacher-logit distillation (PR #136 + PR #137) proved genuinely positive and dosage-scaling: at 25% OPS (`distill_frac = 0.175`), `distill-ops100` beat its sequence-matched `control-ops100` on 11 of 12 metrics (`pdb40` `+0.0020`, `pdb40_2048` `+0.0021`, `cameo` `+0.0020`, `coevo` `+0.0012`, and `-0.0027` to `-0.0061` across all 7 held-out protein and antibody CE sets).

However, 1D site-independent marginals $P(x_i \mid \text{MSA})$ discard the explicit $(i, j)$ pairwise couplings where MSA Pairformer's large 3D contact P@L advantage actually lives. Even across dosage (`alpha=0.2..0.75`), temperature (`T=0.7/1.0/1.5`), divergence (`forward_kl`/`reverse_kl`), and query-clone capping (`--query-identity-cap 0.8`), 1D marginals alone appear capped at roughly `+0.001`–`+0.002` contact P@L in a 25k-step decay.

Before spending multi-day GPU compute extracting ~10M UniRef70 MSAs through MSA Pairformer for scaled-up 1D distillation, we want to test **2D contact-pair distillation** on the fast OpenProteinSet-1M pilot harness first. Because MSA Pairformer computes 1D logits and 2D Cβ contact maps in the same forward pass, validating 2D distillation now lets the eventual ~10M UniRef70 extraction produce **both 1D logits and 2D contacts in a single pass** if 2D works.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- Add optional 2D teacher Cβ contact-pair distillation under the existing `distill:` config block, fully composable with 1D logit distillation (`alpha = 0` for 2D-only, `alpha > 0` for joint 1D + 2D).
- Support two discarded training-only student contact heads behind `distill.contact_head`: `"attention"` (default, upper-layer post-RoPE `(Q, K)` attention maps) and `"pair"` (ablation, symmetric outer-interaction MLP over mixed hidden states), sharing 100% of data storage, collation, soft-BCE loss, and auxiliary-optimizer lifecycle.
- Keep `OplmForMaskedLM` inference checkpoints (`hf/`) 100% unchanged (zero extra parameters or config keys in the inference model) and preserve clean `branch_from` from pre-existing checkpoints (`checkpoint-1000000`) as well as lossless `auto_resume` on Slurm requeue.
- Keep packed `(32, 2048)` `FlexAttention`, `torch.compile(dynamic=False)`, and Selective Activation Checkpointing (SAC) intact with `< 10%` steady-state throughput overhead.
- Define a compact, self-describing `teacher_contacts` Parquet binary column that works both for the 1M OpenProteinSet pilot and for future single-pass 1D+2D extraction at 10M UniRef70 scale.
- Validate `"attention"` and `"pair"` (2D-only first, then 1D+2D if positive) against `control-ops50` in matched 170M 25k-step pilot runs on `coreweave-barcelona`.

### Non-Goals
- Distilling dense 256-dim MSA Pairformer pair representations (`L × L × 256` ≈ 10 TB for 1M chains — infeasible on disk).
- Adding a permanent contact head to `OplmForMaskedLM` or modifying `OplmFold`.
- Running the ~10M UniRef70 MSA generation and teacher extraction in this milestone (gated on the 1M OPS pilot outcome).
- Distilling ConFind side-chain contacts (`confind_contact_head`) in v1.

### Anti-Goals
- Never materialize `(B, H, 2048, 2048)` dense packed attention matrices (kills FlexAttention and OOMs).
- Never leak auxiliary training-head weights into `OplmForMaskedLM` state dicts or `hf/` exports.
- Never break `branch_from` from checkpoints trained without 2D distillation, or break `auto_resume` mid-run.
- Never deadlock under DDP or HSDP when a rank's micro-batch happens to contain zero distilled contact segments.
- Never regress plain MLM or 1D-only distillation when `distill.contact_weight == 0.0`.

---

## 3. Constraints

- **Model & attention kernel (`src/oplm/model/{attention,transformer,modeling_oplm}.py`):** Training uses packed `(32, 2048)` rows with fused `FlexAttention`, `torch.compile(dynamic=False)`, and `gradient_checkpointing_mode: selective` (`_dynamo.config.optimize_ddp = False`). `self.model` in `Trainer` is `torch.compile(DDP(OplmForMaskedLM))` (or `torch.compile(FSDP2(OplmForMaskedLM))` under HSDP). All trainable backbone ops must run inside `self.model(...)` so DDP/HSDP parameter hooks and `DTensor` unsharding work correctly.
- **Checkpointing & `branch_from` (`src/oplm/training/{trainer,checkpoint}.py`):** `branch_from` restores exact DCP model + optimizer state from `checkpoint-1000000`, and `hf/` exports call `save_pretrained` directly on `_unwrapped_model`. Auxiliary 2D head parameters therefore must live **outside** `OplmForMaskedLM` (owned by `DistillMLMTask`) and persist via an optional sidecar (`distill_aux.pt`, mirroring `ema.pt`).
- **Evaluation alignment:** OPLM's built-in contact metrics (`pdb40`, `pdb40_2048`, `cameo`, `casp`, `coevo`) score the **Categorical Jacobian** ([categorical_jacobian.py](file:///home/bryanbriney/git/oplm/src/oplm/eval/metrics/categorical_jacobian.py)), while downstream `OplmFold` reads all-layer hidden states via `LanguageModelShim` + `SingleToPair` ([lm_shim.py](file:///home/bryanbriney/git/oplm/src/oplm/fold/lm_shim.py)).
- **Cluster & storage (`coreweave-barcelona`):** 8-node (64 B200 GPU) `normal` QOS cap; `/mnt/data` has ~3.9 TB free; no loose files on NFS; keep `save_total_limit: 1` on pilot runs; never alter QOS to `admin_override` without explicit user permission.

---

## 4. Approaches Considered

### Approach 1 — Upper-Layer Attention-Map Contact Head (`"attention"`)
Keep the backbone on fused `FlexAttention`. Opt-in to return post-RoPE `(q, k)` (shape `(B, H, T, d_head)` — already computed before FlexAttention, 0 extra QK FLOPs and ~1.2 GB bf16 for 6 layers at `32 × 2048`) from the top `K` executed layers. Outside the compiled backbone, slice `(q, k)` **only for distilled segments** (mean `L ≈ 200`), compute per-segment attention weights in length-sorted mini-chunks, slice residue submatrices `[1:-1, 1:-1]`, symmetrize + APC, and apply a tiny discarded linear layer over the `K × H` head maps → symmetric contact logits → soft BCE against teacher Cβ probabilities.
- **Pros:** Directly trains attention heads to route information between 3D-contacting residues `(i, j)` — the clearest causal mechanism for moving Categorical Jacobian P@L *and* enriching hidden states. Zero QK recomputation; works natively under DDP, HSDP, SAC, and `torch.compile(dynamic=False)`.
- **Tradeoff:** Adds a small opt-in `return_qk_layers` path on `OplmAttention`/`OplmBlock`/`OplmStack`/`OplmForMaskedLM` and ~1.2 GB VRAM for the returned `(q, k)` tensors.

### Approach 2 — Auxiliary Symmetric Pair Head on Hidden States (`"pair"`)
Leave `OplmAttention` untouched. Request `output_hidden_states=True`, slice the top `K` hidden states on distilled segments only, mix layers with a learned softmax weight vector (like `OplmFold`'s `LanguageModelShim`), project to width `d_p = 128`, form symmetric pair features `[u_i * u_j, |u_i - u_j|]`, and pass through a tiny discarded 2-layer GELU MLP → symmetric contact logits → soft BCE.
- **Pros:** Zero edits to `OplmAttention`/`OplmBlock`/`OplmStack`; directly mirrors `OplmFold`'s `SingleToPair`.
- **Tradeoff:** Classic auxiliary-head risk — the backbone may park contact information in a residual-stream subspace read by the auxiliary MLP without strongly altering attention routing, potentially yielding a smaller lift on Categorical Jacobian P@L.

### Approach 3 — Both Heads Behind `distill.contact_head` *(Selected)*
Implement the shared data format, collator, chunked soft-BCE loss, and auxiliary-head optimizer/sidecar lifecycle once, with `"attention"` as the default head and `"pair"` (~35 lines of isolated module code) as an explicit ablation toggle.
- **Why selected:** 90%+ of the engineering and 100% of the teacher contact extraction is identical between Approaches 1 and 2. Supporting both lets us run a clean head-to-head comparison on the exact same OpenProteinSet-1M contact shards (costing one extra ~3h pilot run) to measure whether attention-routing supervision beats hidden-state supervision on Categorical Jacobian P@L and sequence CE.

---

## 5. Detailed Design

### 5.1 Teacher Contact Format (`teacher_contacts` Parquet Column)

Teacher targets come from `msa-pairformer@1.0.2`'s `model.contact_head(pairwise_repr_layer_15)` (`LogisticRegressionContactHead`, `dim_pairwise=256`), which outputs symmetric `(L, L)` Cβ (< 8 Å) contact probabilities.

- **Column name:** `teacher_contacts` (`pyarrow.large_binary()`, nullable), stored alongside `sequence_id`, `sequence`, `teacher_logits`, `teacher_neff`, `teacher_neff_sqrt_l`, `teacher_depth`.
- **Binary payload layout (4-byte self-describing header, exact byte-length validated — *author call*):**
  - Byte 0: `version` (`uint8`, value `1`).
  - Byte 1: `stored_min_sep` (`uint8`, default `6` — i.e. all pairs `(i, j)` with `j - i >= stored_min_sep`).
  - Bytes `2..3`: `stored_len` (`<u2`, little-endian `uint16`, `<= raw_len`). Storing `stored_len` explicitly costs 2 bytes/protein (2 MB across 1M chains) and future-proofs v1 for both policies on long chains at 10M UniRef70 scale (`teacher_contacts = null` OR prefix-cropping chains `> L_cap` to `stored_len = L_cap < len(sequence)`).
  - Bytes `4 .. 4 + n_pairs - 1`: `uint8` row-major upper-triangle entries matching `np.triu_indices(stored_len, k=stored_min_sep)`, quantized as `np.rint(np.clip(p, 0.0, 1.0) * 255.0).astype(np.uint8)`.
  - Where `n_pairs = max(0, (stored_len - stored_min_sep) * (stored_len - stored_min_sep + 1) // 2)`.
  - **Storage size:** At OpenProteinSet-1M `mean L = 198.5` (~19k pairs/protein raw, zstd-compressed heavily because ~95% of entries are `0`–`5`), expect ~10–14 GB for 1M chains (~18–22 GB combined with existing 7.4 GB 1D logits). At 10M UniRef70 (`mean L = 363`), expect ~180–250 GB zstd.
  - **Validation:** The collator checks `len(raw) >= 4`, `version == 1`, `stored_min_sep <= contact_min_sep`, `stored_len <= raw_len`, and `len(raw) == 4 + expected_n_pairs(stored_len, stored_min_sep)`, raising `ValueError` on mismatch (mirroring `teacher_logits` byte-length validation).
  - **Pre-extraction teacher check (50 chains):** Before running 1M extraction, verify inside the `msa-pairformer_1.0.2.sqsh` container:
    1. Whether `LogisticRegressionContactHead` outputs post-sigmoid probabilities in `[0, 1]` or raw logits (apply `sigmoid` if raw logits; assert `[0, 1]` finite symmetric `(L, L)` before quantization).
    2. Agreement between **unmasked** (1 pass) and **fold-0** (query 1/7 masked, matching `/usr/local/bin/msa-pairformer` line 137) Cβ contact maps (mean abs diff + CASP/CAMEO P@L), to confirm whether the future 10M UniRef70 extraction can capture contacts during `fold 0` at **0% extra forward passes**.
  - **OpenProteinSet-1M dataset assembly:** Regenerate deterministic `seed=42` Stage 1 inputs via `/mnt/data/briney/oplm-data/openproteinset-1m/scripts/stage1{a,b}.sbatch` (~12 min), extract 1-pass unmasked contacts across 40 shards using `/mnt/data/briney/containers/msa-pairformer_1.0.2.sqsh` (~30 min on 8 nodes), and merge 1-to-1 with `/mnt/data/briney/oplm-data/openproteinset-1m/train/shard-*.parquet` (asserting exact `(sequence_id, sequence)` match per row) into `/mnt/data/briney/oplm-data/openproteinset-1m-1d2d/train/shard-00000..00039.parquet`. Delete `stage1_inputs/` (`20 GB`) immediately afterward. Leave existing `openproteinset-1m` shards untouched.

### 5.2 Configuration (`src/oplm/config.py`)

Extend `DistillConfig` with 2D contact distillation knobs (all defaulting to 2D-disabled for 100% backward compatibility):

| Field | Type | Default | Semantics & Validation |
| --- | --- | --- | --- |
| `contact_weight` | `float` | `0.0` | Scalar weight on 2D soft-BCE loss. Finite, `>= 0.0`. `0.0` disables 2D loading, QK/hidden taps, and aux head completely. |
| `contact_head` | `str` | `"attention"` | Student 2D head type: `"attention"` or `"pair"`. |
| `contact_layers` | `int` | `6` | Number of top executed layers tapped (`>= 1`; validated `<= effective_depth` in `validate_distill_compat`). |
| `contact_min_sep` | `int` | `6` | Minimum residue separation `j - i` supervised (`>= 1`). Allows training on medium+long (`6`) or long-range-only (`24`) from the same `stored_min_sep=6` shards. |
| `contact_pos_weight` | `float` | `1.0` | Positive-pair multiplier in soft BCE (`> 0.0`; `1.0` = calibrated unweighted soft BCE). |
| `contact_pair_dim` | `int` | `128` | Projection width `d_p` for `contact_head="pair"` (`>= 16`). |
| `contact_head_lr` | `float` | `1e-3` | Base AdamW LR for the discarded auxiliary contact head (`> 0.0`). |
| `contact_head_warmup_steps` | `int` | `200` | Branch-local linear LR warmup steps for a freshly initialized aux head (`>= 0`; `0` disables warmup). |
| `contact_max_chunk_pairs` | `int` | `2_000_000` | Peak pair-element budget (`chunk * L_max^2`) per length-sorted chunk (`>= 10_000`). |

- Existing fields (`alpha`, `temperature`, `loss_type`, `teacher_alphabet`, `min_neff`, `min_neff_sqrt_l`) stay unchanged.
- `validate_distill_compat(cfg)` also verifies `cfg.distill.contact_layers <= effective_depth` (via `resolve_layer_execution_order`) when `contact_weight > 0.0`.
- `cfg.fold` + `cfg.distill` remains rejected.

### 5.3 Dataset, Gating, and Collation (`src/oplm/data/sequence/{dataset,collate,loaders}.py`)

- **`ShardedProteinDataset`:**
  - Add `TEACHER_CONTACTS_COLUMN = "teacher_contacts"` and `load_teacher_contacts: bool = False`.
  - When `load_teacher_contacts=True`: read `teacher_contacts` if present in the shard schema (missing column on non-distilled sources like `deepclust70` / `uniref70` yields `None`, matching `teacher_logits`).
  - **Shared quality gate:** If a row fails `min_teacher_neff` or `min_teacher_neff_sqrt_l`, strip **both** `teacher_logits` and `teacher_contacts` (`None`) so low-Neff MSAs never supervise 2D contacts.
  - Reject `sample_members` on shards carrying `teacher_contacts` (same provenance rule as `teacher_logits`).
- **`build_train_dataloader` (`loaders.py`):**
  - `load_teacher_logits = (distill is not None and distill.alpha > 0.0)`? **No — keep `load_teacher_logits = (distill is not None)` when `alpha > 0.0` OR keep `distill is not None`?** Better:
    - Set `load_teacher_logits = (distill is not None and (distill.alpha > 0.0 or distill.contact_weight == 0.0))` so existing `alpha=0.0` 1D-diagnostic runs still log `loss_kl`, while pure 2D runs (`alpha=0.0, contact_weight > 0.0`) skip decoding 1D logits completely (and work even on shards that have no `teacher_logits` column!).
    - Set `load_teacher_contacts = (distill is not None and distill.contact_weight > 0.0)`.
- **`MLMCollator` (`collate.py`):**
  - Add `load_teacher_contacts: bool = False` and `contact_min_sep: int = 6`.
  - For each sequence in the batch with non-`None` `teacher_contacts`:
    - Validate 4-byte header (`len(raw) >= 4`, `version == 1`, `stored_min_sep <= contact_min_sep`, `stored_len <= raw_len`) and exact byte length `4 + expected_n_pairs(stored_len, stored_min_sep)`.
    - Let `res_len = min(stored_len, raw_len, max_length - 2)` (kept residues between `<cls>` and `<eos>`). If `res_len <= contact_min_sep`, skip (zero scorable pairs).
    - Extract the upper-triangle `uint8` vector for `0 <= i < j < res_len`, `j - i >= contact_min_sep`.
      - Fast path (the 99.9% case where `stored_len == res_len` and `stored_min_sep == contact_min_sep`): zero-copy slice `raw[4:]` directly into `np.frombuffer(..., dtype=np.uint8)`.
      - Truncated or larger-`contact_min_sep` path: index the kept upper-triangle offsets cleanly in NumPy from the `(stored_len, stored_min_sep)` triangular layout.
  - **Batch tensors (both unpacked and packed):**
    - `teacher_contact_segments`: `int64` tensor of shape `(M, 4)` with columns `[row_idx, start_pos, res_len, value_offset]`, where `row_idx` is the (packed or unpacked) row, `start_pos` is the `<cls>` token index (`start_pos + 1 .. start_pos + res_len` are residues, `start_pos + res_len + 1` is `<eos>`), and `value_offset` is the start offset into `teacher_contact_values`.
    - `teacher_contact_values`: `uint8` 1D tensor of shape `(total_pairs,)` concatenating all `M` segments' upper-triangle quantized probabilities.
    - When `pack_sequences=True`, `_pack()` remaps each segment's `(row_idx, start_pos)` to its packed placement (updating `row_idx` and `start_pos` directly while `teacher_contact_values` and `value_offset` stay untouched!).
    - When `M == 0`, emit empty tensors of shapes `(0, 4)` and `(0,)` so `DeviceDataLoader` always sees plain tensors and moves them in one `non_blocking=True` transfer.

### 5.4 Backbone `(Q, K)` Tap (`src/oplm/model/{attention,transformer,modeling_oplm}.py`)

Only active when `return_qk_layers is not None` (`contact_head == "attention"` with `contact_weight > 0.0`). Every existing caller keeps the exact current code path and return types.

- **`OplmAttention.forward(..., return_qk: bool = False)`:**
  - Right after `q, k = self._apply_rope(q, k)` (line 451 of [attention.py](file:///home/bryanbriney/git/oplm/src/oplm/model/attention.py#L451)), `q` and `k` (each `(B, H, T, d_head)`) are already computed and in memory before `_flex_attention` / `_sdpa_attention`.
  - When `return_qk=False` (default): exact existing return tuple (`(out, attn)` or `(out, attn, v1)`).
  - When `return_qk=True`: append `(q, k)` to the return tuple (`(out, attn, (q, k))` or `(out, attn, v1, (q, k))`). FlexAttention runs completely untouched.
- **`OplmBlock._forward_impl` / `OplmBlock.forward(..., return_qk: bool = False)`:**
  - Threads `return_qk` through `torch_checkpoint.checkpoint` (works under both `"full"` and `"selective"` AC; `return_qk` is a constant Python `bool` per layer, so Dynamo traces a single static graph under `torch.compile(dynamic=False)`).
- **`OplmStack.forward(..., return_qk_layers: tuple[int, ...] | None = None)`:**
  - Accepts a sorted tuple of 0-based execution-step indices into `self.layer_execution_order` (e.g., `(18, 19, 20, 21, 22, 23)` for a 24-layer model with `contact_layers=6`).
  - Passes `return_qk=(step_idx in qk_layer_set)` to each block and collects `qk_states: tuple[tuple[Tensor, Tensor], ...]` in execution order.
  - When `return_qk_layers is None`: returns exact 3-tuple `(last_hidden, hidden_states, attentions)`.
  - When `return_qk_layers is not None`: returns 4-tuple `(last_hidden, hidden_states, attentions, qk_states)`.
- **`OplmModel.forward` & `OplmForMaskedLM.forward(..., return_qk_layers: tuple[int, ...] | None = None)`:**
  - Define lightweight `@dataclass class OplmMaskedLMOutput(MaskedLMOutput)` (and `OplmBaseModelOutput(BaseModelOutput)`) adding `qk_states: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None`.
  - When `return_qk_layers is None`: returns plain `BaseModelOutput` / `MaskedLMOutput` (zero behavior change).
  - When `return_qk_layers is not None`: returns `OplmMaskedLMOutput(..., qk_states=qk_states)`.
- **RoPE exactness check:** Inside a packed row, `q` and `k` have already been rotated at their packed positions `start_pos .. end_pos - 1`. Because RoPE inner products depend strictly on relative offset $j - i$, slicing `q[row, :, start_pos:end_pos]` and `k[row, :, start_pos:end_pos]` yields the **exact same** attention logits as FlexAttention computed internally for that segment (including Canon-A, Canon-B, QK-norm, and L2-QK-norm scale).

### 5.5 Auxiliary 2D Contact Heads & Soft-BCE Loss (`src/oplm/training/contact_distill.py`)

Place the 2D contact heads and chunked loss computation in a dedicated, cohesive module `src/oplm/training/contact_distill.py` (keeping `task.py` small and readable):

#### Chunked per-segment execution (shared by both heads)
1. If `M == 0`, return `(logits.new_zeros(()), {})`.
2. Move `teacher_contact_segments` metadata to CPU list (tiny: `M × 4` ints, e.g., ~40 rows) — or sort on CPU before H2D? **Better:** in `MLMCollator`, emit `teacher_contact_segments` **already sorted by `res_len` ascending** so `step()` never triggers an extra GPU→CPU sync! Keep a tiny Python tuple/list of `(row, start, res_len, offset)` (e.g., `batch["teacher_contact_meta"]`, ignored by `DeviceDataLoader` since it's not a tensor, just like `num_sequences` and `stream_position`!) so `step()` slices GPU tensors with **zero `.item()` / `.tolist()` GPU syncs**!
   - *(Builder note: `teacher_contact_values` stays a single `uint8` `Tensor` transferred asynchronously by `DeviceDataLoader`; `teacher_contact_meta` is a plain `list[tuple[int, int, int, int]]` of `(row_idx, start_pos, res_len, value_offset)` sorted by `res_len` ascending.)*
3. Greedily group consecutive segments into chunks such that `len(chunk) * (max_res_len_in_chunk + 2)^2 <= contact_max_chunk_pairs` (always at least 1 segment per chunk so even `L = 2046` works).
4. For each chunk, run the selected head to produce predicted symmetric contact logits `(C, L_max, L_max)` in fp32, extract each segment's upper-triangle (`j - i >= contact_min_sep`, `0 <= i < j < res_len`), pair with `target_p = teacher_contact_values[offset : offset + n_pairs].float() / 255.0`, compute soft BCE:
   ```python
   bce = F.binary_cross_entropy_with_logits(
       pred_triu, target_p, reduction="none"
   )
   if pos_weight != 1.0:
       # Soft positive upweighting: w(p) = 1 + (pos_weight - 1) * p
       bce = bce * (1.0 + (pos_weight - 1.0) * target_p)
   seg_loss = bce.mean()
   ```
5. Average `seg_loss` uniformly across all `M` distilled segments in the micro-batch (`loss_contact = sum(seg_losses) / M`) so a single `L=1000` chain does not outweigh sixteen `L=250` chains.

#### Head A: `AttentionContactHead` (`contact_head == "attention"`)
- Inputs: `qk_states` (tuple of `K` `(q, k)` pairs from the top `K` layers), per-layer `attn_scales: tuple[float, ...]` (read from `OplmAttention.attn_scale`: `1.0` for `qk_norm_mode="l2"`, `1/sqrt(d_head)` otherwise), and `teacher_contact_meta`.
- For each chunk of `C` segments (max token length `T_max = max_res_len + 2`, including `<cls>` and `<eos>`):
  - Gather `q_c, k_c` of shape `(C, H, T_max, d_head)` per layer (zero-padded beyond each segment's `res_len + 2`), plus a boolean key-padding mask `(C, 1, 1, T_max)` where positions `< res_len + 2` are valid.
  - Compute `scores = torch.matmul(q_c.float(), k_c.float().transpose(-2, -1)) * attn_scale`, fill padded key columns with `-inf`, and run `attn = F.softmax(scores, dim=-1)` over the segment's exact `[<cls>, res_0..res_{L-1}, <eos>]` span (matching FlexAttention's exact softmax denominator).
  - Slice residue-by-residue submatrix `A = attn[:, :, 1 : L_max + 1, 1 : L_max + 1]` and zero padded residue rows/cols via residue mask `m` (`(C, 1, L_max, L_max)`).
  - Symmetrize: `A = 0.5 * (A + A.transpose(-1, -2))`.
  - Masked per-segment, per-head **APC**: compute row mean `r_i = sum_j(A_ij) / L_c`, col mean `c_j = sum_i(A_ij) / L_c`, global mean `g = sum_{i,j}(A_ij) / L_c^2`, and `A_apc = A - (r * c) / (g + 1e-8)` (exact match to `apply_apc` on each segment's true `L_c × L_c` square, zero-padded outside).
  - Stack across the `K` layers → `(C, K * H, L_max, L_max)`, apply `self.proj = nn.Linear(K * H, 1, bias=True)` (weights init `trunc_normal_(std=0.01)`, bias init `-3.0` so initial predicted contact prob sits near the ~`0.05` base rate rather than `0.5`).
  - Output: symmetric logits `(C, L_max, L_max)`.
- Total aux parameters for 170M (`K=6, H=12`): **73 parameters** (`72` weights + `1` bias).

#### Head B: `PairContactHead` (`contact_head == "pair"`)
- Inputs: `hidden_states` from `OplmForMaskedLM(..., output_hidden_states=True)` (uses the top `K` block outputs `hidden_states[-K:]`) and `teacher_contact_meta`.
- Architecture (aligned with `OplmFold`'s `LanguageModelShim` + `SingleToPair`, explicitly symmetric in `(i, j)`):
  - `self.layer_weights = nn.Parameter(torch.zeros(K))` (softmax mix across the top `K` layers).
  - `self.input_norm = nn.LayerNorm(hidden_size)` + `self.proj = nn.Linear(hidden_size, d_p, bias=False)`.
  - For each chunk of `C` segments, gather **residues only** `[start_pos + 1 : start_pos + 1 + res_len]` into `(C, L_max, K, D)`, normalize + project + softmax-mix to `u: (C, L_max, d_p)` in fp32.
  - Form symmetric pair features `pair = torch.cat([u[:, :, None, :] * u[:, None, :, :], (u[:, :, None, :] - u[:, None, :, :]).abs()], dim=-1)` of shape `(C, L_max, L_max, 2 * d_p)`.
  - `self.fc1 = nn.Linear(2 * d_p, d_p)` → `F.gelu` → `self.fc2 = nn.Linear(d_p, 1, bias=True)` (`fc2.bias` init `-3.0`).
  - Output: symmetric logits `(C, L_max, L_max)`.
- Total aux parameters for 170M (`D=768, d_p=128, K=6`): **~133K parameters** (~0.08% of 170M).

#### Combined Step Loss & Logged Metrics (`DistillMLMTask.step`)
- When `contact_weight == 0.0`: exact existing `DistillMLMTask.step` (no `return_qk_layers`, no `output_hidden_states`, no aux head).
- When `contact_weight > 0.0`:
  - Call `model(..., return_qk_layers=self._qk_layers)` (for `"attention"`) or `model(..., output_hidden_states=True)` (for `"pair"`).
  - Compute `loss_1d` (exact existing 1D `<mask>`-only CE/KL logic; when `alpha == 0.0`, plain MLM CE).
  - Compute `loss_contact` via the active aux head (outside the compiled model graph, so variable `M` and `L` never trigger Dynamo recompilation!).
  - Total loss: `loss = loss_1d + self._distill.contact_weight * loss_contact`.
  - Additional logged metrics under `train/`:
    - `loss_contact` (when `M > 0`, raw unweighted soft BCE before multiplying by `contact_weight`)
    - `contact_frac` (`M / samples`, fraction of sequences in the micro-batch with teacher contacts)
    - `contact_pos_rate` (when `M > 0`, mean `target_p` across supervised pairs — sanity check that targets decode properly)

### 5.6 Auxiliary Head Optimizer, Distributed Sync, and Checkpointing

Because the auxiliary head lives on `DistillMLMTask` (never attached to `OplmForMaskedLM`), `OplmForMaskedLM`'s parameters, Muon/AdamW groups, DCP state dict, `branch_from`, and `hf/` exports remain **100% untouched**.

Add three optional lifecycle hooks checked via `getattr` (or optional methods with default no-ops on `MLMTask` — keeping `TrainTask` protocol clean and `FoldTask` untouched):

1. **`prepare_aux(self, model: nn.Module, accelerator: Accelerator, cfg: OplmConfig, total_steps: int) -> None`:**
   - Called in `Trainer.__init__` right after `self._unwrapped_model` is resolved (and before `_resume_from_checkpoint` / `_branch_from_checkpoint`).
   - Instantiates `AttentionContactHead` or `PairContactHead` from `self._unwrapped_model.config` (resolving `effective_depth`, `attn_scales`, `hidden_size`), moves it to `accelerator.device`.
   - If `dist.is_initialized()`: broadcasts every parameter tensor from rank 0 (`dist.broadcast(p.data, src=0)`) so all ranks start bit-for-bit identical.
   - Builds a standalone `torch.optim.AdamW(head.parameters(), lr=distill.contact_head_lr, weight_decay=0.01, betas=(0.9, 0.98))` paired with a live-step LR rule (*author call — branch-safe schedule*):
     - Track `self._aux_start_step: int = 0` (persisted in `distill_aux.pt`).
     - At any global step `s`, set the aux optimizer's LR to:
       `lr(s) = contact_head_lr * backbone_schedule_lambda(s) * ramp(s)`
       where `backbone_schedule_lambda` is the exact `get_schedule_fn(cfg.train, total_steps)` multiplier used by the backbone, and `ramp(s) = 1.0` if `contact_head_warmup_steps == 0` else `min(1.0, (s - aux_start_step + 1) / contact_head_warmup_steps)` (clamped `>= 0`).
     - **Why:** On `branch_from: checkpoint-1000000` (where `distill_aux.pt` is absent and `cfg.train` has `stable_steps = 998_000`, `max_steps = 1_025_000`), leaving a PyTorch `LambdaLR` at `last_epoch = 0` would stall on the 998k stable plateau and never decay, while jumping straight to full `1e-3` risks an initial gradient shock from a randomly initialized `PairContactHead` into a converged 1M-step backbone. Calling `sync_aux_schedule(global_step, fresh_start=not restored_from_sidecar)` right after `_resume_from_checkpoint` / `_branch_from_checkpoint` sets `aux_start_step = global_step` when fresh, evaluates `lr(global_step)` immediately for step 1, warms up smoothly over 200 steps, decays cleanly to `min_lr` at `1,025,000` alongside the backbone, and survives `auto_resume` losslessly.
2. **`step_aux(self, accelerator: Accelerator, max_grad_norm: float, global_step: int) -> None`:**
   - Called in `Trainer.train()` inside the `if self.accelerator.sync_gradients:` block, right alongside the main optimizer/scheduler step.
   - If `accelerator.optimizer_step_was_skipped`: `aux_optimizer.zero_grad(set_to_none=True)` and return.
   - For each `p` in `aux_head.parameters()`:
     - If `p.grad is None`: set `p.grad = torch.zeros_like(p)` (prevents any deadlock if a rank had `M == 0` contact segments during this accumulation window!).
     - Note on gradient accumulation scaling: `accelerator.backward(loss)` already divides `loss` by `gradient_accumulation_steps` before backprop, so `p.grad` is already properly accumulation-averaged!
     - If `dist.is_initialized()` and `accelerator.num_processes > 1`: `dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)`. (Works identically under both **DDP** and **HSDP**!)
   - If `max_grad_norm > 0`: `torch.nn.utils.clip_grad_norm_(aux_head.parameters(), max_grad_norm)`.
   - `aux_optimizer.step()`, `aux_optimizer.zero_grad(set_to_none=True)`, and update `aux_optimizer`'s group LR to `lr(global_step + 1)` for the next step.
3. **Checkpoint sidecar (`distill_aux.pt`):**
   - Mirrors `ema.pt` in [checkpoint.py](file:///home/bryanbriney/git/oplm/src/oplm/training/checkpoint.py#L294-L329):
     - `save_checkpoint` writes `distill_aux.pt` (`{"head": ..., "optimizer": ..., "head_type": ..., "aux_start_step": ...}`) on main process into `tmp_dir` before atomic rename when `task.aux_state_dict()` is non-`None`. Also add `"distill_aux.pt"` to `remote._SHARED_ARTIFACT_NAMES`.
     - `load_checkpoint` restores `distill_aux.pt` on every rank via `task.load_aux_state_dict(...)` when present; if absent (e.g., `branch_from: checkpoint-1000000`), logs a warning, leaves the freshly initialized aux head intact, and sets `aux_start_step = global_step` via `sync_aux_schedule`.
     - On `branch_from`, if a checkpoint *does* contain `distill_aux.pt` (e.g. branching from a 2D-distilled checkpoint with a new LR schedule), restore head + optimizer momentum, reset live `contact_head_lr`/`weight_decay`, and evaluate `lr(global_step)` immediately without re-running warmup (`aux_start_step` already `<= global_step - contact_head_warmup_steps`).

### 5.7 Cluster Validation Plan (`coreweave-barcelona`)

Execution happens in two cleanly separated phases: **Phase 1** implements and tests the `oplm` repo PR; **Phase 2** runs contact extraction and the matched 170M pilot bakeoff on `coreweave-barcelona` using our existing pilot infrastructure:

- **Existing cluster assets:**
  - Base checkpoint: `/mnt/data/briney/oplm-runs/miniprod-170M/stable/output/checkpoint-1000000`
  - Pilot harness & baseline configs (`control-ops50.yaml`, `run_arm.sbatch`, `check_wandb.py`): `/mnt/data/briney/oplm-runs/distill-pilot-170M/`
  - OpenProteinSet Stage 1 (`seed=42`) & Stage 2 scripts: `/mnt/data/briney/oplm-data/openproteinset-1m/scripts/` (`stage1a.sbatch`, `stage1b.sbatch`, `stage2_extract_shard.py`)
  - Existing 1D OPS-1M shards: `/mnt/data/briney/oplm-data/openproteinset-1m/train/shard-00000..00039.parquet`
  - Teacher container: `/mnt/data/briney/containers/msa-pairformer_1.0.2.sqsh`
- **Matched `ops50` mix & schedule** (exact match to `control-ops50` `wgp5ey8y` and `distill-ops50-*` arms): `openproteinset: 0.095127`, `uniref70: 0.052470` (`sample_members: true`), `deepclust70: 0.832587`, `antiref96: 0.019816` (`sample_members: true`); `wsd_linear`, `lr: 0.002`, `warmup_steps: 2000`, `stable_steps: 998000`, `max_steps: 1025000`, `tokens_per_step: 4194304`, `batch_tokens: 65536`, `save_total_limit: 1`, W&B project `oplm-distill-pilot`, and **`distill.min_neff_sqrt_l: 2.0`** (matching the `g2` gate across all 1D pilot arms; `91.62%` of OPS-1M chains pass).

1. **50-chain teacher check (~2 min, 1 B200):** Verify `LogisticRegressionContactHead` probability range, symmetry, `uint8` quantization error, and `unmasked` vs `fold-0` contact agreement (MAD + CASP/CAMEO P@L).
2. **1M OpenProteinSet contact extraction (~45 min wall-clock on 8 nodes):**
   - Re-run deterministic `seed=42` Stage 1A + 1B (~12 min).
   - Run 40-shard 1-pass unmasked contact extraction (~30 min on 8 nodes) and merge 1-to-1 onto existing `openproteinset-1m` 1D logits → `/mnt/data/briney/oplm-data/openproteinset-1m-1d2d/train/shard-00000..00039.parquet`.
   - Delete `stage1_inputs/` (`20 GB`) immediately on 40/40 completion.
3. **Single-GPU smoke check (~5 min):** Verify `torch.compile(dynamic=False)` + `FlexAttention` + `SAC` + `branch_from` + `auto_resume` + `hf/` export has zero extra keys + throughput check for both `"attention"` and `"pair"`.
4. **170M 25k-step Pilot Bakeoff (`ops50` mix, `min_neff_sqrt_l: 2.0`, branching from `checkpoint-1000000`, `save_total_limit: 1`):**
   - **Arm 1 (`2d-attn`):** `alpha=0.0`, `contact_weight=1.0`, `contact_head="attention"`, `contact_layers=6`, `contact_min_sep=6`, `min_neff_sqrt_l=2.0`.
   - **Arm 2 (`2d-pair`):** `alpha=0.0`, `contact_weight=1.0`, `contact_head="pair"`, `contact_layers=6`, `contact_min_sep=6`, `min_neff_sqrt_l=2.0`.
   - Benchmark directly against `control-ops50` (`wgp5ey8y`: `pdb40 0.4146`, `pdb40_2048 0.4126`, `cameo 0.4337`, `casp 0.3588`, `coevo 0.3978`, `uniref70_gte2 2.0037`) and `baseline-uniref25` (`eucpfcmx`: `pdb40 0.4199`).
   - **Arm 3 (conditional on Arm 1/2 win):** Combine winning 2D head (and tuned `contact_weight` if loss ratio suggests adjusting) with our winning 1D recipe (`<mask>`-only `forward_kl`, `alpha=0.5`, `T=1.0`, `min_neff_sqrt_l=2.0`) on `ops50` (and optionally `ops100`).

---

## 6. What's Left to the Builder

- Exact helper factoring inside `src/oplm/training/contact_distill.py` (including masked APC vectorization and chunk-slicing helper).
- Exact test file placement mirroring `src/` (`tests/training/test_contact_distill.py`, extensions to `tests/data/sequence/test_collate.py`, `tests/data/sequence/test_dataset.py`, `tests/model/test_attention.py`, `tests/training/test_config.py`, `tests/training/test_checkpoint.py`, `tests/test_docs.py`).
- Documentation updates in `docs/config.md`, `docs/data.md`, and `docs/training.md` (and `docs/methods/` once pilot runs decide production choices).
- Writing the Stage 2 2D-contact extraction + join script in the cluster run directory (or scratch) reusing `stage2_extract_shard.py`'s 32-worker pattern.

---

## 7. Open Decisions / Risks Noted

- **`contact_weight` starting scale:** Soft BCE at base rate `~0.05` starts around `0.20` and drops toward `0.08–0.12`, compared to MLM CE `~1.85`. Starting at `contact_weight: 1.0` keeps 2D gradients meaningful (~5–10% of total loss) without overwhelming MLM CE; the single-GPU smoke run will inspect initial `loss_contact` and backbone gradient-norm ratio before launching the 25k pilot runs.
- **Looped models (`num_loops > 1`):** `return_qk_layers` indexes `self.layer_execution_order` (executed steps `0..D-1`), not physical layer index, so looped models tap their final `contact_layers` executed passes cleanly.
- **Unmasked vs masked query positions during 2D training:** 2D contact loss supervises all valid residue pairs `(i, j)` with `j - i >= contact_min_sep` even when `i` or `j` was replaced by `<mask>` (or random AA) by MLM masking, forcing structural inference across masked positions.
