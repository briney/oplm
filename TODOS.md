# TODO: Implement OPLM Architecture Ablation Best Bets

Implement four behavior-preserving architecture toggles: mask-token embedding
dropout, true L2 QKNorm with learned per-head scale, learnable residual branch
gates, and GEGLU. Defaults must preserve current model behavior and checkpoint
compatibility; attention logit soft-capping remains deferred until learned QKNorm
is evaluated.

## Phase 1: Config Surface And Compatibility

- [ ] Add new `OplmConfig` fields with defaults:
  - `mask_dropout: bool = False`
  - `mask_dropout_reference_ratio: float = 0.12`
  - `qk_norm_mode: str = "channel"` with valid values `channel | l2`
  - `qk_norm_l2_scale_init: float | None = None`
  - `residual_gate: str = "none"` with valid values `none | scalar | channel`
  - `residual_gate_init: float = 1.0`
- [ ] Preserve existing semantics:
  - `qk_norm=True, qk_norm_mode="channel"` keeps current LayerNorm/RMSNorm QK norm.
  - `residual_gate="none"` keeps current fixed `alpha` residual writes.
  - `mask_dropout=False` keeps exact current embedding lookup.
  - `ffn_activation="swiglu"` remains default.
- [ ] Add validation:
  - `mask_dropout_reference_ratio >= 0` and `< 1`.
  - `qk_norm_mode` and `residual_gate` must match enums.
  - `qk_norm_l2_scale_init`, when set, must be positive.
  - `residual_gate_init` must be finite.
- [ ] Update `src/oplm/configs/model/base.yaml`, `docs/CONFIG.md`, and
  `docs/OVERVIEW.md` so new knobs match the existing config style and are not
  hidden implementation details.
- [ ] Update config default, validation, CLI override, YAML typo-guard, and
  save/load round-trip tests for all new fields.

## Phase 2: Implement Mask Dropout

- [ ] Extend embedding forward plumbing so `OplmStack` validates/materializes
  `attention_mask` before embedding lookup and passes it to `OplmEmbedding`.
- [ ] Implement `mask_dropout=True` only for the `input_ids` path; leave
  `inputs_embeds` unchanged because `<mask>` positions cannot be inferred.
- [ ] When enabled:
  - zero every embedding row whose ID equals `mask_token_id`,
  - compute `observed_mask_ratio = count(<mask>) / count(real tokens)` per row,
  - scale embeddings by
    `(1 - mask_dropout_reference_ratio) / (1 - observed_mask_ratio)`,
  - clamp denominators/counts so all-pad or all-mask edge cases remain finite.
- [ ] Apply mask dropout before optional post-embedding norm.
- [ ] Document `mask_dropout_reference_ratio` as the expected fraction of real
  tokens that are `<mask>` under the training masking policy, not as a fraction
  of mask tokens to drop.

## Phase 3: Implement True L2 QKNorm

- [ ] Refactor attention Q/K normalization into explicit modes:
  - `channel`: current `make_norm(norm_type, head_dim)` path plus fixed
    `1/sqrt(head_dim)` attention scaling.
  - `l2`: fp32 L2-normalize Q and K over `d_head`, then apply learned per-head
    scale.
- [ ] Add `qk_l2_scale: nn.Parameter` with shape `(num_attention_heads,)` when
  `qk_norm=True` and `qk_norm_mode="l2"`.
- [ ] Initialize `qk_l2_scale` to `qk_norm_l2_scale_init` when set, otherwise
  `sqrt(head_dim)`.
- [ ] In L2 mode, multiply Q by the per-head scale and use attention kernel scale
  `1.0` in both SDPA and manual attention paths.
- [ ] Keep `qk_norm=False` as the existing no-QK-normalization path with fixed
  attention scaling.

## Phase 4: Implement Residual Gates

- [ ] Keep the current persistent scalar `alpha` buffer as the base residual scale
  from `residual_scaling`.
- [ ] Add optional learnable gates per block:
  - `none`: no new parameters.
  - `scalar`: separate scalar parameters for attention and FFN residual writes.
  - `channel`: separate `(hidden_size,)` parameters for attention and FFN residual
    writes.
- [ ] Initialize gate parameters directly to `residual_gate_init`; do not route
  them through generic weight init.
- [ ] Apply gates as multiplicative refinements on residual writes:
  - `x + alpha * attn_gate * attn_out`
  - `h + alpha * ffn_gate * ffn_out`
- [ ] Rely on existing optimizer grouping so 1D gates land in no-decay AdamW
  groups, including the auxiliary AdamW path under Muon.

## Phase 5: Implement GEGLU

- [x] Add a `GEGLU` FFN class parallel to `SwiGLU`, with the same projection
  shapes and `ffn_bias` handling.
- [x] Implement `GEGLU.forward()` as `down(gelu(gate(x)) * up(x))`.
- [x] Mark `GEGLU.down_proj._is_residual_writer = True` so residual-writer init
  scaling matches SwiGLU.
- [x] Update `make_ffn()` so `ffn_activation="geglu"` constructs `GEGLU` instead
  of raising.
- [x] Update model exports and HuggingFace remote-code dependency imports if needed.

## Phase 6: Tests And Validation

- [x] Add embedding tests for disabled/default behavior, `<mask>` zeroing,
  expected per-row scaling, attention-mask length accounting, `inputs_embeds`
  bypass, and degenerate finite outputs.
- [x] Add attention tests for channel-mode parity, L2 parameter shape/init,
  manual/SDPA agreement, `qk_norm=False` behavior, and gradient flow into
  `qk_l2_scale`.
- [x] Add transformer tests for residual-gate absence, scalar/channel shapes,
  initialization, state-dict persistence, output effect when edited, and gradient
  flow.
- [x] Replace the GEGLU-not-implemented test with GEGLU factory, formula, bias,
  residual-writer marker, shape, and gradient tests.
- [x] Add a targeted integration test covering representative combinations of
  `mask_dropout`, L2 QKNorm, residual gates, GEGLU, sandwich norm, and optimizer
  parameter grouping without exploding the existing full toggle matrix.
- [x] Run focused tests:
  - `pytest tests/model/test_config.py tests/model/test_embedding.py tests/model/test_attention.py tests/model/test_ffn.py tests/model/test_transformer.py tests/model/test_toggles.py tests/model/test_save_load.py tests/training/test_config.py tests/training/test_optim.py`
- [x] Run final gates:
  - `ruff format src/ tests/`
  - `ruff check src/ tests/`
  - `ty check src/`
  - `pytest -m "not slow"`

## Assumptions

- [x] `mask_dropout_reference_ratio=0.12` matches the current default masking
  policy: `mask_prob=0.15` times `mask_token_prob=0.8`.
- [x] `qk_norm_mode="l2"` is the only QK mode that introduces a learned per-head
  scale.
- [x] Residual gates refine the existing residual scaling rather than replacing
  `residual_scaling`.
- [x] No attention logit soft-capping fields, docs, or tests are added in this pass.
