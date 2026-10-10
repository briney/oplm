# Structure Head Milestone 1 — Inference Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port ESMFold2-Fast's inference path into `oplm.fold` so that the released
`biohub/ESMFold2-Fast` head weights load into our modules without renaming and
reproduce the upstream intermediates to recorded tolerances, with a frozen OPLM (or
ESMC hidden states) feeding the shim, and a `fold()` API that writes mmCIF.

**Architecture:** Every module adopts the parameter names of the released HF
checkpoint (verified against its `model.safetensors.index.json`: 213 distinct name
patterns, 1054 head tensors), so the only "remap" is dropping the bundled `esmc.*`
tensors. The model is a pair-only recurrent trunk (`input_embedder` → `language_model`
shim → `lm_encoder` → `parcae` recurrence over `folding_trunk` → `parcae.output_stack`
coda → `distogram_head`), an EDM atom-level diffusion head (`structure_head`) and a
confidence head, all built from milestone 0's `TriangleMultiplication` and attention
primitives. Golden fixtures recorded from the upstream `esm` package on a CPU node
(fp32, dropout off, loop and sampler state pinned) are the parity oracle; the
featurizer, the shim, the recurrence, the distogram, one denoiser evaluation and the
confidence head are each compared stage by stage.

**Tech Stack:** Python 3.11, PyTorch 2.11 (FlexAttention, `torch.utils.checkpoint`),
transformers 4.45–5.3 (`PretrainedConfig`/`PreTrainedModel`, remote code), gemmi for
mmCIF output (`fold` extra), `safetensors`; the fixture generator runs in a separate
venv with `esm==3.4.1.post1` (pins transformers <5.0, which our range also admits).

**Spec:** [Structure prediction head design](../specs/2026-10-07-structure-prediction-head-design.md),
milestone 1 of §10: "Shim, pair features, recurrence/trunk, atoms, diffusion,
confidence, distogram, predict/mmCIF, golden fixture tooling. Acceptance:
Deterministic ESMFold2 parity and block-feature parity pass." Implements §4.1, §4.3
(protein-only reference geometry), §4.5, §4.6, §4.7 (padding), §5.1, §5.3, §5.4,
§5.6, §5.7, §6.6, §9 (ESMFold2 port fixtures), §11 attribution.

**Upstream sources** (read-only, the plan's math is transcribed from them; file:line
references in this plan point at `esm` 3.4.1.post1 unless noted):
`https://github.com/Biohub/esm` — `esm/models/esmfold2/{model,layers,config,prepare_input,protein_utils,constants,conformers,processor,output,hf_checkpoint}.py`;
`https://huggingface.co/biohub/ESMFold2-Fast` (revision `45fe8656f5b3ef493c17fcf9abe9a2968902e712`, public, MIT model card, Apache-2.0 file headers in the code): `config.json`, six safetensors shards (26.1 GB incl. ESMC-6B), `model.safetensors.index.json`, `ccd.pkl` (417 MB `dict[str, rdkit.Chem.Mol]`).

**Planning baseline:** `c0f82e7` on `main` (milestone 0 merged, PR #30). Branch
`feat/fold-m1` (the cluster jobs default to it). Start in an isolated worktree with its own `.venv` (`uv venv --python 3.11 .venv && uv pip install
--python .venv/bin/python -e ".[dev,train]"`). No GPU locally; every GPU test is `slow`
and skip-guarded. Fixture-dependent tests skip unless `OPLM_FOLD_FIXTURES` points at a
generated fixture directory (Task 10 produces it on the cluster; Task 12 runs them).

**Milestone-0 obligations honoured here (docs/FOLD.md §3–§4):** FlexAttention gets a
precomputed-`BlockMask` path and shape bucketing (Task 1); padded-query rows of the
pair-biased attention are zeroed by the calling module (Task 7); the frozen LM is
excluded from `copy.deepcopy` so `build_ema` never duplicates it (Task 9).

## Global Constraints

- `AGENTS.md` rules: Python 3.11+, `from __future__ import annotations` in every file, type hints on every signature, Google-style docstrings on public APIs, ruff line length 100, `pathlib.Path` only, no logic in `__init__.py` beyond imports/registration, tests mirror the source layout, `@pytest.mark.slow` for GPU/E2E work.
- Commands: `.venv/bin/python -m pytest ...`, `.venv/bin/ruff check src/`, `.venv/bin/ruff format <changed files only>`, `VIRTUAL_ENV=.venv .venv/bin/ty check src/`. All three clean before every commit. Framework-boundary diagnostics only via a specific `# ty: ignore[<rule>]` with a reason.
- Dependencies: `torch>=2.10.0,<2.12`, `transformers>=4.45,<5.4` unchanged. The `fold` extra gains `gemmi>=0.7` (Task 11) and keeps `cuequivariance-torch>=0.12.0`, `cuequivariance-ops-torch-cu13>=0.12.0`. No `rdkit`, `biotite`, or `esm` in our dependencies; the fixture generator imports `esm` lazily and runs only in the fixture venv.
- Dependency direction (spec §3): `oplm.fold` imports `oplm.model`, `oplm.training`, `oplm.data`, `oplm.eval`; core packages import nothing from `oplm.fold` except `src/oplm/cli.py`'s registration. `oplm.fold.cli` keeps torch out of module scope.
- **Parameter names are the released checkpoint's names, verbatim.** The complete pattern list is in Task 9's `test_state_dict_matches_released_checkpoint_patterns` and is the contract every module task must satisfy; `TriangleMultiplication` (M0) already matches `tri_mul_{out,in}.{norm_start,norm_mix,proj_bundle,proj_emit,proj_gate}`.
- Spec §4.5: "Each protein chain runs through the LM as its own batch row with BOS and EOS, padded to the longest chain." BOS/EOS/PAD ids are the OPLM tokenizer's (0/2/1); `X` is id 24, as in ESMC's vocabulary, so OPLM and ESMFold2 LM inputs agree token for token.
- Spec §4.6: every pair feature is a function of a row index tensor and a column index tensor; the full tensor is the all-by-all call; "No hidden `arange(L)` anywhere." One test asserts block == slice for every function in the set.
- Spec §4.7: tokens pad to the crop size (multiples of 128 when FlexAttention is used); atoms pad to a multiple of 32 (upstream's layout, also the attention block multiple); "Reject or recrop an over-budget example before collation; never truncate its atom arrays." Padding masks, reference-geometry validity and resolved masks stay distinct.
- Spec §5.1 settings for the port: pair width 256; inputs-embedder atom width 128, 3 blocks, 4 heads, window 128 (half-window 64); relpos `r_max=32`, `s_max=2`; trunk 24 blocks (ESMFold2-Fast) with transition expansion 4; LM pair encoder 4 blocks; coda 2 blocks; diffusion `sigma_data=16`, token width 768, 12 blocks, 16 heads, transition multiplier 2, Fourier dim 256; confidence 4 blocks, 50 pLDDT bins, 64 PAE/PDE bins, 39 distance bins over 3.25–50.75 Å; **distogram 64 bins** (the released config; the spec's "128" is the upstream package default, not the checkpoint); sampler from the released config: 14 steps, `sigma_max_ratio=160`, `sigma_min_ratio=4e-4`, exponent 7, cap 256, `gamma_0=0.8`, `gamma_min=1.0`, `noise_scale=1.003`, `step_scale=1.5`.
- Spec §5.1: "Reference conformer rotations, LM masking, loop count, dropout, and diffusion noise use recorded RNG state." At inference masking and dropout are off. Upstream applies no per-token conformer rotation at inference (`prepare_input.py:1438`), so v1 inference uses the raw reference conformers; rotations are a training-time augmentation (milestone 2).
- Spec §5.4: frozen LM in bf16 under `no_grad` and eval; pair stream bf16 under autocast on CUDA, fp32 on CPU; contractions fp32 (M0 trimul); coordinates, rigid alignment, diffusion sampler math and logits in fp32; the atom attention casts Q/K/V to bf16 as upstream does (that is its precision policy, replicated for parity).
- Spec §5.4: "Preserve the reference port's residual-output initialization and gate biases, including its zero-initialized residual projections and −2 gate biases. A generic HF initialization pass must not overwrite these choices." Zero-init: atom blocks' `adaln_linear`, `structure_head.single_to_token`; weight-zero/bias −2: `attn_gate`, `mlp_gate` of every diffusion block.
- Spec §6.6: "Hold the frozen `OplmModel` outside the fold model's registered module tree ... Verify exclusion from `named_parameters`, `state_dict`, DDP synchronization, EMA, and HF exports." `FoldConfig` records the LM identifier, revision and shape metadata; a documented override applies on load.
- Spec §9: fixtures "Save hidden states, weights/remap metadata, feature inputs, RNG choices, and deterministic intermediate outputs for 3–5 examples. Compare shim output, pair states after 1/2/3 loops, distogram logits, a denoiser evaluation at fixed noisy coordinates and noise level, and confidence on fixed coordinates. Use saved recurrence initial states and conformer rotations; do not compare independently sampled structures as a port oracle. Skip with an explicit reason when fixture artifacts are unavailable."
- Upstream facts the port must reproduce (from the inventories; all file:line in `esm`): the recurrence runs `num_loops_upstream + 1` iterations from a truncated-normal state (`model.py:884-888`, std `sqrt(2/(5·256))`, clipped ±3σ); the confidence trunk result is `pair + Trunk(pair)` (`model.py:229`); pLDDT is on a 0–1 scale; PAE/PDE expected values over 64 bins of 0.5 Å; the sampler's sigma cap truncates the schedule without re-inflating it (`layers.py:1838-1841`) and keys gamma on the *next* sigma (`gammas[1:]`); the atom feature order is `[pos 3 | charge 1 | mask 1 | element 128 | name chars 256]`; `x_inputs = cat[atom-encoder tokens 384 | res_type one-hot 33 | profile 33 | deletion_mean 1]` with profile = the one-hot and deletion_mean = 0 in single-sequence mode.
- Attribution: every ported module names its Biohub source file and modifications in its docstring; `THIRD_PARTY_NOTICES.md`'s table gains a row per new module.

## Review Focus

1. **A chain with no atoms-bearing trailing token.** Upstream derives `n_tokens` from `atom_to_token.max()+1`; a padded trailing token with no atoms would silently shorten the token axis. Our atom encoders take the explicit token count (Task 5, `test_scatter_uses_explicit_token_count`).
2. **A batch row where every token is padding** (crop padding to 128 leaves whole rows empty in a small batch). Attention, pooling and pTM must stay finite (Task 7 `test_diffusion_block_names_gate_init_and_padded_rows`, Task 8 `test_all_padding_row_is_finite`).
3. **A single-chain input through ipTM.** No inter-chain pairs: ipTM must be 0 and `pair_chains_iptm` 1×1, not NaN (Task 8, `test_tm_scores_match_transcription_and_single_chain_iptm_is_zero`).
4. **An unknown residue (`X`).** Must tokenize to res_type 22 / LM id 24, carry the four-atom `UNK` template, and get the upstream peptide-bond quirk in `token_bonds` so parity on the homodimer fixture holds (Task 3, `test_unknown_residue_tokenization_and_bond_quirk`).
5. **The frozen LM under `copy.deepcopy`, `.to()`, `.train()`.** `build_ema` deep-copies the fold model; the copy must share the LM reference, and `train()` must leave the LM in eval mode (Task 9, `test_frozen_lm_is_shared_under_deepcopy_and_stays_eval`).

## File Map and Ownership

| File | Responsibility |
|---|---|
| `src/oplm/fold/attention.py` (M0) | + precomputed `BlockMask` helpers, recompile-limit raise |
| New `src/oplm/fold/configuration_fold.py` | `FoldConfig(PretrainedConfig)` with validation and derived widths |
| New `src/oplm/fold/data/__init__.py`, `data/ccd.py`, `data/reference_conformers.json` | residue vocabulary, atom tables, reference conformers |
| New `src/oplm/fold/data/featurize.py` | `ChainSpec` → `FoldFeatures` (general representation, protein-only data) |
| New `src/oplm/fold/pair.py` | block-local pair features (relpos, bonds, mask, outer sum/product, distance bins) |
| New `src/oplm/fold/trunk.py` | `GatedMLP`, `Transition`, `PairUpdateBlock`, `PairStack`, `Recurrence` |
| New `src/oplm/fold/atoms.py` | 3D RoPE, atom attention block, `AtomEncoder`, `AtomDecoder`, `InputsEmbedder` |
| New `src/oplm/fold/lm_shim.py` | `LanguageModelShim`, per-chain frozen-LM hidden-state runner |
| New `src/oplm/fold/diffusion.py` | conditioning, adaLN, pair-bias block, denoiser, EDM schedule/sampler |
| New `src/oplm/fold/confidence.py` | `ConfidenceHead`, `DistogramHead`, pTM/ipTM |
| New `src/oplm/fold/modeling_fold.py` | `OplmFoldPreTrainedModel`, `OplmForFolding`, `FoldOutput`, frozen-LM ownership |
| `src/oplm/fold/__init__.py` | public exports + Auto registration |
| New `src/oplm/fold/fixtures.py` | upstream config/weight mapping, golden fixture generator (runs in the `esm` venv) |
| New `src/oplm/fold/predict.py` | `fold()`, sample ranking, gemmi mmCIF writer |
| `src/oplm/fold/cli.py` | + `make-fixtures`, `predict` |
| New `docs/fold/b200-fixtures.sbatch` | cluster job: `esm` venv + fixture generation on CPU |
| `docs/FOLD.md`, `THIRD_PARTY_NOTICES.md`, `pyproject.toml`, `AGENTS.md`, `docs/TESTING_E2E.md` | contract, attribution, extras |
| New tests under `tests/fold/` mirroring each module, plus `tests/fold/test_parity.py` and `tests/fold/conftest.py` | |

Execute Tasks 1–11 in order (each builds on the previous interfaces); Task 12 is the cluster acceptance gate.

---

### Task 1: Precomputed block masks and shape bucketing for the attention primitives

**Files:**
- Modify: `src/oplm/fold/attention.py`
- Test: `tests/fold/test_attention.py` (append)

**Interfaces:**
- Consumes: M0 `pair_biased_attention`, `sliding_window_attention`, `_flex`, `resolve_attention_backend`.
- Produces: `pair_bias_block_mask(key_mask: Tensor, n_queries: int | None = None) -> BlockMask`,
  `sliding_window_block_mask(valid: Tensor, half_window: int) -> BlockMask`,
  `pair_biased_attention(..., block_mask: BlockMask | None = None)`,
  `sliding_window_attention(..., block_mask: BlockMask | None = None)`,
  `ensure_flex_recompile_limit(limit: int = 64) -> None`. Tasks 5 and 7 build one mask per forward and pass it to every block.

- [ ] **Step 1: Append the failing tests**

```python
# --- precomputed block masks (milestone 1) ----------------------------------------------
from oplm.fold.attention import (  # noqa: E402  (merge into the top import block)
    ensure_flex_recompile_limit,
    pair_bias_block_mask,
    sliding_window_block_mask,
)


def test_precomputed_block_masks_match_inline_construction_on_cpu_forward() -> None:
    q, k, v = _qkv(n=64)
    bias = torch.randn(2, 2, 64, 64)
    key_mask = torch.ones(2, 64, dtype=torch.bool)
    key_mask[0, 50:] = False
    valid = torch.ones(2, 64, dtype=torch.bool)
    valid[1, 10:20] = False
    with torch.no_grad():
        inline = pair_biased_attention(q, k, v, bias, key_mask, backend="flex")
        pre = pair_biased_attention(
            q, k, v, bias, key_mask, backend="flex", block_mask=pair_bias_block_mask(key_mask)
        )
        torch.testing.assert_close(pre, inline)
        inline_sw = sliding_window_attention(q, k, v, valid, 4, backend="flex")
        pre_sw = sliding_window_attention(
            q, k, v, valid, 4, backend="flex", block_mask=sliding_window_block_mask(valid, 4)
        )
        torch.testing.assert_close(pre_sw, inline_sw)


def test_block_mask_is_ignored_on_the_dense_path() -> None:
    q, k, v = _qkv()
    bias = torch.randn(2, 2, 48, 48)
    key_mask = torch.ones(2, 48, dtype=torch.bool)
    dense = pair_biased_attention(q, k, v, bias, key_mask, backend="dense")
    with_mask = pair_biased_attention(
        q, k, v, bias, key_mask, backend="dense", block_mask=pair_bias_block_mask(key_mask)
    )
    torch.testing.assert_close(dense, with_mask)


def test_ensure_flex_recompile_limit_only_raises() -> None:
    from torch import _dynamo

    before = _dynamo.config.recompile_limit
    ensure_flex_recompile_limit(before + 8)
    assert _dynamo.config.recompile_limit == before + 8
    ensure_flex_recompile_limit(1)
    assert _dynamo.config.recompile_limit == before + 8
    _dynamo.config.recompile_limit = before
```

Run: `.venv/bin/python -m pytest tests/fold/test_attention.py -q`
Expected: FAIL with `ImportError: cannot import name 'ensure_flex_recompile_limit'`.

- [ ] **Step 2: Implement**

In `src/oplm/fold/attention.py` add after `_flex`:

```python
def pair_bias_block_mask(key_mask: Tensor, n_queries: int | None = None) -> BlockMask:
    """Block mask for `pair_biased_attention`: keys valid where `key_mask[b, j]`.

    Build it once per forward (one per diffusion-sampling call) and pass it to every
    block; `create_block_mask` is uncompiled and materialises a `(B, N, N)` mask.
    """
    batch, n_keys = key_mask.shape

    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        return key_mask[b, ki]

    return create_block_mask(
        mask_mod, batch, None, n_queries or n_keys, n_keys, device=key_mask.device
    )


def sliding_window_block_mask(valid: Tensor, half_window: int) -> BlockMask:
    """Block mask for `sliding_window_attention` (rank-based window, self always visible)."""
    batch, n = valid.shape
    rank = torch.cumsum(valid.to(torch.int64), dim=1) - 1

    def mask_mod(b: Tensor, h: Tensor, qi: Tensor, ki: Tensor) -> Tensor:
        in_window = (rank[b, qi] - rank[b, ki]).abs() <= half_window
        return (valid[b, qi] & valid[b, ki] & in_window) | (qi == ki)

    return create_block_mask(mask_mod, batch, None, n, n, device=valid.device)


def ensure_flex_recompile_limit(limit: int = 64) -> None:
    """Raise dynamo's recompile limit so bucketed shapes never fall back to eager flex.

    `_compiled_flex` uses `dynamic=False`; each new `(B, N, dtype)` compiles once. Past
    the limit dynamo silently runs the eager kernel, which materialises the full score
    matrix (docs/FOLD.md §3). Only ever raises the limit.
    """
    from torch import _dynamo

    _dynamo.config.recompile_limit = max(_dynamo.config.recompile_limit, limit)
```

Change the two public functions: add the keyword `block_mask: BlockMask | None = None`
to both signatures and docstrings ("`block_mask`: a precomputed mask from
`pair_bias_block_mask` / `sliding_window_block_mask`; built inline when `None`; ignored
on the dense path"). In `pair_biased_attention`'s flex branch replace the inline
`block_mask: BlockMask | None = None` / `if key_mask is not None:` construction with:

```python
    if block_mask is None and key_mask is not None:
        block_mask = pair_bias_block_mask(key_mask, n_q)
    return _flex(q, k, v, score_mod, block_mask)
```

and in `sliding_window_attention`'s flex branch:

```python
        if block_mask is None:
            block_mask = sliding_window_block_mask(valid, half_window)
        out = _flex(q, k, v, None, block_mask)
```

(the dense branch keeps computing `rank` itself). Add the three new names to `__all__`.

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_attention.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/attention.py tests/fold/test_attention.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all pass (the GPU test still skips).

```bash
git add src/oplm/fold/attention.py tests/fold/test_attention.py
git commit -m "feat(fold): precomputed FlexAttention block masks and a recompile-limit guard"
```

---

### Task 2: `FoldConfig` and the residue/atom reference tables

**Files:**
- Create: `src/oplm/fold/configuration_fold.py`, `src/oplm/fold/data/__init__.py`, `src/oplm/fold/data/ccd.py`, `src/oplm/fold/data/reference_conformers.json`
- Test: `tests/fold/test_configuration.py`, `tests/fold/data/__init__.py`, `tests/fold/data/test_ccd.py`

**Interfaces:**
- Produces: `FoldConfig` (fields below; derived `single_inputs_width`, `atom_feature_dim`, `relpos_feature_dim`, `inputs_token_width`), constants in `oplm.fold.data.ccd`:
  `MOL_TYPE_PROTEIN = 0`, `MOL_TYPE_DNA = 1`, `MOL_TYPE_RNA = 2`, `MOL_TYPE_LIGAND = 3`, `NUM_RES_TYPES = 33`, `PROTEIN_RES_TYPES: dict[str, int]` (`ALA`→2 … `VAL`→21, `UNK`→22), `PROTEIN_1TO3: dict[str, str]`, `UNK_RES_TYPE = 22`, `LM_UNKNOWN_ID = 24`, `MAX_ATOMIC_NUMBER = 128`, `ATOM_NAME_CHARS = 4`, `ATOM_NAME_VOCAB = 64`,
  `encode_atom_name(name: str) -> tuple[int, int, int, int]`, `ResidueTemplate` (frozen dataclass: `name`, `atoms: tuple[str, ...]`, `elements: tuple[int, ...]`, `charges: tuple[int, ...]`, `positions: Tensor[n, 3]`, `representative_atom: int`), `ReferenceConformers.load(path=None) -> ReferenceConformers` with `__getitem__(code) -> ResidueTemplate` and `.codes`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_configuration.py`:

```python
"""FoldConfig: defaults match the released ESMFold2-Fast config, derived widths, validation, round-trip."""

from __future__ import annotations

import pytest

from oplm.fold.configuration_fold import FoldConfig


def test_defaults_match_the_released_fast_checkpoint() -> None:
    cfg = FoldConfig()
    assert (cfg.pair_width, cfg.token_width, cfg.atom_width) == (256, 768, 128)
    assert cfg.inputs_token_width == 384
    assert cfg.single_inputs_width == 451
    assert cfg.atom_feature_dim == 389
    assert cfg.relpos_feature_dim == 139
    assert (cfg.trunk_blocks, cfg.lm_encoder_blocks, cfg.coda_blocks) == (24, 4, 2)
    assert (cfg.diffusion_blocks, cfg.diffusion_heads, cfg.sigma_data) == (12, 16, 16.0)
    assert (cfg.distogram_bins, cfg.plddt_bins, cfg.pae_bins, cfg.pde_bins) == (64, 50, 64, 64)
    assert (cfg.confidence_dist_bins, cfg.confidence_min_dist, cfg.confidence_max_dist) == (
        39, 3.25, 50.75,
    )
    assert (cfg.inference_num_steps, cfg.inference_sigma_cap, cfg.gamma_0) == (14, 256.0, 0.8)
    assert (cfg.gamma_min, cfg.noise_scale, cfg.step_scale) == (1.0, 1.003, 1.5)
    assert (cfg.lm_hidden_size, cfg.lm_num_hidden_states) == (768, 25)
    assert cfg.model_type == "oplm_fold"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("pair_width", 100, "multiple of 32"),
        ("token_width", 770, "divisible by diffusion_heads"),
        ("atom_width", 130, "divisible by atom_encoder_heads"),
        ("recurrence_grad_loops", 0, "recurrence_grad_loops"),
        ("recurrence_max_loops", 0, "recurrence_max_loops"),
        ("inference_num_loops", 0, "inference_num_loops"),
        ("lm_num_hidden_states", 0, "lm_num_hidden_states"),
        ("trimul_backend", "triton", "trimul_backend"),
        ("attention_backend", "sdpa", "attention_backend"),
        ("confidence_min_dist", 60.0, "confidence_min_dist"),
        ("noise_scale", -1.0, "noise_scale"),
    ],
)
def test_validation(field: str, value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FoldConfig(**{field: value})


def test_round_trip_through_save_and_load(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    cfg = FoldConfig(trunk_blocks=2, lm_name_or_path="brineylab/oplm-170M", lm_revision="abc")
    cfg.save_pretrained(tmp_path)
    loaded = FoldConfig.from_pretrained(tmp_path)
    for key, value in cfg.to_dict().items():
        if key != "_name_or_path":  # from_pretrained records the load path there
            assert loaded.to_dict()[key] == value, key
    assert loaded.trunk_blocks == 2 and loaded.lm_revision == "abc"
```

`tests/fold/data/test_ccd.py`:

```python
"""Residue vocabulary, atom-name encoding and the reference conformer table."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.data.ccd import (
    LM_UNKNOWN_ID,
    NUM_RES_TYPES,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB


def test_res_type_ids_are_alphabetical_three_letter_codes_from_two() -> None:
    assert PROTEIN_RES_TYPES["ALA"] == 2 and PROTEIN_RES_TYPES["VAL"] == 21
    assert PROTEIN_RES_TYPES["UNK"] == UNK_RES_TYPE == 22
    assert len(PROTEIN_RES_TYPES) == 21 and NUM_RES_TYPES == 33
    assert PROTEIN_1TO3["M"] == "MET" and PROTEIN_1TO3["X"] == "UNK"
    assert VOCAB["X"] == LM_UNKNOWN_ID == 24  # the LM id ESMFold2 gives X/unknown residues


@pytest.mark.parametrize(
    ("name", "codes"),
    [("CA", (35, 33, 0, 0)), ("N", (46, 0, 0, 0)), ("OXT", (47, 56, 52, 0)), ("CD1", (35, 36, 17, 0))],
)
def test_encode_atom_name_is_left_aligned_ord_minus_32(name: str, codes: tuple[int, ...]) -> None:
    assert encode_atom_name(name) == codes


def test_reference_conformers_table() -> None:
    table = ReferenceConformers.load()
    assert set(table.codes) == set(PROTEIN_RES_TYPES)
    assert sum(len(table[c].atoms) for c in table.codes) == 171
    trp = table["TRP"]
    assert trp.atoms[:5] == ("N", "CA", "C", "O", "CB") and len(trp.atoms) == 14
    assert trp.positions.shape == (14, 3) and trp.positions.dtype == torch.float32
    assert trp.elements[:5] == (7, 6, 6, 8, 6)
    assert table["GLY"].representative_atom == 1  # CA when there is no CB
    assert table["ALA"].representative_atom == 4  # CB
    assert table["LYS"].charges[table["LYS"].atoms.index("NZ")] == 1
    assert table["ARG"].charges[table["ARG"].atoms.index("NH2")] == 1
    assert table["HIS"].charges[table["HIS"].atoms.index("ND1")] == 1
    assert table["UNK"].atoms == ("N", "CA", "C", "O")
    assert torch.equal(table["UNK"].positions, torch.zeros(4, 3))
```

Run: `.venv/bin/python -m pytest tests/fold/test_configuration.py tests/fold/data -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: `src/oplm/fold/configuration_fold.py`**

```python
"""FoldConfig — PretrainedConfig for the OPLM structure prediction head.

Defaults are the released ``biohub/ESMFold2-Fast`` configuration so its weights load for
parity; training experiments override explicitly. The frozen language model is not part of
the fold model's parameters; this config records which LM the head was built against
(spec §6.6) and the shape facts the shim needs.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

__all__ = ["FoldConfig"]

_VALID_TRIMUL_BACKENDS = ("auto", "fused", "fused_forward_reference_backward", "reference")
_VALID_ATTENTION_BACKENDS = ("auto", "dense", "flex")


class FoldConfig(PretrainedConfig):
    """Configuration for :class:`~oplm.fold.modeling_fold.OplmForFolding`.

    Widths and block counts follow ESMFold2-Fast (docs/FOLD.md §2). ``lm_*`` fields describe
    the frozen LM: ``lm_num_hidden_states`` is the number of per-layer states the shim mixes
    (OPLM: ``len(layer_execution_order) + 1``; ESMC-6B: 81). ``pair_dropout`` is the per-loop
    elementwise dropout on the LM pair (upstream ``lm_dropout``, training only in this port);
    ``trunk_dropout`` is the row-shared residual dropout inside every pair-update block
    (upstream ``DropoutResidual``, 0 in the release). ``inference_num_loops`` counts iterations
    executed (upstream's ``num_loops + 1``; spec default 10, the release runs 21).
    """

    model_type = "oplm_fold"

    def __init__(
        self,
        *,
        lm_name_or_path: str | None = None,
        lm_revision: str | None = None,
        lm_hidden_size: int = 768,
        lm_num_hidden_states: int = 25,
        lm_max_chain_length: int | None = None,
        lm_pad_token_id: int = 1,
        lm_bos_token_id: int = 0,
        lm_eos_token_id: int = 2,
        lm_mask_token_id: int = 32,
        lm_input_mask_fraction: float = 0.1,
        num_res_types: int = 33,
        max_atomic_number: int = 128,
        atom_name_chars: int = 4,
        atom_name_vocab: int = 64,
        atom_pad_multiple: int = 32,
        max_atoms_per_token: int = 23,
        pair_width: int = 256,
        token_width: int = 768,
        atom_width: int = 128,
        atom_encoder_blocks: int = 3,
        atom_encoder_heads: int = 4,
        atom_window: int = 128,
        spatial_rope_base: float = 20.0,
        spatial_rope_pairs_per_axis: int = 2,
        uid_rope_pairs: int = 10,
        uid_rope_base: float = 10000.0,
        relpos_r_max: int = 32,
        relpos_s_max: int = 2,
        trunk_blocks: int = 24,
        lm_encoder_blocks: int = 4,
        coda_blocks: int = 2,
        transition_expansion: int = 4,
        pair_dropout: float = 0.25,
        trunk_dropout: float = 0.0,
        recurrence_poisson_mean: float = 3.0,
        recurrence_min_loops: int = 1,
        recurrence_max_loops: int = 6,
        recurrence_grad_loops: int = 2,
        inference_num_loops: int = 10,
        sigma_data: float = 16.0,
        fourier_dim: int = 256,
        diffusion_blocks: int = 12,
        diffusion_heads: int = 16,
        diffusion_transition_multiplier: int = 2,
        diffusion_atom_blocks: int = 3,
        diffusion_atom_heads: int = 4,
        train_noise_log_mean: float = -1.2,
        train_noise_log_std: float = 1.5,
        inference_num_steps: int = 14,
        inference_sigma_max: float = 160.0,
        inference_sigma_min: float = 4e-4,
        inference_rho: float = 7.0,
        inference_sigma_cap: float = 256.0,
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        inference_num_samples: int = 1,
        distogram_bins: int = 64,
        confidence_blocks: int = 4,
        plddt_bins: int = 50,
        pae_bins: int = 64,
        pde_bins: int = 64,
        pae_max_dist: float = 32.0,
        confidence_dist_bins: int = 39,
        confidence_min_dist: float = 3.25,
        confidence_max_dist: float = 50.75,
        trimul_backend: str = "auto",
        trimul_chunk_size: int | None = 64,
        attention_backend: str = "auto",
        layer_norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        self.lm_name_or_path = lm_name_or_path
        self.lm_revision = lm_revision
        self.lm_hidden_size = int(lm_hidden_size)
        self.lm_num_hidden_states = int(lm_num_hidden_states)
        self.lm_max_chain_length = None if lm_max_chain_length is None else int(lm_max_chain_length)
        self.lm_pad_token_id = int(lm_pad_token_id)
        self.lm_bos_token_id = int(lm_bos_token_id)
        self.lm_eos_token_id = int(lm_eos_token_id)
        self.lm_mask_token_id = int(lm_mask_token_id)
        self.lm_input_mask_fraction = float(lm_input_mask_fraction)
        self.num_res_types = int(num_res_types)
        self.max_atomic_number = int(max_atomic_number)
        self.atom_name_chars = int(atom_name_chars)
        self.atom_name_vocab = int(atom_name_vocab)
        self.atom_pad_multiple = int(atom_pad_multiple)
        self.max_atoms_per_token = int(max_atoms_per_token)
        self.pair_width = int(pair_width)
        self.token_width = int(token_width)
        self.atom_width = int(atom_width)
        self.atom_encoder_blocks = int(atom_encoder_blocks)
        self.atom_encoder_heads = int(atom_encoder_heads)
        self.atom_window = int(atom_window)
        self.spatial_rope_base = float(spatial_rope_base)
        self.spatial_rope_pairs_per_axis = int(spatial_rope_pairs_per_axis)
        self.uid_rope_pairs = int(uid_rope_pairs)
        self.uid_rope_base = float(uid_rope_base)
        self.relpos_r_max = int(relpos_r_max)
        self.relpos_s_max = int(relpos_s_max)
        self.trunk_blocks = int(trunk_blocks)
        self.lm_encoder_blocks = int(lm_encoder_blocks)
        self.coda_blocks = int(coda_blocks)
        self.transition_expansion = int(transition_expansion)
        self.pair_dropout = float(pair_dropout)
        self.trunk_dropout = float(trunk_dropout)
        self.recurrence_poisson_mean = float(recurrence_poisson_mean)
        self.recurrence_min_loops = int(recurrence_min_loops)
        self.recurrence_max_loops = int(recurrence_max_loops)
        self.recurrence_grad_loops = int(recurrence_grad_loops)
        self.inference_num_loops = int(inference_num_loops)
        self.sigma_data = float(sigma_data)
        self.fourier_dim = int(fourier_dim)
        self.diffusion_blocks = int(diffusion_blocks)
        self.diffusion_heads = int(diffusion_heads)
        self.diffusion_transition_multiplier = int(diffusion_transition_multiplier)
        self.diffusion_atom_blocks = int(diffusion_atom_blocks)
        self.diffusion_atom_heads = int(diffusion_atom_heads)
        self.train_noise_log_mean = float(train_noise_log_mean)
        self.train_noise_log_std = float(train_noise_log_std)
        self.inference_num_steps = int(inference_num_steps)
        self.inference_sigma_max = float(inference_sigma_max)
        self.inference_sigma_min = float(inference_sigma_min)
        self.inference_rho = float(inference_rho)
        self.inference_sigma_cap = float(inference_sigma_cap)
        self.gamma_0 = float(gamma_0)
        self.gamma_min = float(gamma_min)
        self.noise_scale = float(noise_scale)
        self.step_scale = float(step_scale)
        self.inference_num_samples = int(inference_num_samples)
        self.distogram_bins = int(distogram_bins)
        self.confidence_blocks = int(confidence_blocks)
        self.plddt_bins = int(plddt_bins)
        self.pae_bins = int(pae_bins)
        self.pde_bins = int(pde_bins)
        self.pae_max_dist = float(pae_max_dist)
        self.confidence_dist_bins = int(confidence_dist_bins)
        self.confidence_min_dist = float(confidence_min_dist)
        self.confidence_max_dist = float(confidence_max_dist)
        self.trimul_backend = str(trimul_backend)
        self.trimul_chunk_size = None if trimul_chunk_size is None else int(trimul_chunk_size)
        self.attention_backend = str(attention_backend)
        self.layer_norm_eps = float(layer_norm_eps)
        self._validate()
        super().__init__(**kwargs)

    # Derived widths (ESMFold2: 768 // 2 + 33 + 33 + 1 = 451; 3 + 1 + 1 + 128 + 4 * 64 = 389;
    # 2 * (2 * 32 + 2) + 1 + (2 * 2 + 2) = 139).
    @property
    def inputs_token_width(self) -> int:
        """Width of the inputs-embedder atom aggregation (half the diffusion token width)."""
        return self.token_width // 2

    @property
    def single_inputs_width(self) -> int:
        """Width of ``s_inputs``: atom aggregation + res-type one-hot + profile + deletion mean."""
        return self.inputs_token_width + 2 * self.num_res_types + 1

    @property
    def atom_feature_dim(self) -> int:
        """Width of the per-atom reference features."""
        return 3 + 1 + 1 + self.max_atomic_number + self.atom_name_chars * self.atom_name_vocab

    @property
    def relpos_feature_dim(self) -> int:
        """Width of the relative-position one-hot features."""
        return 2 * (2 * self.relpos_r_max + 2) + 1 + (2 * self.relpos_s_max + 2)

    def _validate(self) -> None:
        if self.pair_width <= 0 or self.pair_width % 32:
            raise ValueError(f"pair_width must be a positive multiple of 32; got {self.pair_width!r}.")
        if self.token_width % 2 or self.token_width % self.diffusion_heads:
            raise ValueError(
                "token_width must be even and divisible by diffusion_heads; "
                f"got {self.token_width!r} / {self.diffusion_heads!r}."
            )
        if self.atom_width % self.atom_encoder_heads or self.atom_width % self.diffusion_atom_heads:
            raise ValueError(
                "atom_width must be divisible by atom_encoder_heads and diffusion_atom_heads; "
                f"got {self.atom_width!r}."
            )
        head_dim = self.atom_width // self.atom_encoder_heads
        if 3 * self.spatial_rope_pairs_per_axis + self.uid_rope_pairs > head_dim // 2:
            raise ValueError("3 * spatial_rope_pairs_per_axis + uid_rope_pairs must fit head_dim // 2.")
        for name in (
            "lm_hidden_size", "lm_num_hidden_states", "atom_window", "trunk_blocks",
            "lm_encoder_blocks", "coda_blocks", "transition_expansion", "recurrence_min_loops",
            "recurrence_max_loops", "recurrence_grad_loops", "inference_num_loops", "fourier_dim",
            "diffusion_blocks", "diffusion_atom_blocks", "inference_num_steps", "inference_num_samples",
            "distogram_bins", "confidence_blocks", "plddt_bins", "pae_bins", "pde_bins",
            "confidence_dist_bins", "atom_pad_multiple", "max_atoms_per_token",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1; got {getattr(self, name)!r}.")
        if self.recurrence_max_loops < self.recurrence_min_loops:
            raise ValueError("recurrence_max_loops must be >= recurrence_min_loops.")
        for name in ("pair_dropout", "trunk_dropout", "lm_input_mask_fraction"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1); got {getattr(self, name)!r}.")
        for name in ("sigma_data", "inference_sigma_max", "inference_sigma_min", "inference_rho",
                     "inference_sigma_cap", "step_scale", "pae_max_dist", "layer_norm_eps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0; got {getattr(self, name)!r}.")
        for name in ("gamma_0", "gamma_min", "noise_scale", "recurrence_poisson_mean"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)!r}.")
        if not 0 < self.confidence_min_dist < self.confidence_max_dist:
            raise ValueError("confidence_min_dist must be positive and below confidence_max_dist.")
        if self.trimul_backend not in _VALID_TRIMUL_BACKENDS:
            raise ValueError(
                f"trimul_backend must be one of {_VALID_TRIMUL_BACKENDS}; got {self.trimul_backend!r}."
            )
        if self.attention_backend not in _VALID_ATTENTION_BACKENDS:
            raise ValueError(
                f"attention_backend must be one of {_VALID_ATTENTION_BACKENDS}; "
                f"got {self.attention_backend!r}."
            )
```

(`ruff format` will rewrap the long tuples.)

- [ ] **Step 3: `src/oplm/fold/data/__init__.py`** — docstring only:

```python
"""Fold data layer: reference chemistry tables and featurization (milestone 1: protein-only)."""
```

- [ ] **Step 4: `src/oplm/fold/data/ccd.py`**

```python
"""Residue vocabulary, atom tables and reference conformers for featurization.

Values transcribed from Biohub's ESMFold2 ``constants.py`` / ``protein_utils.py`` (Apache-2.0;
see THIRD_PARTY_NOTICES.md): the 33-class residue-type vocabulary (ids 2..21 are the twenty
amino acids by alphabetical three-letter code, 22 is UNK), the heavy-atom order per residue,
the three charged atoms, and the atom-name character encoding. Reference conformers come
from ``reference_conformers.json`` next to this file (see its ``source`` field); the fixture
generator can regenerate that file from the ``ccd.pkl`` shipped with the ESMFold2 weights.
Milestone 5 widens this module to ligands and nucleic acids via a gemmi-parsed CCD cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import torch

__all__ = [
    "ATOM_NAME_CHARS",
    "ATOM_NAME_VOCAB",
    "LM_UNKNOWN_ID",
    "MAX_ATOMIC_NUMBER",
    "MOL_TYPE_DNA",
    "MOL_TYPE_LIGAND",
    "MOL_TYPE_PROTEIN",
    "MOL_TYPE_RNA",
    "NUM_RES_TYPES",
    "PROTEIN_1TO3",
    "PROTEIN_RES_TYPES",
    "UNK_RES_TYPE",
    "ReferenceConformers",
    "ResidueTemplate",
    "encode_atom_name",
]

MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_LIGAND = 3
NUM_RES_TYPES = 33  # 0 MSA pad, 1 MSA gap, 2..21 amino acids, 22 UNK, 23..27 RNA, 28..32 DNA
UNK_RES_TYPE = 22
LM_UNKNOWN_ID = 24  # the OPLM/ESMC token id of "X"; ESMFold2 gives it to every non-standard residue
MAX_ATOMIC_NUMBER = 128
ATOM_NAME_CHARS = 4
ATOM_NAME_VOCAB = 64

_THREE_LETTER = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
PROTEIN_RES_TYPES: dict[str, int] = {code: 2 + i for i, code in enumerate(_THREE_LETTER)}
PROTEIN_RES_TYPES["UNK"] = UNK_RES_TYPE
PROTEIN_1TO3: dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL", "X": "UNK",
}


def encode_atom_name(name: str) -> tuple[int, int, int, int]:
    """Left-aligned four-character encoding, ``ord(c) - 32`` per character (space is 0)."""
    padded = name.ljust(ATOM_NAME_CHARS)[:ATOM_NAME_CHARS]
    codes = tuple(ord(c) - 32 for c in padded)
    if any(not 0 <= c < ATOM_NAME_VOCAB for c in codes):
        raise ValueError(f"atom name {name!r} has characters outside the 64-symbol vocabulary")
    return codes  # type: ignore[return-value]  # tuple length fixed by ATOM_NAME_CHARS


@dataclass(frozen=True)
class ResidueTemplate:
    """Heavy atoms of one residue type in model order, with reference geometry."""

    name: str
    atoms: tuple[str, ...]
    elements: tuple[int, ...]
    charges: tuple[int, ...]
    positions: torch.Tensor  # (n_atoms, 3) float32, the reference conformer

    @property
    def representative_atom(self) -> int:
        """Index of CB, or CA when the residue has no CB (distogram/confidence representative)."""
        return self.atoms.index("CB") if "CB" in self.atoms else self.atoms.index("CA")


class ReferenceConformers:
    """Residue templates keyed by three-letter code."""

    def __init__(self, residues: dict[str, ResidueTemplate], source: str) -> None:
        self._residues = residues
        self.source = source

    @classmethod
    def load(cls, path: Path | None = None) -> ReferenceConformers:
        """Load the packaged table, or a JSON file with the same schema."""
        if path is None:
            text = resources.files("oplm.fold.data").joinpath("reference_conformers.json").read_text()
        else:
            text = Path(path).read_text()
        raw = json.loads(text)
        residues = {
            code: ResidueTemplate(
                name=code,
                atoms=tuple(entry["atoms"]),
                elements=tuple(entry["elements"]),
                charges=tuple(entry["charges"]),
                positions=torch.tensor(entry["positions"], dtype=torch.float32),
            )
            for code, entry in raw["residues"].items()
        }
        return cls(residues, raw["source"])

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(self._residues)

    def __getitem__(self, code: str) -> ResidueTemplate:
        return self._residues[code]
```

- [ ] **Step 5: `src/oplm/fold/data/reference_conformers.json`**

Write exactly this file (positions are upstream's `PROTEIN_REF_POS`, `%.9g`; elements
C=6, N=7, O=8, S=16; charges LYS NZ, ARG NH2, HIS ND1 = 1):

```json
{
 "source": "Transcribed from Biohub esm 3.4.1 esm/models/esmfold2/protein_utils.py PROTEIN_REF_POS (single-chain path). Replace with the ccd.pkl 'Computed' conformers dumped by `oplm fold make-fixtures` if the featurizer parity test reports a mismatch.",
 "charge_table": "LYS NZ +1, ARG NH2 +1, HIS ND1 +1 (esm constants.CHARGED_ATOMS)",
 "residues": {
  "ALA": {"atoms": ["N","CA","C","O","CB"], "elements": [7,6,6,8,6], "charges": [0,0,0,0,0], "positions": [[-0.0100318324,-1.20730186,-1.05550611],[-0.0419013835,0.174477637,-0.572936535],[1.21275485,0.473758817,0.195216402],[1.93903291,1.44845629,-0.137597904],[-1.27694333,0.428823054,0.299377054]]},
  "ARG": {"atoms": ["N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2"], "elements": [7,6,6,8,6,6,6,7,6,7,7], "charges": [0,0,0,0,0,0,0,0,0,0,1], "positions": [[-2.01704216,0.671779811,-1.17942333],[-2.05030847,-0.573503673,-0.40972203],[-3.46944046,-1.06128132,-0.275583237],[-3.82184625,-2.13699436,-0.82949698],[-1.4193517,-0.373599142,0.985285878],[0.118788779,-0.311265498,0.963895857],[0.664324582,1.00681853,0.396332949],[2.10902381,1.0977025,0.612095237],[3.09890532,0.321592003,-0.0904717222],[4.46123028,0.384466797,0.341411382],[2.78565097,-0.416636616,-1.11482394]]},
  "ASN": {"atoms": ["N","CA","C","O","CB","CG","OD1","ND2"], "elements": [7,6,6,8,6,6,8,7], "charges": [0,0,0,0,0,0,0,0], "positions": [[-0.75956291,0.750349462,1.13698256],[-0.760878861,0.238763437,-0.235733643],[-1.92110443,-0.698243916,-0.421969295],[-2.67766619,-0.575343966,-1.42231822],[0.550489902,-0.507835031,-0.539033949],[1.72500992,0.426401794,-0.577822864],[1.94703507,1.10863924,-1.61356044],[2.57365346,0.573061883,0.560859978]]},
  "ASP": {"atoms": ["N","CA","C","O","CB","CG","OD1","OD2"], "elements": [7,6,6,8,6,6,8,8], "charges": [0,0,0,0,0,0,0,0], "positions": [[-1.84526968,-1.21695042,0.19437328],[-0.637995958,-0.419743925,0.416816443],[-0.943157256,1.03561974,0.185557172],[-1.51836085,1.40459228,-0.873985589],[0.485945761,-0.897044778,-0.52093637],[1.78034294,-0.19918935,-0.231073037],[2.52029109,-0.604458451,0.704964101],[2.14548802,0.920886159,-0.971298516]]},
  "CYS": {"atoms": ["N","CA","C","O","CB","SG"], "elements": [7,6,6,8,6,16], "charges": [0,0,0,0,0,0], "positions": [[0.0469963513,1.19007516,-1.16072738],[0.113443688,-0.0940042883,-0.459521979],[-1.26520324,-0.68323797,-0.359440625],[-1.46314394,-1.88512206,-0.682679176],[0.691988051,0.090343982,0.952482283],[2.46199274,0.523570776,0.902037263]]},
  "GLN": {"atoms": ["N","CA","C","O","CB","CG","CD","OE1","NE2"], "elements": [7,6,6,8,6,6,6,8,7], "charges": [0,0,0,0,0,0,0,0,0], "positions": [[-2.37000465,-0.963752985,-0.794274926],[-1.37000227,-0.600025892,0.210311145],[-1.75455034,0.709196746,0.843349397],[-1.85206628,0.799928963,2.09649754],[0.0204025973,-0.500446141,-0.4476448],[1.13775122,-0.286807209,0.582992435],[2.47451878,-0.24800165,-0.0936488137],[3.1685524,-1.29662466,-0.171715394],[2.9474256,0.960132957,-0.688836455]]},
  "GLU": {"atoms": ["N","CA","C","O","CB","CG","CD","OE1","OE2"], "elements": [7,6,6,8,6,6,6,8,8], "charges": [0,0,0,0,0,0,0,0,0], "positions": [[-1.5850873,-1.33768415,0.949085116],[-1.05609775,0.027459044,1.03069663],[-1.7741456,0.966439247,0.0925960094],[-1.90124416,2.18134999,0.402479351],[0.470655143,0.0488038696,0.811441481],[0.913360476,-0.421932906,-0.583098531],[2.39882207,-0.309708416,-0.721053779],[3.13893151,-1.27452445,-0.390297651],[2.96478176,0.878134608,-1.17326891]]},
  "GLY": {"atoms": ["N","CA","C","O"], "elements": [7,6,6,8], "charges": [0,0,0,0], "positions": [[-1.39429855,-0.398751289,-0.337032467],[-0.399744302,0.548894525,0.152429625],[0.944005489,-0.103140339,0.198596433],[1.33528996,-0.669218123,1.25412583]]},
  "HIS": {"atoms": ["N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2"], "elements": [7,6,6,8,6,6,7,6,6,7], "charges": [0,0,0,0,0,0,1,0,0,0], "positions": [[-1.45328677,-1.06896269,0.881072462],[-1.3396095,0.247975796,0.249600455],[-2.67525792,0.657155573,-0.304411024],[-3.13113785,1.80797768,-0.0678571537],[-0.304195583,0.217210233,-0.88853091],[1.08875132,0.0289410651,-0.364194691],[1.84045994,1.04117739,0.298045903],[1.78085542,-1.10114896,-0.381425858],[2.95669436,0.492479891,0.647711575],[3.02802038,-0.875196934,0.260843813]]},
  "ILE": {"atoms": ["N","CA","C","O","CB","CG1","CG2","CD1"], "elements": [7,6,6,8,6,6,6,6], "charges": [0,0,0,0,0,0,0,0], "positions": [[-0.716754973,-1.54261398,-0.998333037],[-1.06360853,-0.351692706,-0.213935524],[-1.38967407,0.814214528,-1.11640656],[-1.23777926,0.730291545,-2.36568403],[0.0616670065,0.0159961022,0.805739462],[1.50251997,-0.0889977664,0.241548166],[-0.05317498,-0.852105558,2.07020831],[1.792961,0.899773121,-0.886302769]]},
  "LEU": {"atoms": ["N","CA","C","O","CB","CG","CD1","CD2"], "elements": [7,6,6,8,6,6,6,6], "charges": [0,0,0,0,0,0,0,0], "positions": [[1.96575201,-1.97632241,-0.183915332],[1.30776691,-0.667743087,-0.194924369],[1.99050581,0.241820872,0.787996829],[2.0689671,-0.0788001418,2.00480461],[-0.203069419,-0.809323013,0.112435028],[-0.99162674,0.523495734,0.0672301129],[-2.42280579,0.299493372,0.573042095],[-1.02828562,1.12502646,-1.34601438]]},
  "LYS": {"atoms": ["N","CA","C","O","CB","CG","CD","CE","NZ"], "elements": [7,6,6,8,6,6,6,6,7], "charges": [0,0,0,0,0,0,0,0,1], "positions": [[2.42213726,-0.647331238,0.637057304],[2.03149271,0.278650731,-0.429851204],[2.71685934,1.59575725,-0.209247857],[3.39768171,2.11642742,-1.13325107],[0.501840293,0.487385869,-0.490629733],[-0.250620663,-0.789400995,-0.905553579],[-1.76976264,-0.555270016,-1.04032993],[-2.57653356,-1.02213669,0.184936419],[-2.26915121,-0.242938444,1.38490129]]},
  "MET": {"atoms": ["N","CA","C","O","CB","CG","SD","CE"], "elements": [7,6,6,8,6,6,16,6], "charges": [0,0,0,0,0,0,0,0], "positions": [[1.89039183,-1.52529955,-0.426385939],[1.26305711,-0.244178101,-0.762646258],[2.30391002,0.83677125,-0.725461662],[2.46541452,1.5928632,-1.72077286],[0.105679728,0.108618259,0.197416469],[-1.06580424,-0.873663127,0.0881188363],[-2.45571327,-0.333222598,1.14617002],[-3.26516509,0.703355491,-0.11588376]]},
  "PHE": {"atoms": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ"], "elements": [7,6,6,8,6,6,6,6,6,6,6], "charges": [0,0,0,0,0,0,0,0,0,0,0], "positions": [[-2.84844351,-1.52579081,0.0178981684],[-1.59196961,-0.854516268,0.352144688],[-1.89006317,0.458334148,1.02322221],[-1.34249926,0.74432373,2.12162948],[-0.760358453,-0.634285331,-0.925716043],[0.604112983,-0.0720046833,-0.614811838],[0.846831441,1.24806321,-0.714669466],[1.68276834,-0.975807726,-0.142305419],[2.18017483,1.78757334,-0.374462306],[2.88830781,-0.482775122,0.168049708],[3.14981294,0.965687394,0.0444027111]]},
  "PRO": {"atoms": ["N","CA","C","O","CB","CG","CD"], "elements": [7,6,6,8,6,6,6], "charges": [0,0,0,0,0,0,0], "positions": [[-0.836250365,-0.989980102,0.556130469],[0.3272219,-0.616445839,-0.250725716],[1.61215413,-1.1711241,0.310824126],[1.61277401,-2.27719712,0.915619373],[0.324819893,0.902824402,-0.333681464],[-1.14250839,1.27301288,-0.259060025],[-1.84959686,0.0265758112,0.268128961]]},
  "SER": {"atoms": ["N","CA","C","O","CB","OG"], "elements": [7,6,6,8,6,8], "charges": [0,0,0,0,0,0], "positions": [[0.674650252,1.50187027,-0.536729515],[0.000137928626,0.496646702,0.28510505],[0.994100988,-0.537461758,0.73505038],[1.05452418,-0.868354559,1.94953966],[-1.12792885,-0.165937632,-0.516096354],[-1.81359792,-1.08524966,0.289475143]]},
  "THR": {"atoms": ["N","CA","C","O","CB","OG1","CG2"], "elements": [7,6,6,8,6,8,6], "charges": [0,0,0,0,0,0,0], "positions": [[-1.32583034,-1.37282252,0.688223302],[-0.54333061,-0.163647547,0.416970521],[-1.29438186,0.707737207,-0.554994643],[-1.69396353,0.236544102,-1.65404189],[0.853203297,-0.536380351,-0.141093537],[1.52208209,-1.37900364,0.763516784],[1.72259331,0.705472708,-0.365133107]]},
  "TRP": {"atoms": ["N","CA","C","O","CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"], "elements": [7,6,6,8,6,6,6,6,7,6,6,6,6,6], "charges": [0,0,0,0,0,0,0,0,0,0,0,0,0,0], "positions": [[3.68603086,0.75999999,0.496155709],[2.38409209,0.0907924995,0.532526255],[2.11135721,-0.612106323,-0.773364604],[1.79652631,-1.83231485,-0.777596414],[1.2815212,1.11390364,0.855979145],[-0.0429237559,0.44645074,1.09427929],[-0.423295349,-0.154708743,2.22275543],[-1.10239005,0.215838984,0.115294322],[-1.70303202,-0.76658231,2.05950165],[-2.045645,-0.488117307,0.710669219],[-1.21735024,0.610227168,-1.30010641],[-3.25600934,-0.916439474,-0.00984987337],[-2.31592512,0.230690628,-1.97763109],[-3.38178754,-0.567733765,-1.30320537]]},
  "TYR": {"atoms": ["N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH"], "elements": [7,6,6,8,6,6,6,6,6,6,6,8], "charges": [0,0,0,0,0,0,0,0,0,0,0,0], "positions": [[-1.7900604,-0.840939939,1.31801426],[-1.91388285,0.235528454,0.330669641],[-3.34728074,0.358839989,-0.0983068496],[-3.96781135,-0.644935429,-0.542330205],[-1.00939929,0.000473141321,-0.898155212],[0.45204109,0.021162061,-0.530593276],[1.09924328,1.18779194,-0.357914299],[1.1803174,-1.25340128,-0.311221808],[2.52534509,1.19902563,0.0298046134],[2.47115111,-1.24068761,0.0435342304],[3.18068767,0.046724923,0.221485689],[4.52371979,0.067103073,0.587748587]]},
  "VAL": {"atoms": ["N","CA","C","O","CB","CG1","CG2"], "elements": [7,6,6,8,6,6,6], "charges": [0,0,0,0,0,0,0], "positions": [[0.598751903,-1.5694437,-0.737912476],[0.601435721,-0.105039664,-0.633628666],[1.83916974,0.406785041,0.0635175705],[2.39520621,-0.266619027,0.973116696],[-0.694736898,0.425909638,0.0358147547],[-1.92760313,0.0951582864,-0.817235708],[-0.893842697,-0.0864084214,1.47234976]]},
  "UNK": {"atoms": ["N","CA","C","O"], "elements": [7,6,6,8], "charges": [0,0,0,0], "positions": [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}
 }
}
```

Create `tests/fold/data/__init__.py` as an empty file.

- [ ] **Step 6: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_configuration.py tests/fold/data -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/configuration_fold.py src/oplm/fold/data tests/fold/test_configuration.py tests/fold/data && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: pass. (If hatchling does not ship the JSON in an editable install, nothing changes: `importlib.resources` reads it from `src/`. Confirm the wheel includes it later with `uv build && unzip -l dist/*.whl | grep reference_conformers`.)

```bash
git add src/oplm/fold/configuration_fold.py src/oplm/fold/data tests/fold/test_configuration.py tests/fold/data
git commit -m "feat(fold): FoldConfig and the residue/atom reference tables"
```

---

### Task 3: Block-local pair features and the protein featurizer

**Files:**
- Create: `src/oplm/fold/pair.py`, `src/oplm/fold/data/featurize.py`
- Test: `tests/fold/test_pair.py`, `tests/fold/data/test_featurize.py`

**Interfaces:**
- Consumes: Task 2 tables and `FoldConfig`.
- Produces (`pair.py`): `relpos_features(residue_index, asym_id, sym_id, entity_id, token_index, *, r_max, s_max, rows=None, cols=None) -> Tensor[B, I, J, 139]`,
  `RelativePositionEncoding(width, *, r_max, s_max)` with child `embed: nn.Linear(139, width, bias=False)` and `forward(..., rows=None, cols=None)`,
  `pair_mask(token_mask, rows=None, cols=None) -> Tensor[B, I, J]` (float),
  `outer_sum(row_vec, col_vec) -> Tensor[B, I, J, D]`, `outer_product_difference(x, rows=None, cols=None) -> Tensor[B, I, J, 2D]`,
  `token_bond_features(bonds, rows=None, cols=None) -> Tensor[B, I, J, 1]`,
  `distance_bins(coords, boundaries, rows=None, cols=None) -> Tensor[B, I, J]` (long, strict `>` count),
  `select_rows(x, rows)` / `select_cols(x, cols)` helpers.
- Produces (`featurize.py`): `ChainSpec(sequence: str, chain_id: str, copies: int = 1)`,
  `FoldFeatures` (dataclass of batched tensors, see code; `.to(device)`, `num_tokens`, `num_atoms`, `num_chains`),
  `featurize(chains, *, conformers=None, pad_tokens_to=None, atom_pad_multiple=32) -> FoldFeatures`,
  `lm_rows_for_chains(chains, *, bos, eos, pad) -> tuple[Tensor, Tensor]` (used inside).

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_pair.py`:

```python
"""Block-local pair features (spec §4.6): every function's block call equals the sliced full call."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.pair import (
    RelativePositionEncoding,
    distance_bins,
    outer_product_difference,
    outer_sum,
    pair_mask,
    relpos_features,
    token_bond_features,
)


def _indices(L: int = 12) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    asym = torch.tensor([[0] * 5 + [1] * 4 + [2] * 3])
    residue = torch.tensor([list(range(5)) + list(range(4)) + list(range(3))])
    entity = torch.tensor([[0] * 5 + [1] * 4 + [1] * 3])
    sym = torch.tensor([[0] * 5 + [0] * 4 + [1] * 3])
    token = torch.arange(L)[None]
    mask = torch.ones(1, L, dtype=torch.bool)
    mask[0, 10:] = False
    coords = torch.randn(1, L, 3, generator=g) * 10
    bonds = torch.zeros(1, L, L, 1)
    bonds[0, 2, 3, 0] = bonds[0, 3, 2, 0] = 1.0
    return {
        "asym": asym, "residue": residue, "entity": entity, "sym": sym, "token": token,
        "mask": mask, "coords": coords, "bonds": bonds,
    }


def test_relpos_feature_semantics() -> None:
    ix = _indices()
    f = relpos_features(ix["residue"], ix["asym"], ix["sym"], ix["entity"], ix["token"], r_max=32, s_max=2)
    assert f.shape == (1, 12, 12, 139) and f.dtype == torch.float32
    # same chain, residue delta +1 -> residue bin 33; token delta +1 -> token bin 65 (different residue)
    assert f[0, 1, 0, 33] == 1 and f[0, 1, 0, 66 + 65] == 1
    # different chains: residue bin 65 and token bin 65; chain delta via sym: 0-1+2 = 1 -> bin 1
    assert f[0, 0, 5, 65] == 1 and f[0, 0, 5, 66 + 65] == 1 and f[0, 0, 5, 133 + 1] == 1
    # same chain -> chain bin 5; same entity flag at index 132
    assert f[0, 0, 1, 133 + 5] == 1 and f[0, 0, 1, 132] == 1 and f[0, 0, 5, 132] == 0
    assert f[0, 9, 10, 132] == 1  # entity 1 copies 0 and 1
    assert torch.equal(f.sum(-1), torch.full((1, 12, 12), 3.0) + f[..., 132])


@pytest.mark.parametrize("seed", [0, 1])
def test_every_pair_feature_is_block_local(seed: int) -> None:
    ix = _indices()
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(12, generator=g)[:5]
    cols = torch.randperm(12, generator=g)[:7]
    torch.manual_seed(seed)
    relpos = RelativePositionEncoding(16, r_max=32, s_max=2)
    boundaries = torch.linspace(3.25, 50.75, 38)
    x = torch.randn(1, 12, 8)
    a = torch.randn(1, 12, 8)
    b = torch.randn(1, 12, 8)
    full_and_block = [
        (relpos(ix["residue"], ix["asym"], ix["sym"], ix["entity"], ix["token"]),
         relpos(ix["residue"], ix["asym"], ix["sym"], ix["entity"], ix["token"], rows=rows, cols=cols)),
        (pair_mask(ix["mask"]), pair_mask(ix["mask"], rows=rows, cols=cols)),
        (outer_sum(a, b), outer_sum(a[:, rows], b[:, cols])),
        (outer_product_difference(x), outer_product_difference(x, rows=rows, cols=cols)),
        (token_bond_features(ix["bonds"]), token_bond_features(ix["bonds"], rows=rows, cols=cols)),
        (distance_bins(ix["coords"], boundaries), distance_bins(ix["coords"], boundaries, rows=rows, cols=cols)),
    ]
    for full, block in full_and_block:
        torch.testing.assert_close(block, full[:, rows][:, :, cols])


def test_distance_bins_are_strict_upper_counts() -> None:
    coords = torch.tensor([[[0.0, 0, 0], [3.25, 0, 0], [4.0, 0, 0], [60.0, 0, 0]]])
    boundaries = torch.linspace(3.25, 50.75, 38)
    bins = distance_bins(coords, boundaries)
    assert bins[0, 0, 0] == 0 and bins[0, 0, 1] == 0  # d <= 3.25 -> bin 0 (strict >)
    assert bins[0, 0, 2] == 1 and bins[0, 0, 3] == 38
```

`tests/fold/data/test_featurize.py`:

```python
"""Sequences -> FoldFeatures: ids, atoms, padding, LM rows, the X quirk (spec §4.2, §4.5, §4.7)."""

from __future__ import annotations

import pytest
import torch

from oplm.fold.data.ccd import LM_UNKNOWN_ID, UNK_RES_TYPE, ReferenceConformers
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.model.tokenization_oplm import VOCAB


def test_monomer_features() -> None:
    f = featurize([ChainSpec("MKV", "A")])
    assert f.num_tokens == 3 and f.num_chains == 1
    assert f.res_type.tolist() == [[14, 13, 21]]  # MET, LYS, VAL
    assert f.token_index.tolist() == [[0, 1, 2]] and f.residue_index.tolist() == [[0, 1, 2]]
    assert f.asym_id.tolist() == [[0, 0, 0]] and f.mol_type.tolist() == [[0, 0, 0]]
    assert f.token_mask.all()
    assert f.num_atoms == 32  # 8 + 9 + 7 = 24 real atoms padded to a multiple of 32
    assert f.atom_mask[0].sum() == 24 and not f.atom_mask[0, 24:].any()
    assert f.atom_to_token[0, :24].tolist() == [0] * 8 + [1] * 9 + [2] * 7
    assert f.atom_to_token[0, 24:].eq(0).all()
    assert f.ref_space_uid[0, :24].tolist() == f.atom_to_token[0, :24].tolist()
    assert f.distogram_atom_idx.tolist() == [[4, 8 + 4, 17 + 4]]  # CB of each residue
    assert f.ref_element[0, :8].tolist() == [7, 6, 6, 8, 6, 6, 16, 6]  # MET: N CA C O CB CG SD CE
    assert f.ref_charge[0, 8 + 8] == 1  # LYS NZ
    assert f.ref_atom_name_chars[0, 1].tolist() == [35, 33, 0, 0]  # "CA"
    assert torch.equal(f.ref_pos[0, 24:], torch.zeros(8, 3))
    assert f.token_bonds.shape == (1, 3, 3, 1) and not f.token_bonds.any()
    assert f.lm_input_ids.tolist() == [[0, VOCAB["M"], VOCAB["K"], VOCAB["V"], 2]]
    assert f.lm_attention_mask.all()
    assert f.lm_rows.tolist() == [[0, 0, 0]] and f.lm_positions.tolist() == [[1, 2, 3]]


def test_heterodimer_and_homodimer_numbering() -> None:
    f = featurize([ChainSpec("AG", "A"), ChainSpec("GG", "B", copies=2)])
    assert f.num_chains == 3 and f.num_tokens == 6
    assert f.asym_id.tolist() == [[0, 0, 1, 1, 2, 2]]
    assert f.entity_id.tolist() == [[0, 0, 1, 1, 1, 1]]
    assert f.sym_id.tolist() == [[0, 0, 0, 0, 1, 1]]
    assert f.residue_index.tolist() == [[0, 1, 0, 1, 0, 1]]
    assert f.chain_ids == ["A", "B", "B_2"]
    assert f.lm_input_ids.shape == (3, 4)  # three rows, [BOS a g EOS]
    assert f.lm_rows.tolist() == [[0, 0, 1, 1, 2, 2]] and f.lm_positions.tolist() == [[1, 2, 1, 2, 1, 2]]


def test_lm_rows_pad_to_the_longest_chain() -> None:
    f = featurize([ChainSpec("MKVLA", "A"), ChainSpec("GG", "B")])
    assert f.lm_input_ids.shape == (2, 7)
    assert f.lm_input_ids[1].tolist() == [0, VOCAB["G"], VOCAB["G"], 2, 1, 1, 1]
    assert f.lm_attention_mask[1].tolist() == [1, 1, 1, 1, 0, 0, 0]


def test_unknown_residue_tokenization_and_bond_quirk() -> None:
    f = featurize([ChainSpec("AXG", "A")])
    assert f.res_type[0, 1] == UNK_RES_TYPE and f.lm_input_ids[0, 2] == LM_UNKNOWN_ID
    assert f.atom_to_token[0].tolist()[:13] == [0] * 5 + [1] * 4 + [2] * 4
    assert f.distogram_atom_idx[0, 1] == 5 + 1  # UNK has no CB -> CA
    bonds = f.token_bonds[0, :, :, 0]
    assert bonds[1, 0] == bonds[0, 1] == bonds[1, 2] == bonds[2, 1] == 1  # the upstream X quirk
    assert bonds[0, 2] == 0


def test_lowercase_and_unlisted_letters_are_unknown() -> None:
    f = featurize([ChainSpec("aBZ", "A")])
    assert (f.res_type == UNK_RES_TYPE).all()


def test_token_padding_and_atom_budget() -> None:
    f = featurize([ChainSpec("MKV", "A")], pad_tokens_to=8)
    assert f.num_tokens == 8 and f.token_mask[0].tolist() == [True] * 3 + [False] * 5
    assert f.res_type[0, 3:].eq(0).all() and f.asym_id[0, 3:].eq(0).all()
    assert f.lm_rows[0, 3:].eq(0).all()
    with pytest.raises(ValueError, match="pad_tokens_to"):
        featurize([ChainSpec("MKV", "A")], pad_tokens_to=2)


def test_to_device_and_custom_conformers() -> None:
    f = featurize([ChainSpec("G", "A")], conformers=ReferenceConformers.load()).to("cpu")
    assert f.ref_pos.device.type == "cpu" and f.num_atoms == 32
```

Run: `.venv/bin/python -m pytest tests/fold/test_pair.py tests/fold/data/test_featurize.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: `src/oplm/fold/pair.py`**

```python
"""Block-local pair features (spec §4.6, §5.6).

Every function takes optional ``rows``/``cols`` index tensors and returns the corresponding
block of the all-by-all tensor; the full tensor is the call with both ``None``. No function
contains a hidden ``arange(L)``: indices come from the token features. Semantics follow
ESMFold2's ``ResIdxAsymIdSymIdEntityIdEncoding`` and ``SingleToPair`` (esm/models/esmfold2/
layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md); the relpos feature is the exact AF3-style
layout the released ``input_embedder.rel_pos.embed`` weight expects: ``[residue bins 66 |
token bins 66 | same entity 1 | chain bins 6]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "RelativePositionEncoding",
    "distance_bins",
    "outer_product_difference",
    "outer_sum",
    "pair_mask",
    "relpos_features",
    "select_cols",
    "select_rows",
    "token_bond_features",
]


def select_rows(x: Tensor, rows: Tensor | None) -> Tensor:
    """``x[:, rows]`` or ``x`` when ``rows`` is ``None``."""
    return x if rows is None else x[:, rows]


def select_cols(x: Tensor, cols: Tensor | None) -> Tensor:
    """``x[:, :, cols]`` (for pair tensors) or ``x`` when ``cols`` is ``None``."""
    return x if cols is None else x[:, :, cols]


def pair_mask(token_mask: Tensor, rows: Tensor | None = None, cols: Tensor | None = None) -> Tensor:
    """Outer product of the token mask, as float ``(B, I, J)``."""
    m = token_mask.float()
    return select_rows(m, rows)[:, :, None] * select_rows(m, cols)[:, None, :]


def outer_sum(row_vec: Tensor, col_vec: Tensor) -> Tensor:
    """``row_vec[:, i] + col_vec[:, j]`` -> ``(B, I, J, D)``; pass already-selected rows/cols."""
    return row_vec[:, :, None, :] + col_vec[:, None, :, :]


def outer_product_difference(
    x: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """``cat[x_i * x_j, x_i - x_j]`` -> ``(B, I, J, 2D)`` (ESMFold2 ``SingleToPair``)."""
    xi = select_rows(x, rows)[:, :, None, :]
    xj = select_rows(x, cols)[:, None, :, :]
    return torch.cat([xi * xj, xi - xj], dim=-1)


def token_bond_features(
    bonds: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """``(B, I, J, 1)`` float block of the symmetric token-bond matrix ``(B, L, L, 1)``."""
    return select_cols(select_rows(bonds, rows), cols).float()


def distance_bins(
    coords: Tensor, boundaries: Tensor, rows: Tensor | None = None, cols: Tensor | None = None
) -> Tensor:
    """Bin index ``sum(d > boundary)`` of pairwise distances between representative atoms.

    ``coords`` is ``(B, L, 3)``; the result is ``(B, I, J)`` long with values ``0..len(boundaries)``.
    Uses the exact (non-matmul) Euclidean distance like upstream's ``cdist(..., "donot_use_mm")``.
    """
    ci = select_rows(coords, rows).float()
    cj = select_rows(coords, cols).float()
    d = torch.sqrt(((ci[:, :, None, :] - cj[:, None, :, :]) ** 2).sum(-1))
    return (d[..., None] > boundaries.to(d.device)).sum(-1).long()


def relpos_features(
    residue_index: Tensor,
    asym_id: Tensor,
    sym_id: Tensor,
    entity_id: Tensor,
    token_index: Tensor,
    *,
    r_max: int,
    s_max: int,
    rows: Tensor | None = None,
    cols: Tensor | None = None,
) -> Tensor:
    """AF3/ESMFold2 relative-position one-hots, ``(B, I, J, 2*(2r+2) + 1 + (2s+2))``.

    Residue and token deltas are clipped to ``[0, 2r]`` after adding ``r_max`` and sent to the
    extra bin ``2r+1`` when the pair is not on the same chain (residue) or not on the same chain
    and residue (token). The chain delta uses ``sym_id`` and goes to bin ``2s+1`` on the same chain.
    """
    ri, rj = select_rows(residue_index, rows), select_rows(residue_index, cols)
    ai, aj = select_rows(asym_id, rows), select_rows(asym_id, cols)
    si, sj = select_rows(sym_id, rows), select_rows(sym_id, cols)
    ei, ej = select_rows(entity_id, rows), select_rows(entity_id, cols)
    ti, tj = select_rows(token_index, rows), select_rows(token_index, cols)
    same_chain = ai[:, :, None] == aj[:, None, :]
    same_residue = ri[:, :, None] == rj[:, None, :]
    same_entity = ei[:, :, None] == ej[:, None, :]
    d_res = (ri[:, :, None] - rj[:, None, :] + r_max).clamp(0, 2 * r_max)
    d_res = torch.where(same_chain, d_res, torch.full_like(d_res, 2 * r_max + 1))
    d_tok = (ti[:, :, None] - tj[:, None, :] + r_max).clamp(0, 2 * r_max)
    d_tok = torch.where(same_chain & same_residue, d_tok, torch.full_like(d_tok, 2 * r_max + 1))
    d_chain = (si[:, :, None] - sj[:, None, :] + s_max).clamp(0, 2 * s_max)
    d_chain = torch.where(same_chain, torch.full_like(d_chain, 2 * s_max + 1), d_chain)
    return torch.cat(
        [
            F.one_hot(d_res.long(), 2 * r_max + 2).float(),
            F.one_hot(d_tok.long(), 2 * r_max + 2).float(),
            same_entity.float()[..., None],
            F.one_hot(d_chain.long(), 2 * s_max + 2).float(),
        ],
        dim=-1,
    )


class RelativePositionEncoding(nn.Module):
    """Linear embedding of :func:`relpos_features` (checkpoint name ``rel_pos.embed``)."""

    def __init__(self, width: int, *, r_max: int, s_max: int) -> None:
        super().__init__()
        self.r_max, self.s_max = r_max, s_max
        feature_dim = 2 * (2 * r_max + 2) + 1 + (2 * s_max + 2)
        self.embed = nn.Linear(feature_dim, width, bias=False)

    def forward(
        self,
        residue_index: Tensor,
        asym_id: Tensor,
        sym_id: Tensor,
        entity_id: Tensor,
        token_index: Tensor,
        rows: Tensor | None = None,
        cols: Tensor | None = None,
    ) -> Tensor:
        feats = relpos_features(
            residue_index, asym_id, sym_id, entity_id, token_index,
            r_max=self.r_max, s_max=self.s_max, rows=rows, cols=cols,
        )
        return self.embed(feats.to(self.embed.weight.dtype))
```

- [ ] **Step 3: `src/oplm/fold/data/featurize.py`**

```python
"""Sequences -> model inputs (protein-only data on the general representation; spec §4).

Mirrors ESMFold2's builder path (esm/models/esmfold2/prepare_input.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md): one token per residue, flat ragged atoms padded to a multiple of 32,
``ref_*`` atom features from the reference conformer table, AF3-style chain/entity/symmetry
indices, representative (CB, else CA) atom per token, and the empty token-bond matrix (always
present). Modifications: LM inputs are one row per chain with BOS/EOS (spec §4.5) instead of one
packed row per complex; optional token padding to a crop multiple. Reproduced quirk: an unknown
residue (res_type 22) gets peptide bonds to its chain neighbours in ``token_bonds``, as upstream's
atom-tokenized-residue logic does (prepare_input.py:1075-1110).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import (
    MOL_TYPE_PROTEIN,
    PROTEIN_1TO3,
    PROTEIN_RES_TYPES,
    UNK_RES_TYPE,
    ReferenceConformers,
    encode_atom_name,
)
from oplm.model.tokenization_oplm import VOCAB

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = ["ChainSpec", "FoldFeatures", "featurize"]

_BOS, _EOS, _PAD, _X = VOCAB["<cls>"], VOCAB["<eos>"], VOCAB["<pad>"], VOCAB["X"]


@dataclass(frozen=True)
class ChainSpec:
    """One protein chain and how many identical copies of it the complex contains."""

    sequence: str
    chain_id: str
    copies: int = 1


@dataclass
class FoldFeatures:
    """Batched (batch size 1) model inputs. Token tensors are ``(1, L)``; atom tensors ``(1, A)``."""

    token_index: Tensor
    residue_index: Tensor
    asym_id: Tensor
    entity_id: Tensor
    sym_id: Tensor
    mol_type: Tensor
    res_type: Tensor
    token_mask: Tensor  # bool
    token_bonds: Tensor  # (1, L, L, 1) float
    distogram_atom_idx: Tensor  # (1, L) flat atom index of the representative atom
    ref_pos: Tensor  # (1, A, 3)
    ref_element: Tensor  # (1, A) atomic number, 0 = pad
    ref_charge: Tensor  # (1, A) float formal charge
    ref_atom_name_chars: Tensor  # (1, A, 4) long
    ref_space_uid: Tensor  # (1, A)
    atom_mask: Tensor  # (1, A) bool
    atom_to_token: Tensor  # (1, A) long, 0 on pads
    lm_input_ids: Tensor  # (C, T) one row per chain, BOS ... EOS, right-padded
    lm_attention_mask: Tensor  # (C, T)
    lm_rows: Tensor  # (1, L) chain row of each token (0 on pads)
    lm_positions: Tensor  # (1, L) position of each token inside its LM row (0 on pads)
    chain_ids: list[str]

    @property
    def num_tokens(self) -> int:
        return int(self.token_index.shape[1])

    @property
    def num_atoms(self) -> int:
        return int(self.atom_mask.shape[1])

    @property
    def num_chains(self) -> int:
        return int(self.lm_input_ids.shape[0])

    def to(self, device: torch.device | str) -> FoldFeatures:
        """Return a copy with every tensor on ``device``."""
        moved = {
            f.name: (getattr(self, f.name).to(device) if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in fields(self)
        }
        return FoldFeatures(**moved)


def _expand_chains(chains: Sequence[ChainSpec]) -> list[tuple[str, str, int, int]]:
    """(chain_id, sequence, entity_id, sym_id) per chain copy; entities by first-seen sequence."""
    entities: dict[str, int] = {}
    copies_seen: dict[int, int] = {}
    out = []
    for spec in chains:
        if spec.copies < 1:
            raise ValueError(f"copies must be >= 1 for chain {spec.chain_id!r}")
        entity = entities.setdefault(spec.sequence, len(entities))
        for k in range(spec.copies):
            sym = copies_seen.get(entity, 0)
            copies_seen[entity] = sym + 1
            chain_id = spec.chain_id if k == 0 else f"{spec.chain_id}_{k + 1}"
            out.append((chain_id, spec.sequence, entity, sym))
    return out


def featurize(
    chains: Sequence[ChainSpec],
    *,
    conformers: ReferenceConformers | None = None,
    pad_tokens_to: int | None = None,
    atom_pad_multiple: int = 32,
) -> FoldFeatures:
    """Build :class:`FoldFeatures` for protein chains (no coordinates; prediction inputs).

    Args:
        chains: Chains in order; ``copies`` expands homo-oligomers with increasing ``sym_id``.
        conformers: Reference geometry table; the packaged one when ``None``.
        pad_tokens_to: Pad the token axis to this length (crop/bucket size); ``None`` = no padding.
        atom_pad_multiple: Atom axis padding multiple (upstream: 32).

    Raises:
        ValueError: ``pad_tokens_to`` is smaller than the token count, or a chain is empty.
    """
    table = conformers or ReferenceConformers.load()
    expanded = _expand_chains(chains)
    if any(not seq for _, seq, _, _ in expanded):
        raise ValueError("empty chain sequence")

    tok_residue, tok_asym, tok_entity, tok_sym, tok_res_type = [], [], [], [], []
    tok_rep_atom, tok_lm_row, tok_lm_pos = [], [], []
    atom_pos, atom_elem, atom_charge, atom_chars, atom_uid, atom_tok = [], [], [], [], [], []
    lm_rows: list[list[int]] = []
    chain_ids: list[str] = []
    token = 0
    for asym, (chain_id, seq, entity, sym) in enumerate(expanded):
        chain_ids.append(chain_id)
        lm_rows.append([_BOS] + [VOCAB.get(c, _X) if c in PROTEIN_1TO3 else _X for c in seq] + [_EOS])
        for res_i, letter in enumerate(seq):
            code = PROTEIN_1TO3.get(letter, "UNK")
            template = table[code]
            tok_residue.append(res_i)
            tok_asym.append(asym)
            tok_entity.append(entity)
            tok_sym.append(sym)
            tok_res_type.append(PROTEIN_RES_TYPES[code])
            tok_rep_atom.append(len(atom_pos) + template.representative_atom)
            tok_lm_row.append(asym)
            tok_lm_pos.append(res_i + 1)
            for name, element, charge, pos in zip(
                template.atoms, template.elements, template.charges, template.positions, strict=True
            ):
                atom_pos.append(pos)
                atom_elem.append(element)
                atom_charge.append(float(charge))
                atom_chars.append(encode_atom_name(name))
                atom_uid.append(token)
                atom_tok.append(token)
            token += 1

    n_tokens = token
    L = n_tokens if pad_tokens_to is None else pad_tokens_to
    if L < n_tokens:
        raise ValueError(f"pad_tokens_to={pad_tokens_to} is smaller than the token count {n_tokens}")
    n_atoms = len(atom_pos)
    A = -(-max(n_atoms, 1) // atom_pad_multiple) * atom_pad_multiple

    def tok(values: list[int], dtype: torch.dtype = torch.long) -> Tensor:
        t = torch.zeros(1, L, dtype=dtype)
        t[0, :n_tokens] = torch.tensor(values, dtype=dtype)
        return t

    res_type = tok(tok_res_type)
    token_bonds = torch.zeros(1, L, L, 1)
    asym_t = tok(tok_asym)
    for i in range(n_tokens):  # the upstream unknown-residue bond quirk (same-chain neighbours)
        if tok_res_type[i] != UNK_RES_TYPE:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < n_tokens and tok_asym[j] == tok_asym[i]:
                token_bonds[0, i, j, 0] = token_bonds[0, j, i, 0] = 1.0

    T = max(len(r) for r in lm_rows)
    lm_ids = torch.full((len(lm_rows), T), _PAD, dtype=torch.long)
    lm_mask = torch.zeros(len(lm_rows), T, dtype=torch.long)
    for r, row in enumerate(lm_rows):
        lm_ids[r, : len(row)] = torch.tensor(row)
        lm_mask[r, : len(row)] = 1

    def atoms(values: list, dtype: torch.dtype, trailing: tuple[int, ...] = ()) -> Tensor:
        t = torch.zeros(1, A, *trailing, dtype=dtype)
        if n_atoms:
            t[0, :n_atoms] = torch.tensor(values, dtype=dtype) if not isinstance(values[0], torch.Tensor) else torch.stack(values)
        return t

    token_mask = torch.zeros(1, L, dtype=torch.bool)
    token_mask[0, :n_tokens] = True
    atom_mask = torch.zeros(1, A, dtype=torch.bool)
    atom_mask[0, :n_atoms] = True
    return FoldFeatures(
        token_index=tok(list(range(n_tokens))),
        residue_index=tok(tok_residue),
        asym_id=asym_t,
        entity_id=tok(tok_entity),
        sym_id=tok(tok_sym),
        mol_type=tok([MOL_TYPE_PROTEIN] * n_tokens),
        res_type=res_type,
        token_mask=token_mask,
        token_bonds=token_bonds,
        distogram_atom_idx=tok(tok_rep_atom),
        ref_pos=atoms(atom_pos, torch.float32, (3,)),
        ref_element=atoms(atom_elem, torch.long),
        ref_charge=atoms(atom_charge, torch.float32),
        ref_atom_name_chars=atoms([list(c) for c in atom_chars], torch.long, (4,)),
        ref_space_uid=atoms(atom_uid, torch.long),
        atom_mask=atom_mask,
        atom_to_token=atoms(atom_tok, torch.long),
        lm_input_ids=lm_ids,
        lm_attention_mask=lm_mask,
        lm_rows=tok(tok_lm_row),
        lm_positions=tok(tok_lm_pos),
        chain_ids=chain_ids,
    )
```

(`atoms()` handles both scalar lists and the list of position tensors; `ruff format` wraps the long conditional. Lowercase letters are not in `PROTEIN_1TO3`, so they become `UNK`/`X`, matching upstream's case-sensitive lookup.)

- [ ] **Step 4: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_pair.py tests/fold/data -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/pair.py src/oplm/fold/data/featurize.py tests/fold/test_pair.py tests/fold/data/test_featurize.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`

```bash
git add src/oplm/fold/pair.py src/oplm/fold/data/featurize.py tests/fold/test_pair.py tests/fold/data/test_featurize.py
git commit -m "feat(fold): block-local pair features and the protein featurizer"
```


### Task 4: Pair stack and the recurrence

**Files:**
- Create: `src/oplm/fold/trunk.py`
- Test: `tests/fold/test_trunk.py`

**Interfaces:**
- Consumes: M0 `TriangleMultiplication(width, direction, *, eps, chunk_size, backend)` whose `forward(z, mask)` returns the delta; `FoldConfig` fields `pair_width`, `transition_expansion`, `pair_dropout`, `layer_norm_eps`, `trimul_chunk_size`, `trimul_backend`.
- Produces: `GatedMLP(width, hidden)` (children `gate_up_proj`, `down_proj`), `Transition(width, expansion=4, *, eps)` (children `norm`, `mlp`; returns the delta), `RowSharedDropout(p)`, `PairUpdateBlock(width, *, expansion, dropout, eps, chunk_size, trimul_backend)` (children `tri_mul_out`, `tri_mul_in`, `pair_transition`; returns the updated pair), `PairStack(num_blocks, width, **block_kwargs)` (child `layers`; attribute `gradient_checkpointing`), `Recurrence(width, *, coda_blocks, expansion, dropout, eps, chunk_size, trimul_backend)` with children `input_norm`, `log_delta`, `log_state_decay`, `input_matrix_continuous`, `out_proj`, `output_stack` and methods `dynamics() -> (a, B)`, `init_state(like, generator=None)`, `run(trunk, inject, *, z0, pair_mask, num_loops, grad_loops=None, return_states=False) -> (z, states)`, `readout(z, pair_mask)`. `pair_stack_kwargs(config) -> dict` builds the shared block kwargs from a `FoldConfig`; `cuda_bf16_autocast(enabled) -> ContextManager` is the shared "bf16 autocast on CUDA, nothing elsewhere" helper; `Recurrence.reset_parameters()` re-applies the upstream init without touching tensors HF already loaded.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_trunk.py`:

```python
"""Pair-update blocks, the pair stack and the recurrence against transcribed upstream math."""

from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.trunk import (
    GatedMLP,
    PairStack,
    PairUpdateBlock,
    Recurrence,
    Transition,
    pair_stack_kwargs,
)

_W = 32  # pair width (multiple of 32 for the trimul contract)


def _pair(b: int = 1, n: int = 6, w: int = _W, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, n, w, generator=g)


def test_gated_mlp_matches_upstream_swiglu_order() -> None:
    torch.manual_seed(0)
    mlp = GatedMLP(_W, 4 * _W)
    x = torch.randn(2, 5, _W)
    x1, x2 = mlp.gate_up_proj(x).chunk(2, dim=-1)  # upstream SwiGLUMLP: silu(FIRST half) * SECOND half
    torch.testing.assert_close(mlp(x), mlp.down_proj(F.silu(x1) * x2))
    assert mlp.gate_up_proj.bias is None and mlp.down_proj.bias is None


def test_transition_returns_delta_with_checkpoint_names() -> None:
    t = Transition(_W, 4)
    assert set(t.state_dict()) == {
        "norm.weight", "norm.bias", "mlp.gate_up_proj.weight", "mlp.down_proj.weight",
    }
    assert t.mlp.gate_up_proj.weight.shape == (8 * _W, _W)
    x = torch.randn(1, 3, 3, _W)
    torch.testing.assert_close(t(x), t.mlp(t.norm(x)))


def test_pair_update_block_is_the_upstream_residual_composition() -> None:
    torch.manual_seed(0)
    block = PairUpdateBlock(_W, expansion=4, dropout=0.25, eps=1e-5, chunk_size=64, trimul_backend="reference")
    block.eval()  # dropout is a no-op in eval (and at p=0 in train)
    z = _pair()
    mask = torch.ones(1, 6, 6)
    mask[0, 4:, :] = mask[0, :, 4:] = 0.0
    expected = z + block.tri_mul_out(z, mask)
    expected = expected + block.tri_mul_in(expected, mask)
    expected = expected + block.pair_transition(expected)
    torch.testing.assert_close(block(z, mask), expected)
    names = set(block.state_dict())
    assert "tri_mul_out.proj_bundle.weight" in names and "pair_transition.mlp.down_proj.weight" in names
    assert not any("dropout" in n for n in names)


def test_row_shared_dropout_shares_the_mask_along_rows_and_rescales() -> None:
    torch.manual_seed(0)
    block = PairUpdateBlock(_W, dropout=0.5).train()
    delta = torch.ones(2, 7, 5, _W)
    out = block.dropout(delta)
    assert set(out.unique().tolist()) <= {0.0, 2.0}
    assert torch.equal(out[:, :1].expand_as(out), out)  # identical across the row axis


def test_pair_stack_checkpointing_matches_plain_gradients() -> None:
    torch.manual_seed(0)
    stack = PairStack(2, _W, trimul_backend="reference").train()
    z = _pair().requires_grad_(True)
    out = stack(z)
    (g_plain,) = torch.autograd.grad(out.square().sum(), z)
    stack.gradient_checkpointing = True
    (g_ckpt,) = torch.autograd.grad(stack(z).square().sum(), z)
    torch.testing.assert_close(g_ckpt, g_plain)
    assert [n for n, _ in stack.named_children()] == ["layers"] and len(stack.layers) == 2


def test_recurrence_dynamics_and_initial_values_match_upstream() -> None:
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference")
    a, b = rec.dynamics()
    delta = F.softplus(rec.log_delta)
    torch.testing.assert_close(a, torch.exp(-delta * torch.exp(rec.log_state_decay)))
    torch.testing.assert_close(b, delta[:, None] * rec.input_matrix_continuous)
    # init: delta0 = 0.5 ln 5 -> a = sqrt(1/5); B = delta0 * I; out_proj = I
    torch.testing.assert_close(a, torch.full((_W,), math.sqrt(0.2)))
    torch.testing.assert_close(b, 0.5 * math.log(5.0) * torch.eye(_W))
    torch.testing.assert_close(rec.out_proj.weight, torch.eye(_W))
    assert set(rec.state_dict()) >= {
        "input_norm.weight", "log_delta", "log_state_decay", "input_matrix_continuous",
        "out_proj.weight", "output_stack.layers.0.tri_mul_out.norm_start.weight",
    }


def test_recurrence_init_state_is_truncated_normal_in_fp32_then_cast() -> None:
    rec = Recurrence(256, coda_blocks=1, trimul_backend="reference")
    like = torch.zeros(1, 40, 40, 256, dtype=torch.bfloat16)
    z0 = rec.init_state(like, generator=torch.Generator().manual_seed(0))
    std = math.sqrt(2.0 / (5.0 * 256))
    assert z0.dtype == torch.bfloat16 and z0.shape == like.shape
    assert abs(z0.float().std().item() - std) < 0.1 * std and z0.float().abs().max() <= 3 * std + 1e-3
    again = rec.init_state(like, generator=torch.Generator().manual_seed(0))
    assert torch.equal(z0, again)


def test_recurrence_run_matches_a_transcribed_loop_and_records_states() -> None:
    torch.manual_seed(0)
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference").eval()
    trunk = PairStack(1, _W, trimul_backend="reference").eval()
    z_init, lm = _pair(seed=1), _pair(seed=2)
    mask = torch.ones(1, 6, 6)
    z0 = rec.init_state(z_init, generator=torch.Generator().manual_seed(3))

    with torch.no_grad():
        z, states = rec.run(
            trunk, lambda _t: z_init + lm, z0=z0, pair_mask=mask, num_loops=3, return_states=True
        )
        # upstream _run_one_loop, transcribed
        a, b_mat = rec.dynamics()
        a = a.view(1, 1, 1, -1)
        ref = z0
        for _ in range(3):
            injected = rec.input_norm(z_init + lm)
            ref = a * ref + F.linear(injected, b_mat)
            ref = trunk(ref, mask)
    torch.testing.assert_close(z, ref)
    assert len(states) == 3 and torch.equal(states[-1], z)


def test_recurrence_grad_loops_truncate_backpropagation() -> None:
    torch.manual_seed(0)
    rec = Recurrence(_W, coda_blocks=1, trimul_backend="reference").train()
    trunk = PairStack(1, _W, trimul_backend="reference").train()
    z_init = _pair(seed=1).requires_grad_(True)
    z0 = rec.init_state(z_init.detach()).requires_grad_(True)
    z, _ = rec.run(trunk, lambda _t: z_init, z0=z0, pair_mask=None, num_loops=3, grad_loops=1)
    z.sum().backward()
    assert z0.grad is None  # the first two loops ran under no_grad
    assert z_init.grad is not None and rec.log_delta.grad is not None


def test_pair_stack_kwargs_come_from_config() -> None:
    cfg = FoldConfig(trunk_dropout=0.1, trimul_backend="reference", trimul_chunk_size=None)
    kw = pair_stack_kwargs(cfg)
    assert kw == {
        "expansion": 4, "dropout": 0.1, "eps": 1e-5, "chunk_size": None, "trimul_backend": "reference",
    }
    stack = PairStack(cfg.trunk_blocks, cfg.pair_width, **kw)
    assert len(stack.layers) == 24
    assert stack.layers[0].pair_transition.mlp.gate_up_proj.weight.shape == (2048, 256)


@pytest.mark.parametrize("bad", [{"expansion": 0}, {"dropout": 1.0}])
def test_block_rejects_bad_arguments(bad: dict) -> None:
    with pytest.raises(ValueError):
        PairUpdateBlock(_W, **bad)
```

Run: `.venv/bin/python -m pytest tests/fold/test_trunk.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.trunk'`.

- [ ] **Step 2: `src/oplm/fold/trunk.py`**

```python
"""Pair stack: SwiGLU transitions, pair-update blocks, and the recurrence ("parcae").

Ported from Biohub's ESMFold2 ``Transition``/``PairUpdateBlock``/``FoldingTrunk`` and the
``parcae_*`` recurrence of ``EsmFold2Model`` (esm/models/esmfold2/{layers,model}.py,
Apache-2.0; see THIRD_PARTY_NOTICES.md). Parameter names follow the released HF checkpoint:
``layers.N.{tri_mul_out,tri_mul_in,pair_transition.{norm,mlp.gate_up_proj,mlp.down_proj}}`` and
``parcae.{input_norm,log_delta,log_state_decay,input_matrix_continuous,out_proj,output_stack}``.
Modifications: the triangle updates are milestone 0's :class:`TriangleMultiplication`; the
recurrence takes the loop count, the number of gradient-carrying loops and the initial state
as arguments (spec §5.6; upstream always runs ``num_loops + 1`` loops without gradient);
row-shared dropout is a real module where upstream has ``DropoutResidual(0.0)``; the pair
stack supports activation checkpointing per block.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from oplm.fold.trimul import TriangleMultiplication

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "GatedMLP",
    "PairStack",
    "PairUpdateBlock",
    "Recurrence",
    "RowSharedDropout",
    "Transition",
    "cuda_bf16_autocast",
    "pair_stack_kwargs",
]


def cuda_bf16_autocast(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """bf16 autocast on CUDA when ``enabled``; a null context otherwise (never warns on CPU)."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def unloaded(p: Tensor) -> bool:
    """True unless transformers already loaded ``p`` from a checkpoint (its ``_is_hf_initialized``)."""
    return not getattr(p, "_is_hf_initialized", False)


class GatedMLP(nn.Module):
    """``down_proj(silu(a) * b)`` with ``a, b = gate_up_proj(x).chunk(2)`` (upstream SwiGLU order)."""

    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(a) * b)


class Transition(nn.Module):
    """Pre-norm gated MLP returning the delta; the caller adds the residual."""

    def __init__(self, width: int, expansion: int = 4, *, eps: float = 1e-5) -> None:
        super().__init__()
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1; got {expansion!r}")
        self.norm = nn.LayerNorm(width, eps=eps)
        self.mlp = GatedMLP(width, expansion * width)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.norm(x))


class RowSharedDropout(nn.Module):
    """Dropout with one mask per (batch, column, channel), shared along the row axis.

    AF2-style "row-wise" dropout for the triangle updates; identity in eval or at ``p == 0``.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {p!r}")
        self.p = p

    def forward(self, delta: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return delta
        shape = (delta.shape[0], 1, *delta.shape[2:])
        keep = torch.rand(shape, device=delta.device) >= self.p
        return delta * keep.to(delta.dtype) / (1.0 - self.p)


class PairUpdateBlock(nn.Module):
    """``z += drop(tri_out(z)); z += drop(tri_in(z)); z += transition(z)`` (upstream block)."""

    def __init__(
        self,
        width: int,
        *,
        expansion: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-5,
        chunk_size: int | None = 64,
        trimul_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(
            width, "outgoing", eps=eps, chunk_size=chunk_size, backend=trimul_backend  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        )
        self.tri_mul_in = TriangleMultiplication(
            width, "incoming", eps=eps, chunk_size=chunk_size, backend=trimul_backend  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        )
        self.pair_transition = Transition(width, expansion, eps=eps)
        self.dropout = RowSharedDropout(dropout)

    def forward(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        z = z + self.dropout(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout(self.tri_mul_in(z, pair_mask))
        return z + self.pair_transition(z)


class PairStack(nn.Module):
    """A sequence of :class:`PairUpdateBlock` (checkpoint name ``layers.N``)."""

    def __init__(self, num_blocks: int, width: int, **block_kwargs: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [PairUpdateBlock(width, **block_kwargs) for _ in range(num_blocks)]
        )
        self.gradient_checkpointing = False

    def forward(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(layer, z, pair_mask, use_reentrant=False)  # ty: ignore[invalid-assignment]  # checkpoint is untyped
            else:
                z = layer(z, pair_mask)
        return z


def pair_stack_kwargs(config: FoldConfig) -> dict[str, Any]:
    """Block kwargs shared by every pair stack built from ``config``."""
    return {
        "expansion": config.transition_expansion,
        "dropout": config.trunk_dropout,
        "eps": config.layer_norm_eps,
        "chunk_size": config.trimul_chunk_size,
        "trimul_backend": config.trimul_backend,
    }


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Recurrence(nn.Module):
    """``z_t = trunk(a ⊙ z_{t-1} + input_norm(inject_t) @ Bᵀ)`` and the readout + coda.

    ``delta = softplus(log_delta)``, ``a = exp(-delta · exp(log_state_decay))``,
    ``B = delta[:, None] · input_matrix_continuous`` (upstream ``_discretized_dynamics``). The
    state has the dtype of ``z0`` (bf16 under autocast, fp32 otherwise), as upstream.
    """

    def __init__(self, width: int, *, coda_blocks: int, **block_kwargs: Any) -> None:
        super().__init__()
        eps = block_kwargs.get("eps", 1e-5)
        self.width = width
        self.input_norm = nn.LayerNorm(width, eps=eps)
        self.log_delta = nn.Parameter(torch.empty(width))
        self.log_state_decay = nn.Parameter(torch.empty(width))
        self.input_matrix_continuous = nn.Parameter(torch.empty(width, width))
        self.out_proj = nn.Linear(width, width, bias=False)
        self.out_proj._init_identity = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        self.output_stack = PairStack(coda_blocks, width, **block_kwargs)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Upstream init: ``delta0 = 0.5 ln 5`` (so ``a = sqrt(1/5)``), ``B = delta0 I``, ``out_proj = I``.

        Skips tensors transformers already loaded, so ``_init_weights`` may call it freely.
        """
        delta0 = 0.5 * math.log(5.0)
        if unloaded(self.log_delta):
            nn.init.constant_(self.log_delta, _inverse_softplus(delta0))
        if unloaded(self.log_state_decay):
            nn.init.zeros_(self.log_state_decay)
        for p in (self.input_matrix_continuous, self.out_proj.weight):
            if unloaded(p):
                with torch.no_grad():
                    p.copy_(torch.eye(self.width, device=p.device, dtype=p.dtype))

    def dynamics(self) -> tuple[Tensor, Tensor]:
        """``(a, B)``: the per-channel decay ``(D,)`` and the input matrix ``(D, D)``."""
        delta = F.softplus(self.log_delta)
        a = torch.exp(-delta * torch.exp(self.log_state_decay))
        return a, delta[:, None] * self.input_matrix_continuous

    def init_state(self, like: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Truncated-normal ``z_0`` (std ``sqrt(2 / (5 D))``, clipped at 3σ), drawn in fp32."""
        std = math.sqrt(2.0 / (5.0 * like.shape[-1]))
        state = torch.empty(like.shape, dtype=torch.float32, device=like.device)
        nn.init.trunc_normal_(state, 0.0, std, -3 * std, 3 * std, generator=generator)
        return state.to(like.dtype)

    def run(
        self,
        trunk: PairStack,
        inject: Callable[[int], Tensor],
        *,
        z0: Tensor,
        pair_mask: Tensor | None,
        num_loops: int,
        grad_loops: int | None = None,
        return_states: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        """Iterate the recurrence ``num_loops`` times from ``z0``.

        Args:
            trunk: The shared pair stack applied after every state update.
            inject: ``inject(t)`` returns the pair injected at loop ``t`` (``z_init`` plus the
                LM pair encoding); it is called inside the loop so per-loop dropout and the LM
                encoder run under the loop's gradient mode.
            z0: Initial state; dtype sets the state dtype.
            pair_mask: ``(B, L, L)`` float pair validity, or ``None``.
            num_loops: Iterations executed (upstream's ``num_loops + 1``).
            grad_loops: Only the last ``grad_loops`` iterations build a graph (truncated
                BPTT, spec §5.6); ``None`` keeps the caller's gradient mode throughout.
            return_states: Also return the state after every loop (parity fixtures).
        """
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1; got {num_loops!r}")
        a, b_mat = self.dynamics()
        a = a.to(z0.dtype).view(1, 1, 1, -1)
        b_mat = b_mat.to(z0.dtype)
        no_grad_loops = 0 if grad_loops is None else max(num_loops - grad_loops, 0)
        z, states = z0, []
        for t in range(num_loops):
            ctx = torch.no_grad() if t < no_grad_loops else contextlib.nullcontext()
            with ctx:
                injected = self.input_norm(inject(t))
                z = a * z + F.linear(injected.to(z.dtype), b_mat)
                z = trunk(z, pair_mask)
            if return_states:
                states.append(z)
        return z, states

    def readout(self, z: Tensor, pair_mask: Tensor | None = None) -> Tensor:
        """``output_stack(out_proj(z))`` — the pair every head consumes."""
        return self.output_stack(self.out_proj(z), pair_mask)
```

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_trunk.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/trunk.py tests/fold/test_trunk.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: 11 passed. If ty rejects the `trimul_backend: str` → `Backend` literal mismatch differently than the inline ignores assume, narrow the annotation to `Backend` imported from `oplm.fold.trimul` instead of ignoring.

```bash
git add src/oplm/fold/trunk.py tests/fold/test_trunk.py
git commit -m "feat(fold): pair-update blocks, pair stack and the parcae recurrence"
```

---

### Task 5: Atom machinery and the inputs embedder

**Files:**
- Create: `src/oplm/fold/atoms.py`
- Test: `tests/fold/test_atoms.py`

**Interfaces:**
- Consumes: Task 1 `sliding_window_attention(q, k, v, valid, half_window, *, backend, block_mask)`; Task 3 `RelativePositionEncoding`, `FoldFeatures`, `featurize`; Task 4 `GatedMLP`; `FoldConfig` atom fields.
- Produces: `build_atom_features(ref_pos, ref_charge, atom_mask, ref_element, ref_atom_name_chars, *, max_atomic_number, name_vocab) -> Tensor[B, A, 389]`,
  `build_3d_rope(ref_pos, ref_space_uid, *, head_dim, spatial_pairs_per_axis, uid_pairs, spatial_base, uid_base) -> (cos, sin)` (bf16, `(B, A, head_dim // 2)`), `apply_rotary_3d(x, cos, sin)`,
  `gather_token_to_atom(token_features, atom_to_token)`, `scatter_atom_to_token_mean(atom_features, atom_to_token, n_tokens, atom_mask)`, `intra_token_index(atom_to_token)`, `atom_ffn_hidden(width, expansion=2)`,
  `AtomAttention(width, heads, half_window, *, backend)` (children `q_proj, k_proj, v_proj, gate_proj, o_proj`),
  `AtomBlock(width, heads, half_window, *, expansion=2, backend)` (children `adaln_linear`, `self_attn`, `mlp`),
  `AtomEncoder(config, *, out_width, num_blocks, heads)` with `embed(features) -> c` and `forward(q, c, cos, sin, valid, atom_to_token, n_tokens, *, block_mask=None) -> (a, q)` (children `atom_linear, atom_norm, layers, atom_to_token_linear`),
  `AtomDecoder(config, *, num_blocks, heads)` with `forward(a, q, c, cos, sin, valid, atom_to_token, *, block_mask=None) -> Tensor[B, A, 3]` (children `token_to_atom_linear, layers, norm, output_linear`),
  `InputsEmbedding` dataclass (`s_inputs, z_init, relpos, bonds, atom_features, rope`), `InputsEmbedder(config)` (children `atom_encoder, pair_init_1, pair_init_2, rel_pos, token_bonds`) with `forward(features, *, block_mask=None) -> InputsEmbedding`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_atoms.py`:

```python
"""Atom features, 3D RoPE, SWA atom attention/blocks, encoders and the inputs embedder."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from oplm.fold.atoms import (
    AtomAttention,
    AtomBlock,
    AtomDecoder,
    AtomEncoder,
    InputsEmbedder,
    apply_rotary_3d,
    atom_ffn_hidden,
    build_3d_rope,
    build_atom_features,
    gather_token_to_atom,
    intra_token_index,
    scatter_atom_to_token_mean,
)
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize

_CFG = FoldConfig(trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, attention_backend="dense")


def test_atom_feature_layout_matches_upstream_order() -> None:
    f = featurize([ChainSpec("MK", "A")])
    feats = build_atom_features(
        f.ref_pos, f.ref_charge, f.atom_mask, f.ref_element, f.ref_atom_name_chars,
        max_atomic_number=128, name_vocab=64,
    )
    assert feats.shape == (1, 32, 389)
    torch.testing.assert_close(feats[0, :, :3], f.ref_pos[0])
    assert feats[0, 8 + 8, 3] == 1.0  # LYS NZ charge
    assert feats[0, :17, 4].eq(1).all() and feats[0, 17:, 4].eq(0).all()  # mask channel
    assert feats[0, 0, 5 + 7] == 1.0 and feats[0, 0, 5:133].sum() == 1  # N: element one-hot
    assert feats[0, 1, 133 + 0 * 64 + 35] == 1.0 and feats[0, 1, 133 + 1 * 64 + 33] == 1.0  # "CA"
    assert feats[0, 17:].abs().sum() == 0  # padded atoms contribute nothing


def _upstream_rope(ref_pos: torch.Tensor, uid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ESMFold2 build_3d_rope (layers.py), transcribed with the released constants."""
    B, N = ref_pos.shape[:2]
    half_dim = 16
    sp = 1.0 / (20.0 ** (torch.arange(0, 2, dtype=torch.float32) / 2))
    ui = 1.0 / (10000.0 ** (torch.arange(0, 10, dtype=torch.float32) / 10))
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), sp).reshape(B, N, 6)
    uidf = torch.einsum("bn,k->bnk", uid.float(), ui)
    freqs = torch.cat([spatial, uidf], dim=-1)
    if freqs.shape[-1] < half_dim:
        freqs = torch.cat([freqs, torch.zeros(B, N, half_dim - freqs.shape[-1])], dim=-1)
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def test_3d_rope_matches_transcription_and_neox_rotation() -> None:
    g = torch.Generator().manual_seed(0)
    pos = torch.randn(2, 9, 3, generator=g) * 5
    uid = torch.randint(0, 40, (2, 9), generator=g)
    cos, sin = build_3d_rope(
        pos, uid, head_dim=32, spatial_pairs_per_axis=2, uid_pairs=10, spatial_base=20.0, uid_base=10000.0
    )
    ref_cos, ref_sin = _upstream_rope(pos, uid)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin) and cos.dtype == torch.bfloat16
    x = torch.randn(2, 9, 4, 32, generator=g)
    x1, x2 = x.chunk(2, dim=-1)
    c = cos[:, :, None, :].repeat(1, 1, 1, 2)
    s = sin[:, :, None, :].repeat(1, 1, 1, 2)
    expected = x * c + torch.cat((-x2, x1), dim=-1) * s
    torch.testing.assert_close(apply_rotary_3d(x, cos, sin), expected)


def test_gather_scatter_and_intra_index() -> None:
    tok = torch.randn(1, 3, 4)
    a2t = torch.tensor([[0, 0, 1, 2, 2, 0, 0]])  # last two are pads mapped to 0
    mask = torch.tensor([[True, True, True, True, True, False, False]])
    torch.testing.assert_close(gather_token_to_atom(tok, a2t)[0, 2], tok[0, 1])
    atoms = torch.arange(7, dtype=torch.float32)[None, :, None].expand(1, 7, 2)
    pooled = scatter_atom_to_token_mean(atoms, a2t, 3, mask)
    torch.testing.assert_close(pooled[0, :, 0], torch.tensor([0.5, 2.0, 3.5]))
    assert intra_token_index(a2t)[0].tolist() == [0, 1, 0, 0, 1, 0, 1]


def test_scatter_uses_explicit_token_count() -> None:
    """Review Focus 1: a trailing token with no atoms must not shorten the token axis."""
    atoms = torch.ones(1, 4, 2)
    a2t = torch.tensor([[0, 0, 1, 1]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    pooled = scatter_atom_to_token_mean(atoms, a2t, 3, mask)
    assert pooled.shape == (1, 3, 2) and pooled[0, 2].abs().sum() == 0


def _upstream_swa_attention(attn: AtomAttention, x: torch.Tensor, cos, sin, valid) -> torch.Tensor:
    """ESMFold2 SWA3DRoPEAttention.forward non-flash branch, transcribed onto our projections."""
    B, N, _ = x.shape
    H, D = attn.heads, attn.head_dim
    q = attn.q_proj(x).view(B, N, H, D)
    k = attn.k_proj(x).view(B, N, H, D)
    v = attn.v_proj(x).view(B, N, H, D)
    q = F.rms_norm(q, (D,)).to(q.dtype)
    k = F.rms_norm(k, (D,)).to(k.dtype)
    q, k = apply_rotary_3d(q, cos, sin), apply_rotary_3d(k, cos, sin)
    in_dtype = q.dtype
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    rank = torch.cumsum(valid, dim=1) - 1
    within = (rank.unsqueeze(2) - rank.unsqueeze(1)).abs() <= attn.half_window
    allowed = within & valid.unsqueeze(1) & valid.unsqueeze(2)
    allowed |= torch.eye(N, dtype=torch.bool)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed.unsqueeze(1), scale=D**-0.5
    ).transpose(1, 2)
    out = out * valid.unsqueeze(-1).unsqueeze(-1)
    out = out.to(in_dtype).reshape(B, N, -1)
    out = out * torch.sigmoid(attn.gate_proj(x))
    return attn.o_proj(out)


def test_atom_attention_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    attn = AtomAttention(32, 2, 2, backend="dense")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 10, 32, generator=g)
    pos = torch.randn(2, 10, 3, generator=g)
    uid = torch.arange(10)[None].expand(2, 10)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[1, 7:] = False
    cos, sin = build_3d_rope(pos, uid, head_dim=16, spatial_pairs_per_axis=2, uid_pairs=2, spatial_base=20.0, uid_base=10000.0)
    out = attn(x, cos, sin, valid)
    torch.testing.assert_close(out, _upstream_swa_attention(attn, x, cos, sin, valid), atol=1e-2, rtol=1e-2)
    assert out[1, 7:].abs().sum() == 0  # invalid atoms: attention output zero, gate*0 = 0


def test_atom_block_chunk_order_and_zero_init() -> None:
    torch.manual_seed(0)
    block = AtomBlock(32, 2, 4, backend="dense")
    assert block.adaln_linear.weight.abs().sum() == 0 and block.adaln_linear.bias is None
    assert block.mlp.gate_up_proj.weight.shape == (2 * atom_ffn_hidden(32), 32)
    x, c = torch.randn(1, 6, 32), torch.randn(1, 6, 32)
    pos, uid = torch.randn(1, 6, 3), torch.arange(6)[None]
    cos, sin = build_3d_rope(pos, uid, head_dim=16, spatial_pairs_per_axis=2, uid_pairs=2, spatial_base=20.0, uid_base=10000.0)
    valid = torch.ones(1, 6, dtype=torch.bool)
    torch.testing.assert_close(block(x, c, cos, sin, valid), x)  # all gates zero -> identity
    # gate_a is chunk index 2 of [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]
    with torch.no_grad():
        block.adaln_linear.weight[2 * 32 : 3 * 32] = 0.1  # only gate_a is non-zero
    out = block(x, c, cos, sin, valid)
    gate_a = block.adaln_linear(F.silu(c)).chunk(6, dim=-1)[2]
    expected = x + gate_a * block.self_attn(F.rms_norm(x, (32,)), cos, sin, valid)
    torch.testing.assert_close(out, expected)


def test_encoder_decoder_names_and_shapes() -> None:
    enc = AtomEncoder(_CFG, out_width=_CFG.inputs_token_width, num_blocks=3, heads=4)
    dec = AtomDecoder(_CFG, num_blocks=3, heads=4)
    enc_names, dec_names = set(enc.state_dict()), set(dec.state_dict())
    assert {
        "atom_linear.weight", "atom_norm.weight", "atom_norm.bias", "atom_to_token_linear.weight",
        "layers.2.adaln_linear.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.gate_proj.weight",
        "layers.0.self_attn.o_proj.weight", "layers.0.mlp.gate_up_proj.weight", "layers.0.mlp.down_proj.weight",
    } <= enc_names
    assert {"token_to_atom_linear.weight", "norm.weight", "output_linear.weight", "layers.2.mlp.down_proj.weight"} <= dec_names
    assert enc.atom_linear.weight.shape == (128, 389) and enc.atom_to_token_linear.weight.shape == (384, 128)
    assert enc.layers[0].adaln_linear.weight.shape == (768, 128) and enc.layers[0].mlp.gate_up_proj.weight.shape == (512, 128)
    assert dec.token_to_atom_linear.weight.shape == (128, 768) and dec.output_linear.weight.shape == (3, 128)
    assert len(enc_names) == 22 and len(dec_names) == 22


def test_inputs_embedder_shapes_padding_and_names() -> None:
    torch.manual_seed(0)
    emb = InputsEmbedder(_CFG).eval()
    f = featurize([ChainSpec("MKV", "A")], pad_tokens_to=8)
    with torch.no_grad():
        out = emb(f)
    assert out.s_inputs.shape == (1, 8, 451) and out.z_init.shape == (1, 8, 8, 256)
    assert out.relpos.shape == out.bonds.shape == (1, 8, 8, 256)
    # order: [atom aggregation 384 | res_type one-hot 33 | profile = one-hot 33 | deletion_mean 0]
    assert out.s_inputs[0, 0, 384 + 14] == 1 and out.s_inputs[0, 0, 384 + 33 + 14] == 1
    assert out.s_inputs[0, :, 450].abs().sum() == 0
    assert out.s_inputs[0, 3:].abs().sum() == 0  # padded tokens: no atoms, zeroed one-hots
    assert (out.s_inputs[0, :3, :384] >= 0).all()  # relu'd aggregation
    torch.testing.assert_close(out.z_init[0, 5, 6], out.relpos[0, 5, 6] + out.bonds[0, 5, 6])  # bias-free inits
    assert {
        "atom_encoder.atom_linear.weight", "pair_init_1.weight", "pair_init_2.weight",
        "rel_pos.embed.weight", "token_bonds.weight",
    } <= set(emb.state_dict())
    assert emb.rel_pos.embed.weight.shape == (256, 139) and emb.token_bonds.weight.shape == (256, 1)
```

Run: `.venv/bin/python -m pytest tests/fold/test_atoms.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.atoms'`.

- [ ] **Step 2: `src/oplm/fold/atoms.py`**

```python
"""Atom-level machinery: reference features, 3D RoPE, SWA atom blocks, encoders, inputs embedder.

Ported from Biohub's ESMFold2 ``build_3d_rope``/``apply_rotary_emb_3d``, ``SWA3DRoPEAttention``,
``SWAAtomBlock``, ``EsmFold2AtomEncoder``/``Decoder`` and ``InputsEmbedder``
(esm/models/esmfold2/layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md). Parameter names
follow the released HF checkpoint (``layers.N.{adaln_linear, self_attn.{q,k,v,gate,o}_proj,
mlp.{gate_up_proj,down_proj}}``, ``atom_linear``, ``atom_norm``, ``atom_to_token_linear``,
``token_to_atom_linear``, ``norm``, ``output_linear``, ``pair_init_{1,2}``, ``rel_pos.embed``,
``token_bonds``). Modifications: the windowed attention is milestone 0's
``sliding_window_attention`` (FlexAttention / dense oracle) with a precomputed block mask;
the token count is explicit (upstream derives it from ``atom_to_token.max() + 1``); the
diffusion head's ``coords_linear`` offset enters the encoder through its ``q`` argument.
Precision: Q/K/V are cast to bf16 before attention and the RoPE tables are bf16, exactly as
upstream (parity requires it); everything else follows the caller's autocast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.attention import sliding_window_attention
from oplm.fold.pair import RelativePositionEncoding
from oplm.fold.trunk import GatedMLP

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import FoldFeatures

__all__ = [
    "AtomAttention",
    "AtomBlock",
    "AtomDecoder",
    "AtomEncoder",
    "InputsEmbedder",
    "InputsEmbedding",
    "apply_rotary_3d",
    "atom_ffn_hidden",
    "build_3d_rope",
    "build_atom_features",
    "gather_token_to_atom",
    "intra_token_index",
    "scatter_atom_to_token_mean",
]


def build_atom_features(
    ref_pos: Tensor,
    ref_charge: Tensor,
    atom_mask: Tensor,
    ref_element: Tensor,
    ref_atom_name_chars: Tensor,
    *,
    max_atomic_number: int,
    name_vocab: int,
) -> Tensor:
    """``[pos 3 | charge 1 | mask 1 | element one-hot | name chars one-hot]``, zero on pads."""
    m = atom_mask.to(ref_pos.dtype)[..., None]
    element = F.one_hot(ref_element.long(), max_atomic_number).to(ref_pos.dtype) * m
    chars = F.one_hot(ref_atom_name_chars.long(), name_vocab).to(ref_pos.dtype) * m[..., None]
    return torch.cat(
        [ref_pos, ref_charge.to(ref_pos.dtype)[..., None], m, element, chars.flatten(-2)], dim=-1
    )


def build_3d_rope(
    ref_pos: Tensor,
    ref_space_uid: Tensor,
    *,
    head_dim: int,
    spatial_pairs_per_axis: int,
    uid_pairs: int,
    spatial_base: float,
    uid_base: float,
) -> tuple[Tensor, Tensor]:
    """bf16 ``(cos, sin)`` of shape ``(B, A, head_dim // 2)``: 3 axes × spatial pairs, then uid pairs."""
    device = ref_pos.device
    half = head_dim // 2
    sp_inv = 1.0 / (
        spatial_base
        ** (torch.arange(spatial_pairs_per_axis, dtype=torch.float32, device=device) / spatial_pairs_per_axis)
    )
    uid_inv = 1.0 / (
        uid_base ** (torch.arange(uid_pairs, dtype=torch.float32, device=device) / uid_pairs)
    )
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), sp_inv).flatten(-2)
    uid = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv)
    freqs = torch.cat([spatial, uid], dim=-1)
    if freqs.shape[-1] < half:
        freqs = F.pad(freqs, (0, half - freqs.shape[-1]))
    return freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)


def apply_rotary_3d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """NeoX half-split rotation of the first ``2 * cos.shape[-1]`` channels of ``x (B, A, H, D)``."""
    ro = cos.shape[-1] * 2
    c = cos[:, :, None, :].repeat(1, 1, 1, 2)
    s = sin[:, :, None, :].repeat(1, 1, 1, 2)
    xr, rest = x[..., :ro], x[..., ro:]
    x1, x2 = xr.chunk(2, dim=-1)
    rotated = xr * c + torch.cat((-x2, x1), dim=-1) * s
    return torch.cat([rotated, rest], dim=-1)


def gather_token_to_atom(token_features: Tensor, atom_to_token: Tensor) -> Tensor:
    """``(B, L, D), (B, A) -> (B, A, D)``."""
    idx = atom_to_token[..., None].expand(-1, -1, token_features.shape[-1])
    return torch.gather(token_features, 1, idx)


def scatter_atom_to_token_mean(
    atom_features: Tensor, atom_to_token: Tensor, n_tokens: int, atom_mask: Tensor
) -> Tensor:
    """Mean of each token's valid atoms, ``(B, A, D) -> (B, n_tokens, D)``; atom-less tokens are 0."""
    B, A, D = atom_features.shape
    idx = torch.where(atom_mask, atom_to_token, torch.full_like(atom_to_token, n_tokens))
    out = torch.zeros(B, n_tokens + 1, D, device=atom_features.device, dtype=atom_features.dtype)
    out.scatter_reduce_(1, idx[..., None].expand(B, A, D), atom_features, reduce="mean", include_self=False)
    return out[:, :n_tokens]


def intra_token_index(atom_to_token: Tensor) -> Tensor:
    """0-based slot of each atom inside its (contiguous) token, ``(B, A)``."""
    same_as_prev = F.pad(atom_to_token[:, 1:] == atom_to_token[:, :-1], (1, 0), value=False)
    cumsum = torch.cumsum(torch.ones_like(atom_to_token), dim=-1)
    group_start = torch.cummax(cumsum.masked_fill(same_as_prev, 0), dim=-1).values
    return cumsum - group_start


def atom_ffn_hidden(width: int, expansion: int = 2) -> int:
    """Upstream ``SwiGLUFFN`` hidden size: ``((expansion * (width // 3) * 2) + 255) // 256 * 256``."""
    return ((expansion * (width // 3) * 2) + 255) // 256 * 256


class AtomAttention(nn.Module):
    """Sliding-window attention with qk-RMSNorm, 3D RoPE, bf16 Q/K/V and a sigmoid output gate."""

    def __init__(self, width: int, heads: int, half_window: int, *, backend: str = "auto") -> None:
        super().__init__()
        self.heads, self.head_dim, self.half_window = heads, width // heads, half_window
        self.backend = backend
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, valid: Tensor, block_mask: BlockMask | None = None
    ) -> Tensor:
        B, A, _ = x.shape
        q = self.q_proj(x).view(B, A, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, A, self.heads, self.head_dim)
        v = self.v_proj(x).view(B, A, self.heads, self.head_dim)
        q = F.rms_norm(q, (self.head_dim,)).to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,)).to(k.dtype)
        q, k = apply_rotary_3d(q, cos, sin), apply_rotary_3d(k, cos, sin)
        in_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        out = sliding_window_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), valid, self.half_window,
            backend=self.backend, block_mask=block_mask,  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        )
        out = out.transpose(1, 2).reshape(B, A, -1).to(in_dtype)
        out = out * torch.sigmoid(self.gate_proj(x))
        return self.o_proj(out)


class AtomBlock(nn.Module):
    """adaLN (``rms_norm(x) * (1 + scale) + shift``) + raw-gated residual attention and MLP.

    ``adaln_linear(silu(c))`` chunks to ``shift_a, scale_a, gate_a, shift_f, scale_f, gate_f``;
    it is zero-initialised so a fresh block is the identity (spec §5.4).
    """

    def __init__(
        self, width: int, heads: int, half_window: int, *, expansion: int = 2, backend: str = "auto"
    ) -> None:
        super().__init__()
        self.adaln_linear = nn.Linear(width, 6 * width, bias=False)
        self.adaln_linear._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        nn.init.zeros_(self.adaln_linear.weight)
        self.self_attn = AtomAttention(width, heads, half_window, backend=backend)
        self.mlp = GatedMLP(width, atom_ffn_hidden(width, expansion))

    def forward(
        self, x: Tensor, c: Tensor, cos: Tensor, sin: Tensor, valid: Tensor,
        block_mask: BlockMask | None = None,
    ) -> Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_linear(F.silu(c)).chunk(6, dim=-1)
        h = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_a) + shift_a
        x = x + gate_a * self.self_attn(h, cos, sin, valid, block_mask)
        h = F.rms_norm(x, (x.shape[-1],)) * (1 + scale_f) + shift_f
        return x + gate_f * self.mlp(h)


def _atom_blocks(config: FoldConfig, num_blocks: int, heads: int) -> nn.ModuleList:
    return nn.ModuleList(
        [
            AtomBlock(config.atom_width, heads, config.atom_window // 2, backend=config.attention_backend)
            for _ in range(num_blocks)
        ]
    )


class AtomEncoder(nn.Module):
    """Atom features -> per-atom conditioning ``c`` -> windowed blocks -> relu -> token mean."""

    def __init__(self, config: FoldConfig, *, out_width: int, num_blocks: int, heads: int) -> None:
        super().__init__()
        self.atom_linear = nn.Linear(config.atom_feature_dim, config.atom_width, bias=False)
        self.atom_norm = nn.LayerNorm(config.atom_width, eps=config.layer_norm_eps)
        self.layers = _atom_blocks(config, num_blocks, heads)
        self.atom_to_token_linear = nn.Linear(config.atom_width, out_width, bias=False)

    def embed(self, features: Tensor) -> Tensor:
        """``c = atom_norm(atom_linear(features))`` — the per-atom conditioning and initial ``q``."""
        return self.atom_norm(self.atom_linear(features))

    def forward(
        self, q: Tensor, c: Tensor, cos: Tensor, sin: Tensor, valid: Tensor, atom_to_token: Tensor,
        n_tokens: int, *, block_mask: BlockMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns ``(a (B, n_tokens, out_width), q (B, A, atom_width))``; ``q`` is the decoder skip."""
        for layer in self.layers:
            q = layer(q, c, cos, sin, valid, block_mask)
        a = scatter_atom_to_token_mean(F.relu(self.atom_to_token_linear(q)), atom_to_token, n_tokens, valid)
        return a, q


class AtomDecoder(nn.Module):
    """Token features broadcast to atoms, windowed blocks, LayerNorm, 3-D output projection."""

    def __init__(self, config: FoldConfig, *, num_blocks: int, heads: int) -> None:
        super().__init__()
        self.token_to_atom_linear = nn.Linear(config.token_width, config.atom_width, bias=False)
        self.layers = _atom_blocks(config, num_blocks, heads)
        self.norm = nn.LayerNorm(config.atom_width, eps=config.layer_norm_eps)
        self.output_linear = nn.Linear(config.atom_width, 3, bias=False)

    def forward(
        self, a: Tensor, q: Tensor, c: Tensor, cos: Tensor, sin: Tensor, valid: Tensor,
        atom_to_token: Tensor, *, block_mask: BlockMask | None = None,
    ) -> Tensor:
        q = q + gather_token_to_atom(self.token_to_atom_linear(a), atom_to_token)
        for layer in self.layers:
            q = layer(q, c, cos, sin, valid, block_mask)
        return self.output_linear(self.norm(q))


@dataclass
class InputsEmbedding:
    """What the inputs embedder hands the trunk and the heads."""

    s_inputs: Tensor  # (B, L, 451)
    z_init: Tensor  # (B, L, L, pair)
    relpos: Tensor  # (B, L, L, pair) rel_pos embedding (reused by the structure/confidence heads)
    bonds: Tensor  # (B, L, L, pair) token-bond embedding (reused by the confidence head)
    atom_features: Tensor  # (B, A, 389) (reused by the diffusion atom encoder)
    rope: tuple[Tensor, Tensor]  # bf16 cos/sin (reused by the diffusion atom encoder/decoder)


class InputsEmbedder(nn.Module):
    """Checkpoint ``input_embedder``: atom encoder, single-input assembly and the initial pair."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_encoder = AtomEncoder(
            config, out_width=config.inputs_token_width,
            num_blocks=config.atom_encoder_blocks, heads=config.atom_encoder_heads,
        )
        self.pair_init_1 = nn.Linear(config.single_inputs_width, config.pair_width, bias=False)
        self.pair_init_2 = nn.Linear(config.single_inputs_width, config.pair_width, bias=False)
        self.rel_pos = RelativePositionEncoding(
            config.pair_width, r_max=config.relpos_r_max, s_max=config.relpos_s_max
        )
        self.token_bonds = nn.Linear(1, config.pair_width, bias=False)

    def rope(self, features: FoldFeatures) -> tuple[Tensor, Tensor]:
        cfg = self.config
        return build_3d_rope(
            features.ref_pos, features.ref_space_uid,
            head_dim=cfg.atom_width // cfg.atom_encoder_heads,
            spatial_pairs_per_axis=cfg.spatial_rope_pairs_per_axis, uid_pairs=cfg.uid_rope_pairs,
            spatial_base=cfg.spatial_rope_base, uid_base=cfg.uid_rope_base,
        )

    def forward(self, f: FoldFeatures, *, block_mask: BlockMask | None = None) -> InputsEmbedding:
        cfg = self.config
        feats = build_atom_features(
            f.ref_pos, f.ref_charge, f.atom_mask, f.ref_element, f.ref_atom_name_chars,
            max_atomic_number=cfg.max_atomic_number, name_vocab=cfg.atom_name_vocab,
        )
        rope = self.rope(f)
        c = self.atom_encoder.embed(feats)
        a, _ = self.atom_encoder(c, c, *rope, f.atom_mask, f.atom_to_token, f.num_tokens, block_mask=block_mask)
        res_oh = F.one_hot(f.res_type, cfg.num_res_types).float() * f.token_mask[..., None].float()
        # single-sequence mode: profile = the query one-hot, deletion_mean = 0 (upstream S2)
        s_inputs = torch.cat([a.float(), res_oh, res_oh, torch.zeros_like(res_oh[..., :1])], dim=-1)
        relpos = self.rel_pos(f.residue_index, f.asym_id, f.sym_id, f.entity_id, f.token_index)
        bonds = self.token_bonds(f.token_bonds.to(self.token_bonds.weight.dtype))
        z_init = self.pair_init_1(s_inputs)[:, :, None, :] + self.pair_init_2(s_inputs)[:, None, :, :]
        z_init = z_init + relpos + bonds
        return InputsEmbedding(s_inputs, z_init, relpos, bonds, feats, rope)
```

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_atoms.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/atoms.py tests/fold/test_atoms.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: 9 passed. (`test_atom_attention_matches_upstream_transcription` compares two bf16 attention evaluations, hence the 1e-2 tolerance; the dense path and SDPA agree to bf16 rounding.)

```bash
git add src/oplm/fold/atoms.py tests/fold/test_atoms.py
git commit -m "feat(fold): atom features, 3D RoPE, SWA atom blocks, atom encoders and the inputs embedder"
```

---

### Task 6: Language-model shim and the frozen-LM hidden-state runner

**Files:**
- Create: `src/oplm/fold/lm_shim.py`
- Test: `tests/fold/test_lm_shim.py`

**Interfaces:**
- Consumes: Task 3 `outer_product_difference`, `FoldFeatures`; `FoldConfig` fields `lm_hidden_size`, `lm_num_hidden_states`, `pair_width`, `layer_norm_eps`; `oplm.model.OplmModel.forward(input_ids, attention_mask, output_hidden_states=True, return_dict=True)` whose `hidden_states` is a tuple of `len(backbone.layer_execution_order) + 1` tensors `(C, T, D)`.
- Produces: `SingleToPair(width, hidden, out_width)` (children `downproject`, `output_fc1`, `output_fc2`, all with bias) with `forward(x, rows=None, cols=None)`; `LanguageModelShim(config)` (children `layer_weights`, `pair_input_norm`, `pair_proj`, `single_to_pair`, `pair_output_norm`) with `mix(hidden_states) -> Tensor[B, L, pair]` and `forward(hidden_states, rows=None, cols=None) -> Tensor[B, I, J, pair]`; `frozen_lm_hidden_states(lm, features, *, dtype=None) -> Tensor[1, L, K, D]`; `lm_state_count(lm) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_lm_shim.py`:

```python
"""LanguageModelShim against the transcribed upstream math; the per-chain frozen-LM runner."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.lm_shim import (
    LanguageModelShim,
    SingleToPair,
    frozen_lm_hidden_states,
    lm_state_count,
)
from oplm.model import OplmConfig, OplmModel


def test_single_to_pair_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    stp = SingleToPair(16, 16, 16)
    x = torch.randn(2, 5, 16)
    h = stp.downproject(x)
    pair = torch.cat([h.unsqueeze(2) * h.unsqueeze(1), h.unsqueeze(2) - h.unsqueeze(1)], dim=3)
    expected = stp.output_fc2(F.gelu(stp.output_fc1(pair)))
    torch.testing.assert_close(stp(x), expected)
    assert stp.output_fc1.weight.shape == (16, 32) and stp.downproject.bias is not None
    rows, cols = torch.tensor([4, 0]), torch.tensor([1, 3, 2])
    torch.testing.assert_close(stp(x, rows=rows, cols=cols), expected[:, rows][:, :, cols])


def test_shim_mixes_states_with_softmax_weights_and_names() -> None:
    cfg = FoldConfig(lm_hidden_size=24, lm_num_hidden_states=4, pair_width=32)
    torch.manual_seed(0)
    shim = LanguageModelShim(cfg)
    assert set(shim.state_dict()) == {
        "layer_weights", "pair_input_norm.weight", "pair_input_norm.bias", "pair_proj.weight",
        "single_to_pair.downproject.weight", "single_to_pair.downproject.bias",
        "single_to_pair.output_fc1.weight", "single_to_pair.output_fc1.bias",
        "single_to_pair.output_fc2.weight", "single_to_pair.output_fc2.bias",
        "pair_output_norm.weight", "pair_output_norm.bias",
    }
    assert shim.layer_weights.shape == (4,) and shim.pair_proj.weight.shape == (32, 24)
    hs = torch.randn(1, 6, 4, 24)
    with torch.no_grad():
        shim.layer_weights.copy_(torch.tensor([0.0, 50.0, 0.0, 0.0]))  # ~one-hot on state 1
        mixed = shim.mix(hs)
        torch.testing.assert_close(mixed, shim.pair_proj(shim.pair_input_norm(hs[:, :, 1])), atol=1e-5, rtol=1e-5)
        out = shim(hs)
        torch.testing.assert_close(out, shim.pair_output_norm(shim.single_to_pair(mixed)))
    assert out.shape == (1, 6, 6, 32)


def test_shim_handles_a_single_token() -> None:
    cfg = FoldConfig(lm_hidden_size=8, lm_num_hidden_states=3, pair_width=32)
    out = LanguageModelShim(cfg)(torch.randn(1, 1, 3, 8))
    assert out.shape == (1, 1, 1, 32) and torch.isfinite(out).all()


def _tiny_lm(num_loops: int = 1) -> OplmModel:
    torch.manual_seed(0)
    cfg = OplmConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64, num_loops=num_loops)
    return OplmModel(cfg).eval()


def test_frozen_lm_hidden_states_gathers_per_chain_rows() -> None:
    lm = _tiny_lm()
    f = featurize([ChainSpec("MKVLA", "A"), ChainSpec("GG", "B")], pad_tokens_to=8)
    assert lm_state_count(lm) == 3
    hs = frozen_lm_hidden_states(lm, f, dtype=torch.float32)
    assert hs.shape == (1, 8, 3, 32) and not hs.requires_grad
    with torch.no_grad():
        direct = lm(input_ids=f.lm_input_ids, attention_mask=f.lm_attention_mask, output_hidden_states=True, return_dict=True)
    stacked = torch.stack(direct.hidden_states, dim=2)  # (C, T, K, D)
    torch.testing.assert_close(hs[0, 0], stacked[0, 1])  # chain A, residue 1 (after BOS)
    torch.testing.assert_close(hs[0, 6], stacked[1, 2])  # chain B, residue 2
    assert hs[0, 7:].abs().sum() == 0  # padded tokens are zero


def test_state_count_follows_the_executed_depth() -> None:
    assert lm_state_count(_tiny_lm(num_loops=2)) == 5  # 2 layers x 2 loops + embedding
```

Run: `.venv/bin/python -m pytest tests/fold/test_lm_shim.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.lm_shim'`.

- [ ] **Step 2: `src/oplm/fold/lm_shim.py`**

```python
"""Language-model shim: per-layer LayerNorm + projection, softmax layer mix, single-to-pair.

Ported from Biohub's ESMFold2 ``LanguageModelShim`` and ``SingleToPair``
(esm/models/esmfold2/layers.py, Apache-2.0; see THIRD_PARTY_NOTICES.md), with the released
checkpoint's names (``language_model.{layer_weights, pair_input_norm, pair_proj,
single_to_pair.{downproject, output_fc1, output_fc2}, pair_output_norm}``). Modifications: the
layer mix is an explicit einsum (upstream's ``weights @ x`` + ``squeeze(-2)`` breaks at
``L == 1``); the pair product is block-local (spec §4.6); the hidden states come from any
LM that exposes a ``(B, L, K, D)`` stack — the OPLM runner here feeds every chain as its own
batch row with BOS/EOS (spec §4.5) instead of one packed row per complex.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.pair import outer_product_difference

if TYPE_CHECKING:
    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import FoldFeatures

__all__ = ["LanguageModelShim", "SingleToPair", "frozen_lm_hidden_states", "lm_state_count"]


class SingleToPair(nn.Module):
    """``fc2(gelu(fc1(cat[x_i * x_j, x_i - x_j])))`` after a biased down-projection."""

    def __init__(self, width: int, hidden: int, out_width: int) -> None:
        super().__init__()
        self.downproject = nn.Linear(width, width, bias=True)
        self.output_fc1 = nn.Linear(2 * width, hidden, bias=True)
        self.output_fc2 = nn.Linear(hidden, out_width, bias=True)

    def forward(self, x: Tensor, rows: Tensor | None = None, cols: Tensor | None = None) -> Tensor:
        x = self.downproject(x)
        return self.output_fc2(F.gelu(self.output_fc1(outer_product_difference(x, rows, cols))))


class LanguageModelShim(nn.Module):
    """Hidden states ``(B, L, K, D)`` -> LM pair ``(B, I, J, pair)`` (checkpoint ``language_model``)."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(config.lm_num_hidden_states))
        self.pair_input_norm = nn.LayerNorm(config.lm_hidden_size, eps=config.layer_norm_eps)
        self.pair_proj = nn.Linear(config.lm_hidden_size, config.pair_width, bias=False)
        self.single_to_pair = SingleToPair(config.pair_width, config.pair_width, config.pair_width)
        self.pair_output_norm = nn.LayerNorm(config.pair_width, eps=config.layer_norm_eps)

    def mix(self, hidden_states: Tensor) -> Tensor:
        """Per-state LayerNorm + projection, then the softmax-weighted sum over states."""
        h = self.pair_proj(self.pair_input_norm(hidden_states))
        weights = torch.softmax(self.layer_weights.float(), dim=0).to(h.dtype)
        return torch.einsum("k,blkc->blc", weights, h)

    def forward(self, hidden_states: Tensor, rows: Tensor | None = None, cols: Tensor | None = None) -> Tensor:
        return self.pair_output_norm(self.single_to_pair(self.mix(hidden_states), rows, cols))


def lm_state_count(lm: nn.Module) -> int:
    """Number of hidden states an ``OplmModel`` returns: executed blocks + the embedding output."""
    return len(lm.backbone.layer_execution_order) + 1  # ty: ignore[unresolved-attribute]  # OplmModel.backbone


def frozen_lm_hidden_states(
    lm: nn.Module, features: FoldFeatures, *, dtype: torch.dtype | None = None
) -> Tensor:
    """Run the frozen LM on the per-chain rows and gather every state per token: ``(1, L, K, D)``.

    The LM must already be in eval mode on the features' device (the fold model owns that).
    Padded tokens are zero. Runs under ``torch.no_grad`` (not inference mode: the OPLM RoPE
    cache re-assigns buffers lazily).
    """
    with torch.no_grad():
        out = lm(
            input_ids=features.lm_input_ids, attention_mask=features.lm_attention_mask,
            output_hidden_states=True, return_dict=True,
        )
    states = torch.stack(tuple(out.hidden_states), dim=2)  # (C, T, K, D)
    if dtype is not None:
        states = states.to(dtype)
    gathered = states[features.lm_rows[0], features.lm_positions[0]]  # (L, K, D)
    gathered = gathered * features.token_mask[0][:, None, None].to(gathered.dtype)
    return gathered[None]
```

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_lm_shim.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/lm_shim.py tests/fold/test_lm_shim.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: 5 passed.

```bash
git add src/oplm/fold/lm_shim.py tests/fold/test_lm_shim.py
git commit -m "feat(fold): language-model shim and the per-chain frozen-LM hidden-state runner"
```


### Task 7: Diffusion structure head (conditioning, token transformer, denoiser, sampler)

**Files:**
- Create: `src/oplm/fold/diffusion.py`
- Test: `tests/fold/test_diffusion.py`

**Interfaces:**
- Consumes: Task 1 `pair_biased_attention(..., block_mask)`, `pair_bias_block_mask`, `sliding_window_block_mask`, `resolve_attention_backend`; Task 4 `GatedMLP`, `Transition`, `cuda_bf16_autocast`; Task 5 `AtomEncoder`, `AtomDecoder`; `FoldConfig` diffusion/sampler fields.
- Produces: `FourierEmbedding(dim)` (buffers `frequencies`, `phases`), `AdaptiveLayerNorm(width, cond_width, *, eps)` (children `cond_norm`, `gate_proj`, `shift_proj`), `PairBiasAttention(width, heads, *, backend)` (children `q_proj, k_proj, v_proj, gate_proj, o_proj`), `DiffusionBlock(config)`, `DiffusionTransformer(config)` (child `layers`), `DiffusionConditioning(config)` with `pair(z_trunk, relpos)` and `single(s_inputs, t_hat)`, `DenoiserInputs` dataclass, `StructureHead(config)` with `prepare(...) -> DenoiserInputs`, `denoise(x_noisy, t_hat, inputs) -> Tensor`, `noise_schedule(num_steps, device)`, `sample(inputs, *, num_steps=None, sigma_cap=None, noise_scale=None, step_scale=None, generator=None) -> Tensor[B·S, A, 3]`; functions `random_rotations`, `center_random_augmentation`, `weighted_rigid_align`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_diffusion.py`:

```python
"""Diffusion head: adaLN, pair-bias blocks, conditioning, EDM denoiser, schedule and sampler."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize
from oplm.fold.diffusion import (
    AdaptiveLayerNorm,
    DiffusionBlock,
    DiffusionConditioning,
    PairBiasAttention,
    StructureHead,
    center_random_augmentation,
    random_rotations,
    weighted_rigid_align,
)
from oplm.fold.atoms import InputsEmbedder


def _cfg(**over) -> FoldConfig:
    base = dict(
        pair_width=32, token_width=64, atom_width=32, atom_encoder_blocks=1, atom_encoder_heads=2,
        uid_rope_pairs=2, trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, diffusion_blocks=1,
        diffusion_heads=4, diffusion_atom_blocks=1, diffusion_atom_heads=2, fourier_dim=16,
        confidence_blocks=1, inference_num_steps=3, attention_backend="dense",
        trimul_backend="reference",
    )
    base.update(over)
    return FoldConfig(**base)


def test_adaptive_layer_norm_matches_transcription_and_names() -> None:
    torch.manual_seed(0)
    ada = AdaptiveLayerNorm(8, 6, eps=1e-5)
    assert set(ada.state_dict()) == {"cond_norm.weight", "gate_proj.weight", "gate_proj.bias", "shift_proj.weight"}
    a, s = torch.randn(2, 5, 8), torch.randn(2, 5, 6)
    a_norm = F.layer_norm(a, (8,), None, None, 1e-5)
    s_norm = F.layer_norm(s, (6,), ada.cond_norm.weight, None, 1e-5)
    expected = torch.sigmoid(ada.gate_proj(s_norm)) * a_norm + ada.shift_proj(s_norm)
    torch.testing.assert_close(ada(a, s), expected)


def test_pair_bias_attention_matches_upstream_transcription() -> None:
    torch.manual_seed(0)
    attn = PairBiasAttention(16, 2, backend="dense")
    x = torch.randn(2, 7, 16)
    bias = torch.randn(2, 2, 7, 7)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    out = attn(x, bias, mask)
    q = attn.q_proj(x).view(2, 7, 2, 8)
    k = attn.k_proj(x).view(2, 7, 2, 8)
    v = attn.v_proj(x).view(2, 7, 2, 8)
    g = torch.sigmoid(attn.gate_proj(x)).view(2, 7, 2, 8)
    logits = torch.einsum("...ihd,...jhd->...ijh", q, k) * 8**-0.5 + bias.permute(0, 2, 3, 1)
    logits = logits + torch.where(mask[:, None, :, None], 0.0, torch.finfo(logits.dtype).min)
    ctx = torch.einsum("...ijh,...jhd->...ihd", logits.softmax(dim=-2), v)
    expected = attn.o_proj((g * ctx).reshape(2, 7, 16))
    torch.testing.assert_close(out, expected)
    assert attn.q_proj.bias is not None and attn.k_proj.bias is None


def test_diffusion_block_names_gate_init_and_padded_rows() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    block = DiffusionBlock(cfg)
    names = set(block.state_dict())
    assert names == {
        "input_layernorm.cond_norm.weight", "input_layernorm.gate_proj.weight", "input_layernorm.gate_proj.bias",
        "input_layernorm.shift_proj.weight", "self_attn.q_proj.weight", "self_attn.q_proj.bias",
        "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.gate_proj.weight", "self_attn.o_proj.weight",
        "pair_norm.weight", "pair_norm.bias", "pair_bias_proj.weight", "attn_gate.weight", "attn_gate.bias",
        "post_attention_layernorm.cond_norm.weight", "post_attention_layernorm.gate_proj.weight",
        "post_attention_layernorm.gate_proj.bias", "post_attention_layernorm.shift_proj.weight",
        "mlp.gate_up_proj.weight", "mlp.down_proj.weight", "mlp_gate.weight", "mlp_gate.bias",
    }
    for gate in (block.attn_gate, block.mlp_gate):
        assert gate.weight.abs().sum() == 0 and torch.equal(gate.bias, torch.full((64,), -2.0))
    assert block.pair_bias_proj.weight.shape == (4, 32) and block.mlp.gate_up_proj.weight.shape == (256, 64)
    a, s = torch.randn(2, 6, 64), torch.randn(2, 6, 64)
    z = torch.randn(1, 6, 6, 32)  # base batch 1, two samples
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1] = False  # Review Focus 2: a fully padded batch row
    out = block(a, s, z, mask)
    assert out.shape == a.shape and torch.isfinite(out).all()
    # with zero gates the block is the identity on the residual stream: sigmoid(-2) * ... is small but nonzero,
    # so check the padded row is a pure function of a and s (no attention leak): attention output is zeroed there
    x = block.input_layernorm(a, s)
    residual = a + torch.sigmoid(block.attn_gate(s)) * 0.0
    x2 = block.post_attention_layernorm(residual, s)
    expected_row1 = residual[1] + torch.sigmoid(block.mlp_gate(s))[1] * block.mlp(x2)[1]
    torch.testing.assert_close(out[1], expected_row1)


def test_conditioning_names_and_noise_embedding() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    cond = DiffusionConditioning(cfg)
    names = set(cond.state_dict())
    assert {"fourier.frequencies", "fourier.phases", "pair_input_norm.weight", "pair_proj.weight",
            "pair_transition_0.mlp.gate_up_proj.weight", "pair_transition_1.norm.bias", "single_input_norm.weight",
            "single_proj.weight", "single_transition_0.mlp.down_proj.weight", "noise_norm.weight", "noise_proj.weight"} <= names
    assert cond.pair_proj.weight.shape == (32, 64) and cond.pair_transition_0.mlp.gate_up_proj.weight.shape == (128, 32)
    assert cond.single_proj.weight.shape == (64, cfg.single_inputs_width) and cond.noise_proj.weight.shape == (64, 16)
    t = torch.tensor([16.0, 4.0])
    emb = cond.fourier(0.25 * torch.log(t / 16.0))
    expected = torch.cos(2 * math.pi * ((0.25 * torch.log(t / 16.0))[:, None] * cond.fourier.frequencies + cond.fourier.phases))
    torch.testing.assert_close(emb, expected)
    s_inputs = torch.randn(1, 5, cfg.single_inputs_width)
    s = cond.single(s_inputs.repeat_interleave(2, 0), t)
    assert s.shape == (2, 5, 64) and not torch.equal(s[0], s[1])  # different noise levels differ
    z = cond.pair(torch.randn(1, 5, 5, 32), torch.randn(1, 5, 5, 32))
    assert z.shape == (1, 5, 5, 32) and z.dtype == torch.float32


def _head_and_inputs(cfg: FoldConfig, num_samples: int):
    torch.manual_seed(0)
    emb = InputsEmbedder(cfg).eval()
    head = StructureHead(cfg).eval()
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")], pad_tokens_to=8)
    with torch.no_grad():
        e = emb(f)
        z_trunk = torch.randn(1, 8, 8, cfg.pair_width)
        inp = head.prepare(
            s_inputs=e.s_inputs, z_trunk=z_trunk, relpos=e.relpos, atom_features=e.atom_features, rope=e.rope,
            atom_mask=f.atom_mask, atom_to_token=f.atom_to_token, token_mask=f.token_mask, num_samples=num_samples,
        )
    return head, inp, f


def test_denoiser_shapes_edm_limits_and_zero_init_names() -> None:
    cfg = _cfg()
    head, inp, f = _head_and_inputs(cfg, num_samples=2)
    assert head.single_to_token.weight.abs().sum() == 0 and head.coords_linear.weight.shape == (32, 6)
    assert {"conditioning.fourier.frequencies", "coords_linear.weight", "single_to_token.weight", "single_step_norm.weight",
            "token_norm.bias", "atom_encoder.atom_to_token_linear.weight", "atom_decoder.output_linear.weight",
            "token_transformer.layers.0.attn_gate.bias"} <= set(head.state_dict())
    assert head.atom_encoder.atom_to_token_linear.weight.shape == (64, 32)
    x = torch.randn(2, f.num_atoms, 3)
    with torch.no_grad():
        small = head.denoise(x, torch.full((2,), 1e-6), inp)
        big = head.denoise(x, torch.full((2,), 1e6), inp)
    assert small.shape == (2, f.num_atoms, 3)
    torch.testing.assert_close(small, x, atol=1e-3, rtol=0)  # c_skip -> 1, c_out -> 0 as t -> 0
    assert torch.isfinite(big).all() and (big - x).abs().max() > 1e-3  # c_skip -> 0: pure network output


def test_noise_schedule_matches_transcription_and_cap_truncates() -> None:
    cfg = _cfg(inference_num_steps=14)  # released sampler constants otherwise
    head = StructureHead(cfg)
    sched = head.noise_schedule(14, torch.device("cpu"))
    k = torch.arange(14, dtype=torch.float32)
    base = 160.0 ** (1 / 7) + (k / 13) * (4e-4 ** (1 / 7) - 160.0 ** (1 / 7))
    expected = F.pad(16.0 * base.pow(7.0), (0, 1), value=0.0)
    torch.testing.assert_close(sched, expected)
    assert sched[0] == 2560.0 and sched[-1] == 0.0
    capped = sched[sched <= 256.0]
    capped = F.pad(capped, (1, 0), value=256.0)
    assert capped.shape == (11,) and capped[0] == 256.0  # 10 steps actually run
    torch.testing.assert_close(capped[1], torch.tensor(165.6605), atol=1e-3, rtol=0)
    assert head.noise_schedule(1, torch.device("cpu")).tolist() == [160.0 * 16.0, 0.0]


def test_augmentation_and_kabsch_alignment() -> None:
    g = torch.Generator().manual_seed(0)
    rot = random_rotations(4, device=torch.device("cpu"), dtype=torch.float32, generator=g)
    torch.testing.assert_close(torch.linalg.det(rot), torch.ones(4))
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), torch.eye(3).expand(4, 3, 3), atol=1e-5, rtol=0)
    gt = torch.randn(2, 10, 3, generator=g)
    mask = torch.ones(2, 10)
    mask[1, 8:] = 0.0
    moved = center_random_augmentation(gt, mask, generator=g)
    assert moved.shape == gt.shape
    aligned = weighted_rigid_align(moved, gt, mask)
    torch.testing.assert_close(aligned[0], gt[0], atol=1e-4, rtol=0)
    torch.testing.assert_close(aligned[1, :8], gt[1, :8], atol=1e-4, rtol=0)  # pads carry no weight


def test_sample_is_deterministic_under_a_generator() -> None:
    cfg = _cfg()
    head, inp, f = _head_and_inputs(cfg, num_samples=2)
    with torch.no_grad():
        a = head.sample(inp, generator=torch.Generator().manual_seed(7))
        b = head.sample(inp, generator=torch.Generator().manual_seed(7))
        c = head.sample(inp, generator=torch.Generator().manual_seed(8))
    assert a.shape == (2, f.num_atoms, 3) and torch.isfinite(a).all()
    assert torch.equal(a, b) and not torch.equal(a, c)
```

Run: `.venv/bin/python -m pytest tests/fold/test_diffusion.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.diffusion'`.

- [ ] **Step 2: `src/oplm/fold/diffusion.py`**

```python
"""Diffusion structure head: conditioning, adaptive-norm pair-bias blocks, EDM denoiser, sampler.

Ported from Biohub's ESMFold2 ``FourierEmbedding``, ``AdaptiveLayerNorm``, ``AttentionPairBias``,
``ConditionedTransitionBlock``, ``DiffusionTransformer``, ``DiffusionConditioning``,
``DiffusionModule`` and ``DiffusionStructureHead`` (esm/models/esmfold2/layers.py, Apache-2.0;
see THIRD_PARTY_NOTICES.md). Names follow the released checkpoint: one
``token_transformer.layers.N`` holds the attention and transition halves of a block
(``input_layernorm``/``post_attention_layernorm`` adaptive norms, ``attn_gate``/``mlp_gate``
output gates, ``self_attn.{q,k,v,gate,o}_proj``, ``pair_norm``, ``pair_bias_proj``, ``mlp``).
Modifications: attention is milestone 0's ``pair_biased_attention`` (FlexAttention / dense
oracle) with a precomputed block mask, and padded query rows are zeroed; the denoiser takes the
conditioned pair and the atom conditioning as arguments so the sampler computes them once; the
sampler takes a ``torch.Generator``; the sigma cap truncates the schedule exactly as upstream.
Precision (spec §5.4): coordinates, preconditioning, alignment and the sampler are fp32; the
pair-conditioning transitions run under bf16 autocast on CUDA as upstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.atoms import AtomDecoder, AtomEncoder
from oplm.fold.attention import (
    pair_bias_block_mask,
    pair_biased_attention,
    resolve_attention_backend,
    sliding_window_block_mask,
)
from oplm.fold.trunk import GatedMLP, Transition, cuda_bf16_autocast

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn.attention.flex_attention import BlockMask

    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "AdaptiveLayerNorm",
    "DenoiserInputs",
    "DiffusionBlock",
    "DiffusionConditioning",
    "DiffusionTransformer",
    "FourierEmbedding",
    "PairBiasAttention",
    "StructureHead",
    "center_random_augmentation",
    "random_rotations",
    "weighted_rigid_align",
]


class FourierEmbedding(nn.Module):
    """``cos(2π (t · w + b))`` with random, persistent ``frequencies``/``phases`` (loaded from ckpt)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.randn(dim))
        self.register_buffer("phases", torch.randn(dim))

    def forward(self, t: Tensor) -> Tensor:
        return torch.cos(2 * math.pi * (t[:, None] * self.frequencies + self.phases))


class AdaptiveLayerNorm(nn.Module):
    """``sigmoid(gate(LN_w(s))) · LN(a) + shift(LN_w(s))`` (checkpoint ``cond_norm/gate_proj/shift_proj``)."""

    def __init__(self, width: int, cond_width: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.cond_norm = nn.LayerNorm(cond_width, eps=eps, bias=False)
        self.gate_proj = nn.Linear(cond_width, width, bias=True)
        self.shift_proj = nn.Linear(cond_width, width, bias=False)

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a_norm = F.layer_norm(a, (a.shape[-1],), eps=self.eps)
        s_norm = self.cond_norm(s)
        return torch.sigmoid(self.gate_proj(s_norm)) * a_norm + self.shift_proj(s_norm)


def _output_gate(width: int) -> nn.Linear:
    """Zero-weight, −2-bias gate (spec §5.4); tags keep ``_init_weights`` from overwriting it."""
    gate = nn.Linear(width, width, bias=True)
    gate._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
    gate._init_bias = -2.0  # ty: ignore[unresolved-attribute]  # read by _init_weights
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, -2.0)
    return gate


class PairBiasAttention(nn.Module):
    """Multi-head attention with an additive per-head pair bias and a sigmoid value gate."""

    def __init__(self, width: int, heads: int, *, backend: str = "auto") -> None:
        super().__init__()
        self.heads, self.head_dim, self.backend = heads, width // heads, backend
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(
        self, x: Tensor, bias: Tensor, key_mask: Tensor, block_mask: BlockMask | None = None
    ) -> Tensor:
        B, L, _ = x.shape

        def heads(t: Tensor) -> Tensor:
            return t.view(B, L, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        g = torch.sigmoid(heads(self.gate_proj(x)))
        ctx = pair_biased_attention(
            q, k, v, bias, key_mask, backend=self.backend, block_mask=block_mask  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        )
        return self.o_proj((g * ctx).transpose(1, 2).reshape(B, L, -1))


class DiffusionBlock(nn.Module):
    """One ``token_transformer.layers.N``: gated pair-bias attention then a gated transition."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        w, p, eps = config.token_width, config.pair_width, config.layer_norm_eps
        self.input_layernorm = AdaptiveLayerNorm(w, w, eps=eps)
        self.self_attn = PairBiasAttention(w, config.diffusion_heads, backend=config.attention_backend)
        self.pair_norm = nn.LayerNorm(p, eps=eps)
        self.pair_bias_proj = nn.Linear(p, config.diffusion_heads, bias=False)
        self.attn_gate = _output_gate(w)
        self.post_attention_layernorm = AdaptiveLayerNorm(w, w, eps=eps)
        self.mlp = GatedMLP(w, config.diffusion_transition_multiplier * w)
        self.mlp_gate = _output_gate(w)

    def forward(
        self, a: Tensor, s: Tensor, z: Tensor, token_mask: Tensor, block_mask: BlockMask | None = None
    ) -> Tensor:
        bias = self.pair_bias_proj(self.pair_norm(z)).permute(0, 3, 1, 2)  # (B, H, L, L)
        if bias.shape[0] != a.shape[0]:
            bias = bias.repeat_interleave(a.shape[0] // bias.shape[0], dim=0)
        x = self.input_layernorm(a, s)
        attn = self.self_attn(x, bias, token_mask, block_mask) * token_mask[..., None].to(a.dtype)
        a = a + torch.sigmoid(self.attn_gate(s)) * attn
        x = self.post_attention_layernorm(a, s)
        return a + torch.sigmoid(self.mlp_gate(s)) * self.mlp(x)


class DiffusionTransformer(nn.Module):
    """Checkpoint ``token_transformer``."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DiffusionBlock(config) for _ in range(config.diffusion_blocks)])

    def forward(
        self, a: Tensor, s: Tensor, z: Tensor, token_mask: Tensor, block_mask: BlockMask | None = None
    ) -> Tensor:
        for layer in self.layers:
            a = layer(a, s, z, token_mask, block_mask)
        return a


class DiffusionConditioning(nn.Module):
    """Checkpoint ``conditioning``: pair conditioning (once per call) and noise-aware single."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        p, w, s_in = config.pair_width, config.token_width, config.single_inputs_width
        mult, eps = config.diffusion_transition_multiplier, config.layer_norm_eps
        self.sigma_data = config.sigma_data
        self.fourier = FourierEmbedding(config.fourier_dim)
        self.pair_input_norm = nn.LayerNorm(2 * p, eps=eps)
        self.pair_proj = nn.Linear(2 * p, p, bias=False)
        self.pair_transition_0 = Transition(p, mult, eps=eps)
        self.pair_transition_1 = Transition(p, mult, eps=eps)
        self.single_input_norm = nn.LayerNorm(s_in, eps=eps)
        self.single_proj = nn.Linear(s_in, w, bias=False)
        self.single_transition_0 = Transition(w, mult, eps=eps)
        self.single_transition_1 = Transition(w, mult, eps=eps)
        self.noise_norm = nn.LayerNorm(config.fourier_dim, eps=eps)
        self.noise_proj = nn.Linear(config.fourier_dim, w, bias=False)

    def pair(self, z_trunk: Tensor, relpos: Tensor) -> Tensor:
        """``cat[z_trunk, relpos] -> LN -> proj -> two residual transitions`` (fp32 out)."""
        z = self.pair_proj(self.pair_input_norm(torch.cat([z_trunk.float(), relpos.float()], dim=-1)))
        with cuda_bf16_autocast(z.is_cuda):
            z = z + self.pair_transition_0(z)
            z = z + self.pair_transition_1(z)
        return z.float()

    def single(self, s_inputs: Tensor, t_hat: Tensor) -> Tensor:
        """``proj(LN(s_inputs)) + noise_proj(LN(fourier(0.25 log(t/σ_d))))`` then two transitions."""
        s = self.single_proj(self.single_input_norm(s_inputs.float()))
        c_noise = 0.25 * torch.log((t_hat.float() / self.sigma_data).clamp(min=1e-20))
        n = self.noise_proj(self.noise_norm(self.fourier(c_noise)))
        s = s + n[:, None, :]
        s = s + self.single_transition_0(s)
        return s + self.single_transition_1(s)


@dataclass
class DenoiserInputs:
    """Everything that is constant across the denoising steps of one sampling call.

    All tensors except ``z`` are already expanded to ``B · num_samples`` rows.
    """

    s_inputs: Tensor
    z: Tensor  # (B, L, L, pair) conditioned pair, base batch
    c: Tensor  # (B·S, A, atom) atom conditioning
    rope: tuple[Tensor, Tensor]
    atom_mask: Tensor  # (B·S, A) bool
    atom_to_token: Tensor
    token_mask: Tensor  # (B·S, L) bool
    token_block_mask: BlockMask | None = None
    atom_block_mask: BlockMask | None = None


def random_rotations(
    n: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None
) -> Tensor:
    """``(n, 3, 3)`` rotations from sign-normalised random quaternions (upstream ``_random_rotations``)."""
    q = torch.randn((n, 4), dtype=dtype, device=device, generator=generator)
    scale = torch.sqrt((q * q).sum(dim=1))
    signs = torch.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = torch.unbind(q, dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1)
    return torch.stack(
        [
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    ).reshape(n, 3, 3)


def center_random_augmentation(
    x: Tensor, atom_mask: Tensor, *, generator: torch.Generator | None = None
) -> Tensor:
    """Masked centering, a random rotation per sample, then a ``N(0, I)`` translation (Å)."""
    mask = atom_mask[..., None].to(x.dtype)
    x = x - (x * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)
    rot = random_rotations(x.shape[0], device=x.device, dtype=x.dtype, generator=generator)
    x = torch.einsum("bmd,bds->bms", x, rot)
    return x + torch.randn((x.shape[0], 1, 3), dtype=x.dtype, device=x.device, generator=generator)


def weighted_rigid_align(x: Tensor, x_gt: Tensor, weights: Tensor) -> Tensor:
    """Weighted Kabsch in fp32: ``x`` superposed onto ``x_gt`` (upstream ``_weighted_rigid_align``)."""
    w = weights[..., None].float()
    denom = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    mu = (x.float() * w).sum(dim=1, keepdim=True) / denom
    mu_gt = (x_gt.float() * w).sum(dim=1, keepdim=True) / denom
    x_c, gt_c = x.float() - mu, x_gt.float() - mu_gt
    h = torch.einsum("bni,bnj->bij", w * gt_c, x_c)
    u, _, vh = torch.linalg.svd(h, driver="gesvd" if h.is_cuda else None)
    det = torch.linalg.det(u @ vh)
    d = torch.ones(h.shape[0], 3, device=h.device, dtype=h.dtype)
    d[:, 2] = det
    rot = u @ torch.diag_embed(d) @ vh
    return x_c @ rot.transpose(-1, -2) + mu_gt


class StructureHead(nn.Module):
    """Checkpoint ``structure_head``: the EDM denoiser and the Karras sampler."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        self.sigma_data = config.sigma_data
        eps = config.layer_norm_eps
        self.conditioning = DiffusionConditioning(config)
        self.coords_linear = nn.Linear(6, config.atom_width, bias=False)
        self.atom_encoder = AtomEncoder(
            config, out_width=config.token_width,
            num_blocks=config.diffusion_atom_blocks, heads=config.diffusion_atom_heads,
        )
        self.single_to_token = nn.Linear(config.token_width, config.token_width, bias=False)
        self.single_to_token._init_zero = True  # ty: ignore[unresolved-attribute]  # read by _init_weights
        nn.init.zeros_(self.single_to_token.weight)
        self.single_step_norm = nn.LayerNorm(config.token_width, eps=eps)
        self.token_transformer = DiffusionTransformer(config)
        self.token_norm = nn.LayerNorm(config.token_width, eps=eps)
        self.atom_decoder = AtomDecoder(
            config, num_blocks=config.diffusion_atom_blocks, heads=config.diffusion_atom_heads
        )

    def prepare(
        self,
        *,
        s_inputs: Tensor,
        z_trunk: Tensor,
        relpos: Tensor,
        atom_features: Tensor,
        rope: tuple[Tensor, Tensor],
        atom_mask: Tensor,
        atom_to_token: Tensor,
        token_mask: Tensor,
        num_samples: int,
    ) -> DenoiserInputs:
        """Condition the pair, embed the atoms and expand the masks once for ``num_samples``."""
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples!r}")

        def rep(t: Tensor) -> Tensor:
            return t.repeat_interleave(num_samples, dim=0)

        z = self.conditioning.pair(z_trunk, relpos)
        c = self.atom_encoder.embed(atom_features)
        atom_mask_s, token_mask_s = rep(atom_mask), rep(token_mask)
        flex = resolve_attention_backend(z, self.config.attention_backend) == "flex"  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        return DenoiserInputs(
            s_inputs=rep(s_inputs), z=z, c=rep(c), rope=(rep(rope[0]), rep(rope[1])),
            atom_mask=atom_mask_s, atom_to_token=rep(atom_to_token), token_mask=token_mask_s,
            token_block_mask=pair_bias_block_mask(token_mask_s) if flex else None,
            atom_block_mask=(
                sliding_window_block_mask(atom_mask_s, self.config.atom_window // 2) if flex else None
            ),
        )

    def denoise(self, x_noisy: Tensor, t_hat: Tensor, inp: DenoiserInputs) -> Tensor:
        """EDM-preconditioned denoiser: ``c_skip · x + c_out · F(c_in · x, c_noise)`` (fp32)."""
        sigma = self.sigma_data
        t = t_hat.float().reshape(-1)
        if t.numel() == 1:
            t = t.expand(x_noisy.shape[0])
        x_noisy = x_noisy.float()
        s = self.conditioning.single(inp.s_inputs, t)
        r_noisy = x_noisy / torch.sqrt(t * t + sigma * sigma)[:, None, None]
        coords = torch.cat([r_noisy, torch.zeros_like(r_noisy)], dim=-1)  # upstream: pred_r1 is always 0
        q = inp.c + self.coords_linear(coords.to(inp.c.dtype))
        n_tokens = inp.token_mask.shape[1]
        a, q_skip = self.atom_encoder(
            q, inp.c, *inp.rope, inp.atom_mask, inp.atom_to_token, n_tokens, block_mask=inp.atom_block_mask
        )
        a = a + self.single_to_token(self.single_step_norm(s))
        a = self.token_transformer(a, s, inp.z, inp.token_mask, inp.token_block_mask)
        a = self.token_norm(a)
        r_update = self.atom_decoder(
            a, q_skip, inp.c, *inp.rope, inp.atom_mask, inp.atom_to_token, block_mask=inp.atom_block_mask
        )
        c_skip = (sigma * sigma / (sigma * sigma + t * t))[:, None, None]
        c_out = (sigma * t / torch.sqrt(sigma * sigma + t * t))[:, None, None]
        return c_skip * x_noisy + c_out * r_update.float()

    def noise_schedule(self, num_steps: int, device: torch.device) -> Tensor:
        """Karras schedule ``σ_d · (s_max^(1/ρ) + k/(n-1) (s_min^(1/ρ) − s_max^(1/ρ)))^ρ``, trailing 0."""
        cfg = self.config
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1; got {num_steps!r}")
        if num_steps == 1:
            return torch.tensor([cfg.inference_sigma_max * self.sigma_data, 0.0], device=device)
        rho = cfg.inference_rho
        inv = 1.0 / rho
        k = torch.arange(num_steps, device=device, dtype=torch.float32)
        base = cfg.inference_sigma_max**inv + (k / (num_steps - 1)) * (
            cfg.inference_sigma_min**inv - cfg.inference_sigma_max**inv
        )
        return F.pad(self.sigma_data * base.pow(rho), (0, 1), value=0.0)

    def sample(
        self,
        inp: DenoiserInputs,
        *,
        num_steps: int | None = None,
        sigma_cap: float | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Upstream ``DiffusionStructureHead.sample``: churned Euler steps with per-step alignment.

        RNG draw order per upstream: initial noise; then per step a rotation quaternion, a
        translation and the churn noise (drawn even when ``noise_scale == 0``). The sigma cap
        truncates the schedule (``schedule[schedule <= cap]`` prefixed by ``cap``) without
        re-inflating the step count, and the churn gamma is keyed on the *next* sigma.
        """
        cfg = self.config
        steps = cfg.inference_num_steps if num_steps is None else num_steps
        cap = cfg.inference_sigma_cap if sigma_cap is None else sigma_cap
        lam = cfg.noise_scale if noise_scale is None else noise_scale
        eta = cfg.step_scale if step_scale is None else step_scale
        device = inp.z.device
        schedule = self.noise_schedule(steps, device)
        schedule = F.pad(schedule[schedule <= cap], (1, 0), value=cap)
        sigmas = schedule.tolist()
        gammas = [cfg.gamma_0 if s > cfg.gamma_min else 0.0 for s in sigmas]
        n, n_atoms = inp.atom_mask.shape
        atom_mask = inp.atom_mask.float()
        x = sigmas[0] * torch.randn(n, n_atoms, 3, device=device, generator=generator)
        for sigma_tm, sigma_t, gamma in zip(sigmas[:-1], sigmas[1:], gammas[1:], strict=True):
            x = center_random_augmentation(x, atom_mask, generator=generator)
            t_hat = sigma_tm * (1.0 + gamma)
            eps_std = lam * max(t_hat**2 - sigma_tm**2, 0.0) ** 0.5
            x_noisy = x + eps_std * torch.randn(x.shape, device=device, generator=generator)
            x_denoised = self.denoise(x_noisy, torch.full((n,), t_hat, device=device), inp)
            x_noisy = weighted_rigid_align(x_noisy, x_denoised, atom_mask)
            x = x_noisy + eta * (sigma_t - t_hat) * (x_noisy - x_denoised) / t_hat
        return x
```

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_diffusion.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/diffusion.py tests/fold/test_diffusion.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: 8 passed.

```bash
git add src/oplm/fold/diffusion.py tests/fold/test_diffusion.py
git commit -m "feat(fold): diffusion structure head with EDM denoiser and the upstream sampler"
```

---

### Task 8: Confidence head, pTM/ipTM and the distogram readout

**Files:**
- Create: `src/oplm/fold/confidence.py`
- Test: `tests/fold/test_confidence.py`

**Interfaces:**
- Consumes: Task 3 `distance_bins`; Task 4 `PairStack`, `pair_stack_kwargs`, `cuda_bf16_autocast`; Task 5 `gather_token_to_atom`, `scatter_atom_to_token_mean`, `intra_token_index`.
- Produces: `ConfidenceInputEmbedder(config)`, `RowAttentionPooling(pair_width, single_width)`, `ConfidenceOutput` dataclass (`plddt_logits, plddt_per_atom, plddt, pae_logits, pae, pde_logits, pde, resolved_logits, ptm, iptm, pair_chains_iptm, complex_plddt`), `ConfidenceHead(config)` with `forward(*, s_inputs, z, relpos, bonds, coords, distogram_atom_idx, token_mask, atom_to_token, atom_mask, asym_id) -> ConfidenceOutput`, `categorical_mean(logits, start, end)`, `tm_scores(pae_logits, token_mask, asym_id, *, max_dist) -> (ptm, iptm, pair_chains_iptm)`, `symmetrized_distogram(head: nn.Linear, z) -> Tensor`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_confidence.py`:

```python
"""Confidence head against the transcribed upstream forward; pTM/ipTM edge cases; distogram."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from oplm.fold.atoms import gather_token_to_atom, intra_token_index
from oplm.fold.confidence import (
    ConfidenceHead,
    categorical_mean,
    symmetrized_distogram,
    tm_scores,
)
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.featurize import ChainSpec, featurize


def _cfg() -> FoldConfig:
    return FoldConfig(
        pair_width=32, token_width=64, atom_width=32, atom_encoder_blocks=1, atom_encoder_heads=2,
        uid_rope_pairs=2, trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, diffusion_blocks=1,
        diffusion_heads=4, diffusion_atom_blocks=1, diffusion_atom_heads=2, fourier_dim=16,
        confidence_blocks=1, plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5,
        attention_backend="dense", trimul_backend="reference",
    )


def test_names_and_shapes() -> None:
    cfg = _cfg()
    head = ConfidenceHead(cfg)
    names = set(head.state_dict())
    assert {
        "boundaries", "dist_bin_pairwise_embed.weight", "input_embedder.single_inputs_norm.weight",
        "input_embedder.pair_norm.bias", "input_embedder.single_to_pair.weight",
        "input_embedder.single_to_pair_transpose.weight", "input_embedder.single_to_pair_prod_in1.weight",
        "input_embedder.single_to_pair_prod_in2.weight", "input_embedder.single_to_pair_prod_out.weight",
        "folding_trunk.layers.0.tri_mul_out.proj_bundle.weight", "row_attention_pooling.attn_proj.weight",
        "row_attention_pooling.out_proj.weight", "plddt_layernorm.weight", "plddt_weight", "pae_layernorm.bias",
        "pae_head.weight", "pde_layernorm.weight", "pde_head.weight", "resolved_layernorm.bias", "resolved_weight",
    } <= names
    assert len(names) == 25 + 18  # 25 head tensors + one PairUpdateBlock (18)
    assert head.boundaries.shape == (4,) and torch.equal(head.boundaries, torch.linspace(3.25, 50.75, 4))
    assert head.dist_bin_pairwise_embed.weight.shape == (5, 32)
    assert head.plddt_weight.shape == (23, 32, 10) and head.plddt_weight.abs().sum() == 0
    assert head.resolved_weight.shape == (23, 32, 2)
    assert head.row_attention_pooling.out_proj.weight.shape == (32, 32)  # single width = token_width // 2
    assert head.pae_head.weight.shape == (8, 32) and head.pae_head.bias is None


def _run(head: ConfidenceHead, cfg: FoldConfig, chains: list[ChainSpec], num_samples: int, pad: int | None = None):
    f = featurize(chains, pad_tokens_to=pad)
    g = torch.Generator().manual_seed(0)
    L, A = f.num_tokens, f.num_atoms
    s_inputs = torch.randn(1, L, cfg.single_inputs_width, generator=g)
    z = torch.randn(1, L, L, cfg.pair_width, generator=g)
    relpos = torch.randn(1, L, L, cfg.pair_width, generator=g)
    bonds = torch.randn(1, L, L, cfg.pair_width, generator=g)
    coords = torch.randn(num_samples, A, 3, generator=g) * 5
    with torch.no_grad():
        out = head(
            s_inputs=s_inputs, z=z, relpos=relpos, bonds=bonds, coords=coords,
            distogram_atom_idx=f.distogram_atom_idx, token_mask=f.token_mask, atom_to_token=f.atom_to_token,
            atom_mask=f.atom_mask, asym_id=f.asym_id,
        )
    return f, (s_inputs, z, relpos, bonds, coords), out


def test_forward_matches_transcribed_upstream_confidence_head() -> None:
    cfg = _cfg()
    torch.manual_seed(0)
    head = ConfidenceHead(cfg).eval()
    with torch.no_grad():
        head.plddt_weight.normal_()
        head.resolved_weight.normal_()
    f, (s_inputs, z, relpos, bonds, coords), out = _run(head, cfg, [ChainSpec("MKV", "A"), ChainSpec("GG", "B")], 2)
    ie = head.input_embedder
    with torch.no_grad():
        # upstream ConfidenceHead.forward (model.py:179-300), transcribed
        s = ie.single_inputs_norm(s_inputs)
        z_base = ie.pair_norm(z) + relpos + bonds
        z_base = z_base + ie.single_to_pair(s).unsqueeze(2) + ie.single_to_pair_transpose(s).unsqueeze(1)
        z_base = z_base + ie.single_to_pair_prod_out(ie.single_to_pair_prod_in1(s)[:, :, None, :] * ie.single_to_pair_prod_in2(s)[:, None, :, :])
        pair = z_base.repeat_interleave(2, 0)
        rep_idx = f.distogram_atom_idx.repeat_interleave(2, 0)
        rep = torch.gather(coords, 1, rep_idx[..., None].expand(-1, -1, 3))
        d = torch.cdist(rep, rep, compute_mode="donot_use_mm_for_euclid_dist")
        bins = (d.unsqueeze(-1) > head.boundaries).sum(-1).long()
        pair = pair + head.dist_bin_pairwise_embed(bins)
        mask = f.token_mask.repeat_interleave(2, 0)
        pair_mask = mask[:, :, None].float() * mask[:, None, :].float()
        pair = pair + head.folding_trunk(pair, pair_mask).float()  # the upstream residual quirk
        scores = head.row_attention_pooling.attn_proj(pair).squeeze(-1) + torch.where(mask[:, None, :], 0.0, -1e9)
        single = head.row_attention_pooling.out_proj(torch.einsum("bnm,bnmd->bnd", scores.softmax(-1), pair))
        pae_logits = head.pae_head(head.pae_layernorm(pair))
        a2t = f.atom_to_token.repeat_interleave(2, 0)
        s_atoms = gather_token_to_atom(single, a2t)
        slot = intra_token_index(a2t).clamp(max=22)
        plddt_logits = torch.einsum("...c,...cb->...b", head.plddt_layernorm(s_atoms), head.plddt_weight[slot])
    torch.testing.assert_close(out.pae_logits, pae_logits)
    torch.testing.assert_close(out.plddt_logits, plddt_logits)
    torch.testing.assert_close(out.plddt_per_atom, categorical_mean(plddt_logits, 0.0, 1.0))
    torch.testing.assert_close(out.pae, categorical_mean(pae_logits, 0.0, 32.0))
    assert out.plddt.shape == (2, 5) and out.resolved_logits.shape == (2, f.num_atoms, 2)
    assert out.pair_chains_iptm.shape == (2, 2, 2) and out.complex_plddt.shape == (2,)
    assert (out.plddt >= 0).all() and (out.plddt <= 1).all()


def test_categorical_mean_bin_centers() -> None:
    logits = torch.full((1, 4), -1e9)
    logits[0, 2] = 0.0
    torch.testing.assert_close(categorical_mean(logits, 0.0, 1.0), torch.tensor([0.625]))  # center of bin 2 of 4
    torch.testing.assert_close(categorical_mean(logits, 0.0, 32.0), torch.tensor([20.0]))


def test_tm_scores_match_transcription_and_single_chain_iptm_is_zero() -> None:
    g = torch.Generator().manual_seed(0)
    pae_logits = torch.randn(2, 6, 6, 8, generator=g)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 5] = False
    asym = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    ptm, iptm, chains = tm_scores(pae_logits, mask, asym, max_dist=32.0)
    # upstream (model.py:340-386), transcribed
    bw = 32.0 / 8
    centers = torch.arange(0.5 * bw, 32.0, bw)
    mask_f = mask.float()
    n_res = mask_f.sum(-1, keepdim=True)
    d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1 / (1 + (centers / d0) ** 2)
    tm_expected = (F.softmax(pae_logits, -1) * tm_per_bin[:, None, None, :]).sum(-1)
    pair = mask_f[..., None] * mask_f[:, None, :]
    ptm_ref = ((tm_expected * pair).sum(-1) / (pair.sum(-1) + 1e-6)).max(-1).values
    inter = (asym[..., None] != asym[:, None, :]).float() * pair
    iptm_ref = ((tm_expected * inter).sum(-1) / (inter.sum(-1) + 1e-6)).max(-1).values
    torch.testing.assert_close(ptm, ptm_ref)
    torch.testing.assert_close(iptm, iptm_ref)
    assert iptm[1] == 0.0  # Review Focus 3: single chain -> no inter-chain pairs -> 0, not NaN
    assert chains.shape == (2, 2, 2) and torch.isfinite(chains).all()
    assert chains[1, 1, 1] == 0.0 and chains[0, 0, 1] >= 0


def test_all_padding_row_is_finite() -> None:
    """Review Focus 2: a batch row with every token padded must not produce NaN anywhere."""
    cfg = _cfg()
    torch.manual_seed(0)
    head = ConfidenceHead(cfg).eval()
    f = featurize([ChainSpec("MK", "A")], pad_tokens_to=4)
    L, A = 4, f.num_atoms
    with torch.no_grad():
        out = head(
            s_inputs=torch.randn(2, L, cfg.single_inputs_width), z=torch.randn(2, L, L, 32),
            relpos=torch.zeros(2, L, L, 32), bonds=torch.zeros(2, L, L, 32), coords=torch.randn(2, A, 3),
            distogram_atom_idx=f.distogram_atom_idx.expand(2, L),
            token_mask=torch.stack([f.token_mask[0], torch.zeros(L, dtype=torch.bool)]),
            atom_to_token=f.atom_to_token.expand(2, A),
            atom_mask=torch.stack([f.atom_mask[0], torch.zeros(A, dtype=torch.bool)]),
            asym_id=f.asym_id.expand(2, L),
        )
    for name in ("plddt", "pae", "pde", "ptm", "iptm", "pair_chains_iptm", "complex_plddt"):
        assert torch.isfinite(getattr(out, name)).all(), name
    assert out.ptm[1] == 0.0 and out.complex_plddt[1] == 0.0


def test_symmetrized_distogram() -> None:
    torch.manual_seed(0)
    head = nn.Linear(32, 8)
    z = torch.randn(1, 5, 5, 32)
    out = symmetrized_distogram(head, z)
    torch.testing.assert_close(out, head(z + z.transpose(1, 2)))
    torch.testing.assert_close(out, out.transpose(1, 2))
```

Run: `.venv/bin/python -m pytest tests/fold/test_confidence.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.confidence'`.

- [ ] **Step 2: `src/oplm/fold/confidence.py`**

```python
"""Confidence head (pLDDT, PAE, PDE, resolved, pTM/ipTM) and the symmetrised distogram readout.

Ported from Biohub's ESMFold2 ``ConfidenceHead``, ``RowAttentionPooling``, ``_categorical_mean``
and the pTM/ipTM code (esm/models/esmfold2/{model,layers}.py, Apache-2.0; see
THIRD_PARTY_NOTICES.md), with the released checkpoint's names (``confidence_head.{boundaries,
dist_bin_pairwise_embed, input_embedder.*, folding_trunk, row_attention_pooling,
{plddt,pae,pde,resolved}_layernorm, plddt_weight, pae_head, pde_head, resolved_weight}``).
Modifications: the distance binning is the block-local ``distance_bins`` (spec §4.6); the
pair trunk's residual quirk (``pair + Trunk(pair)`` although ``Trunk`` already carries its own
residuals) is reproduced deliberately; unused upstream modules (``s_norm``,
``s_inputs_to_single``, ``s_input_to_s``) are not ported; interface pLDDT is deferred to
milestone 2. pLDDT is on the 0–1 scale, PAE/PDE in Å (64 bins over 0–32 Å by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from oplm.fold.atoms import gather_token_to_atom, intra_token_index, scatter_atom_to_token_mean
from oplm.fold.pair import distance_bins
from oplm.fold.trunk import PairStack, cuda_bf16_autocast, pair_stack_kwargs

if TYPE_CHECKING:
    from torch import Tensor

    from oplm.fold.configuration_fold import FoldConfig

__all__ = [
    "ConfidenceHead",
    "ConfidenceInputEmbedder",
    "ConfidenceOutput",
    "RowAttentionPooling",
    "categorical_mean",
    "symmetrized_distogram",
    "tm_scores",
]

_EPS = 1e-6


def symmetrized_distogram(head: nn.Linear, z: Tensor) -> Tensor:
    """``distogram_head(z + zᵀ)`` on the fp32 final pair (upstream reads the symmetrised pair)."""
    return head(z + z.transpose(1, 2))


def categorical_mean(logits: Tensor, start: float, end: float) -> Tensor:
    """Expected value over equal-width bins spanning ``[start, end]`` (bin centers)."""
    n_bins = logits.shape[-1]
    edges = torch.linspace(start, end, n_bins + 1, device=logits.device, dtype=torch.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    return logits.float().softmax(dim=-1) @ centers


def tm_scores(
    pae_logits: Tensor, token_mask: Tensor, asym_id: Tensor, *, max_dist: float = 32.0
) -> tuple[Tensor, Tensor, Tensor]:
    """pTM, ipTM and per-chain-pair ipTM from PAE logits (upstream ``model.py:340-386``).

    ``d0 = 1.24 (max(N, 19) − 15)^(1/3) − 1.8`` with ``N`` the valid-token count; pTM is the
    max over frame rows of the masked mean of the expected TM term; ipTM restricts columns to
    other chains (0 for a single chain); ``pair_chains_iptm[c1, c2]`` is the max over rows in
    chain ``c2`` of the mean over columns in chain ``c1``.
    """
    n_bins = pae_logits.shape[-1]
    bin_width = max_dist / n_bins
    centers = torch.arange(0.5 * bin_width, max_dist, bin_width, device=pae_logits.device)
    mask_f = token_mask.float()
    n_res = mask_f.sum(dim=-1, keepdim=True)
    d0 = 1.24 * (n_res.clamp(min=19) - 15) ** (1 / 3) - 1.8
    tm_per_bin = 1 / (1 + (centers[None, :] / d0) ** 2)
    tm_expected = (pae_logits.float().softmax(dim=-1) * tm_per_bin[:, None, None, :]).sum(-1)
    pair = mask_f[:, :, None] * mask_f[:, None, :]
    ptm = ((tm_expected * pair).sum(-1) / (pair.sum(-1) + _EPS)).max(-1).values
    inter = (asym_id[:, :, None] != asym_id[:, None, :]).float() * pair
    iptm = ((tm_expected * inter).sum(-1) / (inter.sum(-1) + _EPS)).max(-1).values
    n_chains = int(asym_id.max().item()) + 1
    chains = torch.zeros(pae_logits.shape[0], n_chains, n_chains, device=pae_logits.device)
    for c1 in range(n_chains):
        cols = (asym_id == c1).float() * mask_f
        row_vals = (tm_expected * cols[:, None, :]).sum(-1) / (cols[:, None, :].sum(-1) + _EPS)
        for c2 in range(n_chains):
            rows = (asym_id == c2) & token_mask
            masked = row_vals.masked_fill(~rows, float("-inf")).max(-1).values
            chains[:, c1, c2] = masked.clamp(min=0.0)
    return ptm, iptm, chains


class ConfidenceInputEmbedder(nn.Module):
    """Checkpoint ``confidence_head.input_embedder``: the pair the confidence trunk starts from."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        s_in, p, eps = config.single_inputs_width, config.pair_width, config.layer_norm_eps
        self.single_inputs_norm = nn.LayerNorm(s_in, eps=eps)
        self.pair_norm = nn.LayerNorm(p, eps=eps)
        self.single_to_pair = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_transpose = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_in1 = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_in2 = nn.Linear(s_in, p, bias=False)
        self.single_to_pair_prod_out = nn.Linear(p, p, bias=False)

    def forward(self, s_inputs: Tensor, z: Tensor, relpos: Tensor, bonds: Tensor) -> Tensor:
        s = self.single_inputs_norm(s_inputs)
        z = self.pair_norm(z) + relpos + bonds
        z = z + self.single_to_pair(s)[:, :, None, :] + self.single_to_pair_transpose(s)[:, None, :, :]
        prod = self.single_to_pair_prod_in1(s)[:, :, None, :] * self.single_to_pair_prod_in2(s)[:, None, :, :]
        return z + self.single_to_pair_prod_out(prod)


class RowAttentionPooling(nn.Module):
    """Softmax over columns of a learned score, masked at padded columns, then a projection."""

    def __init__(self, pair_width: int, single_width: int) -> None:
        super().__init__()
        self.attn_proj = nn.Linear(pair_width, 1, bias=False)
        self.out_proj = nn.Linear(pair_width, single_width, bias=False)

    def forward(self, z: Tensor, token_mask: Tensor) -> Tensor:
        scores = self.attn_proj(z).squeeze(-1) + torch.where(token_mask[:, None, :], 0.0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return self.out_proj(torch.einsum("bnm,bnmd->bnd", weights, z))


@dataclass
class ConfidenceOutput:
    """Per-sample confidence tensors; leading dim is ``B · num_samples``."""

    plddt_logits: Tensor  # (N, A, plddt_bins)
    plddt_per_atom: Tensor  # (N, A) in [0, 1]
    plddt: Tensor  # (N, L) masked mean over each token's atoms
    pae_logits: Tensor  # (N, L, L, pae_bins)
    pae: Tensor  # (N, L, L) Å
    pde_logits: Tensor
    pde: Tensor
    resolved_logits: Tensor  # (N, A, 2)
    ptm: Tensor  # (N,)
    iptm: Tensor  # (N,)
    pair_chains_iptm: Tensor  # (N, C, C)
    complex_plddt: Tensor  # (N,) mean over valid atoms


class ConfidenceHead(nn.Module):
    """Checkpoint ``confidence_head``."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__()
        self.config = config
        p, single, eps = config.pair_width, config.inputs_token_width, config.layer_norm_eps
        self.register_buffer(
            "boundaries",
            torch.linspace(
                config.confidence_min_dist, config.confidence_max_dist, config.confidence_dist_bins - 1
            ),
        )
        self.dist_bin_pairwise_embed = nn.Embedding(config.confidence_dist_bins, p)
        self.input_embedder = ConfidenceInputEmbedder(config)
        self.folding_trunk = PairStack(config.confidence_blocks, p, **pair_stack_kwargs(config))
        self.row_attention_pooling = RowAttentionPooling(p, single)
        self.plddt_layernorm = nn.LayerNorm(single, eps=eps)
        self.plddt_weight = nn.Parameter(torch.zeros(config.max_atoms_per_token, single, config.plddt_bins))
        self.pae_layernorm = nn.LayerNorm(p, eps=eps)
        self.pae_head = nn.Linear(p, config.pae_bins, bias=False)
        self.pde_layernorm = nn.LayerNorm(p, eps=eps)
        self.pde_head = nn.Linear(p, config.pde_bins, bias=False)
        self.resolved_layernorm = nn.LayerNorm(single, eps=eps)
        self.resolved_weight = nn.Parameter(torch.zeros(config.max_atoms_per_token, single, 2))

    def forward(
        self,
        *,
        s_inputs: Tensor,
        z: Tensor,
        relpos: Tensor,
        bonds: Tensor,
        coords: Tensor,
        distogram_atom_idx: Tensor,
        token_mask: Tensor,
        atom_to_token: Tensor,
        atom_mask: Tensor,
        asym_id: Tensor,
    ) -> ConfidenceOutput:
        """Score ``coords (B·S, A, 3)`` against the base-batch trunk tensors (repeated per sample)."""
        num_samples = coords.shape[0] // z.shape[0]

        def rep(t: Tensor) -> Tensor:
            return t.repeat_interleave(num_samples, dim=0)

        pair = rep(self.input_embedder(s_inputs, z.float(), relpos.float(), bonds.float()))
        token_mask, atom_to_token, atom_mask, asym_id = map(rep, (token_mask, atom_to_token, atom_mask, asym_id))
        rep_idx = rep(distogram_atom_idx)
        rep_coords = torch.gather(coords.float(), 1, rep_idx[..., None].expand(-1, -1, 3))
        pair = pair + self.dist_bin_pairwise_embed(distance_bins(rep_coords, self.boundaries))
        pair_mask = token_mask[:, :, None].float() * token_mask[:, None, :].float()
        with cuda_bf16_autocast(pair.is_cuda):
            delta = self.folding_trunk(pair, pair_mask)
        pair = pair + delta.float()  # upstream quirk: ``delta`` already includes the residual
        single = self.row_attention_pooling(pair, token_mask)
        pae_logits = self.pae_head(self.pae_layernorm(pair))
        pde_logits = self.pde_head(self.pde_layernorm(pair))
        s_atoms = gather_token_to_atom(single, atom_to_token)
        slot = intra_token_index(atom_to_token).clamp(max=self.plddt_weight.shape[0] - 1)
        plddt_logits = torch.einsum("...c,...cb->...b", self.plddt_layernorm(s_atoms), self.plddt_weight[slot])
        resolved_logits = torch.einsum(
            "...c,...cb->...b", self.resolved_layernorm(s_atoms), self.resolved_weight[slot]
        )
        plddt_per_atom = categorical_mean(plddt_logits, 0.0, 1.0)
        plddt = scatter_atom_to_token_mean(
            plddt_per_atom[..., None], atom_to_token, token_mask.shape[1], atom_mask
        )[..., 0]
        atom_f = atom_mask.float()
        complex_plddt = (plddt_per_atom * atom_f).sum(-1) / atom_f.sum(-1).clamp(min=1.0)
        ptm, iptm, pair_chains_iptm = tm_scores(
            pae_logits, token_mask, asym_id, max_dist=self.config.pae_max_dist
        )
        return ConfidenceOutput(
            plddt_logits=plddt_logits, plddt_per_atom=plddt_per_atom, plddt=plddt,
            pae_logits=pae_logits, pae=categorical_mean(pae_logits, 0.0, self.config.pae_max_dist),
            pde_logits=pde_logits, pde=categorical_mean(pde_logits, 0.0, self.config.pae_max_dist),
            resolved_logits=resolved_logits, ptm=ptm, iptm=iptm, pair_chains_iptm=pair_chains_iptm,
            complex_plddt=complex_plddt,
        )
```

- [ ] **Step 3: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_confidence.py -q && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/confidence.py tests/fold/test_confidence.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: 6 passed.

```bash
git add src/oplm/fold/confidence.py tests/fold/test_confidence.py
git commit -m "feat(fold): confidence head with pTM/ipTM and the symmetrised distogram readout"
```

---

### Task 9: `OplmForFolding`: the HF model, frozen-LM ownership, registration, save/load

**Files:**
- Create: `src/oplm/fold/modeling_fold.py`, `tests/fold/data/esmfold2_fast_head_keys.txt`
- Modify: `src/oplm/fold/__init__.py`
- Test: `tests/fold/test_modeling.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8; `oplm.model.OplmModel`; `oplm.training.ema.build_ema`.
- Produces: `OplmFoldPreTrainedModel(PreTrainedModel)` (`config_class = FoldConfig`, tag-aware `_init_weights`), `FoldOutput` dataclass (`distogram_logits, coords, confidence, s_inputs, pair, intermediates`), `OplmForFolding(config)` with children `input_embedder, language_model, lm_encoder, folding_trunk, parcae, distogram_head, structure_head, confidence_head`; `lm` property; `attach_lm(lm, *, name_or_path=None, revision=None)`; `load_frozen_lm(name_or_path, *, revision=None, dtype=None, device=None) -> OplmModel`; classmethod `from_pretrained(path, *, lm=None, lm_name_or_path=None, lm_revision=None, **kwargs)`; `forward(features, *, lm_hidden_states=None, num_loops=None, num_samples=None, num_steps=None, z0=None, generator=None, return_intermediates=False) -> FoldOutput`. `oplm.fold.__init__` lazily exports `FoldConfig`, `OplmForFolding`, `FoldOutput`, `ChainSpec`, `FoldFeatures`, `featurize`; importing `oplm.fold.modeling_fold` registers `oplm_fold` with `AutoConfig`/`AutoModel`.

- [ ] **Step 1: Write the checkpoint-pattern file**

`tests/fold/data/esmfold2_fast_head_keys.txt` — every tensor name in the released
`biohub/ESMFold2-Fast` index except the `esmc.*` ones, with layer indices replaced by `N`
(213 lines, derived from `model.safetensors.index.json` at revision `45fe8656`):

```text
confidence_head.boundaries
confidence_head.dist_bin_pairwise_embed.weight
confidence_head.folding_trunk.layers.N.pair_transition.mlp.down_proj.weight
confidence_head.folding_trunk.layers.N.pair_transition.mlp.gate_up_proj.weight
confidence_head.folding_trunk.layers.N.pair_transition.norm.bias
confidence_head.folding_trunk.layers.N.pair_transition.norm.weight
confidence_head.folding_trunk.layers.N.tri_mul_in.norm_mix.bias
confidence_head.folding_trunk.layers.N.tri_mul_in.norm_mix.weight
confidence_head.folding_trunk.layers.N.tri_mul_in.norm_start.bias
confidence_head.folding_trunk.layers.N.tri_mul_in.norm_start.weight
confidence_head.folding_trunk.layers.N.tri_mul_in.proj_bundle.weight
confidence_head.folding_trunk.layers.N.tri_mul_in.proj_emit.weight
confidence_head.folding_trunk.layers.N.tri_mul_in.proj_gate.weight
confidence_head.folding_trunk.layers.N.tri_mul_out.norm_mix.bias
confidence_head.folding_trunk.layers.N.tri_mul_out.norm_mix.weight
confidence_head.folding_trunk.layers.N.tri_mul_out.norm_start.bias
confidence_head.folding_trunk.layers.N.tri_mul_out.norm_start.weight
confidence_head.folding_trunk.layers.N.tri_mul_out.proj_bundle.weight
confidence_head.folding_trunk.layers.N.tri_mul_out.proj_emit.weight
confidence_head.folding_trunk.layers.N.tri_mul_out.proj_gate.weight
confidence_head.input_embedder.pair_norm.bias
confidence_head.input_embedder.pair_norm.weight
confidence_head.input_embedder.single_inputs_norm.bias
confidence_head.input_embedder.single_inputs_norm.weight
confidence_head.input_embedder.single_to_pair.weight
confidence_head.input_embedder.single_to_pair_prod_in1.weight
confidence_head.input_embedder.single_to_pair_prod_in2.weight
confidence_head.input_embedder.single_to_pair_prod_out.weight
confidence_head.input_embedder.single_to_pair_transpose.weight
confidence_head.pae_head.weight
confidence_head.pae_layernorm.bias
confidence_head.pae_layernorm.weight
confidence_head.pde_head.weight
confidence_head.pde_layernorm.bias
confidence_head.pde_layernorm.weight
confidence_head.plddt_layernorm.bias
confidence_head.plddt_layernorm.weight
confidence_head.plddt_weight
confidence_head.resolved_layernorm.bias
confidence_head.resolved_layernorm.weight
confidence_head.resolved_weight
confidence_head.row_attention_pooling.attn_proj.weight
confidence_head.row_attention_pooling.out_proj.weight
distogram_head.bias
distogram_head.weight
folding_trunk.layers.N.pair_transition.mlp.down_proj.weight
folding_trunk.layers.N.pair_transition.mlp.gate_up_proj.weight
folding_trunk.layers.N.pair_transition.norm.bias
folding_trunk.layers.N.pair_transition.norm.weight
folding_trunk.layers.N.tri_mul_in.norm_mix.bias
folding_trunk.layers.N.tri_mul_in.norm_mix.weight
folding_trunk.layers.N.tri_mul_in.norm_start.bias
folding_trunk.layers.N.tri_mul_in.norm_start.weight
folding_trunk.layers.N.tri_mul_in.proj_bundle.weight
folding_trunk.layers.N.tri_mul_in.proj_emit.weight
folding_trunk.layers.N.tri_mul_in.proj_gate.weight
folding_trunk.layers.N.tri_mul_out.norm_mix.bias
folding_trunk.layers.N.tri_mul_out.norm_mix.weight
folding_trunk.layers.N.tri_mul_out.norm_start.bias
folding_trunk.layers.N.tri_mul_out.norm_start.weight
folding_trunk.layers.N.tri_mul_out.proj_bundle.weight
folding_trunk.layers.N.tri_mul_out.proj_emit.weight
folding_trunk.layers.N.tri_mul_out.proj_gate.weight
input_embedder.atom_encoder.atom_linear.weight
input_embedder.atom_encoder.atom_norm.bias
input_embedder.atom_encoder.atom_norm.weight
input_embedder.atom_encoder.atom_to_token_linear.weight
input_embedder.atom_encoder.layers.N.adaln_linear.weight
input_embedder.atom_encoder.layers.N.mlp.down_proj.weight
input_embedder.atom_encoder.layers.N.mlp.gate_up_proj.weight
input_embedder.atom_encoder.layers.N.self_attn.gate_proj.weight
input_embedder.atom_encoder.layers.N.self_attn.k_proj.weight
input_embedder.atom_encoder.layers.N.self_attn.o_proj.weight
input_embedder.atom_encoder.layers.N.self_attn.q_proj.weight
input_embedder.atom_encoder.layers.N.self_attn.v_proj.weight
input_embedder.pair_init_1.weight
input_embedder.pair_init_2.weight
input_embedder.rel_pos.embed.weight
input_embedder.token_bonds.weight
language_model.layer_weights
language_model.pair_input_norm.bias
language_model.pair_input_norm.weight
language_model.pair_output_norm.bias
language_model.pair_output_norm.weight
language_model.pair_proj.weight
language_model.single_to_pair.downproject.bias
language_model.single_to_pair.downproject.weight
language_model.single_to_pair.output_fc1.bias
language_model.single_to_pair.output_fc1.weight
language_model.single_to_pair.output_fc2.bias
language_model.single_to_pair.output_fc2.weight
lm_encoder.layers.N.pair_transition.mlp.down_proj.weight
lm_encoder.layers.N.pair_transition.mlp.gate_up_proj.weight
lm_encoder.layers.N.pair_transition.norm.bias
lm_encoder.layers.N.pair_transition.norm.weight
lm_encoder.layers.N.tri_mul_in.norm_mix.bias
lm_encoder.layers.N.tri_mul_in.norm_mix.weight
lm_encoder.layers.N.tri_mul_in.norm_start.bias
lm_encoder.layers.N.tri_mul_in.norm_start.weight
lm_encoder.layers.N.tri_mul_in.proj_bundle.weight
lm_encoder.layers.N.tri_mul_in.proj_emit.weight
lm_encoder.layers.N.tri_mul_in.proj_gate.weight
lm_encoder.layers.N.tri_mul_out.norm_mix.bias
lm_encoder.layers.N.tri_mul_out.norm_mix.weight
lm_encoder.layers.N.tri_mul_out.norm_start.bias
lm_encoder.layers.N.tri_mul_out.norm_start.weight
lm_encoder.layers.N.tri_mul_out.proj_bundle.weight
lm_encoder.layers.N.tri_mul_out.proj_emit.weight
lm_encoder.layers.N.tri_mul_out.proj_gate.weight
parcae.input_matrix_continuous
parcae.input_norm.bias
parcae.input_norm.weight
parcae.log_delta
parcae.log_state_decay
parcae.out_proj.weight
parcae.output_stack.layers.N.pair_transition.mlp.down_proj.weight
parcae.output_stack.layers.N.pair_transition.mlp.gate_up_proj.weight
parcae.output_stack.layers.N.pair_transition.norm.bias
parcae.output_stack.layers.N.pair_transition.norm.weight
parcae.output_stack.layers.N.tri_mul_in.norm_mix.bias
parcae.output_stack.layers.N.tri_mul_in.norm_mix.weight
parcae.output_stack.layers.N.tri_mul_in.norm_start.bias
parcae.output_stack.layers.N.tri_mul_in.norm_start.weight
parcae.output_stack.layers.N.tri_mul_in.proj_bundle.weight
parcae.output_stack.layers.N.tri_mul_in.proj_emit.weight
parcae.output_stack.layers.N.tri_mul_in.proj_gate.weight
parcae.output_stack.layers.N.tri_mul_out.norm_mix.bias
parcae.output_stack.layers.N.tri_mul_out.norm_mix.weight
parcae.output_stack.layers.N.tri_mul_out.norm_start.bias
parcae.output_stack.layers.N.tri_mul_out.norm_start.weight
parcae.output_stack.layers.N.tri_mul_out.proj_bundle.weight
parcae.output_stack.layers.N.tri_mul_out.proj_emit.weight
parcae.output_stack.layers.N.tri_mul_out.proj_gate.weight
structure_head.atom_decoder.layers.N.adaln_linear.weight
structure_head.atom_decoder.layers.N.mlp.down_proj.weight
structure_head.atom_decoder.layers.N.mlp.gate_up_proj.weight
structure_head.atom_decoder.layers.N.self_attn.gate_proj.weight
structure_head.atom_decoder.layers.N.self_attn.k_proj.weight
structure_head.atom_decoder.layers.N.self_attn.o_proj.weight
structure_head.atom_decoder.layers.N.self_attn.q_proj.weight
structure_head.atom_decoder.layers.N.self_attn.v_proj.weight
structure_head.atom_decoder.norm.bias
structure_head.atom_decoder.norm.weight
structure_head.atom_decoder.output_linear.weight
structure_head.atom_decoder.token_to_atom_linear.weight
structure_head.atom_encoder.atom_linear.weight
structure_head.atom_encoder.atom_norm.bias
structure_head.atom_encoder.atom_norm.weight
structure_head.atom_encoder.atom_to_token_linear.weight
structure_head.atom_encoder.layers.N.adaln_linear.weight
structure_head.atom_encoder.layers.N.mlp.down_proj.weight
structure_head.atom_encoder.layers.N.mlp.gate_up_proj.weight
structure_head.atom_encoder.layers.N.self_attn.gate_proj.weight
structure_head.atom_encoder.layers.N.self_attn.k_proj.weight
structure_head.atom_encoder.layers.N.self_attn.o_proj.weight
structure_head.atom_encoder.layers.N.self_attn.q_proj.weight
structure_head.atom_encoder.layers.N.self_attn.v_proj.weight
structure_head.conditioning.fourier.frequencies
structure_head.conditioning.fourier.phases
structure_head.conditioning.noise_norm.bias
structure_head.conditioning.noise_norm.weight
structure_head.conditioning.noise_proj.weight
structure_head.conditioning.pair_input_norm.bias
structure_head.conditioning.pair_input_norm.weight
structure_head.conditioning.pair_proj.weight
structure_head.conditioning.pair_transition_0.mlp.down_proj.weight
structure_head.conditioning.pair_transition_0.mlp.gate_up_proj.weight
structure_head.conditioning.pair_transition_0.norm.bias
structure_head.conditioning.pair_transition_0.norm.weight
structure_head.conditioning.pair_transition_1.mlp.down_proj.weight
structure_head.conditioning.pair_transition_1.mlp.gate_up_proj.weight
structure_head.conditioning.pair_transition_1.norm.bias
structure_head.conditioning.pair_transition_1.norm.weight
structure_head.conditioning.single_input_norm.bias
structure_head.conditioning.single_input_norm.weight
structure_head.conditioning.single_proj.weight
structure_head.conditioning.single_transition_0.mlp.down_proj.weight
structure_head.conditioning.single_transition_0.mlp.gate_up_proj.weight
structure_head.conditioning.single_transition_0.norm.bias
structure_head.conditioning.single_transition_0.norm.weight
structure_head.conditioning.single_transition_1.mlp.down_proj.weight
structure_head.conditioning.single_transition_1.mlp.gate_up_proj.weight
structure_head.conditioning.single_transition_1.norm.bias
structure_head.conditioning.single_transition_1.norm.weight
structure_head.coords_linear.weight
structure_head.single_step_norm.bias
structure_head.single_step_norm.weight
structure_head.single_to_token.weight
structure_head.token_norm.bias
structure_head.token_norm.weight
structure_head.token_transformer.layers.N.attn_gate.bias
structure_head.token_transformer.layers.N.attn_gate.weight
structure_head.token_transformer.layers.N.input_layernorm.cond_norm.weight
structure_head.token_transformer.layers.N.input_layernorm.gate_proj.bias
structure_head.token_transformer.layers.N.input_layernorm.gate_proj.weight
structure_head.token_transformer.layers.N.input_layernorm.shift_proj.weight
structure_head.token_transformer.layers.N.mlp.down_proj.weight
structure_head.token_transformer.layers.N.mlp.gate_up_proj.weight
structure_head.token_transformer.layers.N.mlp_gate.bias
structure_head.token_transformer.layers.N.mlp_gate.weight
structure_head.token_transformer.layers.N.pair_bias_proj.weight
structure_head.token_transformer.layers.N.pair_norm.bias
structure_head.token_transformer.layers.N.pair_norm.weight
structure_head.token_transformer.layers.N.post_attention_layernorm.cond_norm.weight
structure_head.token_transformer.layers.N.post_attention_layernorm.gate_proj.bias
structure_head.token_transformer.layers.N.post_attention_layernorm.gate_proj.weight
structure_head.token_transformer.layers.N.post_attention_layernorm.shift_proj.weight
structure_head.token_transformer.layers.N.self_attn.gate_proj.weight
structure_head.token_transformer.layers.N.self_attn.k_proj.weight
structure_head.token_transformer.layers.N.self_attn.o_proj.weight
structure_head.token_transformer.layers.N.self_attn.q_proj.bias
structure_head.token_transformer.layers.N.self_attn.q_proj.weight
structure_head.token_transformer.layers.N.self_attn.v_proj.weight
```

- [ ] **Step 2: Write the failing tests**

`tests/fold/test_modeling.py`:

```python
"""OplmForFolding: checkpoint-name contract, custom inits, frozen-LM ownership, forward, save/load."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from transformers import AutoConfig, AutoModel

from oplm.fold import ChainSpec, FoldConfig, OplmForFolding, featurize
from oplm.fold.lm_shim import lm_state_count
from oplm.model import OplmConfig, OplmModel
from oplm.training.ema import build_ema

if TYPE_CHECKING:
    from torch import Tensor

_KEYS = Path(__file__).parent / "data" / "esmfold2_fast_head_keys.txt"


def _tiny_cfg(**over) -> FoldConfig:
    base = dict(
        pair_width=32, token_width=64, atom_width=32, atom_encoder_blocks=1, atom_encoder_heads=2,
        uid_rope_pairs=2, trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, diffusion_blocks=1,
        diffusion_heads=4, diffusion_atom_blocks=1, diffusion_atom_heads=2, fourier_dim=16,
        confidence_blocks=1, plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5,
        distogram_bins=8, inference_num_steps=2, inference_num_loops=2, lm_hidden_size=32,
        lm_num_hidden_states=3, attention_backend="dense", trimul_backend="reference",
    )
    base.update(over)
    return FoldConfig(**base)


def _tiny_lm() -> OplmModel:
    torch.manual_seed(1)
    cfg = OplmConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)
    return OplmModel(cfg).eval()


def test_state_dict_matches_released_checkpoint_patterns() -> None:
    expected = set(_KEYS.read_text().split())
    assert len(expected) == 213
    with torch.device("meta"):
        model = OplmForFolding(FoldConfig())
    keys = list(model.state_dict())
    assert len(keys) == 1054
    patterns = {re.sub(r"\.\d+\.", ".N.", k) for k in keys}
    assert patterns == expected
    counts = {}
    for k in keys:
        m = re.match(r"(.*?)\.layers\.(\d+)\.", k)
        if m:
            counts[m.group(1)] = max(counts.get(m.group(1), 0), int(m.group(2)) + 1)
    assert counts == {
        "confidence_head.folding_trunk": 4, "folding_trunk": 24, "input_embedder.atom_encoder": 3,
        "lm_encoder": 4, "parcae.output_stack": 2, "structure_head.atom_decoder": 3,
        "structure_head.atom_encoder": 3, "structure_head.token_transformer": 12,
    }
    sd = model.state_dict()
    assert sd["distogram_head.weight"].shape == (64, 256)
    assert sd["structure_head.conditioning.pair_transition_0.mlp.gate_up_proj.weight"].shape == (1024, 256)
    assert sd["structure_head.token_transformer.layers.0.mlp.gate_up_proj.weight"].shape == (3072, 768)
    assert sd["confidence_head.plddt_weight"].shape == (23, 384, 50)
    assert sd["language_model.layer_weights"].shape == (25,)


def _assert_custom_inits(model: OplmForFolding) -> None:
    sd = model.state_dict()
    for i in range(model.config.diffusion_blocks):
        for g in ("attn_gate", "mlp_gate"):
            assert sd[f"structure_head.token_transformer.layers.{i}.{g}.weight"].abs().sum() == 0
            assert (sd[f"structure_head.token_transformer.layers.{i}.{g}.bias"] == -2.0).all()
    assert sd["structure_head.single_to_token.weight"].abs().sum() == 0
    assert sd["input_embedder.atom_encoder.layers.0.adaln_linear.weight"].abs().sum() == 0
    torch.testing.assert_close(sd["parcae.out_proj.weight"], torch.eye(model.config.pair_width))
    torch.testing.assert_close(sd["parcae.input_matrix_continuous"], torch.eye(model.config.pair_width))
    assert sd["confidence_head.plddt_weight"].abs().sum() == 0 and sd["language_model.layer_weights"].abs().sum() == 0
    assert sd["input_embedder.pair_init_1.weight"].abs().sum() > 0  # generic init still runs elsewhere


def test_custom_inits_survive_post_init_and_reload(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg())
    _assert_custom_inits(model)
    model.save_pretrained(tmp_path)
    reloaded = OplmForFolding.from_pretrained(tmp_path)
    _assert_custom_inits(reloaded)
    for k, v in model.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k]), k


def test_frozen_lm_is_shared_under_deepcopy_and_stays_eval(tmp_path: Path) -> None:
    """Review Focus 5 and spec §6.6."""
    model = OplmForFolding(_tiny_cfg())
    lm = _tiny_lm()
    model.attach_lm(lm, name_or_path="tiny-lm", revision="r1")
    assert model.lm is lm and model.config.lm_name_or_path == "tiny-lm" and model.config.lm_revision == "r1"
    assert all(not p.requires_grad for p in lm.parameters())
    assert not any(n.startswith("_lm") or "backbone" in n for n, _ in model.named_parameters())
    assert not any("backbone" in k for k in model.state_dict())
    clone = copy.deepcopy(model)
    assert clone.lm is lm and clone.parcae is not model.parcae
    ema = build_ema(model, 0.99)
    assert ema.module.lm is lm
    model.train()
    assert model.training and not model.lm.training
    model.save_pretrained(tmp_path)
    from safetensors import safe_open

    with safe_open(str(tmp_path / "model.safetensors"), "pt") as f:
        assert not any("backbone" in k for k in f.keys())


def test_attach_lm_validates_the_shape_contract() -> None:
    model = OplmForFolding(_tiny_cfg(lm_hidden_size=16))
    with pytest.raises(ValueError, match="lm_hidden_size"):
        model.attach_lm(_tiny_lm())
    model = OplmForFolding(_tiny_cfg(lm_num_hidden_states=4))
    with pytest.raises(ValueError, match="lm_num_hidden_states"):
        model.attach_lm(_tiny_lm())


def _features():
    return featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B", copies=2)])


def test_forward_end_to_end_is_deterministic_under_a_generator() -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg()).eval()
    model.attach_lm(_tiny_lm())
    f = _features()
    with torch.no_grad():
        out = model(f, num_samples=2, generator=torch.Generator().manual_seed(3), return_intermediates=True)
        again = model(f, num_samples=2, generator=torch.Generator().manual_seed(3))
    L, A = f.num_tokens, f.num_atoms
    assert out.coords.shape == (2, A, 3) and out.distogram_logits.shape == (1, L, L, 8)
    assert out.pair.shape == (1, L, L, 32) and out.s_inputs.shape == (1, L, model.config.single_inputs_width)
    assert out.confidence.ptm.shape == (2,) and out.confidence.pair_chains_iptm.shape == (2, 3, 3)
    assert torch.isfinite(out.coords).all() and torch.isfinite(out.confidence.pae).all()
    assert out.intermediates is not None and len(out.intermediates["states"]) == 2
    assert set(out.intermediates) >= {"s_inputs", "z_init", "relpos", "bonds", "lm_z", "z0", "states"}
    torch.testing.assert_close(out.coords, again.coords)
    torch.testing.assert_close(out.distogram_logits, again.distogram_logits)


def test_forward_accepts_precomputed_lm_states_and_z0() -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg()).eval()
    f = _features()
    hs = torch.randn(1, f.num_tokens, 3, 32)
    z0 = torch.zeros(1, f.num_tokens, f.num_tokens, 32)
    with torch.no_grad():
        out = model(f, lm_hidden_states=hs, z0=z0, num_loops=1, generator=torch.Generator().manual_seed(0))
    assert out.coords.shape == (1, f.num_atoms, 3)
    with pytest.raises(RuntimeError, match="language model"):
        model(f)


def test_save_load_round_trip_and_auto_classes(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg()).eval()
    model.save_pretrained(tmp_path)
    assert (tmp_path / "config.json").read_text().find('"model_type": "oplm_fold"') >= 0
    cfg = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(cfg, FoldConfig) and cfg.pair_width == 32
    auto = AutoModel.from_pretrained(tmp_path)
    assert type(auto) is OplmForFolding
    reloaded, info = OplmForFolding.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert reloaded.lm is None
    with_lm = OplmForFolding.from_pretrained(tmp_path, lm=_tiny_lm())
    assert with_lm.lm is not None


def test_from_pretrained_resolves_the_lm_from_config(tmp_path: Path) -> None:
    lm = _tiny_lm()
    lm.save_pretrained(tmp_path / "lm")
    model = OplmForFolding(_tiny_cfg(lm_name_or_path=str(tmp_path / "lm")))
    model.save_pretrained(tmp_path / "fold")
    reloaded = OplmForFolding.from_pretrained(tmp_path / "fold")
    assert reloaded.lm is not None and lm_state_count(reloaded.lm) == 3
    override = OplmForFolding.from_pretrained(tmp_path / "fold", lm_name_or_path=str(tmp_path / "lm"))
    assert override.lm is not None
    bad = OplmForFolding(_tiny_cfg(lm_name_or_path=str(tmp_path / "lm"), lm_hidden_size=16))
    bad.save_pretrained(tmp_path / "bad")
    with pytest.raises(ValueError, match="lm_hidden_size"):
        OplmForFolding.from_pretrained(tmp_path / "bad")


def test_gradient_checkpointing_reaches_every_pair_stack() -> None:
    model = OplmForFolding(_tiny_cfg())
    model.gradient_checkpointing_enable()
    assert model.folding_trunk.gradient_checkpointing and model.confidence_head.folding_trunk.gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.lm_encoder.gradient_checkpointing
```

Run: `.venv/bin/python -m pytest tests/fold/test_modeling.py -q`
Expected: FAIL with `ImportError: cannot import name 'OplmForFolding' from 'oplm.fold'`.

- [ ] **Step 3: `src/oplm/fold/modeling_fold.py`**

```python
"""OplmForFolding — the HF model that owns the heads and a frozen, unregistered language model.

Composition and data flow follow Biohub's ESMFold2 ``EsmFold2Model.forward``
(esm/models/esmfold2/model.py, Apache-2.0; see THIRD_PARTY_NOTICES.md); parameter names are
the released checkpoint's (``input_embedder``, ``language_model``, ``lm_encoder``,
``folding_trunk``, ``parcae``, ``distogram_head``, ``structure_head``, ``confidence_head``).
Modifications: the LM is any ``OplmModel`` (or precomputed hidden states) held *outside* the
module tree (spec §6.6); loop count, gradient loops, initial state and the sampler RNG are
explicit; per-loop LM-pair dropout is training-only; the generic HF init respects the port's
zero/identity/−2 initialisations through module tags (spec §5.4).

Loading a fold checkpoint requires ``oplm`` to be installed (``import oplm.fold`` registers
``oplm_fold`` with the Auto classes); ``trust_remote_code`` bundling is not provided in this
milestone.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel

from oplm.fold.atoms import InputsEmbedder
from oplm.fold.attention import ensure_flex_recompile_limit, resolve_attention_backend, sliding_window_block_mask
from oplm.fold.confidence import ConfidenceHead, ConfidenceOutput, symmetrized_distogram
from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.diffusion import StructureHead
from oplm.fold.lm_shim import LanguageModelShim, frozen_lm_hidden_states, lm_state_count
from oplm.fold.trunk import PairStack, Recurrence, pair_stack_kwargs
from oplm.model import OplmModel

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from oplm.fold.data.featurize import FoldFeatures

__all__ = ["FoldOutput", "OplmFoldPreTrainedModel", "OplmForFolding", "load_frozen_lm"]


class OplmFoldPreTrainedModel(PreTrainedModel):
    """HF plumbing for fold models: config class, init policy, checkpointing hooks."""

    config_class = FoldConfig
    base_model_prefix = "fold"
    main_input_name = "lm_input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PairUpdateBlock", "DiffusionBlock", "AtomBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        """Generic init plus the port's tagged exceptions; never touches loaded tensors."""
        if isinstance(module, nn.Linear):
            if getattr(module, "_init_zero", False):
                nn.init.zeros_(module.weight)
            elif not getattr(module, "_init_identity", False):  # identity is applied by its owner
                nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.constant_(module.bias, float(getattr(module, "_init_bias", 0.0)))
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, Recurrence):
            module.reset_parameters()
        elif isinstance(module, LanguageModelShim):
            nn.init.zeros_(module.layer_weights)
        elif isinstance(module, ConfidenceHead):
            nn.init.zeros_(module.plddt_weight)
            nn.init.zeros_(module.resolved_weight)


@dataclass
class FoldOutput:
    """What one forward pass produces (``coords``/``confidence`` have ``B · num_samples`` rows)."""

    distogram_logits: Tensor  # (B, L, L, bins) fp32, symmetric
    coords: Tensor  # (B·S, A, 3) Å, padded atoms meaningless
    confidence: ConfidenceOutput
    s_inputs: Tensor  # (B, L, single_inputs_width)
    pair: Tensor  # (B, L, L, pair) fp32 final pair (after readout + coda)
    intermediates: dict[str, Any] | None = None


def load_frozen_lm(
    name_or_path: str | Path,
    *,
    revision: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> OplmModel:
    """Load the bare OPLM encoder (local dir, ``<ckpt>/hf``, or Hub id) frozen in eval mode.

    Raises:
        ValueError: ``name_or_path`` has the ``<repo>#esmc`` form written by
            ``fold_config_from_upstream``: the head was trained against the ESMC bundled in
            that repo, which is not an OPLM; pass ``lm_hidden_states`` to ``forward`` instead.
    """
    if "#" in str(name_or_path):
        raise ValueError(
            f"{name_or_path!r} names a language model bundled in an upstream checkpoint, not an "
            "OPLM; run the head with precomputed lm_hidden_states (see docs/FOLD.md §7)"
        )
    lm = OplmModel.from_pretrained(str(name_or_path), revision=revision)
    if dtype is not None:
        lm = lm.to(dtype)
    if device is not None:
        lm = lm.to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm


class OplmForFolding(OplmFoldPreTrainedModel):
    """Frozen LM -> shim -> recurrent pair trunk -> distogram, diffusion and confidence heads."""

    def __init__(self, config: FoldConfig) -> None:
        super().__init__(config)
        kw = pair_stack_kwargs(config)
        self.input_embedder = InputsEmbedder(config)
        self.language_model = LanguageModelShim(config)
        self.lm_encoder = PairStack(config.lm_encoder_blocks, config.pair_width, **kw)
        self.folding_trunk = PairStack(config.trunk_blocks, config.pair_width, **kw)
        self.parcae = Recurrence(config.pair_width, coda_blocks=config.coda_blocks, **kw)
        self.distogram_head = nn.Linear(config.pair_width, config.distogram_bins, bias=True)
        self.structure_head = StructureHead(config)
        self.confidence_head = ConfidenceHead(config)
        object.__setattr__(self, "_lm", None)  # bypass nn.Module registration (spec §6.6)
        self.post_init()

    # --- frozen LM ownership -------------------------------------------------------------

    @property
    def lm(self) -> nn.Module | None:
        """The attached frozen language model, or ``None``."""
        return self._lm

    def attach_lm(
        self, lm: nn.Module, *, name_or_path: str | None = None, revision: str | None = None
    ) -> None:
        """Attach a frozen LM after checking it against the config's shape contract.

        Raises:
            ValueError: ``lm.config.hidden_size`` or its hidden-state count disagrees with
                ``lm_hidden_size`` / ``lm_num_hidden_states``.
        """
        hidden = int(lm.config.hidden_size)  # ty: ignore[unresolved-attribute]  # PreTrainedModel
        if hidden != self.config.lm_hidden_size:
            raise ValueError(
                f"lm_hidden_size mismatch: config expects {self.config.lm_hidden_size}, LM has {hidden}"
            )
        states = lm_state_count(lm)
        if states != self.config.lm_num_hidden_states:
            raise ValueError(
                f"lm_num_hidden_states mismatch: config expects {self.config.lm_num_hidden_states}, "
                f"LM produces {states}"
            )
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "_lm", lm)
        if name_or_path is not None:
            self.config.lm_name_or_path = name_or_path
        if revision is not None:
            self.config.lm_revision = revision

    def train(self, mode: bool = True) -> OplmForFolding:  # ty: ignore[invalid-method-override]  # narrower return
        super().train(mode)
        if self._lm is not None:
            self._lm.eval()
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> OplmForFolding:
        """Deep-copy the head; share the frozen LM (``build_ema`` must not duplicate it)."""
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            object.__setattr__(clone, name, value if name == "_lm" else copy.deepcopy(value, memo))
        return clone

    @classmethod
    def from_pretrained(  # ty: ignore[invalid-method-override]  # adds keyword-only LM options
        cls,
        pretrained_model_name_or_path: str | Path | None,
        *args: Any,
        lm: nn.Module | None = None,
        lm_name_or_path: str | Path | None = None,
        lm_revision: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Load the head, then attach ``lm`` or resolve the LM from the config (override wins).

        No LM is attached when neither is given and ``config.lm_name_or_path`` is unset; call
        :meth:`attach_lm` or pass ``lm_hidden_states`` to ``forward``.
        """
        result = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        source = lm_name_or_path or model.config.lm_name_or_path
        if lm is not None:
            model.attach_lm(lm)
        elif source is not None:
            revision = lm_revision or (model.config.lm_revision if lm_name_or_path is None else None)
            model.attach_lm(load_frozen_lm(source, revision=revision, device=model.device))
        return result

    # --- forward -------------------------------------------------------------------------

    def forward(
        self,
        features: FoldFeatures,
        *,
        lm_hidden_states: Tensor | None = None,
        num_loops: int | None = None,
        num_samples: int | None = None,
        num_steps: int | None = None,
        z0: Tensor | None = None,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
    ) -> FoldOutput:
        """Predict a structure for ``features`` (batch size 1).

        Args:
            features: Output of :func:`oplm.fold.featurize`.
            lm_hidden_states: ``(1, L, K, D)`` precomputed LM states (skips the attached LM).
            num_loops: Recurrence iterations; default ``config.inference_num_loops``.
            num_samples: Diffusion samples; default ``config.inference_num_samples``.
            num_steps: Sampler steps before the sigma cap; default ``config.inference_num_steps``.
            z0: Initial pair state; drawn from ``generator`` when ``None``.
            generator: RNG for the initial state and the sampler (deterministic when set).
            return_intermediates: Also return the stage tensors used by the parity tests.

        Raises:
            RuntimeError: No LM is attached and ``lm_hidden_states`` is ``None``.
        """
        cfg = self.config
        f = features.to(self.device)
        if lm_hidden_states is None:
            if self._lm is None:
                raise RuntimeError(
                    "no language model attached: call attach_lm(), load with lm_name_or_path, "
                    "or pass lm_hidden_states"
                )
            lm_device = next(self._lm.parameters()).device
            lm_dtype = torch.bfloat16 if lm_device.type == "cuda" else None
            lm_hidden_states = frozen_lm_hidden_states(self._lm, f.to(lm_device), dtype=lm_dtype)
        lm_hidden_states = lm_hidden_states.to(self.device)
        loops = cfg.inference_num_loops if num_loops is None else num_loops
        samples = cfg.inference_num_samples if num_samples is None else num_samples
        on_cuda = self.device.type == "cuda"
        if on_cuda:
            ensure_flex_recompile_limit()
        flex = on_cuda and resolve_attention_backend(f.ref_pos, cfg.attention_backend) == "flex"  # ty: ignore[invalid-argument-type]  # validated by FoldConfig
        atom_block_mask = sliding_window_block_mask(f.atom_mask, cfg.atom_window // 2) if flex else None
        pair_mask = f.token_mask[:, :, None].float() * f.token_mask[:, None, :].float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=on_cuda):
            emb = self.input_embedder(f, block_mask=atom_block_mask)
            lm_z = self.language_model(lm_hidden_states)
            if z0 is None:
                z0 = self.parcae.init_state(emb.z_init, generator=generator)
            drop = cfg.pair_dropout if self.training else 0.0

            def inject(_t: int) -> Tensor:
                lm_t = F.dropout(lm_z, drop, training=True) if drop > 0 else lm_z
                return emb.z_init + self.lm_encoder(lm_t.to(emb.z_init.dtype), pair_mask)

            z, states = self.parcae.run(
                self.folding_trunk, inject, z0=z0.to(emb.z_init.dtype), pair_mask=pair_mask,
                num_loops=loops, grad_loops=cfg.recurrence_grad_loops if self.training else None,
                return_states=return_intermediates,
            )
            z = self.parcae.readout(z, pair_mask)
        z = z.float()
        distogram_logits = symmetrized_distogram(self.distogram_head, z)
        inp = self.structure_head.prepare(
            s_inputs=emb.s_inputs, z_trunk=z, relpos=emb.relpos, atom_features=emb.atom_features,
            rope=emb.rope, atom_mask=f.atom_mask, atom_to_token=f.atom_to_token,
            token_mask=f.token_mask, num_samples=samples,
        )
        coords = self.structure_head.sample(inp, num_steps=num_steps, generator=generator)
        confidence = self.confidence_head(
            s_inputs=emb.s_inputs, z=z, relpos=emb.relpos, bonds=emb.bonds, coords=coords,
            distogram_atom_idx=f.distogram_atom_idx, token_mask=f.token_mask,
            atom_to_token=f.atom_to_token, atom_mask=f.atom_mask, asym_id=f.asym_id,
        )
        intermediates = None
        if return_intermediates:
            intermediates = {
                "s_inputs": emb.s_inputs, "z_init": emb.z_init, "relpos": emb.relpos, "bonds": emb.bonds,
                "lm_z": lm_z, "z0": z0, "states": states, "pair": z, "conditioned_pair": inp.z,
            }
        return FoldOutput(
            distogram_logits=distogram_logits, coords=coords, confidence=confidence,
            s_inputs=emb.s_inputs, pair=z, intermediates=intermediates,
        )


AutoConfig.register("oplm_fold", FoldConfig, exist_ok=True)
AutoModel.register(FoldConfig, OplmForFolding, exist_ok=True)
```

`OplmForFolding.train`'s `# ty: ignore[invalid-method-override]` is only needed if ty rejects the narrower return type; drop it otherwise.

- [ ] **Step 4: `src/oplm/fold/__init__.py`** (replace the docstring-only file; lazy so `oplm fold --help` never imports torch)

```python
"""Structure prediction head: kernels (milestone 0) and the ESMFold2 inference port (milestone 1).

Importing the model classes registers ``oplm_fold`` with ``AutoConfig``/``AutoModel``.
Exports are lazy so the CLI can import ``oplm.fold.cli`` without loading torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oplm.fold.configuration_fold import FoldConfig
    from oplm.fold.data.featurize import ChainSpec, FoldFeatures, featurize
    from oplm.fold.modeling_fold import FoldOutput, OplmForFolding

__all__ = ["ChainSpec", "FoldConfig", "FoldFeatures", "FoldOutput", "OplmForFolding", "featurize"]

_LAZY = {
    "FoldConfig": ("oplm.fold.configuration_fold", "FoldConfig"),
    "ChainSpec": ("oplm.fold.data.featurize", "ChainSpec"),
    "FoldFeatures": ("oplm.fold.data.featurize", "FoldFeatures"),
    "featurize": ("oplm.fold.data.featurize", "featurize"),
    "FoldOutput": ("oplm.fold.modeling_fold", "FoldOutput"),
    "OplmForFolding": ("oplm.fold.modeling_fold", "OplmForFolding"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'oplm.fold' has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
```

(`AutoModel.from_pretrained` on a fold directory works once `oplm.fold.modeling_fold` has been imported, e.g. via `from oplm.fold import OplmForFolding`; the test does exactly that.)

- [ ] **Step 5: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold -q -m "not slow" && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/modeling_fold.py src/oplm/fold/__init__.py tests/fold/test_modeling.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: all fold tests pass (the meta-device construction test takes a few seconds). If transformers reports the `_lm` attribute during `save_pretrained`/`from_pretrained` (it only walks registered modules, so it should not), the `object.__setattr__` path is the fix point, not the config.

```bash
git add src/oplm/fold/modeling_fold.py src/oplm/fold/__init__.py tests/fold/test_modeling.py tests/fold/data/esmfold2_fast_head_keys.txt
git commit -m "feat(fold): OplmForFolding with frozen-LM ownership, tagged init and Auto registration"
```


### Task 10: Golden fixtures: released-config map, head-weight extraction, generator, parity tests, cluster job

**Files:**
- Create: `src/oplm/fold/fixtures.py`, `tests/fold/conftest.py`, `tests/fold/test_fixtures.py`, `tests/fold/test_parity.py`, `tests/fold/data/esmfold2_fast_config.json`, `docs/fold/b200-fixtures.sbatch`
- Modify: `src/oplm/fold/cli.py` (add `make-fixtures`)

**Interfaces:**
- Consumes: Task 9 `OplmForFolding`, Task 3 `featurize`/`ChainSpec`, Task 2 `ReferenceConformers`.
- Produces: `FixtureCase(name, chains, num_loops=2, num_steps=3)`, `FIXTURE_CASES`, `UPSTREAM_REPO = "biohub/ESMFold2-Fast"`, `UPSTREAM_REVISION = "45fe8656f5b3ef493c17fcf9abe9a2968902e712"`, `fold_config_from_upstream(config: dict, **overrides) -> FoldConfig`, `extract_head_weights(snapshot_dir, out) -> Path`, `load_fixture(fixtures_dir, case_name) -> dict[str, Tensor]`, `generate_fixtures(out_dir, *, repo, revision, cases, seed) -> Path` (esm venv only), `upstream_lm_rows(input_ids, bos, eos, pad) -> list[list[int]]`.
- Fixture directory layout (`$OPLM_FOLD_FIXTURES`): `manifest.json`, `config.json` (released, `esmc_config` stripped), `head.safetensors` (1054 head tensors, stored dtype), `reference_conformers.json` (dumped through upstream's `get_idealized_atom_pos`), `<case>.safetensors` with keys listed in the generator.

- [ ] **Step 1: Commit the released config for the offline config-map test**

```bash
.venv/bin/python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
p = hf_hub_download("biohub/ESMFold2-Fast", "config.json", revision="45fe8656f5b3ef493c17fcf9abe9a2968902e712")
cfg = json.loads(Path(p).read_text())
cfg.pop("esmc_config", None)
Path("tests/fold/data/esmfold2_fast_config.json").write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n")
EOF
```

(Offline alternative: the same file is cached at the planning session's scratchpad `hf/fast/config.json`; strip `esmc_config` the same way.)

- [ ] **Step 2: Write the failing tests (CPU, no fixtures needed)**

`tests/fold/test_fixtures.py`:

```python
"""Released-config -> FoldConfig map, head-weight extraction, LM-row splitting, case table."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.fixtures import (
    FIXTURE_CASES,
    extract_head_weights,
    fold_config_from_upstream,
    upstream_lm_rows,
)

_CFG = Path(__file__).parent / "data" / "esmfold2_fast_config.json"


def test_released_config_maps_onto_fold_config_defaults() -> None:
    cfg = fold_config_from_upstream(json.loads(_CFG.read_text()))
    defaults = FoldConfig().to_dict()
    mapped = cfg.to_dict()
    differing = {k for k in defaults if defaults[k] != mapped.get(k)}
    assert differing == {"inference_num_loops", "lm_hidden_size", "lm_num_hidden_states", "lm_name_or_path"}
    assert cfg.inference_num_loops == 21 and cfg.lm_hidden_size == 2560 and cfg.lm_num_hidden_states == 81
    assert cfg.lm_name_or_path == "biohub/ESMFold2-Fast#esmc"
    assert cfg.single_inputs_width == 451 and cfg.atom_feature_dim == 389
    over = fold_config_from_upstream(json.loads(_CFG.read_text()), attention_backend="dense", trunk_blocks=1)
    assert over.attention_backend == "dense" and over.trunk_blocks == 1


def test_extract_head_weights_drops_esmc_and_keeps_shards_order(tmp_path: Path) -> None:
    (tmp_path / "snap").mkdir()
    save_file({"esmc.a": torch.zeros(2), "distogram_head.bias": torch.ones(3)}, tmp_path / "snap" / "model-00001-of-00002.safetensors")
    save_file({"parcae.log_delta": torch.full((4,), 2.0), "esmc.b": torch.zeros(1)}, tmp_path / "snap" / "model-00002-of-00002.safetensors")
    index = {"weight_map": {"esmc.a": "model-00001-of-00002.safetensors", "distogram_head.bias": "model-00001-of-00002.safetensors",
                            "parcae.log_delta": "model-00002-of-00002.safetensors", "esmc.b": "model-00002-of-00002.safetensors"}}
    (tmp_path / "snap" / "model.safetensors.index.json").write_text(json.dumps(index))
    out = extract_head_weights(tmp_path / "snap", tmp_path / "head.safetensors")
    head = load_file(out)
    assert set(head) == {"distogram_head.bias", "parcae.log_delta"}
    assert torch.equal(head["parcae.log_delta"], torch.full((4,), 2.0))


def test_upstream_lm_rows_split_the_packed_sequence() -> None:
    ids = torch.tensor([[0, 20, 15, 2, 0, 6, 6, 2, 1, 1]])
    assert upstream_lm_rows(ids, bos=0, eos=2, pad=1) == [[0, 20, 15, 2], [0, 6, 6, 2]]


def test_fixture_cases_cover_the_spec_shapes() -> None:
    names = [c.name for c in FIXTURE_CASES]
    assert names == ["trp_cage", "villin", "gb1", "heterodimer", "homodimer_x"]
    assert sum(len(ch.sequence) * ch.copies for ch in FIXTURE_CASES[-1].chains) == 16
    assert any("X" in ch.sequence for ch in FIXTURE_CASES[-1].chains)
    assert all(c.num_loops == 2 and c.num_steps == 3 for c in FIXTURE_CASES)
```

Run: `.venv/bin/python -m pytest tests/fold/test_fixtures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.fixtures'`.

- [ ] **Step 3: `src/oplm/fold/fixtures.py`**

```python
"""Golden-fixture tooling for the ESMFold2 parity oracle (spec §9).

Two halves. Pure-torch helpers that run anywhere: the released-config -> :class:`FoldConfig`
map, head-only weight extraction from the HF shards, fixture loading, LM-row splitting. And
the generator, which imports the upstream ``esm`` package lazily and runs only inside the
pinned fixture venv (docs/fold/b200-fixtures.sbatch): it featurizes each case with upstream's
own input builder, runs the upstream model on CPU in fp32 with dropout, masking and kernels
off and the loop count and sampler pinned, records every stage through forward hooks, and
writes one safetensors file per case plus a manifest. Fixture tensors keep upstream's dtype.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import load_file, safe_open, save_file

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.ccd import PROTEIN_RES_TYPES, ReferenceConformers
from oplm.fold.data.featurize import ChainSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

__all__ = [
    "FIXTURE_CASES",
    "UPSTREAM_REPO",
    "UPSTREAM_REVISION",
    "FixtureCase",
    "extract_head_weights",
    "fold_config_from_upstream",
    "generate_fixtures",
    "load_fixture",
    "upstream_lm_rows",
]

UPSTREAM_REPO = "biohub/ESMFold2-Fast"
UPSTREAM_REVISION = "45fe8656f5b3ef493c17fcf9abe9a2968902e712"


@dataclass(frozen=True)
class FixtureCase:
    """One parity example: short enough for a CPU run of the 6B LM and the 24-block trunk."""

    name: str
    chains: tuple[ChainSpec, ...]
    num_loops: int = 2  # upstream runs num_loops + 1 = 3 iterations (spec §9: states after 1/2/3)
    num_steps: int = 3  # sampler steps before the cap; the denoiser oracle is the first call


FIXTURE_CASES: tuple[FixtureCase, ...] = (
    FixtureCase("trp_cage", (ChainSpec("NLYIQWLKDGGPSSGRPPPS", "A"),)),
    FixtureCase("villin", (ChainSpec("LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF", "A"),)),
    FixtureCase("gb1", (ChainSpec("MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "A"),)),
    FixtureCase("heterodimer", (ChainSpec("GSHMKTAYIAKQRQ", "A"), ChainSpec("WKLLSDEDFKAV", "B"))),
    FixtureCase("homodimer_x", (ChainSpec("MKVLAXGG", "A", copies=2),)),
)


def fold_config_from_upstream(config: dict[str, Any], **overrides: Any) -> FoldConfig:
    """Map the released ESMFold2 HF ``config.json`` onto :class:`FoldConfig` fields.

    ``inference_num_loops`` is ``num_loops + 1`` (iterations executed); ``trunk_dropout`` is 0
    (upstream's block dropout is ``DropoutResidual(0.0)``; its ``folding_trunk_dropout`` key is
    never read); ``inference_num_samples`` is 1 (the ``fold()`` default, not the forward
    default of 32). ``lm_name_or_path`` records the bundled ESMC as ``<repo>#esmc`` so a load
    never silently attaches an OPLM to a head trained against ESMC.

    Raises:
        ValueError: The derived ``single_inputs_width``/``atom_feature_dim`` disagree with the
            released ``single_inputs_size``/``atom_feature_dim``.
    """
    ae, sh, ch = config["atom_encoder"], config["structure_head"], config["confidence_head"]
    dm = sh["diffusion_module"]
    pair = int(config["pairwise_hidden_size"])
    fields: dict[str, Any] = dict(
        lm_name_or_path=f"{UPSTREAM_REPO}#esmc",
        lm_hidden_size=config["lm_d_model"],
        lm_num_hidden_states=config["lm_num_layers"] + 1,
        lm_input_mask_fraction=config["lm_mask_pct"],
        num_res_types=config["num_res_types"],
        max_atomic_number=config["max_atomic_number"],
        atom_name_chars=config["max_chars"],
        atom_name_vocab=config["char_vocab_size"],
        max_atoms_per_token=config["max_atoms_per_token"],
        pair_width=pair,
        token_width=ae["token_hidden_size"],
        atom_width=ae["hidden_size"],
        atom_encoder_blocks=ae["num_hidden_layers"],
        atom_encoder_heads=ae["num_attention_heads"],
        atom_window=config["sliding_window"],
        spatial_rope_base=ae["spatial_rope_base_frequency"],
        spatial_rope_pairs_per_axis=ae["num_spatial_rope_pairs_per_axis"],
        uid_rope_pairs=ae["num_uid_rope_pairs"],
        uid_rope_base=ae["uid_rope_base_frequency"],
        relpos_r_max=config["num_relative_residx_bins"],
        relpos_s_max=config["num_relative_chain_bins"],
        trunk_blocks=config["folding_trunk_num_hidden_layers"],
        lm_encoder_blocks=config["lm_encoder"]["num_hidden_layers"],
        coda_blocks=config["parcae_num_coda_layers"],
        transition_expansion=config["pair_transition_intermediate_size"] // pair,
        pair_dropout=config["lm_encoder"]["lm_dropout"],
        trunk_dropout=0.0,
        recurrence_poisson_mean=config["parcae"]["poisson_mean"],
        recurrence_min_loops=config["parcae"]["min_steps"],
        recurrence_max_loops=config["parcae"]["max_steps"],
        inference_num_loops=config["num_loops"] + 1,
        sigma_data=dm["sigma_data"],
        fourier_dim=dm["fourier_dim"],
        diffusion_blocks=dm["token_num_blocks"],
        diffusion_heads=dm["token_num_heads"],
        diffusion_transition_multiplier=dm["transition_multiplier"],
        diffusion_atom_blocks=dm["atom_encoder"]["num_hidden_layers"],
        diffusion_atom_heads=dm["atom_encoder"]["num_attention_heads"],
        inference_num_steps=sh["inference_num_steps"],
        inference_sigma_max=sh["inference_sigma_max_ratio"],
        inference_sigma_min=sh["inference_sigma_min_ratio"],
        inference_rho=sh["inference_exponent"],
        inference_sigma_cap=sh["inference_sigma_cap"],
        gamma_0=sh["gamma_0"],
        gamma_min=sh["gamma_min"],
        noise_scale=sh["noise_scale"],
        step_scale=sh["step_scale"],
        inference_num_samples=1,
        distogram_bins=sh["num_distogram_bins"],
        confidence_blocks=ch["num_hidden_layers"],
        plddt_bins=ch["num_plddt_bins"],
        pae_bins=ch["num_pae_bins"],
        pde_bins=ch["num_pde_bins"],
        confidence_dist_bins=ch["distogram_bins"],
        confidence_min_dist=ch["min_dist"],
        confidence_max_dist=ch["max_dist"],
    )
    fields.update(overrides)
    cfg = FoldConfig(**fields)
    if cfg.single_inputs_width != config["single_inputs_size"]:
        raise ValueError(
            f"single_inputs_width {cfg.single_inputs_width} != released {config['single_inputs_size']}"
        )
    if cfg.atom_feature_dim != config["atom_feature_dim"]:
        raise ValueError(f"atom_feature_dim {cfg.atom_feature_dim} != released {config['atom_feature_dim']}")
    return cfg


def extract_head_weights(snapshot_dir: Path, out: Path) -> Path:
    """Write every non-``esmc.*`` tensor of a sharded HF snapshot to one safetensors file."""
    index = json.loads((Path(snapshot_dir) / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for key, shard in index["weight_map"].items():
        if not key.startswith("esmc."):
            by_shard.setdefault(shard, []).append(key)
    tensors: dict[str, Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(Path(snapshot_dir) / shard), "pt") as f:
            for key in keys:
                tensors[key] = f.get_tensor(key)
    save_file(tensors, str(out), metadata={"source": f"{UPSTREAM_REPO}@{UPSTREAM_REVISION}"})
    return Path(out)


def load_fixture(fixtures_dir: Path, case_name: str) -> dict[str, Tensor]:
    """Load one case's recorded tensors."""
    return load_file(str(Path(fixtures_dir) / f"{case_name}.safetensors"))


def upstream_lm_rows(input_ids: Tensor, *, bos: int, eos: int, pad: int) -> list[list[int]]:
    """Split upstream's packed ``[BOS c1 EOS BOS c2 EOS PAD...]`` LM row into per-chain rows."""
    rows: list[list[int]] = []
    for tok in input_ids[0].tolist():
        if tok == pad:
            break
        if tok == bos:
            rows.append([tok])
        else:
            rows[-1].append(tok)
    return rows


# --- generator (runs only in the esm venv) ---------------------------------------------------


class _Recorder:
    """Forward hooks that stash tensors under ``<name>.in.<k>`` / ``<name>.out.<k>`` per call."""

    def __init__(self) -> None:
        self.tensors: dict[str, Tensor] = {}
        self.calls: dict[str, int] = {}
        self.handles: list[Any] = []

    def remove_all(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _put(self, key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            self.tensors[key] = value.detach().clone().cpu()

    def attach(self, module: torch.nn.Module, name: str, *, inputs: bool = False, kwargs: bool = False) -> None:
        def hook(_m: Any, args: Any, kw: Any = None, output: Any = None) -> None:
            if output is None:  # with_kwargs=False signature: (module, args, output)
                output, kw = kw, {}
            i = self.calls.get(name, 0)
            self.calls[name] = i + 1
            if inputs:
                for j, a in enumerate(args):
                    self._put(f"{name}.in.{i}.arg{j}", a)
                for k, v in (kw or {}).items():
                    self._put(f"{name}.in.{i}.{k}", v)
            if isinstance(output, dict):
                for k, v in output.items():
                    self._put(f"{name}.out.{i}.{k}", v)
            elif isinstance(output, tuple):
                for j, v in enumerate(output):
                    self._put(f"{name}.out.{i}.{j}", v)
            else:
                self._put(f"{name}.out.{i}", output)

        self.handles.append(module.register_forward_hook(hook, with_kwargs=kwargs))


def _dump_reference_conformers(out: Path, base: ReferenceConformers) -> None:
    """Our residue/atom tables with positions from upstream's ``get_idealized_atom_pos``."""
    from esm.models.esmfold2.conformers import get_idealized_atom_pos  # ty: ignore[unresolved-import]  # esm venv only

    residues = {}
    for code in base.codes:
        t = base[code]
        positions = []
        for name in t.atoms:
            pos = get_idealized_atom_pos(PROTEIN_RES_TYPES[code], name)
            positions.append([0.0, 0.0, 0.0] if pos is None else [float(v) for v in pos])
        residues[code] = {
            "atoms": list(t.atoms), "elements": list(t.elements), "charges": list(t.charges),
            "positions": positions,
        }
    out.write_text(json.dumps({
        "source": f"esm {version('esm')} get_idealized_atom_pos over ccd.pkl of {UPSTREAM_REPO}@{UPSTREAM_REVISION} (Computed conformer, raw)",
        "charge_table": base.source, "residues": residues,
    }, indent=1) + "\n")


def generate_fixtures(
    out_dir: Path,
    *,
    repo: str = UPSTREAM_REPO,
    revision: str = UPSTREAM_REVISION,
    cases: Sequence[FixtureCase] = FIXTURE_CASES,
    seed: int = 0,
) -> Path:
    """Run upstream ESMFold2 on CPU in fp32 for each case and record the parity oracle.

    Requires the ``esm`` package (``esm==3.4.1.post1``) and network/cache access to ``repo``.
    Determinism: ``set_kernel_backend(None)``, ``set_chunk_size(None)``, LM dropout off
    (``_lm_dropout_context(model, None)``), ``lm_mask_pct=0``, ``torch.manual_seed(seed)``
    before every forward; the initial pair state is captured by wrapping ``_init_pair_state``.
    """
    from esm.models.esmfold2 import (  # ty: ignore[unresolved-import]  # esm venv only
        ESMFold2InputBuilder,
        EsmFold2Model,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.models.esmfold2.processor import _lm_dropout_context  # ty: ignore[unresolved-import]  # esm venv only
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.pkl"]))
    config = json.loads((snapshot / "config.json").read_text())
    config.pop("esmc_config", None)
    (out_dir / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    extract_head_weights(snapshot, out_dir / "head.safetensors")

    torch.set_default_dtype(torch.float32)
    # If this signature rejects `revision`, pass the `snapshot` path instead (same files).
    model = EsmFold2Model.from_pretrained(repo, revision=revision, device="cpu").eval().float()
    model.set_kernel_backend(None)
    model.set_chunk_size(None)
    builder = ESMFold2InputBuilder()
    _dump_reference_conformers(out_dir / "reference_conformers.json", ReferenceConformers.load())

    manifest: dict[str, Any] = {
        "repo": repo, "revision": revision, "seed": seed, "python": sys.version, "platform": platform.platform(),
        "esm": version("esm"), "torch": torch.__version__, "transformers": version("transformers"),
        "cases": {},
    }
    for case in cases:
        rec = _Recorder()
        rec.attach(model.inputs_embedder, "inputs_embedder")
        for name in ("z_init_1", "z_init_2", "rel_pos", "token_bonds", "language_model", "lm_encoder",
                     "folding_trunk", "parcae_input_norm", "parcae_readout", "parcae_coda", "distogram_head"):
            rec.attach(getattr(model, name), name, inputs=(name == "language_model"))
        rec.attach(model.structure_head.diffusion_module, "diffusion", inputs=True, kwargs=True)
        rec.attach(model.confidence_head.folding_trunk, "confidence_trunk", inputs=True)
        original_init = model._init_pair_state

        def init_pair_state(ref: Tensor) -> Tensor:
            z0 = original_init(ref)
            rec.tensors["z0"] = z0.detach().clone().cpu()
            return z0

        model._init_pair_state = init_pair_state  # ty: ignore[invalid-assignment]  # instance override
        spi = StructurePredictionInput(sequences=[
            ProteinInput(id=f"{spec.chain_id}{'' if k == 0 else k + 1}", sequence=spec.sequence)
            for spec in case.chains for k in range(spec.copies)
        ])
        features, _chain_infos = builder.prepare_input(spi, seed=seed, device="cpu")
        torch.manual_seed(seed)
        with torch.no_grad(), _lm_dropout_context(model, None):
            output = model(
                **features, num_loops=case.num_loops, num_diffusion_samples=1,
                num_sampling_steps=case.num_steps, lm_mask_pct=0.0,
            )
        model._init_pair_state = original_init  # ty: ignore[invalid-assignment]  # restore
        tensors = {f"features.{k}": v.cpu() for k, v in features.items() if isinstance(v, torch.Tensor)}
        tensors.update({f"output.{k}": v.detach().cpu() for k, v in output.items() if isinstance(v, torch.Tensor)})
        tensors.update(rec.tensors)
        tensors["lm_hidden_states"] = rec.tensors["language_model.in.0.arg0"].clone()  # no shared storage
        # Hooks ran under upstream's @inference_mode; clone outside it so safetensors gets plain tensors.
        save_file(
            {k: v.detach().clone().contiguous() for k, v in tensors.items()},
            str(out_dir / f"{case.name}.safetensors"),
        )
        manifest["cases"][case.name] = {
            "chains": [[c.sequence, c.chain_id, c.copies] for c in case.chains],
            "num_loops": case.num_loops, "num_steps": case.num_steps, "keys": sorted(tensors),
            "denoiser_calls": rec.calls.get("diffusion", 0),
        }
        rec.remove_all()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return out_dir
```

(The recorder keys the parity tests read: `inputs_embedder.out.0` (x_inputs), `z_init_1.out.0`, `z_init_2.out.0`, `rel_pos.out.0`, `token_bonds.out.0`, `lm_hidden_states`, `language_model.out.0` (lm_z), `z0`, `lm_encoder.out.{0,1,2}`, `folding_trunk.out.{0,1,2}`, `parcae_readout.out.0`, `parcae_coda.out.0`, `distogram_head.out.0`, `diffusion.in.0.x_noisy`, `diffusion.in.0.t_hat`, `diffusion.out.0.x_denoised`, `confidence_trunk.in.0.arg0`, `confidence_trunk.out.0`, `output.sample_atom_coords`, `output.{plddt_logits,pae_logits,pde_logits,plddt,pae,pde,ptm,iptm,pair_chains_iptm,resolved_logits}`, `features.*`. The denoiser's conditioned pair is not a module output; the parity test rebuilds it and checks it indirectly through `x_denoised`.)

- [ ] **Step 4: `make-fixtures` command in `src/oplm/fold/cli.py`** (append; torch and esm stay inside the function)

```python
@app.command("make-fixtures")
def make_fixtures(
    out: Annotated[Path, typer.Option("--out", help="Fixture directory to create")],
    repo: Annotated[str, typer.Option(help="Upstream HF repo")] = "biohub/ESMFold2-Fast",
    revision: Annotated[str, typer.Option(help="Upstream HF revision (sha)")] = "45fe8656f5b3ef493c17fcf9abe9a2968902e712",
    cases: Annotated[str, typer.Option(help="Comma-separated case names, or 'all'")] = "all",
    seed: Annotated[int, typer.Option(help="torch.manual_seed before every upstream forward")] = 0,
) -> None:
    """Record ESMFold2 golden fixtures (requires the pinned `esm` venv; CPU, fp32)."""
    from oplm.fold.fixtures import FIXTURE_CASES, generate_fixtures

    chosen = FIXTURE_CASES if cases == "all" else tuple(c for c in FIXTURE_CASES if c.name in cases.split(","))
    if not chosen:
        raise typer.BadParameter(f"no fixture case matches {cases!r}")
    path = generate_fixtures(out, repo=repo, revision=revision, cases=chosen, seed=seed)
    console.print(f"[green]fixtures written to {path}[/green]")
```

(`from typing import Annotated` and `from pathlib import Path` are already imported by the M0 CLI; add them if not.)

- [ ] **Step 5: `tests/fold/conftest.py`**

```python
"""Fixture-directory gate for the parity tests (spec §9: skip with an explicit reason)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    env = os.environ.get("OPLM_FOLD_FIXTURES")
    if not env:
        pytest.skip("OPLM_FOLD_FIXTURES is unset: ESMFold2 golden fixtures unavailable (docs/FOLD.md §7)")
    path = Path(env)
    if not (path / "manifest.json").exists():
        pytest.skip(f"no manifest.json under OPLM_FOLD_FIXTURES={path}")
    return path
```

- [ ] **Step 6: `tests/fold/test_parity.py`**

```python
"""Stage-by-stage parity against the recorded upstream ESMFold2-Fast run (spec §9, §10 M1).

CPU, fp32, dense attention, reference trimul: the same arithmetic upstream used on its CPU
run. Tolerances are initial; Task 12 records the observed maxima in docs/FOLD.md §7 and
tightens them to <= 10x observed.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import load_file

from oplm.fold.configuration_fold import FoldConfig
from oplm.fold.data.ccd import ReferenceConformers
from oplm.fold.data.featurize import featurize
from oplm.fold.fixtures import FIXTURE_CASES, fold_config_from_upstream, load_fixture, upstream_lm_rows
from oplm.fold.modeling_fold import OplmForFolding

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

_CASES = [pytest.param(c, id=c.name) for c in FIXTURE_CASES]


@pytest.fixture(scope="module")
def released(fixtures_dir: Path) -> OplmForFolding:
    cfg = fold_config_from_upstream(
        json.loads((fixtures_dir / "config.json").read_text()),
        attention_backend="dense", trimul_backend="reference", trimul_chunk_size=None,
    )
    model = OplmForFolding(cfg).eval()
    missing, unexpected = model.load_state_dict(load_file(str(fixtures_dir / "head.safetensors")), strict=True)
    assert not missing and not unexpected
    return model.float()


@pytest.fixture(scope="module")
def conformers(fixtures_dir: Path) -> ReferenceConformers:
    return ReferenceConformers.load(fixtures_dir / "reference_conformers.json")


def _close(a: Tensor, b: Tensor, *, atol: float, rtol: float = 0.0, what: str = "") -> None:
    a, b = a.float(), b.float()
    err = (a - b).abs().max().item()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=f"{what}: max abs err {err:.3e} (atol {atol})")


def test_head_weights_load_strictly_and_config_matches(released: OplmForFolding) -> None:
    assert released.config.trunk_blocks == 24 and released.config.distogram_bins == 64
    assert len(released.state_dict()) == 1054


@pytest.mark.parametrize("case", _CASES)
def test_featurizer_matches_upstream(fixtures_dir: Path, conformers: ReferenceConformers, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers)
    for ours, theirs in [
        ("token_index", "token_index"), ("residue_index", "residue_index"), ("asym_id", "asym_id"),
        ("sym_id", "sym_id"), ("entity_id", "entity_id"), ("mol_type", "mol_type"), ("res_type", "res_type"),
        ("token_mask", "token_attention_mask"), ("ref_element", "ref_element"), ("ref_space_uid", "ref_space_uid"),
        ("atom_mask", "atom_attention_mask"), ("atom_to_token", "atom_to_token"),
        ("distogram_atom_idx", "distogram_atom_idx"), ("ref_atom_name_chars", "ref_atom_name_chars"),
    ]:
        assert torch.equal(getattr(f, ours).long(), fx[f"features.{theirs}"].long()), ours
    assert torch.equal(f.token_bonds.bool(), fx["features.token_bonds"].bool())
    _close(f.ref_charge, fx["features.ref_charge"], atol=0, what="ref_charge")
    _close(f.ref_pos, fx["features.ref_pos"], atol=1e-6, what="ref_pos")
    rows = upstream_lm_rows(fx["features.input_ids"], bos=0, eos=2, pad=1)
    ours_rows = [[t for t in row if t != 1] for row in f.lm_input_ids.tolist()]
    assert ours_rows == rows


@pytest.mark.parametrize("case", _CASES)
def test_shim_matches(fixtures_dir: Path, released: OplmForFolding, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    with torch.no_grad():
        lm_z = released.language_model(fx["lm_hidden_states"].float())
    _close(lm_z, fx["language_model.out.0"], atol=1e-4, rtol=1e-4, what="lm_z")


@pytest.mark.parametrize("case", _CASES)
def test_inputs_embedder_matches(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        emb = released.input_embedder(f)
    _close(emb.s_inputs, fx["inputs_embedder.out.0"], atol=1e-4, rtol=1e-4, what="x_inputs")
    _close(emb.relpos, fx["rel_pos.out.0"], atol=1e-5, what="relpos")
    _close(emb.bonds, fx["token_bonds.out.0"], atol=1e-5, what="bonds")
    z_init = fx["z_init_1.out.0"][:, :, None] + fx["z_init_2.out.0"][:, None] + fx["rel_pos.out.0"] + fx["token_bonds.out.0"]
    _close(emb.z_init, z_init, atol=1e-4, rtol=1e-4, what="z_init")


@pytest.mark.parametrize("case", _CASES)
def test_recurrence_readout_and_distogram_match(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers)
    pair_mask = f.token_mask[:, :, None].float() * f.token_mask[:, None, :].float()
    z_init, lm_z, z0 = fx["z_init_1.out.0"][:, :, None] + fx["z_init_2.out.0"][:, None] + fx["rel_pos.out.0"] + fx["token_bonds.out.0"], fx["language_model.out.0"].float(), fx["z0"].float()
    with torch.no_grad():
        refined = released.lm_encoder(lm_z, pair_mask)
        _close(refined, fx["lm_encoder.out.0"], atol=1e-3, rtol=1e-3, what="lm_encoder")
        z, states = released.parcae.run(
            released.folding_trunk, lambda _t: z_init.float() + refined, z0=z0, pair_mask=pair_mask,
            num_loops=case.num_loops + 1, return_states=True,
        )
        for i, state in enumerate(states):
            _close(state, fx[f"folding_trunk.out.{i}"], atol=2e-3, rtol=2e-3, what=f"state {i + 1}")
        readout = released.parcae.readout(z, pair_mask)
        _close(released.parcae.out_proj(z), fx["parcae_readout.out.0"], atol=2e-3, rtol=2e-3, what="readout")
        _close(readout, fx["parcae_coda.out.0"], atol=3e-3, rtol=3e-3, what="coda")
        logits = released.distogram_head(readout + readout.transpose(1, 2))
    _close(logits, fx["distogram_head.out.0"], atol=5e-3, rtol=3e-3, what="distogram")


@pytest.mark.parametrize("case", _CASES)
def test_denoiser_matches(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        emb = released.input_embedder(f)
        inp = released.structure_head.prepare(
            s_inputs=fx["inputs_embedder.out.0"].float(), z_trunk=fx["parcae_coda.out.0"].float(),
            relpos=fx["rel_pos.out.0"].float(), atom_features=emb.atom_features, rope=emb.rope,
            atom_mask=f.atom_mask, atom_to_token=f.atom_to_token, token_mask=f.token_mask, num_samples=1,
        )
        x_denoised = released.structure_head.denoise(fx["diffusion.in.0.x_noisy"].float(), fx["diffusion.in.0.t_hat"].float(), inp)
    _close(x_denoised, fx["diffusion.out.0.x_denoised"], atol=5e-2, what="x_denoised (Å)")


@pytest.mark.parametrize("case", _CASES)
def test_sampler_schedule_matches_recorded_t_hat(fixtures_dir: Path, released: OplmForFolding, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    calls = json.loads((fixtures_dir / "manifest.json").read_text())["cases"][case.name]["denoiser_calls"]
    sched = released.structure_head.noise_schedule(case.num_steps, torch.device("cpu"))
    cap = released.config.inference_sigma_cap
    sched = torch.nn.functional.pad(sched[sched <= cap], (1, 0), value=cap).tolist()
    gammas = [released.config.gamma_0 if s > released.config.gamma_min else 0.0 for s in sched]
    t_hats = [s * (1 + g) for s, g in zip(sched[:-1], gammas[1:], strict=True)]
    assert len(t_hats) == calls
    for i, t in enumerate(t_hats):
        assert abs(fx[f"diffusion.in.{i}.t_hat"][0].item() - t) < 1e-3 * max(t, 1.0), i


@pytest.mark.parametrize("case", _CASES)
def test_confidence_matches(fixtures_dir: Path, released: OplmForFolding, conformers, case) -> None:
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers)
    with torch.no_grad():
        out = released.confidence_head(
            s_inputs=fx["inputs_embedder.out.0"].float(), z=fx["parcae_coda.out.0"].float(),
            relpos=fx["rel_pos.out.0"].float(), bonds=fx["token_bonds.out.0"].float(),
            coords=fx["output.sample_atom_coords"].float(), distogram_atom_idx=f.distogram_atom_idx,
            token_mask=f.token_mask, atom_to_token=f.atom_to_token, atom_mask=f.atom_mask, asym_id=f.asym_id,
        )
    _close(out.pae_logits, fx["output.pae_logits"], atol=3e-3, rtol=3e-3, what="pae_logits")
    _close(out.pde_logits, fx["output.pde_logits"], atol=3e-3, rtol=3e-3, what="pde_logits")
    _close(out.plddt_logits, fx["output.plddt_logits"], atol=3e-3, rtol=3e-3, what="plddt_logits")
    _close(out.plddt, fx["output.plddt"], atol=1e-3, what="plddt")
    _close(out.ptm, fx["output.ptm"], atol=1e-3, what="ptm")
    _close(out.iptm, fx["output.iptm"], atol=1e-3, what="iptm")
    _close(out.pair_chains_iptm, fx["output.pair_chains_iptm"], atol=1e-3, what="pair_chains_iptm")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", _CASES[:1])
def test_gpu_pipeline_tracks_the_cpu_oracle(fixtures_dir: Path, conformers, case) -> None:
    """bf16 autocast + FlexAttention + fused trimul: finite end to end, pair within bf16 drift."""
    cfg = fold_config_from_upstream(json.loads((fixtures_dir / "config.json").read_text()))
    model = OplmForFolding(cfg).eval().cuda()
    model.load_state_dict(load_file(str(fixtures_dir / "head.safetensors")), strict=True)
    fx = load_fixture(fixtures_dir, case.name)
    f = featurize(case.chains, conformers=conformers, pad_tokens_to=128)
    hs = torch.zeros(1, 128, cfg.lm_num_hidden_states, cfg.lm_hidden_size)
    L = fx["lm_hidden_states"].shape[1]
    hs[:, :L] = fx["lm_hidden_states"].float()
    z0 = torch.zeros(1, 128, 128, cfg.pair_width)
    z0[:, :L, :L] = fx["z0"].float()
    with torch.no_grad():
        out = model(f, lm_hidden_states=hs.cuda(), z0=z0.cuda(), num_loops=case.num_loops + 1,
                    num_steps=case.num_steps, generator=torch.Generator(device="cuda").manual_seed(0))
    assert torch.isfinite(out.coords).all() and torch.isfinite(out.confidence.pae).all()
    ref = fx["parcae_coda.out.0"].float().cuda()
    ours = out.pair[:, :L, :L]
    rel = ((ours - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, f"relative Frobenius error of the final pair: {rel:.3e}"
```

- [ ] **Step 7: `docs/fold/b200-fixtures.sbatch`**

```bash
#!/bin/bash
# Milestone-1 golden-fixture generation + CPU parity run (docs/superpowers/plans/
# 2026-10-09-fold-m1-inference-port.md, Tasks 10 and 12). Builds a pinned `esm==3.4.1.post1`
# venv (Python 3.12, transformers 4.57) on /mnt/data, installs this repo into it, records
# the upstream ESMFold2-Fast oracle on CPU in fp32, then runs tests/fold/test_parity.py
# against it in the same venv. CPU-only: the GPU is requested only because the partition
# schedules by GPU; drop --gres if a CPU partition exists. One-shot job, no requeue.
#
# Submit from a login node:
#   sbatch docs/fold/b200-fixtures.sbatch
# Overrides (environment variables, all optional):
#   OPLM_REPO=/mnt/home/<you>/git/oplm       mounted checkout instead of cloning
#   OPLM_GIT_REF=feat/fold-m1                branch/tag/sha to clone when OPLM_REPO is unset
#   OPLM_ENV_FILE=/mnt/home/<you>/.env       credentials/JOB_WORK_DIR script to source
#   OPLM_CONTAINER=/mnt/data/containers/deeplearning_v2026-05-26.sqsh
#   OPLM_FIXTURES=/mnt/data/<you>/fold-fixtures/<jobid>   where the fixtures land (persistent)
#   OPLM_ESM_VENV=/mnt/data/<you>/fold-fixtures/esm-venv  reused across jobs
#   OPLM_RESULTS=/mnt/home/<you>/fold-m1-fixtures/<jobid> logs, manifest copy, junit XML
#   OPLM_CASES=all                           comma-separated subset of fixture cases
#   OPLM_STEPS="venv install env make-fixtures parity"
#
#SBATCH --job-name=fold-m1-fixtures
#SBATCH --partition=hpc-mid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/home/briney/logs/%x_%j.out
#SBATCH --error=/mnt/home/briney/logs/%x_%j.err

set -euo pipefail

ENV_FILE="${OPLM_ENV_FILE:-/mnt/home/${SLURM_JOB_USER}/.env}"
CONTAINER="${OPLM_CONTAINER:-/mnt/data/containers/deeplearning_v2026-05-26.sqsh}"
RESULTS="${OPLM_RESULTS:-/mnt/home/${SLURM_JOB_USER}/fold-m1-fixtures/${SLURM_JOB_ID}}"
FIXTURES="${OPLM_FIXTURES:-/mnt/data/${SLURM_JOB_USER}/fold-fixtures/${SLURM_JOB_ID}}"
ESM_VENV="${OPLM_ESM_VENV:-/mnt/data/${SLURM_JOB_USER}/fold-fixtures/esm-venv}"
GIT_REF="${OPLM_GIT_REF:-feat/fold-m1}"
REPO="${OPLM_REPO:-}"
CASES="${OPLM_CASES:-all}"
STEPS="${OPLM_STEPS:-venv install env make-fixtures parity}"
export RESULTS FIXTURES ESM_VENV GIT_REF REPO CASES STEPS

source "$ENV_FILE"
echo "=== $(date -Is) start; job=$SLURM_JOB_ID node=$(hostname) results=$RESULTS fixtures=$FIXTURES ==="
mkdir -p "$RESULTS" "$FIXTURES" "$(dirname "$ESM_VENV")" "$JOB_WORK_DIR"

INNER="$RESULTS/run-inner.sh"
cat > "$INNER" <<'EOF'
set -uo pipefail

log() { echo "=== $(date -Is) $*"; }
run() {
  local name=$1; shift
  case " $STEPS " in
    *" $name "*) ;;
    *) log "step $name skipped (OPLM_STEPS)"; echo "$name skipped" >> "$RESULTS/status.txt"; return 0 ;;
  esac
  log "step $name: $*"
  "$@" > "$RESULTS/$name.log" 2>&1
  local status=$?
  echo "$name $status" >> "$RESULTS/status.txt"
  log "step $name exit $status (log: $RESULTS/$name.log)"
  return 0
}

export CUDA_VISIBLE_DEVICES=""            # CPU oracle on purpose (fp32, no autocast)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export HF_HOME="/mnt/data/${SLURM_JOB_USER}/hf-cache"   # the 26 GB snapshot is reused across jobs
export UV_PYTHON_INSTALL_DIR="$(dirname "$ESM_VENV")/uv-python"
export WANDB_MODE=offline
: > "$RESULTS/status.txt"

if [ -z "$REPO" ]; then
  REPO="$JOB_WORK_DIR/oplm"
  log "cloning https://github.com/briney/oplm.git@$GIT_REF -> $REPO"
  git clone --quiet --branch "$GIT_REF" https://github.com/briney/oplm.git "$REPO"
fi
cd "$REPO"
git rev-parse HEAD > "$RESULTS/git-rev.txt"

# Pinned upstream environment: esm 3.4.1.post1 needs Python >=3.12 and transformers <5; this
# repo accepts transformers >=4.45, so one venv serves both (and exercises OPLM under 4.x).
run venv bash -c "pip install --quiet uv && ([ -x '$ESM_VENV/bin/python' ] || uv venv --python 3.12 '$ESM_VENV')"
run install bash -c "source '$ESM_VENV/bin/activate' && pip install --quiet 'esm==3.4.1.post1' rdkit && pip install --quiet -e '$REPO[dev]' && pip freeze | grep -i -E '^(esm|torch|transformers|rdkit|safetensors|huggingface)' > '$RESULTS/pip-freeze.txt'"
run env bash -c "source '$ESM_VENV/bin/activate' && python -c 'import esm, torch, transformers; print(esm.__version__, torch.__version__, transformers.__version__)'"
run make-fixtures bash -c "source '$ESM_VENV/bin/activate' && oplm fold make-fixtures --out '$FIXTURES' --cases '$CASES'"
cp "$FIXTURES/manifest.json" "$RESULTS/manifest.json" 2>/dev/null || true
run parity bash -c "source '$ESM_VENV/bin/activate' && OPLM_FOLD_FIXTURES='$FIXTURES' python -m pytest tests/fold/test_parity.py -v -rs --junitxml='$RESULTS/parity.xml'"

log "status summary:"
cat "$RESULTS/status.txt"
EOF

srun --nodes=1 --ntasks-per-node=1 \
  --export=ALL \
  --container-image="$CONTAINER" \
  --container-mounts="/mnt/home/${SLURM_JOB_USER}:/mnt/home/${SLURM_JOB_USER},/mnt/data:/mnt/data,/tmp:/tmp" \
  --container-workdir="$JOB_WORK_DIR" \
  --no-container-mount-home \
  bash "$INNER"

echo "=== $(date -Is) done; fixtures at $FIXTURES; copy $RESULTS/{manifest.json,parity.log,parity.xml,status.txt,pip-freeze.txt} into docs/fold/m1/ ==="
```

- [ ] **Step 8: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold/test_fixtures.py tests/fold/test_parity.py tests/fold/test_cli.py -q -rs && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/fixtures.py src/oplm/fold/cli.py tests/fold/conftest.py tests/fold/test_fixtures.py tests/fold/test_parity.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: `test_fixtures.py` passes; every `test_parity.py` test reports `SKIPPED (OPLM_FOLD_FIXTURES is unset ...)`; `test_cli.py` still passes. `ty` must not complain about the lazy `esm` imports (each carries a specific ignore).

```bash
git add src/oplm/fold/fixtures.py src/oplm/fold/cli.py tests/fold/conftest.py tests/fold/test_fixtures.py tests/fold/test_parity.py tests/fold/data/esmfold2_fast_config.json docs/fold/b200-fixtures.sbatch
git commit -m "feat(fold): ESMFold2 golden-fixture generator, released-config map and stage-wise parity tests"
```

---

### Task 11: `fold()` prediction API, mmCIF output, `oplm fold predict`, docs and attribution

**Files:**
- Create: `src/oplm/fold/predict.py`, `tests/fold/test_predict.py`
- Modify: `src/oplm/fold/cli.py`, `tests/fold/test_cli.py`, `pyproject.toml`, `docs/FOLD.md`, `THIRD_PARTY_NOTICES.md`, `AGENTS.md`, `docs/TESTING_E2E.md`

**Interfaces:**
- Consumes: Task 9 `OplmForFolding`, `FoldOutput`; Task 3 `featurize`, `ChainSpec`, `FoldFeatures`; Task 2 `PROTEIN_RES_TYPES`.
- Produces: `FoldResult` dataclass, `fold(model, chains, *, lm_hidden_states=None, num_samples=1, num_loops=None, num_steps=None, seed=None, pad_to_multiple=None) -> FoldResult`, `rank_samples(output, num_chains) -> Tensor`, `write_mmcif(result, path, *, name="pred") -> Path`, `parse_chain_arg(text, default_id) -> ChainSpec`, CLI `oplm fold predict`.

- [ ] **Step 1: Write the failing tests**

`tests/fold/test_predict.py`:

```python
"""fold(): ranking, result layout, mmCIF round trip through gemmi; chain-argument parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from oplm.fold import ChainSpec, FoldConfig, OplmForFolding, featurize
from oplm.fold.predict import FoldResult, fold, parse_chain_arg, rank_samples, write_mmcif
from oplm.model import OplmConfig, OplmModel

if TYPE_CHECKING:
    from pathlib import Path

gemmi = pytest.importorskip("gemmi")


def _model() -> OplmForFolding:
    torch.manual_seed(0)
    cfg = FoldConfig(
        pair_width=32, token_width=64, atom_width=32, atom_encoder_blocks=1, atom_encoder_heads=2,
        uid_rope_pairs=2, trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, diffusion_blocks=1,
        diffusion_heads=4, diffusion_atom_blocks=1, diffusion_atom_heads=2, fourier_dim=16,
        confidence_blocks=1, plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5,
        distogram_bins=8, inference_num_steps=2, inference_num_loops=1, lm_hidden_size=32,
        lm_num_hidden_states=3, attention_backend="dense", trimul_backend="reference",
    )
    model = OplmForFolding(cfg).eval()
    lm = OplmModel(OplmConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)).eval()
    model.attach_lm(lm)
    return model


def test_parse_chain_arg() -> None:
    assert parse_chain_arg("MKV", "A") == ChainSpec("MKV", "A")
    assert parse_chain_arg("B:GG", "A") == ChainSpec("GG", "B")
    assert parse_chain_arg("B:GG*3", "A") == ChainSpec("GG", "B", copies=3)
    with pytest.raises(ValueError, match="copies"):
        parse_chain_arg("GG*0", "A")


def test_rank_samples_uses_iptm_for_complexes_and_ptm_for_monomers() -> None:
    model = _model()
    f = featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B")])
    with torch.no_grad():
        out = model(f, num_samples=3, generator=torch.Generator().manual_seed(0))
    out.confidence.iptm = torch.tensor([0.1, 0.9, 0.5])
    out.confidence.ptm = torch.tensor([0.9, 0.1, 0.5])
    assert rank_samples(out, num_chains=2).tolist() == [1, 2, 0]
    assert rank_samples(out, num_chains=1).tolist() == [0, 2, 1]


def test_fold_result_layout_and_determinism() -> None:
    model = _model()
    chains = [ChainSpec("MKV", "A"), ChainSpec("GG", "B", copies=2)]
    a = fold(model, chains, num_samples=2, seed=1)
    b = fold(model, chains, num_samples=2, seed=1)
    assert isinstance(a, FoldResult) and a.chain_ids == ["A", "B", "B_2"]
    assert a.coords.shape == (24 + 8, 3) and a.plddt_per_atom.shape == (32,) and a.plddt.shape == (7,)
    assert a.atom_names[:5] == ["N", "CA", "C", "O", "CB"] and a.residue_names[:3] == ["MET", "LYS", "VAL"]
    assert a.atom_chain_index.tolist() == [0] * 24 + [1] * 4 + [2] * 4
    assert 0.0 <= a.ptm <= 1.0 and 0.0 <= a.iptm <= 1.0 and a.pae.shape == (7, 7)
    assert a.best_sample in (0, 1) and a.output.coords.shape[0] == 2
    torch.testing.assert_close(a.coords, b.coords)


def test_write_mmcif_round_trips_through_gemmi(tmp_path: Path) -> None:
    model = _model()
    result = fold(model, [ChainSpec("MKV", "A"), ChainSpec("GG", "B")], seed=0)
    path = write_mmcif(result, tmp_path / "pred.cif", name="unit")
    st = gemmi.read_structure(str(path))
    assert st.name.lower() == "unit" and len(st) == 1
    chains = {ch.name: ch for ch in st[0]}
    assert set(chains) == {"A", "B"}
    assert [r.name for r in chains["A"]] == ["MET", "LYS", "VAL"] and len(chains["B"]) == 2
    atoms = [a for ch in st[0] for r in ch for a in r]
    assert len(atoms) == 24 + 8
    assert all(0.0 <= a.b_iso <= 100.0 for a in atoms) and atoms[0].name == "N" and atoms[0].element.name == "N"
    assert chains["A"][0].seqid.num == 1 and chains["B"][1].seqid.num == 2
```

Add to `tests/fold/test_cli.py`:

```python
def test_fold_help_lists_predict_and_make_fixtures() -> None:
    result = runner.invoke(app, ["fold", "--help"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "predict" in out and "make-fixtures" in out


def test_predict_cli_writes_a_cif(tmp_path: Path) -> None:
    pytest.importorskip("gemmi")
    from oplm.fold import FoldConfig, OplmForFolding
    from oplm.model import OplmConfig, OplmModel

    torch.manual_seed(0)
    cfg = FoldConfig(
        pair_width=32, token_width=64, atom_width=32, atom_encoder_blocks=1, atom_encoder_heads=2,
        uid_rope_pairs=2, trunk_blocks=1, lm_encoder_blocks=1, coda_blocks=1, diffusion_blocks=1,
        diffusion_heads=4, diffusion_atom_blocks=1, diffusion_atom_heads=2, fourier_dim=16,
        confidence_blocks=1, plddt_bins=10, pae_bins=8, pde_bins=8, confidence_dist_bins=5,
        distogram_bins=8, inference_num_steps=2, inference_num_loops=1, lm_hidden_size=32,
        lm_num_hidden_states=3, attention_backend="dense", trimul_backend="reference",
    )
    OplmForFolding(cfg).save_pretrained(tmp_path / "head")
    OplmModel(OplmConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64)).save_pretrained(tmp_path / "lm")
    result = runner.invoke(app, [
        "fold", "predict", "MKV", "B:GG", "--model", str(tmp_path / "head"), "--lm", str(tmp_path / "lm"),
        "--out", str(tmp_path / "pred.cif"), "--seed", "0", "--device", "cpu",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "pred.cif").exists() and "pTM" in plain(result.output)
```

Run: `.venv/bin/python -m pytest tests/fold/test_predict.py tests/fold/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'oplm.fold.predict'` (or `gemmi` skip until the extra is installed: run `uv pip install --python .venv/bin/python -e ".[dev]"` after Step 3).

- [ ] **Step 2: `src/oplm/fold/predict.py`**

```python
"""Sequence(s) -> ranked structure -> mmCIF (spec §10 M1 "predict/mmCIF").

Ranking follows the ESMFold2 HF adapter's "best-ranked sample" convention: ipTM for complexes,
pTM for monomers (upstream's own ``fold()`` returns samples unranked). mmCIF is written with
gemmi; B-factors carry pLDDT × 100 (upstream's PDB writer leaves it on the 0–1 scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from oplm.fold.data.ccd import PROTEIN_RES_TYPES
from oplm.fold.data.featurize import ChainSpec, featurize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from oplm.fold.data.featurize import FoldFeatures
    from oplm.fold.modeling_fold import FoldOutput, OplmForFolding

__all__ = ["FoldResult", "fold", "parse_chain_arg", "rank_samples", "write_mmcif"]

_RES_NAMES = {v: k for k, v in PROTEIN_RES_TYPES.items()}
_ELEMENT_SYMBOLS = {6: "C", 7: "N", 8: "O", 16: "S"}


@dataclass
class FoldResult:
    """The best-ranked sample, unpadded, plus the raw model output."""

    chain_ids: list[str]
    sequences: list[str]
    coords: Tensor  # (A_real, 3) Å
    atom_names: list[str]
    atom_elements: list[int]
    atom_chain_index: Tensor  # (A_real,) index into chain_ids
    atom_residue_index: Tensor  # (A_real,) 0-based within the chain
    residue_names: list[str]  # per token
    plddt_per_atom: Tensor  # (A_real,) in [0, 1]
    plddt: Tensor  # (L,) per token
    pae: Tensor  # (L, L) Å
    ptm: float
    iptm: float
    best_sample: int
    features: FoldFeatures
    output: FoldOutput


def parse_chain_arg(text: str, default_id: str) -> ChainSpec:
    """``"SEQ"``, ``"ID:SEQ"`` or ``"ID:SEQ*N"`` -> :class:`ChainSpec`."""
    chain_id, _, rest = text.rpartition(":") if ":" in text else ("", "", text)
    seq, _, copies = rest.partition("*")
    n = int(copies) if copies else 1
    if n < 1:
        raise ValueError(f"copies must be >= 1 in {text!r}")
    if not seq:
        raise ValueError(f"empty sequence in {text!r}")
    return ChainSpec(seq.upper(), chain_id or default_id, copies=n)


def rank_samples(output: FoldOutput, num_chains: int) -> Tensor:
    """Sample indices from best to worst: by ipTM for complexes, pTM otherwise."""
    score = output.confidence.iptm if num_chains > 1 else output.confidence.ptm
    return torch.argsort(score.detach().cpu(), descending=True)


def fold(
    model: OplmForFolding,
    chains: Sequence[ChainSpec],
    *,
    lm_hidden_states: Tensor | None = None,
    num_samples: int = 1,
    num_loops: int | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
    pad_to_multiple: int | None = None,
) -> FoldResult:
    """Featurize, run the model, rank the samples and unpad the best one.

    ``pad_to_multiple`` defaults to 128 on CUDA (FlexAttention shape buckets, docs/FOLD.md §3)
    and to no padding on CPU. ``seed`` makes the initial pair state and the sampler
    deterministic for a given device.
    """
    device = model.device
    if pad_to_multiple is None and device.type == "cuda":
        pad_to_multiple = 128
    n_tokens = sum(len(c.sequence) * c.copies for c in chains)
    pad_to = None if pad_to_multiple is None else -(-n_tokens // pad_to_multiple) * pad_to_multiple
    f = featurize(chains, pad_tokens_to=pad_to)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        output = model(
            f, lm_hidden_states=lm_hidden_states, num_loops=num_loops, num_samples=num_samples,
            num_steps=num_steps, generator=generator,
        )
    order = rank_samples(output, f.num_chains)
    best = int(order[0])
    real = f.atom_mask[0].cpu()
    tok = f.token_mask[0].cpu()
    atom_to_token = f.atom_to_token[0].cpu()[real]
    conf = output.confidence
    names = ["".join(chr(int(c) + 32) for c in row).strip() for row in f.ref_atom_name_chars[0].cpu()[real].tolist()]
    return FoldResult(
        chain_ids=list(f.chain_ids),
        sequences=[c.sequence for c in chains for _ in range(c.copies)],
        coords=output.coords[best].detach().cpu()[real],
        atom_names=names,
        atom_elements=f.ref_element[0].cpu()[real].tolist(),
        atom_chain_index=f.asym_id[0].cpu()[atom_to_token],
        atom_residue_index=f.residue_index[0].cpu()[atom_to_token],
        residue_names=[_RES_NAMES[int(r)] for r in f.res_type[0].cpu()[tok].tolist()],
        plddt_per_atom=conf.plddt_per_atom[best].detach().cpu()[real],
        plddt=conf.plddt[best].detach().cpu()[tok],
        pae=conf.pae[best].detach().cpu()[tok][:, tok],
        ptm=float(conf.ptm[best]),
        iptm=float(conf.iptm[best]),
        best_sample=best,
        features=f,
        output=output,
    )


def write_mmcif(result: FoldResult, path: Path, *, name: str = "pred") -> Path:
    """Write the best sample as mmCIF (one model; B-factor = pLDDT × 100)."""
    import gemmi

    structure = gemmi.Structure()
    structure.name = name
    model = gemmi.Model("1")
    chains = {chain_id: gemmi.Chain(chain_id) for chain_id in result.chain_ids}
    residues: dict[tuple[int, int], gemmi.Residue] = {}
    token = 0
    seen: dict[tuple[int, int], int] = {}
    for atom_idx in range(result.coords.shape[0]):
        key = (int(result.atom_chain_index[atom_idx]), int(result.atom_residue_index[atom_idx]))
        if key not in residues:
            res = gemmi.Residue()
            res.name = result.residue_names[token]
            res.seqid = gemmi.SeqId(key[1] + 1, " ")
            res.het_flag = "A"
            residues[key] = res
            seen[key] = token
            token += 1
        atom = gemmi.Atom()
        atom.name = result.atom_names[atom_idx]
        z = result.atom_elements[atom_idx]
        atom.element = gemmi.Element(_ELEMENT_SYMBOLS.get(z, "X"))
        x, y, zc = result.coords[atom_idx].tolist()
        atom.pos = gemmi.Position(x, y, zc)
        atom.occ = 1.0
        atom.b_iso = float(result.plddt_per_atom[atom_idx]) * 100.0
        residues[key].add_atom(atom)
    for (chain_idx, _), res in residues.items():
        chains[result.chain_ids[chain_idx]].add_residue(res)
    for chain in chains.values():
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    path = Path(path)
    structure.make_mmcif_document().write_file(str(path))
    return path
```

- [ ] **Step 3: `predict` command in `src/oplm/fold/cli.py`**

```python
@app.command("predict")
def predict(
    chains: Annotated[list[str], typer.Argument(help="Chains as SEQ, ID:SEQ or ID:SEQ*N (homo-oligomer copies)")],
    model: Annotated[Path, typer.Option("--model", help="Fold checkpoint directory")],
    out: Annotated[Path, typer.Option("--out", help="Output .cif path")],
    lm: Annotated[str | None, typer.Option("--lm", help="Frozen LM path or Hub id (overrides the checkpoint's)")] = None,
    samples: Annotated[int, typer.Option(help="Diffusion samples; the best by ipTM (complex) or pTM is written")] = 1,
    loops: Annotated[int | None, typer.Option(help="Recurrence iterations (default: config)")] = None,
    steps: Annotated[int | None, typer.Option(help="Sampler steps before the sigma cap (default: config)")] = None,
    seed: Annotated[int | None, typer.Option(help="Seed for the initial pair state and the sampler")] = None,
    device: Annotated[str, typer.Option(help="cuda, cpu or auto")] = "auto",
) -> None:
    """Predict a protein (complex) structure and write it as mmCIF."""
    import torch

    from oplm.fold.modeling_fold import OplmForFolding
    from oplm.fold.predict import fold, parse_chain_arg, write_mmcif

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    specs = [parse_chain_arg(text, chr(ord("A") + i)) for i, text in enumerate(chains)]
    folder = OplmForFolding.from_pretrained(model, lm_name_or_path=lm).to(dev).eval()
    if folder.lm is None:
        raise typer.BadParameter("no language model: pass --lm or save the checkpoint with lm_name_or_path")
    folder.lm.to(dev)
    result = fold(folder, specs, num_samples=samples, num_loops=loops, num_steps=steps, seed=seed)
    write_mmcif(result, out)
    console.print(
        f"wrote {out}  pTM {result.ptm:.3f}  ipTM {result.iptm:.3f}  "
        f"mean pLDDT {float(result.plddt.mean()) * 100:.1f}  sample {result.best_sample}/{samples}"
    )
```

- [ ] **Step 4: `pyproject.toml`** — add `"gemmi>=0.7"` to the `fold` extra and to `dev` (so the test suite exercises mmCIF). Reinstall: `uv pip install --python .venv/bin/python -e ".[dev]"`.

- [ ] **Step 5: Docs and attribution**

`docs/FOLD.md` — append:

```markdown
## 7. Milestone 1: ESMFold2 inference port (`oplm.fold.modeling_fold`)

**What exists.** `FoldConfig` (defaults = the released `biohub/ESMFold2-Fast` config; `docs/
fold/m1/` records the parity run), `featurize()` (protein chains -> `FoldFeatures`; one LM row
per chain with BOS/EOS; atoms padded to 32; tokens optionally padded to a crop multiple),
`OplmForFolding` (checkpoint-identical module names; frozen LM held outside the module tree via
`attach_lm` / `lm_name_or_path`; `forward(features) -> FoldOutput`), `fold()` + `write_mmcif()`
and `oplm fold predict`. `oplm fold make-fixtures` + `docs/fold/b200-fixtures.sbatch` record
the parity oracle; `tests/fold/test_parity.py` runs when `OPLM_FOLD_FIXTURES` points at it.

**Deviations from upstream, all deliberate.**
- Per-loop LM-pair dropout (`pair_dropout`) and LM input masking are training-only; upstream
  forces dropout on at inference. Inference is deterministic given `generator`.
- `inference_num_loops` counts iterations executed (upstream `num_loops + 1`); the spec default
  is 10, the released config maps to 21.
- The initial pair state and the sampler take a `torch.Generator`; upstream draws from the
  global RNG. Independently sampled structures are therefore not an oracle (spec §9).
- Padded query rows of the diffusion token transformer are zeroed (upstream leaves them finite
  garbage); interface pLDDT (`complex_iplddt`) is deferred to milestone 2.
- Loading a fold checkpoint needs `oplm` installed; `trust_remote_code` bundling is not
  provided (the featurizer, tokenizer vocabulary and LM live in this package).
- `fold()` ranks samples by ipTM (complex) / pTM (monomer); upstream's `fold()` does not rank.

**Parity (filled in by Task 12).** | stage | atol used | max abs err observed | cases |
```

`THIRD_PARTY_NOTICES.md` — add rows to the table:

```markdown
| `src/oplm/fold/pair.py` | `esm/models/esmfold2/layers.py` (`ResIdxAsymIdSymIdEntityIdEncoding`, `SingleToPair`) | Relative-position one-hot layout; outer product/difference |
| `src/oplm/fold/trunk.py` | `esm/models/esmfold2/{layers,model}.py` (`PairUpdateBlock`, `FoldingTrunk`, `Transition`, parcae recurrence) | Block composition, SwiGLU order, recurrence dynamics and init |
| `src/oplm/fold/atoms.py` | `esm/models/esmfold2/layers.py` (`build_3d_rope`, `SWA3DRoPEAttention`, `SWAAtomBlock`, `EsmFold2AtomEncoder/Decoder`, `InputsEmbedder`) | Atom feature layout, 3D RoPE, adaLN atom blocks, token aggregation |
| `src/oplm/fold/lm_shim.py` | `esm/models/esmfold2/layers.py` (`LanguageModelShim`) | Per-layer norm/projection, softmax layer mix |
| `src/oplm/fold/diffusion.py` | `esm/models/esmfold2/layers.py` (`DiffusionConditioning`, `AttentionPairBias`, `ConditionedTransitionBlock`, `DiffusionModule`, `DiffusionStructureHead`) | Conditioning, adaLN blocks, EDM preconditioning, Karras schedule, churned sampler, Kabsch |
| `src/oplm/fold/confidence.py` | `esm/models/esmfold2/model.py` (`ConfidenceHead`, pTM/ipTM) | Pair init, distance bins, pooling, pLDDT/PAE/PDE heads, TM-score formula |
| `src/oplm/fold/modeling_fold.py` | `esm/models/esmfold2/model.py` (`EsmFold2Model.forward`) | Stage order and precision policy |
| `src/oplm/fold/data/ccd.py`, `src/oplm/fold/data/reference_conformers.json` | `esm/models/esmfold2/{constants,protein_utils}.py` | Residue vocabulary, heavy-atom order, charged atoms, reference conformers |
| `src/oplm/fold/data/featurize.py` | `esm/models/esmfold2/prepare_input.py` | Token/atom feature construction, unknown-residue bond quirk |
| `src/oplm/fold/fixtures.py` | `esm/models/esmfold2/config.py` | Config field correspondence |
```

`AGENTS.md` — update the `fold/` line in the test-layout tree to: `fold/  # structure prediction head: kernels, model, data, predict; parity tests need OPLM_FOLD_FIXTURES (docs/FOLD.md §7)`. `docs/TESTING_E2E.md` §2 — add one line: "`OPLM_FOLD_FIXTURES=<dir>` enables `tests/fold/test_parity.py` (skipped with a reason otherwise)".

- [ ] **Step 6: Run, lint, commit**

Run: `.venv/bin/python -m pytest tests/fold -q -m "not slow" && .venv/bin/ruff check src/ && .venv/bin/ruff format src/oplm/fold/predict.py src/oplm/fold/cli.py tests/fold/test_predict.py tests/fold/test_cli.py && VIRTUAL_ENV=.venv .venv/bin/ty check src/`
Expected: pass (parity tests skipped with reason).

```bash
git add src/oplm/fold/predict.py src/oplm/fold/cli.py tests/fold/test_predict.py tests/fold/test_cli.py pyproject.toml docs/FOLD.md THIRD_PARTY_NOTICES.md AGENTS.md docs/TESTING_E2E.md
git commit -m "feat(fold): fold() prediction API, gemmi mmCIF output and oplm fold predict; M1 docs and attribution"
```

---

### Task 12: Cluster acceptance: fixtures, parity, GPU run, recorded tolerances

**Files:**
- Create: `docs/fold/m1/{manifest.json,parity.log,parity.xml,status.txt,pip-freeze.txt,gpu-tests.log}`
- Modify: `docs/FOLD.md` §7 (parity table), `tests/fold/test_parity.py` (tolerances), possibly `src/oplm/fold/data/reference_conformers.json`

This task needs the SUNK cluster; the user submits the jobs (as for milestone 0) and copies results back. Everything else is done here.

- [ ] **Step 1: Push the branch and submit the fixture job**

```bash
git push -u origin feat/fold-m1
# on a login node:
sbatch docs/fold/b200-fixtures.sbatch
```

Expected: `status.txt` shows `venv 0`, `install 0`, `env 0`, `make-fixtures 0`, `parity <0 or 1>`. The fixture directory (`/mnt/data/<user>/fold-fixtures/<jobid>`) holds `manifest.json`, `config.json`, `head.safetensors`, `reference_conformers.json` and five `<case>.safetensors`.

- [ ] **Step 2: Triage parity failures in order**

1. `test_head_weights_load_strictly_and_config_matches` fails → a module name or shape differs from the checkpoint; fix the module (the name contract test in Task 9 should already agree, so this is a shape).
2. `test_featurizer_matches_upstream` fails on `ref_pos` only → replace `src/oplm/fold/data/reference_conformers.json` with the fixture directory's copy (its `source` names `get_idealized_atom_pos`), rerun. Fails on `entity_id`/`sym_id` → align `_expand_chains` with upstream's numbering (read `build_chains_from_input` in the installed `esm`), never special-case the test. Fails on `token_bonds` → the unknown-residue quirk; compare against `compute_token_bonds`.
3. Shim / embedder / recurrence / denoiser / confidence: a stage failing while every earlier stage passes localises the bug to that module; fix the module, keep the tolerance.
4. Re-run only the parity step: `OPLM_STEPS="parity" OPLM_FIXTURES=<dir> sbatch docs/fold/b200-fixtures.sbatch` (no regeneration needed unless the generator changed).

- [ ] **Step 3: Record observed errors and tighten tolerances**

Extract the `max abs err` values that `_close` prints on failure, or add `-s` printing of the error per stage by setting `atol=0` in a scratch run; fill the FOLD.md §7 table (stage, atol, observed max, cases) and set each `atol` in `test_parity.py` to the smallest round number ≥ 10× the observed maximum. Copy `manifest.json`, `parity.log`, `parity.xml`, `status.txt`, `pip-freeze.txt` into `docs/fold/m1/` (`git add -f` for `.log`).

- [ ] **Step 4: GPU run**

```bash
OPLM_GIT_REF=feat/fold-m1 OPLM_STEPS="install env-torch gpu-tests" \
  sbatch --export=ALL,OPLM_FOLD_FIXTURES=/mnt/data/<user>/fold-fixtures/<jobid> docs/fold/b200-task7.sbatch
```

(The M0 job's `gpu-tests` step runs `pytest tests/fold -m slow`, which now includes
`test_gpu_pipeline_tracks_the_cpu_oracle` and the milestone-0 GPU tests; `--export=ALL`
forwards the variable into the container.) Expected: all slow fold tests pass; record
`gpu-tests.log` under `docs/fold/m1/` and the relative pair error in FOLD.md §7.

- [ ] **Step 5: End-to-end prediction with the released head**

Inside the fixture venv on the cluster (CPU is fine for one 20-mer):

```bash
OPLM_FOLD_FIXTURES=<dir> python - <<'EOF'
import json, os, torch
from pathlib import Path
from safetensors.torch import load_file
from oplm.fold import ChainSpec, OplmForFolding
from oplm.fold.fixtures import FIXTURE_CASES, fold_config_from_upstream, load_fixture
from oplm.fold.predict import fold, write_mmcif
d = Path(os.environ["OPLM_FOLD_FIXTURES"])
cfg = fold_config_from_upstream(json.loads((d / "config.json").read_text()), attention_backend="dense", trimul_backend="reference")
model = OplmForFolding(cfg).eval()
model.load_state_dict(load_file(str(d / "head.safetensors")), strict=True)
case = FIXTURE_CASES[0]
fx = load_fixture(d, case.name)
r = fold(model, case.chains, lm_hidden_states=fx["lm_hidden_states"].float(), num_loops=21, seed=0)
write_mmcif(r, Path("trp_cage.cif"))
print("ours  pTM %.3f mean pLDDT %.3f" % (r.ptm, float(r.plddt.mean())))
print("upstream (3 loops, 3 steps) pTM %.3f mean pLDDT %.3f" % (float(fx["output.ptm"][0]), float(fx["output.plddt"][0].mean())))
EOF
```

Expected: a valid `trp_cage.cif` (opens in gemmi/PyMOL), pTM and mean pLDDT in the same range as upstream's recorded values (they are not expected to match: different loop/step counts and RNG). Record the two lines in FOLD.md §7.

- [ ] **Step 6: Commit the evidence**

```bash
git add docs/fold/m1 docs/FOLD.md tests/fold/test_parity.py src/oplm/fold/data/reference_conformers.json
git add -f docs/fold/m1/*.log
git commit -m "docs(fold): record the milestone-1 parity run, GPU run and tolerances"
```

Milestone-1 acceptance (spec §10): every `tests/fold/test_parity.py` test passes on the recorded fixtures (deterministic ESMFold2 parity), `tests/fold/test_pair.py::test_every_pair_feature_is_block_local` passes (block-feature parity), and the GPU slow suite is green.
