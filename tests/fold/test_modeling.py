"""OplmForFolding: checkpoint names, custom inits, frozen-LM ownership, forward, save/load."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoConfig, AutoModel

from oplm.fold import ChainSpec, FoldConfig, OplmForFolding, featurize
from oplm.fold.lm_shim import lm_state_count
from oplm.fold.modeling_fold import load_frozen_lm
from oplm.training.ema import build_ema
from tests.fold.helpers import tiny_fold_config, tiny_lm

_KEYS = Path(__file__).parent / "data" / "esmfold2_fast_head_keys.txt"

_TINY: dict[str, Any] = dict(
    plddt_bins=10,
    pae_bins=8,
    pde_bins=8,
    confidence_dist_bins=5,
    distogram_bins=8,
    inference_num_steps=2,
    inference_num_loops=2,
    lm_hidden_size=32,
    lm_num_hidden_states=3,
)


def _tiny_cfg(**over: Any) -> FoldConfig:
    return tiny_fold_config(**{**_TINY, **over})


def test_state_dict_matches_released_checkpoint_patterns() -> None:
    expected = set(_KEYS.read_text().split())
    assert len(expected) == 213
    with torch.device("meta"):
        model = OplmForFolding(FoldConfig())
    keys = list(model.state_dict())
    assert len(keys) == 1054
    # transformers 4.x matches the prefix without a dot; a hit there silently skips loading
    assert not any(k.startswith(OplmForFolding.base_model_prefix) for k in keys)
    patterns = {re.sub(r"\.\d+\.", ".N.", k) for k in keys}
    assert patterns == expected
    counts = {}
    for k in keys:
        m = re.match(r"(.*?)\.layers\.(\d+)\.", k)
        if m:
            counts[m.group(1)] = max(counts.get(m.group(1), 0), int(m.group(2)) + 1)
    assert counts == {
        "confidence_head.folding_trunk": 4,
        "folding_trunk": 24,
        "input_embedder.atom_encoder": 3,
        "lm_encoder": 4,
        "parcae.output_stack": 2,
        "structure_head.atom_decoder": 3,
        "structure_head.atom_encoder": 3,
        "structure_head.token_transformer": 12,
    }
    sd = model.state_dict()
    assert sd["distogram_head.weight"].shape == (64, 256)
    assert sd["structure_head.conditioning.pair_transition_0.mlp.gate_up_proj.weight"].shape == (
        1024,
        256,
    )
    assert sd["structure_head.token_transformer.layers.0.mlp.gate_up_proj.weight"].shape == (
        3072,
        768,
    )
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
    torch.testing.assert_close(
        sd["parcae.input_matrix_continuous"], torch.eye(model.config.pair_width)
    )
    assert sd["confidence_head.plddt_weight"].abs().sum() == 0
    assert sd["language_model.layer_weights"].abs().sum() == 0
    assert sd["input_embedder.pair_init_1.weight"].abs().sum() > 0  # generic init still runs


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
    lm = tiny_lm()
    model.attach_lm(lm, name_or_path="tiny-lm", revision="r1")
    assert model.lm is lm
    assert model.config.lm_name_or_path == "tiny-lm" and model.config.lm_revision == "r1"
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
        assert not any("backbone" in k for k in f.keys())  # noqa: SIM118  # safe_open is not a dict


def test_attach_lm_validates_the_shape_contract() -> None:
    model = OplmForFolding(_tiny_cfg(lm_hidden_size=16))
    with pytest.raises(ValueError, match="lm_hidden_size"):
        model.attach_lm(tiny_lm())
    model = OplmForFolding(_tiny_cfg(lm_num_hidden_states=4))
    with pytest.raises(ValueError, match="lm_num_hidden_states"):
        model.attach_lm(tiny_lm())


def _features():
    return featurize([ChainSpec("MKV", "A"), ChainSpec("GG", "B", copies=2)])


def test_forward_end_to_end_is_deterministic_under_a_generator() -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg()).eval()
    model.attach_lm(tiny_lm())
    f = _features()
    with torch.no_grad():
        out = model(
            f, num_samples=2, generator=torch.Generator().manual_seed(3), return_intermediates=True
        )
        again = model(f, num_samples=2, generator=torch.Generator().manual_seed(3))
    L, A = f.num_tokens, f.num_atoms
    assert out.coords.shape == (2, A, 3) and out.distogram_logits.shape == (1, L, L, 8)
    assert out.pair.shape == (1, L, L, 32)
    assert out.s_inputs.shape == (1, L, model.config.single_inputs_width)
    assert out.confidence.ptm.shape == (2,) and out.confidence.pair_chains_iptm.shape == (2, 3, 3)
    assert torch.isfinite(out.coords).all() and torch.isfinite(out.confidence.pae).all()
    assert out.intermediates is not None and len(out.intermediates["states"]) == 2
    assert set(out.intermediates) >= {
        "s_inputs",
        "z_init",
        "relpos",
        "bonds",
        "lm_z",
        "z0",
        "states",
    }
    torch.testing.assert_close(out.coords, again.coords)
    torch.testing.assert_close(out.distogram_logits, again.distogram_logits)


def test_forward_accepts_precomputed_lm_states_and_z0() -> None:
    torch.manual_seed(0)
    model = OplmForFolding(_tiny_cfg()).eval()
    f = _features()
    hs = torch.randn(1, f.num_tokens, 3, 32)
    z0 = torch.zeros(1, f.num_tokens, f.num_tokens, 32)
    with torch.no_grad():
        out = model(
            f, lm_hidden_states=hs, z0=z0, num_loops=1, generator=torch.Generator().manual_seed(0)
        )
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
    with_lm = OplmForFolding.from_pretrained(tmp_path, lm=tiny_lm())
    assert with_lm.lm is not None


def test_from_pretrained_resolves_the_lm_from_config(tmp_path: Path) -> None:
    lm = tiny_lm()
    lm.save_pretrained(tmp_path / "lm")
    model = OplmForFolding(_tiny_cfg(lm_name_or_path=str(tmp_path / "lm")))
    model.save_pretrained(tmp_path / "fold")
    reloaded = OplmForFolding.from_pretrained(tmp_path / "fold")
    assert reloaded.lm is not None and lm_state_count(reloaded.lm) == 3
    override = OplmForFolding.from_pretrained(
        tmp_path / "fold", lm_name_or_path=str(tmp_path / "lm")
    )
    assert override.lm is not None
    bad = OplmForFolding(_tiny_cfg(lm_name_or_path=str(tmp_path / "lm"), lm_hidden_size=16))
    bad.save_pretrained(tmp_path / "bad")
    with pytest.raises(ValueError, match="lm_hidden_size"):
        OplmForFolding.from_pretrained(tmp_path / "bad")


def test_esmc_heads_reload_without_an_lm(tmp_path: Path) -> None:
    """A ``<repo>#esmc`` head skips LM auto-resolution; only that suffix is rejected."""
    OplmForFolding(_tiny_cfg(lm_name_or_path="biohub/ESMFold2-Fast#esmc")).save_pretrained(
        tmp_path / "fold"
    )
    assert OplmForFolding.from_pretrained(tmp_path / "fold").lm is None
    with pytest.raises(ValueError, match="bundled"):
        load_frozen_lm("x#esmc")
    with pytest.raises(ValueError, match="bundled"):  # an explicit #esmc override still fails
        OplmForFolding.from_pretrained(tmp_path / "fold", lm_name_or_path="x#esmc")
    tiny_lm().save_pretrained(tmp_path / "dir#1" / "lm")
    assert lm_state_count(load_frozen_lm(tmp_path / "dir#1" / "lm")) == 3


def test_gradient_checkpointing_reaches_every_pair_stack() -> None:
    model = OplmForFolding(_tiny_cfg())
    model.gradient_checkpointing_enable()
    assert model.folding_trunk.gradient_checkpointing
    assert model.confidence_head.folding_trunk.gradient_checkpointing
    model.gradient_checkpointing_disable()
    assert not model.lm_encoder.gradient_checkpointing
