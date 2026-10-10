"""FoldConfig: ESMFold2-Fast defaults, derived widths, validation and save/load round-trip."""

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
        39,
        3.25,
        50.75,
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
