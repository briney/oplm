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
    assert differing == {
        "inference_num_loops",
        "lm_hidden_size",
        "lm_num_hidden_states",
        "lm_name_or_path",
        "lm_input_mask_fraction",
    }
    assert (
        cfg.inference_num_loops == 21
        and cfg.lm_hidden_size == 2560
        and cfg.lm_num_hidden_states == 81
    )
    assert cfg.lm_name_or_path == "biohub/ESMFold2-Fast#esmc"
    assert cfg.lm_input_mask_fraction == 0.0
    assert cfg.single_inputs_width == 451 and cfg.atom_feature_dim == 389
    over = fold_config_from_upstream(
        json.loads(_CFG.read_text()), attention_backend="dense", trunk_blocks=1
    )
    assert over.attention_backend == "dense" and over.trunk_blocks == 1


def test_extract_head_weights_drops_esmc_and_keeps_shards_order(tmp_path: Path) -> None:
    (tmp_path / "snap").mkdir()
    save_file(
        {"esmc.a": torch.zeros(2), "distogram_head.bias": torch.ones(3)},
        tmp_path / "snap" / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"parcae.log_delta": torch.full((4,), 2.0), "esmc.b": torch.zeros(1)},
        tmp_path / "snap" / "model-00002-of-00002.safetensors",
    )
    index = {
        "weight_map": {
            "esmc.a": "model-00001-of-00002.safetensors",
            "distogram_head.bias": "model-00001-of-00002.safetensors",
            "parcae.log_delta": "model-00002-of-00002.safetensors",
            "esmc.b": "model-00002-of-00002.safetensors",
        }
    }
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
